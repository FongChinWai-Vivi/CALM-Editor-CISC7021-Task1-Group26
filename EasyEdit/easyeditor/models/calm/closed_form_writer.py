"""
closed_form_writer.py -- MEMIT / AlphaEdit closed-form writer for CALM (W4, Fong Chin Wai).

Motivation (W4 quality lift, 2026-11-06)
----------------------------------------
Every evaluated WRITE path so far lands at 1.6%-3.9% *true* model-level
rewrite rate (22.2% raw was REJECT false hits).  The diagnosis is that the
hypernetwork's learned low-rank ``delta = scale * U @ V^T`` does not point in
the direction that actually moves the model's next-token distribution toward
``target_new``.  MEMIT / AlphaEdit instead *solve* for the weight update in
closed form:

    min_delta  || (W + delta) K - (W K + R) ||^2  +  lambda ||delta||^2
    =>  delta = R K^T (K K^T + lambda I)^-1

where ``K`` holds the keys (input activations of the rewrite MLP at the
subject's last token) and ``R`` holds residual errors towards an optimised
value vector ``z`` (computed by :func:`alphaedit.compute_z`).  No meta-training
is involved, so the direction is derived from the actual model rather than from
a generator trained on a different objective.

This module exposes a *drop-in* replacement for
:class:`easyeditor.models.calm.hypernetwork.HypernetworkEditGenerator`:

    generate_update(request) -> {param_name: delta}
    apply_update(deltas)     -> None (in-place add)
    generate_and_apply(request)

so :class:`easyeditor.editors.calm_editor.CALMEditor` can swap writers via
``hparams.hypernet_alg = "closedform"`` with no other change.

Scope of this first skeleton (v1)
---------------------------------
* MEMIT closed-form solve on the MLP ``c_proj`` ("rewrite module") weights of
  ``layers`` (default ``[9, 10, 11]`` for gpt2-small, matching the W1/W3
  ``inner_params`` MLP layers).
* ``z`` value vectors via EasyEdit's AlphaEdit ``compute_z`` (Adam on a single
  hidden-state vector; the loss is tied to the last layer).
* Optional AlphaEdit-style key statistics term ``cache_c``
  (``cf_kv_cache=True``): previously written keys regularise later solves,
  which is the cheap half of AlphaEdit that needs no offline covariance.
* Null-space projection (the ``P`` matrix of AlphaEdit) is *not* enabled by
  default: it requires mom2 statistics over wikipedia plus a precomputed
  ``P_loc``; ``null_space="stats"`` raises a clear error instead of silently
  degrading to plain MEMIT.

Every applied edit records ``delta_norm`` / ``weight_norm`` /
``rel_delta`` per parameter in ``self.last_apply_stats`` so the W4 audit
(`audit_per_edit.py`) can separate "delta landed" from "delta pointed the
wrong way".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...util import nethook
from ...util.device import get_model_device, normalize_device
from ..alphaedit.compute_z import compute_z, get_module_input_output_at_words


@dataclass
class _CFHParams:
    """Attribute bag consumed by ``alphaedit.compute_z``.

    ``compute_z`` only reads plain attributes, so a dataclass mirroring the
    fields it touches keeps this writer decoupled from ``AlphaEditHyperParams``
    (which forces a yaml-backed constructor).
    """

    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    rewrite_module_tmp: str
    layer_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: Any


class ClosedFormEditGenerator:
    """MEMIT/AlphaEdit closed-form writer bound to a base model.

    Parameters
    ----------
    layers:
        Layer indices whose ``rewrite_module_tmp.format(layer)`` weight is
        solved for.  **The last entry must be strictly before the final
        transformer block**: MEMIT writes into the residual stream at that
        layer, and only the *remaining* blocks can propagate the change to the
        token being predicted.  gpt2-small has 12 blocks, so the W3
        ``inner_params`` set {9, 10, 11} leaves no downstream block and yields
        a provably zero gradient (verified 2026-11-06: delta stayed at
        ``norm 0.0``).  Defaults to ``[4, 5, 6, 7, 8]``, the gpt2-small analogue
        of MEMIT's ``[13..17]`` on 48-layer gpt2-xl.
    l2:
        Ridge coefficient ``lambda`` of the closed-form solve, i.e. MEMIT's
        ``mom2_update_weight``.  MEMIT ships 15000 for gpt2-xl and that order of
        magnitude is *load-bearing*: the Gram matrix entries scale like the
        squared residual norm (~1e4-1e6), so ``lambda = 10`` is numerically the
        same as no regularisation at all and the delta explodes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tok,
        inner_params: Optional[List[str]] = None,
        layers: Optional[List[int]] = None,
        l2: float = 10.0,
    l2_mode: str = "rel",
    delta_scale: float = 1.0,
    resid_ref: str = "z_layer",
    z_clamp: float = 1.0,
        v_lr: float = 0.5,
        v_num_grad_steps: int = 20,
        v_loss_layer: int = -1,
        v_weight_decay: float = 0.5,
        clamp_norm_factor: float = 0.75,
        kl_factor: float = 0.0625,
        fact_token: str = "subject_last",
        rewrite_module_tmp: str = "transformer.h.{}.mlp.c_proj",
        layer_module_tmp: str = "transformer.h.{}",
        ln_f_module: str = "transformer.ln_f",
        lm_head_module: str = "lm_head",
        kv_cache: bool = False,
        null_space: str = "none",
        proj_mix: float = 1.0,
        proj_l2: float = -1.0,
        delta_rank: int = 0,
        delta_cap: float = -1.0,
        replay_cache: bool = False,
        max_contexts: int = 10,
        device: Any = None,
        verbose: bool = False,
    ):
        self.model = model
        self.tok = tok
        self.inner_params = list(inner_params or [])
        self.layers = [int(x) for x in (layers or [4, 5, 6, 7, 8])]
        self.l2 = float(l2)
        self.delta_scale = float(delta_scale)
        self.resid_ref = str(resid_ref)
        self.z_clamp = float(z_clamp)
        if l2_mode not in ("rel", "abs"):
            raise ValueError(f"[CALM-CF] l2_mode must be 'rel' or 'abs', got {l2_mode!r}")
        self.l2_mode = l2_mode
        self.kv_cache = bool(kv_cache)
        self.null_space = str(null_space).lower()
        self.proj_mix = float(proj_mix)
        self.proj_l2 = float(proj_l2)
        # Fluency guards on the solved delta itself (independent of z clamp):
        #   delta_rank > 0 : keep only the top-r singular directions (SVD) of
        #                    the [d_in, d_out] update.  A full-rank solve spends
        #                    energy in directions that carry no key information,
        #                    which is pure fluency damage.
        #   delta_cap > 0  : cap ||delta||_F at delta_cap * ||W||_F per layer.
        self.delta_rank = int(delta_rank)
        self.delta_cap = float(delta_cap)
        # Replay cache: remember the (keys, residual) rows of every chunk that
        # has already been written and re-impose them in later solves, so a new
        # chunk cannot take the space back.  Without it, chunk k+1 solves only
        # its own constraints and silently overwrites chunk k (measured 0/50
        # retention at n=500, chunked solves included).
        self.replay_cache = bool(replay_cache)
        self._replay: Dict[int, tuple] = {}
        self.max_contexts = max(1, int(max_contexts))
        self.verbose = bool(verbose)

        self.rewrite_module_tmp = rewrite_module_tmp
        self.layer_module_tmp = layer_module_tmp

        # gpt2 tokenizer has no pad token by default; compute_z pads a batch.
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token

        dev = normalize_device(device) if device is not None else get_model_device(model)
        self.device = dev

        # -1: tie the z objective to the last transformer layer (MEMIT default).
        if int(v_loss_layer) < 0:
            cfg = getattr(model, "config", None)
            n_layer = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None)
            if n_layer is None:
                raise ValueError(
                    "[CALM-CF] cannot infer the layer count for v_loss_layer; "
                    "pass cf_v_loss_layer explicitly"
                )
            v_loss_layer = int(n_layer) - 1

        cfg = getattr(model, "config", None)
        self._n_layer = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None)

        self.hparams = _CFHParams(
            layers=self.layers,
            fact_token=fact_token,
            v_num_grad_steps=int(v_num_grad_steps),
            v_lr=float(v_lr),
            v_loss_layer=int(v_loss_layer),
            v_weight_decay=float(v_weight_decay),
            clamp_norm_factor=float(clamp_norm_factor),
            kl_factor=float(kl_factor),
            rewrite_module_tmp=rewrite_module_tmp,
            layer_module_tmp=layer_module_tmp,
            ln_f_module=ln_f_module,
            lm_head_module=lm_head_module,
            device=dev,
        )

        # AlphaEdit-style running key statistics (``cache_c`` in AlphaEdit).
        # Per layer: sum of K K^T over previously written keys.  Requires
        # cf_kv_cache=True; it is the offline-statistics-free half of AlphaEdit.
        self._key_cache: Dict[int, torch.Tensor] = {}
        self._acc_n: Dict[int, int] = {}

        self.last_apply_stats: Dict[str, Dict[str, float]] = {}
        self.edit_count: int = 0

        # MEMIT writes into the residual stream at ``layers[-1]``; the blocks
        # after it are what propagate the change to the predicted token.  If the
        # rewrite layer *is* the last block, the value gradient of the z
        # objective is identically zero and nothing can be learned/landed.
        if self._n_layer is not None and self.layers[-1] >= int(self._n_layer) - 1:
            raise ValueError(
                f"[CALM-CF] layers={self.layers} puts the rewrite layer at "
                f"{self.layers[-1]}, but this model has {self._n_layer} blocks: "
                f"MEMIT needs at least one downstream block to propagate the "
                f"write (otherwise the z gradient is exactly zero). Use layers "
                f"strictly below {int(self._n_layer) - 1}, e.g. [4, 5, 6, 7, 8]."
            )

        if self.null_space not in ("none", "stats", "keys"):
            raise ValueError(
                f"[CALM-CF] unknown null_space mode {self.null_space!r} "
                f"(expected 'none', 'stats' or 'keys')"
            )
        if self.null_space == "stats":
            raise NotImplementedError(
                "[CALM-CF] null_space='stats' needs mom2 wikipedia statistics "
                "+ a precomputed null-space matrix P (AlphaEdit P_loc); this "
                "first skeleton only ships plain MEMIT (null_space='none'). "
                "Generate P offline, then wire it into _solve() to enable."
            )

    # ------------------------------------------------------------------ #
    # Request preparation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_mit_request(request: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a CALM request into the MEMIT/AlphaEdit request convention.

        MEMIT expects ``prompt`` to still carry the ``{}`` placeholder for the
        subject and ``target_new`` to be a plain string with a leading space
        (GPT-2 tokenisation).
        """
        prompt = request["prompt"]
        subject = request.get("subject") or ""
        if "{}" not in prompt:
            if subject and subject in prompt:
                prompt = prompt.replace(subject, "{}")
            else:
                prompt = prompt + " {}"
        target_new = request["target_new"]
        if isinstance(target_new, dict):
            target_new = target_new.get("str", "")
        if not target_new.startswith(" "):
            target_new = " " + target_new
        return {"prompt": prompt, "subject": subject, "target_new": target_new}

    def _context_templates(self, request: Dict[str, Any]) -> List[List[str]]:
        """Sub-templates handed to ``compute_z`` (MEMIT nesting convention)."""
        return [["{}"]]

    def _key_contexts(self, request: Dict[str, Any]) -> List[str]:
        """Flat context templates used for the **key** extraction and the
        residual measurement (``{}`` marks the subject slot).

        Conditioning matters more than cost here: with a single context the
        Gram matrix ``K K^T`` is rank one, so the ridge solve is free to spend
        an enormous delta in every orthogonal direction -- that is what
        destroyed the model's fluency in the n=64 single-context run
        (probe NLL 3.57 -> 10.4).  Adding the dataset's paraphrase prompts
        (and, when present, the generation prompts) makes the solve
        over-determined instead of exactly determined.
        """
        prompt = request["prompt"]
        if "{}" not in prompt:
            subject = request.get("subject") or ""
            prompt = prompt.replace(subject, "{}") if subject and subject in prompt else prompt + " {}"
        subject = request.get("subject") or "#"
        contexts = [self._brace_safe(prompt)]
        for field in ("rephrase_prompts", "rephrase_prompt", "paraphrase_prompts",
                      "generation_prompts"):
            extra = request.get(field)
            if not extra:
                continue
            if isinstance(extra, str):
                extra = [extra]
            for text in extra:
                if not isinstance(text, str) or not text.strip():
                    continue
                if subject not in text and "{}" not in text:
                    continue          # cannot anchor the subject -> useless key
                tpl = self._brace_safe(text, subject)
                if tpl not in contexts:
                    contexts.append(tpl)
            if len(contexts) >= self.max_contexts:
                break
        return contexts[: self.max_contexts]

    @staticmethod
    def _brace_safe(text: str, subject: str = "") -> str:
        """Turn a raw sentence into a ``str.format`` template.

        CounterFact prompts occasionally contain literal braces, which makes
        ``format`` raise ``unmatched '{' in format spec``.  Route the subject
        slot through a sentinel so every other brace can be escaped.
        """
        sentinel = "\x00SUBJ\x00"
        if subject and subject in text:
            text = text.replace(subject, sentinel)
        elif "{}" in text:
            text = text.replace("{}", sentinel)
        text = text.replace("{", "{{").replace("}", "}}")
        return text.replace(sentinel, "{}")

    # ------------------------------------------------------------------ #
    # Core: closed-form solve
    # ------------------------------------------------------------------ #

    def _weight_name(self, layer: int) -> str:
        return f"{self.rewrite_module_tmp.format(layer)}.weight"

    def _layer_keys(self, mreq: Dict[str, Any], layer: int,
                    flat_contexts: List[str]) -> torch.Tensor:
        """Keys ``K`` [d_in, n_ctx] = current input acts of the rewrite module."""
        words = [mreq["subject"] for _ in flat_contexts]
        ks = get_module_input_output_at_words(
            self.model, self.tok, layer,
            context_templates=flat_contexts,
            words=words,
            module_template=self.rewrite_module_tmp,
            fact_token_strategy=self.hparams.fact_token,
        )[0]                      # [n_ctx, d_in]
        return ks.T.detach().float()   # [d_in, n_ctx]

    def _solve(self, keys: torch.Tensor, resid: torch.Tensor,
               layer: int) -> torch.Tensor:
        """Ridge-regularised closed form: ``R K^T (K K^T + lambda I)^-1``."""
        d_in = keys.shape[0]
        eye = torch.eye(d_in, dtype=keys.dtype, device=keys.device)
        gram = keys @ keys.T
        if self.kv_cache and self.null_space != "keys" and layer in self._key_cache:
            gram = gram + self._key_cache[layer].to(gram.device, gram.dtype)
        lam = self._lambda(gram, int(resid.shape[1]))
        upd = torch.linalg.solve(gram + lam * eye, keys @ resid.T)
        return upd

    def _project_to_key_null_space(self, upd: torch.Tensor, layer: int) -> torch.Tensor:
        """AlphaEdit-lite: remove the component that would disturb old keys.

        AlphaEdit projects every weight update onto the null space of the
        corpus-level key matrix, ``P = lam (K K^T + lam I)^-1``, so that new
        edits do not overwrite earlier ones.  The official P needs offline
        Wikipedia covariance statistics that this repo does not carry, but the
        same projection can be built from the keys of the edits we have already
        written: ``G = sum_j K_j K_j^T`` is exactly the running statistics kept
        by ``kv_cache``.  ``upd`` is [d_in, d_out], so P acts on the key side.
        """
        gram = self._key_cache.get(layer)
        if gram is None or self._acc_n.get(layer, 0) <= 0:
            return upd
        gram = gram.to(upd.device, upd.dtype)
        coef = self.l2 if self.proj_l2 < 0 else self.proj_l2
        lam = self._lambda(gram, self._acc_n[layer]) * (coef / max(self.l2, 1e-12))
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        # P @ upd  ==  lam * (G + lam I)^-1 @ upd
        projected = torch.linalg.solve(gram + lam * eye, lam * upd)
        if self.proj_mix >= 1.0:
            return projected
        return (1.0 - self.proj_mix) * upd + self.proj_mix * projected

    def _lambda(self, gram: torch.Tensor, n_ctx: int) -> float:
        """Ridge strength actually used.

        ``abs`` reproduces MEMIT's literal ``mom2_update_weight``; it is only
        sensible when the Gram matrix comes from a large activation corpus.
        Here the keys are the ~6 in-context c_proj inputs of one edit, whose
        mean diagonal is ~0.2 -- so an absolute 15000 regularises the problem
        into "do nothing" (rewrite hits collapse to 12%) while an absolute 10
        is effectively no regularisation at all.  ``rel`` scales lambda to the
        observed Gram scale instead: lambda = coef * mean(diag(K K^T)).
        """
        if self.l2_mode == "abs":
            return float(self.l2)
        # Scale to the *eigenvalue* level of K K^T, not to its mean diagonal:
        # the informative eigenvalues are ~|k|^2 = trace(gram)/n_ctx (hundreds
        # for GPT-2 activations), so a lambda derived from mean(diag) (~0.2)
        # is numerically invisible and the solve is effectively unregularised.
        n_ctx = max(1, n_ctx)
        mean_eig = float(gram.trace().item()) / n_ctx
        return float(self.l2) * max(mean_eig, 1e-8)

    def _upd_to_param_shape(self, matrix: torch.Tensor,
                            weight: torch.Tensor) -> torch.Tensor:
        """GPT-2 stores ``c_proj.weight`` as [out, in]; solve yields [in, out]."""
        if matrix.shape == weight.shape:
            return matrix
        if matrix.T.shape == weight.shape:
            return matrix.T
        raise ValueError(
            f"[CALM-CF] solved update shape {tuple(matrix.shape)} does not "
            f"match weight shape {tuple(weight.shape)}"
        )

    def generate_update(self, request: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Return ``{param_name: delta}`` for one edit (model left unchanged).

        MEMIT computes layer ``i+1``'s keys *after* layer ``i`` has been
        updated, so the solve loop applies each update tentatively and the
        original weights are restored before returning.  Because every weight
        receives exactly one delta, re-applying the returned dict in
        ``apply_update`` reproduces the sequential-loop end state exactly.
        """
        t0 = time.time()
        mreq = self._as_mit_request(request)
        contexts = self._context_templates(request)      # z optimisation
        flat_contexts = self._key_contexts(request)      # keys + residual
        z_layer = self.layers[-1]

        # 1) value vector z: optimise a residual-stream vector at z_layer that
        #    makes the final layer predict target_new.
        z = compute_z(self.model, self.tok, mreq, self.hparams, z_layer, contexts)

        # 1b) norm clamp.  ``compute_z`` is an unconstrained Adam ascent and on
        #     gpt2-small (|h_8| ~ 100 for ordinary prompts) it routinely returns
        #     |z| ~ 300-1400, i.e. an activation 3-14x off-manifold.  Writing such
        #     a target demands deltas larger than the activation they act on,
        #     which flips the answer but wrecks fluency.  Rescale z to a small
        #     multiple of the observed activation norm instead.
        z_norm_raw = float(z.norm().item())
        if self.z_clamp > 0:
            with torch.no_grad():
                h_ref = get_module_input_output_at_words(
                    self.model, self.tok, z_layer,
                    context_templates=flat_contexts,
                    words=[mreq["subject"] for _ in flat_contexts],
                    module_template=self.layer_module_tmp,
                    fact_token_strategy=self.hparams.fact_token,
                )[1].T.detach().float()          # [d_model, n_ctx]
                h_norm = float(h_ref.norm(dim=0).mean().item())
            cap = self.z_clamp * max(h_norm, 1e-6)
            if z_norm_raw > cap:
                z = z * (cap / z_norm_raw)
        z_norm_used = float(z.norm().item())

        deltas: Dict[str, torch.Tensor] = {}
        diag: Dict[str, Any] = {"z_norm_raw": round(z_norm_raw, 1), "z_norm_used": round(z_norm_used, 1)}
        snapshots: Dict[str, torch.Tensor] = {}
        keys_per_layer: Dict[int, torch.Tensor] = {}

        try:
            for i, layer in enumerate(self.layers):
                keys = self._layer_keys(mreq, layer, flat_contexts)
                keys_per_layer[layer] = keys

                # 2) residual error: (target value) - (current act).
                # Canonical MEMIT measures the *reference* activation ONCE at
                # z_layer and then splits (z - h_z) over the edited layers by a
                # 1/(N-i) schedule.  Measuring per-layer (z - h_i) instead makes
                # every layer solve the full remaining error, so the deltas sum
                # to roughly 2-3x the correction actually needed -> the rewrite
                # succeeds but the model is pushed far off-manifold (NLL blow-up,
                # degenerate repetition).  ``resid_ref="cur_layer"`` keeps the
                # old behaviour for ablation.
                ref_layer = z_layer if self.resid_ref == "z_layer" else layer
                cur = get_module_input_output_at_words(
                    self.model, self.tok, ref_layer,
                    context_templates=flat_contexts,
                    words=[mreq["subject"] for _ in flat_contexts],
                    module_template=self.layer_module_tmp,
                    fact_token_strategy=self.hparams.fact_token,
                )[1].T.detach().float()          # [d_model, n_ctx]
                n_ctx = cur.shape[1]
                target = z.to(cur.device, cur.dtype).reshape(-1, 1).repeat(1, n_ctx)
                resid = (target - cur) / float(len(self.layers) - i)

                gram_trace = float((keys @ keys.T).diagonal().mean().item())
                upd = self._solve(keys, resid, layer)
                proj_gain = None
                if self.null_space == "keys":
                    before = float(upd.norm().item())
                    upd = self._project_to_key_null_space(upd, layer)
                    proj_gain = round(float(upd.norm().item()) / max(before, 1e-12), 4)
                diag.setdefault("gram_trace", {})[layer] = round(gram_trace, 1)
                diag.setdefault("resid_norm", {})[layer] = round(float(resid.norm().item()), 3)
                if proj_gain is not None:
                    diag.setdefault("proj_gain", {})[layer] = proj_gain
                weight = nethook.get_parameter(self.model, self._weight_name(layer))
                upd = self._upd_to_param_shape(upd, weight).to(weight.device, weight.dtype)
                if self.delta_scale != 1.0:
                    upd = upd * self.delta_scale
                upd = self._limit_delta(upd, weight)

                snapshots[self._weight_name(layer)] = weight.detach().clone()
                with torch.no_grad():
                    weight.add_(upd)
                deltas[self._weight_name(layer)] = upd.detach().cpu()
        finally:
            # restore: apply_update() will re-apply the returned deltas
            with torch.no_grad():
                for name, original in snapshots.items():
                    nethook.get_parameter(self.model, name).data.copy_(original)

        # 3) AlphaEdit key statistics bookkeeping (post-write keys)
        if self.kv_cache:
            for layer, keys in keys_per_layer.items():
                self._acc_n[layer] = self._acc_n.get(layer, 0) + int(keys.shape[1])
                kk = (keys @ keys.T).detach().cpu()
                if layer in self._key_cache:
                    self._key_cache[layer] = self._key_cache[layer] + kk
                else:
                    self._key_cache[layer] = kk

        self._last_deltas = deltas
        self._last_meta = {
            "z_norm": float(z.detach().norm().item()),
            "solve_seconds": round(time.time() - t0, 3),
            "layers": list(self.layers),
            "n_contexts": len(flat_contexts),
            "l2_mode": self.l2_mode,
            "delta_scale": self.delta_scale,
            "resid_ref": self.resid_ref,
            "z_clamp": self.z_clamp,
            "null_space": self.null_space,
            "proj_gain": diag.get("proj_gain", {}),
            "proj_mix": self.proj_mix,
            "proj_l2": self.proj_l2,
            "l2_over_gram_trace": {
                int(layer): round(self.l2 / max(1e-9, g), 4)
                for layer, g in (diag.get("gram_trace") or {}).items()
            },
            "gram_trace": diag.get("gram_trace", {}),
            "resid_norm": diag.get("resid_norm", {}),
        }
        return deltas

    def generate_update_batch(self, requests: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Joint closed-form solve for a whole batch of edits.

        Sequential single-key solves overwrite each other: edit 500 re-solves
        the same key directions that edits 1..499 already used, so the oldest
        writes are always the losers (measured ``recheck 0/50`` at n=500).
        Solving a chunk at once makes every edit in it a simultaneous
        constraint of one least-squares problem, so conflicting residuals are
        distributed by the solve instead of being reassigned by arrival order.
        One delta per parameter is produced for the whole batch, then applied
        in a single ``apply_update`` call.
        """
        t0 = time.time()
        if not requests:
            return {}
        preps: List[Dict[str, Any]] = []
        for request in requests:
            mreq = self._as_mit_request(request)
            flat = self._key_contexts(request)
            z = compute_z(self.model, self.tok, mreq, self.hparams,
                          self.layers[-1], self._context_templates(request))
            z_raw = float(z.norm().item())
            if self.z_clamp > 0:
                with torch.no_grad():
                    h_ref = get_module_input_output_at_words(
                        self.model, self.tok, self.layers[-1],
                        context_templates=flat,
                        words=[mreq["subject"] for _ in flat],
                        module_template=self.layer_module_tmp,
                        fact_token_strategy=self.hparams.fact_token,
                    )[1].T.detach().float()
                    h_norm = float(h_ref.norm(dim=0).mean().item())
                cap = self.z_clamp * max(h_norm, 1e-6)
                if z_raw > cap:
                    z = z * (cap / z_raw)
            preps.append({"mreq": mreq, "flat": flat, "z": z, "z_raw": z_raw})

        n_layers = len(self.layers)
        deltas: Dict[str, torch.Tensor] = {}
        snapshots: Dict[str, torch.Tensor] = {}
        keys_per_layer: Dict[int, torch.Tensor] = {}
        chunk_rows: Dict[int, tuple] = {}
        try:
            for i, layer in enumerate(self.layers):
                ks, rs = [], []
                for p in preps:
                    ks.append(self._layer_keys(p["mreq"], layer, p["flat"]))
                    ref_layer = self.layers[-1] if self.resid_ref == "z_layer" else layer
                    cur = get_module_input_output_at_words(
                        self.model, self.tok, ref_layer,
                        context_templates=p["flat"],
                        words=[p["mreq"]["subject"] for _ in p["flat"]],
                        module_template=self.layer_module_tmp,
                        fact_token_strategy=self.hparams.fact_token,
                    )[1].T.detach().float()
                    n_ctx = cur.shape[1]
                    target = p["z"].to(cur.device, cur.dtype).reshape(-1, 1).repeat(1, n_ctx)
                    rs.append((target - cur) / float(n_layers - i))
                keys = torch.cat(ks, dim=1)
                resid = torch.cat(rs, dim=1)
                chunk_rows[layer] = (keys.detach().cpu(), resid.detach().cpu())
                if self.replay_cache and layer in self._replay:
                    prev_k, prev_r = self._replay[layer]
                    keys = torch.cat([prev_k.to(keys.device), keys], dim=1)
                    resid = torch.cat([prev_r.to(resid.device), resid], dim=1)
                keys_per_layer[layer] = keys
                upd = self._solve(keys, resid, layer)
                if self.null_space == "keys":
                    upd = self._project_to_key_null_space(upd, layer)
                weight = nethook.get_parameter(self.model, self._weight_name(layer))
                upd = self._upd_to_param_shape(upd, weight).to(weight.device, weight.dtype)
                if self.delta_scale != 1.0:
                    upd = upd * self.delta_scale
                upd = self._limit_delta(upd, weight)
                snapshots[self._weight_name(layer)] = weight.detach().clone()
                with torch.no_grad():
                    weight.add_(upd)
                deltas[self._weight_name(layer)] = upd.detach().cpu()
        finally:
            with torch.no_grad():
                for name, original in snapshots.items():
                    nethook.get_parameter(self.model, name).data.copy_(original)

        if self.kv_cache:
            for layer, keys in keys_per_layer.items():
                self._acc_n[layer] = self._acc_n.get(layer, 0) + int(keys.shape[1])
                kk = (keys @ keys.T).detach().cpu()
                if layer in self._key_cache:
                    self._key_cache[layer] = self._key_cache[layer] + kk
                else:
                    self._key_cache[layer] = kk

        if self.replay_cache:
            for layer, (k_new, r_new) in chunk_rows.items():
                if layer in self._replay:
                    pk, pr = self._replay[layer]
                    self._replay[layer] = (torch.cat([pk, k_new], dim=1),
                                           torch.cat([pr, r_new], dim=1))
                else:
                    self._replay[layer] = (k_new, r_new)

        self._last_deltas = deltas
        self._last_meta = {
            "z_norm": [round(float(p["z"].detach().norm().item()), 1) for p in preps],
            "batch_size": len(preps),
            "solve_seconds": round(time.time() - t0, 3),
            "layers": list(self.layers),
            "n_contexts": sum(int(p["z"].numel() > 0) * len(p["flat"]) for p in preps),
            "l2_mode": self.l2_mode,
            "delta_scale": self.delta_scale,
            "delta_rank": self.delta_rank,
            "delta_cap": self.delta_cap,
            "resid_ref": self.resid_ref,
            "z_clamp": self.z_clamp,
            "null_space": self.null_space,
        }
        return deltas

    def _limit_delta(self, upd: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Low-rank truncation + Frobenius cap on one layer's solved delta."""
        if self.delta_rank > 0 and min(upd.shape) > self.delta_rank:
            u, s, vh = torch.linalg.svd(upd, full_matrices=False)
            r = self.delta_rank
            upd = (u[:, :r] * s[:r]) @ vh[:r]
        if self.delta_cap > 0:
            cap = self.delta_cap * float(torch.norm(weight.detach(), p="fro").item())
            dnorm = float(torch.norm(upd, p="fro").item())
            if dnorm > cap > 0:
                upd = upd * (cap / dnorm)
        return upd

    def apply_update(self, deltas: Dict[str, torch.Tensor]) -> None:
        """Add the solved deltas in place to the base model weights."""
        stats: Dict[str, Dict[str, float]] = {}
        with torch.no_grad():
            for name, delta in deltas.items():
                param = nethook.get_parameter(self.model, name)
                before = float(torch.norm(param.data, p="fro").item())
                dnorm = float(torch.norm(delta, p="fro").item())
                param.data.add_(delta.to(device=param.device, dtype=param.dtype))
                after = float(torch.norm(param.data, p="fro").item())
                stats[name] = {
                    "delta_norm": dnorm,
                    "weight_norm_before": before,
                    "weight_norm_after": after,
                    "rel_delta": dnorm / max(before, 1e-12),
                }
        self.last_apply_stats = stats
        self.edit_count += 1

    def generate_and_apply(self, request: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        deltas = self.generate_update(request)
        self.apply_update(deltas)
        return deltas

    # ------------------------------------------------------------------ #
    # Instrumentation / interop
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict[str, Any]:
        """Compact description of the writer state (for audit dumps)."""
        return {
            "writer": "closedform",
            "layers": list(self.layers),
            "l2": self.l2,
            "kv_cache": self.kv_cache,
            "null_space": self.null_space,
            "edits_applied": self.edit_count,
            "last_meta": getattr(self, "_last_meta", {}),
        }
