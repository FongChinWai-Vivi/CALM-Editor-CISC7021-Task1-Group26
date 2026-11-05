"""
hypernetwork.py -- CALM Hypernetwork Edit Generator (W3, Fong Chin Wai).

Implements the MEND/MALMEN-style low-rank weight update generator that the
CALMEditor invokes on the WRITE path.  Given a single
subject-relation-object edit request, the generator maps the edit text into a
low-rank factor pair (U, V) for each editable MLP parameter and returns the
additive delta ``scale * U @ V^T`` -- no per-edit gradient descent on the
full model is required.

Backends
--------
* ``lowrank`` (default): a self-contained MLP hypernetwork.  The edit feature
  is the mean token embedding of ``prompt + target_new`` through the base
  model's input embedding; a shared backbone + per-parameter linear head
  emits U [out, r] and V [in, r] for every inner parameter.
* ``mend``: delegates to EasyEdit's trained MEND archive (GradientTransform /
  IDMLP low-rank factors) via ``MendRewriteExecutor`` -- wired up in
  ``calm_editor.CALMEditor`` when ``hparams.hypernet_alg == "mend"``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...util import nethook
from ...util.device import get_model_device


def _matrix_shapes(model: nn.Module, inner_params: List[str]) -> List[Tuple[str, int, int]]:
    """Return ``(name, out_dim, in_dim)`` for each 2-D inner parameter."""
    shapes: List[Tuple[str, int, int]] = []
    for name in inner_params:
        param = nethook.get_parameter(model, name)
        w = param.detach()
        if w.dim() == 2:
            out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
            shapes.append((name, out_dim, in_dim))
        else:
            # 1-D biases are not low-rank factored; skip them.
            print(f"[CALM] skipping non-matrix inner parameter: {name} {tuple(w.shape)}")
    return shapes


class LowRankHypernetwork(nn.Module):
    """MLP hypernetwork that emits rank-r factor pairs for inner parameters.

    Architecture: a shared backbone (``feature -> hidden -> ReLU``) plus one
    linear head per inner parameter.  Each head produces
    ``out_dim * r + in_dim * r`` values that are reshaped into
    U [out, r] and V [in, r]; the generated delta is ``scale * U @ V^T``.
    """

    def __init__(self, feature_dim: int, inner_shapes: List[Tuple[str, int, int]],
                 rank: int = 4, hidden: int = 512, scale: float = 1.0,
                 seed: int = 42):
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank
        self.scale = scale
        self.inner_shapes = inner_shapes

        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
        )
        # Post-hoc delta control (W4 norm-growth experiments, 2026-11-02):
        # norm_cap rescales each emitted delta so that ||delta||_F <= norm_cap
        # at inference (locality-safe by construction); track_norms records
        # per-parameter emitted norms in self.last_norms for instrumentation.
        # norm_tau rescales each delta to ||delta||_F <= tau * base_norm(param),
        # matching the differentiable cap used during W4 re-training.
        self.norm_cap: Optional[float] = None
        self.norm_tau: Optional[float] = None
        self.base_norms: Optional[Dict[str, float]] = None
        self.track_norms: bool = False
        self.last_norms: Dict[str, float] = {}
        # nn.ModuleDict keys must not contain ".", so we use positional keys
        # and keep a name -> index map for the forward pass.
        self.heads = nn.ModuleDict()
        self._name_to_idx: Dict[str, int] = {}
        for i, (name, out_dim, in_dim) in enumerate(inner_shapes):
            self.heads[f"head_{i}"] = nn.Linear(hidden, out_dim * rank + in_dim * rank)
            self._name_to_idx[name] = i

        torch.manual_seed(seed)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map an edit feature ``[B, feature_dim]`` to per-parameter deltas."""
        h = self.backbone(feature)                       # [B, hidden]
        deltas: Dict[str, torch.Tensor] = {}
        for name, out_dim, in_dim in self.inner_shapes:
            idx = self._name_to_idx[name]
            raw = self.heads[f"head_{idx}"](h)           # [B, out*r + in*r]
            u = raw[:, : out_dim * self.rank].view(-1, out_dim, self.rank)
            v = raw[:, out_dim * self.rank:].view(-1, in_dim, self.rank)
            delta = self.scale * torch.bmm(u, v.transpose(1, 2))   # [B, out, in]
            delta = delta.squeeze(0) if delta.size(0) == 1 else delta
            cap = self.norm_cap
            if isinstance(cap, dict):
                cap = cap.get(name)
            if cap is not None:
                dnorm = float(torch.norm(delta, p="fro").item())
                if dnorm > cap:
                    delta = delta * (cap / dnorm)
            if self.norm_tau is not None and self.base_norms is not None:
                base = self.base_norms.get(name)
                if base is not None:
                    dnorm = float(torch.norm(delta, p="fro").item())
                    limit = self.norm_tau * base
                    if dnorm > limit:
                        delta = delta * (limit / dnorm)
            if self.track_norms:
                self.last_norms[name] = float(torch.norm(delta, p="fro").item())
            deltas[name] = delta
        return deltas

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HypernetworkEditGenerator:
    """MEND/MALMEN-style low-rank edit generator bound to a base model.

    Generates weight deltas from an edit request without per-edit gradient
    descent.  The feature representation is the mean token embedding of
    ``prompt + target_new`` through the base model's input embedding.
    """

    def __init__(self, model: nn.Module, tok, inner_params: List[str],
                 rank: int = 4, hidden: int = 512, scale: float = 1.0,
                 seed: int = 42, max_length: int = 64,
                 ckpt: Optional[str] = None,
                 feature_mode: str = "mean-emb"):
        self.model = model
        self.tok = tok
        self.inner_params = list(inner_params)
        self.max_length = max_length
        # Claude #3 (2026-11-02): key-conditioned feature modes.
        #   "mean-emb"  : mean token embedding of "prompt target_new" (original)
        #   "key-hidden": last-token hidden state of the *subject* through the
        #                 penultimate transformer block (key-conditioned).
        # Both modes emit a [1, d] vector with d == embedding dim, so the
        # low-rank hypernetwork architecture is unchanged; only the input
        # signal is sharper.  Training must use the same mode as inference.
        self.feature_mode = feature_mode

        emb = model.get_input_embeddings()
        feature_dim = int(emb.weight.shape[1])
        shapes = _matrix_shapes(model, inner_params)
        if not shapes:
            raise ValueError("[CALM] hypernetwork: no 2-D inner parameters found.")

        self.hypernet = LowRankHypernetwork(
            feature_dim, shapes, rank=rank, hidden=hidden, scale=scale, seed=seed,
        )
        self.device = get_model_device(model)
        self.hypernet.to(self.device)

        # Per-edit norm instrumentation (W4, 2026-11-02): when track_norms is
        # True, apply_update records per-(edit,param) delta / param norms.
        self.track_norms: bool = False
        self._norm_log: List[Dict[str, Any]] = []

        # W4 (2026-11-02): null-space projection (AlphaEdit-style).  When
        # enabled, each emitted delta is projected onto the null space of the
        # most recent written deltas (per inner param) so new writes cannot
        # clobber earlier edits' directions; retention-per-slot improves
        # without retraining.  ``ns_keep`` bounds the number of stacked delta
        # directions (memory: ns_keep * param_size floats per inner param).
        # Projection is applied gently: only when the new delta overlaps
        # strongly with a previous direction (cosine >= ns_cos), and the
        # projected delta keeps at least ``ns_min_keep`` of its original norm.
        self.ns_proj: bool = False
        self.ns_keep: int = 8
        self.ns_cos: float = 0.60
        self.ns_min_keep: float = 0.60
        self._ns_stack: Dict[str, List[torch.Tensor]] = {}
        self._ns_proj_stats: Dict[str, float] = {}

        # Optional meta-trained checkpoint (W3 enhancement).  The checkpoint is
        # a plain ``LowRankHypernetwork.state_dict()`` plus a small metadata
        # dict; when ``ckpt`` is None the generator keeps its untrained
        # behaviour exactly as before.
        if ckpt is not None:
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"[CALM] hypernet checkpoint not found: {ckpt}")
            data = torch.load(ckpt, map_location=self.device, weights_only=False)
            meta = data.get("meta", {})
            if meta:
                for key, expected in (
                    ("rank", rank), ("hidden", hidden),
                    ("feature_dim", feature_dim),
                ):
                    loaded = meta.get(key)
                    if loaded is not None and int(loaded) != int(expected):
                        raise ValueError(
                            f"[CALM] checkpoint meta mismatch for {key}: "
                            f"loaded={loaded} expected={expected}"
                        )
                loaded_mode = meta.get("feature_mode")
                if loaded_mode is not None and loaded_mode != self.feature_mode:
                    raise ValueError(
                        f"[CALM] checkpoint feature_mode mismatch: "
                        f"loaded={loaded_mode} expected={self.feature_mode} -- "
                        f"train and evaluate with the same --feature-mode"
                    )
                # The trained checkpoint may use a different global scale
                # (training amplifies the low-rank factors); adopt it so the
                # deployed generator behaves exactly like the trained one.
                if meta.get("scale") is not None:
                    self.hypernet.scale = float(meta["scale"])
            state = data.get("state_dict", data)  # tolerate both forms
            self.hypernet.load_state_dict(state)
            print(f"[CALM] loaded hypernetwork checkpoint: {ckpt} "
                  f"(scale={self.hypernet.scale})")

    # -- feature construction -------------------------------------------- #

    def edit_feature(self, prompt: str, target_new: str,
                     subject: Optional[str] = None) -> torch.Tensor:
        """Build the edit feature vector [1, d].

        ``mean-emb`` (default): mean token embedding of "prompt target_new".
        ``key-hidden`` (Claude #3): hidden state of the subject's last token
        from the penultimate transformer layer.  The feature is computed from
        the *prompt* (which contains the subject) so the conditioning signal
        reflects the key itself, not the edited target.
        """
        if self.feature_mode == "key-hidden":
            if not subject:
                raise ValueError("[CALM] key-hidden mode requires a subject "
                                 "in the edit request")
            ids = self.tok(prompt, return_tensors="pt", truncation=True,
                           max_length=self.max_length).input_ids.to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=ids,
                                 output_hidden_states=True,
                                 use_cache=False)
                # hidden_states: [embed, h0, ..., hL-1]; penultimate layer.
                hs = out.hidden_states[-2]                 # [1, L, d]
            # Locate the subject's token span inside the prompt tokens by
            # sub-sequence matching, then take the LAST subject token index.
            sub_ids = self.tok(subject, return_tensors="pt",
                               truncation=True,
                               max_length=self.max_length).input_ids[0]
            pid = ids[0].tolist()
            sid = sub_ids.tolist()
            last_match = -1
            if sid:
                for i in range(len(pid) - len(sid) + 1):
                    if pid[i:i + len(sid)] == sid:
                        last_match = i + len(sid) - 1
            if last_match < 0:
                # Fall back to the last token before any separator tokens.
                last_match = min(len(pid) - 1, len(sid) - 1)
            feat = hs[:, last_match, :]                   # [1, d]
            # Raw penultimate hidden states have ~100x larger magnitude than
            # token embeddings (std ~8 vs ~0.07), which explodes the linear
            # backbone.  LayerNorm to unit variance keeps the key-conditioned
            # direction signal while matching the mean-emb scale regime.
            return torch.nn.functional.layer_norm(
                feat, feat.shape[-1:])                    # [1, d]
        text = f"{prompt} {target_new}".strip()
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_length).input_ids.to(self.device)
        with torch.no_grad():
            emb = self.model.get_input_embeddings()(ids)   # [1, L, d]
        return emb.mean(dim=1)                             # [1, d]

    # -- public API ------------------------------------------------------- #

    def generate_update(self, request: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Return ``{param_name: delta}`` for one edit request."""
        prompt = request.get("prompt", "")
        target_new = request.get("target_new", "")
        feature = self.edit_feature(prompt, target_new,
                                    subject=request.get("subject"))
        with torch.no_grad():
            deltas = self.hypernet(feature)
        if self.ns_proj:
            deltas = self._null_space_project(deltas)
        return deltas

    # -- null-space projection (AlphaEdit-style, W4) ----------------------- #

    def _null_space_project(
        self, deltas: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Project each delta onto the null space of recently written deltas.

        For inner param ``name`` with stack K = [d_1; ...; d_m] (flattened
        previously written deltas), the new flattened delta d is replaced by
        ``d - K^T (K K^T + eps I)^-1 K d``, i.e. the component orthogonal to
        the span of previous edits.  If the projection shrinks the delta too
        much (the direction is nearly exhausted), a partial blend is used so
        the edit still lands with meaningful strength.
        """
        out: Dict[str, torch.Tensor] = {}
        for name, delta in deltas.items():
            stack = self._ns_stack.get(name)
            if not stack:
                out[name] = delta
                continue
            dvec = delta.reshape(-1).to(delta.dtype)
            K = torch.stack(stack).to(device=delta.device, dtype=delta.dtype)
            m = K.size(0)
            # Overlap with the most recent previous direction; if the new
            # delta is not strongly aligned, leave it untouched (no benefit).
            recent = K[-1]
            dnorm = float(dvec.norm().item())
            rnorm = float(recent.norm().item())
            if dnorm <= 1e-12 or rnorm <= 1e-12:
                out[name] = delta
                continue
            cos_sim = float((dvec @ recent).item() / (dnorm * rnorm))
            if cos_sim < self.ns_cos:
                out[name] = delta
                continue
            eps = 1e-4 * K.norm() ** 2 / max(1, m * K.size(1))
            KKt = K @ K.t() + eps * torch.eye(m, device=K.device, dtype=K.dtype)
            coeff = torch.linalg.solve(KKt, K @ dvec)       # [m]
            proj = dvec - (K.t() @ coeff)                   # null-space comp.
            orig_norm = dnorm
            proj_norm = float(proj.norm().item())
            keep = float(proj_norm / max(orig_norm, 1e-12))
            if keep < self.ns_min_keep:
                # Direction nearly exhausted; blend so the edit still lands.
                scale = self.ns_min_keep * orig_norm / max(proj_norm, 1e-12)
                proj = proj * scale
                keep = self.ns_min_keep
            out[name] = proj.reshape_as(delta)
            self._ns_proj_stats[name] = keep
        return out

    def _ns_commit(self, deltas: Dict[str, torch.Tensor]) -> None:
        """Append applied deltas to the per-param null-space stacks."""
        for name, delta in deltas.items():
            if name not in self._ns_stack:
                self._ns_stack[name] = []
            self._ns_stack[name].append(delta.detach().reshape(-1).cpu())
            if len(self._ns_stack[name]) > self.ns_keep:
                del self._ns_stack[name][0]

    def apply_update(self, deltas: Dict[str, torch.Tensor]) -> None:
        """Add deltas in-place to the base model's inner parameters."""
        with torch.no_grad():
            for name, delta in deltas.items():
                param = nethook.get_parameter(self.model, name)
                param.data.add_(delta.to(device=param.device, dtype=param.dtype))
                if self.track_norms:
                    base_norm = float(torch.norm(param.data, p="fro").item())
                    self._norm_log.append({
                        "name": name,
                        "delta_norm": float(torch.norm(delta, p="fro").item()),
                        "param_norm_after": base_norm,
                    })
        if self.ns_proj:
            self._ns_commit(deltas)

    def generate_and_apply(self, request: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        deltas = self.generate_update(request)
        self.apply_update(deltas)
        return deltas
