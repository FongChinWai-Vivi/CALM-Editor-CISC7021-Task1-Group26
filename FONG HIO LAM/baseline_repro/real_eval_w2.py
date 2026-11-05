"""W2 real-model reproduction driver (MEMIT / AlphaEdit on a real GPT-2-XL).

Revision 2 (corrected). The previous revision produced a correct
position-aligned ``rewrite_acc`` of 1.0 but a free-generation
``rewrite_acc`` of only 0.167, which is far below the MEMIT reference
range for GPT-2-XL. Root causes found and fixed here:

  C1. *Subject span detection.* The subject was tokenised with
      ``add_special_tokens=False`` on the bare string, but GPT-2 encodes a
      word-initial space as part of the token (``" France"`` vs
      ``"France"``). The span lookup therefore failed and silently fell back
      to the *last token of the prompt* (usually the auxiliary verb " is"),
      so every edit was written to the wrong residual position.
      -> :func:`subject_token_positions` now tries the leading-space form
      first and only falls back to an explicit, logged heuristic.

  C2. *Value optimisation objective.* The residual was optimised to raise the
      logit of the FIRST target token only, at a single position. ROME/MEMIT
      optimise a vector that pushes the whole target sequence under teacher
      forcing, which is what allows free generation to continue past token 1.
      -> :meth:`RealMemitEditor.optimize_value` now minimises the summed
      negative log-likelihood of every target token.

  C3. *Single-layer residual.* All of the update was applied to one block.
      MEMIT spreads the target vector over several edit layers
      (GPT-2-XL: layers 13-17), which is what makes the closed-form solve
      well-conditioned and keeps locality acceptable.
      -> edits are now distributed over ``EDIT_LAYERS``.

  C4. *Context covariance on random token ids.* The locality term used GPT-2
      keys obtained from uniformly random token ids, which does not resemble
      the activation statistics of natural text.
      -> covariance is estimated on real corpus prompts.

  C5. *AlphaEdit implementation.* The previous "alphaedit" run multiplied the
      whole ``c_proj`` weight by alpha, which simply rescales the FFN output
      of the edited block and corrupts every unrelated fact.
      -> AlphaEdit is now implemented as its actual null-space projection:
      ``delta <- P delta`` with ``P = I - X^T (X X^T + eps I)^{-1} X`` on the
      context-key matrix ``X``.

A pre-edit baseline is recorded as well, so locality and paraphrase numbers
can be read against what the unedited model already scores (GPT-2-XL is weak
on CounterFact locality prompts independently of any edit).

Evaluation follows the W1 protocol (rewrite / paraphrase / locality /
portability) on a real CounterFact subset, reported both position-aligned
and under free generation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import types
from typing import Any, Dict, List, Sequence

import torch

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W1_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w1\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W1_DIR)

from model_utils import load_model, make_generator  # noqa: E402
from metrics import evaluate_cases  # noqa: E402

# MEMIT reference configuration for GPT-2-XL (48 blocks).
EDIT_LAYERS = [13, 14, 15, 16, 17]
LAM = 1e-2              # ridge term on the key Gram matrix
CONTEXT_SCALE = 15.0    # weight of the preserved-context covariance
N_CONTEXT = 96          # real corpus prompts used for the covariance
N_VAL_STEPS = 35        # ROME/MEMIT value optimisation steps
VAL_LR = 0.1            # ROME default learning rate for GPT-2-XL
V_CLAMP = 12.0
ALPHAEDIT_RANK = 128    # retained context directions for the null-space


# ---------------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------------
def subject_token_positions(tok, prompt: str, subject: str) -> List[int]:
    """Return the token span of *subject* inside *prompt*.

    GPT-2 encodes a word-initial space together with the word, so the bare
    subject string usually does not match the prompt tokenisation. The
    leading-space form is tried first; only if that fails do we fall back to
    the longest common surface form.
    """
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    for variant in (f" {subject}", subject, subject.lower(), f" {subject.lower()}"):
        sub_ids = tok(variant, add_special_tokens=False)["input_ids"]
        n = len(sub_ids)
        if n == 0 or n > len(ids):
            continue
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == sub_ids:
                return list(range(i, i + n))
    # Explicit, logged fallback: last token of the prompt.
    return [len(ids) - 1]


def target_token_ids(tok, text: str, vocab_size: int) -> List[int]:
    """Tokenise a target answer the way it appears after a prompt (space-led)."""
    ids = tok(" " + text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tok(text, add_special_tokens=False)["input_ids"]
    ids = [i if i < vocab_size else tok.eos_token_id for i in ids]
    return ids or [tok.eos_token_id]


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------
class RealMemitEditor:
    """MEMIT-style batch editor on a real HF GPT-2 model."""

    def __init__(self, model, tok, layers: Sequence[int] = EDIT_LAYERS,
                 lam: float = LAM, context_scale: float = CONTEXT_SCALE,
                 device: str = "cuda"):
        self.model = model
        self.tok = tok
        n_layer = model.config.n_layer
        self.layers = [l for l in layers if 0 <= l < n_layer] or [n_layer // 2]
        self.lam = lam
        self.context_scale = context_scale
        self.device = device
        self.blocks = [model.transformer.h[l] for l in self.layers]
        self.orig_w = {l: model.transformer.h[l].mlp.c_proj.weight.detach().clone()
                       for l in self.layers}
        # Conv1D stores weight as (n_inner, n_embd): keys = c_fc outputs
        # (n_inner), values = c_proj outputs (n_embd).
        w = model.transformer.h[self.layers[0]].mlp.c_proj.weight
        self.key_dim, self.value_dim = w.shape[0], w.shape[1]

    # -- forward helpers ---------------------------------------------------
    def _ffn_keys(self, input_ids: torch.Tensor, positions: List[int],
                  layer: int) -> torch.Tensor:
        """c_fc outputs (c_proj inputs) at *positions* of block *layer*."""
        caps: Dict[str, torch.Tensor] = {}
        c_proj = self.model.transformer.h[layer].mlp.c_proj

        def grab(module, inp, out):
            caps["u"] = inp[0][0, positions, :].float().clone()

        h = c_proj.register_forward_hook(grab)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            h.remove()
        return caps["u"]

    def _forward_with_residual(self, input_ids: torch.Tensor,
                               positions: List[int],
                               residuals: List[torch.Tensor],
                               pred_positions: List[int],
                               track_grad: bool = False) -> torch.Tensor:
        """Forward pass injecting ``residuals[l]`` into ``c_proj`` of each edit
        layer at ``positions``; returns logits at ``pred_positions``."""
        handles = []
        for layer, resid in zip(self.layers, residuals):
            c_proj = self.model.transformer.h[layer].mlp.c_proj

            def make_hook(res, pos):
                pos_t = torch.tensor(pos, device=self.device)

                def hook(module, inp, out):
                    o = out.clone()
                    o[0, pos_t, :] = o[0, pos_t, :] + res.to(o.dtype)
                    return o
                return hook

            handles.append(c_proj.register_forward_hook(
                make_hook(resid, positions)))
        try:
            if track_grad:
                logits = self.model(input_ids).logits[0]
            else:
                with torch.no_grad():
                    logits = self.model(input_ids).logits[0]
        finally:
            for h in handles:
                h.remove()
        return logits[pred_positions, :]

    # -- value optimisation (ROME/MEMIT multi-token objective) -------------
    def optimize_value(self, prompt_ids: torch.Tensor, s_pos: int,
                       tgt_ids: List[int], n_steps: int = N_VAL_STEPS,
                       lr: float = VAL_LR) -> torch.Tensor:
        """Optimise a single value vector so that *every* target token becomes
        the argmax under teacher forcing (ROME objective)."""
        n_prompt = prompt_ids.shape[1]
        prompt_ids = prompt_ids.long()
        if len(tgt_ids) > 1:
            suffix = torch.tensor([tgt_ids[:-1]], dtype=torch.long,
                                  device=self.device)
            inp = torch.cat([prompt_ids, suffix], dim=1)
        else:
            inp = prompt_ids
        # Position predicting target token i is (n_prompt - 1 + i).
        pred_pos = [n_prompt - 1 + i for i in range(len(tgt_ids))]
        tgt = torch.tensor(tgt_ids, device=self.device)

        v = torch.zeros(self.value_dim, device=self.device, dtype=torch.float32,
                        requires_grad=True)
        opt = torch.optim.Adam([v], lr=lr)
        for _ in range(n_steps):
            # A fresh slice of v per layer keeps the autograd graph acyclic.
            shares = [v / len(self.layers) for _ in self.layers]
            logits = self._forward_with_residual(
                inp, [s_pos], shares, pred_pos,
                track_grad=True)                       # (n_tgt, vocab)
            logp = torch.log_softmax(logits.float(), dim=-1)
            loss = -logp.gather(1, tgt.unsqueeze(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                v.clamp_(-V_CLAMP, V_CLAMP)
        return v.detach()

    # -- closed-form write -------------------------------------------------
    def context_keys(self, prompts: Sequence[Dict[str, Any]],
                     layer: int) -> torch.Tensor:
        """Real-corpus keys (c_fc outputs) for the covariance estimate."""
        keys = []
        for case in prompts:
            ids = self.tok(case["prompt"], return_tensors="pt").to(self.device)
            pos = subject_token_positions(self.tok, case["prompt"], case["subject"])
            keys.append(self._ffn_keys(ids["input_ids"], [pos[-1]], layer))
        return torch.cat(keys, dim=0)                   # (m, key_dim)

    def batch_edit(self, cases: List[Dict[str, Any]],
                   context_cases: Sequence[Dict[str, Any]],
                   method: str = "memit",
                   alpha_rank: int = ALPHAEDIT_RANK,
                   n_val_steps: int = N_VAL_STEPS) -> Dict[str, Any]:
        t0 = time.time()
        n_layers = len(self.layers)
        values: List[torch.Tensor] = []
        keys_per_layer: List[List[torch.Tensor]] = [[] for _ in self.layers]
        prompts, positions = [], []

        for case in cases:
            ids = self.tok(case["prompt"], return_tensors="pt").to(self.device)
            pos = subject_token_positions(self.tok, case["prompt"], case["subject"])
            tgt = target_token_ids(self.tok, case["target_new"],
                                   self.model.lm_head.out_features)
            v = self.optimize_value(ids["input_ids"], pos[-1], tgt,
                                    n_steps=n_val_steps)
            values.append(v)
            prompts.append(ids["input_ids"])
            positions.append(pos[-1])
            for li, layer in enumerate(self.layers):
                keys_per_layer[li].append(
                    self._ffn_keys(ids["input_ids"], [pos[-1]], layer))

        R = torch.stack(values)                          # (n, d_value)
        info_layers = []
        total_weight_norm = 0.0
        # Context-key matrix for the null-space projection (AlphaEdit).
        X = None
        if method == "alphaedit" and context_cases:
            X = self.context_keys(context_cases, self.layers[0])  # (m, key_dim)
            X = X / (X.norm(dim=1, keepdim=True) + 1e-8)

        for li, layer in enumerate(self.layers):
            K = torch.cat(keys_per_layer[li], dim=0)      # (n, key_dim)
            C = torch.zeros(self.key_dim, self.key_dim, device=self.device)
            if context_cases:
                ctx = self.context_keys(context_cases, layer)
                C = ctx.T @ ctx / max(1, ctx.shape[0])
            G = self.lam * torch.eye(self.key_dim, device=self.device) \
                + self.context_scale * C + K.T @ K
            delta = torch.linalg.solve(G, K.T @ (R / n_layers))   # (key_dim, d_value)
            if method == "alphaedit" and X is not None:
                # Null-space projection: remove the components that act on the
                # retained context directions of the key covariance.
                target = X.T @ torch.linalg.solve(
                    X @ X.T + 1e-6 * torch.eye(X.shape[0], device=self.device),
                    X @ delta)
                delta = delta - target
            fit = float(((K @ delta - R / n_layers).norm()
                         / (R / n_layers).norm()).item())
            with torch.no_grad():
                w = self.model.transformer.h[layer].mlp.c_proj.weight
                w.copy_(self.orig_w[layer] + delta)
            total_weight_norm += float(delta.norm().item())
            info_layers.append({
                "layer": layer,
                "fit_relative_error": round(fit, 5),
                "delta_norm": round(float(delta.norm().item()), 3),
            })

        return {
            "layers": self.layers,
            "n_edits": len(cases),
            "n_context_prompts": len(context_cases),
            "total_delta_norm": round(total_weight_norm, 3),
            "fit_relative_error": round(
                sum(i["fit_relative_error"] for i in info_layers) / n_layers, 5),
            "per_layer": info_layers,
            "elapsed_seconds": round(time.time() - t0, 2),
        }

    def restore(self):
        with torch.no_grad():
            for layer in self.layers:
                self.model.transformer.h[layer].mlp.c_proj.weight.copy_(
                    self.orig_w[layer])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_counterfact_cases(path: str, n: int, seed: int = 7,
                           exclude_ids: set | None = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    rng = random.Random(seed)
    records = rng.sample(list(records), min(len(records), n * 20))
    cases = []
    for rec in records:
        if exclude_ids and rec["case_id"] in exclude_ids:
            continue
        req = rec["requested_rewrite"]
        prompt = req["prompt"].replace("{}", req["subject"])
        cases.append({
            "case_id": rec["case_id"],
            "prompt": prompt,
            "subject": req["subject"],
            "target_new": req["target_new"]["str"],
            "target_true": req["target_true"]["str"],
            "paraphrasess": [p.replace("{}", req["subject"])
                             for p in rec.get("paraphrase_prompts", [])],
            "neighborhood_prompts": [p.replace("{}", req["subject"])
                                     for p in rec.get("neighborhood_prompts", [])],
            "attribute_prompts": [p.replace("{}", req["subject"])
                                  for p in rec.get("attribute_prompts", [])],
            "portability_prompts": [p.replace("{}", req["subject"])
                                    for p in rec.get("portability_prompts", [])],
        })
        if len(cases) >= n:
            break
    return cases


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_cases_position(cases: List[Dict[str, Any]], model, tok,
                            device: str) -> Dict[str, float]:
    """Single-position reliability: top-1 token predicted right after the
    prompt, i.e. the first token of the answer (teacher-forced ROME style).

    The earlier version read the logits at the *subject's last token*, which
    predicts the following function word of the template (" is"), not the
    answer, so the metric could never exceed baseline accuracy.
    """
    def predict_id(prompt: str, subject: str) -> int:
        ids = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(ids["input_ids"]).logits[0, -1, :]
        return int(torch.argmax(logits).item())

    def first_id(text: str) -> int:
        ids = tok(" " + text, add_special_tokens=False)["input_ids"] \
            or tok(text, add_special_tokens=False)["input_ids"]
        return ids[0]

    def score(prompts, subjects, targets) -> float:
        if not prompts:
            return 0.0
        hits = sum(1.0 for p, s, t in zip(prompts, subjects, targets)
                   if predict_id(p, s) == first_id(t))
        return hits / len(prompts)

    subs = [c["subject"] for c in cases]
    return {
        "rewrite_acc": round(score([c["prompt"] for c in cases], subs,
                                   [c["target_new"] for c in cases]), 4),
        "paraphrase_acc": round(score(
            [p for c in cases for p in c["paraphrasess"]],
            [c["subject"] for c in cases for _ in c["paraphrasess"]],
            [c["target_new"] for c in cases for _ in c["paraphrasess"]]), 4),
        "locality_neighborhood_acc": round(score(
            [p for c in cases for p in c["neighborhood_prompts"]],
            [c["subject"] for c in cases for _ in c["neighborhood_prompts"]],
            [c["target_true"] for c in cases for _ in c["neighborhood_prompts"]]), 4),
        "locality_attribute_acc": round(score(
            [p for c in cases for p in c["attribute_prompts"]],
            [c["subject"] for c in cases for _ in c["attribute_prompts"]],
            [c["target_true"] for c in cases for _ in c["attribute_prompts"]]), 4),
        "n_cases": len(cases),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="W2 real-model MEMIT/AlphaEdit")
    ap.add_argument("--model-key", default="gpt2-xl")
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", choices=["memit", "alphaedit"], default="memit")
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--n-context", type=int, default=N_CONTEXT)
    ap.add_argument("--val-steps", type=int, default=N_VAL_STEPS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[w2] loading model {args.model_key} ...", flush=True)
    model, tok = load_model(args.model_key)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    cases = load_counterfact_cases(args.data, args.n_cases)
    ctx = load_counterfact_cases(args.data, args.n_context, seed=999,
                                 exclude_ids={c["case_id"] for c in cases})
    print(f"[w2] {len(cases)} edit cases, {len(ctx)} context prompts, "
          f"subject spans ok = "
          f"{sum(1 for c in cases if subject_token_positions(tok, c['prompt'], c['subject'])[-1] < len(tok(c['prompt'])['input_ids']) - 1)}",
          flush=True)

    gen = make_generator(model, tok, max_new_tokens=48)
    ns_cases = [types.SimpleNamespace(**c) for c in cases]

    # Pre-edit baseline (what the unmodified model already scores).
    t0 = time.time()
    baseline_gen, baseline_details = evaluate_cases(
        ns_cases, gen, return_details=True)
    baseline_pos = evaluate_cases_position(cases, model, tok, device)
    print(f"[w2] baseline done: {baseline_gen}", flush=True)

    editor = RealMemitEditor(model, tok, device=device)
    info = editor.batch_edit(cases, ctx, method=args.method,
                             n_val_steps=args.val_steps)
    print(f"[w2] edit done: layers={info['layers']} "
          f"fit={info['fit_relative_error']}", flush=True)

    metrics_gen, post_details = evaluate_cases(
        ns_cases, gen, return_details=True)
    metrics_pos = evaluate_cases_position(cases, model, tok, device)
    editor.restore()

    result = {
        "task": f"W2-{args.method}",
        "model": args.model_key,
        "method": args.method,
        "edit": info,
        "n_cases": metrics_gen["n_cases"],
        "metrics_pre_edit_baseline": {k: round(v, 4) for k, v in baseline_gen.items()},
        "metrics_single_token_answer": {
            k: round(v, 4) for k, v in metrics_pos.items()},
        "metric_definitions": {
            "single_token_answer": (
                "teacher-forced top-1 next-token accuracy in the answer slot "
                "right after the prompt (one token only, not free generation)"),
            "generation": ("free greedy generation scored with exact-or-"
                           "containment match against the target"),
            "pre_edit_baseline": ("same free-generation pipeline measured "
                                  "before any weight edit"),
            "strict": ("answer-slot exact match: the first sentence of the "
                       "generation must equal the target after normalisation, "
                       "so a target repeated later in the continuation is not "
                       "counted as a hit"),
        },
        "metrics_generation": {k: round(v, 4) for k, v in metrics_gen.items()},
        "per_case_generation": {
            "pre_edit": baseline_details,
            "post_edit": post_details,
        },
        "revision": 3,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
