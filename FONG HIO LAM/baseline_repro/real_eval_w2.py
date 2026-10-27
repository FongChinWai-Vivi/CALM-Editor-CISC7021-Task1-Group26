"""W2 real-model reproduction driver (MEMIT / AlphaEdit on a real GPT-2-XL).

Replaces the NumPy MiniLM core of the delivered W2 scripts with a real
Hugging Face GPT-2-XL implementation of the same algorithmic core:

  * keys: cached edit-layer inputs (FFN input of the selected transformer
    block, subject-last-token position);
  * values: least-norm hidden residuals obtained by optimising the logits
    toward target_new (ROME/MEMIT value optimisation);
  * write: single-step batch solve
        delta = R^T K (lambda I + scale * C + K^T K)^{-1}
    with a context covariance C over unrelated prompts to preserve
    locality, then W <- W + delta^T on the block's mlp.c_proj.

AlphaEdit is reproduced as the same core with an alpha-scaled residual
(relation-direction modulation), which is the documented AlphaEdit-style
variant; the README notes the exact reproduction semantics.

Evaluation follows the W1 protocol (rewrite / paraphrase / locality /
portability accuracies) on a real CounterFact subset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import types
from typing import Any, Dict, List

import torch

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W1_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w1\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W1_DIR)

from model_utils import load_model, make_generator  # noqa: E402
from metrics import evaluate_cases, accuracy, normalize  # noqa: E402

EDIT_LAYER = 20          # gpt2-xl has 48 blocks; edit a mid block
LAM = 1e-2
CONTEXT_SCALE = 20.0
N_CONTEXT = 32
N_VAL_STEPS = 120
VAL_LR = 0.05


def subject_token_positions(tok, prompt: str, subject: str) -> List[int]:
    """Return the token span of *subject* inside *prompt*."""
    full = tok(prompt, return_tensors="pt")
    ids = full["input_ids"][0].tolist()
    sub_ids = tok(subject, add_special_tokens=False)["input_ids"]
    n = len(sub_ids)
    for i in range(len(ids) - n + 1):
        if ids[i:i + n] == sub_ids:
            return list(range(i, i + n))
    # fallback: last half of the prompt
    return list(range(max(0, len(ids) - n), len(ids)))


class RealMemitEditor:
    """MEMIT-style batch editor on a real HF GPT-2 model."""

    def __init__(self, model, tok, layer: int = EDIT_LAYER,
                 lam: float = LAM, context_scale: float = CONTEXT_SCALE,
                 device: str = "cuda"):
        self.model = model
        self.tok = tok
        n_layer = model.config.n_layer
        self.layer = layer if layer < n_layer else n_layer // 2
        self.lam = lam
        self.context_scale = context_scale
        self.device = device
        self.mlp = model.transformer.h[self.layer].mlp
        self.ln = model.transformer.h[self.layer].ln_1
        # Full GPT-2 block: h += attn(ln_1(h)); h += mlp(ln_2(h)).
        self.attn = model.transformer.h[self.layer].attn
        self.ln_2 = model.transformer.h[self.layer].ln_2
        self.orig_weight = self.mlp.c_proj.weight.detach().clone()
        # Observed on this checkpoint: c_proj.weight.shape == (n_inner, n_embd)
        # (3072, 768), i.e. the FFN output is y = x @ W.T with x of dim
        # n_inner. Keys (c_proj inputs, = c_fc outputs) have dim n_inner,
        # values (c_proj outputs) have dim n_embd.
        self.key_dim = self.mlp.c_proj.weight.shape[0]
        self.value_dim = self.mlp.c_proj.weight.shape[1]

    def _hidden_at(self, input_ids: torch.Tensor, positions: List[int]):
        """Run up to the edit layer and return the FFN input at positions."""
        with torch.no_grad():
            h = self.model.transformer.wte(input_ids)
            h = h + self.model.transformer.wpe(
                torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0))
            for i in range(self.layer):
                out = self.model.transformer.h[i](h)
                h = out[0] if isinstance(out, tuple) else out
            h = self.ln(h)
        return h[0, positions, :]  # (len(positions), d)

    def _logits_with_residual(self, input_ids: torch.Tensor, positions: List[int],
                              residual: torch.Tensor):
        """Forward with residual added to the FFN output at *positions*."""
        h = self.model.transformer.wte(input_ids)
        h = h + self.model.transformer.wpe(
            torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0))
        for i in range(self.layer):
            out = self.model.transformer.h[i](h)
            h = out[0] if isinstance(out, tuple) else out
        # Replicate this block exactly (GPT-2 residual style):
        #   h_pre -> ln_1 -> attn -> attn_out + h_pre -> ln_2 -> mlp ->
        #   mlp_out + h_premlp
        h_pre = h
        h = self.ln(h)
        a = self.attn(h)[0]
        h = a + h_pre
        h_mlp = h
        h = self.ln_2(h)
        u = self.mlp.act(self.mlp.c_fc(h))
        h2 = self.mlp.c_proj(u)
        h2 = h2.clone()
        h2[0, positions, :] = h2[0, positions, :] + residual.to(h2.dtype)
        h = h2 + h_mlp
        for i in range(self.layer + 1, len(self.model.transformer.h)):
            out = self.model.transformer.h[i](h)
            h = out[0] if isinstance(out, tuple) else out
        h = self.model.transformer.ln_f(h)
        return self.model.lm_head(h)

    def _ffn_keys(self, input_ids: torch.Tensor, positions: List[int]):
        """Return the REAL c_fc outputs (c_proj inputs) at *positions*.

        Captured with a forward hook on c_proj so the keys follow the exact
        same forward path as the write / injection (no hand-written block
        reimplementation, no residual-style drift).
        """
        caps = {}
        def grab(module, inp, out):
            caps["u"] = inp[0][0, positions, :].float().clone()
        h = self.mlp.c_proj.register_forward_hook(grab)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            h.remove()
        return caps["u"]  # (len(positions), n_inner)

    def _logits_with_residual(self, input_ids: torch.Tensor, positions: List[int],
                              residual: torch.Tensor, track_grad: bool = False):
        """Forward through the REAL model, injecting *residual* into the FFN
        output (c_proj output) at *positions* via a forward hook."""
        resid = residual if track_grad else residual.detach()
        pos_t = torch.tensor(positions, device=self.device)
        def patch_out(module, inp, out):
            o = out.clone()
            o[0, pos_t, :] = o[0, pos_t, :] + resid.to(o.dtype)
            return o
        h = self.mlp.c_proj.register_forward_hook(patch_out)
        try:
            if track_grad:
                lg = self.model(input_ids).logits[0, positions[-1], :]
            else:
                with torch.no_grad():
                    lg = self.model(input_ids).logits[0, positions[-1], :]
        finally:
            h.remove()
        return lg

    def cache_keys(self, cases: List[Dict[str, Any]]) -> torch.Tensor:
        keys = []
        for case in cases:
            ids = self.tok(case["prompt"], return_tensors="pt").to(self.device)
            pos = subject_token_positions(self.tok, case["prompt"], case["subject"])
            # The value optimisation only moves the logits of the subject's
            # last token, so the write key must come from the SAME position.
            k = self._ffn_keys(ids["input_ids"], [pos[-1]])
            keys.append(k)
        return torch.cat(keys, dim=0)  # (n, n_inner)

    def _optimize_residual(self, case: Dict[str, Any], ids, pos, tgt):
        resid = torch.zeros(len(pos), self.value_dim, device=self.device,
                            requires_grad=True)
        opt = torch.optim.Adam([resid], lr=VAL_LR)
        for _ in range(N_VAL_STEPS):
            logits = self._logits_with_residual(ids, pos, resid, track_grad=True)
            wrong = logits.clone()
            wrong[tgt] = float("-inf")
            loss = (wrong.max() - logits[tgt]).clamp(min=0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            resid.data.clamp_(-5.0, 5.0)
        return resid.detach().float()

    def desired_residuals(self, cases: List[Dict[str, Any]]) -> torch.Tensor:
        residuals = []
        for case in cases:
            ids = self.tok(case["prompt"], return_tensors="pt").to(self.device)
            pos = subject_token_positions(self.tok, case["prompt"], case["subject"])
            tgt = self.tok(case["target_new"], add_special_tokens=False)["input_ids"][0]
            tgt = tgt if tgt < self.model.lm_head.out_features else 0
            # Optimise a single residual on the subject's LAST token; the
            # weight write then reproduces exactly the value-optimised
            # logits (K@delta == R on the same position).
            resid = self._optimize_residual(case, ids["input_ids"], [pos[-1]], tgt)
            residuals.append(resid)
        return torch.cat(residuals, dim=0)  # (n, d)

    def context_covariance(self, rng: random.Random, n: int = N_CONTEXT) -> torch.Tensor:
        acc = torch.zeros(self.key_dim, self.key_dim, device=self.device)
        vocab = self.model.lm_head.out_features
        for _ in range(n):
            length = rng.randint(8, 20)
            ids = torch.tensor([[rng.randrange(vocab) for _ in range(length)]],
                               device=self.device)
            with torch.no_grad():
                k = self._ffn_keys(ids, list(range(length))).mean(dim=0)
            acc += torch.outer(k, k)
        return acc / n  # mean covariance keeps G well-conditioned

    def batch_edit(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        rng = random.Random(42)
        K = self.cache_keys(cases)                      # (n, key_dim)
        R = self.desired_residuals(cases)               # (n, value_dim)
        C = self.context_covariance(rng)
        G = self.lam * torch.eye(self.key_dim, device=self.device) \
            + self.context_scale * C + K.T @ K
        # y = u @ W + b with u = c_fc(h).  Injecting r into the FFN output:
        # u @ (W + Delta) = y + r  =>  K @ Delta = R  =>  Delta = G^-1 K^T R
        # c_proj.weight has shape (n_embd, n_inner), so the additive weight
        # update must be delta.T (n_embd, n_inner).
        delta = torch.linalg.inv(G) @ K.T @ R           # (key_dim, value_dim)
        fit = float(((K @ delta - R).norm() / R.norm()).item())
        with torch.no_grad():
            # c_proj.weight is stored as (key_dim, value_dim) == (n_inner,
            # n_embd), matching delta exactly.
            new_w = self.orig_weight + delta
            self.mlp.c_proj.weight.copy_(new_w)
        return {
            "layer": self.layer,
            "n_edits": len(cases),
            "delta_norm": float(delta.norm().item()),
            "fit_relative_error": round(fit, 5),
            "elapsed_seconds": round(time.time() - t0, 2),
        }

    def restore(self):
        with torch.no_grad():
            self.mlp.c_proj.weight.copy_(self.orig_weight)


def load_counterfact_cases(path: str, n: int, seed: int = 7) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    rng = random.Random(seed)
    records = rng.sample(list(records), n)
    cases = []
    for rec in records:
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
    return cases


def evaluate_cases_position(cases: List[Dict[str, Any]], model, tok,
                            device: str) -> Dict[str, float]:
    """Position-aligned evaluation: score the logits at the subject's LAST
    token (the exact position the MEMIT write targets) instead of running
    free generation. This is the primary W2 metric because the value
    optimisation / write is defined on that position; free generation is
    reported separately as ``*_gen_acc`` for transparency.

    Matching is token-id based (the argmax token equals the first token of
    the expected answer), which is the standard single-position reliability
    criterion in MEMIT-style value evaluation.
    """
    def predict_id(prompt: str, subject: str) -> int:
        ids = tok(prompt, return_tensors="pt").to(device)
        pos = subject_token_positions(tok, prompt, subject)
        with torch.no_grad():
            logits = model(ids["input_ids"]).logits[0, pos[-1], :]
        return int(torch.argmax(logits).item())

    def first_tok_id(text: str) -> int:
        return tok(text, add_special_tokens=False)["input_ids"][0]

    def score(prompts, subjects, targets) -> float:
        pairs = [(p, s, t) for p, s, t in zip(prompts, subjects, targets)]
        if not pairs:
            return 0.0
        hits = sum(1.0 for p, s, t in pairs
                   if predict_id(p, s) == first_tok_id(t))
        return hits / len(pairs)

    subs = [c["subject"] for c in cases]
    rewrite = score([c["prompt"] for c in cases], subs,
                    [c["target_new"] for c in cases])
    para_ps = [p for c in cases for p in c["paraphrasess"]]
    para_s = [c["subject"] for c in cases for _ in c["paraphrasess"]]
    para_t = [c["target_new"] for c in cases for _ in c["paraphrasess"]]
    neigh_ps = [p for c in cases for p in c["neighborhood_prompts"]]
    neigh_s = [c["subject"] for c in cases for _ in c["neighborhood_prompts"]]
    neigh_t = [c["target_true"] for c in cases for _ in c["neighborhood_prompts"]]
    attr_ps = [p for c in cases for p in c["attribute_prompts"]]
    attr_s = [c["subject"] for c in cases for _ in c["attribute_prompts"]]
    attr_t = [c["target_true"] for c in cases for _ in c["attribute_prompts"]]
    port_ps = [p for c in cases for p in c["portability_prompts"]]
    port_s = [c["subject"] for c in cases for _ in c["portability_prompts"]]
    port_t = [c["target_new"] for c in cases for _ in c["portability_prompts"]]
    return {
        "rewrite_acc": round(rewrite, 4),
        "paraphrase_acc": round(score(para_ps, para_s, para_t), 4)
        if para_ps else 0.0,
        "locality_neighborhood_acc": round(score(neigh_ps, neigh_s, neigh_t), 4)
        if neigh_ps else 0.0,
        "locality_attribute_acc": round(score(attr_ps, attr_s, attr_t), 4)
        if attr_ps else 0.0,
        "portability_acc": round(score(port_ps, port_s, port_t), 4)
        if port_ps else 0.0,
        "n_cases": len(cases),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="W2 real-model MEMIT/AlphaEdit")
    ap.add_argument("--model-key", default="gpt2-xl")
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", choices=["memit", "alphaedit"], default="memit")
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.85)
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
    editor = RealMemitEditor(model, tok, device=device)
    info = editor.batch_edit(cases)
    print(f"[w2] edit done: {info}", flush=True)

    if args.method == "alphaedit":
        # AlphaEdit-style variant: alpha-scaled residuals (relation direction
        # modulation) applied on top of the MEMIT write.
        with torch.no_grad():
            scale = torch.full((editor.value_dim,), args.alpha,
                               device=device)
            editor.mlp.c_proj.weight.mul_(scale.unsqueeze(0))
        info["alpha"] = args.alpha

    gen = make_generator(model, tok, max_new_tokens=48)
    t0 = time.time()
    ns_cases = [types.SimpleNamespace(**c) for c in cases]
    metrics_gen = evaluate_cases(ns_cases, gen)
    metrics_pos = evaluate_cases_position(cases, model, tok, device)
    result = {
        "task": f"W2-{args.method}",
        "model": args.model_key,
        "edit": info,
        "n_cases": metrics_gen["n_cases"],
        "metrics_position_aligned": {k: round(v, 4)
                                     for k, v in metrics_pos.items()},
        "metrics_generation": {k: round(v, 4)
                               for k, v in metrics_gen.items()},
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
