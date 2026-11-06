"""Likelihood probe for the W3 ConflictGate REJECT branch.

Motivation
----------
The free-generation pre-edit probe is capped at ~45% recall on gpt2-xl for a
structural reason, not a tuning one: for most CounterFact subjects the model's
greedy *continuation* of a paraphrase prompt never states the attribute at all,
whatever the budget or template (probe_sweep.py: 4-9 hits / 20 across
budget 8..32 and across both paraphrase templates).

This script replaces "did the model say the target?" with "does the model
already assign probability to the target?" -- the same protocol CounterFact
uses for efficacy (target token probability under a paraphrase prompt).  It is
one forward pass, deterministic, and does not depend on decoding quality.

Two signals are measured for every case:

  1. absolute probe probability  P(requested_target | paraphrase_prompt),
     length-normalised log-probability, i.e. exp(mean log p) per token.
     This is available to the gate at decision time (it only needs the
     requested target_new and the paraphrase prompt) -- so it is *deployable*.

  2. relative margin  logp(requested) - logp(other), where "other" is the
     competing target of the same item (target_true for a normal edit,
     target_new for a canary item).  This needs both targets of an item, so
     it is reported as an upper bound / diagnostic only.

Labels: canary items (requested target = the TRUE fact, i.e. information the
model already has) should score high; normal CounterFact items (requested
target = a FALSE fact) should score low.  A gate that rejects when the
absolute signal exceeds a threshold therefore rejects redundant edits while
leaving genuine edits alone.

Usage:
    python probe_likelihood.py --data-path <counterfact.json> --n 100 \
        --out probe_likelihood_gpt2xl.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, List

import torch

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W3_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w3\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W3_DIR)

from model_utils import load_model  # noqa: E402
from real_w3_components import load_records  # noqa: E402


def norm_logprob(model, tok, prompt: str, target: str) -> float:
    """Length-normalised log-probability of `target` after `prompt`."""
    target = (target or "").strip()
    if not target:
        return -20.0
    p_ids = tok(prompt, return_tensors="pt")["input_ids"]
    full = tok(prompt + " " + target, return_tensors="pt")["input_ids"]
    n_prompt = p_ids.shape[1]
    if full.shape[1] <= n_prompt:
        return -20.0
    full = full.to(model.device)
    with torch.no_grad():
        logits = model(full).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    tgt = full[0, n_prompt:]
    lps = logprobs[0, n_prompt - 1:-1]
    token_lp = lps.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.mean().item())


def auc(pos: List[float], neg: List[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def recall_at_fpr(pos: List[float], neg: List[float], threshold: float) -> Dict[str, float]:
    rec = sum(1 for p in pos if p >= threshold) / max(1, len(pos))
    fpr = sum(1 for n in neg if n >= threshold) / max(1, len(neg))
    return {"recall": round(rec, 4), "fpr": round(fpr, 4),
            "threshold": round(threshold, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description="W3 likelihood probe")
    ap.add_argument("--model-key", default="gpt2-xl")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--canary-n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_records(args.data_path)
    items: List[Dict[str, Any]] = []
    for i, rec in enumerate(records[: args.n]):
        rw = rec.get("requested_rewrite", rec)
        subject = rw.get("subject", "")
        t_new = rw.get("target_new", {})
        t_true = rw.get("target_true", {})
        new = t_new.get("str", t_new) if isinstance(t_new, dict) else t_new
        true = t_true.get("str", t_true) if isinstance(t_true, dict) else t_true
        para = rec.get("paraphrase_prompts") or []
        if not (subject and new and true and para):
            continue
        items.append({"sample_id": i, "subject": subject,
                      "target_new": new, "target_true": true,
                      "probe_prompt": para[0].replace("{}", subject),
                      "probe_prompt_2": (para[1].replace("{}", subject)
                                         if len(para) > 1 else "")})
    print(f"[lik] {len(items)} items", flush=True)

    model, tok = load_model(args.model_key)

    rows: List[Dict[str, Any]] = []
    for j, it in enumerate(items):
        p = it["probe_prompt"]
        lp_new = norm_logprob(model, tok, p, it["target_new"])
        lp_true = norm_logprob(model, tok, p, it["target_true"])
        r = {"sample_id": it["sample_id"], "subject": it["subject"],
             "target_new": it["target_new"], "target_true": it["target_true"],
             "logp_target_new": round(lp_new, 4),
             "logp_target_true": round(lp_true, 4),
             "margin_new_vs_true": round(lp_new - lp_true, 4)}
        if it["probe_prompt_2"]:
            q = it["probe_prompt_2"]
            lp2_new = norm_logprob(model, tok, q, it["target_new"])
            lp2_true = norm_logprob(model, tok, q, it["target_true"])
            r["logp_target_new_p2"] = round(lp2_new, 4)
            r["logp_target_true_p2"] = round(lp2_true, 4)
            r["margin_new_vs_true_p2"] = round(lp2_new - lp2_true, 4)
        rows.append(r)
        if j % 10 == 0:
            print(f"[lik] {j}/{len(items)}", flush=True)

    # canary items are the first `canary_n` items (same slice the W3 run uses)
    canary_rows = rows[: args.canary_n]
    main_rows = rows[args.canary_n:]

    def collect(sel, key):
        return [r[key] for r in sel if key in r]

    # Signal 1 (deployable): absolute prob of the REQUESTED target.
    #   canary  -> probability of target_true
    #   main    -> probability of target_new
    pos_abs = collect(canary_rows, "logp_target_true")
    neg_abs = collect(main_rows, "logp_target_new")
    # Signal 2 (diagnostic): margin, requested minus competing target.
    pos_mrg = [-r["margin_new_vs_true"] for r in canary_rows]  # true - new
    neg_mrg = list(collect(main_rows, "margin_new_vs_true"))

    grid = [round(-12 + 0.25 * k, 3) for k in range(49)]
    pos_abs_sorted = sorted(pos_abs)
    # sweep for the threshold with zero false positives and max recall
    fpr0 = None
    for th in sorted(grid, reverse=True):
        rec = sum(1 for p in pos_abs if p >= th) / max(1, len(pos_abs))
        fp = sum(1 for n in neg_abs if n >= th) / max(1, len(neg_abs))
        if fp == 0.0:
            fpr0 = {"threshold": th, "recall": round(rec, 4), "fpr": 0.0}
            break
    # operating point: threshold placed between the two class medians
    med_pos = sorted(pos_abs)[len(pos_abs) // 2] if pos_abs else 0.0
    med_neg = sorted(neg_abs)[len(neg_abs) // 2] if neg_abs else 0.0
    mid = round((med_pos + med_neg) / 2, 3)

    result = {
        "model_key": args.model_key,
        "n_items": len(rows),
        "n_canary": len(canary_rows),
        "n_main": len(main_rows),
        "signal_definitions": {
            "absolute": ("length-normalised log p(requested_target | paraphrase_prompt); "
                         "canary requests target_true, main requests target_new"),
            "margin": "log p(requested) - log p(competing); needs both targets, diagnostic only",
        },
        "absolute": {
            "canary_mean": round(sum(pos_abs) / max(1, len(pos_abs)), 4),
            "main_mean": round(sum(neg_abs) / max(1, len(neg_abs)), 4),
            "canary_median": round(med_pos, 4),
            "main_median": round(med_neg, 4),
            "auc": round(auc(pos_abs, neg_abs), 4),
            "best_zero_fpr_point": fpr0,
            "midpoint_threshold": recall_at_fpr(pos_abs, neg_abs, mid),
        },
        "margin": {
            "canary_mean": round(sum(pos_mrg) / max(1, len(pos_mrg)), 4),
            "main_mean": round(sum(neg_mrg) / max(1, len(neg_mrg)), 4),
            "auc": round(auc(pos_mrg, neg_mrg), 4),
        },
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[lik] absolute  canary_mean=%.3f main_mean=%.3f AUC=%.3f"
          % (result["absolute"]["canary_mean"], result["absolute"]["main_mean"],
             result["absolute"]["auc"]), flush=True)
    print("[lik] zero-FPR point: %s" % (fpr0,), flush=True)
    print("[lik] midpoint point: %s" % (result["absolute"]["midpoint_threshold"],), flush=True)
    print("[lik] margin    canary_mean=%.3f main_mean=%.3f AUC=%.3f"
          % (result["margin"]["canary_mean"], result["margin"]["main_mean"],
             result["margin"]["auc"]), flush=True)
    print(f"[lik] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

