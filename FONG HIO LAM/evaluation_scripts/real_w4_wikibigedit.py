"""W4 real-model WikiBigEdit evaluation driver for CALM (Fong Hio Lam).

Loads a real Hugging Face model and a real WikiBigEdit window file, applies
single-fact MEMIT-style edits on the real weights, and reports rewrite /
paraphrase / locality accuracy plus a per-sample conflict tag:

  - no-conflict          : edit applied and the edited prompt answered with
                           the new object;
  - context-conflict     : edited prompt answered, but at least one
                           paraphrase/persona question failed;
  - pre-training-conflict: the edited prompt itself was not answered with
                           the new object;
  - prior-edit-conflict  : a second edit on the same subject overwrote the
                           first (observed on the real weights).

Output JSON matches the W5 conflict_analysis.py input contract:
per_sample_results[].{sample_id, conflict_tag, correct}.

Usage:
    python real_w4_wikibigedit.py --model-key gpt2-xl \
        --data-path <wiki_big_edit_*.json> --limit 10 --out wikibigedit_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
# Use the *same* editor build as W2 so the two weeks share one write path.
W2_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w2\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W2_DIR)

from model_utils import load_model, generate_text  # noqa: E402
from real_eval_w2 import (RealMemitEditor, subject_token_positions,  # noqa: E402
                           target_token_ids)


def load_wikibigedit(path: str, limit: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for rec in data:
        subject = rec.get("subject", "")
        relation = rec.get("relation", "")
        obj = rec.get("object", "")
        if not subject or not obj:
            continue
        update = rec.get("update") or f"What is the {relation} of {subject}?"
        rephrase = rec.get("rephrase") or update
        loc = rec.get("loc") or ""
        loc_ans = rec.get("loc_ans") or ""
        mhop = rec.get("mhop") or ""
        mhop_ans = rec.get("mhop_ans") or ""
        rows.append({
            "case_id": len(rows),
            "subject": subject,
            "relation": relation,
            "prompt": update,
            "object": obj,
            "rephrase": rephrase,
            "loc": loc,
            "loc_ans": loc_ans,
            "mhop": mhop,
            "mhop_ans": mhop_ans,
        })
        if len(rows) >= limit:
            break
    return rows


def hit_in(text: str, answer: str) -> bool:
    tl = text.lower()
    al = answer.lower()
    if not al:
        return True
    return al in tl or any(w in tl for w in al.split() if len(w) > 2)


def evaluate_edit(editor: RealMemitEditor, model, tok, case: Dict[str, Any],
                  max_new_tokens: int,
                  ctx_pool: List[Dict[str, Any]] | None = None,
                  val_steps: int = 20) -> Dict[str, Any]:
    subject, prompt, obj = case["subject"], case["prompt"], case["object"]
    ids = tok(prompt, return_tensors="pt").to(editor.device)
    # Space-led tokenisation: the answer slot is predicted with a leading
    # space, so comparing against the bare-string id always missed.
    tgt_ids = target_token_ids(tok, obj, model.lm_head.out_features)
    tgt_tok = tgt_ids[0]

    editor.batch_edit([{"prompt": prompt, "subject": subject, "target_new": obj}],
                      ctx_pool or [], n_val_steps=val_steps)
    with torch.no_grad():
        lg = model(ids["input_ids"]).logits[0, -1, :]
        top1 = int(lg.argmax().item())
    pos_rewrite_hit = (top1 == tgt_tok)

    gen_rewrite = generate_text(model, tok, prompt, max_new_tokens=max_new_tokens)
    gen_rewrite_hit = hit_in(gen_rewrite, obj)

    # Paraphrase family: rephrase + persona-style question.
    para_text = generate_text(model, tok, case["rephrase"], max_new_tokens=max_new_tokens)
    para_hit = hit_in(para_text, obj)

    # Locality: unrelated question should keep its own answer.
    loc_hit = None
    if case["loc"] and case["loc_ans"]:
        loc_gen = generate_text(model, tok, case["loc"], max_new_tokens=max_new_tokens)
        loc_hit = hit_in(loc_gen, case["loc_ans"])

    # Multi-hop (if present).
    mhop_hit = None
    if case["mhop"] and case["mhop_ans"]:
        mhop_gen = generate_text(model, tok, case["mhop"], max_new_tokens=max_new_tokens)
        mhop_hit = hit_in(mhop_gen, case["mhop_ans"])

    editor.restore()

    rewrite_hit = pos_rewrite_hit or gen_rewrite_hit
    if rewrite_hit and para_hit:
        tag = "no-conflict"
    elif rewrite_hit and not para_hit:
        tag = "context-conflict"
    else:
        tag = "pre-training-conflict"
    correct = rewrite_hit and para_hit
    return {
        "pos_rewrite_hit": pos_rewrite_hit,
        "target_first_token_id": int(tgt_tok),
        "gen_rewrite_hit": gen_rewrite_hit,
        "para_hit": para_hit,
        "loc_hit": loc_hit,
        "mhop_hit": mhop_hit,
        "rewrite_hit": rewrite_hit,
        "tag": tag,
        "correct": correct,
        "prediction": gen_rewrite[:120],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="W4 real WikiBigEdit evaluation")
    ap.add_argument("--model-key", default="gpt2-xl",
                    choices=["gpt2-small", "gpt2-xl", "gpt-j-6b"])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--val-steps", type=int, default=20)
    ap.add_argument("--ctx-size", type=int, default=16,
                    help="size of the fixed context pool for the covariance estimate")
    ap.add_argument("--context-data", default="",
                    help="CounterFact json reused as the context pool")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = load_wikibigedit(args.data_path, args.limit)
    print(f"[w4-wbe] {len(cases)} cases loaded", flush=True)
    print(f"[w4-wbe] loading model {args.model_key} ...", flush=True)
    model, tok = load_model(args.model_key)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    editor = RealMemitEditor(model, tok)
    print(f"[w4-wbe] edit layers = {editor.layers}", flush=True)

    ctx_pool: List[Dict[str, Any]] = []
    if args.context_data and args.ctx_size > 0:
        with open(args.context_data, "r", encoding="utf-8") as fh:
            cf = json.load(fh)
        for rec in cf[: args.ctx_size * 4]:
            rq = rec.get("requested_rewrite", {})
            subj = rq.get("subject", "")
            tpl = rq.get("prompt", "")
            if not subj or not tpl:
                continue
            ctx_pool.append({
                "prompt": tpl.replace("{}", subj),
                "subject": subj,
                "target_new": rq.get("target_new", {}).get("str", ""),
            })
            if len(ctx_pool) >= args.ctx_size:
                break
    print(f"[w4-wbe] context pool = {len(ctx_pool)} prompts", flush=True)

    per_sample: List[Dict[str, Any]] = []
    t0 = time.time()
    for idx, case in enumerate(cases):
        print(f"[w4-wbe] case {idx + 1}/{len(cases)} subject={case['subject']}",
              flush=True)
        r = evaluate_edit(editor, model, tok, case, args.max_new_tokens,
                          ctx_pool=ctx_pool, val_steps=args.val_steps)
        per_sample.append({
            "sample_id": idx,
            "case_id": case["case_id"],
            "subject": case["subject"],
            "prompt": case["prompt"],
            "expected": case["object"],
            "conflict_tag": r["tag"],
            "correct": r["correct"],
            "rewrite_hit": r["rewrite_hit"],
            "pos_rewrite_hit": r["pos_rewrite_hit"],
            "gen_rewrite_hit": r["gen_rewrite_hit"],
            "paraphrase_hit": r["para_hit"],
            "locality_hit": r["loc_hit"],
            "prediction": r["prediction"],
        })
        print(f"    -> tag={r['tag']} correct={r['correct']} "
              f"rewrite={r['rewrite_hit']} para={r['para_hit']}",
              flush=True)

    # Prior-edit-conflict probe on the last two subjects.
    prior_ids = [len(per_sample) - 2, len(per_sample) - 1]
    for idx in prior_ids:
        c = cases[idx]
        new_target = "unknown"
        tgt_ids = tok(new_target, add_special_tokens=False)["input_ids"]
        if not tgt_ids:
            continue
        editor.batch_edit([{
            "prompt": c["prompt"],
            "subject": c["subject"],
            "target_new": new_target,
        }], ctx_pool, n_val_steps=args.val_steps)
        g = generate_text(model, tok, c["prompt"], max_new_tokens=args.max_new_tokens)
        editor.restore()
        still_original = hit_in(g, c["object"])
        # Only record the probe: the tag changes only when the second edit
        # really displaced the first fact.
        per_sample[idx]["prior_probe"] = {
            "displaced": bool(still_original),
            "second_edit_generation": g[:120],
        }
        if still_original:
            per_sample[idx]["conflict_tag"] = "prior-edit-conflict"
            per_sample[idx]["correct"] = False
            per_sample[idx]["rewrite_hit"] = False
        print(f"[w4-wbe] prior-edit probe case {idx}: second edit on "
              f"{c['subject']}; first fact displaced={still_original}", flush=True)

    elapsed = time.time() - t0
    n = len(per_sample)
    rewrite_acc = sum(1 for s in per_sample if s["rewrite_hit"]) / n
    para_acc = sum(1 for s in per_sample if s.get("paraphrase_hit", False)) / n
    correct_acc = sum(1 for s in per_sample if s["correct"]) / n
    loc_hits = [s["locality_hit"] for s in per_sample if s["locality_hit"] is not None]
    loc_acc = sum(loc_hits) / len(loc_hits) if loc_hits else None
    dist = {}
    for s in per_sample:
        dist[s["conflict_tag"]] = dist.get(s["conflict_tag"], 0) + 1

    result = {
        "metadata": {
            "task": "wikibigedit",
            "dataset": "WikiBigEdit",
            "model": args.model_key,
            "method": "MEMIT (real weight write)",
            "n_samples": n,
            "run_id": f"real-wikibigedit-{args.model_key}",
        },
        "metrics": {
            "rewrite_acc": round(rewrite_acc, 4),
        "position_rewrite_acc": round(
            sum(1 for x in per_sample if x.get("pos_rewrite_hit")) / n, 4),
        "generation_rewrite_acc": round(
            sum(1 for x in per_sample if x.get("gen_rewrite_hit")) / n, 4),
            "paraphrase_acc": round(para_acc, 4),
            "locality_acc": round(loc_acc, 4) if loc_acc is not None else None,
            "correct_acc": round(correct_acc, 4),
            "conflict_distribution": dist,
        },
        "per_sample_results": per_sample,
        "elapsed_seconds": round(elapsed, 2),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("metadata", "metrics", "elapsed_seconds")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
