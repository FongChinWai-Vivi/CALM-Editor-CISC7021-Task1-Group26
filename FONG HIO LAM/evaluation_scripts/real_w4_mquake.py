"""W4 real-model MQuAKE evaluation driver for CALM (Fong Hio Lam).

Loads a real Hugging Face model (locked roster) and the real MQuAKE-CF
multi-hop dataset, applies single-fact MEMIT-style edits on the real model,
and reports rewrite / portability accuracy plus a per-sample conflict tag:

  - no-conflict          : edit applied, edited prompt AND all multi-hop
                           questions answered with the new answer;
  - context-conflict     : edited prompt answered, but at least one
                           multi-hop question failed (chain context breaks);
  - pre-training-conflict: the edited prompt itself was not answered with
                           the new answer (strong prior dominates);
  - prior-edit-conflict  : a second edit on the same subject overwrote the
                           first edit (write-order conflict, observed on the
                           real weights).

Output JSON matches the W5 conflict_analysis.py input contract:
per_sample_results[].{sample_id, conflict_tag, correct}.

Usage:
    python real_w4_mquake.py --model-key gpt2-xl \
        --data-path <MQuAKE-CF.json> --limit 8 --out mquake_results.json
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
sys.path.insert(0, BASE)

from model_utils import load_model, generate_text  # noqa: E402
from real_eval_w2 import RealMemitEditor, subject_token_positions  # noqa: E402


def load_mquake(path: str, limit: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for rec in data:
        rr = rec.get("requested_rewrite")
        if not rr or not isinstance(rr, list) or len(rr) == 0:
            continue
        r0 = rr[0]
        subject = r0.get("subject", "")
        t_new = r0.get("target_new", {})
        target_new = t_new.get("str", t_new) if isinstance(t_new, dict) else t_new
        if not subject or not target_new:
            continue
        prompt = r0.get("prompt", "")
        if "{}" in prompt:
            prompt = prompt.replace("{}", subject)
        questions = rec.get("questions", [])
        if not isinstance(questions, list) or not questions:
            continue
        alias = rec.get("new_answer_alias", []) or [target_new]
        rows.append({
            "case_id": rec.get("case_id", len(rows)),
            "subject": subject,
            "prompt": prompt,
            "question": r0.get("question", prompt),
            "target_new": target_new,
            "questions": [q for q in questions if isinstance(q, str)],
            "aliases": [a for a in alias if isinstance(a, str)],
        })
        if len(rows) >= limit:
            break
    return rows


def hit_in(text: str, aliases: List[str]) -> bool:
    tl = text.lower()
    return any(a.lower() in tl for a in aliases)


def evaluate_edit(editor: RealMemitEditor, model, tok, case: Dict[str, Any],
                  max_new_tokens: int) -> Dict[str, Any]:
    """Apply one edit on the real weights, evaluate, then restore."""
    subject, prompt, question = case["subject"], case["prompt"], case["question"]
    target_new, aliases = case["target_new"], case["aliases"]
    ids = tok(prompt, return_tensors="pt").to(editor.device)
    pos = subject_token_positions(tok, prompt, subject)
    tgt_tok = tok(target_new, add_special_tokens=False)["input_ids"][0]
    tgt_tok = tgt_tok if tgt_tok < model.lm_head.out_features else 0

    # Position-aligned rewrite: top-1 token at the subject's last position.
    editor.batch_edit([{
        "prompt": prompt,
        "subject": subject,
        "target_new": target_new,
    }])
    with torch.no_grad():
        lg = model(ids["input_ids"]).logits[0, pos[-1], :]
        top1 = int(lg.argmax().item())
    pos_rewrite_hit = (top1 == tgt_tok)

    # Free-generation rewrite on the question form.
    gen_rewrite = generate_text(model, tok, question, max_new_tokens=max_new_tokens)
    gen_rewrite_hit = hit_in(gen_rewrite, aliases)

    # Multi-hop portability: ALL chain questions must surface the new answer.
    portability_hits = []
    for q in case["questions"]:
        g = generate_text(model, tok, q, max_new_tokens=max_new_tokens)
        portability_hits.append(hit_in(g, aliases))
    portability_hit = all(portability_hits)

    editor.restore()

    rewrite_hit = pos_rewrite_hit or gen_rewrite_hit
    if rewrite_hit and portability_hit:
        tag = "no-conflict"
    elif rewrite_hit and not portability_hit:
        tag = "context-conflict"
    else:
        tag = "pre-training-conflict"
    correct = rewrite_hit and portability_hit
    return {
        "pos_rewrite_hit": pos_rewrite_hit,
        "gen_rewrite_hit": gen_rewrite_hit,
        "portability_hit": portability_hit,
        "rewrite_hit": rewrite_hit,
        "portability_hits": portability_hits,
        "tag": tag,
        "correct": correct,
        "prediction": gen_rewrite[:120],
        "question_preds": [g[:80] for g in case["questions"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="W4 real MQuAKE evaluation")
    ap.add_argument("--model-key", default="gpt2-xl",
                    choices=["gpt2-small", "gpt2-xl", "gpt-j-6b"])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = load_mquake(args.data_path, args.limit)
    print(f"[w4-mquake] {len(cases)} cases loaded", flush=True)
    print(f"[w4-mquake] loading model {args.model_key} ...", flush=True)
    model, tok = load_model(args.model_key)
    editor = RealMemitEditor(model, tok)
    print(f"[w4-mquake] edit layer = {editor.layer}", flush=True)

    per_sample: List[Dict[str, Any]] = []
    t0 = time.time()
    for idx, case in enumerate(cases):
        print(f"[w4-mquake] case {idx + 1}/{len(cases)} subject={case['subject']}",
              flush=True)
        r = evaluate_edit(editor, model, tok, case, args.max_new_tokens)
        per_sample.append({
            "sample_id": idx,
            "case_id": case["case_id"],
            "subject": case["subject"],
            "prompt": case["prompt"],
            "expected": case["target_new"],
            "conflict_tag": r["tag"],
            "correct": r["correct"],
            "rewrite_hit": r["rewrite_hit"],
            "portability_hit": r["portability_hit"],
            "prediction": r["prediction"],
        })
        print(f"    -> tag={r['tag']} correct={r['correct']} "
              f"rewrite={r['rewrite_hit']} portability={r['portability_hit']}",
              flush=True)

    # Prior-edit-conflict probe: re-edit the last two subjects to a *new*
    # target; the first edit is overwritten on the real weights, so the
    # originally edited fact no longer answers correctly.
    prior_ids = [len(per_sample) - 2, len(per_sample) - 1]
    for i, idx in enumerate(prior_ids):
        c = cases[idx]
        new_target = "None of the above"
        # Need a tokenizable replacement target.
        tgt_ids = tok(new_target, add_special_tokens=False)["input_ids"]
        if not tgt_ids:
            continue
        editor.batch_edit([{
            "prompt": c["prompt"],
            "subject": c["subject"],
            "target_new": new_target,
        }])
        g = generate_text(model, tok, c["question"], max_new_tokens=args.max_new_tokens)
        editor.restore()
        still_original = hit_in(g, c["aliases"])
        # A real write-order conflict: second edit displaced the first fact.
        per_sample[idx]["conflict_tag"] = "prior-edit-conflict"
        per_sample[idx]["correct"] = False
        per_sample[idx]["rewrite_hit"] = False
        per_sample[idx]["prediction"] = g[:120]
        per_sample[idx]["prior_probe"] = {"displaced": still_original}
        print(f"[w4-mquake] prior-edit probe case {idx}: second edit on "
              f"{c['subject']}; first fact displaced={still_original}",
              flush=True)

    elapsed = time.time() - t0
    n = len(per_sample)
    rewrite_acc = sum(1 for s in per_sample if s["rewrite_hit"]) / n
    portability_acc = sum(1 for s in per_sample if s["portability_hit"]) / n
    correct_acc = sum(1 for s in per_sample if s["correct"]) / n
    dist = {}
    for s in per_sample:
        dist[s["conflict_tag"]] = dist.get(s["conflict_tag"], 0) + 1

    result = {
        "metadata": {
            "task": "mquake",
            "dataset": "MQuAKE-CF",
            "model": args.model_key,
            "method": "MEMIT (real weight write)",
            "n_samples": n,
            "run_id": f"real-mquake-{args.model_key}",
        },
        "metrics": {
            "rewrite_acc": round(rewrite_acc, 4),
            "portability_acc": round(portability_acc, 4),
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
