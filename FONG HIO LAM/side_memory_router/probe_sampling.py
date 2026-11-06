"""Is the 0.55 canary recall a probe ceiling or a model-knowledge ceiling?

The deployed gate probes with up to three prompts (original + two CounterFact
paraphrases) and a greedy decode of 16 tokens, and rejects when any answer
contains the requested target.  On the 20-item canary group that recovers 11/20
(report in probe_sweep.py, probe_likelihood.py).

Greedy decoding only ever samples the mode of the distribution, so a fact the
model holds weakly -- present under sampling but never the argmax string -- is
invisible to that probe.  This script separates the two failure modes:

  * greedy misses that *are* recoverable by sampling  -> probe ceiling
  * greedy misses that never surface under sampling    -> model-knowledge gap

It is the cheap, local version of the SliCK categorisation (Gekhman et al.,
"Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"): for every
probe prompt it takes one greedy decode plus `--n-samples` temperature samples,
scores each with the same containment rule the gate uses, and labels each canary
item HighlyKnown (greedy hit), WeaklyKnown (greedy miss, sample hit) or Unknown
(never hit).

Usage:
    python probe_sampling.py --data-path <counterfact.json> --n 20 \
        --out probe_sampling_gpt2xl.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List

import torch

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W3_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w3\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W3_DIR)

from model_utils import load_model  # noqa: E402
from conflict_gate import ConflictGate  # noqa: E402
from probe_sweep import build_canary, generate  # noqa: E402
from real_w3_components import load_records  # noqa: E402


def generate_sample(model, tok, prompt: str, budget: int,
                    temperature: float, top_p: float) -> str:
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs.to(model.device),
            max_new_tokens=budget,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="SliCK-style probe sampling study")
    ap.add_argument("--model-key", default="gpt2-xl")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-templates", type=int, default=3)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    started = time.time()
    records = load_records(args.data_path)
    cases = build_canary(records, args.n)
    print(f"[sampling] {len(cases)} canary cases", flush=True)

    model, tok = load_model(args.model_key)
    scorer = ConflictGate(seed=0)

    for j, case in enumerate(cases):
        prompts = [case["orig"]] + case["paras"][:args.max_templates - 1]
        target = case["target_true"]
        per_template: List[Dict[str, Any]] = []
        for t, prompt in enumerate(prompts):
            greedy = generate(model, tok, prompt, args.budget)
            greedy_hit = scorer._confidence_score(greedy, target) >= 1.0
            sampled = [generate_sample(model, tok, prompt, args.budget,
                                       args.temperature, args.top_p)
                       for _ in range(args.n_samples)]
            sample_hits = [scorer._confidence_score(s, target) >= 1.0
                           for s in sampled]
            per_template.append({
                "template": t,
                "probe_prompt": prompt,
                "greedy_answer": greedy[:160],
                "greedy_hit": greedy_hit,
                "n_samples": args.n_samples,
                "sample_hits": int(sum(sample_hits)),
                "sample_hit_rate": round(sum(sample_hits) / args.n_samples, 4),
                "sampled_answers": [s[:100] for s in sampled],
            })
        greedy_any = any(x["greedy_hit"] for x in per_template)
        sample_any = any(x["sample_hits"] > 0 for x in per_template)
        if greedy_any:
            category = "HighlyKnown"
        elif sample_any:
            category = "WeaklyKnown"
        else:
            category = "Unknown"
        case["templates"] = per_template
        case["greedy_any_hit"] = greedy_any
        case["sample_any_hit"] = sample_any
        case["category"] = category
        case["max_sample_hit_count"] = max(x["sample_hits"] for x in per_template)
        print(f"[sampling] {j}/{len(cases)} {case['subject'][:28]} "
              f"target={target[:26]!r} greedy={int(greedy_any)} "
              f"sample={int(sample_any)} -> {category}", flush=True)

    n = len(cases)
    counts = {c: sum(1 for x in cases if x["category"] == c)
              for c in ("HighlyKnown", "WeaklyKnown", "Unknown")}
    greedy_recall = sum(1 for x in cases if x["greedy_any_hit"]) / n
    sampled_recall = sum(1 for x in cases if x["sample_any_hit"]) / n
    recovered = [x["subject"] for x in cases
                 if not x["greedy_any_hit"] and x["sample_any_hit"]]
    unresolved = [x["subject"] for x in cases if x["category"] == "Unknown"]

    result = {
        "model_key": args.model_key,
        "data_path": args.data_path,
        "n_cases": n,
        "config": {
            "max_templates": args.max_templates,
            "budget": args.budget,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "greedy_recall": round(greedy_recall, 4),
        "sampled_recall": round(sampled_recall, 4),
        "category_counts": counts,
        "recovered_by_sampling": recovered,
        "still_unknown": unresolved,
        "elapsed_seconds": round(time.time() - started, 2),
        "cases": cases,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[sampling] greedy union recall = %.2f" % greedy_recall, flush=True)
    print("[sampling] sampled union recall = %.2f" % sampled_recall, flush=True)
    print("[sampling] categories = %s" % counts, flush=True)
    print("[sampling] recovered by sampling = %s" % recovered, flush=True)
    print("[sampling] still unknown = %s" % unresolved, flush=True)
    print("[sampling] elapsed = %.1fs" % (time.time() - started), flush=True)
    print(f"[sampling] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
