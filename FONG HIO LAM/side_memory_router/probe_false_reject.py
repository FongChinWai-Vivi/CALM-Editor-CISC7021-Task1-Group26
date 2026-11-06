"""What would it cost, in false rejects, to put sampling behind the REJECT branch?

The canary study (probe_sampling.py) shows temperature sampling recovers facts the
greedy probe misses: 3 of the 9 canary misses surface under sampling, taking the
canary ceiling from 0.55 to 0.70.  That result is deliberately *not* wired into the
gate, and the README says why -- the false-reject cost was never measured.  A sample
can contain the requested target for reasons that have nothing to do with the model
knowing the fact (tail drift landing on a plausible string), and every such hit would
turn a legitimate edit into a redundant one.

This script measures that cost on the split where it matters: the 100 main-split
edits, i.e. edits that *should* be applied.  For each request it runs the deployed
probe configuration (up to 3 prompts, 16 tokens, greedy) plus 8 temperature samples
per prompt, and scores containment of `target_new` exactly as the gate does.

The quantity of interest is `sample_only_hit` -- greedy missed but a sample hit.  Those
are precisely the requests where a sampling-augmented gate would newly say REJECT.
`greedy_hit` reproduces the deployed behaviour and must match the run report (2/100).

Usage:
    python probe_false_reject.py --data-path <counterfact.json> --n 100 \
        --out probe_false_reject_gpt2xl.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W3_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w3\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W3_DIR)

from model_utils import load_model  # noqa: E402
from conflict_gate import ConflictGate  # noqa: E402
from probe_sweep import generate, MAX_PARAS  # noqa: E402
from probe_sampling import generate_sample  # noqa: E402
from real_w3_components import load_records  # noqa: E402


def build_main_cases(records: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """The main split, probing for the *requested* new target."""
    cases = []
    for i, rec in enumerate(records[:n]):
        request = rec.get("request", rec)
        rw = request.get("requested_rewrite", request)
        subject = rw.get("subject", request.get("subject", ""))
        prompt_tpl = rw.get("prompt", request.get("prompt", ""))
        t_new = rw.get("target_new", {})
        target_new = t_new.get("str", t_new) if isinstance(t_new, dict) else t_new
        t_old = rw.get("target_true", {})
        target_true = t_old.get("str", t_old) if isinstance(t_old, dict) else t_old
        relation = rw.get("relation_id", "")
        para = rec.get("paraphrase_prompts") or rw.get("paraphrase_prompts") or []
        paras = [p.replace("{}", subject) for p in para][:MAX_PARAS]
        if not (subject and target_new and paras):
            continue
        cases.append({"sample_id": i, "subject": subject, "relation": relation,
                      "target_new": target_new, "target_true": target_true,
                      "paras": paras, "orig": prompt_tpl.replace("{}", subject)})
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description="False-reject cost of sampling probes")
    ap.add_argument("--model-key", default="gpt2-xl")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-templates", type=int, default=3)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    started = time.time()
    records = load_records(args.data_path)
    cases = build_main_cases(records, args.n)
    print(f"[false-reject] {len(cases)} main-split cases", flush=True)

    model, tok = load_model(args.model_key)
    scorer = ConflictGate(seed=0)

    for j, case in enumerate(cases):
        prompts = [case["orig"]] + case["paras"][:args.max_templates - 1]
        target = case["target_new"]
        per_template: List[Dict[str, Any]] = []
        for t, prompt in enumerate(prompts):
            greedy = generate(model, tok, prompt, args.budget)
            greedy_hit = scorer._confidence_score(greedy, target) >= 1.0
            sampled = [generate_sample(model, tok, prompt, args.budget,
                                       args.temperature, args.top_p)
                       for _ in range(args.n_samples)]
            hits = [scorer._confidence_score(s, target) >= 1.0 for s in sampled]
            per_template.append({
                "template": t,
                "probe_prompt": prompt,
                "greedy_answer": greedy[:160],
                "greedy_hit": greedy_hit,
                "sample_hits": int(sum(hits)),
                "sampled_answers": [s[:100] for s in sampled],
            })
        greedy_any = any(x["greedy_hit"] for x in per_template)
        sample_any = any(x["sample_hits"] > 0 for x in per_template)
        case["templates"] = per_template
        case["greedy_any_hit"] = greedy_any
        case["sample_any_hit"] = sample_any
        # greedy missed, a sample hit: this is a request the deployed gate WRITEs and
        # a sampling-augmented gate would REJECT
        case["sample_only_hit"] = sample_any and not greedy_any
        print(f"[false-reject] {j}/{len(cases)} {case['subject'][:26]} "
              f"greedy={int(greedy_any)} sample={int(sample_any)} "
              f"sample_only={int(case['sample_only_hit'])}", flush=True)

    n = len(cases)
    greedy_hits = [c for c in cases if c["greedy_any_hit"]]
    sample_only = [c for c in cases if c["sample_only_hit"]]
    both = [c for c in cases if c["greedy_any_hit"] and c["sample_any_hit"]]

    result = {
        "model_key": args.model_key,
        "data_path": args.data_path,
        "split": "main (edits that should be applied)",
        "n_cases": n,
        "config": {
            "max_templates": args.max_templates,
            "budget": args.budget,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "deployed_gate_rejects": len(greedy_hits),
        "deployed_gate_reject_rate": round(len(greedy_hits) / n, 4),
        "sample_only_hits": len(sample_only),
        "sample_only_rate": round(len(sample_only) / n, 4),
        "sample_only_rate_ci95_upper": round(min(1.0, (len(sample_only) + 3) / n), 4),
        "greedy_and_sample_hit": len(both),
        "augmented_gate_reject_rate": round(
            (len(greedy_hits) + len(sample_only)) / n, 4),
        "sample_only_cases": [
            {"sample_id": c["sample_id"], "subject": c["subject"],
             "relation": c["relation"], "target_new": c["target_new"],
             "greedy_answers": [t["greedy_answer"][:90] for t in c["templates"]],
             "hit_samples": [
                 {"template": t["template"],
                  "answers": [a[:90] for a in t["sampled_answers"]]}
                 for t in c["templates"] if t["sample_hits"] > 0]}
            for c in sample_only],
        "elapsed_seconds": round(time.time() - started, 2),
        "cases": cases,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[false-reject] deployed reject rate = {len(greedy_hits)}/{n}", flush=True)
    print(f"[false-reject] sample-only hits    = {len(sample_only)}/{n} "
          f"(new false rejects if sampling is wired in)", flush=True)
    print(f"[false-reject] augmented reject    = "
          f"{len(greedy_hits) + len(sample_only)}/{n}", flush=True)
    print("[false-reject] sample-only subjects = "
          + ", ".join(c["subject"] for c in sample_only), flush=True)
    print("[false-reject] elapsed = %.1fs" % (time.time() - started), flush=True)
    print(f"[false-reject] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
