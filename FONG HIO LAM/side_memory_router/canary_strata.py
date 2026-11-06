"""Re-base the canary recall on the subset of cases the model actually knows.

The canary group is built from CounterFact by asking the gate to write a fact whose
true answer is known to the world.  That label conflates two very different things:
the model holding the answer, and the world holding it.  probe_sampling.py shows the
split is 11 HighlyKnown / 3 WeaklyKnown / 6 Unknown, so 6 of the 20 "should be
rejected" items are facts gpt2-xl demonstrably does not hold under any probe
(30 greedy answers, 480 sampled answers, zero containment).

A gate cannot reject on evidence the model never produces, so those 6 belong out of
the recall denominator rather than in the numerator of a failure.  This script joins

  * the probe strata from probe_sampling_gpt2xl.json  (HighlyKnown / WeaklyKnown /
    Unknown), and
  * the gate's own canary decisions from w3_components_gpt2xl.json
    (reject_canary.cases),

and reports recall under three denominators: all 20 items (the old number, kept for
continuity with earlier sections), the 14 model-known items, and the 6 model-unknown
items, which are listed explicitly so a reviewer can check them by hand.

Usage:
    python canary_strata.py --run w3_components_gpt2xl.json \
        --sampling probe_sampling_gpt2xl.json --out canary_strata_gpt2xl.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Canary recall by model-knownness stratum")
    ap.add_argument("--run", required=True, help="w3_components_gpt2xl.json")
    ap.add_argument("--sampling", required=True, help="probe_sampling_gpt2xl.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = load(args.run)
    sampling = load(args.sampling)

    decisions = {c["subject"]: c["decision"]
                 for c in run["reject_canary"]["cases"]}
    strata = {c["subject"]: c["category"] for c in sampling["cases"]}
    max_hits = {c["subject"]: c["max_sample_hit_count"] for c in sampling["cases"]}
    greedy = {c["subject"]: c["greedy_any_hit"] for c in sampling["cases"]}
    sampled = {c["subject"]: c["sample_any_hit"] for c in sampling["cases"]}

    missing = [s for s in strata if s not in decisions]
    if missing:
        raise SystemExit(f"subjects absent from the run file: {missing}")

    rows: List[Dict[str, Any]] = []
    for subject, category in strata.items():
        decision = decisions[subject]
        rows.append({
            "subject": subject,
            "stratum": category,
            "gate_decision": decision,
            "rejected": decision == "REJECT",
            "greedy_hit": greedy[subject],
            "sample_hit": sampled[subject],
            # the deployed gate probes greedily first, so the achievable bound is
            # greedy OR sampled -- sampling alone can also drift away from an answer
            # greedy found (Edwin of Northumbria: greedy hit, 0/24 samples hit).
            "recoverable": greedy[subject] or sampled[subject],
            "max_sample_hit_count_per_template": max_hits[subject],
        })

    def block(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(subset)
        if not n:
            return {"n": 0}
        rejected = sum(1 for r in subset if r["rejected"])
        recoverable = sum(1 for r in subset if r["recoverable"])
        return {
            "n": n,
            "rejected_by_deployed_gate": rejected,
            "greedy_recall": round(rejected / n, 4),
            "sampled_upper_bound": recoverable,
            "sampled_recall": round(recoverable / n, 4),
            "subjects": [r["subject"] for r in subset],
        }

    known = [r for r in rows if r["stratum"] in ("HighlyKnown", "WeaklyKnown")]
    unknown = [r for r in rows if r["stratum"] == "Unknown"]

    result = {
        "model_key": sampling.get("model_key"),
        "n_canary": len(rows),
        "label_rule": (
            "A canary item counts as a gate miss only if the model can be shown to "
            "hold the target answer; model-unknown items are excluded from the recall "
            "denominator instead of being scored as failures."),
        "all_items": block(rows),
        "model_known": block(known),
        "model_unknown": block(unknown),
        "weakly_known_items": [
            {k: r[k] for k in ("subject", "gate_decision",
                               "max_sample_hit_count_per_template")}
            for r in rows if r["stratum"] == "WeaklyKnown"],
        "unknown_items": [
            {"subject": r["subject"], "gate_decision": r["gate_decision"]}
            for r in unknown],
        "cases": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    for key in ("all_items", "model_known", "model_unknown"):
        b = result[key]
        print(f"[strata] {key:<13} n={b['n']:>2} rejected={b['rejected_by_deployed_gate']:>2} "
              f"greedy={b['greedy_recall']:.4f} sampled={b['sampled_recall']:.4f}",
              flush=True)
    print("[strata] model-unknown items: "
          + ", ".join(f"{r['subject']} [{r['gate_decision']}]" for r in unknown),
          flush=True)
    print(f"[strata] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
