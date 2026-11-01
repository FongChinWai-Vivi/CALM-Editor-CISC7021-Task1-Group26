"""W3 (Tang I Choi) - how much of the learned gate survives the online path?

The W3 classifier is trained on static records that carry ``prior_edits`` and
``retrieval_context``.  Inside the CALM edit stream ``CALMEditor`` calls
``gate.decide(..., retrieval_context=None)`` and the prior-edit window is only
populated by edits that already happened in that same stream.  This script
quantifies the resulting gap between the two settings:

  * ``offline`` - the recorded features (what W3 trained and evaluated on);
  * ``online``  - context dropped and the prior-edit window empty, i.e. the
    first encounter of an edit in a CALM stream.

Both the learned gate and the W1 rule gate are scored on the same 835-record
test split, so the comparison is paired.

Output: ``results/gate_online_gap.json`` (+ a Markdown summary on stdout).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ARMS = os.path.join(REPO, "TANG I CHOI", "w4_gate_ablations", "gate_arms")
sys.path.insert(0, ARMS)

from _arm_common import EditRecord  # noqa: E402

TEST = os.path.join(HERE, "gate_datasets", "gate_test.jsonl")
OUT_DIR = os.path.join(HERE, "results")
LABELS = ["WRITE", "ROUTE", "REJECT"]
MODES = ["offline", "online"]


def load_records() -> List[dict]:
    with open(TEST) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_gate(gate, records: Sequence[dict], mode: str) -> List[str]:
    preds = []
    for r in records:
        if mode == "offline":
            # the recorded window is plain JSON; arms expect EditRecord objects
            prior = [EditRecord(d.get("subject") or "",
                                d.get("relation") or "fact",
                                d.get("target") or d.get("target_new") or "",
                                d.get("current_answer") or "",
                                float(d.get("current_confidence") or 0.0))
                     for d in (r.get("prior_edits") or [])]
            ctx = r.get("retrieval_context") or []
        else:
            prior, ctx = [], None
        gate.clear_history()
        d = gate.decide(
            subject=r.get("subject") or "",
            relation=r.get("relation") or "fact",
            target_new=r.get("target_new") or "",
            current_answer=r.get("current_answer") or "",
            current_confidence=float(r.get("current_confidence") or 0.0),
            recent_edits=prior,
            retrieval_context=ctx,
        )
        preds.append(d if isinstance(d, str) else str(d))
    return preds


def scores(gold: Sequence[str], pred: Sequence[str]) -> dict:
    n = len(gold)
    acc = sum(1 for g, p in zip(gold, pred) if g == p) / n
    per, f1s = {}, []
    for lab in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"support": sum(1 for g in gold if g == lab),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4)}
        f1s.append(f1)
    conf = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        if g in conf and p in conf[g]:
            conf[g][p] += 1
    return {"accuracy": round(acc, 4),
            "macro_f1": round(sum(f1s) / len(f1s), 4),
            "per_class": per, "confusion": conf}


def main() -> None:
    from scipy import stats

    records = load_records()
    gold = [r["action"] for r in records]
    print(f"[gate-gap] {len(records)} records, gold={ {l: gold.count(l) for l in LABELS} }")

    import rule_gate as rule_mod
    import learned_gate as learned_mod

    rule = rule_mod.ConflictGate()
    learned = learned_mod.ConflictGate()

    result: Dict[str, dict] = {"n_records": len(records),
                               "gold_distribution": {l: gold.count(l) for l in LABELS},
                               "gates": {}}
    for name, gate in (("rule_gate", rule), ("learned_gate", learned)):
        result["gates"][name] = {}
        for mode in MODES:
            pred = run_gate(gate, records, mode)
            result["gates"][name][mode] = scores(gold, pred)
            r = result["gates"][name][mode]
            print(f"[gate-gap] {name:12s} {mode:8s} acc={r['accuracy']:.3f} "
                  f"macroF1={r['macro_f1']:.3f}")

    # paired test: learned vs rule inside the online setting
    preds = {n: {m: run_gate(g, records, m) for m in MODES}
             for n, g in (("rule_gate", rule), ("learned_gate", learned))}
    for mode in MODES:
        a = [p == g for p, g in zip(preds["learned_gate"][mode], gold)]
        b = [p == g for p, g in zip(preds["rule_gate"][mode], gold)]
        bb = sum(1 for i in range(len(a)) if a[i] and not b[i])
        cc = sum(1 for i in range(len(a)) if b[i] and not a[i])
        p = float(stats.binomtest(bb, bb + cc, 0.5).pvalue) if bb + cc else 1.0
        result.setdefault("mcnemar_learned_vs_rule", {})[mode] = {
            "learned_only_correct": bb, "rule_only_correct": cc,
            "p_value": round(p, 6),
        }
        print(f"[gate-gap] McNemar {mode}: learned-only={bb} rule-only={cc} p={p:.4f}")

    # degradation inside one gate, offline -> online
    for name in ("rule_gate", "learned_gate"):
        off = result["gates"][name]["offline"]["accuracy"]
        on = result["gates"][name]["online"]["accuracy"]
        result["gates"][name]["accuracy_drop_offline_to_online"] = round(off - on, 4)
        print(f"[gate-gap] {name}: offline->online drop = {off - on:+.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "gate_online_gap.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"[gate-gap] wrote {path}")


if __name__ == "__main__":
    main()
