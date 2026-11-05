"""W4 (Tang I Choi) - statistical analysis of the conflict-gate ablations.

Reads the per-arm result files produced by ``run_w4_gate_ablation.py`` and
writes:

  * ``w4_statistics.json``       - every test result in machine-readable form
  * ``w4_ablation_table.md``     - the ablation table used by the report

Tests implemented (no network, no extra dependencies beyond scipy):

  1. Exact McNemar test on paired per-edit outcomes (end-to-end retention,
     model-only retention, immediate rewrite) for every pair of arms.
  2. Paired bootstrap 95% CI for the difference of two paired retention
     proportions (McNemar's test says *whether*, the CI says *how much*).
  3. Gate-vs-gold agreement: exact McNemar test of W3 learned gate against the
     rule gate and against the always-* baselines, plus Cohen's kappa.
  4. Per-conflict-type breakdown (situation) of end-to-end retention and of
     decision correctness, with Wilson 95% CI per cell.

Usage::

    python analyze_w4_statistics.py                    # uses ./results
    python analyze_w4_statistics.py --results-dir ...  # custom directory
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ARMS = ["always_write", "always_route", "rule_gate", "learned_gate"]
DECISIONS = ["WRITE", "ROUTE", "REJECT"]
# situation -> short name used in the report
SITUATIONS = [
    "no-conflict",
    "pre-training-conflict",
    "prior-edit-conflict",
    "context-conflict",
]
SITUATION_LABELS = {
    "no-conflict": "no-conflict",
    "pre-training-conflict": "pre-training-conflict",
    "prior-edit-conflict": "prior-edit-conflict",
    "context-conflict": "context-conflict",
}
GATE_TEST = os.path.join(HERE, "..", "w3_conflict_gate_training_and_classifier",
                         "gate_datasets", "gate_test.jsonl")
METRICS = [
    ("final_e2e_hit", "E2E retention"),
    ("final_model_hit", "model-only retention"),
    ("immediate_rewrite", "immediate rewrite"),
]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_arms(results_dir: str, stream: str = "canonical") -> Dict[str, dict]:
    """Load one arm per name.

    ``stream="canonical"`` skips the ``*_deduped.json`` reruns so that the report
    table keeps describing the 191-edit / four-conflict-type stream; pass
    ``stream="deduped"`` to analyse the deduped rerun instead. Both sets of files
    can coexist in ``results/``.
    """
    data = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "w4_ablation_*.json"))):
        if "_smoke" in path:
            continue
        is_deduped = "_deduped" in os.path.basename(path)
        if stream == "canonical" and is_deduped:
            continue
        if stream == "deduped" and not is_deduped:
            continue
        with open(path) as f:
            obj = json.load(f)
        data[obj["arm"]] = obj
    if not data:
        raise SystemExit(f"[w4-stats] no w4_ablation_*.json under {results_dir} "
                         f"for stream={stream}")
    return data


def paired_vectors(a: dict, b: dict, metric: str):
    """Align two arms on edit id and return (x, y) boolean vectors."""
    va = {d["id"]: bool(d.get(metric)) for d in a["per_edit"]}
    vb = {d["id"]: bool(d.get(metric)) for d in b["per_edit"]}
    ids = sorted(set(va) & set(vb), key=str)
    return [va[i] for i in ids], [vb[i] for i in ids]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def mcnemar(x: Sequence[bool], y: Sequence[bool]) -> dict:
    """Exact (binomial) McNemar test for paired binary outcomes."""
    b = sum(1 for i in range(len(x)) if x[i] and not y[i])
    c = sum(1 for i in range(len(x)) if y[i] and not x[i])
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0,
                "statistic": 0.0, "note": "no discordant pairs"}
    p = float(stats.binomtest(b, n, 0.5).pvalue)
    return {"b": b, "c": c, "n_discordant": n,
            "statistic": (abs(b - c) - 1) ** 2 / n,  # continuity-corrected chi2
            "p_value": round(p, 6)}


def paired_bootstrap(x: Sequence[bool], y: Sequence[bool], draws: int = 5000,
                     seed: int = 20261101) -> dict:
    """Paired bootstrap CI for P(x) - P(y)."""
    rng = np.random.default_rng(seed)
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(xa)
    if n == 0:
        return {"delta": 0.0, "ci95": [0.0, 0.0], "draws": draws}
    idx = rng.integers(0, n, size=(draws, n))
    diffs = xa[idx].mean(axis=1) - ya[idx].mean(axis=1)
    return {
        "delta": round(float(xa.mean() - ya.mean()), 4),
        "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                 round(float(np.percentile(diffs, 97.5)), 4)],
        "draws": draws,
    }


def cohen_kappa(pred: Sequence[str], gold: Sequence[str]) -> float:
    n = len(pred)
    if n == 0:
        return 0.0
    po = sum(1 for p, g in zip(pred, gold) if p == g) / n
    labels = sorted(set(pred) | set(gold))
    pe = 0.0
    for lab in labels:
        pe += (pred.count(lab) / n) * (gold.count(lab) / n)
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1.0 - pe), 4)


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def summary_table(arms: Dict[str, dict]) -> List[dict]:
    rows = []
    for arm in ARMS:
        if arm not in arms:
            continue
        a = arms[arm]
        rows.append({
            "arm": arm,
            "n_edits": a["n_edits"],
            "write": a["decisions"].get("WRITE", 0),
            "route": a["decisions"].get("ROUTE", 0),
            "reject": a["decisions"].get("REJECT", 0),
            "gate_acc": a["gate_accuracy_vs_gold"],
            "immediate_rewrite": a["immediate_rewrite_rate"],
            "model_retention": a["final_model_retention"],
            "e2e_retention": a["final_e2e_retention"],
            "locality": a["locality_unchanged"],
            "memory_entries": a["memory_entries"],
            "sec_per_edit": a["seconds_per_edit"],
        })
    return rows


def label4_map() -> Dict[str, str]:
    """id -> four-label annotation from the W3 test split."""
    mapping: Dict[str, str] = {}
    if not os.path.exists(GATE_TEST):
        return mapping
    with open(GATE_TEST) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            mapping[r.get("id")] = r.get("label4")
    return mapping


def situation_table(arm: dict, labels: Optional[Dict[str, str]] = None) -> List[dict]:
    per = arm["per_edit"]
    labels = labels or {}
    for d in per:
        if not d.get("label4"):
            d["label4"] = labels.get(d.get("id"))
    out = []
    for sit in SITUATIONS:
        cells = [d for d in per if d.get("label4") == sit]
        n = len(cells)
        e2e = sum(1 for d in cells if d.get("final_e2e_hit"))
        correct = sum(1 for d in cells if d.get("decision") == d.get("gold_action"))
        out.append({
            "situation": SITUATION_LABELS[sit],
            "n": n,
            "e2e_retention": round(e2e / n, 4) if n else None,
            "e2e_ci95": wilson(e2e, n),
            "decision_correct": round(correct / n, 4) if n else None,
            "decision_ci95": wilson(correct, n),
        })
    return out


def confusion(arm: dict) -> Dict[str, Dict[str, int]]:
    table = {g: {p: 0 for p in DECISIONS} for g in DECISIONS}
    for d in arm["per_edit"]:
        g, p = d.get("gold_action"), d.get("decision")
        if g in table and p in table[g]:
            table[g][p] += 1
    return table


def markdown(arms: Dict[str, dict], summary: List[dict],
             pairwise: List[dict], gates: List[dict],
             situations: Dict[str, List[dict]]) -> str:
    lines = ["# W4 gate ablation - results table", ""]
    first = summary[0] if summary else {}
    lines += [
        f"Model: `{arms[ARMS[0]]['model']}` | hparams: "
        f"`{arms[ARMS[0]]['hparams']}` | "
        f"edits: {first.get('n_edits')} over {arms[ARMS[0]]['n_subjects']} subjects | "
        f"held-out locality probes: {arms[ARMS[0]]['n_heldout']}",
        "",
        "## 1. Ablation table",
        "",
        "| arm | WRITE | ROUTE | REJECT | gate acc. | immediate rewrite "
        "| model-only retention | E2E retention | locality | memory entries | s/edit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| `{r['arm']}` | {r['write']} | {r['route']} | {r['reject']} "
            f"| {r['gate_acc']:.3f} | {r['immediate_rewrite']:.3f} "
            f"| {r['model_retention']:.3f} | {r['e2e_retention']:.3f} "
            f"| {r['locality']:.3f} | {r['memory_entries']} "
            f"| {r['sec_per_edit']} |")
    lines += [
        "",
        "> **Metric caveat.** `immediate rewrite` and `model-only retention` are "
        "scored against the *base model's* answer for the same prompt, so they "
        "fire only on edits whose subject the untouched model already answers. "
        "The identical value in all four arms is therefore expected: the gate "
        "cannot change it. On this 191-row stream (40 distinct prompts) it equals "
        "the base-model hit rate; on a deduped stream the same four arms report "
        "0.325 (= 13/40 base hits) and the writer-attributable rewrite after "
        "correction is 0.000. Diagnosis: `docs/w4_immediate_rewrite_diagnosis.md`; "
        "report section 3.4.",
        "",
        "## 2. Paired comparisons (exact McNemar + paired bootstrap)",
        "",
        "`b` = arm A hits / arm B misses, `c` = the opposite; "
              "delta = P(A) - P(B).", "",
              "| metric | A vs B | b | c | p | delta | delta 95% CI |",
              "|---|---|---|---|---|---|---|"]
    for row in pairwise:
        ci = row["ci95"]
        lines.append(
            f"| {row['metric']} | `{row['a']}` vs `{row['b']}` | {row['b_only']} "
            f"| {row['c_only']} | {row['p_value']:.4f} | {row['delta']:+.3f} "
            f"| [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines += ["", "## 3. Gate quality (learned gate versus rules and baselines)", "",
              "| comparison | b | c | p | kappa(A) | kappa(B) | agreement(A) |",
              "|---|---|---|---|---|---|---|"]
    for g in gates:
        lines.append(
            f"| {g['label']} | {g['b']} | {g['c']} | {g['p_value']:.4f} "
            f"| {g['kappa_a']:.3f} | {g['kappa_b']:.3f} | {g['agreement_a']:.3f} |")
    lines += ["", "## 4. Breakdown by conflict type", ""]
    for arm in ["learned_gate", "rule_gate", "always_write", "always_route"]:
        if arm not in situations:
            continue
        lines += [f"### `{arm}`", "",
                  "| situation | n | E2E retention (95% CI) | decision correct (95% CI) |",
                  "|---|---|---|---|"]
        for s in situations[arm]:
            if s["n"] == 0:
                lines.append(f"| {s['situation']} | 0 | - | - |")
                continue
            lines.append(
                f"| {s['situation']} | {s['n']} "
                f"| {s['e2e_retention']:.3f} [{s['e2e_ci95'][0]:.3f}, "
                f"{s['e2e_ci95'][1]:.3f}] "
                f"| {s['decision_correct']:.3f} "
                f"[{s['decision_ci95'][0]:.3f}, {s['decision_ci95'][1]:.3f}] |")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--stream", choices=["canonical", "deduped"], default="canonical",
                    help="canonical = the 191-edit four-type stream (default); "
                         "deduped = the 40-prompt rerun. The two write to different "
                         "file names so they never overwrite each other.")
    args = ap.parse_args()

    arms = load_arms(args.results_dir, stream=args.stream)
    suffix = "" if args.stream == "canonical" else "_deduped"
    summary = summary_table(arms)

    pairwise = []
    present = [r["arm"] for r in summary]
    for a, b in combinations(present, 2):
        for metric, label in METRICS:
            x, y = paired_vectors(arms[a], arms[b], metric)
            t = mcnemar(x, y)
            bs = paired_bootstrap(x, y, draws=args.draws)
            pairwise.append({
                "metric": label, "metric_key": metric, "a": a, "b": b,
                "n": len(x), "b_only": t["b"], "c_only": t["c"],
                "p_value": t["p_value"], "delta": bs["delta"],
                "ci95": bs["ci95"],
            })

    gates = []
    ref = "learned_gate"
    if ref in arms:
        gold = {d["id"]: d.get("gold_action") for d in arms[ref]["per_edit"]}
        pred_ref = {d["id"]: d.get("decision") for d in arms[ref]["per_edit"]}
        ids = sorted(gold, key=str)
        for other in present:
            if other == ref:
                continue
            pred_o = {d["id"]: d.get("decision") for d in arms[other]["per_edit"]}
            x = [pred_ref[i] == gold[i] for i in ids]
            y = [pred_o[i] == gold[i] for i in ids]
            t = mcnemar(x, y)
            gates.append({
                "label": f"learned vs {other}",
                "b": t["b"], "c": t["c"], "p_value": t["p_value"],
                "kappa_a": cohen_kappa([pred_ref[i] for i in ids],
                                       [gold[i] for i in ids]),
                "kappa_b": cohen_kappa([pred_o[i] for i in ids],
                                       [gold[i] for i in ids]),
                "agreement_a": round(sum(x) / len(x), 4),
            })

    labels = label4_map()
    situations = {arm: situation_table(arms[arm], labels) for arm in present}
    out = {
        "meta": {
            "model": arms[present[0]]["model"],
            "hparams": arms[present[0]]["hparams"],
            "n_edits": arms[present[0]]["n_edits"],
            "n_subjects": arms[present[0]]["n_subjects"],
            "n_heldout": arms[present[0]]["n_heldout"],
            "hypernet_scale": arms[present[0]].get("hypernet_scale"),
            "tests": ["exact McNemar (binomial, two-sided)",
                      f"paired bootstrap ({args.draws} draws, seed 20261101)",
                      "Cohen's kappa", "Wilson 95% CI"],
        },
        "summary": summary,
        "pairwise": pairwise,
        "gate_quality": gates,
        "confusion_learned_gate": confusion(arms[ref]) if ref in arms else {},
        "by_situation": situations,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stats_name = f"w4_statistics{suffix}.json"
    table_name = f"w4_ablation_table{suffix}.md"
    with open(os.path.join(args.results_dir, stats_name), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    md = markdown(arms, summary, pairwise, gates, situations)
    with open(os.path.join(args.results_dir, table_name), "w") as f:
        f.write(md)

    print(md)
    print(f"[w4-stats] wrote {args.results_dir}/{stats_name} and {table_name}")


if __name__ == "__main__":
    main()
