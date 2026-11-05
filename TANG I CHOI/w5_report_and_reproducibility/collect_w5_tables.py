"""W5 (Tang I Choi) - assemble the results tables used by the report.

Reads (read-only) both the shared team artefacts under ``output/`` and Tang's
own W3/W4 artefacts, and writes ``report/results_tables.md``:

  Table 1  stream / lifelong editing results (team, gpt2-small)
  Table 2  gate ablation (Tang, gpt2-xl, this repo state)
  Table 3  gate quality: offline features vs. the CALM stream path
  Table 4  statistical tests (McNemar / bootstrap) for the ablation

The script never writes outside ``TANG I CHOI/w5_report_coordination``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTPUT = os.path.join(REPO, "output")
W3_RES = os.path.join(REPO, "TANG I CHOI",
                      "w3_conflict_gate_training_and_classifier", "results")
W4_RES = os.path.join(REPO, "TANG I CHOI", "w4_gate_ablations", "results")
REPORT = os.path.join(HERE, "report")
ARMS = ["always_write", "always_route", "rule_gate", "learned_gate"]


def load(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100.0 * float(x):.2f}"


def table1() -> str:
    """Team stream results (produced by the W3/W4 stream owners)."""
    w3_1k = load(os.path.join(OUTPUT, "w3_calm_stream_1k.json"))
    seq = load(os.path.join(OUTPUT, "w4_sequential_5k_results.json"))
    fix = load(os.path.join(OUTPUT, "w4_calm_fix_5k_results.json"))
    rows = []
    if w3_1k:
        d = w3_1k.get("decision_counts", {})
        rows.append(("W3 stream, 1K (gate as shipped)", w3_1k.get("model", ""),
                     w3_1k.get("max_edits"), pct(w3_1k.get("immediate_rewrite_rate")),
                     "-", d.get("WRITE"), d.get("ROUTE"), d.get("REJECT")))
    if seq:
        for key, label in (("trained_5k", "W4 5K, trained hypernet"),
                           ("baseline_5k", "W4 5K, baseline (no meta-training)")):
            r = seq.get("runs", {}).get(key, {})
            if not r:
                continue
            rows.append((label, seq.get("environment", {}).get("model", ""),
                         r.get("edits"),
                         pct(r.get("immediate_rewrite_rate_end_to_end")),
                         pct(r.get("immediate_model_rewrite_rate")),
                         "-", "-", "-"))
    if fix:
        d = fix.get("decision_counts", {})
        rows.append(("W4 5K, guards on (first-write + budget + dual-write)",
                     fix.get("model", ""), fix.get("max_edits"),
                     pct(fix.get("immediate_rewrite_rate")),
                     pct(fix.get("immediate_model_rewrite_rate")),
                     d.get("WRITE"), d.get("ROUTE"), d.get("REJECT")))
    if not rows:
        return "_Table 1: shared stream artefacts not found under `output/`._"
    lines = ["### Table 1 - Sequential / lifelong editing (team run, gpt2-small)", "",
             "| setting | model | edits | E2E rewrite % | model-only rewrite % "
             "| WRITE | ROUTE | REJECT |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines += ["", "E2E rewrite counts a memory hit as a successful rewrite; "
              "model-only rewrite only looks at the weights."]
    return "\n".join(lines)


def table2(stats: Dict) -> str:
    rows = {r["arm"]: r for r in stats["summary"]}
    lines = ["### Table 2 - Conflict-gate ablation (Tang, W4)", "",
             f"Model `{stats['meta']['model']}`, {stats['meta']['n_edits']} edits over "
             f"{stats['meta']['n_subjects']} subjects, "
             f"{stats['meta']['n_heldout']} held-out locality probes, "
             f"hypernet scale = {stats['meta']['hypernet_scale']}.", "",
             "| arm | WRITE | ROUTE | REJECT | gate acc. vs gold "
             "| immediate rewrite % | model-only retention % | E2E retention % "
             "| locality unchanged % | memory entries |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        r = rows.get(arm)
        if not r:
            continue
        lines.append(
            f"| `{arm}` | {r['write']} | {r['route']} | {r['reject']} "
            f"| {r['gate_acc']:.3f} | {pct(r['immediate_rewrite'])} "
            f"| {pct(r['model_retention'])} | {pct(r['e2e_retention'])} "
            f"| {pct(r['locality'])} | {r['memory_entries']} |")
    lines += ["",
              "> **Metric caveat.** `immediate rewrite %` and `model-only "
              "retention %` are scored against the base model's answer, so they "
              "measure the untouched model, not the writer: the value is "
              "identical in all four arms (0.110 here) because the gate cannot "
              "influence it. Deduped rerun: 0.325 in all four arms (= 13/40 base "
              "hits); writer-attributable rewrite after correction = 0.000. See "
              "`w4_gate_ablations/docs/w4_immediate_rewrite_diagnosis.md` and "
              "`w4_gate_ablations/results/w4_ablation_table_deduped.md`."]
    return "\n".join(lines)


def table3(gap: Dict) -> str:
    lines = ["### Table 3 - Gate quality on the same 835-record test split", "",
             "Offline = recorded `prior_edits` + `retrieval_context`; online = "
             "`retrieval_context=None` and an empty history, i.e. the CALM stream "
             "path.", "",
             "| gate | mode | accuracy | macro-F1 | WRITE recall | ROUTE recall "
             "| REJECT recall |",
             "|---|---|---|---|---|---|---|"]
    for gate in ("rule_gate", "learned_gate"):
        for mode in ("offline", "online"):
            g = gap["gates"][gate][mode]
            pc = g["per_class"]
            lines.append(
                f"| `{gate}` | {mode} | {g['accuracy']:.3f} | {g['macro_f1']:.3f} "
                f"| {pc['WRITE']['recall']:.3f} | {pc['ROUTE']['recall']:.3f} "
                f"| {pc['REJECT']['recall']:.3f} |")
    m = gap.get("mcnemar_learned_vs_rule", {})
    if m:
        lines += ["", "Learned vs. rule gate (exact McNemar): "
                  f"offline p={m['offline']['p_value']:.4f} "
                  f"(learned-only {m['offline']['learned_only_correct']}, "
                  f"rule-only {m['offline']['rule_only_correct']}); "
                  f"online p={m['online']['p_value']:.4f} "
                  f"(learned-only {m['online']['learned_only_correct']}, "
                  f"rule-only {m['online']['rule_only_correct']})."]
    return "\n".join(lines)


def table4(stats: Dict) -> str:
    key = [p for p in stats["pairwise"]
           if p["metric_key"] == "final_e2e_hit"]
    lines = ["### Table 4 - Paired tests on end-to-end retention", "",
             "`b` = A retains and B does not, `c` = the opposite.", "",
             "| A | B | b | c | McNemar p | delta | delta 95% CI |",
             "|---|---|---|---|---|---|---|"]
    for p in key:
        ci = p["ci95"]
        lines.append(f"| `{p['a']}` | `{p['b']}` | {p['b_only']} | {p['c_only']} "
                     f"| {p['p_value']:.4f} | {p['delta']:+.3f} "
                     f"| [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    return "\n".join(lines)


def main() -> None:
    stats = load(os.path.join(W4_RES, "w4_statistics.json"))
    gap = load(os.path.join(W3_RES, "gate_online_gap.json"))
    if stats is None:
        raise SystemExit("[w5-tables] w4_statistics.json missing; run W4 first")
    if gap is None:
        raise SystemExit("[w5-tables] gate_online_gap.json missing; run W3 first")

    blocks = ["# Results tables (generated)", "",
              "Generated by `collect_w5_tables.py` - do not edit by hand; "
              "re-run `reproduce_all.sh` instead.", "",
              table1(), "", table2(stats), "", table3(gap), "", table4(stats), ""]
    md = "\n".join(blocks)
    os.makedirs(REPORT, exist_ok=True)
    path = os.path.join(REPORT, "results_tables.md")
    with open(path, "w") as f:
        f.write(md)
    print(md)
    print(f"[w5-tables] wrote {path}")


if __name__ == "__main__":
    main()
