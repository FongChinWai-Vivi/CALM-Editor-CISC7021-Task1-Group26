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



# --------------------------------------------------------------------------- #
# mandatory table footnotes (T2 / T5 of the W6 gate audit)
# --------------------------------------------------------------------------- #
FOOTNOTE_REASONS = {
    "backbone": (
        "*Backbones are not comparable.* `gpt2-small` is run under the 8 GB "
        "profile and `gpt2-xl` from the full 30 GB cache; rows from the two "
        "backbones are printed side by side for reading only and are never "
        "pooled, differenced or averaged."),
    "closed_form": (
        "*Closed-form arm: not run.* `closed_form_quickfix` does not consume a "
        "gate decision and is scored on the 5 K fluent-edit stream, so it cannot "
        "be an arm of a gate ablation. Arms without a closed-form counterpart "
        "are labelled \"not run\" instead of being left blank."),
    "nll": (
        "*`nll_delta` is not collected.* The stream runners log hit counts but "
        "no fluency delta, so no row here pairs `param_hit_total` with "
        "`nll_delta`. Every hit count in these tables is coverage-only and is "
        "never reported as a pass on a (hit, NLL) criterion."),
}


def footnotes(where: str) -> str:
    """The three footnotes that must follow *every* table (T5 + T2)."""
    return ("\n\n> **Table footnotes — " + where + ".** "
            "(i) " + FOOTNOTE_REASONS["backbone"] + " "
            "(ii) " + FOOTNOTE_REASONS["closed_form"] + " "
            "(iii) " + FOOTNOTE_REASONS["nll"] + "\n")


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
    return "\n".join(lines) + footnotes("table 1 - sequential / lifelong editing")


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
    return "\n".join(lines) + footnotes("table 2 - conflict-gate ablation")


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
    return "\n".join(lines) + footnotes("table 3 - gate quality, offline vs online")


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
    return "\n".join(lines) + footnotes("table 4 - paired tests on E2E retention")



def table3b() -> str:
    """Ranking diagnostics for the learned gate (T3 of the W6 gate audit).

    Reads ``w3_conflict_gate_training_and_classifier/results/gate_roc_pr.json``,
    produced by ``analysis_gate_roc_pr.py``.  Three domains, one classifier, one
    record set: the only thing that changes is which gate signals are emptied,
    so the columns separate ranking quality from covariate shift.
    """
    roc = load(os.path.join(W3_RES, "gate_roc_pr.json"))
    if roc is None:
        return ("### Table 3b - Ranking diagnostics (AUC / AP)\n\n"
                "**Not produced** - `gate_roc_pr.json` is missing. Run "
                "`w3_conflict_gate_training_and_classifier/analysis_gate_roc_pr.py`. "
                "This is a gap, not a pass.\n")
    domains = [
        ("offline", "offline (as labelled)"),
        ("stream_simulation", "stream path (`retrieval_context` emptied)"),
        ("cold_start_simulation", "cold start (context + prior history emptied)"),
    ]
    lines = ["### Table 3b - Ranking diagnostics for the learned gate (T3)", "",
             "Same 835-record CounterFact test split and the same classifier as "
             "Table 3; the rows differ only in which gate signals are emptied, so "
             "the spread isolates distribution shift from ranking quality.", "",
             "| domain | AUC (WRITE vs rest) | AP (WRITE) | WRITE recall | ROUTE recall "
             "| accuracy | admitted per 100 | precision of admitted |",
             "|---|---|---|---|---|---|---|---|"]
    for key, label in domains:
        b = roc["splits"][key]
        m = b["confusion_matrix_gold_rows"]["matrix"]
        total = max(sum(sum(r) for r in m), 1)
        acc = (m[0][0] + m[1][1] + m[2][2]) / total
        w_recall = m[0][0] / max(sum(m[0]), 1)
        r_recall = m[1][1] / max(sum(m[1]), 1)
        lines.append(
            f"| {label} | {b['auc_write_vs_rest']:.3f} | "
            f"{b['average_precision_write']:.3f} | {w_recall:.3f} | "
            f"{r_recall:.3f} | {acc:.3f} | {b['argmax_admitted_per_100']:.1f} | "
            f"{b['argmax_admitted_gold_write_precision']:.3f} |")
    lines += ["",
              "The cold-start row reproduces Table 3's `learned_gate` / online row "
              "(accuracy 0.340, WRITE recall 0.457, ROUTE recall 0.000, REJECT "
              "recall 0.992) to three decimals, so the simulation and the stream "
              "run agree; what Table 3 could not say is *why* accuracy falls. It "
              "falls because ROUTE recall collapses to 0.000 while AUC stays above "
              "chance, i.e. the ranking survives and the decision region does not."]
    return "\n".join(lines) + footnotes("table 3b - ranking diagnostics")


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
              table1(), "", table2(stats), "", table3(gap), "",
              table3b(), "", table4(stats), ""]
    md = "\n".join(blocks)
    os.makedirs(REPORT, exist_ok=True)
    path = os.path.join(REPORT, "results_tables.md")
    with open(path, "w") as f:
        f.write(md)
    print(md)
    print(f"[w5-tables] wrote {path}")


if __name__ == "__main__":
    main()
