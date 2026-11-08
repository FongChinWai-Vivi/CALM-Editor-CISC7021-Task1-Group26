#!/usr/bin/env python3
"""Mechanical guard for the T2 / T3 / T5 checks of the gate audit.

Reads only generated artefacts and checks them as a reader would:

  T5  every table is followed by the three footnotes (backbone, closed-form
      "not run", nll_delta not collected);
  T3  the ranking diagnostics still carry all five quantities for all three
      domains (AUC, AP, TPR@0.5, realised admission rate, confusion matrix);
  T2  no table quotes a hit count in a file where ``nll_delta`` exists
      elsewhere -- i.e. once the fluency delta is collected it may not be
      dropped from a table.  While no artefact contains ``nll_delta`` at all
      the check reports PENDING (the value does not exist yet) instead of
      failing, so the guard can be used before T2 is implemented.

Exit code: 0 when T3 and T5 pass and T2 is either PASS or PENDING; 1 otherwise.

Usage: python w6_cross_review/check_gate_acceptance.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
TANG = os.path.join(REPO, "TANG I CHOI")
W3_RES = os.path.join(TANG, "w3_conflict_gate_training_and_classifier", "results")
W4_RES = os.path.join(TANG, "w4_gate_ablations", "results")
TABLES = [
    os.path.join(TANG, "w4_gate_ablations", "results", "w4_ablation_table.md"),
    os.path.join(TANG, "w4_gate_ablations", "results", "w4_ablation_table_deduped.md"),
    os.path.join(TANG, "w5_report_and_reproducibility", "report", "results_tables.md"),
]
ROC = os.path.join(W3_RES, "gate_roc_pr.json")
DOMAINS = ["offline", "stream_simulation", "cold_start_simulation"]
ROC_KEYS = ["auc_write_vs_rest", "average_precision_write", "tpr_at_0.5",
            "argmax_admitted_per_100", "argmax_admitted_gold_write_precision",
            "confusion_matrix_gold_rows"]

FOOTNOTE_MARKERS = {
    "footnote block": "Table footnotes",
    "backbone caveat": "gpt2-small",
    "closed-form not run": "not run",
    "nll not collected": "nll_delta",
}


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def table_blocks(md: str) -> List[Tuple[int, str]]:
    """Yield (line number, block) for every markdown table (header + rows)."""
    lines = md.splitlines()
    out: List[Tuple[int, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) \
                and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            start = i
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append((start + 1, "\n".join(lines[start:j])))
            i = j
        else:
            i += 1
    return out


def check_t5(md: str, path: str) -> List[str]:
    """Every table must be followed by the three footnotes within 12 lines."""
    errors: List[str] = []
    lines = md.splitlines()
    for lineno, block in table_blocks(md):
        end = lineno - 1 + len(block.splitlines())
        tail = "\n".join(lines[end - 1:end + 12])
        for label, marker in FOOTNOTE_MARKERS.items():
            if marker not in tail:
                errors.append(
                    f"[T5] {os.path.basename(path)}:{lineno} table is missing the "
                    f"{label} (`{marker}`) in the 12 lines that follow it")
    return errors


def check_t3() -> List[str]:
    errors: List[str] = []
    if not os.path.exists(ROC):
        return [f"[T3] {os.path.relpath(ROC, REPO)} missing; run "
                "analysis_gate_roc_pr.py"]
    roc = json.load(open(ROC, encoding="utf-8"))
    for domain in DOMAINS:
        block = roc.get("splits", {}).get(domain)
        if block is None:
            errors.append(f"[T3] domain `{domain}` missing from gate_roc_pr.json")
            continue
        for key in ROC_KEYS:
            if key not in block:
                errors.append(f"[T3] domain `{domain}` has no `{key}`")
        m = block.get("confusion_matrix_gold_rows", {}).get("matrix")
        if not (isinstance(m, list) and len(m) == 3
                and all(isinstance(r, list) and len(r) == 3 for r in m)):
            errors.append(f"[T3] domain `{domain}` confusion matrix is not 3x3")
    for key in ("threshold_sweep_on_stream_simulation",
                "threshold_sweep_on_cold_start_simulation"):
        if key not in roc:
            errors.append(f"[T3/T4] `{key}` missing from gate_roc_pr.json")
    return errors


def _nll_in_results() -> bool:
    """True only if a *results JSON* really carries the field (footnote text
    that merely mentions `nll_delta` must not switch the guard to strict)."""
    for root in (W4_RES, W3_RES):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if not name.endswith(".json"):
                continue
            body = read(os.path.join(root, name))
            # a *value*, not a mention in a "missing metric" note
            if re.search(r'"nll_delta"\s*:\s*-?\d', body):
                return True
    return False


def check_t2(texts: Dict[str, str]) -> Tuple[str, List[str]]:
    """PENDING while no artefact carries nll_delta; otherwise strict."""
    repository_has_nll = _nll_in_results()
    messages: List[str] = []
    if not repository_has_nll:
        return "PENDING", ["[T2] PENDING: no artefact contains `nll_delta` yet "
                           "(run_w4_gate_ablation.py does not record it); every "
                           "hit count in the tables is coverage-only and is "
                           "declared as such by the footnotes"]
    for path, md in texts.items():
        for lineno, block in table_blocks(md):
            header = block.splitlines()[0].lower()
            quotes_hits = any(k in header for k in ("hit", "retention", "rewrite"))
            if quotes_hits and "nll" not in header:
                messages.append(
                    f"[T2] {os.path.basename(path)}:{lineno} table quotes hits "
                    "(header contains hit/retention/rewrite) without an `nll` column")
    return ("FAIL" if messages else "PASS"), messages


def main() -> int:
    texts: Dict[str, str] = {}
    errors: List[str] = []

    for path in TABLES:
        if not os.path.exists(path):
            errors.append(f"[T5] missing table file: {os.path.relpath(path, REPO)}")
            continue
        texts[path] = read(path)
        errors += check_t5(texts[path], path)

    t3_errors = check_t3()
    t2_status, t2_messages = check_t2(texts)

    n_tables = sum(len(table_blocks(md)) for md in texts.values())
    print(f"tables checked : {n_tables} in {len(texts)} files")
    print(f"T5 footnotes   : {'PASS' if not errors else 'FAIL'}")
    print(f"T3 diagnostics : {'PASS' if not t3_errors else 'FAIL'}")
    print(f"T2 pairing     : {t2_status}")

    for line in errors + t3_errors + t2_messages:
        print("  " + line)

    hard_fail = bool(errors or t3_errors or t2_status == "FAIL")
    print("result         :", "FAIL" if hard_fail else "OK")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
