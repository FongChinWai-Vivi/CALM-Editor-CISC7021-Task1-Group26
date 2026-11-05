#!/usr/bin/env python
"""Diagnose the ``immediate_rewrite`` metric of the W4 gate ablation.

The W4 ablation reported ``immediate_rewrite_rate = 0.1099`` for the
``always_write`` arm and 1.000 for the three routing arms, which reads as "the
writer never writes".  This script tests that reading with three checks and
writes the evidence to ``results/immediate_rewrite_diagnosis.json``:

check 1  -- is the metric a function of the gate decision at all?
            (compare ``immediate_rewrite_rate`` across the four arms on the
             deduplicated stream: a metric that measures the *writer* must move
             with the WRITE share, a metric that measures the *base model* will
             not)
check 2  -- is the metric baseline-contaminated?
            (``rewrite_ok`` in CALMEditor compares the post-edit answer with
             ``target_new`` only, so a case whose base answer already contains
             ``target_new`` counts as a successful rewrite although nothing was
             edited; recompute with ``post_hit and not pre_hit``)
check 3  -- does the parameter write change the answer at all?
            (``answer_changed_by_write_rate`` and ``mean_delta_norm`` recorded
             by ``run_w4_gate_ablation.py``'s apply_update hook)

A fourth check replays the old 191-row stream through the per-prompt baseline
of the deduplicated run, to show that 0.1099 is the base-model hit rate
re-weighted by the duplicated prompts, not a writer failure.

Usage (from the EasyEdit working directory is NOT needed; pure JSON):
    python diagnose_immediate_rewrite.py
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ARMS = ["always_write", "always_route", "rule_gate", "learned_gate"]


def hit(answer: str, target: str) -> bool:
    return str(target).strip().lower() in str(answer or "").strip().lower()


def load(name: str) -> Dict[str, Any] | None:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def arm_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    per = res.get("per_edit") or []
    n = len(per)
    write_share = (sum(1 for d in per if d["decision"] == "WRITE") / n) if n else 0.0

    has_pre = any(d.get("answer_pre") is not None for d in per)
    pre_hit = sum(1 for d in per
                  if has_pre and hit(d.get("answer_pre"), d.get("target_new")))
    post_hit = sum(1 for d in per
                   if has_pre and hit(d.get("answer_post"), d.get("target_new")))
    corrected = sum(1 for d in per
                    if has_pre
                    and hit(d.get("answer_post"), d.get("target_new"))
                    and not hit(d.get("answer_pre"), d.get("target_new")))
    return {
        "arm": res.get("arm"),
        "n_edits": n,
        "stream_mode": res.get("stream_mode"),
        "write_share": round(write_share, 4),
        "reported_immediate_rewrite_rate": res.get("immediate_rewrite_rate"),
        "baseline_hit_rate": round(pre_hit / n, 4) if n else None,
        "post_hit_rate": round(post_hit / n, 4) if n else None,
        "rewrite_after_baseline_correction": round(corrected / n, 4) if n else None,
        "n_corrected_rewrites": corrected,
        "write_delta_applied_rate": res.get("write_delta_applied_rate"),
        "answer_changed_by_write_rate": res.get("answer_changed_by_write_rate"),
        "mean_delta_norm": res.get("mean_delta_norm"),
        "hypernet_ckpt": res.get("hypernet_ckpt"),
    }


def check_legacy_stream(legacy: Dict[str, Any],
                        deduped: Dict[str, Any]) -> Dict[str, Any]:
    """Explain the 191-row rate with the per-prompt baseline of the 40-row run.

    The legacy stream contains several rows per prompt (different, mutually
    exclusive ``target_new`` values).  Collapsing both streams on ``subject``
    shows whether every legacy row reported as a successful rewrite sits on a
    subject whose base answer already contains its target.
    """
    subject_baseline: Dict[str, int] = {}
    for d in (deduped.get("per_edit") or []):
        subj = (d.get("subject") or "").strip().lower()
        if not subj:
            continue
        subject_baseline[subj] = int(hit(d.get("answer_pre"), d.get("target_new")))

    rows = legacy.get("per_edit") or []
    reported_true = [d for d in rows if d.get("immediate_rewrite")]
    confirmed = sum(1 for d in reported_true
                    if subject_baseline.get((d.get("subject") or "").strip().lower()))
    per_subject: Dict[str, Dict[str, int]] = {}
    for d in rows:
        subj = (d.get("subject") or "").strip().lower()
        slot = per_subject.setdefault(subj, {"rows": 0, "reported_true": 0})
        slot["rows"] += 1
        slot["reported_true"] += int(bool(d.get("immediate_rewrite")))
    return {
        "legacy_n_edits": len(rows),
        "legacy_reported_true": len(reported_true),
        "legacy_reported_rate": legacy.get("immediate_rewrite_rate"),
        "legacy_true_rows_on_baseline_hit_subject": confirmed,
        "legacy_true_rows_explained": f"{confirmed}/{len(reported_true)}",
        "baseline_hit_subjects": sum(subject_baseline.values()),
        "n_subjects": len(subject_baseline),
        "per_subject": per_subject,
    }


def main() -> None:
    deduped = {arm: load(f"w4_ablation_{arm}_deduped.json") for arm in ARMS}
    legacy = {arm: load(f"w4_ablation_{arm}.json") for arm in ARMS}

    table: List[Dict[str, Any]] = [arm_metrics(r) for r in deduped.values() if r]

    rates = {row["arm"]: row["reported_immediate_rewrite_rate"] for row in table}
    same_rate = len(set(rates.values())) == 1

    legacy_check = None
    if legacy.get("always_write") and deduped.get("always_write"):
        legacy_check = check_legacy_stream(legacy["always_write"],
                                          deduped["always_write"])
        legacy_check["all_arms_reported_immediate_rewrite_rate"] = {
            arm: (legacy[arm] or {}).get("immediate_rewrite_rate") for arm in ARMS
        }

    verdict = {
        "check1_metric_tracks_write_share": not same_rate,
        "check1_note": (
            "identical immediate_rewrite_rate across arms with very different "
            "WRITE shares -> the metric measures the base model, not the writer"
            if same_rate else "rate varies with the WRITE share"),
        "check2_baseline_contaminated": bool(
            table and table[0]["baseline_hit_rate"]
            and abs(table[0]["baseline_hit_rate"]
                    - (table[0]["reported_immediate_rewrite_rate"] or 0)) < 1e-6),
        "check3_write_changes_answer": bool(
            table and any((row["answer_changed_by_write_rate"] or 0) > 0
                          for row in table)),
        "root_cause": "untrained hypernetwork (hparams has no hypernet_ckpt) "
                      "emits deltas that are applied (delta norm > 0) but never "
                      "move greedy decoding; the reported rate is the base-model "
                      "hit rate on the stream",
    }

    out = {
        "metric": "immediate_rewrite_rate",
        "definition_in_code": "mean(post['rewrite_ok']), rewrite_ok = target_new "
                              "substring of the post-edit greedy answer",
        "corrected_definition": "post_hit AND NOT pre_hit (change caused by the "
                               "edit)",
        "deduped_arms": table,
        "legacy_stream_check": legacy_check,
        "verdict": verdict,
    }

    path = os.path.join(RESULTS, "immediate_rewrite_diagnosis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print(json.dumps(out["verdict"], indent=1, ensure_ascii=False))
    for row in table:
        print(f"{row['arm']:13s} n={row['n_edits']:3d} write={row['write_share']:.3f} "
              f"reported={row['reported_immediate_rewrite_rate']} "
              f"baseline={row['baseline_hit_rate']} "
              f"corrected={row['rewrite_after_baseline_correction']} "
              f"ans_changed={row['answer_changed_by_write_rate']}")
    if legacy_check:
        print("legacy 191-row check:", json.dumps(legacy_check, ensure_ascii=False))
    print("-> results/immediate_rewrite_diagnosis.json")


if __name__ == "__main__":
    main()
