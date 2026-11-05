"""run_w4_gate_ablation.py -- W4: gate ablations on the CALM edit stream.

Owner: TANG I CHOI (MC631517).

Runs one ablation arm at a time over the same edit stream and reports the
metrics that decide whether conflict-aware gating is worth it:

    route mix            how many edits are WRITE / ROUTE / REJECT
    immediate rewrite    is the new target in the model's answer right after
                         the edit is applied (parametric success)
    final model ret.     share of *all* stream edits still answered correctly
                         by the parameters at the end of the stream
                         (this is what parameter pollution destroys)
    final e2e ret.       answered by parameters *or* found in side memory
    locality             share of held-out facts whose answer is unchanged by
                         the whole stream (1.0 = no collateral damage)
    memory               number of stored side-memory entries

Arms: ``always_write`` (lower bound on locality), ``always_route`` (upper bound
on locality, no parametric gain), ``rule_gate`` (W1 manual rules) and
``learned_gate`` (W3 trained classifier).

The arm is injected through ``hparams.conflict_gate_path``; the two CALMEditor
routing guards of W4 (``_first_write_guard`` / ``_write_budget_guard``) are
switched off on the instance so that the ablation measures the gate alone and
nothing else.  No teammate file is edited by this script.

Usage::

    python run_w4_gate_ablation.py --arm learned_gate --subjects 40
    python run_w4_gate_ablation.py --all-arms --subjects 40
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EASYEDIT = os.path.join(REPO, "EasyEdit")
ARMS_DIR = os.path.join(HERE, "gate_arms")
W3_DATA = os.path.join(REPO, "TANG I CHOI",
                       "w3_conflict_gate_training_and_classifier",
                       "gate_datasets")
DEFAULT_HPARAMS = os.path.join(EASYEDIT, "hparams", "CALM", "gpt2-xl.yaml")
OUT_DIR = os.path.join(HERE, "results")

ARMS = ["always_write", "always_route", "rule_gate", "learned_gate"]

# arm name -> module file inside gate_arms/
ARM_FILES = {
    "always_write": "always_write_gate.py",
    "always_route": "always_route_gate.py",
    "rule_gate": "rule_gate.py",
    "learned_gate": "learned_gate.py",
}

sys.path.insert(0, EASYEDIT)


# --------------------------------------------------------------------------- #
# Stream construction
# --------------------------------------------------------------------------- #

def load_stream(n_subjects: int, max_per_subject: int = 6,
                dedupe_prompt: bool = False):
    """Build a deterministic edit stream from the W3 held-out test split.

    Edits are grouped by subject so that the stream contains repeated edits on
    the same entity (the realistic lifelong-editing regime, and the only way
    prior-edit conflicts can arise inside one stream).  Within a subject the
    rows keep the dataset order.

    Returns ``(stream, heldout)`` where ``heldout`` are facts on subjects that
    never appear in the stream (used for the locality measurement).
    """
    rows = [json.loads(l) for l in open(os.path.join(W3_DATA, "gate_test.jsonl"))]
    order: List[str] = []
    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        s = r.get("subject") or ""
        if s not in by_subject:
            by_subject[s] = []
            order.append(s)
        by_subject[s].append(r)

    stream: List[Dict[str, Any]] = []
    used = set()
    for s in order[:n_subjects]:
        chunk = by_subject[s][:max_per_subject]
        stream.extend(chunk)
        used.add(s)

    heldout = [r for r in rows if (r.get("subject") or "") not in used]
    return stream, heldout[:60]


def dedupe_by_prompt(stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one edit per distinct prompt.

    The CounterFact situations behind W3 share a single prompt per subject but
    carry *different* ``target_new`` values (e.g. ``Winchester railway station
    was named for`` is asked to become ``Winchester``, ``Alps`` and ``Cairo``).
    Two edits on the same prompt with contradictory targets cannot both be
    satisfied by any editor, so ``immediate_rewrite`` on the raw stream is
    bounded by ``1 / edits_per_prompt`` (~0.21 here) no matter how good the
    gate or the writer is.  Deduplicating gives a stream on which the metric
    actually measures the writer.
    """
    seen = set()
    out = []
    for r in stream:
        key = (r.get("prompt") or "").strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #

def answer_of(editor, prompt: str, max_new_tokens: int = 5):
    return editor._current_answer_and_confidence(prompt, max_new_tokens=max_new_tokens)


def hit(answer: str, target: str) -> bool:
    return str(target).strip().lower() in str(answer).strip().lower()


def build_editor(arm: str, hparams_path: str, hypernet_scale: float = None):
    from easyeditor.editors.calm_editor import CALMEditor  # noqa: E402
    from easyeditor.models.calm.calm_hparams import CALMHyperParams  # noqa: E402

    import torch
    # CALMEditor caches the gate module under a fixed name; drop it so that a
    # second arm in the same process really loads its own file.
    sys.modules.pop("calm_conflict_gate", None)
    hparams = CALMHyperParams.from_hparams(hparams_path)
    hparams.conflict_gate_path = os.path.join(ARMS_DIR, ARM_FILES[arm])
    hparams.write_budget = -1      # guards off: measure the gate only
    hparams.dual_write = False
    if hypernet_scale is not None:
        hparams.hypernet_scale = float(hypernet_scale)
    if torch.backends.mps.is_available():
        hparams.device = "mps"

    editor = CALMEditor.from_hparams(hparams)
    # ablation hygiene: neutralise the W4 routing guards on this instance only
    editor._first_write_guard = types.MethodType(
        lambda self, request, decision: decision, editor)
    editor._write_budget_guard = types.MethodType(
        lambda self, request, decision: decision, editor)

    # diagnosis hook: record the Frobenius norm of every parameter delta that
    # the hypernetwork actually applies, so the results can tell "the gate said
    # WRITE" apart from "the WRITE changed the weights" (see
    # docs/w4_immediate_rewrite_diagnosis.md).
    if getattr(editor, "hypernetwork", None) is not None:
        editor._last_delta_norm = 0.0
        _raw_apply = editor.hypernetwork.apply_update

        def _measured_apply(deltas, _raw=_raw_apply, _ed=editor):
            import torch as _t

            norm = 0.0
            for d in deltas.values():
                norm += float(_t.linalg.matrix_norm(d.float())) ** 2
            _ed._last_delta_norm = norm ** 0.5
            return _raw(deltas)

        editor.hypernetwork.apply_update = _measured_apply
    return editor, hparams


def run_arm(arm: str, subjects: int, hparams_path: str, max_per_subject: int,
            verbose: bool = False, hypernet_scale: float = None,
            dedupe_prompt: bool = False) -> Dict[str, Any]:
    stream, heldout = load_stream(subjects, max_per_subject)
    raw_len = len(stream)
    if dedupe_prompt:
        stream = dedupe_by_prompt(stream)
    print(f"[{arm}] stream={len(stream)} edits over {subjects} subjects "
          f"(raw {raw_len}, dedupe_prompt={dedupe_prompt}), "
          f"heldout={len(heldout)}", flush=True)

    t0 = time.time()
    editor, hparams = build_editor(arm, hparams_path, hypernet_scale)
    print(f"[{arm}] editor ready in {time.time() - t0:.1f}s "
          f"(device={hparams.device})", flush=True)

    # --- locality baseline before any edit ----------------------------- #
    held_before = []
    for r in heldout:
        ans, _ = answer_of(editor, r["prompt"])
        held_before.append(ans)

    # --- run the stream -------------------------------------------------- #
    decisions: List[Dict[str, Any]] = []
    t_run = time.time()
    for i, r in enumerate(stream):
        req = {
            "prompt": r["prompt"],
            "target_new": r["target_new"],
            "ground_truth": r.get("current_answer", ""),
            "subject": r.get("subject") or "",
            "relation": r.get("relation") or "fact",
        }
        metrics, _, extra = editor.edit_requests([req], sequential_edit=True,
                                                 verbose=False)
        m = metrics[0]
        decisions.append({
            "id": r.get("id"),
            "subject": req["subject"],
            "relation": req["relation"],
            "target_new": req["target_new"],
            "situation": r.get("situation"),
            "label4": r.get("label4"),
            "gold_action": r.get("action"),
            "decision": m["calm"]["decision"],
            "route": m["calm"]["route"],
            "immediate_rewrite": m["post"]["rewrite_ok"],
            "answer_pre": m["pre"].get("answer"),
            "answer_post": m["post"].get("answer"),
            "delta_norm": round(float(getattr(editor, "_last_delta_norm", 0.0)), 6),
        })
        if verbose or (i + 1) % 25 == 0:
            print(f"[{arm}] {i + 1}/{len(stream)} {decisions[-1]['decision']} "
                  f"rewrite={decisions[-1]['immediate_rewrite']}", flush=True)
    stream_seconds = time.time() - t_run

    # --- final retention -------------------------------------------------- #
    model_hits, e2e_hits = 0, 0
    for r, d in zip(stream, decisions):
        ans, _ = answer_of(editor, r["prompt"])
        ok_model = hit(ans, r["target_new"])
        ok_mem = editor.memory.lookup(r.get("subject") or "",
                                     r.get("relation") or "fact",
                                     r["target_new"]) is not None
        model_hits += int(ok_model)
        e2e_hits += int(ok_model or ok_mem)
        d["final_model_hit"] = bool(ok_model)
        d["final_e2e_hit"] = bool(ok_model or ok_mem)
        d["final_answer"] = ans

    # --- locality after the stream --------------------------------------- #
    held_after = []
    for r in heldout:
        ans, _ = answer_of(editor, r["prompt"])
        held_after.append(ans)
    unchanged = sum(1 for a, b in zip(held_before, held_after)
                    if a.strip().lower() == b.strip().lower())

    n = len(stream)
    counts = {"WRITE": 0, "ROUTE": 0, "REJECT": 0}
    for d in decisions:
        counts[d["decision"]] = counts.get(d["decision"], 0) + 1
    gate_accuracy = (sum(1 for d in decisions if d["decision"] == d["gold_action"])
                     / n) if n else 0.0

    result = {
        "arm": arm,
        "model": hparams.model_name,
        "hparams": os.path.basename(hparams_path),
        "n_edits": n,
        "n_subjects": subjects,
        "stream_mode": "deduped-prompt" if dedupe_prompt else "grouped-multi-target",
        "n_stream_rows_before_dedupe": raw_len,
        "n_distinct_prompts": len({(d.get("prompt") or "").strip()
                                   for d in stream}),
        "hypernet_ckpt": getattr(hparams, "hypernet_ckpt", None)
        or "untrained (random init)",
        "write_delta_applied_rate": round(
            sum(1 for d in decisions if d["delta_norm"] > 0) / n, 4) if n else 0.0,
        "mean_delta_norm": round(
            sum(d["delta_norm"] for d in decisions) / n, 6) if n else 0.0,
        "answer_changed_by_write_rate": round(
            sum(1 for d in decisions
                if d["decision"] == "WRITE"
                and (d.get("answer_pre") or "").strip().lower()
                != (d.get("answer_post") or "").strip().lower())
            / max(1, sum(1 for d in decisions if d["decision"] == "WRITE")), 4),
        "decisions": counts,
        "decision_share": {k: round(v / n, 4) for k, v in counts.items()} if n else {},
        "immediate_rewrite_rate": round(
            sum(1 for d in decisions if d["immediate_rewrite"]) / n, 4) if n else 0.0,
        "final_model_retention": round(model_hits / n, 4) if n else 0.0,
        "final_e2e_retention": round(e2e_hits / n, 4) if n else 0.0,
        "locality_unchanged": round(unchanged / len(heldout), 4) if heldout else 1.0,
        "n_heldout": len(heldout),
        "gate_accuracy_vs_gold": round(gate_accuracy, 4),
        "memory_entries": editor.memory.stats().get("total_stored"),
        "memory_stats": editor.memory.stats(),
        "stream_seconds": round(stream_seconds, 1),
        "seconds_per_edit": round(stream_seconds / n, 3) if n else None,
        "hypernet_scale": hparams.hypernet_scale,
        "per_edit": decisions,
    }
    return result


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="learned_gate", choices=ARMS)
    ap.add_argument("--all-arms", action="store_true")
    ap.add_argument("--subjects", type=int, default=40,
                    help="number of distinct subjects in the stream")
    ap.add_argument("--max-per-subject", type=int, default=6)
    ap.add_argument("--hparams", default=DEFAULT_HPARAMS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tag", default="", help="suffix for the result file")
    ap.add_argument("--hypernet-scale", type=float, default=None,
                    help="override hparams.hypernet_scale (stress probe)")
    ap.add_argument("--dedupe-prompts", action="store_true",
                    help="one edit per distinct prompt (removes contradictory "
                         "targets on the same prompt; makes immediate_rewrite "
                         "interpretable -- see docs/w4_immediate_rewrite_diagnosis.md)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    arms = ARMS if args.all_arms else [args.arm]
    os.makedirs(args.out_dir, exist_ok=True)
    # The CALM hparams file uses paths relative to the EasyEdit root
    # (./hugging_cache/gpt2-xl, ../FONG HIO LAM/...), so run from there.
    cwd = os.getcwd()
    os.chdir(EASYEDIT)
    try:
        for arm in arms:
            res = run_arm(arm, args.subjects, args.hparams, args.max_per_subject,
                          verbose=args.verbose, hypernet_scale=args.hypernet_scale,
                          dedupe_prompt=args.dedupe_prompts)
            name = f"w4_ablation_{arm}{args.tag}.json"
            with open(os.path.join(args.out_dir, name), "w") as f:
                json.dump(res, f, indent=1, ensure_ascii=False)
            print(f"[{arm}] decisions={res['decisions']} "
                  f"imm={res['immediate_rewrite_rate']} "
                  f"model_ret={res['final_model_retention']} "
                  f"e2e_ret={res['final_e2e_retention']} "
                  f"locality={res['locality_unchanged']} "
                  f"({res['stream_seconds']}s) -> results/{name}", flush=True)
            # release the 1.5B model before the next arm
            import gc
            import torch
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
