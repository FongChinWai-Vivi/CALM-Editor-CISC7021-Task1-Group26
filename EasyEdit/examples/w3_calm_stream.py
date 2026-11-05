"""
w3_calm_stream.py -- CALM Gate 3: end-to-end sequential (lifelong) edit stream.

Runs a sequential edit stream over the first ``--max-edits`` records of
``EasyEdit/data/counterfact.json`` with gpt2-small.  Every edit goes through
the CALMEditor pipeline (reusing the W3 implementation, no changes):

    ConflictGate.decide -> WRITE  (hypernetwork low-rank weight update)
                       -> ROUTE  (side memory, no parameter change)
                       -> REJECT (skip)

The script reports:
  * route distribution (WRITE / ROUTE / REJECT counts and shares)
  * side-memory usage statistics
  * total wall time and average time per edit
  * a sampled rewrite-retention check: a few edits from the middle and from
    the end of the stream are re-measured on the *final* model to see whether
    the edited fact is still visible in the model output.

Usage (from the repository root, with .venv active)::

    .venv\\Scripts\\python.exe EasyEdit\\examples\\w3_calm_stream.py
    .venv\\Scripts\\python.exe EasyEdit\\examples\\w3_calm_stream.py --max-edits 200
    .venv\\Scripts\\python.exe EasyEdit\\examples\\w3_calm_stream.py --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # EasyEdit/
sys.path.insert(0, ROOT)
_CALL_CWD = os.getcwd()
# The CALM hparams file uses paths relative to the EasyEdit root
# (e.g. ./hugging_cache/gpt2-small, ../FONG HIO LAM/...), so switch the
# working directory to EasyEdit before constructing the editor.
os.chdir(ROOT)

from easyeditor.editors.calm_editor import CALMEditor          # noqa: E402
from easyeditor.models.calm.calm_hparams import CALMHyperParams  # noqa: E402


def load_requests(data_path: str, max_edits: int) -> List[Dict[str, Any]]:
    """Load the first ``max_edits`` CounterFact records as CALM edit requests.

    ``requested_rewrite.prompt`` uses the EasyEdit ``{}`` placeholder, so the
    subject is substituted in to obtain a concrete prompt per record.
    """
    with open(data_path, encoding="utf-8") as fh:
        records = json.load(fh)
    records = records[:max_edits]

    requests: List[Dict[str, Any]] = []
    for rec in records:
        rr = rec["requested_rewrite"]
        prompt = rr["prompt"].replace("{}", rr["subject"])
        target_new = rr["target_new"]["str"] if isinstance(rr["target_new"], dict) else rr["target_new"]
        target_true = rr["target_true"]["str"] if isinstance(rr["target_true"], dict) else rr["target_true"]
        requests.append({
            "prompt": prompt,
            "subject": rr["subject"],
            "relation": rr.get("relation_id", "fact"),
            "target_new": target_new,
            "ground_truth": target_true,
        })
    return requests


def pick_sample_indices(n: int, k: int = 8) -> List[int]:
    """Sample ``k`` indices spread over the middle and the end of the stream."""
    mid = n // 2
    half = k // 2
    step_mid = max(1, (n // 4) // max(1, half))
    mid_idx = list(range(max(0, mid - half * step_mid), mid, step_mid))[:half]
    end_start = max(mid, n - half * max(1, n // (half * 10)))
    step_end = max(1, n // (half * 4))
    end_idx = list(range(end_start, n, step_end))[:half]
    idx = sorted(set(mid_idx + end_idx))
    return idx


def measure_retention(editor: CALMEditor, request: Dict[str, Any]) -> Dict[str, Any]:
    """Re-measure one edit on the final model (is target_new still produced?)."""
    m = editor._measure(request)
    return {
        "subject": request["subject"],
        "prompt": request["prompt"],
        "target_new": request["target_new"],
        "answer": m["answer"],
        "rewrite_ok": m["rewrite_ok"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CALM Gate 3 sequential edit stream")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "counterfact.json"))
    ap.add_argument("--hparams", default=os.path.join(ROOT, "hparams", "CALM", "gpt2-small.yaml"))
    ap.add_argument("--max-edits", type=int, default=1000)
    ap.add_argument("--sample-edits", type=int, default=8,
                    help="number of middle/end edits to re-measure for retention")
    ap.add_argument("--hypernet-ckpt", default=None,
                    help="optional trained hypernetwork checkpoint to load "
                         "(W3 meta-trained LowRankHypernetwork)")
    ap.add_argument("--count-memory-rewrites", action="store_true",
                    help="also count side-memory ROUTE hits as successful "
                         "rewrites (CALM end-to-end semantics); default keeps "
                         "the model-only metric")
    ap.add_argument("--e2e-retention", action="store_true",
                    help="retention check uses CALM end-to-end semantics: a "
                         "ROUTE edit is retained iff its (subject, relation, "
                         "target_new) entry is found in side memory; default "
                         "keeps the model-only retention metric")
    ap.add_argument("--gate-write-threshold", type=float, default=None,
                    help="override ConflictGate.write_threshold (lower = fewer "
                         "parameter WRITEs, more edits routed to side memory)")
    ap.add_argument("--gate-route-threshold", type=float, default=None,
                    help="override ConflictGate.route_threshold")
    ap.add_argument("--write-budget", type=int, default=None,
                    help="cap parameter WRITEs over the stream; further edits "
                         "are routed to side memory (W4 fix, -1 = unlimited)")
    ap.add_argument("--dual-write", action="store_true",
                    help="also store WRITE edits into side memory so end-to-end "
                         "retention is memory-backed for parameter edits too")
    ap.add_argument("--verify-commit", action="store_true",
                    help="Claude #9: tentatively apply WRITE deltas; if the "
                         "post-edit answer misses target_new, roll back the "
                         "parameter delta and store the edit in side memory "
                         "instead (prevents failed writes from polluting the "
                         "weights on long streams)")
    ap.add_argument("--feature-mode", default=None,
                    choices=["mean-emb", "key-hidden"],
                    help="Claude #3: hypernetwork edit-feature representation "
                         "(must match the ckpt's training mode; default is "
                         "mean-emb, the trained ckpt meta is validated)")
    ap.add_argument("--hypernet-rank", type=int, default=None,
                    help="override hypernetwork rank (must match ckpt meta)")
    ap.add_argument("--hypernet-hidden", type=int, default=None,
                    help="override hypernetwork hidden size (must match ckpt meta)")
    ap.add_argument("--delta-norm-cap", type=float, default=None,
                    help="post-hoc cap: rescale each emitted delta to "
                         "||delta||_F <= cap (absolute Frobenius bound)")
    ap.add_argument("--infer-scale", type=float, default=None,
                    help="override the hypernetwork scale at inference "
                         "(sweep scale without retraining)")
    ap.add_argument("--norm-tau", type=float, default=None,
                    help="relative cap matching W4 retrain: rescale each delta "
                         "to ||delta||_F <= tau * ||W_orig||_F (use with the "
                         "tau-trained hypernetwork ckpt)")
    ap.add_argument("--ns-proj", action="store_true",
                    help="project each WRITE delta onto the null space of the "
                         "most recent written deltas (AlphaEdit-style, no "
                         "retraining; retention-per-slot fix)")
    ap.add_argument("--ns-keep", type=int, default=8,
                    help="number of recent deltas kept in the null-space stack")
    ap.add_argument("--track-norm-log", default=None,
                    help="write a per-edit per-param norm log (JSON) for "
                         "norm-growth instrumentation (W4)")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()
    # Resolve a possibly-relative --output-dir against the caller's cwd
    # (os.chdir(ROOT) above changed the process cwd already).
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(_CALL_CWD, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[CALM] stream | data={args.data} max_edits={args.max_edits}")
    requests = load_requests(args.data, args.max_edits)
    print(f"[CALM] loaded {len(requests)} requests")

    hparams = CALMHyperParams.from_hparams(args.hparams)
    if args.hypernet_ckpt:
        hparams.hypernet_ckpt = os.path.abspath(args.hypernet_ckpt)
        print(f"[CALM] loading trained hypernetwork: {hparams.hypernet_ckpt}")
    if args.hypernet_rank is not None:
        hparams.hypernet_rank = args.hypernet_rank
        print(f"[CALM] hypernet rank -> {args.hypernet_rank}")
    if args.hypernet_hidden is not None:
        hparams.hypernet_hidden = args.hypernet_hidden
        print(f"[CALM] hypernet hidden -> {args.hypernet_hidden}")
    if args.gate_write_threshold is not None:
        hparams.gate_write_threshold = args.gate_write_threshold
        print(f"[CALM] gate write_threshold -> {args.gate_write_threshold}")
    if args.gate_route_threshold is not None:
        hparams.gate_route_threshold = args.gate_route_threshold
        print(f"[CALM] gate route_threshold -> {args.gate_route_threshold}")
    if args.write_budget is not None:
        hparams.write_budget = args.write_budget
        print(f"[CALM] write_budget -> {args.write_budget}")
    if args.dual_write:
        hparams.dual_write = True
        print("[CALM] dual_write -> True")
    if args.verify_commit:
        hparams.verify_commit = True
        print("[CALM] verify_commit -> True (Claude #9)")
    if args.feature_mode:
        hparams.feature_mode = args.feature_mode
        print(f"[CALM] feature_mode -> {args.feature_mode} (Claude #3)")
    editor = CALMEditor.from_hparams(hparams)
    print(f"[CALM] editor ready (model={hparams.model_name} alg={hparams.hypernet_alg} "
          f"ckpt={hparams.hypernet_ckpt})")
    if args.delta_norm_cap is not None:
        cap = args.delta_norm_cap
        if isinstance(cap, str):
            try:
                cap = json.loads(cap)   # allow {"param": cap} per-param dict
            except json.JSONDecodeError:
                cap = float(cap)
        editor.hypernetwork.hypernet.norm_cap = cap
        print(f"[CALM] delta-norm cap -> {cap}")
    if args.norm_tau is not None:
        from easyeditor.util import nethook
        inner_params = list(hparams.inner_params)
        base_norms = {}
        with torch.no_grad():
            for name in inner_params:
                w = nethook.get_parameter(editor.model, name)
                base_norms[name] = float(w.data.norm(p="fro").item())
        editor.hypernetwork.hypernet.norm_tau = args.norm_tau
        editor.hypernetwork.hypernet.base_norms = base_norms
        print(f"[CALM] norm-tau -> {args.norm_tau} "
              f"(base norms for {len(base_norms)} inner params)")
    if args.ns_proj:
        editor.hypernetwork.ns_proj = True
        editor.hypernetwork.ns_keep = args.ns_keep
        print(f"[CALM] null-space projection ON (keep={args.ns_keep})")
    if args.infer_scale is not None:
        editor.hypernetwork.hypernet.scale = args.infer_scale
        print(f"[CALM] infer-scale override -> {args.infer_scale}")
    if args.track_norm_log:
        editor.hypernetwork.track_norms = True
        print(f"[CALM] tracking per-edit norms -> {args.track_norm_log}")

    sample_idx = pick_sample_indices(len(requests), args.sample_edits)

    t0 = time.time()
    all_metrics, _model, extra = editor.edit_requests(
        requests, sequential_edit=True, verbose=False
    )
    total_seconds = time.time() - t0

    route_counts: Dict[str, int] = {}
    decision_counts: Dict[str, int] = {}
    for m in all_metrics:
        calm = m["calm"]
        route_counts[calm["route"]] = route_counts.get(calm["route"], 0) + 1
        decision_counts[calm["decision"]] = decision_counts.get(calm["decision"], 0) + 1

    n = len(all_metrics)
    route_share = {k: round(v / n, 4) for k, v in sorted(route_counts.items())}

    # Immediate (stream-internal) rewrite success: measured right after each
    # edit.  Kept separate from the final retention check below.
    # model-only metric: the edited weights themselves must answer target_new.
    immediate_model_ok = sum(1 for m in all_metrics if m["post"]["rewrite_ok"])
    # CALM end-to-end metric (optional): a ROUTE to side memory counts as a
    # successful rewrite when the stored (subject, relation, target_new) entry
    # is found by the memory (the memory stores the edit exactly at ROUTE).
    if args.count_memory_rewrites:
        immediate_ok = 0
        memory_hits = 0
        for i, m in enumerate(all_metrics):
            calm = m["calm"]
            if calm["route"] in ("side_memory", "verify_rollback"):
                req = requests[i]
                hit = editor.memory.lookup(
                    req.get("subject") or "",
                    req.get("relation") or "fact",
                    req["target_new"],
                )
                memory_hits += int(hit is not None)
                immediate_ok += int(hit is not None)
            else:
                immediate_ok += int(m["post"]["rewrite_ok"])
    else:
        immediate_ok = immediate_model_ok
        memory_hits = 0

    # Retention check on the final model (after the whole stream).
    # CALM end-to-end semantics (--e2e-retention): a ROUTE edit is retained
    # iff its entry is still present in side memory (memory is part of the
    # CALM architecture); otherwise we measure the final model weights only.
    retention = []
    for i in sample_idx:
        calm = all_metrics[i]["calm"]
        routed = calm["route"] in ("side_memory", "verify_rollback")
        if args.e2e_retention and routed:
            req = requests[i]
            hit = editor.memory.lookup(
                req.get("subject") or "",
                req.get("relation") or "fact",
                req["target_new"],
            )
            retention.append({
                "index": i,
                "position": "mid" if i < n // 2 else "end",
                "immediate_ok": bool(all_metrics[i]["post"]["rewrite_ok"]),
                "route": calm["route"],
                "e2e": True,
                "memory_hit": bool(hit is not None),
                "subject": req["subject"],
                "prompt": req["prompt"],
                "target_new": req["target_new"],
                "answer": hit if hit is not None else "",
                "rewrite_ok": bool(hit is not None),
            })
        else:
            m = measure_retention(editor, requests[i])
            retention.append({
                "index": i,
                "position": "mid" if i < n // 2 else "end",
                "immediate_ok": bool(all_metrics[i]["post"]["rewrite_ok"]),
                "route": calm["route"],
                "e2e": False,
                **m,
            })
    retention_ok = sum(1 for r in retention if r["rewrite_ok"])

    result = {
        "model": hparams.model_name,
        "hparams_yaml": args.hparams,
        "data": args.data,
        "max_edits": len(requests),
        "total_seconds": round(total_seconds, 3),
        "avg_seconds_per_edit": round(total_seconds / max(1, n), 4),
        "route_counts": route_counts,
        "route_share": route_share,
        "decision_counts": decision_counts,
        "immediate_rewrite_ok": immediate_ok,
        "immediate_rewrite_rate": round(immediate_ok / max(1, n), 4),
        "immediate_model_rewrite_ok": immediate_model_ok,
        "immediate_model_rewrite_rate": round(immediate_model_ok / max(1, n), 4),
        "memory_rewrite_hits": memory_hits,
        "memory_stats": extra.get("memory_stats"),
        "gate_summary": extra.get("gate_summary"),
        "sample_retention": retention,
        "retention_ok": retention_ok,
        "delta_norm_cap": args.delta_norm_cap,
        "norm_tau": args.norm_tau,
        "ns_proj": args.ns_proj,
        "verify_commit": args.verify_commit,
        "feature_mode": getattr(hparams, "feature_mode", "mean-emb"),
        "status": "PASS",
    }

    if args.track_norm_log:
        log_path = args.track_norm_log
        if not os.path.isabs(log_path):
            log_path = os.path.join(_CALL_CWD, log_path)
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump({
                "max_edits": len(requests),
                "delta_norm_cap": args.delta_norm_cap,
                "norm_log": editor.hypernetwork._norm_log,
            }, fh, ensure_ascii=False, indent=2)
        print(f"[CALM] norm log written to: {log_path} "
              f"({len(editor.hypernetwork._norm_log)} records)")

    out_path = os.path.join(
        args.output_dir,
        f"w3_calm_stream_{'1k' if len(requests) == 1000 else len(requests)}"
        f"{'_trained' if args.hypernet_ckpt else ''}.json",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"\n=== CALM Gate 3 stream summary ===")
    print(f"model={hparams.model_name} edits={n} total={total_seconds:.1f}s "
          f"avg={total_seconds / max(1, n):.3f}s/edit")
    print(f"route_counts={route_counts}")
    print(f"route_share={route_share}")
    print(f"decision_counts={decision_counts}")
    print(f"immediate_rewrite_rate={result['immediate_rewrite_rate']} "
          f"({immediate_ok}/{n})")
    print(f"memory_stats={extra.get('memory_stats')}")
    print(f"retention_samples={len(retention)} "
          f"retained={sum(1 for r in retention if r['rewrite_ok'])}")
    print(f"JSON result written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
