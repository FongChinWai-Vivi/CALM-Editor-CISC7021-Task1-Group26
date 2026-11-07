# -*- coding: utf-8 -*-
"""W6 end-to-end scorer for a finished deferred-write run.

The deferred-write work fixed the *parameter* channel only. Whether the stream is
"end-to-end fixed" is a different question: after all 100 edits, how many edits are
answered correctly by *some* channel, and is each channel's credit leak-free?

This script does not run a model. It joins
  - a `results_defer_W*.json` produced by `w6_defer_stream.py` (per-edit route + final_ok), and
  - a memory store dump (or an assumed memory recall),
and reports end-to-end accuracy with per-channel attribution and leakage assertions.

Two modes:

  --assume-memory-recall 1.0   # ceiling: what end-to-end becomes once the memory channel reads
  --assume-memory-recall 0.0   # floor: what it is today (memory is not read on the answer path)
  --memory-json store.json     # real store: [{"subject":..., "relation":..., "target":...}, ...]

Leakage guards (hard failures, not warnings):
  1. the lookup key may only contain (subject, relation); a key containing the target is rejected.
  2. any prompt used for retrieval that contains the new target is rejected -- a query that
     already contains the answer cannot be used as evidence of retrieval.

Usage
-----
  python examples/w6_end2end_eval.py \
      --run w6run/R_o0_K/results_defer_W100.json --offset 0 --edits 100 \
      --assume-memory-recall 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAM_ROUTES = {"parameter", "hypernetwork_batched", "deferred_batched"}
MEMORY_ROUTES = {"side_memory", "batched_route_fallback", "memory"}
DEAD_ROUTES = {"gate_reject"}


def load_edits(data_path: str, offset: int, edits: int) -> List[Dict[str, Any]]:
    with open(data_path, encoding="utf-8") as fh:
        records = json.load(fh)[offset:offset + edits]
    out: List[Dict[str, Any]] = []
    for rec in records:
        rr = rec["requested_rewrite"]
        out.append({
            "subject": rr["subject"],
            "relation": rr.get("relation_id", "fact"),
            "prompt": rr["prompt"].replace("{}", rr["subject"]),
            "target_new": rr["target_new"]["str"] if isinstance(rr["target_new"], dict)
            else rr["target_new"],
            "paraphrases": rec.get("paraphrase_prompts") or [],
        })
    return out


def channel_of(route: str) -> str:
    r = (route or "").lower()
    if r in PARAM_ROUTES:
        return "parameter"
    if r in MEMORY_ROUTES:
        return "memory"
    if r in DEAD_ROUTES:
        return "reject"
    return "other"


def memory_key(edit: Dict[str, Any]) -> Tuple[str, str]:
    """The only key the memory channel is allowed to use. Never the target."""
    return (str(edit["subject"]).strip().lower(), str(edit["relation"]).strip().lower())


def build_store(memory_json: Optional[str], edits: List[Dict[str, Any]],
                recall: Optional[float]) -> Dict[Tuple[str, str], Optional[str]]:
    """Map key -> stored target (None means 'not stored' / dropped)."""
    if memory_json:
        with open(memory_json, encoding="utf-8") as fh:
            raw = json.load(fh)
        store: Dict[Tuple[str, str], Optional[str]] = {}
        for item in raw:
            bad = {k for k in item if "target" in k.lower()} - {"target"}
            if bad:
                raise SystemExit(
                    "LEAKAGE: memory store entry carries target-side fields in its key: %s" % sorted(bad))
            key = (str(item["subject"]).strip().lower(), str(item.get("relation", "fact")).strip().lower())
            store[key] = str(item["target"])
        return store
    # assumed recall: recall=1.0 -> every memory-routed edit is retrievable
    assumed = {} if recall is None else {memory_key(e): e["target_new"] for e in edits} if recall >= 1.0 else {}
    return assumed


def score(run: Dict[str, Any], edits: List[Dict[str, Any]],
          store: Dict[Tuple[str, str], Optional[str]],
          assume_recall: Optional[float]) -> Dict[str, Any]:
    rows = run["rows"]
    if len(rows) != len(edits):
        raise SystemExit("run has %d rows but data slice has %d edits" % (len(rows), len(edits)))

    per_channel: Counter = Counter()
    per_channel_hit: Counter = Counter()
    attributions: List[Dict[str, Any]] = []
    leakage_hits = 0

    for row, edit in zip(rows, edits):
        ch = channel_of(row.get("route"))
        per_channel[ch] += 1
        ok = False
        if ch == "parameter":
            ok = bool(row.get("final_ok"))
        elif ch == "memory":
            key = memory_key(edit)
            # leakage guard: the query must not contain the answer
            if edit["target_new"].strip().lower() in edit["prompt"].lower():
                leakage_hits += 1
                ok = False
            elif assume_recall is not None:
                ok = assume_recall >= 1.0
            else:
                stored = store.get(key)
                ok = stored is not None and stored.strip().lower() == str(edit["target_new"]).strip().lower()
        if ok:
            per_channel_hit[ch] += 1
        attributions.append({"slot": row.get("slot"), "route": row.get("route"), "channel": ch,
                             "e2e_ok": ok, "final_ok": row.get("final_ok")})

    n = len(rows)
    e2e = sum(1 for a in attributions if a["e2e_ok"])
    mem_n = per_channel.get("memory", 0)
    return {
        "n": n,
        "end2end_hit": round(e2e / n, 4),
        "channels": {c: {"n": per_channel[c], "hit": per_channel_hit[c]} for c in per_channel},
        "parameter_hit": per_channel_hit.get("parameter", 0),
        "memory_hit": per_channel_hit.get("memory", 0),
        "memory_routed": mem_n,
        "memory_recall_on_routed": (round(per_channel_hit.get("memory", 0) / mem_n, 4) if mem_n else None),
        "dead_edits": per_channel.get("reject", 0),
        "leakage_queries": leakage_hits,
        "attribution": attributions,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="W6 end-to-end scorer (per-channel, leakage-guarded)")
    ap.add_argument("--run", required=True, help="results_defer_W*.json from w6_defer_stream.py")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "counterfact.json"))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--edits", type=int, default=100)
    ap.add_argument("--memory-json", default=None,
                    help="store dump: [{'subject','relation','target'}, ...] (key never contains the target)")
    ap.add_argument("--assume-memory-recall", type=float, default=None,
                    help="skip the store and assume a recall: 1.0 = ceiling, 0.0 = floor")
    ap.add_argument("--out", default=None, help="write the full attribution to this json")
    args = ap.parse_args()

    run = json.load(open(args.run, encoding="utf-8"))
    edits = load_edits(args.data, args.offset, args.edits)
    store = build_store(args.memory_json, edits, args.assume_memory_recall)
    res = score(run, edits, store, args.assume_memory_recall)

    print("run                : %s" % args.run)
    print("slice              : records[%d:%d]" % (args.offset, args.offset + args.edits))
    print("end-to-end hit     : %.4f  (%d/%d)" % (res["end2end_hit"], int(res["end2end_hit"] * res["n"]), res["n"]))
    print("  parameter channel: %d hits / %d routed" % (res["parameter_hit"], res["channels"].get("parameter", {}).get("n", 0)))
    print("  memory channel   : %d hits / %d routed (recall %s)"
          % (res["memory_hit"], res["memory_routed"], res["memory_recall_on_routed"]))
    print("  dead (no channel): %d" % res["dead_edits"])
    if res["leakage_queries"]:
        print("LEAKAGE: %d retrieval queries contain the new target; those edits were scored 0"
              % res["leakage_queries"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
