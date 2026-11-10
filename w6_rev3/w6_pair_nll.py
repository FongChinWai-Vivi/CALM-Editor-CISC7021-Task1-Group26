"""Tang ②/③ — per-entry pairing of in-gate hits with NLL, and the W=50
double-objective retag.

Two outputs, both built from the same per-entry rows so no number can be quoted
without its counterpart:

1. ``pairing`` — for every route recorded by the defer stream, how many entries
   are hits and what the marginal probe-NLL change of those entries is. A row
   that reports "param hits = 37" without the NLL column next to it is, by
   construction, unproducible here: both coordinates come out of one dict.

2. ``retag`` — an offline threshold sweep on the logged ``gate_p_write``. For
   each threshold the admitted set is re-derived and reported as
   (admitted per 100, measured NLL cost of the admitted entries). This is the
   double objective at W=50 without re-running the stream once per threshold.

Usage:
    python w6_pair_nll.py <results_defer_W50.json> [more.json ...] --out paired.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def pairing(rows):
    by_route = defaultdict(lambda: {"n": 0, "hits": 0, "nll": [], "hit_nll": []})

    for r in rows:
        route = r.get("route") or "unknown"
        b = by_route[route]
        b["n"] += 1
        ok = bool(r.get("final_ok"))
        b["hits"] += int(ok)
        d = r.get("nll_delta")
        if d is not None:
            b["nll"].append(d)
            if ok:
                b["hit_nll"].append(d)

    out = {}
    for route, b in sorted(by_route.items()):
        out[route] = {
            "n": b["n"],
            "hits": b["hits"],
            "hit_rate": round(b["hits"] / b["n"], 4) if b["n"] else None,
            "nll_delta_mean": round(mean(b["nll"]), 4),
            "nll_delta_mean_on_hits": round(mean(b["hit_nll"]), 4),
            "paired_row": {
                "hits": b["hits"],
                "nll_delta_mean": round(mean(b["nll"]), 4),
                "note": "coverage and its measured fluency cost in one row",
            },
        }
    return out


def stream_nll(summary, rows):
    """The honest fluency coordinate: what the stream cost, and what a write costs.

    The per-entry trace records the probe NLL after every edit, so the change
    lands at the edit where the flush happened and is ~0 elsewhere.  Averaging
    those zeros would understate the cost, so the stream-level delta is reported
    next to a per-write normalisation and the flush attribution is kept as-is.
    """
    n = len(rows)
    n_writes = sum(1 for r in rows if (r.get("route") or "").startswith("hypernetwork"))
    delta = summary.get("nll_delta")
    trace = summary.get("nll_trace") or []
    flush_costs = [round(trace[i] - trace[i - 1], 4) for i in range(1, len(trace))
                   if abs((trace[i] - trace[i - 1]) or 0.0) > 1e-9]
    return {
        "nll_pre": summary.get("nll_pre"),
        "nll_post": summary.get("nll_post"),
        "nll_delta_stream": delta,
        "n_per_100_edits": delta,
        "nll_delta_per_write": round(delta / n_writes, 4) if n_writes else None,
        "n_writes": n_writes,
        "n_flush_events": len(flush_costs),
        "flush_costs_top5": sorted(flush_costs, key=abs, reverse=True)[:5],
        "note": ("nll_delta_per_write is the number to quote beside a hit count on a "
                 "parameter channel; per-entry nll_delta is 0 for edits that were "
                 "buffered and only becomes non-zero at the flush edit."),
    }


def retag(rows):
    scored = [r for r in rows if r.get("gate_p_write") is not None]
    if not scored:
        return {"available": False,
                "why": "rows carry no gate_p_write (run predates the T4 logging patch)"}

    n = len(rows)
    curve = []
    for t in THRESHOLDS:
        admitted = [r for r in scored if r["gate_p_write"] >= t]
        if not admitted:
            curve.append({"threshold": t, "admitted_per_100": 0.0,
                          "nll_delta_mean_on_admitted": None,
                          "hit_rate_on_admitted": None})
            continue
        curve.append({
            "threshold": t,
            "admitted_per_100": round(100.0 * len(admitted) / n, 2),
            "nll_delta_mean_on_admitted": round(
                mean([r.get("nll_delta") for r in admitted]), 4),
            "hit_rate_on_admitted": round(
                mean([1 if r.get("final_ok") else 0 for r in admitted]), 4),
            "hit_rate_over_stream": round(
                100.0 * sum(1 for r in admitted if r.get("final_ok")) / n, 2),
        })
    return {
        "available": True,
        "n_scored": len(scored),
        "n_rows": n,
        "p_write_summary": {
            "min": round(min(r["gate_p_write"] for r in scored), 4),
            "max": round(max(r["gate_p_write"] for r in scored), 4),
            "mean": round(mean([r["gate_p_write"] for r in scored]), 4),
        },
        "curve": curve,
        "note": ("admitted_per_100 and nll_delta_mean_on_admitted are measured on the "
                 "same rows, at W=50, from one run: this is the double objective the "
                 "audit asked for, computed offline instead of re-running per threshold."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "rev3_paired_nll.json"))
    a = ap.parse_args()

    all_rows, per_run = [], {}
    for p in a.results:
        d = json.load(open(p, encoding="utf-8"))
        rows = d.get("rows") or []
        tag = os.path.basename(os.path.dirname(p))
        all_rows += rows
        r = {"n": len(rows), "pairing": pairing(rows), "retag": retag(rows),
             "fluency": stream_nll(d["summary"], rows)}
        per_run[tag] = r
        print(f"[{tag}] n={len(rows)} "
              f"routes={dict(Counter(x.get('route') for x in rows))} "
              f"p_write_logged={sum(1 for x in rows if x.get('gate_p_write') is not None)}")

    deltas = [per_run[t]["fluency"]["nll_delta_stream"] for t in per_run
              if per_run[t]["fluency"]["nll_delta_stream"] is not None]
    per_write = [per_run[t]["fluency"]["nll_delta_per_write"] for t in per_run
                 if per_run[t]["fluency"]["nll_delta_per_write"] is not None]
    out = {"per_run": per_run,
           "pooled": {"n": len(all_rows),
                      "pairing": pairing(all_rows),
                      "retag": retag(all_rows),
                      "fluency": {
                          "nll_delta_stream_mean_of_slices": round(mean(deltas), 4),
                          "nll_delta_per_write_mean_of_slices": round(mean(per_write), 4),
                          "slices": len(deltas),
                          "note": "quote beside the pooled hit rate, never alone",
                      }}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print()
    print("route-level pairing (hits with their NLL cost, same row):")
    print(f"{'route':>26} {'n':>5} {'hits':>5} {'hit_rate':>9} {'nll_delta':>10} {'nll|hit':>9}")
    for route, b in out["pooled"]["pairing"].items():
        print(f"{route:>26} {b['n']:>5} {b['hits']:>5} {b['hit_rate']:>9.3f}"
              f" {b['nll_delta_mean']:>10.4f} {b['nll_delta_mean_on_hits']:>9.4f}")
    rt = out["pooled"]["retag"]
    print()
    if rt["available"]:
        print("W=50 double-objective retag (threshold applied offline to gate_p_write):")
        print(f"{'thr':>5} {'admitted/100':>13} {'nll_delta|admitted':>19} {'hit_rate|admitted':>18}")
        for c in rt["curve"]:
            print(f"{c['threshold']:>5.2f} {c['admitted_per_100']:>13.2f}"
                  f" {str(c['nll_delta_mean_on_admitted']):>19} "
                  f"{str(c['hit_rate_on_admitted']):>18}")
    else:
        print(f"retag unavailable on the pooled rows: {rt['why']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
