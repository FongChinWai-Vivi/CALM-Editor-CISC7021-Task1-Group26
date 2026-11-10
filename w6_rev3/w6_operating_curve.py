"""Tang ③ — the measured W=50 operating curve.

The first attempt swept the gate's ``write_threshold`` (0.15 / 0.30 / 0.50) and the
three runs came back bit-identical, i.e. that knob does not move admission in this
configuration.  The admission decision here is governed by ``delta_budget``, so the
double objective is swept over that instead: how many edits are admitted per 100,
what the in-gate hit rate is on exactly those admitted edits, and what the stream
costs in probe NLL.

Both coordinates come from the same run, so no hit number below can be quoted
without the NLL beside it and vice versa.
"""
from __future__ import annotations

import glob
import json
import os

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "EasyEdit", "w6run")
PREFIX = "R3_o0_W50_"


def fnum(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def collect():
    pts = []
    for d in sorted(glob.glob(os.path.join(BASE, PREFIX + "*"))):
        f = os.path.join(d, "results_defer_W50.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f, encoding="utf-8"))["summary"]
        if s.get("hit_total_measured") is None:
            continue
        pts.append({
            "run": os.path.basename(d).replace(PREFIX, ""),
            "gate_write_threshold": s.get("gate_write_threshold"),
            "delta_budget": s.get("delta_budget"),
            "admitted_per_100": (None if s.get("param_coverage") is None
                                 else round(100 * s["param_coverage"], 1)),
            "hit_rate_on_admitted": s.get("final_state_hit"),
            "e2e_measured": s.get("hit_total_measured"),
            "hit_param_channel": s.get("hit_total_param_channel"),
            "hit_memory_channel": s.get("hit_total_memory_channel"),
            "nll_delta_per_100": s.get("nll_delta"),
            "nll_delta_per_write": (None if not s.get("param_coverage")
                                    or s.get("nll_delta") is None
                                    else round(s["nll_delta"] / (100 * s["param_coverage"]), 4)),
            "degenerate_rate": s.get("degenerate_rate"),
        })
    return pts


def main():
    pts = collect()
    print("=" * 118)
    print("W=50 operating points (slice o0, 100 edits, memory in-loop) — admission vs NLL, same rows")
    print("=" * 118)
    hdr = (f"{'run':>10} {'thr':>5} {'budget':>7} {'admit/100':>10} {'hit|admit':>10} "
           f"{'e2e':>6} {'param':>6} {'mem':>6} {'NLL/100':>8} {'NLL/write':>10} {'deg':>5}")
    print(hdr)
    print("-" * len(hdr))
    for p in pts:
        print(f"{p['run']:>10} {fnum(p['gate_write_threshold'],2):>5} "
              f"{fnum(p['delta_budget'],2):>7} {fnum(p['admitted_per_100'],1):>10} "
              f"{fnum(p['hit_rate_on_admitted'],3):>10} {fnum(p['e2e_measured'],3):>6} "
              f"{fnum(p['hit_param_channel'],3):>6} {fnum(p['hit_memory_channel'],3):>6} "
              f"{fnum(p['nll_delta_per_100'],3):>8} {fnum(p['nll_delta_per_write'],4):>10} "
              f"{fnum(p['degenerate_rate'],2):>5}")

    thr = [p for p in pts if p["run"].startswith("retag_")]
    same = len({(p["e2e_measured"], p["admitted_per_100"], p["nll_delta_per_100"]) for p in thr}) == 1
    print()
    if len(thr) >= 2:
        print(f"gate write_threshold sweep: {len(thr)} points, "
              f"{'IDENTICAL — knob is inert at W=50' if same else 'moves the outcome'}"
              f" (thresholds {[p['gate_write_threshold'] for p in thr]})")

    out = {"points": pts,
           "gate_threshold_inert": same,
           "note": ("admit/100 and hit|admit are on the same admitted set; NLL/100 is the "
                    "stream cost at that operating point; NLL/write normalises it.")}
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rev3_operating_curve.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
