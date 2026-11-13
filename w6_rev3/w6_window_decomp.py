"""W6 R12 -- decompose the window effect instead of quoting one e2e number.

The W=100 arm moves three things at once, and a report that quotes only e2e would
be hiding two of them:

  coverage        route == "hypernetwork_batched"  (how many edits the gate admits)
  write fidelity  param hits / admitted writes     (how well one write-out lands)
  memory channel  final_ok & not param_ok          (how much the store has to cover)
  NLL             per-slice stream delta, plus the guard allowance it implies

It also reports the admitted-set overlap with the R6 baseline per slice, because
the interesting question is not only "how many" writes were admitted but
"were they the same rows".

Usage: python w6_window_decomp.py [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(REPO, "EasyEdit", "w6run")

ARMS = [
    ("R6_b025_W50", "R6_o*_W50_b025_inject_instruct_para", 50),
    ("R11_b025_W25", "R11_o*_W25_W25b025_inject", 25),
    ("R12_b025_W100", "R12_o*_W100_W100b025_inject", 100),
]


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def slice_stats(path):
    d = load(path)
    rows, s = d["rows"], d["summary"]
    n = len(rows)
    admitted = [i for i, r in enumerate(rows) if r.get("route") == "hypernetwork_batched"]
    hits = sum(1 for r in rows if r.get("param_ok"))
    mem = sum(1 for r in rows if r.get("final_ok") and not r.get("param_ok"))
    e2e = sum(1 for r in rows if r.get("final_ok"))
    flushes = s.get("flush_hist") or []
    return {
        "slice": os.path.basename(os.path.dirname(path)),
        "n": n,
        "admitted": len(admitted),
        "coverage": len(admitted) / n,
        "write_fidelity": (hits / len(admitted)) if admitted else None,
        "param": hits / n,
        "memory": mem / n,
        "e2e": e2e / n,
        "nll_delta": s.get("nll_delta"),
        "window": s.get("defer_window"),
        "budget": s.get("delta_budget"),
        "flush_count": len(flushes),
        "flush_seconds": round(sum(f.get("seconds", 0.0) for f in flushes), 1),
        "total_seconds": s.get("total_seconds"),
        "admitted_slots": admitted,
        "para_recall": s.get("paraphrase_recall"),
        "para_n": s.get("paraphrase_n"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(REPO, "w6_rev3", "r12_window_decomp.json"))
    args = ap.parse_args()

    out = {}
    for name, prefix, W in ARMS:
        files = sorted(glob.glob(os.path.join(RUN_ROOT, prefix, "results_defer_W%d.json" % W)))
        if not files:
            print("MISSING", name)
            continue
        per = [slice_stats(f) for f in files]
        adm = sum(p["admitted"] for p in per)
        par = sum(round(p["param"] * p["n"]) for p in per)
        pooled = {
            "slices": len(per),
            "n": sum(p["n"] for p in per),
            "admitted": adm,
            "coverage": statistics.mean(p["coverage"] for p in per),
            "write_fidelity": (par / adm) if adm else None,
            "param": statistics.mean(p["param"] for p in per),
            "memory": statistics.mean(p["memory"] for p in per),
            "e2e": statistics.mean(p["e2e"] for p in per),
            "nll_delta": statistics.mean(p["nll_delta"] for p in per if p["nll_delta"] is not None),
            "flush_count": statistics.mean(p["flush_count"] for p in per),
            "flush_seconds": statistics.mean(p["flush_seconds"] for p in per),
            "nll_per_flush": statistics.mean(
                (p["nll_delta"] or 0.0) / max(1, p["flush_count"]) for p in per),
        }
        out[name] = {"per_slice": per, "pooled": pooled}

    # admitted-set overlap against the frozen baseline, per slice
    base = "R6_b025_W50"
    if base in out:
        for name in list(out):
            if name == base:
                continue
            ov = []
            for a, b in zip(out[name]["per_slice"], out[base]["per_slice"]):
                if a["slice"].split("_")[1] != b["slice"].split("_")[1]:
                    continue
                A, B = set(a["admitted_slots"]), set(b["admitted_slots"])
                ov.append({"slice": a["slice"], "n_arm": len(A), "n_base": len(B),
                           "jaccard": (len(A & B) / len(A | B)) if (A | B) else 1.0})
            out[name]["overlap_vs_base"] = ov

    hdr = "%-15s %5s %7s %7s %6s %6s %6s %7s %5s %8s" % (
        "arm", "n", "cover", "writeF", "param", "mem", "e2e", "nll", "flush", "nll/fl")
    print(hdr)
    print("-" * len(hdr))

    def f3(v):
        return "  n/a" if v is None else "%5.3f" % v

    for name, v in out.items():
        p = v["pooled"]
        print("%-15s %5d %7.3f %7s %6.3f %6.3f %6.3f %7.3f %5.1f %8.3f" % (
            name, p["n"], p["coverage"], f3(p["write_fidelity"]), p["param"], p["memory"],
            p["e2e"], p["nll_delta"], p["flush_count"], p["nll_per_flush"]))
    for name, v in out.items():
        if "overlap_vs_base" in v:
            js = [o["jaccard"] for o in v["overlap_vs_base"]]
            print("admit-set Jaccard vs %s for %s: %.3f (per slice %s)" % (
                base, name, statistics.mean(js) if js else float("nan"),
                ", ".join("%.2f" % j for j in js)))

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", args.json)


if __name__ == "__main__":
    main()
