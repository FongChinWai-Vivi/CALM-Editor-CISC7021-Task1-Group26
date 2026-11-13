# -*- coding: utf-8 -*-
"""R13: does the W=100 window effect survive on edits nobody has ever run?

R12 measured W=100 on offsets 0-700 -- the same eight slices every arm in this
report has used, so those slices are in-sample for the project even though they
are held-out for the model.  R13 therefore re-runs BOTH windows on offsets
1600-1900, which no earlier arm touched, and pools the paired per-slice
differences.  Written to be usable while the last run is still in flight: a
missing slice is dropped from the pairing, never imputed.

  python w6_pool_holdout.py
"""
from __future__ import annotations

import glob
import json
import os
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W6RUN = os.path.join(REPO, "EasyEdit", "w6run")
OFFSETS = [1600, 1700, 1800, 1900]
ARMS = [("W50_b025", 50), ("W100_b025", 100)]


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = d["rows"]
    n = len(rows)
    s = d["summary"]
    para_rows = s.get("paraphrase_rows") or []
    if para_rows:
        para = sum(1 for r in para_rows if r.get("ok")) / len(para_rows)
        pn = len(para_rows)
    else:
        para = s.get("paraphrase_recall")
        pn = s.get("paraphrase_n") or 0
    return {
        "n": n,
        "e2e": sum(1 for r in rows if r["final_ok"]) / n,
        "param": sum(1 for r in rows if r["param_ok"]) / n,
        "mem": sum(1 for r in rows if r["final_ok"] and not r["param_ok"]) / n,
        "admitted": sum(1 for r in rows if r["param_ok"] or r.get("route") == "hypernetwork_batched"),
        "nll": s.get("nll_delta"),
        "para": para,
        "para_n": pn,
        "flushes": len(s.get("flush_hist") or []),
        "seconds": s.get("total_seconds"),
        "settings": {k: s.get(k) for k in ("defer_window", "delta_budget", "layers",
                                           "refine_passes", "memory_read_template",
                                           "memory_read_arbitrate")},
    }


def ci(d, n_boot=5000, seed=42):
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def main():
    table = {}
    for tag, W in ARMS:
        per = {}
        for off in OFFSETS:
            pat = os.path.join(W6RUN, "R13_hold_o%d_%s" % (off, tag), "results_defer_W%d.json" % W)
            hit = glob.glob(pat)
            if hit:
                per[off] = load(hit[0])
        if not per:
            continue
        n = sum(v["n"] for v in per.values())
        adm = sum(v["admitted"] for v in per.values())
        table[tag] = {
            "W": W,
            "slices": sorted(per),
            "per_slice": per,
            "pooled": {
                "n": n,
                "e2e": sum(v["e2e"] * v["n"] for v in per.values()) / n,
                "param": sum(v["param"] * v["n"] for v in per.values()) / n,
                "mem": sum(v["mem"] * v["n"] for v in per.values()) / n,
                "admitted": adm,
                "write_fidelity": (sum(v["param"] * v["n"] for v in per.values()) / adm) if adm else None,
                "nll_delta_mean_of_slices": sum(v["nll"] or 0 for v in per.values()) / len(per),
                "para": (sum(v["para"] * v["para_n"] for v in per.values())
                         / sum(v["para_n"] for v in per.values())) if sum(v["para_n"] for v in per.values()) else None,
                "flushes_mean": sum(v["flushes"] for v in per.values()) / len(per),
                "seconds_mean": sum(v["seconds"] or 0 for v in per.values()) / len(per),
            },
        }

    print("R13 held-out window check (offsets %s)" % ", ".join(str(o) for o in OFFSETS))
    print("%-12s %4s %6s %6s %6s %6s %6s %5s %7s" % ("arm", "n", "e2e", "param", "mem", "nll", "para", "flsh", "sec"))
    for tag, res in table.items():
        p = res["pooled"]
        print("%-12s %4d %6.3f %6.3f %6.3f %6.3f %6s %5.1f %7.0f" % (
            tag, p["n"], p["e2e"], p["param"], p["mem"], p["nll_delta_mean_of_slices"],
            ("%.3f" % p["para"]) if p["para"] is not None else "-",
            p["flushes_mean"], p["seconds_mean"]))

    b = table.get("W50_b025")
    t = table.get("W100_b025")
    if b and t:
        common = sorted(set(b["per_slice"]) & set(t["per_slice"]))
        print("\npaired per-slice deltas (W100 - W50), %d paired slices" % len(common))
        for label, key in (("e2e", "e2e"), ("nll", "nll"), ("para", "para"), ("param", "param"), ("mem", "mem")):
            d = [t["per_slice"][o][key] - b["per_slice"][o][key] for o in common]
            lo, hi = ci(d)
            print("  %-6s mean %+.3f  95%% CI [%+.3f, %+.3f]  per-slice %s" % (
                label, sum(d) / len(d), lo, hi,
                " ".join("%+.2f" % v for v in d)))
            t.setdefault("paired_vs_w50", {})[label] = {
                "mean": sum(d) / len(d), "ci95": [lo, hi],
                "per_slice_offsets": common, "per_slice": d}
        # slice-level difficulty spread: how much does an offset move e2e at all?
        for tag, res in table.items():
            vals = [res["per_slice"][o]["e2e"] for o in res["slices"]]
            print("  slice spread %-10s min %.2f max %.2f range %.2f" % (
                tag, min(vals), max(vals), max(vals) - min(vals)))

    out = os.path.join(REPO, "w6_rev3", "r13_holdout_pooled.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
