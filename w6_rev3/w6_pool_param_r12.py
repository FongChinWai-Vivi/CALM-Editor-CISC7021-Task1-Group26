"""W6 R12 -- pool the W=100 arm against the W=50 operating point.

Reuses the R11 pooling machinery (values recomputed from rows, provenance tuple
checked, normal + bootstrap + 3-seed CIs) so the two tables are directly
comparable.  Two differences:

  1. the arm list is the R6 baseline, the R11 budget-0.25 / W=50 arm, and the
     new W=100 arm -- the single lever between the last two is flush count;
  2. in-loop paraphrase recall is pooled as well, because the W=100 runs carry
     `paraphrase_rows` (absent from the R6/R11 summaries, which is why their
     para column shows as "-" with its own n from the earlier note).
"""
from __future__ import annotations

import json
import os
import statistics

import w6_pool_param as base

ARMS = [
    ("R6_b025_W50", "R6_o*_W50_b025_inject_instruct_para", 50),
    ("R12_b025_W100", "R12_o*_W100_W100b025_inject", 100),
]


def pool_with_para(name, prefix, W):
    res = base.pool_arm(name, prefix, W)
    if res is None:
        return None
    import glob
    pattern = os.path.join(base.RUN_ROOT, prefix, "results_defer_W%d.json" % W)
    files = sorted(glob.glob(pattern))
    n_para, n_para_hit, flush_counts = 0, 0, []
    per_slice_para = []
    for f in files:
        s = base.load(f)["summary"]
        rows = s.get("paraphrase_rows") or []
        if not rows:
            # R6/R11 predate the per-row field; fall back to their own aggregate
            # so the column is comparable instead of silently empty.
            fn = s.get("paraphrase_n") or 0
            fh = round((s.get("paraphrase_recall") or 0.0) * fn)
            n_para += fn
            n_para_hit += fh
            per_slice_para.append({"slice": os.path.basename(os.path.dirname(f)),
                                   "n": fn,
                                   "para": s.get("paraphrase_recall")})
            flush_counts.append(len(s.get("flush_hist") or []))
            continue
        hit = sum(1 for r in rows if r.get("ok"))
        n_para += len(rows)
        n_para_hit += hit
        flush_counts.append(len(s.get("flush_hist") or []))
        per_slice_para.append({"slice": os.path.basename(os.path.dirname(f)),
                               "n": len(rows),
                               "para": (hit / len(rows)) if rows else None})
    res["paraphrase"] = {
        "n": n_para,
        "recall": (n_para_hit / n_para) if n_para else None,
        "per_slice": per_slice_para,
    }
    res["flush_counts"] = flush_counts
    res["flush_count_mean"] = statistics.mean(flush_counts) if flush_counts else None
    return res


def main():
    out = {}
    hdr = "%-16s %5s %6s %6s %6s %8s %6s %-14s %5s %6s" % (
        "arm", "n", "e2e", "param", "mem", "nll", "deg", "E2E 95% CI", "flsh", "para")
    print(hdr)
    print("-" * len(hdr))
    for name, prefix, W in ARMS:
        res = pool_with_para(name, prefix, W)
        if res is None:
            continue
        pl = res["pooled"]
        para = res["paraphrase"]
        out[name] = res
        print("%-16s %5d %6.3f %6.3f %6.3f %8.3f %6.2f [%5.3f,%5.3f] %5.1f %6s" % (
            name, pl["n"], pl["e2e"], pl["param_channel"], pl["memory_channel"],
            pl["nll_delta_mean_of_slices"], pl["degenerate_rate"],
            pl["ci95_normal"][0], pl["ci95_normal"][1],
            res["flush_count_mean"] or -1,
            ("%.3f" % para["recall"]) if para["recall"] is not None else "-"))
        if not res["poolable"]:
            print("  WARNING: not poolable (mixed setting tuples)")

    # Paired slice-level difference against the frozen baseline.  The arms see
    # the same 100 edits per offset, so the slices pair exactly; resampling the
    # eight paired differences is the honest CI for "the window changed e2e".
    if "R6_b025_W50" in out:
        import random
        rng = random.Random(42)
        b = out["R6_b025_W50"]["per_slice"]
        print("\npaired per-slice deltas vs R6_b025_W50 (95% CI, 5000 slice resamples)")
        for name, res in out.items():
            if name == "R6_b025_W50":
                continue
            a = res["per_slice"]
            if len(a) != len(b):
                continue
            d_e2e = [x["e2e"] - y["e2e"] for x, y in zip(a, b)]
            d_nll = [x["nll_delta"] - y["nll_delta"] for x, y in zip(a, b)]
            for label, d in (("e2e", d_e2e), ("nll", d_nll)):
                boots = sorted(
                    sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
                lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
                print("  %-16s %-4s mean %+.3f  95%% CI [%+.3f, %+.3f]  win/lose %d/%d" % (
                    name, label, sum(d) / len(d), lo, hi,
                    sum(1 for v in d if v > 0), sum(1 for v in d if v < 0)))
                res.setdefault("paired_vs_r6", {})[label] = {
                    "mean": sum(d) / len(d), "ci95": [lo, hi], "per_slice": d}
    dst = os.path.join(base.OUT_ROOT, "r12_window_probe_pooled.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
