"""W6 Rev3 — pooling + provenance + 3-seed CI over the measured end-to-end sweep.

Reads every ``results_defer_W50.json`` produced by the Day-2 sweep, verifies that
the runs are poolable (same W, same recorder, same reader template), and prints
one table with:

  * per-slice hit_total_measured (memory in-loop) and its two channel split:
    hit_total_param_channel and hit_total_memory_channel
  * the fluency coordinate that must travel with every hit number: nll_delta
  * a 3-seed random-subset CI (Vivi ②) and a bootstrap CI over the pooled rows

Nothing is estimated from the summaries alone: the pooled number is recomputed
from ``rows`` so a summary/rows disagreement would be visible.
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics
import sys

RUN_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "EasyEdit", "w6run")
# default sweep = the first Rev3 pass at delta_budget=0.50; pass a prefix on the
# CLI to pool a different operating point (e.g. R3b_o*_W50_b025).
RUN_GLOB = os.path.join(RUN_ROOT, "R3_o*_W50_memread", "results_defer_W50.json")
SEEDS = (0, 1, 2)
SUBSET = 0.6          # fraction drawn per seed, same protocol as w6_ci_rows.py
N_BOOT = 2000


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def ci95(xs):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    m, se = mean(xs), statistics.stdev(xs) / (len(xs) ** 0.5)
    return (m - 1.96 * se, m + 1.96 * se)


def bootstrap_ci(vals, n=N_BOOT, seed=0):
    rnd = random.Random(seed)
    n_obs = len(vals)
    ms = [mean([vals[rnd.randrange(n_obs)] for _ in range(n_obs)]) for _ in range(n)]
    ms.sort()
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def seed_subset_ci(vals, seeds=SEEDS, frac=SUBSET):
    rnd_means = []
    for s in seeds:
        rnd = random.Random(s)
        k = max(2, int(round(frac * len(vals))))
        rnd_means.append(mean(rnd.sample(vals, k)))
    return rnd_means


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else RUN_GLOB
    tag_out = sys.argv[2] if len(sys.argv) > 2 else "rev3_pooled.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print("no sweep results found", file=sys.stderr)
        return 2

    print("=" * 100)
    print("W6 Rev3 — measured end-to-end, memory in-loop, 8 held-out slices (W=50, 100 edits each)")
    print("=" * 100)
    hdr = (f"{'slice':>8} {'n':>4} {'e2e':>6} {'param':>6} {'mem':>6} {'nll_delta':>10}"
           f" {'deg':>5} {'E2E 95% CI':>16} {'3-seed':>22}")
    print(hdr)
    print("-" * len(hdr))

    pooled_ok, pooled_param, pooled_mem, pooled_nll, pooled_deg = [], [], [], [], []
    per_slice = []
    for f in files:
        d = load(f)
        s, rows = d["summary"], d["rows"]
        tag = os.path.basename(os.path.dirname(f)).replace("_W50_memread", "")
        ok = [1 if r["final_ok"] else 0 for r in rows]
        param = [1 if r["param_ok"] else 0 for r in rows]
        mem = [1 if (r["final_ok"] and not r["param_ok"]) else 0 for r in rows]
        nll = s.get("nll_delta")
        deg = s.get("degenerate_rate", 0.0)
        lo, hi = ci95(ok)
        seeds = seed_subset_ci(ok)
        print(f"{tag:>8} {len(rows):>4} {mean(ok):>6.3f} {mean(param):>6.3f} {mean(mem):>6.3f}"
              f" {nll:>10.3f} {deg:>5.2f} [{lo:>6.3f},{hi:>6.3f}] "
              f"{'/'.join(f'{x:.3f}' for x in seeds):>22}")
        pooled_ok += ok
        pooled_param += param
        pooled_mem += mem
        pooled_nll.append(nll)
        pooled_deg.append(deg)
        per_slice.append({"slice": tag, "n": len(rows), "e2e": mean(ok),
                          "param": mean(param), "mem": mean(mem),
                          "nll_delta": nll, "degenerate": deg,
                          "ci95": list(ci95(ok)), "seed_subset": seeds})

    print("-" * len(hdr))
    lo, hi = ci95(pooled_ok)
    blo, bhi = bootstrap_ci([float(x) for x in pooled_ok])
    seeds = seed_subset_ci([float(x) for x in pooled_ok])
    print(f"{'POOLED':>8} {len(pooled_ok):>4} {mean(pooled_ok):>6.3f} {mean(pooled_param):>6.3f}"
          f" {mean(pooled_mem):>6.3f} {mean(pooled_nll):>10.3f} {mean(pooled_deg):>5.2f}"
          f" [{lo:>6.3f},{hi:>6.3f}] {'/'.join(f'{x:.3f}' for x in seeds):>22}")
    print(f"         bootstrap CI95 over pooled rows: [{blo:.3f},{bhi:.3f}]  "
          f"3-seed subset means (frac={SUBSET}): {[round(x,3) for x in seeds]}")

    # provenance check: every slice must agree on the recorder settings
    keys = ("defer_window", "refine_passes", "delta_budget", "memory_read",
            "memory_read_template", "memory_read_arbitrate", "edits")
    settings = {}
    for f in files:
        s = load(f)["summary"]
        settings[os.path.basename(os.path.dirname(f))] = {k: s.get(k) for k in keys}
    distinct = {json.dumps(v, sort_keys=True) for v in settings.values()}
    print()
    print(f"provenance: {len(distinct)} distinct setting tuple(s) across {len(files)} slices "
          f"-> {'POOLABLE' if len(distinct) == 1 else 'NOT POOLABLE'}")
    if len(distinct) == 1:
        print("  " + json.dumps(next(iter(distinct)), sort_keys=True))

    out = {
        "protocol": {"pattern": pattern, "W": 50, "edits_per_slice": 100, "slices": len(files),
                     "reader": "memory_read=True, template=instruct, arbitrate=True",
                     "poolable": len(distinct) == 1},
        "settings": settings,
        "per_slice": per_slice,
        "pooled": {
            "n": len(pooled_ok),
            "e2e": mean(pooled_ok),
            "param_channel": mean(pooled_param),
            "memory_channel": mean(pooled_mem),
            "nll_delta_mean_of_slices": mean(pooled_nll),
            "degenerate_rate": mean(pooled_deg),
            "ci95_normal": list(ci95(pooled_ok)),
            "ci95_bootstrap": list(bootstrap_ci([float(x) for x in pooled_ok])),
            "seed_subset_means": seeds,
        },
    }
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), tag_out)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
