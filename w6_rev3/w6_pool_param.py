"""W6 R11 -- pool the three parameter-channel probe arms against the R6 baseline.

Same protocol as w6_pool.py (values recomputed from rows, never trusting the
summary alone; provenance tuple checked across slices; normal + bootstrap +
3-seed subset CIs). Differences: the results file name follows the arm's defer
window, and each arm is pooled separately so the two levers (write budget,
window length) stay attributable.
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(REPO, "EasyEdit", "w6run")
OUT_ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
SUBSET = 0.6
N_BOOT = 2000

ARMS = [
    ("R6_b025_W50", "R6_o*_W50_b025_inject_instruct_para", 50),
    ("R11_b015_W50", "R11_o*_W50_b015_inject", 50),
    ("R11_b035_W50", "R11_o*_W50_b035_inject", 50),
    ("R11_b025_W25", "R11_o*_W25_W25b025_inject", 25),
]


def load(p):
    with open(p, encoding="utf-8") as f:
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
    k = len(vals)
    ms = sorted(mean([vals[rnd.randrange(k)] for _ in range(k)]) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def seed_subset_ci(vals, seeds=SEEDS, frac=SUBSET):
    out = []
    for s in seeds:
        rnd = random.Random(s)
        k = max(2, int(round(frac * len(vals))))
        out.append(mean(rnd.sample(vals, k)))
    return out


def pool_arm(name, prefix, W):
    pattern = os.path.join(RUN_ROOT, prefix, "results_defer_W%d.json" % W)
    files = sorted(glob.glob(pattern))
    if not files:
        print("MISSING", name, pattern)
        return None
    per_slice = []
    p_ok, p_param, p_mem, nlls, degs = [], [], [], [], []
    for f in files:
        d = load(f)
        s, rows = d["summary"], d["rows"]
        tag = os.path.basename(os.path.dirname(f))
        ok = [1 if r["final_ok"] else 0 for r in rows]
        param = [1 if r["param_ok"] else 0 for r in rows]
        mem = [1 if (r["final_ok"] and not r["param_ok"]) else 0 for r in rows]
        nll = s.get("nll_delta")
        deg = s.get("degenerate_rate", 0.0)
        per_slice.append({"slice": tag, "n": len(rows), "e2e": mean(ok),
                          "param": mean(param), "mem": mean(mem),
                          "nll_delta": nll, "degenerate": deg,
                          "ci95": list(ci95(ok))})
        p_ok += ok
        p_param += param
        p_mem += mem
        nlls.append(nll if nll is not None else 0.0)
        degs.append(deg)
    keys = ("defer_window", "refine_passes", "delta_budget", "memory_read",
            "memory_read_template", "memory_read_arbitrate", "edits")
    settings = {os.path.basename(os.path.dirname(f)): {k: load(f)["summary"].get(k) for k in keys}
                for f in files}
    distinct = {json.dumps(v, sort_keys=True) for v in settings.values()}
    res = {
        "arm": name, "prefix": prefix, "W": W, "slices": len(files),
        "poolable": len(distinct) == 1,
        "settings_tuple": json.loads(next(iter(distinct))) if len(distinct) == 1 else None,
        "per_slice": per_slice,
        "pooled": {
            "n": len(p_ok),
            "e2e": mean(p_ok),
            "param_channel": mean(p_param),
            "memory_channel": mean(p_mem),
            "nll_delta_mean_of_slices": mean(nlls),
            "degenerate_rate": mean(degs),
            "ci95_normal": list(ci95(p_ok)),
            "ci95_bootstrap": list(bootstrap_ci([float(x) for x in p_ok])),
            "seed_subset_means": seed_subset_ci([float(x) for x in p_ok]),
        },
    }
    return res


def main():
    out = {}
    hdr = "%-16s %5s %6s %6s %6s %8s %6s %18s %20s" % (
        "arm", "n", "e2e", "param", "mem", "nll", "deg", "E2E 95% CI", "3-seed")
    print(hdr)
    print("-" * len(hdr))
    for name, prefix, W in ARMS:
        res = pool_arm(name, prefix, W)
        if res is None:
            continue
        pl = res["pooled"]
        out[name] = res
        print("%-16s %5d %6.3f %6.3f %6.3f %8.3f %6.2f  [%6.3f,%6.3f] %20s" % (
            name, pl["n"], pl["e2e"], pl["param_channel"], pl["memory_channel"],
            pl["nll_delta_mean_of_slices"], pl["degenerate_rate"],
            pl["ci95_normal"][0], pl["ci95_normal"][1],
            "/".join("%.3f" % x for x in pl["seed_subset_means"])))
        if not res["poolable"]:
            print("  WARNING: not poolable (mixed setting tuples)")
    dst = os.path.join(OUT_ROOT, "r11_param_probe_pooled.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
