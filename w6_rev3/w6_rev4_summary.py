"""W6 Rev4 — one measured summary for the three questions this round asked.

1. e2e / channels / DeltaNLL on the eight frozen slices, re-measured with the
   paraphrase read added to the runner (R6).  Pooled from ``rows`` so a
   summary/rows mismatch cannot hide.
2. paraphrase delivery measured ON THE EDITED MODEL (R6 ``paraphrase_recall``),
   pooled over items, with its own denominator -- this replaces the 0.343 that
   Rev3 borrowed from a clean gpt2-small harness.
3. the clean-LM template sweep: 300 in-slice edits (o0/o300/o700) plus 500
   held-out edits (o800-o1299) that no earlier table touched, and the in-loop
   counter-test (R5, fewshot) that says whether the clean-LM win transfers.

Usage:  python w6_rev3/w6_rev4_summary.py
Writes: w6_rev3/rev4_summary.json and prints the markdown cells used in the
report, so the numbers in the document are copied from a single artefact.
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUN = os.path.join(REPO, "EasyEdit", "w6run")
REC = os.path.join(HERE, "recall4")
N_BOOT = 2000
SEEDS = (0, 1, 2)
SUBSET = 0.6


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def ci95(xs):
    if len(xs) < 2:
        return float("nan"), float("nan")
    m, se = mean(xs), statistics.stdev(xs) / (len(xs) ** 0.5)
    return m - 1.96 * se, m + 1.96 * se


def boot(xs, seed=0):
    rnd = random.Random(seed)
    n = len(xs)
    ms = sorted(mean([xs[rnd.randrange(n)] for _ in range(n)]) for _ in range(N_BOOT))
    return ms[int(0.025 * N_BOOT)], ms[int(0.975 * N_BOOT)]


def seed_ci(xs):
    out = []
    for s in SEEDS:
        rnd = random.Random(s)
        k = max(2, int(round(SUBSET * len(xs))))
        out.append(round(mean(rnd.sample(xs, k)), 3))
    return out


def pool_stream(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[warn] no files for {label}", file=sys.stderr)
        return None
    rows, slices = [], []
    for f in files:
        d = load(f)
        rows.extend(d["rows"])
        slices.append(d["summary"])
    oks = [1 if r["final_ok"] else 0 for r in rows]
    param = [1 if (r["final_ok"] and r.get("answered_by") == "parameter") else 0 for r in rows]
    # the channel split the report quotes is the runner's own definition:
    # param = answered by the parameters, memory = answered and not by params.
    p_ch = [1 if r.get("param_ok") else 0 for r in rows]
    m_ch = [1 if (r["final_ok"] and not r.get("param_ok")) else 0 for r in rows]
    d_nll = [s["nll_delta"] for s in slices]
    para = [(s.get("paraphrase_recall"), s.get("paraphrase_n") or 0) for s in slices]
    para_items = sum(n for _, n in para)
    para_pooled = (sum(r * n for r, n in para) / para_items) if para_items else None
    lo, hi = boot(oks)
    slo, shi = ci95(oks)
    return {
        "label": label,
        "slices": len(files),
        "edits": len(rows),
        "e2e": round(mean(oks), 4),
        "e2e_ci95_normal": [round(slo, 3), round(shi, 3)],
        "e2e_ci95_bootstrap": [round(lo, 3), round(hi, 3)],
        "e2e_3seed": seed_ci(oks),
        "param_channel": round(mean(p_ch), 4),
        "memory_channel": round(mean(m_ch), 4),
        "nll_delta_per_slice_mean": round(mean(d_nll), 4),
        "paraphrase_recall_pooled": (round(para_pooled, 4) if para_pooled is not None else None),
        "paraphrase_items": para_items,
        "degenerate_mean": round(mean([s["degenerate_rate"] for s in slices]), 4),
        "template": slices[0].get("memory_read_template"),
        "serve": slices[0].get("memory_serve"),
        "files": [os.path.basename(os.path.dirname(f)) for f in files],
    }


def pool_recall(files, label):
    agg, tot = {}, 0
    for f in files:
        d = load(os.path.join(REC, f))
        n = d["n_routed_edits"]
        tot += n
        for t, m in d["templates"].items():
            a = agg.setdefault(t, dict(c=0.0, p=0.0, pi=0, ps=0.0, any_=0.0, n=0))
            a["c"] += m["canonical_delivery"] * n
            a["p"] += m["paraphrase_delivery"] * m["paraphrase_instances"]
            a["pi"] += m["paraphrase_instances"]
            a["ps"] += m["paraphrase_delivery_strict"] * m["paraphrase_instances"]
            a["any_"] += m["paraphrase_case_any"] * n
            a["n"] += n
    return {"label": label, "edits": tot,
            "templates": {t: {"canonical": round(a["c"] / a["n"], 3),
                              "paraphrase": round(a["p"] / a["pi"], 3),
                              "paraphrase_strict": round(a["ps"] / a["pi"], 3),
                              "case_any": round(a["any_"] / a["n"], 3)}
                          for t, a in agg.items()}}


def main():
    r6 = pool_stream(os.path.join(RUN, "R6_o*_instruct_para", "results_defer_W50.json"),
                     "R6 inject+instruct, paraphrase in-loop")
    r5 = pool_stream(os.path.join(RUN, "R5_o*_fewshot", "results_defer_W50.json"),
                     "R5 inject+retro+fewshot")
    r3 = pool_stream(os.path.join(RUN, "R3b_o*_b025*", "results_defer_W50.json"),
                     "R3b inject+instruct (frozen Rev3)")
    names = sorted(os.listdir(REC))
    orig = [f for f in names if "gpt2-small" in f and any(f"_o{o}_" in f for o in (0, 300, 700))]
    fresh = [f for f in names if "gpt2-small" in f
             and any(f"_o{o}_" in f for o in (800, 900, 1000, 1100, 1200))]
    xl = [f for f in names if "gpt2-xl" in f]
    out = {
        "streams": {k: v for k, v in
                    [("r6", r6), ("r5_fewshot", r5), ("r3b_frozen", r3)] if v},
        "recall": {},
    }
    if orig:
        out["recall"]["in_slice_300"] = pool_recall(orig, "o0/o300/o700, 300 edits")
    if fresh:
        out["recall"]["held_out_500"] = pool_recall(fresh, "o800-o1299, 500 edits")
    if xl:
        out["recall"]["gpt2_xl_cpu"] = pool_recall(xl, "gpt2-xl, CPU, o0, n=40")
    js = os.path.join(HERE, "rev4_summary.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 96)
    for key, v in out["streams"].items():
        print(f"{key:12s} n={v['edits']:4d} e2e={v['e2e']:.3f} "
              f"[{v['e2e_ci95_bootstrap'][0]:.3f}-{v['e2e_ci95_bootstrap'][1]:.3f}] "
              f"param={v['param_channel']:.3f} mem={v['memory_channel']:.3f} "
              f"nllD={v['nll_delta_per_slice_mean']:.3f} "
              f"para={v['paraphrase_recall_pooled']} (n={v['paraphrase_items']}) "
              f"tmpl={v['template']} serve={v['serve']}")
    print("=" * 96)
    for key, v in out["recall"].items():
        print(f"\n{key}  ({v['label']}, {v['edits']} edits)")
        for t, m in v["templates"].items():
            print(f"  {t:10s} canon={m['canonical']:.3f} para={m['paraphrase']:.3f} "
                  f"strict={m['paraphrase_strict']:.3f} case_any={m['case_any']:.3f}")
    print(f"\nwrote {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
