# eval_judges.py -- CALM-Editor evaluation contract v1 (auto-generated).
#
# Usage in any eval script:
#     from eval_judges import judge_L1, judge_L2, judge_L3, bootstrap_ci, assert_contract, summarise
import re, math, random
import numpy as np

_NONWORD, _WS = re.compile(r"[^\w\s]"), re.compile(r"\s+")

def normalise(text):
    s = _NONWORD.sub(" ", str(text or "").lower().strip())
    return _WS.sub(" ", s).strip()

def first_sentences(text, k=1):
    parts = [p for p in re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip()) if p.strip()]
    return " ".join(parts[:k])

def judge_L1(gen, target):
    g, t = normalise(gen), normalise(target)
    return bool(t) and t in g

def judge_L2(gen, target):
    return judge_L1(first_sentences(gen, 1), target)

def judge_L3(gen, target):
    g, t = normalise(first_sentences(gen, 1)), normalise(target)
    return bool(t) and g == t

def bootstrap_ci(bits, B=4000, seed=42):
    n = len(bits)
    if n == 0:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    d = np.array([sum(bits[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)])
    return (sum(bits) / n, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)))

def summarise(rows, rot=None):
    # full contract-compliant metrics dict, built from per-sample rows
    L1 = [judge_L1(r["prediction"], r["expected"]) for r in rows]
    L2 = [judge_L2(r["prediction"], r["expected"]) for r in rows]
    L3 = [judge_L3(r["prediction"], r["expected"]) for r in rows]
    out = {"n": len(rows)}
    for name, bits in (("L1_contain", L1), ("L2_contain_1sent", L2), ("L3_slot_exact", L3)):
        p, lo, hi = bootstrap_ci(bits)
        out[name] = round(p, 4)
        out[name + "_ci"] = [round(lo, 4), round(hi, 4)]
    if rot is None:
        rot = [r["expected"] for r in rows[1:]] + [r["expected"] for r in rows[:1]]
    floor = [judge_L2(r["prediction"], t) for r, t in zip(rows, rot)]
    fp, flo, fhi = bootstrap_ci(floor)
    out["negative_control_floor"] = {"L2_floor": round(fp, 4),
                                     "ci": [round(flo, 4), round(fhi, 4)],
                                     "L2_margin": round(out["L2_contain_1sent"] - fp, 4)}
    strata = {}
    for tag in sorted({r["conflict_tag"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["conflict_tag"] == tag]
        strata[tag] = {"n": len(sel),
                       "L2_contain_1sent": round(sum(L2[i] for i in sel) / len(sel), 4),
                       "L3_slot_exact": round(sum(L3[i] for i in sel) / len(sel), 4)}
    out["conflict_tag_strata"] = strata
    return out
