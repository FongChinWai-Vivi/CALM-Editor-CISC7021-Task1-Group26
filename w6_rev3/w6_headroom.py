"""R16 - where is the remaining head-room?  A decomposition of the frozen end-to-end number.

No model calls: every field needed is already in the run summaries.

For each (offset, slot) we know
  param_ok              did the parameter channel answer correctly
  memory_reason         was a store note served at all ("ok" vs "memory_miss")
  memory_answer         what the memory channel would say
  final_ok, answered_by what the system actually delivered

so three separate ceilings can be computed:

  delivered              what we report today
  arbitration ceiling    served rows only: best of the two channels, i.e. a perfect arbiter
  admission ceiling      served rows OR-ed with what the parameter channel kept, plus the
                         counterfactual that an unserved row would have been served at the
                         store's own read accuracy (reported as a range, not a point)

The margins are recorded per row, so a margin-threshold arbitration is also swept offline.
"""
import json
import os
import glob
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(ROOT, "EasyEdit", "w6run")

ARMS = [
    ("in-sample W50", "R6_o{off}_W50_b025_inject_instruct_para/results_defer_W50.json", range(0, 800, 100)),
    ("held-out W50", "R13_hold_o{off}_W50_b025/results_defer_W50.json", range(1600, 2000, 100)),
]

MEM_READ_ACC = 0.723  # canonical store read accuracy, §10.1 (used only for the counterfactual)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", (s or "").casefold())).strip()


def contains(answer, target):
    if not answer or not target:
        return False
    return norm(target) in norm(answer)


def load(offsets, pattern):
    rows = []
    for off in offsets:
        f = os.path.join(RUN_ROOT, pattern.format(off=off))
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding="utf-8"))
        for r in d.get("rows", []):
            r["_slice"] = off
            # the memory channel counts as able to deliver when either the injected read was
            # correct, or the arbiter emitted the stored note verbatim (answered_by == memory_arb)
            r["_mem_ok"] = (contains(r.get("memory_answer"), r.get("target_new"))
                            or (r.get("final_ok") and r.get("answered_by") in ("memory", "memory_arb")))
            r["_served"] = r.get("memory_reason") == "ok"
            rows.append(r)
    return rows


def decompose(rows):
    n = len(rows)
    if not n:
        return None
    served = [r for r in rows if r["_served"]]
    unserved = [r for r in rows if not r["_served"]]
    delivered = sum(bool(r.get("final_ok")) for r in rows) / n
    param = sum(bool(r.get("param_ok")) for r in rows) / n
    mem_served = len(served) / n
    # perfect arbiter on served rows only, unserved rows keep their delivered outcome
    arb = (sum(bool(r.get("param_ok")) or bool(r["_mem_ok"]) for r in served)
           + sum(bool(r.get("final_ok")) for r in unserved)) / n
    # store accuracy on the rows it actually served
    mem_acc_when_served = (sum(bool(r["_mem_ok"]) for r in served) / len(served)) if served else None
    # counterfactual: had the unserved rows been routed to the store, they would be served at
    # the same accuracy the store achieves on rows it does serve (range: that accuracy or 0)
    cf_lo = (sum(bool(r.get("param_ok")) for r in unserved)
             + sum(bool(r.get("param_ok")) or bool(r["_mem_ok"]) for r in served)) / n
    cf_hi = (sum(bool(r.get("param_ok")) or bool(r["_mem_ok"]) for r in unserved)
             + sum(bool(r.get("param_ok")) or bool(r["_mem_ok"]) for r in served)) / n
    cf_exp = (sum(bool(r.get("param_ok")) or (MEM_READ_ACC > 0.5) for r in unserved)
              + sum(bool(r.get("param_ok")) or bool(r["_mem_ok"]) for r in served)) / n
    both_wrong = sum((not r.get("param_ok")) and (not r["_mem_ok"]) for r in rows) / n
    return {
        "n": n,
        "delivered": round(delivered, 4),
        "param_channel": round(param, 4),
        "served_share": round(mem_served, 4),
        "mem_acc_when_served": None if mem_acc_when_served is None else round(mem_acc_when_served, 4),
        "arbitration_ceiling": round(arb, 4),
        "admission_ceiling_lo": round(cf_lo, 4),
        "admission_ceiling_hi": round(cf_hi, 4),
        "admission_ceiling_expected": round(cf_exp, 4),
        "both_channels_wrong": round(both_wrong, 4),
        "arb_gain": round(arb - delivered, 4),
        "admission_gain_expected": round(cf_exp - arb, 4),
    }


def margin_sweep(rows):
    """Offline arbitration sweep: on served rows, keep the param answer unless the memory margin
    exceeds tau.  Only rows that carry a margin can be re-decided; report coverage."""
    with_margin = [r for r in rows if r.get("mem_arb_margin_ctx") is not None]
    if not with_margin:
        return None
    out = []
    for tau in (-10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8):
        ok = 0
        for r in rows:
            m = r.get("mem_arb_margin_ctx")
            if m is None:
                ok += bool(r.get("final_ok"))
            elif m > tau:
                ok += bool(r["_mem_ok"])
            else:
                ok += bool(r.get("param_ok"))
        out.append({"tau": tau, "e2e": round(ok / len(rows), 4)})
    return {"rows_with_margin": len(with_margin), "coverage": round(len(with_margin) / len(rows), 4),
            "sweep": out, "delivered": round(sum(bool(r.get("final_ok")) for r in rows) / len(rows), 4)}


def main():
    report = {}
    pooled = []
    for label, pattern, offsets in ARMS:
        rows = load(offsets, pattern)
        if not rows:
            print(f"[skip] {label}")
            continue
        dec = decompose(rows)
        sweep = margin_sweep(rows)
        report[label] = {"decomposition": dec, "margin_sweep": sweep,
                         "per_slice": {str(o): decompose([r for r in rows if r["_slice"] == o])
                                       for o in offsets}}
        pooled.extend(rows)
        print(f"\n== {label} (n={dec['n']})")
        for k, v in dec.items():
            if k != "n":
                print(f"   {k:26s} {v}")
        if sweep:
            print(f"   margin rows {sweep['rows_with_margin']} ({sweep['coverage']:.2f}), delivered "
                  f"{sweep['delivered']}; best sweep point "
                  f"{max(sweep['sweep'], key=lambda x: x['e2e'])}")

    if pooled:
        dec = decompose(pooled)
        report["pooled_12_slices"] = {"decomposition": dec, "margin_sweep": margin_sweep(pooled)}
        print(f"\n== pooled (n={dec['n']})")
        for k, v in dec.items():
            if k != "n":
                print(f"   {k:26s} {v}")
        sw = report["pooled_12_slices"]["margin_sweep"]
        if sw:
            print("   sweep:", sw["sweep"])

    out = os.path.join(ROOT, "w6_rev3", "r16_headroom.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
