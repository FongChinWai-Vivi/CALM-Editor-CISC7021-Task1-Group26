"""Rev3 / Vivi (2): 3-seed random-subset CI for a run's answer-path hit_total.

Reads the per-edit rows dumped by ``examples/w6_defer_stream.py`` and reports,
for the run's measured end-to-end hit_total:

  * ``full``            : all rows, plus a 2000-resample percentile bootstrap CI
  * ``subset_seed0/1/2``: three independent random subsets (default 60% of rows)
                          each with its own bootstrap CI

The point is honesty about n=100: a single hit_total without a spread invites
being quoted as if it were a population value.  Every number here is derived
from rows already on disk - no model is reloaded.

Usage:
  python w6_rev3/w6_ci_rows.py --run EasyEdit/w6run/K2_W50_refine1_guarded_heldout
  python w6_rev3/w6_ci_rows.py --run <dir> --frac 0.6 --seeds 0,1,2 --out x.json
"""
import argparse
import glob
import json
import os
import random
from typing import Any, Dict, List, Optional


def _load_rows(run_dir: str) -> Dict[str, Any]:
    cands = sorted(glob.glob(os.path.join(run_dir, "results_defer_W*.json")))
    if not cands:
        raise SystemExit(f"no results_defer_W*.json under {run_dir}")
    data = json.load(open(cands[0], encoding="utf-8"))
    if "rows" in data and "summary" in data:
        payload, tag = data, f"W{data['summary'].get('defer_window')}"
    else:
        tag = next(iter(data))
        payload = data[tag]
    return {"tag": tag, "path": cands[0], "payload": payload}


def _boot_ci(vals: List[float], resamples: int, rng: random.Random,
             level: float = 0.95) -> Optional[List[float]]:
    if not vals:
        return None
    n = len(vals)
    means = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((1 - level) / 2 * len(means))]
    hi = means[min(len(means) - 1, int((1 + level) / 2 * len(means)))]
    return [round(lo, 4), round(hi, 4)]


def _summ(vals: List[float], resamples: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 4) if vals else None,
        "ci95": _boot_ci(vals, resamples, rng),
        "se": round((sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
                     / max(1, len(vals) - 1)) ** 0.5 / max(1, len(vals)) ** 0.5, 4)
        if len(vals) > 1 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory holding results_defer_W*.json")
    ap.add_argument("--frac", type=float, default=0.6, help="subset fraction")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loaded = _load_rows(args.run)
    payload = loaded["payload"]
    payload = {**payload.get("summary", {}), **{k: v for k, v in payload.items() if k != "summary"}}
    rows = payload.get("rows") or []
    if not rows:
        raise SystemExit(f"{loaded['path']} has no rows")

    # answer-path verdict; pre-Rev3 runs only have final_ok (parameter channel)
    def _ok(r: Dict[str, Any]) -> float:
        v = r.get("final_ok")
        return float(bool(v))

    allv = [_ok(r) for r in rows]
    out: Dict[str, Any] = {
        "run_dir": os.path.abspath(args.run),
        "rows_file": os.path.abspath(loaded["path"]),
        "defer_window": payload.get("defer_window"),
        "memory_read": payload.get("memory_read", False),
        "edits": payload.get("edits", len(rows)),
        "hit_total_measured": payload.get("hit_total_measured",
                                         round(sum(allv) / len(allv), 4)),
        "hit_total_param_channel": payload.get("hit_total_param_channel",
                                               round(sum(allv) / len(allv), 4)),
        "full": _summ(allv, args.resamples, 12345),
        "subsets": {},
        "note": ("3 independent random subsets of the same rows; each with a "
                 "2000-resample percentile bootstrap CI. Subset dispersion is "
                 "the honest n=100 spread and must be quoted with the point "
                 "estimate."),
    }
    k = max(1, int(round(args.frac * len(rows))))
    idx_all = list(range(len(rows)))
    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = random.Random(seed)
        idx = sorted(rng.sample(idx_all, k))
        vals = [allv[i] for i in idx]
        rec = _summ(vals, args.resamples, seed)
        rec["slots"] = [rows[i]["slot"] for i in idx]
        out["subsets"][f"seed{seed}"] = rec

    dst = args.out or os.path.join(args.run, "ci_3seed.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps({kk: out[kk] for kk in
                      ("run_dir", "edits", "hit_total_measured",
                       "hit_total_param_channel", "full")},
                     ensure_ascii=False, indent=1))
    for name, rec in out["subsets"].items():
        print(f"  {name}: n={rec['n']} mean={rec['mean']} ci95={rec['ci95']}")
    print(f"  -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
