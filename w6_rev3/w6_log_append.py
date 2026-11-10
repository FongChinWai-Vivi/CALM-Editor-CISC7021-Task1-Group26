"""Append the Rev3 sweep to the append-only quality log.

Every line records the artefact it came from, that artefact's SHA1 and the full
setting tuple, so a later reader can tell which numbers were re-run and which
were merely restated.  The pooled lines are appended separately and point at the
per-slice lines by run path.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, "EasyEdit", "w6run", "quality_log.jsonl")
REV3 = os.path.join(REPO, "w6_rev3")

SETS = [
    ("b050", os.path.join(REPO, "EasyEdit", "w6run", "R3_o*_W50_memread",
                          "results_defer_W50.json"), "rev3_pooled_b050.json"),
    ("b025", os.path.join(REPO, "EasyEdit", "w6run", "R3b_o*_W50_b025",
                          "results_defer_W50.json"), "rev3_pooled_b025.json"),
]


def sha1(path, n=12):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def entry_for(path, tag):
    d = json.load(open(path, encoding="utf-8"))
    s = d["summary"]
    rel = os.path.relpath(path, os.path.join(REPO, "EasyEdit")).replace("\\", "/")
    off = os.path.basename(os.path.dirname(path)).split("_")[1]
    return {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checker": "Vivi",
        "note": (f"Rev3 measured end-to-end, memory in-loop (delta_budget="
                 f"{s.get('delta_budget')}); supersedes any ceiling-only statement "
                 f"for this slice"),
        "set": tag,
        "run": "w6run/" + rel.replace("w6run/", ""),
        "sha1": sha1(path),
        "offset": int(off.lstrip("o")),
        "n": s.get("edits"),
        "W": s.get("defer_window"),
        "layers": json.dumps(s.get("layers")),
        "refine": s.get("refine_passes"),
        "delta_budget": s.get("delta_budget"),
        "memory_read": s.get("memory_read"),
        "memory_read_template": s.get("memory_read_template"),
        "gate_write_threshold": s.get("gate_write_threshold"),
        "coverage": s.get("param_coverage"),
        "immediate": s.get("immediate_hit"),
        "final_in_gate": s.get("final_state_hit"),
        "hit_total_param_channel": s.get("hit_total_param_channel"),
        "hit_total_memory_channel": s.get("hit_total_memory_channel"),
        "hit_total": s.get("hit_total_measured"),
        "e2e_measured": s.get("hit_total_measured"),
        "e2e_ceiling_note": "not used in Rev3; measured value stands alone",
        "nll_delta": s.get("nll_delta"),
        "degen": s.get("degenerate_rate"),
        "param_n": round(100 * (s.get("param_coverage") or 0), 1),
        "answered_by": s.get("answered_by"),
    }


def main():
    lines = []
    for tag, pattern, pooled in SETS:
        files = sorted(glob.glob(pattern))
        for f in files:
            lines.append(entry_for(f, tag))
        pp = os.path.join(REV3, pooled)
        if os.path.exists(pp):
            p = json.load(open(pp, encoding="utf-8"))
            pooled_row = p["pooled"]
            lines.append({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "checker": "Vivi",
                "note": "Rev3 pooled over 8 held-out slices (800 edits) — the citable row",
                "set": tag,
                "run": f"w6_rev3/{pooled}",
                "sha1": sha1(pp),
                "slices": p["protocol"]["slices"],
                "n": pooled_row["n"],
                "W": 50,
                "delta_budget": (0.25 if tag == "b025" else 0.5),
                "hit_total": round(pooled_row["e2e"], 4),
                "e2e_measured": round(pooled_row["e2e"], 4),
                "hit_total_param_channel": round(pooled_row["param_channel"], 4),
                "hit_total_memory_channel": round(pooled_row["memory_channel"], 4),
                "ci95_normal": [round(x, 4) for x in pooled_row["ci95_normal"]],
                "ci95_bootstrap": [round(x, 4) for x in pooled_row["ci95_bootstrap"]],
                "seed_subset_means": [round(x, 4) for x in pooled_row["seed_subset_means"]],
                "nll_delta": round(pooled_row["nll_delta_mean_of_slices"], 4),
                "degen": round(pooled_row["degenerate_rate"], 4),
                "poolable": p["protocol"]["poolable"],
            })
    with open(LOG, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    print(f"appended {len(lines)} lines to {LOG}")


if __name__ == "__main__":
    main()
