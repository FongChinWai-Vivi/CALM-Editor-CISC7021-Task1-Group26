# -*- coding: utf-8 -*-
"""Append the R11 parameter-channel lever sweep to the append-only quality log.

Same contract as w6_log_append.py: every line names the artefact, its SHA1 and
the full setting tuple, so a later checker can tell what was re-run from what was
merely restated.  R11 = three arms, one lever each, on the frozen R6 protocol.
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

ARMS = [
    ("r11_b015", "budget 0.15, W=50",
     os.path.join(REPO, "EasyEdit", "w6run", "R11_o*_W50_b015_inject", "results_defer_W50.json"), 50, 0.15),
    ("r11_b035", "budget 0.35, W=50",
     os.path.join(REPO, "EasyEdit", "w6run", "R11_o*_W50_b035_inject", "results_defer_W50.json"), 50, 0.35),
    ("r11_w25", "budget 0.25, W=25",
     os.path.join(REPO, "EasyEdit", "w6run", "R11_o*_W25_W25b025_inject", "results_defer_W25.json"), 25, 0.25),
]

NOTE = ("R11 參數通道槓桿探針：在 R6 凍結協議上只改一個槓桿（寫入 budget 或 defer window），"
        "用來判定 §4 所述 0.369 上限是調參缺失還是結構限制；結論見報告 §11")


def sha1(path, n=12):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def entry_for(path, tag, lever, W, budget):
    d = json.load(open(path, encoding="utf-8"))
    s = d["summary"]
    rel = os.path.relpath(path, os.path.join(REPO, "EasyEdit")).replace("\\", "/")
    off = os.path.basename(os.path.dirname(path)).split("_")[1]
    return {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checker": "Vivi",
        "note": NOTE,
        "set": tag,
        "lever": lever,
        "run": "w6run/" + rel.replace("w6run/", ""),
        "sha1": sha1(path),
        "offset": int(off.lstrip("o")),
        "slice": f"{int(off.lstrip('o'))}-{int(off.lstrip('o')) + 99}",
        "n": s.get("edits"),
        "W": s.get("defer_window"),
        "layers": json.dumps(s.get("layers")),
        "refine": s.get("refine_passes"),
        "delta_budget": s.get("delta_budget"),
        "memory_read": s.get("memory_read"),
        "memory_read_template": s.get("memory_read_template"),
        "memory_read_arbitrate": s.get("memory_read_arbitrate"),
        "gate_write_threshold": s.get("gate_write_threshold"),
        "coverage": s.get("param_coverage"),
        "immediate": s.get("immediate_hit"),
        "final_in_gate": s.get("final_state_hit"),
        "hit_total_param_channel": s.get("hit_total_param_channel"),
        "hit_total_memory_channel": s.get("hit_total_memory_channel"),
        "hit_total": s.get("hit_total_measured"),
        "e2e_measured": s.get("hit_total_measured"),
        "nll_delta": s.get("nll_delta"),
        "degen": s.get("degenerate_rate"),
    }


def main():
    lines = []
    for tag, lever, pattern, _W, _B in ARMS:
        files = sorted(glob.glob(pattern))
        if len(files) != 8:
            raise SystemExit(f"{tag}: expected 8 slices, found {len(files)}")
        for f in files:
            lines.append(entry_for(f, tag, lever, None, None))

    pp = os.path.join(REV3, "r11_param_probe_pooled.json")
    pooled = json.load(open(pp, encoding="utf-8"))
    for tag, arm in pooled.items():
        p = arm["pooled"]
        lines.append({
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checker": "Vivi",
            "note": "R11 池化（8 切片 × 100 編集 = 800）；所有數字從 rows 重算，不讀 summary；"
                    "arms poolable 已驗證",
            "set": tag,
            "run": "w6_rev3/r11_param_probe_pooled.json",
            "sha1": sha1(pp),
            "slices": 8,
            "n": p["n"],
            "W": arm["settings_tuple"].get("defer_window"),
            "delta_budget": arm["settings_tuple"].get("delta_budget"),
            "settings_tuple": arm["settings_tuple"],
            "hit_total": round(p["e2e"], 4),
            "e2e_measured": round(p["e2e"], 4),
            "hit_total_param_channel": round(p["param_channel"], 4),
            "hit_total_memory_channel": round(p["memory_channel"], 4),
            "ci95_normal": [round(x, 4) for x in p["ci95_normal"]],
            "ci95_bootstrap": [round(x, 4) for x in p["ci95_bootstrap"]],
            "seed_subset_means": [round(x, 4) for x in p["seed_subset_means"]],
            "nll_delta": round(p["nll_delta_mean_of_slices"], 4),
            "degen": round(p["degenerate_rate"], 4),
            "poolable": arm["poolable"],
        })

    with open(LOG, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    print(f"appended {len(lines)} lines to {LOG}")


if __name__ == "__main__":
    main()
