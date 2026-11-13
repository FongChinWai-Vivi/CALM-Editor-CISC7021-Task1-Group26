# -*- coding: utf-8 -*-
"""Append R12 (window W=100), R13 (held-out window check) and the R12 read-path
probes to the append-only quality log.

Same contract as w6_log_append.py / w6_log_append_r11.py: one line per slice run
plus one line per pooled arm, each carrying the file SHA1 and the full setting
tuple, so a later checker can tell what was re-run from what was restated.

Idempotent: a (set, offset) pair already present in the log is skipped, so a
re-run after a crash does not double-count.
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

RUNS = [
    ("r12_w100_b025", "budget 0.25, W=100",
     os.path.join(REPO, "EasyEdit", "w6run", "R12_o*_W100_W100b025_inject", "results_defer_W100.json"), 8),
    ("r13_hold_w50_b025", "held-out offsets 1600-1900, budget 0.25, W=50",
     os.path.join(REPO, "EasyEdit", "w6run", "R13_hold_o*_W50_b025", "results_defer_W50.json"), 4),
    ("r13_hold_w100_b025", "held-out offsets 1600-1900, budget 0.25, W=100",
     os.path.join(REPO, "EasyEdit", "w6run", "R13_hold_o*_W100_b025", "results_defer_W100.json"), 4),
]

NOTES = {
    "r12_w100_b025": ("R12 窗口反向探針：W=50 凍結協議只把 defer window 拉到 100（每 100 編集一次寫出），"
                      "用來檢驗 §11 的「flush 次數而非窗口長度主導干擾」假設；結論見報告 §13"),
    "r13_hold_w50_b025": ("R13 留出集對照（W=50 臂）：offset 1600-1900 先前所有 arm 從未使用，"
                          "用來檢驗 R12 的增益是否只在評估切片上成立"),
    "r13_hold_w100_b025": ("R13 留出集對照（W=100 臂）：與同切片 W=50 臂成對，"
                           "作為 R12 窗口結論的樣本外驗證"),
}


def sha1(path, n=12):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def existing_pairs():
    seen = set()
    if not os.path.exists(LOG):
        return seen
    with open(LOG, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if "offset" in d and "set" in d:
                seen.add((d["set"], d["offset"]))
    return seen


def entry_for(path, tag):
    d = json.load(open(path, encoding="utf-8"))
    s = d["summary"]
    rel = os.path.relpath(path, os.path.join(REPO, "EasyEdit")).replace("\\", "/")
    off = int(os.path.basename(os.path.dirname(path)).split("_o")[1].split("_")[0])
    return {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checker": "Vivi",
        "note": NOTES[tag],
        "set": tag,
        "run": "w6run/" + rel.replace("w6run/", ""),
        "sha1": sha1(path),
        "offset": off,
        "slice": f"{off}-{off + 99}",
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
        "paraphrase_recall": s.get("paraphrase_recall"),
        "paraphrase_n": s.get("paraphrase_n"),
        "flush_count": len(s.get("flush_hist") or []),
        "total_seconds": s.get("total_seconds"),
    }


def main():
    seen = existing_pairs()
    lines = []
    for tag, _lever, pattern, expect in RUNS:
        files = sorted(glob.glob(pattern))
        if len(files) != expect:
            print("WARN %s: expected %d slices, found %d" % (tag, expect, len(files)))
        for f in files:
            off = int(os.path.basename(os.path.dirname(f)).split("_o")[1].split("_")[0])
            if (tag, off) in seen:
                continue
            lines.append(entry_for(f, tag))

    pp = os.path.join(REV3, "r12_window_probe_pooled.json")
    if os.path.exists(pp):
        pooled = json.load(open(pp, encoding="utf-8"))
        for tag, arm in pooled.items():
            if (tag, "pooled") in seen:
                continue
            p = arm["pooled"]
            lines.append({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "checker": "Vivi",
                "note": "R12 池化（8 切片 × 100 編集 = 800）；所有數字從 rows 重算，不讀 summary；"
                        "arms poolable 已驗證；paraphrase 為 in-loop 實測",
                "set": tag,
                "run": "w6_rev3/r12_window_probe_pooled.json",
                "sha1": sha1(pp),
                "slices": len(arm["per_slice"]),
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
                "paraphrase_recall": p.get("paraphrase_recall"),
                "poolable": arm["poolable"],
                "paired_vs_r6": arm.get("paired_vs_r6"),
            })

    # Read-path probes: cross-backbone recall + the decision criteria on each.
    probes = [
        ("r12_probe_recall_gpt2medium", "gpt2-medium 讀取路徑 recall（o0,o100 各 100 edits）",
         os.path.join(REV3, "recall_gpt2medium")),
        ("r12_probe_decision_gpt2medium", "gpt2-medium 決策判據探針（top-k / 決鬥 / 旋轉對照）",
         os.path.join(REV3, "para_decision_probe_gpt2medium.json")),
    ]
    for tag, note, target in probes:
        if (tag, "probe") in seen:
            continue
        if os.path.isdir(target):
            files = sorted(f for f in glob.glob(os.path.join(target, "*.json")))
        else:
            files = [target] if os.path.exists(target) else []
        for f in files:
            d = json.load(open(f, encoding="utf-8"))
            if os.path.isdir(target):
                keep = {k: v for k, v in d.items() if k != "per_case"}
            else:
                keep = {"channels": {ch: {k: v for k, v in c.items() if not isinstance(v, dict) or k.startswith("control")}
                                     for ch, c in d.get("channels", {}).items()}}
            lines.append({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "checker": "Vivi",
                "note": note,
                "set": tag,
                "run": os.path.relpath(f, REPO).replace("\\", "/"),
                "sha1": sha1(f),
                "probe": keep,
            })

    with open(LOG, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    print("appended %d lines to %s" % (len(lines), LOG))


if __name__ == "__main__":
    main()
