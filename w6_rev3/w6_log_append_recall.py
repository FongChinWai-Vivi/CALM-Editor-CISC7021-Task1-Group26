# -*- coding: utf-8 -*-
"""Append the gpt2-small recall pass to the append-only quality log."""
import glob
import hashlib
import json
import os
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, "EasyEdit", "w6run", "quality_log.jsonl")


def sha1(path, n=12):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


lines = []
for f in sorted(glob.glob(os.path.join(REPO, "w6_rev3", "recall",
                                       "recall_gpt2small_o*_n100.json"))):
    d = json.load(open(f, encoding="utf-8"))
    lines.append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checker": "Sonia/Vivi",
        "note": ("Rev3 recall on the sweep backbone at n=100/slice, with shuffled-key "
                 "control; supersedes the gpt2-xl n=20 approximation"),
        "kind": "recall",
        "model": d["model"],
        "run": "w6_rev3/recall/" + os.path.basename(f),
        "sha1": sha1(f),
        "slice": d["slice_tag"],
        "n": d["n_routed_edits"],
        "key_material": d["key_material"],
        "canonical_recall": d["memory_canonical_recall"],
        "paraphrase_recall": d["memory_paraphrase_recall"],
        "paraphrase_recall_case_any": d["memory_paraphrase_recall_case_any"],
        "paraphrase_denominator": d["memory_paraphrase_denominator"],
        "baseline_recall": d["memory_baseline_recall"],
        "shuffled_recall": d["memory_shuffled_recall"],
        "shuffled_n": d["memory_shuffled_n"],
        "logp_read_delta": d["logp_read_delta"],
        "note_echo_rate": d["note_echo_rate"],
    })

rows = [json.load(open(f, encoding="utf-8")) for f in sorted(
    glob.glob(os.path.join(REPO, "w6_rev3", "recall", "recall_gpt2small_o*_n100.json")))]
if rows:
    mean = lambda k: round(sum(r[k] for r in rows) / len(rows), 4)
    lines.append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checker": "Sonia/Vivi",
        "note": "Rev3 recall, mean over 8 slices × 100 edits — the citable recall row",
        "kind": "recall_pooled",
        "model": "gpt2-small",
        "run": "w6_rev3/recall/recall_summary.json",
        "sha1": sha1(os.path.join(REPO, "w6_rev3", "recall", "recall_summary.json")),
        "slices": len(rows),
        "n": sum(r["n_routed_edits"] for r in rows),
        "canonical_recall": mean("memory_canonical_recall"),
        "paraphrase_recall": mean("memory_paraphrase_recall"),
        "paraphrase_recall_case_any": mean("memory_paraphrase_recall_case_any"),
        "baseline_recall": mean("memory_baseline_recall"),
        "shuffled_recall": mean("memory_shuffled_recall"),
        "logp_read_delta": mean("logp_read_delta"),
    })

with open(LOG, "a", encoding="utf-8") as fh:
    for ln in lines:
        fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
print(f"appended {len(lines)} recall lines")
