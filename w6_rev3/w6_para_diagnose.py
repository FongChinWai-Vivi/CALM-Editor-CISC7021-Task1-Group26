# -*- coding: utf-8 -*-
"""Offline diagnosis of the paraphrase gap — judging matrix + failure taxonomy.

Inputs are the already-persisted read-path sweeps (8 slices x 100 subjects,
gpt2-small, note injected, greedy decode).  No model is run here: every number
is recomputed from the saved per-item prompts and answers.

The point is to answer one question before spending any more GPU time: the
paraphrase number is 0.29-0.33 while canonical is 0.72.  Is that gap
  (a) a judging artefact (right answer, different surface form),
  (b) a routing/key artefact (the note is never found), or
  (c) a genuine reader failure (note present in context, model answers anyway)?
`memory_lookup_rate` in the same files already answers (b) for this harness.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECALL = os.path.join(REPO, "w6_rev3", "recall")
OUT = os.path.join(REPO, "w6_rev3", "para_diagnose.json")

WORD = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "the", "of", "in", "is", "was", "are", "to", "and", "or",
        "for", "on", "at", "by", "with", "as", "from", "that", "this", "it",
        "he", "she", "they", "his", "her", "their", "be", "been"}


def toks(s):
    return WORD.findall((s or "").lower())


def content(s):
    return [w for w in toks(s) if w not in STOP]


def strict_hit(answer, target):
    return target.strip().lower() in (answer or "").lower()


def norm_hit(answer, target):
    """Target present after collapsing punctuation/case/possessives."""
    a = " ".join(toks(answer))
    t = " ".join(toks(target))
    return bool(t) and t in a


def lead_hit(answer, target, k=6):
    """Target's content words appear within the first k words of the answer."""
    a = content(answer)[:k]
    t = content(target)
    return bool(t) and all(w in a for w in t)


def token_f1(answer, target):
    a, t = set(content(answer)), set(content(target))
    if not a or not t:
        return 0.0
    inter = len(a & t)
    if not inter:
        return 0.0
    p, r = inter / len(a), inter / len(t)
    return 2 * p * r / (p + r)


def old_value_retained(answer, base_answer):
    """Does the answer still lead with the pre-edit value?"""
    b = content(base_answer)
    a = content(answer)
    return bool(b) and bool(a) and b[0] == a[0]


def classify(case, para):
    ans, tgt = para["answer"], case["target_new"]
    if strict_hit(ans, tgt):
        return "hit"
    if not (ans or "").strip():
        return "empty"
    words = toks(ans)
    if len(words) >= 4 and len(set(words)) == 1:
        return "degenerate"
    if norm_hit(ans, tgt):
        return "surface_only"          # right entity, punctuation/case blocked it
    if lead_hit(ans, tgt):
        return "lead_hit_late"         # right entity, but buried past the read window
    if token_f1(ans, tgt) >= 0.5:
        return "partial_entity"
    if old_value_retained(ans, case["base_answer"]):
        return "old_value"
    if note_echo(ans, case.get("note") or ""):
        return "note_echo_no_target"   # copies the note's frame, not the answer
    return "other"


def note_echo(answer, note):
    a, n = set(content(answer)), set(content(note))
    return bool(a & n) and len(a & n) / max(1, len(a)) >= 0.5


def main():
    files = sorted(glob.glob(os.path.join(RECALL, "recall_gpt2small_o*_n100.json")))
    if not files:
        raise SystemExit("no recall files")

    agg = Counter()
    per_slice = []
    canon = Counter()
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        c = Counter()
        for case in d["per_case"]:
            canon["n"] += 1
            canon["strict"] += int(strict_hit(case["canon_answer"], case["target_new"]))
            canon["norm"] += int(norm_hit(case["canon_answer"], case["target_new"]))
            for para in case.get("paraphrase") or []:
                c["n"] += 1
                c["strict"] += int(strict_hit(para["answer"], case["target_new"]))
                c["norm"] += int(norm_hit(para["answer"], case["target_new"]))
                c["lead"] += int(lead_hit(para["answer"], case["target_new"]))
                c["f1_50"] += int(token_f1(para["answer"], case["target_new"]) >= 0.5)
                c[classify(case, para)] += 1
        agg.update(c)
        per_slice.append({
            "slice": d["slice_tag"], "n": c["n"],
            "strict": round(c["strict"] / c["n"], 4),
            "norm": round(c["norm"] / c["n"], 4),
            "lead6": round(c["lead"] / c["n"], 4),
            "f1_50": round(c["f1_50"] / c["n"], 4),
            "lookup_rate": d.get("memory_lookup_rate"),
            "file_para_recall": d.get("memory_paraphrase_recall"),
        })

    n = agg["n"]
    out = {
        "scope": ("read-path harness, clean gpt2-small, note injected before the "
                  "question, greedy decode, 8 slices x 100 subjects x 2 paraphrases"),
        "paraphrase": {
            "n": n,
            "strict": round(agg["strict"] / n, 4),
            "normalised": round(agg["norm"] / n, 4),
            "lead6": round(agg["lead"] / n, 4),
            "token_f1_ge_0.5": round(agg["f1_50"] / n, 4),
            "taxonomy": {k: round(agg[k] / n, 4) for k in
                         ("hit", "surface_only", "lead_hit_late", "partial_entity",
                          "old_value", "note_echo_no_target", "degenerate", "empty", "other")},
            "taxonomy_counts": {k: agg[k] for k in
                                ("hit", "surface_only", "lead_hit_late", "partial_entity",
                                 "old_value", "note_echo_no_target", "degenerate", "empty", "other")},
        },
        "canonical": {
            "n": canon["n"],
            "strict": round(canon["strict"] / canon["n"], 4),
            "normalised": round(canon["norm"] / canon["n"], 4),
        },
        "per_slice": per_slice,
        "read": None,
    }

    p = out["paraphrase"]
    surface_gain = p["normalised"] - p["strict"]
    out["read"] = {
        "judging_artefact_gain": round(surface_gain, 4),
        "judging_artefact_share_of_gap": round(
            surface_gain / max(1e-9, out["canonical"]["normalised"] - p["strict"]), 4),
        "routing_is_not_the_bottleneck": (
            "memory_lookup_rate = 1.0 on every slice, so the note was found for "
            "every routed case; the gap is downstream of the key."),
        "verdict": ("judging explains only %.1f pp of the %.1f pp canonical-paraphrase gap"
                    % (100 * surface_gain,
                       100 * (out["canonical"]["normalised"] - p["strict"]))),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
