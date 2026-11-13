"""R15 - can a better key layer recover the store entry from a paraphrase prompt?

Offline, no model calls.  For every paraphrase item that the edited model was actually asked
(paraphrase_rows in the run summaries, joined to rows by slot) we ask a narrower question than
"did it answer": *is the entity that the store is keyed on still present in the query at all?*

Three tests, weakest first:

  verbatim   subject string appears in the prompt (case-folded, punctuation folded)
  tokens     every content token of the subject appears somewhere in the prompt (order-free)
  strict     verbatim, and the relation's target is not the pre-edit value

If the entity is present in most paraphrase prompts, then a normalisation / approximate-key
layer cannot be the missing piece: the key is recoverable, so the failure must be downstream.
That is the claim this probe is allowed to settle, and nothing more.
"""
import json
import os
import re
import glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(ROOT, "EasyEdit", "w6run")

# Only runs made after the Rev4b field `paraphrase_rows` exists carry the per-item read, so the
# R6 / R11 W=50 arms cannot be scored here; R12 (W=100, same eight slices) is the in-sample
# substitute and the two W=100/W=50 held-out blocks cover fresh data.
ARMS = [
    ("in-sample W100", "R12_o{off}_W100_W100b025_inject/results_defer_W100.json", range(0, 800, 100)),
    ("held-out W50", "R13_hold_o{off}_W50_b025/results_defer_W50.json", range(1600, 2000, 100)),
    ("held-out W100", "R13_hold_o{off}_W100_b025/results_defer_W100.json", range(1600, 2000, 100)),
]

STOP = {"the", "of", "and", "de", "la", "le", "les", "du", "des", "van", "von", "da", "di"}


def norm(s: str) -> str:
    s = s.casefold()
    s = re.sub(r"[^0-9a-z\u00c0-\u024f\u4e00-\u9fff]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str):
    return [t for t in norm(s).split() if len(t) >= 3 and t not in STOP]


def load_arm(pattern, offsets):
    per_slice = []
    for off in offsets:
        f = os.path.join(RUN_ROOT, pattern.format(off=off))
        if not os.path.exists(f):
            continue
        d = json.load(open(f, encoding="utf-8"))
        rows = {r.get("slot"): r for r in d.get("rows", [])}
        pr = d.get("summary", {}).get("paraphrase_rows") or []
        items = []
        for item in pr:
            r = rows.get(item.get("slot"))
            if not r:
                continue
            subj = r.get("subject") or ""
            prompt = item.get("prompt") or ""
            np, ns = norm(prompt), norm(subj)
            toks = tokens(subj)
            items.append({
                "slice": off,
                "ok": bool(item.get("ok")),
                "verbatim": bool(ns) and ns in np,
                "tokens": bool(toks) and all(t in np for t in toks),
                "subject": subj,
            })
        if items:
            per_slice.append((off, items))
    return per_slice


def summarise(per_slice):
    allitems = [i for _, items in per_slice for i in items]
    n = len(allitems)
    if not n:
        return None
    out = {"n": n, "recall": sum(i["ok"] for i in allitems) / n}
    for key in ("verbatim", "tokens"):
        hit = [i for i in allitems if i[key]]
        miss = [i for i in allitems if not i[key]]
        out[key] = {
            "share": len(hit) / n,
            "recall_when_present": (sum(i["ok"] for i in hit) / len(hit)) if hit else None,
            "recall_when_absent": (sum(i["ok"] for i in miss) / len(miss)) if miss else None,
            "n_present": len(hit),
            "n_absent": len(miss),
            "ok_recoverable": sum(i["ok"] for i in hit),
            "ok_total": sum(i["ok"] for i in allitems),
        }
    return out


def main():
    report = {}
    for label, pattern, offsets in ARMS:
        per_slice = load_arm(pattern, offsets)
        summary = summarise(per_slice)
        if summary is None:
            print(f"[skip] {label}: no paraphrase rows")
            continue
        report[label] = {
            "slices": [off for off, _ in per_slice],
            "summary": summary,
            "per_slice": {
                str(off): summarise([(off, items)]) for off, items in per_slice
            },
        }
        s = summary
        print(f"\n== {label}: {s['n']} paraphrase items over {len(per_slice)} slices, "
              f"recall {s['recall']:.4f}")
        for key in ("verbatim", "tokens"):
            k = s[key]
            print(f"   {key:9s} present {k['share']:.3f} (n={k['n_present']}) "
                  f"recall|present {k['recall_when_present']} "
                  f"recall|absent {k['recall_when_absent']}")

    out = os.path.join(ROOT, "w6_rev3", "r15_key_recoverability.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
