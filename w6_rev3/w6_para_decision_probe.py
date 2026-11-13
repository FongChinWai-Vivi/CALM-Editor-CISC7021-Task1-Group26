# -*- coding: utf-8 -*-
"""Decision-level probe: is the paraphrase gap a *knowledge* failure or a
*decoding* failure?

For every (note + prompt) context in the read-path slices we ask the model, in a
single teacher-forced forward pass:

  rank      : rank of the target's first token in the next-token distribution
  top1/5/10 : is the target's first token the argmax / in the top k
  lp_target : mean per-token log P(target | note + prompt)
  lp_greedy : mean per-token log P(measured greedy answer | note + prompt)
  duel      : would a likelihood rerank between {target, greedy answer} pick the
              target?

`duel` is the number that matters: it is the recall a copy/rerank read channel
could reach without touching the weights, i.e. an upper bound on what a decode
fix buys, measured on exactly the items the greedy read already failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = r"C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
sys.path.insert(0, os.path.join(REPO, "FONG HIO LAM", "side_memory_router"))

from side_memory import SideMemory, build_memory_note  # noqa: E402

MODEL_DIR = os.path.join(REPO, "EasyEdit", "hugging_cache", "gpt2-small")
DATA = os.path.join(REPO, "EasyEdit", "data", "counterfact.json")
RECALL = os.path.join(REPO, "w6_rev3", "recall")
OUT = os.path.join(REPO, "w6_rev3", "para_decision_probe.json")


def _norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def subject_prompts() -> Dict[str, str]:
    """subject -> canonical prompt, rebuilt from the dataset (per_case does not
    store the canonical prompt, only the paraphrase ones)."""
    recs = json.load(open(DATA, encoding="utf-8"))
    out: Dict[str, str] = {}
    for rec in recs:
        rr = rec.get("requested_rewrite", rec)
        subj = rr.get("subject")
        if subj and subj not in out:
            out[subj] = (rr.get("prompt") or "").replace("{}", subj)
    return out


@torch.no_grad()
def score(model, tok, ctx: str, cand: str) -> float:
    """Mean per-token log P(cand | ctx)."""
    pid = tok(ctx, return_tensors="pt", truncation=True, max_length=96).input_ids.to(model.device)
    tid = tok(" " + cand.strip(), add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    if tid.size(1) == 0:
        return -1e9
    ids = torch.cat([pid, tid], dim=1)
    logits = model(input_ids=ids).logits[0].float()
    start = pid.size(1)
    total = 0.0
    for j in range(tid.size(1)):
        lp = torch.log_softmax(logits[start + j - 1], dim=-1)
        total += float(lp[tid[0, j]])
    return total / tid.size(1)


@torch.no_grad()
def first_token_rank(model, tok, ctx: str, cand: str) -> Dict[str, Any]:
    pid = tok(ctx, return_tensors="pt", truncation=True, max_length=96).input_ids.to(model.device)
    tid = tok(" " + cand.strip(), add_special_tokens=False).input_ids
    if not tid:
        return {"rank": None, "top1": False, "top5": False, "top10": False}
    first = int(tid[0])
    logits = model(input_ids=pid).logits[0, -1].float()
    order = torch.argsort(logits, descending=True)
    rank = int((order == first).nonzero()[0].item()) + 1
    return {"rank": rank, "top1": rank == 1, "top5": rank <= 5, "top10": rank <= 10}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap items per slice (0 = all)")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="backbone to score; default is the frozen gpt2-small path")
    ap.add_argument("--recall-glob", default=os.path.join(RECALL, "recall_gpt2small_o*_n100.json"),
                    help="recall slices whose greedy answers are re-scored")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to("cuda").eval()

    files = sorted(__import__("glob").glob(args.recall_glob))
    SUBJ2PROMPT = subject_prompts()
    acc: Dict[str, Dict[str, float]] = {}
    per_item: List[Dict[str, Any]] = []
    t0 = time.time()

    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        cases = d["per_case"]
        if args.limit:
            cases = cases[:args.limit]
        notes = [(c.get("subject"), c.get("relation"), c.get("note")) for c in cases]
        for ci, c in enumerate(cases):
            note = c.get("note")
            if not note:
                continue
            note_pref = build_memory_note(c["subject"], c["relation"], note)
            # negative control: the same context with ANOTHER edit's note.  Without
            # it a top-k rate cannot be told apart from "the model likes this token
            # anyway", which is exactly what makes top-k claims worthless.
            donor = None
            for step in range(1, len(notes) + 1):
                cand = notes[(ci + step) % len(notes)]
                if cand[2] and cand[0] != c["subject"]:
                    donor = cand
                    break
            shuf_pref = (build_memory_note(donor[0], donor[1], donor[2])
                         if donor else None)
            tgt = c["target_new"]

            canon_prompt = SUBJ2PROMPT.get(c["subject"], "")
            for kind, pairs in (("canonical", [(c["canon_answer"], c["canon_hit"], canon_prompt)]
                                 if canon_prompt else []),
                                ("paraphrase", [(p["answer"], p["hit"], p["prompt"])
                                                for p in (c.get("paraphrase") or [])])):
                for greedy, hit, prompt in pairs:
                    ctx = note_pref + prompt
                    rk = first_token_rank(model, tok, ctx, tgt)
                    lp_t = score(model, tok, ctx, tgt)
                    lp_g = score(model, tok, ctx, greedy) if (greedy or "").strip() else -1e9
                    rk_s = (first_token_rank(model, tok, shuf_pref + prompt, tgt)
                            if shuf_pref else {"rank": None, "top1": False,
                                               "top5": False, "top10": False})
                    a = acc.setdefault(kind, {"n": 0, "hit": 0, "top1": 0, "top5": 0,
                                              "top10": 0, "duel_win": 0, "rank_sum": 0.0,
                                              "rank_n": 0, "hit_and_top10": 0,
                                              "s_top1": 0, "s_top5": 0, "s_top10": 0, "s_n": 0})
                    a["n"] += 1
                    a["hit"] += int(hit)
                    a["top1"] += int(rk["top1"])
                    a["top5"] += int(rk["top5"])
                    a["top10"] += int(rk["top10"])
                    if rk["rank"]:
                        a["rank_sum"] += rk["rank"]
                        a["rank_n"] += 1
                    a["s_n"] += 1
                    a["s_top1"] += int(rk_s["top1"])
                    a["s_top5"] += int(rk_s["top5"])
                    a["s_top10"] += int(rk_s["top10"])
                    win = lp_t > lp_g
                    a["duel_win"] += int(win)
                    a["hit_and_top10"] += int(hit and rk["top10"])
                    per_item.append({"slice": d["slice_tag"], "kind": kind, "subject": c["subject"],
                                     "target": tgt, "greedy": greedy, "greedy_hit": hit,
                                     "rank": rk["rank"], "top10": rk["top10"],
                                     "shuf_top1": rk_s["top1"], "shuf_top10": rk_s["top10"],
                                     "lp_target": round(lp_t, 4), "lp_greedy": round(lp_g, 4),
                                     "duel_win": win})
        print(f"[{d['slice_tag']}] done  {time.time() - t0:.0f}s", flush=True)

    out: Dict[str, Any] = {
        "model": MODEL_DIR,
        "scope": "read-path contexts (side-memory note prepended), greedy answers reused from recall/*.json",
        "seconds": round(time.time() - t0, 1),
        "channels": {},
        "per_item_file": None,
    }
    for kind, a in acc.items():
        n = a["n"]
        out["channels"][kind] = {
            "n": n,
            "greedy_recall": round(a["hit"] / n, 4),
            "target_first_token_top1": round(a["top1"] / n, 4),
            "top5": round(a["top5"] / n, 4),
            "top10": round(a["top10"] / n, 4),
            "median_rank": (sorted([x["rank"] for x in per_item
                                    if x["kind"] == kind and x["rank"]])[n // 2]
                            if n else None),
            "mean_rank": round(a["rank_sum"] / max(1, a["rank_n"]), 1),
            "likelihood_duel_win": round(a["duel_win"] / n, 4),
            "greedy_hit_and_top10": round(a["hit_and_top10"] / n, 4),
            "control_shuffled_note": {
                "n": a["s_n"],
                "top1": round(a["s_top1"] / max(1, a["s_n"]), 4),
                "top5": round(a["s_top5"] / max(1, a["s_n"]), 4),
                "top10": round(a["s_top10"] / max(1, a["s_n"]), 4),
            },
            "top10_margin_over_control": round(
                (a["top10"] - a["s_top10"]) / n, 4),
        }
    # split the duel by whether the greedy read already failed -- that is the
    # population a decode fix would actually have to rescue.
    for kind in acc:
        fails = [x for x in per_item if x["kind"] == kind and not x["greedy_hit"]]
        if fails:
            out["channels"][kind]["duel_win_on_greedy_failures"] = round(
                sum(x["duel_win"] for x in fails) / len(fails), 4)
            out["channels"][kind]["n_greedy_failures"] = len(fails)
            out["channels"][kind]["top10_on_greedy_failures"] = round(
                sum(x["top10"] for x in fails) / len(fails), 4)

    p = os.path.join(REPO, "w6_rev3", "para_decision_probe_items.json")
    json.dump(per_item, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    out["per_item_file"] = os.path.relpath(p, REPO)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out["channels"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
