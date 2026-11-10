"""w6_recall_gpt2small.py -- Sonia slice of the Day-1 plan, run locally.

Measures the SideMemory *read* path on gpt2-small with n>=100 per slice:

  canonical recall   : note + original prompt        -> answer contains target?
  paraphrase recall  : note + paraphrase prompt      -> answer contains target?
  baseline recall    : original prompt, no note      -> answer contains target?
  shuffled recall    : note of ANOTHER edit injected -> answer contains target?
  lookup rate        : does lookup(subject, relation) resolve at all?

Plus a likelihood read-out that does not depend on greedy decoding luck:
  logp_delta = mean log P(target_new | note + prompt) - log P(target_new | prompt)

Key = (subject, relation) only; the answer is the value, never part of the key.
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

RELATION_FALLBACK = "is"


def infer_relation(prompt_tmpl: str, subject: str) -> str:
    """Same relation signature as real_w3_components.infer_relation.

    Re-implemented here because that module also imports ``model_utils``, which
    only exists on the teammate's machine.
    """
    tmpl = (prompt_tmpl or "").strip()
    if not tmpl:
        return RELATION_FALLBACK
    head, _, tail = tmpl.partition("{}")
    rel = " ".join(tail.replace("{}", subject).split()).strip(" .?!,")
    if not rel:
        rel = " ".join(head.replace("{}", subject).split()).strip(" .?!,")
    return rel or RELATION_FALLBACK

MODEL_DIR = os.path.join(REPO, "EasyEdit", "hugging_cache", "gpt2-small")
DATA = os.path.join(REPO, "EasyEdit", "data", "counterfact.json")


def _norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def contains(answer: str, target: str) -> bool:
    a, t = _norm(answer), _norm(target)
    return bool(t) and t in a


def _as_list(v) -> List[str]:
    if isinstance(v, str):
        try:
            v = json.loads(v.replace("\\'", "'"))
        except Exception:
            v = [v]
    return [x for x in (v or []) if isinstance(x, str)]


def parse_records(path: str, offset: int, n: int) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)[offset:offset + n]
    cases = []
    for rec in records:
        rr = rec.get("requested_rewrite", rec)
        tmpl = rr["prompt"]
        subject = rr["subject"]
        t_new = rr["target_new"]
        target_new = t_new.get("str") if isinstance(t_new, dict) else t_new
        para = _as_list(rec.get("paraphrase_prompts")) or _as_list(rr.get("paraphrase_prompts"))
        cases.append({
            "subject": subject,
            "relation": infer_relation(tmpl, subject),
            "target_new": target_new,
            "prompt": tmpl.replace("{}", subject),
            "paraphrase": para,
        })
    return [c for c in cases if c["subject"] and c["target_new"] and c["prompt"]]


@torch.no_grad()
def gen(model, tok, prompt: str, max_new: int) -> str:
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=96).input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True).strip()


@torch.no_grad()
def logp_target(model, tok, prompt: str, target: str) -> float:
    """Mean per-token log P(target | prompt)."""
    pid = tok(prompt, return_tensors="pt", truncation=True, max_length=96).input_ids.to(model.device)
    tid = tok(" " + target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    ids = torch.cat([pid, tid], dim=1)
    out = model(input_ids=ids)
    logits = out.logits[0].float()
    start = pid.size(1)
    logp = 0.0
    for j in range(tid.size(1)):
        lp = torch.log_softmax(logits[start + j - 1], dim=-1)
        logp += float(lp[tid[0, j]])
    return logp / max(1, tid.size(1))


def run_slice(model, tok, cases, max_new: int, tag: str) -> Dict[str, Any]:
    mem = SideMemory(n_shards=64, max_per_shard=500, seed=42)
    for c in cases:
        mem.store(c["subject"], c["relation"], c["target_new"])

    stats = dict(n=len(cases), lookup_hits=0, base_hits=0, canon_hits=0,
                 para_hits=0, para_instances=0, para_any=0,
                 shuffle_hits=0, shuffle_checked=0, lp_delta_sum=0.0,
                 lp_delta_n=0, canon_note_echo=0)
    per_case = []
    t0 = time.time()

    for idx, c in enumerate(cases):
        note = mem.lookup(c["subject"], c["relation"])
        stats["lookup_hits"] += int(note is not None)

        base_ans = gen(model, tok, c["prompt"], max_new)
        base_hit = contains(base_ans, c["target_new"])
        stats["base_hits"] += int(base_hit)

        read_ans, read_hit = "", False
        if note is not None:
            note_pref = build_memory_note(c["subject"], c["relation"], note)
            read_ans = gen(model, tok, note_pref + c["prompt"], max_new)
            read_hit = contains(read_ans, c["target_new"])
            stats["canon_note_echo"] += int(contains(read_ans, note))
            lp_with = logp_target(model, tok, note_pref + c["prompt"], c["target_new"])
            lp_without = logp_target(model, tok, c["prompt"], c["target_new"])
            stats["lp_delta_sum"] += (lp_with - lp_without)
            stats["lp_delta_n"] += 1
        stats["canon_hits"] += int(read_hit)

        para_rows = []
        if note is not None:
            note_pref = build_memory_note(c["subject"], c["relation"], note)
            for p in c["paraphrase"]:
                pa = gen(model, tok, note_pref + p, max_new)
                ph = contains(pa, c["target_new"])
                para_rows.append({"prompt": p, "answer": pa[:120], "hit": ph})
                stats["para_hits"] += int(ph)
                stats["para_instances"] += 1
            stats["para_any"] += int(any(r["hit"] for r in para_rows))

        other = cases[(idx + 1) % len(cases)]
        wrong = mem.lookup(other["subject"], other["relation"])
        sh_ans, sh_hit = "", False
        if wrong is not None and other["subject"] != c["subject"] and wrong != c["target_new"]:
            sh_ans = gen(model, tok,
                         build_memory_note(other["subject"], other["relation"], wrong) + c["prompt"],
                         max_new)
            sh_hit = contains(sh_ans, c["target_new"])
            stats["shuffle_checked"] += 1
        stats["shuffle_hits"] += int(sh_hit)

        per_case.append({"subject": c["subject"], "relation": c["relation"],
                         "target_new": c["target_new"], "note": note,
                         "base_answer": base_ans[:120], "base_hit": base_hit,
                         "canon_answer": read_ans[:120], "canon_hit": read_hit,
                         "paraphrase": para_rows,
                         "shuffle_answer": sh_ans[:120], "shuffle_hit": sh_hit})
        if (idx + 1) % 10 == 0:
            print(f"  [{tag}] {idx+1}/{len(cases)} canon={stats['canon_hits']} "
                  f"para={stats['para_hits']}/{stats['para_instances']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    n = stats["n"]
    out = {
        "model": "gpt2-small",
        "slice_tag": tag,
        "n_routed_edits": n,
        "key_material": "sha256(subject + '|' + relation)[:16]",
        "memory_lookup_rate": round(stats["lookup_hits"] / n, 4),
        "memory_baseline_recall": round(stats["base_hits"] / n, 4),
        "memory_canonical_recall": round(stats["canon_hits"] / n, 4),
        "memory_paraphrase_recall": (round(stats["para_hits"] / stats["para_instances"], 4)
                                     if stats["para_instances"] else None),
        "memory_paraphrase_recall_case_any": (round(stats["para_any"] / n, 4) if n else None),
        "memory_paraphrase_denominator": stats["para_instances"],
        "memory_shuffled_recall": (round(stats["shuffle_hits"] / stats["shuffle_checked"], 4)
                                   if stats["shuffle_checked"] else None),
        "memory_shuffled_n": stats["shuffle_checked"],
        "logp_read_delta": (round(stats["lp_delta_sum"] / stats["lp_delta_n"], 4)
                            if stats["lp_delta_n"] else None),
        "note_echo_rate": round(stats["canon_note_echo"] / n, 4),
        "seconds": round(time.time() - t0, 1),
        "denominator_note": "all recall columns use n_routed_edits as denominator unless a "
                            "dedicated denominator field is given",
        "per_case": per_case,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default="0")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "w6_rev3", "recall"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    summary = {}
    for off in [int(x) for x in args.offsets.split(",") if x.strip()]:
        cases = parse_records(DATA, off, args.n)
        print(f"=== slice offset={off} n={len(cases)} ===", flush=True)
        res = run_slice(model, tok, cases, args.max_new, f"o{off}")
        path = os.path.join(args.out_dir, f"recall_gpt2small_o{off}_n{len(cases)}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        summary[f"o{off}"] = {k: v for k, v in res.items() if k != "per_case"}
        print(json.dumps(summary[f"o{off}"], ensure_ascii=False), flush=True)

    with open(os.path.join(args.out_dir, "recall_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
