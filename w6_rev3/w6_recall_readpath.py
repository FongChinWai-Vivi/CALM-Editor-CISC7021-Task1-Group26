"""w6_recall_readpath.py -- Rev4 read-path harness (template sweep + strict metric).

Why this file exists.  The Rev3 recall harness scored a note "delivered" as soon
as the target string appeared anywhere in the continuation, and it only ever
used one note template.  Both choices flatter the read path: a rambling answer
that happens to mention the entity counts the same as a clean one, and the
template is a free hyper-parameter that was never swept.  This harness keeps the
same key material and store (so numbers stay comparable with
``w6_recall_gpt2small.py``) and adds:

  * ``--templates``  prefix / instruct / qa / fewshot / repeat, scored side by
    side on the same cases, so the read path is chosen by measurement;
  * a strict column: the target has to appear within the first ``--strict-words``
    words of the answer, not merely somewhere in it;
  * retrieval vs delivery split, because lookup rate is 1.0 by construction and
    the interesting quantity is whether the LM actually hands the note over;
  * ``--model-dir`` so the same harness can be pointed at a bigger backbone.

Nothing here touches the parameter channel or the stream runner; it is a pure
read-path measurement.
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
MODEL_DIR = os.path.join(REPO, "EasyEdit", "hugging_cache", "gpt2-small")
DATA = os.path.join(REPO, "EasyEdit", "data", "counterfact.json")


def infer_relation(prompt_tmpl: str, subject: str) -> str:
    tmpl = (prompt_tmpl or "").strip()
    if not tmpl:
        return RELATION_FALLBACK
    head, _, tail = tmpl.partition("{}")
    rel = " ".join(tail.replace("{}", subject).split()).strip(" .?!,")
    if not rel:
        rel = " ".join(head.replace("{}", subject).split()).strip(" .?!,")
    return rel or RELATION_FALLBACK


def _norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def contains(answer: str, target: str) -> bool:
    a, t = _norm(answer), _norm(target)
    return bool(t) and t in a


def contains_strict(answer: str, target: str, words: int) -> bool:
    """Target inside the first ``words`` words -- no credit for rambling."""
    toks = _norm(answer).split()
    t = _norm(target)
    return bool(t) and t in " ".join(toks[:words])


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


# --------------------------------------------------------------------------
# Note templates.  ``prefix`` is the Rev3 template (build_memory_note), the
# rest are candidates.  keep them side by side so the sweep is one number per
# template rather than a memory of what each one used to say.
# --------------------------------------------------------------------------
FEWSHOT_HEAD = (
    "Q: what is the capital city of Zorba?\n"
    "Note: Zorba is Tarn.\n"
    "Answer: Tarn\n"
    "Q: who wrote Night Whispers?\n"
    "Note: Night Whispers is by Elias Moore.\n"
    "Answer: Elias Moore\n"
)


def render_note(template: str, subject: str, relation: str, note: str, prompt: str) -> str:
    tmpl = (template or "prefix").lower()
    if tmpl == "prefix":
        return build_memory_note(subject, relation, note) + prompt
    if tmpl == "instruct":
        return f"Answer with {note}.\n" + prompt
    if tmpl == "qa":
        return f"Q: what is the {relation} of {subject}?\nA: {note}\n" + prompt
    if tmpl == "fewshot":
        return (FEWSHOT_HEAD
                + f"Q: {prompt.strip()}\n"
                + f"Note: {subject} is {note}.\n"
                + "Answer:")
    if tmpl == "repeat":
        return (f"Note: {subject} is {note}. Repeat: {note}.\n" + prompt)
    if tmpl == "fewshot_prefix":
        # fewshot wins on paraphrase but loses a little canonical delivery;
        # this variant keeps the prefix cue as well to see whether the two
        # gains are additive or whether they compete for the same slot.
        return (FEWSHOT_HEAD
                + build_memory_note(subject, relation, note)
                + prompt)
    raise SystemExit(f"unknown template {template}")


@torch.no_grad()
def gen(model, tok, prompt: str, max_new: int, max_len: int) -> str:
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=max_len).input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True).strip()


@torch.no_grad()
def logp_target(model, tok, prompt: str, target: str, max_len: int) -> float:
    pid = tok(prompt, return_tensors="pt", truncation=True,
              max_length=max_len).input_ids.to(model.device)
    tid = tok(" " + target, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    ids = torch.cat([pid, tid], dim=1)
    logits = model(input_ids=ids).logits[0].float()
    start = pid.size(1)
    acc = 0.0
    for j in range(tid.size(1)):
        lp = torch.log_softmax(logits[start + j - 1], dim=-1)
        acc += float(lp[tid[0, j]])
    return acc / max(1, tid.size(1))


def run_slice(model, tok, cases, templates, max_new, max_len, strict_words, tag):
    mem = SideMemory(n_shards=64, max_per_shard=500, seed=42)
    for c in cases:
        mem.store(c["subject"], c["relation"], c["target_new"])

    per_t: Dict[str, Dict[str, float]] = {
        t: dict(lookup_hits=0, base_hits=0, canon_hits=0, canon_strict=0,
                para_hits=0, para_instances=0, para_any=0, para_strict=0,
                lp_delta_sum=0.0, lp_n=0, echo=0)
        for t in templates
    }
    per_case = []
    t0 = time.time()

    for idx, c in enumerate(cases):
        note = mem.lookup(c["subject"], c["relation"])
        base_ans = gen(model, tok, c["prompt"], max_new, max_len)
        base_hit = contains(base_ans, c["target_new"])
        row = {"subject": c["subject"], "relation": c["relation"],
               "target_new": c["target_new"], "note": note,
               "base_answer": base_ans[:120], "base_hit": base_hit, "templates": {}}

        for t in templates:
            st = per_t[t]
            st["base_hits"] += int(base_hit)
            st["lookup_hits"] += int(note is not None)
            rec = {"canon_answer": "", "canon_hit": False, "canon_strict": False,
                   "paraphrase": [], "logp_delta": None, "echo": False}
            if note is not None:
                pref = render_note(t, c["subject"], c["relation"], note, c["prompt"])
                ans = gen(model, tok, pref, max_new, max_len)
                hit = contains(ans, c["target_new"])
                strict = contains_strict(ans, c["target_new"], strict_words)
                echo = contains(ans, note)
                st["canon_hits"] += int(hit)
                st["canon_strict"] += int(strict)
                st["echo"] += int(echo)
                lp_with = logp_target(model, tok, pref, c["target_new"], max_len)
                lp_without = logp_target(model, tok, c["prompt"], c["target_new"], max_len)
                st["lp_delta_sum"] += (lp_with - lp_without)
                st["lp_n"] += 1
                rec.update(canon_answer=ans[:120], canon_hit=hit,
                           canon_strict=strict, echo=echo,
                           logp_delta=round(lp_with - lp_without, 4))
                para_rows = []
                for p in c["paraphrase"]:
                    pa = gen(model, tok,
                             render_note(t, c["subject"], c["relation"], note, p),
                             max_new, max_len)
                    ph = contains(pa, c["target_new"])
                    ps = contains_strict(pa, c["target_new"], strict_words)
                    para_rows.append({"prompt": p, "answer": pa[:120],
                                      "hit": ph, "strict": ps})
                    st["para_hits"] += int(ph)
                    st["para_strict"] += int(ps)
                    st["para_instances"] += 1
                st["para_any"] += int(any(r["hit"] for r in para_rows))
                rec["paraphrase"] = para_rows
            row["templates"][t] = rec
        per_case.append(row)
        if (idx + 1) % 10 == 0:
            brief = " ".join(f"{t}:{per_t[t]['canon_hits']}/{per_t[t]['para_hits']}"
                             for t in templates)
            print(f"  [{tag}] {idx+1}/{len(cases)} {brief} ({time.time()-t0:.0f}s)",
                  flush=True)

    n = len(cases)
    out = {"slice_tag": tag, "n_routed_edits": n, "templates": {}, "per_case": per_case,
           "strict_words": strict_words,
           "denominator_note": "recall columns use n_routed_edits; paraphrase columns "
                               "use template-level instance counts"}
    for t in templates:
        st = per_t[t]
        out["templates"][t] = {
            "retrieval_rate": round(st["lookup_hits"] / n, 4),
            "baseline_recall": round(st["base_hits"] / n, 4),
            "canonical_delivery": round(st["canon_hits"] / n, 4),
            "canonical_delivery_strict": round(st["canon_strict"] / n, 4),
            "paraphrase_delivery": (round(st["para_hits"] / st["para_instances"], 4)
                                    if st["para_instances"] else None),
            "paraphrase_delivery_strict": (round(st["para_strict"] / st["para_instances"], 4)
                                           if st["para_instances"] else None),
            "paraphrase_case_any": round(st["para_any"] / n, 4) if n else None,
            "paraphrase_instances": st["para_instances"],
            "note_echo_rate": round(st["echo"] / n, 4),
            "logp_read_delta": (round(st["lp_delta_sum"] / st["lp_n"], 4)
                                if st["lp_n"] else None),
        }
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default="0")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--strict-words", type=int, default=4)
    ap.add_argument("--templates", default="prefix,instruct,qa,fewshot,repeat")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu -- cpu keeps a 6 GB backbone off a full GPU")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "w6_rev3", "recall4"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model.to(device).eval()
    model_name = os.path.basename(args.model_dir.rstrip("\\/"))

    summary = {"model": model_name, "templates": templates, "slices": {}}
    for off in [int(x) for x in args.offsets.split(",") if x.strip()]:
        cases = parse_records(DATA, off, args.n)
        tag = args.tag or f"o{off}"
        print(f"=== {model_name} slice offset={off} n={len(cases)} ===", flush=True)
        res = run_slice(model, tok, cases, templates, args.max_new, args.max_len,
                        args.strict_words, tag)
        res["model"] = model_name
        path = os.path.join(args.out_dir, f"readpath_{model_name}_o{off}_n{len(cases)}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        summary["slices"][f"o{off}"] = {k: v for k, v in res.items()
                                        if k not in ("per_case",)}
        print(json.dumps({t: res["templates"][t] for t in templates},
                         ensure_ascii=False), flush=True)

    with open(os.path.join(args.out_dir, f"readpath_summary_{model_name}{args.tag}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
