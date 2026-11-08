"""W3 memory-channel readout (SideMemory answer path, gpt2-xl).

Why this script exists
----------------------
The first SideMemory build keyed entries on ``subject | relation | target_new``
and ``lookup()`` required ``target_new`` as an argument.  The caller therefore
had to know the answer before asking, so every recall number produced by that
path was 1.0 *by construction* -- it measured an index hit, not memory recall.

This script measures the channel the way the quality handbook requires:

  1. key   : ``(subject, relation)`` only; ``target_new`` is the value;
  2. read  : ``lookup(subject, relation)`` -> note -> inject before generation;
  3. recall: split canonical / paraphrase, denominator = routed edits;
  4. control: a cue-only baseline on the same edits, and shuffled keys that
             must yield recall ~ 0;
  5. label : the metric names carry the denominator, nothing is called
             "memory accuracy" unless it was measured on the answer path.

Outputs (JSON): memory_n, memory_canonical_recall, memory_paraphrase_recall,
memory_shuffled_recall, plus a diagnostic ``legacy_constructive_recall`` that
reproduces the old (invalid) keying for the record.

Usage:
    python w3_memory_channel.py --model-key gpt2-xl \
        --data-path <counterfact.json> --n 20 --out w3_memory_channel_gpt2xl.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W3_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w3\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W3_DIR)

from model_utils import load_model  # noqa: E402
from side_memory import SideMemory, build_memory_note  # noqa: E402
from real_w3_components import infer_relation  # noqa: E402


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else data.get("data", [])


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def contains(answer: str, target: str) -> bool:
    """Containment match, the metric used by the W3 probe scripts."""
    a, t = _norm(answer), _norm(target)
    return bool(t) and t in a


def generate_text(model, tok, prompt: str, max_new_tokens: int) -> str:
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def parse_case(rec: Dict[str, Any]) -> Dict[str, Any]:
    request = rec.get("request", rec)
    rw = request.get("requested_rewrite", request)
    prompt_tpl = rw.get("prompt", request.get("prompt", ""))
    subject = rw.get("subject", request.get("subject", ""))
    t_new = rw.get("target_new", {})
    target_new = t_new.get("str", t_new) if isinstance(t_new, dict) else t_new
    para = rec.get("paraphrase_prompts") or rw.get("paraphrase_prompts") or []
    prompt = prompt_tpl.replace("{}", subject) if "{}" in prompt_tpl else prompt_tpl
    return {
        "subject": subject,
        "relation": infer_relation(prompt_tpl, subject),
        "target_new": target_new,
        "prompt": prompt,
        "paraphrase": para[0] if para else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="W3 memory-channel readout")
    ap.add_argument("--model-key", default="gpt2-xl",
                    choices=["gpt2-small", "gpt2-xl", "gpt-j-6b"])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = [parse_case(r) for r in load_records(args.data_path)[: args.n]]
    cases = [c for c in cases if c["subject"] and c["target_new"] and c["prompt"]]
    print(f"[memory] {len(cases)} routed edits loaded", flush=True)

    model, tok = load_model(args.model_key)

    # ---- write path: key = (subject, relation); answer = value ----
    mem = SideMemory(n_shards=64, max_per_shard=200, seed=42)
    for c in cases:
        mem.store(c["subject"], c["relation"], c["target_new"])
    print(f"[memory] stored={mem.stats()['total_stored']}", flush=True)

    # ---- legacy keying, kept only as a diagnostic reproduction ----
    legacy_hits = 0
    for c in cases:
        legacy_key = f"{c['subject']}|{c['relation']}|{c['target_new']}"
        legacy_hits += int(legacy_key.split("|")[2] == c["target_new"])

    canonical_hits = 0
    baseline_hits = 0
    paraphrase_hits = 0
    paraphrase_n = 0
    shuffled_hits = 0
    per_case: List[Dict[str, Any]] = []

    for idx, c in enumerate(cases):
        note = mem.lookup(c["subject"], c["relation"])

        # Baseline: the same cue with no note injected. Without this row the
        # injected number cannot be attributed to memory rather than to the cue.
        baseline_answer = generate_text(model, tok, c["prompt"],
                                        args.max_new_tokens)
        baseline_hit = contains(baseline_answer, c["target_new"])
        baseline_hits += int(baseline_hit)

        read_hit = False
        read_answer = ""
        if note is not None:
            read_prompt = build_memory_note(c["subject"], c["relation"], note) + c["prompt"]
            read_answer = generate_text(model, tok, read_prompt, args.max_new_tokens)
            read_hit = contains(read_answer, c["target_new"])
        canonical_hits += int(read_hit)

        para_hit = None
        para_answer = ""
        if c["paraphrase"] and note is not None:
            para_prompt = (build_memory_note(c["subject"], c["relation"], note)
                           + c["paraphrase"])
            para_answer = generate_text(model, tok, para_prompt, args.max_new_tokens)
            para_hit = contains(para_answer, c["target_new"])
            paraphrase_hits += int(para_hit)
            paraphrase_n += 1

        # Shuffled-key control: inject the note of a different edit.
        other = cases[(idx + 1) % len(cases)]
        wrong_note = mem.lookup(other["subject"], other["relation"])
        shuffle_hit = False
        shuffle_answer = ""
        if wrong_note is not None and other["subject"] != c["subject"]:
            shuffle_prompt = (build_memory_note(other["subject"], other["relation"],
                                                wrong_note) + c["prompt"])
            shuffle_answer = generate_text(model, tok, shuffle_prompt,
                                           args.max_new_tokens)
            shuffle_hit = contains(shuffle_answer, c["target_new"])
        shuffled_hits += int(shuffle_hit)

        per_case.append({
            "subject": c["subject"], "relation": c["relation"],
            "target_new": c["target_new"], "note_found": note is not None,
            "baseline_answer": baseline_answer[:120],
            "canonical_answer": read_answer[:120], "canonical_hit": read_hit,
            "paraphrase_answer": para_answer[:120], "paraphrase_hit": para_hit,
            "shuffled_answer": shuffle_answer[:120],
            "shuffled_hit": shuffle_hit,
        })
        print(f"[memory] {idx} {c['subject']} baseline={baseline_hit} "
              f"canonical={read_hit} "
              f"paraphrase={para_hit} shuffled={shuffle_hit}", flush=True)

    memory_n = len(cases)
    result = {
        "task": "W3-memory-channel",
        "model": args.model_key,
        "n_routed_edits": memory_n,
        "key_material": "sha256(subject + '|' + relation)[:16]",
        "memory_n": memory_n,
        "memory_baseline_recall": round(baseline_hits / memory_n, 4) if memory_n else 0.0,
        "memory_canonical_recall": round(canonical_hits / memory_n, 4) if memory_n else 0.0,
        "memory_paraphrase_recall": (round(paraphrase_hits / paraphrase_n, 4)
                                     if paraphrase_n else None),
        "memory_paraphrase_denominator": paraphrase_n,
        "memory_shuffled_recall": round(shuffled_hits / memory_n, 4) if memory_n else 0.0,
        "legacy_constructive_recall": round(legacy_hits / memory_n, 4) if memory_n else 0.0,
        "legacy_note": ("old keying stored target_new inside the key, so this number "
                        "is an index hit by construction and must not be quoted as "
                        "memory accuracy"),
        "denominator_note": ("baseline, canonical and shuffled recall use the routed "
                             "edits as denominator; paraphrase recall uses the subset "
                             "that carries a paraphrase prompt"),
        "per_case": per_case,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_case"},
                     ensure_ascii=False, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
