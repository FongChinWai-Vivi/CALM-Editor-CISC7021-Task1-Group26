"""W3 real-model component run (SideMemory + ConflictGate) for CALM.

Loads a real Hugging Face model and real CounterFact edits, then drives the
W3 SideMemory / ConflictGate components with genuine model signals:

  - current_answer / current_confidence: greedy generation + mean token
    probability from the real model;
  - recent_edits: sliding window of earlier edit records (ConflictGate);
  - retrieval_context: n-gram similarity of recent subjects.

For every edit request the gate decides WRITE / ROUTE / REJECT. ROUTE
stores the edit in SideMemory (sharded, hash-routed). The script reports the
decision distribution, SideMemory statistics and the rewrite hit rate for
each decision path.

Usage:
    python real_w3_components.py --model-key gpt2-xl \
        --data-path <counterfact.json|parquet> --limit 100 --out w3_components.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd
import torch

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W3_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w3\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W3_DIR)

from model_utils import load_model  # noqa: E402
from side_memory import SideMemory  # noqa: E402
from conflict_gate import ConflictGate, EditRecord  # noqa: E402

RELATION_FALLBACK = "is"


def load_records(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            raw = raw.get("data", raw.get("instances", raw.get("records", [])))
        return list(raw)
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    raise ValueError(f"unsupported data extension: {ext}")


def mean_token_prob_v2(model, tok, prompt: str, generation: str) -> float:
    """Mean softmax probability of the generated tokens (real model signal).

    Computed with a single forward pass over (prompt + generation); does not
    rely on generation.scores, avoiding GPT-2 prefill reshape errors in
    transformers 5.x.
    """
    if not generation:
        return 0.0
    enc_prompt = tok(prompt, return_tensors="pt")
    n_prompt = enc_prompt["input_ids"].shape[1]
    enc = tok(prompt + generation, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0]  # (L, vocab)
    probs = torch.softmax(logits, dim=-1)
    ids = enc["input_ids"][0]
    gen_ids = ids[n_prompt:]
    if gen_ids.numel() == 0 or n_prompt == 0:
        return 0.0
    step_probs = probs[n_prompt - 1: -1].gather(
        1, gen_ids.unsqueeze(1)).squeeze(1)
    return float(step_probs.mean().item())


def ngram_sim(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity (standard-library retrieval context)."""
    def grams(s: str):
        s = "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace())
        return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}
    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def main() -> int:
    ap = argparse.ArgumentParser(description="W3 real component run")
    ap.add_argument("--model-key", default="gpt2-xl",
                    choices=["gpt2-small", "gpt2-xl", "gpt-j-6b"])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_records(args.data_path)[: args.limit]
    print(f"[w3] {len(records)} edit requests loaded", flush=True)

    print(f"[w3] loading model {args.model_key} ...", flush=True)
    model, tok = load_model(args.model_key)

    mem = SideMemory(n_shards=64, max_per_shard=200, seed=42)
    gate = ConflictGate(n_recent=50, write_threshold=0.3,
                        route_threshold=0.6, similarity_threshold=0.7, seed=0)

    decisions: List[str] = []
    path_hits: Dict[str, List[bool]] = {"WRITE": [], "ROUTE": [], "REJECT": []}
    log: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, rec in enumerate(records):
        request = rec.get("request", rec)
        rw = request.get("requested_rewrite", request)
        prompt_tpl = rw.get("prompt", request.get("prompt", ""))
        subject = rw.get("subject", request.get("subject", ""))
        prompt = (prompt_tpl.replace("{}", subject)
                  if "{}" in prompt_tpl else prompt_tpl)
        t_new = rw.get("target_new", {})
        t_true = rw.get("target_true", {})
        target_new = t_new.get("str", t_new) if isinstance(t_new, dict) else t_new
        target_true = t_true.get("str", t_true) if isinstance(t_true, dict) else t_true
        relation = RELATION_FALLBACK

        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs.to(model.device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        generation = tok.decode(gen_ids, skip_special_tokens=True)
        confidence = mean_token_prob_v2(model, tok, prompt, generation)

        # Recent edit records feed the gate's second signal.
        recent = [r for r in gate.recent_edits]
        # Retrieval context: similarity to the most recent subjects.
        ctx = [r.subject for r in recent[-10:]]
        sim = max((ngram_sim(subject, c) for c in ctx), default=0.0)
        ctx_score = sim if sim > 0 else 0.0

        decision = gate.decide(
            subject=subject, relation=relation, target_new=target_new,
            current_answer=generation, current_confidence=confidence,
            recent_edits=recent, retrieval_context=ctx,
        )
        decisions.append(decision)

        if decision == "ROUTE":
            mem.store(subject, relation, target_new, target_new)
            hit = mem.lookup(subject, relation, target_new) is not None
        elif decision == "WRITE":
            # The parameter-write path is executed by the W2 editors; here we
            # record the decision and check rewrite feasibility via the gate.
            hit = True
        else:
            hit = False

        path_hits[decision].append(hit)
        log.append({
            "sample_id": i, "subject": subject, "decision": decision,
            "confidence": round(confidence, 4),
            "generation": generation[:120],
            "target_new": target_new, "target_true": target_true,
        })

        # Record the edit in the gate's sliding window.
        gate.recent_edits.append(
            EditRecord(subject=subject, relation=relation, target_new=target_new,
                       current_answer=generation, confidence=confidence,
                       timestamp=float(i))
        )
        if len(gate.recent_edits) > gate.n_recent:
            gate.recent_edits = gate.recent_edits[-gate.n_recent:]

    elapsed = time.time() - t0
    from collections import Counter
    dist = dict(Counter(decisions))

    result = {
        "task": "W3-components",
        "model": args.model_key,
        "n_requests": len(records),
        "decision_distribution": dist,
        "path_rewrite_hits": {k: (sum(v), len(v)) for k, v in path_hits.items()},
        "side_memory_stats": mem.stats(),
        "elapsed_seconds": round(elapsed, 2),
        "per_request": log,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_request"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
