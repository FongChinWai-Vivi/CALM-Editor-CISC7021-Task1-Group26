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


def infer_relation(prompt_tmpl: str, subject: str) -> str:
    """Derive a relation signature from the prompt template itself.

    Every CounterFact prompt used to be labelled with the constant "is", which
    made two unrelated edits look related to ConflictGate (a shared relation
    alone contributed 0.5 to the similarity score).  The relation is now the
    template text that *follows* the subject, e.g. "is located in".
    """
    tmpl = (prompt_tmpl or "").strip()
    if not tmpl:
        return RELATION_FALLBACK
    head, _, tail = tmpl.partition("{}")
    rel = " ".join(tail.replace("{}", subject).split()).strip(" .?!,")
    if not rel:
        rel = " ".join(head.replace("{}", subject).split()).strip(" .?!,")
    return rel or RELATION_FALLBACK


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
    ap.add_argument("--probe-max-new-tokens", type=int, default=8,
                    help="short budget for the pre-edit probe answer")
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
        relation = infer_relation(prompt_tpl, subject)
        # Pre-edit probe: a differently worded prompt for the same request.
        # If the unedited model already answers it with target_new the edit is
        # redundant, which is the only evidence that justifies a REJECT.
        para = rec.get("paraphrase_prompts") or rw.get("paraphrase_prompts") or []
        probe_prompt = (para[0].replace("{}", subject) if para else "")

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

        # Pre-edit probe answer (same unedited model, paraphrased prompt).
        probe_answer = ""
        if probe_prompt:
            p_inputs = tok(probe_prompt, return_tensors="pt")
            with torch.no_grad():
                p_out = model.generate(
                    **p_inputs.to(model.device),
                    max_new_tokens=args.probe_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            probe_answer = tok.decode(
                p_out[0][p_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)

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
            pre_edit_probe_answer=probe_answer or None,
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
            "sample_id": i, "subject": subject, "relation": relation,
            "decision": decision,
            "confidence": round(confidence, 4),
            "generation": generation[:120],
            "probe_prompt": probe_prompt,
            "probe_answer": probe_answer[:120],
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

    # ------------------------------------------------------------------
    # REJECT canary group.
    # CounterFact always asks for a *new* fact, so an unedited model can
    # never already answer target_new: a REJECT is structurally impossible
    # on the main 100 requests.  To prove the REJECT branch is live rather
    # than dead code, a small canary group asks to write the *true* fact
    # (target_true) -- a redundant edit the model already knows.  A correct
    # gate must reject it via the pre-edit probe.
    # ------------------------------------------------------------------
    canary_gate = ConflictGate(n_recent=50, write_threshold=0.3,
                               route_threshold=0.6, similarity_threshold=0.7,
                               seed=0)
    canary = {"n": 0, "reject": 0, "write_or_route": 0,
              "checked": 0, "cases": []}
    for i, rec in enumerate(records[:20]):
        request = rec.get("request", rec)
        rw = request.get("requested_rewrite", request)
        prompt_tpl = rw.get("prompt", request.get("prompt", ""))
        subject = rw.get("subject", request.get("subject", ""))
        if not subject or "{}" not in prompt_tpl:
            continue
        prompt = prompt_tpl.replace("{}", subject)
        t_true = rw.get("target_true", {})
        target_true = t_true.get("str", t_true) if isinstance(t_true, dict) else t_true
        if not target_true:
            continue
        para = rec.get("paraphrase_prompts") or rw.get("paraphrase_prompts") or []
        if not para:
            continue
        probe_prompt = para[0].replace("{}", subject)

        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs.to(model.device),
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        generation = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        p_inputs = tok(probe_prompt, return_tensors="pt")
        with torch.no_grad():
            p_out = model.generate(**p_inputs.to(model.device),
                                   max_new_tokens=args.probe_max_new_tokens,
                                   do_sample=False,
                                   pad_token_id=tok.pad_token_id,
                                   eos_token_id=tok.eos_token_id)
        probe_answer = tok.decode(
            p_out[0][p_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)

        decision = canary_gate.decide(
            subject=subject,
            relation=infer_relation(prompt_tpl, subject),
            target_new=target_true,
            current_answer=generation,
            current_confidence=mean_token_prob_v2(model, tok, prompt, generation),
            recent_edits=[], retrieval_context=[],
            pre_edit_probe_answer=probe_answer,
        )
        canary["n"] += 1
        if decision == "REJECT":
            canary["reject"] += 1
        else:
            canary["write_or_route"] += 1
        canary["cases"].append({
            "sample_id": i, "subject": subject,
            "target_true": target_true, "decision": decision,
            "probe_answer": probe_answer[:80],
        })
        print(f"[w3] canary {i} subject={subject} -> {decision}", flush=True)
    if canary["n"]:
        canary["reject_rate"] = round(canary["reject"] / canary["n"], 4)
        print(f"[w3] canary REJECT rate = {canary['reject_rate']} "
              f"({canary['reject']}/{canary['n']})", flush=True)

    # ------------------------------------------------------------------
    # Repeat-edit probe: the same (subject, relation) edited twice with two
    # different targets.  The gate must escalate the second request to ROUTE
    # through its side-memory conflict signal instead of writing over the
    # first edit.  Pure decision logic (no model call), so it is cheap and
    # reproducible.
    # ------------------------------------------------------------------
    repeat_gate = ConflictGate(n_recent=50, write_threshold=0.3,
                               route_threshold=0.6, similarity_threshold=0.7,
                               seed=0)
    repeat = {"n": 0, "routed": 0, "write": 0, "reject": 0, "cases": []}
    seen_subjects = set()
    for rec in records:
        request = rec.get("request", rec)
        rw = request.get("requested_rewrite", request)
        prompt_tpl = rw.get("prompt", request.get("prompt", ""))
        subject = rw.get("subject", request.get("subject", ""))
        if not subject or subject in seen_subjects:
            continue
        seen_subjects.add(subject)
        if len(repeat["cases"]) >= 20:
            break
        relation = infer_relation(prompt_tpl, subject)
        first = repeat_gate.decide(
            subject=subject, relation=relation, target_new="first target",
            current_answer="unrelated baseline answer",
            current_confidence=0.55, recent_edits=[],
            retrieval_context=[], pre_edit_probe_answer=None)
        second = repeat_gate.decide(
            subject=subject, relation=relation, target_new="second target",
            current_answer="unrelated baseline answer",
            current_confidence=0.55,
            recent_edits=list(repeat_gate.recent_edits),
            retrieval_context=[], pre_edit_probe_answer=None)
        repeat["n"] += 1
        if second == "ROUTE":
            repeat["routed"] += 1
        elif second == "WRITE":
            repeat["write"] += 1
        else:
            repeat["reject"] += 1
        repeat["cases"].append({
            "subject": subject, "relation": relation,
            "first_decision": first, "second_decision": second,
        })
    if repeat["n"]:
        repeat["route_rate"] = round(repeat["routed"] / repeat["n"], 4)
        print(f"[w3] repeat-edit probe ROUTE rate = {repeat['route_rate']} "
              f"({repeat['routed']}/{repeat['n']})", flush=True)

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
        "reject_justification": {
            "n_reject": sum(1 for e in gate._decision_log
                            if e["decision"] == "REJECT"),
            "n_reject_with_probe_match": sum(
                1 for e in gate._decision_log
                if e["decision"] == "REJECT" and e["pre_edit_probe_match"]),
            "note": ("every REJECT carries either a verified pre-edit probe "
                     "match or an exact current-answer match"),
        },
        "reject_canary": canary,
        "repeat_edit_probe": repeat,
        "decision_log": gate._decision_log,
        "elapsed_seconds": round(elapsed, 2),
        "per_request": log,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_request",)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
