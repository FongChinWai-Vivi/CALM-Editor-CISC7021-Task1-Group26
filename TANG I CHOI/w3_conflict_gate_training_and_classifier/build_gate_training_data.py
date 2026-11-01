"""build_gate_training_data.py -- build the W3 conflict-gate training data.

W3 (Tang I Choi, MC631517).  Owner: TANG I CHOI.

What this script produces
-------------------------
Three gate datasets in ``gate_datasets/``:

1. ``gate_train.jsonl`` / ``gate_dev.jsonl`` / ``gate_test.jsonl``
   CounterFact-derived gate situations.  Each CounterFact edit is turned into
   four *situations* that differ only in the three gate signals (current answer,
   side-memory state, retrieved context), so the derived action label is a
   property of the situation rather than of the edit:

   ==========================  ===========================================  ============
   situation                   construction                                 label4
   ==========================  ===========================================  ============
   ``true_fact``               target = target_true, no prior edit, no ctx   no-conflict
   ``false_fact``              target = a different value of the same        pre-training-
                               relation (contradicts model knowledge)        conflict
   ``repeat_edit``             same (subject, relation) already edited with  prior-edit-
                               another value                                 conflict
   ``context_override``        retrieved context asserts another value       context-
                                                                             conflict
   ==========================  ===========================================  ============

2. ``gate_test_pilot100.jsonl``
   The 100 hand-labelled W1 pilot edits
   (``conflict annotation manual + 100 pilot annotations/data/conflict_pilot_100.jsonl``)
   turned into gate samples.  Their four-way labels are *human* labels, so this
   file is the honest held-out test set: training labels are derived from the
   annotation manual's decision rules, while this test set is not.

Label -> action mapping (documented in ``docs/conflict_annotation_manual.md``
and restated in ``w3_report.md``):

    no-conflict            + model already answers target  -> REJECT
    no-conflict            + model cannot answer target     -> WRITE
    pre-training-conflict                                  -> REJECT
    prior-edit-conflict                                    -> ROUTE
    context-conflict                                       -> ROUTE

The current answer / confidence of every prompt is measured with the real CALM
verification model (``gpt2-xl``, same checkpoint as W3/W4), so feature A of the
gate is measured and not simulated.  Answers are cached in
``gate_datasets/model_answer_cache.json``.

Usage::

    python "TANG I CHOI/w3_conflict_gate_training_and_classifier/build_gate_training_data.py"
    python .../build_gate_training_data.py --n-train-bases 700 --n-eval-bases 180
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EASYEDIT = os.path.join(REPO, "EasyEdit")
COUNTERFACT = os.path.join(EASYEDIT, "data", "counterfact.json")
PILOT = os.path.join(
    REPO, "conflict annotation manual + 100 pilot annotations", "data",
    "conflict_pilot_100.jsonl")
MODEL_DIR = os.path.join(EASYEDIT, "hugging_cache", "gpt2-xl")

sys.path.insert(0, HERE)

ACTIONS = ("WRITE", "ROUTE", "REJECT")
LABELS4 = ("no-conflict", "pre-training-conflict", "prior-edit-conflict",
           "context-conflict")


# --------------------------------------------------------------------------- #
# Model answers
# --------------------------------------------------------------------------- #
class Answerer:
    """Greedy-decode the model's current answer + confidence for a prompt."""

    def __init__(self, model_dir: str, device: Optional[str] = None,
                 max_new_tokens: int = 5, batch_size: int = 24):
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.torch = torch
        self.tok = GPT2Tokenizer.from_pretrained(model_dir)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # decoder-only generation with padded batches needs left padding
        self.tok.padding_side = "left"
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model.to(device).eval()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def answer_many(self, prompts: List[str]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=64)
            ids = enc.input_ids.to(self.device)
            mask = enc.attention_mask.to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    ids, attention_mask=mask,
                    max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                    output_scores=True, return_dict_in_generate=True,
                )
                logits = self.model(gen.sequences).logits
            new_tokens = gen.sequences[:, ids.size(1):]
            # mean probability of the chosen tokens (same definition as
            # CALMEditor._current_answer_and_confidence)
            start_pos = ids.size(1) - 1
            for row, prompt in enumerate(chunk):
                n_new = int((new_tokens[row] != self.tok.pad_token_id).sum().item())
                n_new = max(n_new, 1)
                lp = logits[row, start_pos:start_pos + n_new]
                probs = lp.softmax(dim=-1)
                chosen = probs.gather(
                    1, new_tokens[row, :n_new].unsqueeze(-1)).squeeze(-1)
                text = self.tok.decode(new_tokens[row, :n_new],
                                       skip_special_tokens=True)
                out[prompt] = {"answer": _first_clause(text),
                               "confidence": float(chosen.mean().item())}
            print(f"  [answers] {min(start + len(chunk), len(prompts))}"
                  f"/{len(prompts)}", flush=True)
        return out


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def _first_clause(text: str) -> str:
    """Keep the first clause of a greedy continuation (gpt2 often rambles)."""
    cleaned = str(text or "").strip()
    for sep in (".", "\n", ";", ",", "?", "!"):
        idx = cleaned.find(sep)
        if idx > 0:
            cleaned = cleaned[:idx]
    return cleaned.strip()


def _model_knows(answer: str, target: str, confidence: float) -> bool:
    """Does the model already produce the target with reasonable confidence?"""
    a, t = _norm(answer), _norm(target)
    if not t:
        return False
    return (t in a) and confidence >= 0.5


def action_from_label4(label4: str, model_knows_target: bool) -> str:
    """The manual's label -> action mapping (single source of truth)."""
    if label4 == "no-conflict":
        return "REJECT" if model_knows_target else "WRITE"
    if label4 == "pre-training-conflict":
        return "REJECT"
    if label4 in ("prior-edit-conflict", "context-conflict"):
        return "ROUTE"
    raise ValueError(f"unknown label4: {label4}")


# --------------------------------------------------------------------------- #
# CounterFact -> gate situations
# --------------------------------------------------------------------------- #
def load_counterfact_bases(max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    records = _load_json(COUNTERFACT)
    if max_records:
        records = records[:max_records]
    bases, seen = [], set()
    for rec in records:
        rr = rec.get("requested_rewrite") or {}
        subject = rr.get("subject")
        if not subject or subject in seen:
            continue
        seen.add(subject)
        prompt_tpl = rr.get("prompt", "")
        bases.append({
            "case_id": rec.get("case_id"),
            "subject": subject,
            "relation": rr.get("relation_id") or "fact",
            "prompt": prompt_tpl.replace("{}", str(subject)),
            "target_true": (rr.get("target_true") or {}).get("str", "")
            if isinstance(rr.get("target_true"), dict) else str(rr.get("target_true")),
        })
    return bases


def other_values_by_relation(bases: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    pool: Dict[str, set] = defaultdict(set)
    for b in bases:
        if b["target_true"]:
            pool[b["relation"]].add(b["target_true"])
    return {k: sorted(v) for k, v in pool.items()}


def build_counterfact_samples(
    bases: List[Dict[str, Any]],
    answers: Dict[str, Dict[str, float]],
    value_pool: Dict[str, List[str]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(base: Dict[str, Any], situation: str, target_new: str,
            label4: str, prior_edits: List[Dict[str, str]],
            context: List[str], rationale: str) -> None:
        ans = answers.get(base["prompt"], {"answer": "", "confidence": 0.0})
        knows = _model_knows(ans["answer"], target_new, ans["confidence"])
        rows.append({
            "source": "counterfact",
            "situation": situation,
            "prompt": base["prompt"],
            "subject": base["subject"],
            "relation": base["relation"],
            "target_new": target_new,
            "current_answer": ans["answer"],
            "current_confidence": round(ans["confidence"], 4),
            "prior_edits": prior_edits,
            "retrieval_context": context,
            "label4": label4,
            "action": action_from_label4(label4, knows),
            "rationale": rationale,
        })

    for base in bases:
        true_val = base["target_true"]
        pool = [v for v in value_pool.get(base["relation"], [])
                if _norm(v) and _norm(v) != _norm(true_val)]
        if not true_val or len(pool) < 2:
            continue
        ans = answers.get(base["prompt"], {"answer": "", "confidence": 0.0})
        model_answer = str(ans.get("answer") or "").strip()
        conf = float(ans.get("confidence") or 0.0)
        model_knows_true = _model_knows(model_answer, true_val, conf)

        # a contradicting value that is clearly different from what the model
        # currently believes, so "false fact" really contradicts the model
        pool_vs_model = [
            v for v in pool
            if _norm(v) not in _norm(model_answer)
            and _norm(model_answer) not in _norm(v)
        ] or pool
        fake1, fake2 = rng.sample(pool_vs_model, 2)

        # 1) the model already produces this answer -> the edit is redundant
        #    (no-conflict, model knows the target)
        if len(_norm(model_answer)) >= 2 and conf >= 0.5:
            add(base, "redundant_answer", model_answer, "no-conflict", [], [],
                "Model already answers this (subject, relation) with high "
                "confidence, so writing the edit changes nothing.")

        # 2) a true fact the model cannot currently answer -> no conflict, but
        #    the parameters do need the write
        if not model_knows_true:
            add(base, "true_fact_unknown", true_val, "no-conflict", [], [],
                "Edit states the value the relation actually holds and no "
                "prior edit or context contradicts it; the model merely does "
                "not know it yet.")

        # 3) the model holds a confident belief and the edit contradicts it
        if model_answer and conf >= 0.3:
            add(base, "false_fact", fake1, "pre-training-conflict", [], [],
                "Edit contradicts the value the model already produces with "
                "confidence; the conflict sits in the parameters.")

        # 4) same (subject, relation) already edited with another value
        add(base, "repeat_edit", fake2, "prior-edit-conflict",
            [{"subject": base["subject"], "relation": base["relation"],
              "target": fake1}],
            [], "A previous edit in this stream already set this "
                "(subject, relation) to another value.")

        # 5) retrieved context asserts another value
        add(base, "context_override", fake1, "context-conflict", [],
            [f"The document retrieved for {base['subject']} gives {fake2} "
             f"for the relation {base['relation']}."],
            "Inference-time context asserts a different value for the same "
            "relation, so the edit conflicts with the context.")

        # 6) context supports the edit -> still no-conflict
        add(base, "context_supported", true_val, "no-conflict", [],
            [f"The document retrieved for {base['subject']} gives {true_val} "
             f"for the relation {base['relation']}."],
            "Context agrees with the edit; none of the three signals shows a "
            "conflict.")
    return rows


# --------------------------------------------------------------------------- #
# W1 pilot 100 -> gate samples (human labels, honest test set)
# --------------------------------------------------------------------------- #
def build_pilot_samples(pilot_path: str, answers: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    pilot = [json.loads(line) for line in open(pilot_path, encoding="utf-8") if line.strip()]
    by_id = {int(p["id"]): p for p in pilot}
    rows: List[Dict[str, Any]] = []
    for item in pilot:
        subject = str(item.get("subject") or "")
        relation = str(item.get("relation") or "")
        target = str(item.get("target") or "")
        prompt = f"{subject} {relation}".strip()
        prior_edits = []
        for pid in item.get("prior_edit_ids") or []:
            src = by_id.get(int(pid))
            if src:
                prior_edits.append({
                    "subject": str(src.get("subject") or ""),
                    "relation": str(src.get("relation") or ""),
                    "target": str(src.get("target") or ""),
                })
        context = [str(item["context"])] if item.get("context") else []
        ans = answers.get(prompt, {"answer": "", "confidence": 0.0})
        knows = _model_knows(ans["answer"], target, ans["confidence"])
        rows.append({
            "id": int(item["id"]),
            "source": "w1_pilot_100",
            "situation": "human_annotated",
            "prompt": prompt,
            "subject": subject,
            "relation": relation,
            "target_new": target,
            "current_answer": ans["answer"],
            "current_confidence": round(ans["confidence"], 4),
            "prior_edits": prior_edits,
            "retrieval_context": context,
            "label4": item["label"],
            "action": action_from_label4(item["label"], knows),
            "rationale": item.get("rationale", ""),
        })
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-bases", type=int, default=700)
    ap.add_argument("--n-eval-bases", type=int, default=180)
    ap.add_argument("--n-pool-bases", type=int, default=4000,
                    help="CounterFact prefix scanned to build the same-relation "
                         "value pool used for contradicting targets")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "gate_datasets"))
    ap.add_argument("--cache", default=None,
                    help="model answer cache path (default: <out-dir>/model_answer_cache.json)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = args.cache or os.path.join(args.out_dir, "model_answer_cache.json")
    rng = random.Random(args.seed)

    print("[1/5] loading CounterFact bases ...", flush=True)
    pool_bases = load_counterfact_bases(args.n_pool_bases)
    value_pool = other_values_by_relation(pool_bases)
    print(f"      {len(pool_bases)} unique-subject bases, "
          f"{len(value_pool)} relations")

    usable = [b for b in pool_bases if len(value_pool.get(b["relation"], [])) >= 3
              and b["target_true"]]
    rng.shuffle(usable)
    n_total = args.n_train_bases + args.n_eval_bases
    if len(usable) < n_total:
        raise SystemExit(f"only {len(usable)} usable bases, need {n_total}")
    train_bases = usable[:args.n_train_bases]
    dev_bases = usable[args.n_train_bases:n_total]

    print("[2/5] measuring model answers (gpt2-xl) ...", flush=True)
    cache: Dict[str, Dict[str, float]] = {}
    if os.path.exists(cache_path):
        cache = _load_json(cache_path)
    wanted = [b["prompt"] for b in usable[:n_total]]
    if os.path.exists(PILOT):
        wanted += [f"{json.loads(l)['subject']} {json.loads(l)['relation']}".strip()
                   for l in open(PILOT, encoding="utf-8") if l.strip()]
    todo = sorted({p for p in wanted if p not in cache})
    if todo:
        t0 = time.time()
        answerer = Answerer(args.model_dir)
        cache.update(answerer.answer_many(todo))
        print(f"      answered {len(todo)} prompts in {time.time() - t0:.1f}s")
    else:
        print("      cache hit, no model call needed")
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=1)

    print("[3/5] building gate situations ...", flush=True)
    train_rows = build_counterfact_samples(train_bases, cache, value_pool, rng)
    test_rows = build_counterfact_samples(dev_bases, cache, value_pool, rng)
    for i, row in enumerate(train_rows):
        row["id"] = f"train_{i:05d}"
        row["split"] = "train"
    for i, row in enumerate(test_rows):
        row["id"] = f"test_{i:05d}"
        row["split"] = "test_counterfact"

    print("[4/5] building W1 pilot-100 gate samples ...", flush=True)
    pilot_rows = build_pilot_samples(PILOT, cache) if os.path.exists(PILOT) else []
    for row in pilot_rows:
        row["split"] = "test_pilot100"

    # dev split: 20% of train situations, kept apart from the test bases
    rng.shuffle(train_rows)
    n_dev = max(1, int(0.2 * len(train_rows)))
    dev_rows, train_rows = train_rows[:n_dev], train_rows[n_dev:]
    for row in dev_rows:
        row["split"] = "dev"

    print("[5/5] writing datasets ...", flush=True)
    _dump_jsonl(os.path.join(args.out_dir, "gate_train.jsonl"), train_rows)
    _dump_jsonl(os.path.join(args.out_dir, "gate_dev.jsonl"), dev_rows)
    _dump_jsonl(os.path.join(args.out_dir, "gate_test.jsonl"), test_rows)
    _dump_jsonl(os.path.join(args.out_dir, "gate_test_pilot100.jsonl"), pilot_rows)

    card = {
        "owner": "Tang I Choi (MC631517)",
        "work_package": "W3 -- conflict-gate training data",
        "model_for_answer_signal": args.model_dir,
        "counterfact_source": COUNTERFACT,
        "pilot_source": PILOT,
        "seed": args.seed,
        "splits": {
            "train": {
                "n": len(train_rows),
                "action": dict(Counter(r["action"] for r in train_rows)),
                "label4": dict(Counter(r["label4"] for r in train_rows)),
            },
            "dev": {
                "n": len(dev_rows),
                "action": dict(Counter(r["action"] for r in dev_rows)),
                "label4": dict(Counter(r["label4"] for r in dev_rows)),
            },
            "test_counterfact": {
                "n": len(test_rows),
                "action": dict(Counter(r["action"] for r in test_rows)),
                "label4": dict(Counter(r["label4"] for r in test_rows)),
            },
            "test_pilot100": {
                "n": len(pilot_rows),
                "action": dict(Counter(r["action"] for r in pilot_rows)),
                "label4": dict(Counter(r["label4"] for r in pilot_rows)),
            },
        },
        "label_action_mapping": {
            "no-conflict + model knows target": "REJECT",
            "no-conflict + model cannot answer": "WRITE",
            "pre-training-conflict": "REJECT",
            "prior-edit-conflict": "ROUTE",
            "context-conflict": "ROUTE",
        },
        "caveats": [
            "CounterFact situations are constructed, not human-labelled: the "
            "label follows from how the situation was built, per the W1 manual.",
            "The pilot-100 split keeps its human labels and is never used for "
            "training or model selection; it is reported separately as the "
            "honest out-of-distribution test set.",
            "The current answer/confidence signal is measured with gpt2-xl, the "
            "same model used for the W3/W4 runs; other backbones would shift it.",
        ],
    }
    with open(os.path.join(args.out_dir, "dataset_card.json"), "w", encoding="utf-8") as fh:
        json.dump(card, fh, ensure_ascii=False, indent=2)

    for name in ("train", "dev", "test_counterfact", "test_pilot100"):
        s = card["splits"][name]
        print(f"  {name:16s} n={s['n']:5d} action={s['action']} label4={s['label4']}")
    print(f"datasets written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
