"""W1 real-model evaluation driver for the CALM project.

Loads a real Hugging Face model (locked roster: gpt2-small / gpt2-xl /
gpt-j-6b), runs the W1 loaders on the real datasets (CounterFact / zsRE /
MQuAKE) and scores the full W1 metric battery:

  rewrite_acc, paraphrase_acc, locality_neighborhood_acc,
  locality_attribute_acc, portability_acc.

Usage:
    python real_eval_w1.py --model-key gpt2-xl --dataset counterfact \
        --data-path <path-to-data> --limit 200 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W1_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w1\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W1_DIR)

from model_utils import load_model, make_generator  # noqa: E402
from loaders import build_loader  # noqa: E402
from metrics import evaluate_cases  # noqa: E402


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load dataset records from json / jsonl / parquet into a list of dicts."""
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


def normalize_records(records: List[Dict[str, Any]], dataset: str
                      ) -> List[Dict[str, Any]]:
    """Normalise real dataset records into the W1 loader schema.

    The official ROME/EasyEdit files store the edit inside
    ``requested_rewrite`` with ``{}`` placeholders in the prompt; the W1
    loaders expect ``request`` plus concrete prompts, so we map fields here
    without modifying the delivered loader modules.
    """
    norm = []
    for rec in records:
        # zsRE (MEND file layout) stores the edit flat on the record.
        if dataset == "zsre" and "requested_rewrite" not in rec \
                and "request" not in rec:
            out = dict(rec)
            rephrase = rec.get("rephrase", [])
            rephrase = rephrase if isinstance(rephrase, list) else [rephrase]
            out["request"] = {
                "prompt": rec.get("src", ""),
                "subject": rec.get("subject", ""),
                "target_new": rec.get("alt", ""),
                "target_true": rec.get("pred", ""),
            }
            out["paraphrasess"] = rephrase
            loc = rec.get("loc", [])
            loc = loc if isinstance(loc, list) else [loc]
            out["neighborhood_prompts"] = loc
            norm.append(out)
            continue
        req = rec.get("request") or rec.get("requested_rewrite") or rec
        out = dict(rec)
        prompt = req.get("prompt", "")
        subject = req.get("subject", "")
        if "{}" in prompt and subject:
            prompt = prompt.replace("{}", subject)
        out["request"] = {
            "prompt": prompt,
            "subject": subject,
            "target_new": req.get("target_new", ""),
            "target_true": req.get("target_true", ""),
        }
        # MQuAKE raw records keep the multi-hop chain as a list of question
        # strings; expose them as portability_prompts for the W1 metrics.
        if dataset == "mquake" and not out.get("portability_prompts"):
            qs = rec.get("questions", [])
            if qs and all(isinstance(q, str) for q in qs):
                out["portability_prompts"] = qs
        norm.append(out)
    return norm


def main() -> int:
    ap = argparse.ArgumentParser(description="W1 real-model evaluation")
    ap.add_argument("--model-key", required=True,
                    choices=["gpt2-small", "gpt2-xl", "gpt-j-6b"])
    ap.add_argument("--dataset", required=True,
                    choices=["counterfact", "zsre", "mquake"])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Normalise the raw records into a JSON file the W1 loader can consume.
    records = normalize_records(load_records(args.data_path), args.dataset)
    if args.limit is not None:
        records = records[: args.limit]
    norm_path = os.path.join(BASE, f"w1_{args.dataset}_norm.json")
    with open(norm_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False)

    print(f"[w1] loading model {args.model_key} ...", flush=True)
    model, tok = load_model(args.model_key)
    gen = make_generator(model, tok, max_new_tokens=args.max_new_tokens)

    loader = build_loader(args.dataset, data_path=norm_path)
    cases = loader.load()
    print(f"[w1] {args.dataset}: {len(cases)} cases loaded", flush=True)

    t0 = time.time()
    metrics = evaluate_cases(cases, gen)
    elapsed = time.time() - t0

    result = {
        "task": f"W1-{args.dataset}",
        "model": args.model_key,
        "n_cases": metrics["n_cases"],
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
