"""W2 real-model editing reproduction (MEMIT / AlphaEdit) for CALM.

Loads a real Hugging Face model (gpt2-xl preferred, gpt-j-6b optional),
runs the official EasyEdit implementations of MEMIT / AlphaEdit on a
subset of the real CounterFact dataset, then scores the edited model with
the W1 metric battery (rewrite / paraphrase / locality / portability).

The mom2 statistics are collected from the real wikitext-2 corpus (the
official MEMIT/AlphaEdit statistics are computed over wikipedia; we use
the smaller wikitext-2 corpus locally so the run is reproducible on this
machine without downloading the multi-GB wikipedia dump).

Usage:
    python real_w2_edit.py --algo memit --model-key gpt2-xl \
        --data-path <counterfact.json|parquet> --limit 50 --out w2_memit.json
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
EASYEDIT_DIR = os.path.join(
    BASE, "CALM-Editor-CISC7021-Task1-Group26", "EasyEdit"
)
sys.path.insert(0, BASE)
sys.path.insert(0, EASYEDIT_DIR)

from model_utils import load_model  # noqa: E402
from easyeditor.models.memit import apply_memit_to_model, MEMITHyperParams  # noqa: E402
from easyeditor.models.alphaedit import (  # noqa: E402
    apply_AlphaEdit_to_model, AlphaEditHyperParams,
)

LOCAL_MODEL_PATHS = {
    "gpt2-xl": os.path.join(BASE, "models", "gpt2-xl"),
    "gpt-j-6b": os.path.join(BASE, "models", "EleutherAI__gpt-j-6b"),
}


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


def to_requests(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw CounterFact records into EasyEdit request dicts."""
    reqs = []
    for i, rec in enumerate(records):
        t_new = rec.get("target_new", rec.get("target", ""))
        if isinstance(t_new, dict):
            t_new = t_new.get("str", "")
        t_true = rec.get("target_true", "")
        if isinstance(t_true, dict):
            t_true = t_true.get("str", "")
        reqs.append({
            "case_id": rec.get("case_id", i),
            "prompt": rec.get("prompt", rec.get("request", {}).get("prompt", "")),
            "subject": rec.get("subject", rec.get("request", {}).get("subject", "")),
            "target_new": t_new,
            "target_true": t_true,
            "paraphrase_prompts": rec.get("paraphrase_prompts",
                                          rec.get("paraphrasess", [])),
            "neighborhood_prompts": rec.get("neighborhood_prompts", []),
            "attribute_prompts": rec.get("attribute_prompts", []),
        })
    return reqs


def load_hparams(algo: str, model_key: str):
    if algo == "memit":
        yaml_path = os.path.join(EASYEDIT_DIR, "hparams", "MEMIT",
                                 "gpt2-xl.yaml" if model_key == "gpt2-xl" else "gpt-j-6B.yaml")
        hp = MEMITHyperParams.from_hparams(yaml_path)
    else:
        yaml_path = os.path.join(EASYEDIT_DIR, "hparams", "AlphaEdit",
                                 "gpt2-xl.yaml" if model_key == "gpt2-xl" else "gpt-j-6B.yaml")
        hp = AlphaEditHyperParams.from_hparams(yaml_path)
    # Override with local configuration.
    hp.model_name = LOCAL_MODEL_PATHS[model_key]
    hp.device = 0
    hp.stats_dir = os.path.join(BASE, "stats")
    hp.mom2_dataset = "wikitext2"
    hp.mom2_n_samples = 20000
    hp.mom2_dtype = "float32"
    if algo == "alphaedit":
        hp.P_loc = os.path.join(BASE, "stats", "null_space_project.pt")
    return hp


def main() -> int:
    ap = argparse.ArgumentParser(description="W2 real editing reproduction")
    ap.add_argument("--algo", required=True, choices=["memit", "alphaedit"])
    ap.add_argument("--model-key", default="gpt2-xl",
                    choices=["gpt2-xl", "gpt-j-6b"])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_records(args.data_path)[: args.limit]
    requests = to_requests(records)
    print(f"[w2] {args.algo}: {len(requests)} requests loaded", flush=True)

    print(f"[w2] loading model {args.model_key} ...", flush=True)
    model, tok = load_model(args.model_key)
    hparams = load_hparams(args.algo, args.model_key)

    # --- scoring helpers (generation based, same protocol as W1) ---------
    from metrics import evaluate_cases  # noqa: E402
    from loaders import EditCase  # noqa: E402

    def to_case(req: Dict[str, Any], idx: int) -> EditCase:
        return EditCase(
            case_id=idx,
            prompt=req["prompt"],
            subject=req["subject"],
            target_new=req["target_new"],
            target_true=req["target_true"],
            paraphrasess=req["paraphrase_prompts"],
            neighborhood_prompts=req["neighborhood_prompts"],
            attribute_prompts=req["attribute_prompts"],
        )

    def run_generation(prompt: str) -> str:
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs.to(model.device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)

    # --- pre-edit metrics ------------------------------------------------
    pre_cases = [to_case(r, i) for i, r in enumerate(requests)]
    pre_metrics = evaluate_cases(pre_cases, run_generation)
    print(f"[w2] PRE-EDIT metrics: {pre_metrics}", flush=True)

    # --- apply the edit --------------------------------------------------
    t0 = time.time()
    if args.algo == "memit":
        model, _ = apply_memit_to_model(model, tok, requests, hparams,
                                        copy=False, keep_original_weight=False)
    else:
        model, _ = apply_AlphaEdit_to_model(model, tok, requests, hparams,
                                            copy=False, keep_original_weight=False)
    edit_elapsed = time.time() - t0
    print(f"[w2] edit applied in {edit_elapsed:.1f}s", flush=True)

    # --- post-edit metrics -----------------------------------------------
    post_metrics = evaluate_cases(pre_cases, run_generation)

    result = {
        "task": f"W2-{args.algo}",
        "model": args.model_key,
        "algo": args.algo,
        "n_edits": len(requests),
        "pre_edit_metrics": {k: round(v, 4) for k, v in pre_metrics.items()},
        "post_edit_metrics": {k: round(v, 4) for k, v in post_metrics.items()},
        "edit_elapsed_seconds": round(edit_elapsed, 2),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
