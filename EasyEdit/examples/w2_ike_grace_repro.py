# W2 IKE / GRACE memory-based baseline reproduction (CALM project, Tang I Choi)
# Reproduces the two memory-type baselines used by CALM under the same protocol
# as the teammate ROME/MEMIT reproduction (docs/evaluation_protocol.md):
#   * IKE   (In-Context Knowledge Editing, Zheng et al. 2023): does NOT change
#            weights. It retrieves k in-context examples from a fact memory bank
#            (zsre_mend_train) via a sentence-transformer and prepends them to
#            the prompt during generation (EasyEdit IKEHyperParams).
#   * GRACE (Gradient Ascent with Contrastive Regularization, Hartvigsen et al.
#            2023): performs n_iter gradient-ascent edits on a single MLP
#            weight with early-stop contrastive regularization
#            (EasyEdit GraceHyperParams).
#
# Both algorithms are driven through the same BaseEditor interface and evaluated
# with the same rewrite / rephrase / locality (and optional portability) slices,
# so the numbers are directly comparable with the W2 ROME reproduction and the
# W3 conflict-gate training set built by Tang I Choi.
#
# Data:
#   eval:  official CounterFact split (counterfact.json, memit.baulab.info)
#          or official zsRE eval split (zsre_mend_eval.json)
#   IKE memory bank: zsre_mend_train.json (memit.baulab.info), optionally
#          truncated with --train-size (default 10000, matching the EasyEdit
#          IKE example). The truncated slice is cached under EasyEdit/data/.
#
# Usage:
#   IKE:
#     python examples/w2_ike_grace_repro.py --alg ike --data zsre \
#         --model gpt2-xl --n 50 --train-size 10000 --device mps
#   GRACE:
#     python examples/w2_ike_grace_repro.py --alg grace --data counterfact \
#         --model gpt2-xl --n 50 --device mps
#   GPU (default device 0 = cuda:0):
#     python examples/w2_ike_grace_repro.py --alg grace --data zsre \
#         --model gpt-j-6B --n 50 --fp16
#
# Output: results/eval/{IKE|GRACE}_{model}_{YYYYmmdd_HHMMSS}.json
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

EASYEDIT = str(Path(__file__).resolve().parent.parent)
os.chdir(EASYEDIT)
sys.path.insert(0, EASYEDIT)

import torch

from easyeditor import BaseEditor, IKEHyperParams, GraceHyperParams, ZsreDataset
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer

DATA_URL = {
    "counterfact": "https://memit.baulab.info/data/dsets/counterfact.json",
    "zsre": "https://memit.baulab.info/data/dsets/zsre_mend_eval.json",
    "zsre_train": "https://memit.baulab.info/data/dsets/zsre_mend_train.json",
}

HPARAMS_DIR = {
    "IKE": "hparams/IKE",
    "GRACE": "hparams/GRACE",
}


def ensure_data(path: Path, data_type: str):
    if not path.exists():
        print(f"[W2] downloading {data_type} data ...", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(DATA_URL[data_type], str(path))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def counterfact_to_edit_inputs(records):
    prompts, subjects, ground_truths, target_news, rephrases = [], [], [], [], []
    nq_prompts, nq_ground_truths = [], []
    for r in records:
        req = r["requested_rewrite"]
        prompt = req["prompt"].format(req["subject"])
        prompts.append(prompt)
        subjects.append(req["subject"])
        ground_truths.append(req["target_true"]["str"])
        target_news.append(req["target_new"]["str"])
        rephrases.append(r["paraphrase_prompts"][0])
        nq_prompts.append(r["neighborhood_prompts"])
        nq_ground_truths.append([req["target_true"]["str"]] * len(r["neighborhood_prompts"]))
    localities = {"neighborhood": {"prompt": nq_prompts, "ground_truth": nq_ground_truths}}
    return prompts, subjects, ground_truths, target_news, rephrases, localities


def zsre_to_edit_inputs(records):
    prompts, subjects, ground_truths, target_news, rephrases = [], [], [], [], []
    nq_prompts, nq_ground_truths = [], []
    for r in records:
        assert "subject" in r and isinstance(r["answers"], list) and len(r["answers"]) > 0, r
        prompts.append(r["src"])
        subjects.append(r["subject"])
        ground_truths.append(r["answers"][0])
        target_news.append(r["alt"])
        rephrases.append(r["rephrase"])
        nq_prompts.append(r["loc"])
        nq_ground_truths.append(r["loc_ans"])
    localities = {"nq": {"prompt": nq_prompts, "ground_truth": nq_ground_truths}}
    return prompts, subjects, ground_truths, target_news, rephrases, localities


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def aggregate(metrics, locality_key):
    """Aggregate EasyEdit post metrics to protocol-level rewrite/rephrase/locality."""
    rewrite_acc, rephrase_acc, loc_accs = [], [], []
    per_case = []
    for i, m in enumerate(metrics):
        post = m.get("post", {})
        case = {
            "case_id": i,
            "prompt": m.get("requested_rewrite", {}).get("prompt"),
            "target_new": m.get("requested_rewrite", {}).get("target_new"),
            "rewrite_acc": post.get("rewrite_acc"),
            "rephrase_acc": post.get("rephrase_acc"),
        }
        loc = post.get("locality", {})
        case[f"{locality_key}_acc"] = loc.get(f"{locality_key}_acc")
        per_case.append(case)
        rewrite_acc.extend(_as_list(case["rewrite_acc"]))
        rephrase_acc.extend(_as_list(case["rephrase_acc"]))
        loc_accs.extend(_as_list(case[f"{locality_key}_acc"]))
    return {
        "rewrite_acc": mean(rewrite_acc),
        "rephrase_acc": mean(rephrase_acc),
        f"locality_{locality_key}_acc": mean(loc_accs),
        "per_case": per_case,
    }


def aggregate_portability(metrics):
    accs = {}
    for m in metrics:
        post = m.get("post", {})
        port = post.get("portability", {})
        for key, val in port.items():
            if key.endswith("_acc"):
                accs.setdefault(key, []).extend(_as_list(val))
    return {key: mean(vals) for key, vals in accs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", choices=["ike", "grace"], required=True)
    ap.add_argument("--data", choices=["counterfact", "zsre"], default="zsre")
    ap.add_argument("--model", choices=["gpt2-xl", "gpt-j-6B"], default="gpt2-xl",
                    help="model to edit (gpt-j-6B expects fp16 weights in hugging_cache/gpt-j-6B)")
    ap.add_argument("--n", type=int, default=50, help="number of edit cases (0 = all)")
    ap.add_argument("--train-size", type=int, default=10000,
                    help="IKE memory bank size (top-N of zsre_mend_train; 0 = full train set)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="load model in fp16 (GPU only)")
    ap.add_argument("--device", default="0",
                    help="torch device: '0'/'1' = cuda:N (GPU), 'mps' (Apple Silicon), 'cpu'")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    alg = args.alg.upper()

    data_dir = Path(EASYEDIT) / "data"
    data_fname = {"counterfact": "counterfact.json", "zsre": "zsre_mend_eval.json"}[args.data]
    data_path = data_dir / data_fname
    raw = ensure_data(data_path, args.data)
    records = raw[: args.n] if args.n else raw

    if args.data == "counterfact":
        prompts, subjects, ground_truths, target_news, rephrases, localities = counterfact_to_edit_inputs(records)
        locality_key = "neighborhood"
        portability_inputs = None
    else:  # zsre
        prompts, subjects, ground_truths, target_news, rephrases, localities = zsre_to_edit_inputs(records)
        locality_key = "nq"
        portability_inputs = None
    print(f"[W2] alg={alg} data={args.data} n={len(records)} device={args.device}", flush=True)

    # NOTE: hparams filename casing differs between the IKE dir (gpt-j-6B.yaml)
    # and the GRACE dir (gpt-j-6b.yaml); normalise per algorithm.
    hparams_fname = f"{args.model}.yaml"
    if alg == "GRACE" and args.model == "gpt-j-6B":
        hparams_fname = "gpt-j-6b.yaml"
    hparams_path = os.path.join(EASYEDIT, HPARAMS_DIR[alg], hparams_fname)
    hparams = (IKEHyperParams if alg == "IKE" else GraceHyperParams).from_hparams(hparams_path)
    hparams.model_name = os.path.join(EASYEDIT, "hugging_cache", args.model)
    if alg == "IKE":
        hparams.sentence_model_name = os.path.join(EASYEDIT, "hugging_cache", "all-MiniLM-L6-v2")
    hparams.device = args.device  # override yaml (int 0 -> cuda:0 unless overridden)
    if args.fp16:
        hparams.fp16 = True
    print(f"[W2] hparams: {hparams_path} model={hparams.model_name} device={hparams.device}", flush=True)

    # ---- IKE: prepare the fact memory bank (zsre_mend_train) once ----
    train_ds = None
    if alg == "IKE":
        train_fname = "zsre_mend_train.json" if args.train_size == 0 else f"zsre_mend_train_{args.train_size}.json"
        train_path = data_dir / train_fname
        if not train_path.exists():
            print(f"[W2] building IKE memory bank slice (top-{args.train_size}) ...", flush=True)
            full = ensure_data(data_dir / "zsre_mend_train.json", "zsre_train")
            with open(train_path, "w", encoding="utf-8") as f:
                json.dump(full[: args.train_size], f)
            print(f"[W2] memory bank slice saved to {train_path}", flush=True)
        train_ds = ZsreDataset(str(train_path))
        print(f"[W2] encoding {len(train_ds)} memory facts with {hparams.sentence_model_name} ...", flush=True)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(args.device)
        encode_ike_facts(sentence_model, train_ds, hparams)
        print("[W2] IKE memory embeddings ready", flush=True)

    print(f"[W2] loading editor (model={hparams.model_name}) ...", flush=True)
    editor = BaseEditor.from_hparams(hparams)
    print("[W2] editor loaded, starting edits ...", flush=True)

    t0 = time.time()
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        subject=subjects,
        ground_truth=ground_truths,
        target_new=target_news,
        rephrase_prompts=rephrases,
        locality_inputs=localities,
        portability_inputs=portability_inputs,
        train_ds=train_ds,
        keep_original_weight=True,
        sequential_edit=False,
    )
    elapsed = time.time() - t0

    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
        print(f"[W2] peak CUDA memory allocated: {peak_vram_mb} MB", flush=True)

    agg = aggregate(metrics, locality_key)
    port_agg = aggregate_portability(metrics)
    result = {
        "alg": alg,
        "model": args.model,
        "split": f"{args.data} (official, first {len(records)})",
        "n": len(records),
        "seed": args.seed,
        "rewrite_acc": agg["rewrite_acc"],
        "rephrase_acc": agg["rephrase_acc"],
        "locality": {f"{locality_key}_acc": agg[f"locality_{locality_key}_acc"]},
        "portability": port_agg,
        "memory_bank": {"name": "zsre_mend_train", "size": (args.train_size or "full")} if alg == "IKE" else None,
        "smoke": 0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
        "device": args.device,
        "fp16": bool(args.fp16),
        "elapsed_s": round(elapsed, 1),
        "peak_vram_mb": peak_vram_mb,
        "hparams": hparams_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = Path(EASYEDIT) / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{alg}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[W2] result saved to {out_path}", flush=True)

    print("[W2-METRICS]", json.dumps(result, indent=2, default=str), flush=True)
    print("[W2-PER-CASE]", json.dumps(agg["per_case"], indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
