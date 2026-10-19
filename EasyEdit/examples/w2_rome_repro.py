# W2 ROME single-fact baseline reproduction (CALM project, Fong Chin Wai)
# Runs ROME edits on gpt2-xl (local cache) and aggregates reliability /
# generalisation / locality metrics following docs/evaluation_protocol.md.
#
# Data: official CounterFact split (counterfact.json, memit.baulab.info) or
# official zsRE split (zsre_mend_eval.json); converted with the same mapping
# used by EasyEdit examples/run_AlphaEdit_editing.py.
#
# Usage:
#   python examples/w2_rome_repro.py [--data counterfact|zsre] [--n 50]
#                                     [--mom2-dataset wikitext2] [--mom2-n-samples 10000]
#                                     [--seed 42] [--fp16]
#
# Output: results/eval/ROME_gpt2-xl_<date>.json
import argparse
import json
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path

EASYEDIT = str(Path(__file__).resolve().parent.parent)
os.chdir(EASYEDIT)
sys.path.insert(0, EASYEDIT)

import torch
from easyeditor import ROMEHyperParams, BaseEditor

DATA_URL = {
    "counterfact": "https://memit.baulab.info/data/dsets/counterfact.json",
    "zsre": "https://memit.baulab.info/data/dsets/zsre_mend_eval.json",
    # Multi-hop portability benchmark: MQuAKE (CF-3k v2).
    # portability(p1): after editing hop-1 fact, the model should answer
    # the multi-hop "New Question" with the edited "new_answer".
    "mquake": "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k-v2.json",
}


def ensure_data(path: Path, data_type: str):
    if not path.exists():
        print(f"[W2] downloading {data_type} data ...", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)  # data dir may not exist in a fresh env
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


def mquake_to_edit_inputs(records):
    """MQuAKE-CF-3k-v2 -> ROME edit inputs + portability (p1) slice.

    Edit only the first hop of each chain (requested_rewrite[0]); the
    multi-hop `questions` with the post-edit `new_answer` form the p1
    portability evaluation (EasyEdit icl_lm_eval, prefix
    "New Fact: <prompt> <target_new>\\nPrompt: <question>").
    """
    prompts, subjects, ground_truths, target_news = [], [], [], []
    p1_prompts, p1_ground_truths = [], []
    for r in records:
        req = r["requested_rewrite"][0]
        prompt = req["prompt"].format(req["subject"])
        prompts.append(prompt)
        subjects.append(req["subject"])
        ground_truths.append(req["target_true"]["str"])
        target_news.append(req["target_new"]["str"])
        # _prepare_requests requires one portability input per edit case, but
        # evaluate.py accepts a list of prompts/ground_truths per case, so we
        # pack all 3 multi-hop questions (with the post-edit new_answer) here.
        p1_prompts.append(r["questions"])
        p1_ground_truths.append([r["new_answer"]] * len(r["questions"]))
    portability = {"p1": {"prompt": p1_prompts, "ground_truth": p1_ground_truths}}
    return prompts, subjects, ground_truths, target_news, portability


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
    """Aggregate EasyEdit post.portability.{key}_acc across edited cases."""
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
    ap.add_argument("--data", choices=["counterfact", "zsre", "mquake"], default="counterfact")
    ap.add_argument("--model", choices=["gpt2-xl", "gpt-j-6B"], default="gpt2-xl",
                    help="model to edit (gpt-j-6B expects fp16 weights in hugging_cache/gpt-j-6B)")
    ap.add_argument("--n", type=int, default=50, help="number of edit cases (0 = all)")
    ap.add_argument("--mom2-dataset", default="wikitext2",
                    help="mom2 statistics corpus (wikitext2 is small/fast; official uses wikipedia)")
    ap.add_argument("--mom2-n-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="enable fp16 to reduce VRAM")
    ap.add_argument("--v-steps", type=int, default=None,
                    help="override v_num_grad_steps (default 20; lower saves VRAM during backward)")
    ap.add_argument("--ctx-len", type=str, default=None,
                    help="override context_template_length_params, e.g. '[3,5]' -> [[3,5],[3,5]]")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(EASYEDIT) / "data"
    data_fname = {"counterfact": "counterfact.json",
                  "zsre": "zsre_mend_eval.json",
                  "mquake": "MQuAKE-CF-3k-v2.json"}[args.data]
    data_path = data_dir / data_fname
    raw = ensure_data(data_path, args.data)
    records = raw[: args.n] if args.n else raw

    if args.data == "counterfact":
        prompts, subjects, ground_truths, target_news, rephrases, localities = (
            counterfact_to_edit_inputs(records))
        locality_key = "neighborhood"
        portability_inputs = None
    elif args.data == "zsre":
        prompts, subjects, ground_truths, target_news, rephrases, localities = (
            zsre_to_edit_inputs(records))
        locality_key = "nq"
        portability_inputs = None
    else:  # mquake: ROME edit hop-1, evaluate multi-hop portability (p1)
        prompts, subjects, ground_truths, target_news, portability_inputs = (
            mquake_to_edit_inputs(records))
        rephrases, localities, locality_key = None, None, None
    print(f"[W2] data={args.data} n={len(records)} portability={portability_inputs is not None}", flush=True)

    hparams = ROMEHyperParams.from_hparams(
        os.path.join(EASYEDIT, "hparams/ROME", f"{args.model}.yaml")
    )
    hparams.model_name = os.path.join(EASYEDIT, "hugging_cache", args.model)
    hparams.stats_dir = os.path.join(EASYEDIT, "data", "stats")
    hparams.device = 0  # override (gpt-j yaml defaults to device 1)
    hparams.mom2_dataset = args.mom2_dataset
    hparams.mom2_n_samples = args.mom2_n_samples
    if args.fp16:
        hparams.fp16 = True
    if args.v_steps is not None:
        hparams.v_num_grad_steps = args.v_steps
    if args.ctx_len is not None:
        lo, hi = map(int, args.ctx_len.strip("[]").split(","))
        hparams.context_template_length_params = [[lo, hi], [lo, hi]]
    print(f"[W2] hparams: v_steps={hparams.v_num_grad_steps} ctx={hparams.context_template_length_params} fp16={hparams.fp16}", flush=True)

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
        keep_original_weight=True,
        sequential_edit=False,
    )
    elapsed = time.time() - t0

    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
        print(f"[W2] peak CUDA memory allocated: {peak_vram_mb} MB", flush=True)

    rw, rp = [], []
    for m in metrics:
        post = m.get("post", {})
        rw.extend(_as_list(post.get("rewrite_acc")))
        rp.extend(_as_list(post.get("rephrase_acc")))
    rewrite_acc, rephrase_acc = mean(rw), mean(rp)
    locality = {}
    per_case = None
    if locality_key is not None:
        agg = aggregate(metrics, locality_key)
        locality = {f"{locality_key}_acc": agg[f"locality_{locality_key}_acc"]}
        per_case = agg["per_case"]
    port_agg = aggregate_portability(metrics)

    result = {
        "alg": "ROME",
        "model": args.model,
        "split": f"{args.data} (official, first {len(records)})",
        "n": len(records),
        "seed": args.seed,
        "rewrite_acc": rewrite_acc,
        "rephrase_acc": rephrase_acc,
        "locality": locality,
        "portability": port_agg,
        "smoke": 0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "fp16": bool(args.fp16),
        "mom2_dataset": args.mom2_dataset,
        "mom2_n_samples": args.mom2_n_samples,
        "elapsed_s": round(elapsed, 1),
        "peak_vram_mb": peak_vram_mb,
        "hparams": f"hparams/ROME/{args.model}.yaml",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = Path(EASYEDIT) / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ROME_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[W2] result saved to {out_path}", flush=True)

    print("[W2-METRICS]", json.dumps(result, indent=2, default=str), flush=True)
    print("[W2-PER-CASE]", json.dumps(per_case, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
