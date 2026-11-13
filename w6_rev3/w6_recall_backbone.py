# -*- coding: utf-8 -*-
"""Run the memory read-path recall sweep on a second backbone.

Why: every read-path number in this project is measured on gpt2-small, so the
copy failure reported in Sec.12 (canonical 0.72 vs paraphrase 0.34) can always
be waved away with "your model is too small".  The cheapest honest answer is to
re-run the SAME sweep, with no code path changed, on gpt2-medium -- 355M against
124M, same tokenizer family, same note format, same prompts.

This is deliberately not a new headline: it is one diagnostic slice, reported
next to the small-backbone number it is meant to qualify.

  python w6_recall_backbone.py --model gpt2-medium --tag gpt2medium --offsets 0 --n 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import w6_recall_gpt2small as R  # noqa: E402  (reuse parse_records / run_slice verbatim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--tag", default="gpt2medium")
    ap.add_argument("--offsets", default="0")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    model_dir = os.path.join(R.REPO, "EasyEdit", "hugging_cache", args.model)
    out_dir = os.path.join(R.REPO, "w6_rev3", "recall_%s" % args.tag)
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    for off in [int(x) for x in args.offsets.split(",") if x.strip()]:
        cases = R.parse_records(R.DATA, off, args.n)
        print("=== %s slice offset=%d n=%d ===" % (args.model, off, len(cases)), flush=True)
        res = R.run_slice(model, tok, cases, args.max_new, "o%d" % off)
        res["model"] = args.model
        path = os.path.join(out_dir, "recall_%s_o%d_n%d.json" % (args.tag, off, len(cases)))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        keep = {k: v for k, v in res.items() if k != "per_case"}
        print(json.dumps(keep, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
