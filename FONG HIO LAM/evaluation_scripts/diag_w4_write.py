"""Diagnose why the MEMIT write path fails on MQuAKE cases.

Compares the top-1 answer-slot probability before/after an edit for a few
configurations (context-pool size x optimisation steps).  Prints only.
"""
from __future__ import annotations

import json
import sys

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W2_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w2\FONG HIO LAM"
W4_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w4\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W2_DIR)
sys.path.insert(0, W4_DIR)

import torch  # noqa: E402
from model_utils import load_model  # noqa: E402
from real_eval_w2 import RealMemitEditor  # noqa: E402
from real_w4_mquake import load_mquake  # noqa: E402


def main() -> int:
    mq = BASE + r"\data\MQuAKE\datasets\MQuAKE-CF.json"
    cf = BASE + r"\data\counterfact.json"
    cases = load_mquake(mq, 3)
    with open(cf, "r", encoding="utf-8") as fh:
        cf_raw = json.load(fh)
    pool = []
    for rec in cf_raw[:120]:
        rw = rec.get("request", rec).get("requested_rewrite", {})
        p = rw.get("prompt", "")
        s = rw.get("subject", "")
        t = rw.get("target_new", {})
        t = t.get("str", t) if isinstance(t, dict) else t
        if p and s and t:
            pool.append({"prompt": p.replace("{}", s), "subject": s, "target_new": t})

    model, tok = load_model("gpt2-xl")
    editor = RealMemitEditor(model, tok)

    for case in cases:
        prompt, tgt = case["prompt"], case["target_new"]
        ids = tok(prompt, return_tensors="pt").to(editor.device)
        first_tok = tok(tgt, add_special_tokens=False)["input_ids"][0]
        with torch.no_grad():
            base_lg = model(ids["input_ids"]).logits[0, -1, :]
        base_top = tok.decode([int(base_lg.argmax())])
        base_p = float(torch.softmax(base_lg, -1)[first_tok])
        print(f"\n=== {case['subject']} | target={tgt!r} | first_tok={first_tok} "
              f"({tok.decode([first_tok])!r}))", flush=True)
        print(f"  pre-edit top1={base_top!r} p(target)={base_p:.4f}", flush=True)

        for ctx_n, steps in ((16, 20), (16, 35), (96, 35)):
            editor.batch_edit([{"prompt": prompt, "subject": case["subject"],
                                "target_new": tgt}], pool[:ctx_n],
                              n_val_steps=steps)
            with torch.no_grad():
                lg = model(ids["input_ids"]).logits[0, -1, :]
            top = tok.decode([int(lg.argmax())])
            p_t = float(torch.softmax(lg, -1)[first_tok])
            print(f"  ctx={ctx_n:3d} steps={steps:2d} -> top1={top!r} "
                  f"p(target)={p_t:.4f} delta_p={p_t - base_p:+.4f}", flush=True)
            editor.restore()
    return 0


if __name__ == "__main__":
    sys.exit(main())
