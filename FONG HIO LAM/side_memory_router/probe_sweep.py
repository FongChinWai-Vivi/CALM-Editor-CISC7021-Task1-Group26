"""Pre-edit probe recall sweep for the W3 ConflictGate REJECT branch.

The canary group asks the gate to write a fact the model already knows
(target_true).  A correct gate must REJECT it, and the only admissible
evidence is a pre-edit probe whose answer already contains target_true.
The first real run rejected 9/20 (recall 0.45) with a 8-token probe on the
first paraphrase prompt only.  This script isolates *why* the other 11 were
missed by sweeping the two knobs that control the probe:

  - probe budget (max_new_tokens): 8 / 16 / 24 / 32
  - probe template: each individual paraphrase prompt, and the union of all
    paraphrase prompts ("any-hit" over the templates)

For every (budget, template) cell it reports the containment recall, i.e. the
fraction of canary cases whose probe answer already contains target_true under
the same containment rule ConflictGate._confidence_score uses (=1.0 when the
target appears as a substring).  No model weights are edited here; this is a
pure read of the unedited model, so it is cheap and deterministic.

Usage:
    python probe_sweep.py --data-path <counterfact.json> --n 20 \
        --out probe_sweep_gpt2xl.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import torch

BASE = r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp"
W3_DIR = r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w3\FONG HIO LAM"
sys.path.insert(0, BASE)
sys.path.insert(0, W3_DIR)

from model_utils import load_model  # noqa: E402
from conflict_gate import ConflictGate  # noqa: E402
from real_w3_components import load_records  # noqa: E402

BUDGETS = [8, 16, 24, 32]
MAX_PARAS = 4


def build_canary(records: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    cases = []
    for i, rec in enumerate(records[:n]):
        request = rec.get("request", rec)
        rw = request.get("requested_rewrite", request)
        subject = rw.get("subject", request.get("subject", ""))
        prompt_tpl = rw.get("prompt", request.get("prompt", ""))
        t_true = rw.get("target_true", {})
        target_true = t_true.get("str", t_true) if isinstance(t_true, dict) else t_true
        para = rec.get("paraphrase_prompts") or rw.get("paraphrase_prompts") or []
        # CounterFact paraphrase prompts are already subject-prefilled
        # continuations (no "{}" slot); .replace is a no-op then, which is
        # exactly what real_w3_components.py does.
        paras = [p.replace("{}", subject) for p in para][:MAX_PARAS]
        if not (subject and target_true and paras):
            continue
        cases.append({"sample_id": i, "subject": subject,
                      "target_true": target_true, "paras": paras,
                      "orig": prompt_tpl.replace("{}", subject)})
    return cases


def generate(model, tok, prompt: str, budget: int) -> str:
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs.to(model.device),
            max_new_tokens=budget,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="W3 pre-edit probe recall sweep")
    ap.add_argument("--model-key", default="gpt2-xl")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_records(args.data_path)
    cases = build_canary(records, args.n)
    print(f"[sweep] {len(cases)} canary cases", flush=True)

    model, tok = load_model(args.model_key)
    scorer = ConflictGate(seed=0)

    n_paras = max(len(c["paras"]) for c in cases)
    n_tpl = n_paras + 1  # template 0 = the original prompt, 1.. = paraphrases
    # cells[(template_idx, budget)] -> hits
    cells = {(t, b): 0 for t in range(n_tpl) for b in BUDGETS}
    union = {b: 0 for b in BUDGETS}       # every template
    union_para = {b: 0 for b in BUDGETS}  # paraphrase templates only

    for j, case in enumerate(cases):
        target = case["target_true"]
        hits = {(t, b): False for t in range(n_tpl) for b in BUDGETS}
        answers: List[Dict[str, Any]] = []
        for t, prompt in enumerate([case["orig"]] + case["paras"]):
            for b in BUDGETS:
                ans = generate(model, tok, prompt, b)
                ok = scorer._confidence_score(ans, target) >= 1.0
                hits[(t, b)] = ok
                if ok:
                    cells[(t, b)] += 1
                answers.append({"template": t, "budget": b,
                                "probe_prompt": prompt,
                                "probe_answer": ans[:120],
                                "hit": ok})
        for b in BUDGETS:
            union[b] += int(any(hits[(t, b)] for t in range(n_tpl)))
            union_para[b] += int(any(hits[(t, b)] for t in range(1, n_tpl)))
        case["answers"] = answers
        case["hit_union_by_budget"] = {
            str(b): any(answers) and any(
                a["hit"] for a in answers if a["budget"] == b) for b in BUDGETS}
        print(f"[sweep] {j}/{len(cases)} {case['subject'][:30]} "
              f"target={target[:30]!r} union_hits="
              f"{[int(case['hit_union_by_budget'][str(b)]) for b in BUDGETS]}",
              flush=True)

    n = len(cases)
    recall = {
        "per_template_budget": {
            f"template{t}_budget{b}": round(cells[(t, b)] / n, 4)
            for t in range(n_tpl) for b in BUDGETS},
        "union_any_template": {f"budget{b}": round(union[b] / n, 4)
                               for b in BUDGETS},
        "union_paraphrase_only": {f"budget{b}": round(union_para[b] / n, 4)
                                  for b in BUDGETS},
    }
    best_union = max(BUDGETS, key=lambda b: union[b])
    result = {
        "model_key": args.model_key,
        "data_path": args.data_path,
        "n_cases": n,
        "n_paraphrase_templates_used": n_paras,
        "budgets": BUDGETS,
        "recall": recall,
        "best_union_budget": best_union,
        "best_union_recall": round(union[best_union] / n, 4),
        "cases": cases,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("[sweep] recall matrix (template x budget):", flush=True)
    for t in range(n_tpl):
        print("  t%d " % t + " ".join(
            f"b{b}={cells[(t, b)]}/{n}" for b in BUDGETS), flush=True)
    print("  any(all) " + " ".join(f"b{b}={union[b]}/{n}" for b in BUDGETS),
          flush=True)
    print("  any(para) " + " ".join(f"b{b}={union_para[b]}/{n}"
                                    for b in BUDGETS), flush=True)
    print(f"[sweep] best union budget={best_union} "
          f"recall={union[best_union] / n:.2f}", flush=True)
    print(f"[sweep] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


