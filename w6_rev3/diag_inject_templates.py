"""Rev3 routing debug: why does the injected note stop working after a stream?

The Day-1 smoke run (o0-99, W=50, memory read on) showed the note being
retrieved and injected for 33 edits but answering 0 of them
(``memory_answered=0``), while the same note on a clean model answers
immediately.  So the failure is state-dependent, and the fix has to be found
on the post-stream weights, not in the plumbing.

This script runs the stream once, then - at the final state, for the rows the
parameter channel fails on - evaluates several injection templates against the
same retrieved note and reports the hit rate of each.  Whichever template wins
is the one the read path should use.

Usage:
  python w6_rev3/diag_inject_templates.py --edits 100 --defer-window 50
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "EasyEdit"))
sys.path.insert(0, os.path.join(ROOT, "FONG HIO LAM", "side_memory_router"))
sys.path.insert(0, os.path.join(ROOT, "EasyEdit", "examples"))

import torch  # noqa: E402

from easyeditor.editors.calm_editor import CALMEditor  # noqa: E402
from easyeditor.models.calm.calm_hparams import CALMHyperParams  # noqa: E402
from w6_defer_stream import load_requests, mean_nll, rewrite_ok  # noqa: E402

HP = os.path.join(ROOT, "EasyEdit", "hparams", "CALM", "gpt2-small.yaml")
TEMPLATES = {
    "prefix_note":       lambda n, s, r, p: f"[{s}] [{r}] is [{n}]\n" + p,
    "note_then_answer":  lambda n, s, r, p: f"Fact: [{s}] [{r}] is [{n}]\n" + p,
    "repeat_note":       lambda n, s, r, p: (f"[{s}] [{r}] is [{n}]\n"
                                             f"[{s}] [{r}] is [{n}]\n" + p),
    "qa_style":          lambda n, s, r, p: (f"Q: what is the {r} of {s}?\n"
                                             f"A: {n}\n" + p),
    "append_note":       lambda n, s, r, p: p + f" The {r} of {s} is",
    "instruct":          lambda n, s, r, p: (f"Answer with {n}.\n" + p),
}


def gen(model, tok, prompt, max_new=5):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=96
              ).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", type=int, default=100)
    ap.add_argument("--data-offset", type=int, default=0)
    ap.add_argument("--defer-window", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(ROOT, "w6_rev3",
                                                  "diag_inject_templates.json"))
    args = ap.parse_args()

    hp = CALMHyperParams.from_hparams(HP)
    hp.hypernet_alg = "closedform"
    hp.cf_layers = [4, 5, 6, 7, 8]
    hp.cf_l2 = 1.0
    hp.cf_l2_mode = "rel"
    hp.cf_z_clamp = 1.0
    hp.cf_max_contexts = 10
    hp.cf_v_num_grad_steps = 20
    hp.defer_window = args.defer_window
    hp.defer_refine_passes = 1
    hp.defer_delta_budget = 0.5
    hp.memory_read = False          # stream exactly as the smoke run wrote it
    ed = CALMEditor.from_hparams(hp)

    requests = load_requests(os.path.join(ROOT, "EasyEdit", "data", "counterfact.json"),
                             args.edits, args.data_offset)
    nll_pre = mean_nll(ed.model, ed.tok, [
        "The capital of France is", "Alan Turing was born in",
        "Water boils at a temperature of", "The novel was written by",
        "In 1990 the population of the city was", "He played for the club until",
        "The film was directed by", "Her first album was released in",
    ])
    metrics, model, extra = ed.edit_requests(requests, sequential_edit=True,
                                             verbose=False)
    nll_post = mean_nll(model, ed.tok, [
        "The capital of France is", "Alan Turing was born in",
        "Water boils at a temperature of", "The novel was written by",
        "In 1990 the population of the city was", "He played for the club until",
        "The film was directed by", "Her first album was released in",
    ])
    print(f"stream done: nll {nll_pre:.4f} -> {nll_post:.4f} "
          f"({nll_post - nll_pre:+.4f})")

    # rows the parameter channel fails on, with whatever memory holds for them
    fail_rows = []
    for i, req in enumerate(requests):
        base = gen(model, ed.tok, req["prompt"])
        if rewrite_ok(base, req["target_new"]):
            continue
        note = ed.memory.lookup(req.get("subject") or "",
                                req.get("relation") or "fact")
        fail_rows.append({"slot": i, "subject": req.get("subject"),
                          "relation": req.get("relation"),
                          "target": req["target_new"],
                          "prompt": req["prompt"], "note": note,
                          "base": base})
    with_note = [r for r in fail_rows if r["note"]]
    print(f"param-fail rows at final state: {len(fail_rows)}, "
          f"with a memory note: {len(with_note)}")

    result = {"edits": args.edits, "defer_window": args.defer_window,
              "nll_pre": round(nll_pre, 4), "nll_post": round(nll_post, 4),
              "param_fail_rows": len(fail_rows),
              "rows_with_note": len(with_note),
              "templates": {}, "samples": {}}
    for name, fn in TEMPLATES.items():
        hits, samples = 0, []
        for r in with_note:
            prompt = fn(r["note"], r["subject"], r["relation"], r["prompt"])
            ans = gen(model, ed.tok, prompt)
            ok = rewrite_ok(ans, r["target"])
            hits += int(ok)
            if len(samples) < 4:
                samples.append({"slot": r["slot"], "target": r["target"],
                                "note": r["note"], "answer": ans, "ok": ok})
        rate = round(hits / max(1, len(with_note)), 4)
        result["templates"][name] = {"hits": hits, "n": len(with_note), "rate": rate}
        result["samples"][name] = samples
        print(f"  {name:18s} {hits}/{len(with_note)} = {rate}")

    result["param_fail_detail"] = fail_rows
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("->", args.out)


if __name__ == "__main__":
    main()
