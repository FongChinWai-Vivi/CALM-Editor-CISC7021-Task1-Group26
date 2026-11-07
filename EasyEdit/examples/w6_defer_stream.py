"""
w6_defer_stream.py -- W6 harness: deferred (windowed) writes + survival@k
(Fong Chin Wai, CISC7021 Task 1 Group 26).

What this measures
------------------
The W4/W6 diagnosis is that the *sequential* CALM loop, not the writer, caps
in-gate retention:

    immediate hit  (W=1, own write still in the window) ......... 100%
    final state    (recheck after the whole stream) ............. 0.6% (1/173)

i.e. a closed-form delta lands target_new when it is applied, and loses the
answer as later writes reuse the same key directions.  This runner turns that
claim into a curve instead of two endpoints:

    immediate hit  : measured right after the window containing the edit lands
    survival@k     : is the edit still answered after k further edits landed?
    final state    : strict recheck of every edit at the end of the stream

Modes (all through the production CALMEditor path, gate + side memory live)
--------------------------------------------------------------------------
    --defer-window 1        sequential baseline (W=1, current behaviour)
    --defer-window 10/25/50 buffered windows, joint closed-form solve
    --replay-cache          cumulative re-solve: earlier windows' (keys,
                            residual) rows are re-imposed in every new solve

Usage (repo root, venv active)
------------------------------
    .venv\\Scripts\\python.exe EasyEdit\\examples\\w6_defer_stream.py --edits 100 \
        --defer-window 1,10,25 --survival-every 10

Every run reports per-mode JSON under ``--out-dir`` with the survival table so
the W6 report can quote a number *and* its scope (W, replay, layers, n).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # EasyEdit/
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_requests(data_path: str, max_edits: int, offset: int = 0) -> List[Dict[str, Any]]:
    with open(data_path, encoding="utf-8") as fh:
        records = json.load(fh)[offset:offset + max_edits]
    requests: List[Dict[str, Any]] = []
    for rec in records:
        rr = rec["requested_rewrite"]
        prompt = rr["prompt"].replace("{}", rr["subject"])
        target_new = (rr["target_new"]["str"] if isinstance(rr["target_new"], dict)
                      else rr["target_new"])
        target_true = (rr["target_true"]["str"] if isinstance(rr["target_true"], dict)
                       else rr["target_true"])
        rephrase: List[str] = []
        for field in ("paraphrase_prompts", "generation_prompts"):
            val = rec.get(field)
            if isinstance(val, str):
                try:
                    val = json.loads(val.replace("\\'", "'"))
                except Exception:
                    val = [val]
            if isinstance(val, list):
                rephrase.extend([v for v in val if isinstance(v, str)])
        requests.append({
            "prompt": prompt,
            "subject": rr["subject"],
            "relation": rr.get("relation_id", "fact"),
            "target_new": target_new,
            "ground_truth": target_true,
            "case_id": rec.get("case_id"),
            "rephrase_prompts": rephrase,
        })
    return requests


# --------------------------------------------------------------------------
# measurement helpers (same protocols as w4_closedform_smoke.py)
# --------------------------------------------------------------------------

PROBE_TEXTS = [
    "The river flows slowly through the wide valley before it reaches the sea.",
    "She opened the window and listened to the rain falling on the roof.",
    "A committee of local residents met on Tuesday to discuss the new library.",
    "The company reported a modest increase in revenue for the third quarter.",
    "Many species of birds migrate south during the colder months of the year.",
    "He studied the map carefully before choosing a route through the mountains.",
    "The museum's collection includes paintings, sculptures, and ancient coins.",
    "After the long discussion, the council agreed to postpone the final vote.",
    "Trains between the two cities run every hour except on public holidays.",
    "The old bridge was repaired last summer after years of structural damage.",
    "Scientists measured the temperature of the lake at several different depths.",
    "The teacher explained the problem again with a simpler example on the board.",
]


@torch.no_grad()
def greedy_answer(model, tok, prompt: str, max_new_tokens: int = 5) -> str:
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=64).input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = out[0, ids.size(1):]
    return tok.decode(gen, skip_special_tokens=True).strip()


def rewrite_ok(answer: str, target: str) -> bool:
    return target.strip().lower() in (answer or "").lower()


def is_degenerate(answer: str) -> bool:
    """Repeated-token / empty answer = fluency damage, not a hit."""
    if not answer or not answer.strip():
        return True
    toks = answer.split()
    return len(toks) >= 4 and len(set(toks)) == 1


@torch.no_grad()
def mean_nll(model, tok, texts) -> float:
    total_tok, total_nll = 0, 0.0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=64).input_ids.to(model.device)
        if ids.size(1) < 3:
            continue
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
        logits = out.logits[:, :-1, :].float()
        tgt = ids[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
        total_nll += float(nll)
        total_tok += tgt.numel()
    return total_nll / max(1, total_tok)


# --------------------------------------------------------------------------
# survival bookkeeping
# --------------------------------------------------------------------------

AGE_BUCKETS = [(1, 10), (11, 25), (26, 50), (51, 10 ** 9)]


def _age_bucket(age: int) -> str:
    for lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
    return "?"


def survival_table(checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn the (checkpoint x edit) hit matrix into survival@k buckets.

    ``age`` = number of edits that landed *after* this edit at the moment of
    the checkpoint, i.e. how much interference the write has survived.
    """
    hits: Dict[str, List[int]] = {}
    for cp in checkpoints:
        at = cp["at_edit"]
        for slot, ok in cp["hits"].items():
            age = at - int(slot)
            if age <= 0:
                continue
            key = _age_bucket(age)
            hits.setdefault(key, []).append(1 if ok else 0)
    table = {}
    for key, vals in hits.items():
        table[key] = {"n": len(vals),
                      "rate": round(sum(vals) / len(vals), 4)}
    # monotone read-out across buckets, in declared order
    ordered = {k: table[k] for k, _ in
               [(_age_bucket(lo), None) for lo, _ in AGE_BUCKETS] if k in table}
    return ordered


# --------------------------------------------------------------------------
# one stream
# --------------------------------------------------------------------------


def run_one(args, defer_window: int, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    from easyeditor.editors.calm_editor import CALMEditor
    from easyeditor.models.calm.calm_hparams import CALMHyperParams

    hp = CALMHyperParams.from_hparams(args.hparams)
    hp.hypernet_alg = "closedform"
    hp.cf_layers = [int(x) for x in args.layers.split(",")]
    hp.cf_l2 = args.l2
    hp.cf_l2_mode = args.l2_mode
    hp.cf_z_clamp = args.z_clamp
    hp.cf_max_contexts = args.contexts
    hp.cf_v_num_grad_steps = args.v_steps
    hp.cf_delta_rank = args.delta_rank
    hp.cf_replay_cache = bool(args.replay_cache and defer_window > 1)
    hp.defer_window = int(defer_window)
    hp.defer_group_keys = bool(args.group_keys)
    hp.defer_chunk = int(args.defer_chunk)
    hp.defer_delta_budget = float(args.delta_budget)
    hp.defer_guard_scale = float(args.guard_scale)
    hp.defer_refine_passes = int(args.refine_passes)
    hp.write_budget = int(args.write_budget)
    hp.dual_write = False              # keep the parameter channel clean

    editor = CALMEditor.from_hparams(hp)
    nll_pre = mean_nll(editor.model, editor.tok, PROBE_TEXTS)

    # ---- checkpoint bookkeeping: recheck every written edit mid-stream ----
    written: List[int] = []
    checkpoints: List[Dict[str, Any]] = []

    def _recheck(slots: List[int], at_edit: int) -> Optional[Dict[str, Any]]:
        if not slots:
            return None
        hits: Dict[int, bool] = {}
        for slot in slots:
            ans = greedy_answer(editor.model, editor.tok, requests[slot]["prompt"])
            hits[slot] = rewrite_ok(ans, requests[slot]["target_new"])
        cp = {"at_edit": at_edit, "n": len(hits),
              "rate": round(sum(hits.values()) / len(hits), 4),
              "hits": hits}
        checkpoints.append(cp)
        return cp

    def _hook(idx: int, metrics: Dict[str, Any], all_metrics: List[Dict[str, Any]]):
        # an edit counts as "written" once a delta for it actually landed in the
        # weights -- immediate (W=1) or via its window flush (W>1)
        written[:] = [i for i, m in enumerate(all_metrics)
                      if m["calm"]["route"] in ("hypernetwork", "hypernetwork_batched")
                      and m["calm"].get("flush_ok", True)]
        if args.survival_every > 0 and (idx + 1) % args.survival_every == 0:
            cp = _recheck(written, idx)
            if cp is not None:
                print(f"    [survival@{idx + 1}] {cp['rate']:.3f} over {cp['n']} writes")

    editor.after_edit_hook = _hook

    t0 = time.time()
    metrics, model, extra = editor.edit_requests(requests, sequential_edit=True,
                                                 verbose=False)

    # ---- final state: strict recheck of every edit ----------------------
    final_rows = []
    for i, req in enumerate(requests):
        ans = greedy_answer(model, editor.tok, req["prompt"])
        final_rows.append({"slot": i, "answer": ans,
                           "ok": rewrite_ok(ans, req["target_new"]),
                           "degenerate": is_degenerate(ans)})
    nll_post = mean_nll(model, editor.tok, PROBE_TEXTS)

    # ---- attribution --------------------------------------------------
    param_rows = [i for i, m in enumerate(metrics)
                  if m["calm"]["route"] in ("hypernetwork", "hypernetwork_batched")]
    fallback_rows = [i for i, m in enumerate(metrics)
                     if m["calm"]["route"] == "batched_route_fallback"]
    mem_rows = [i for i, m in enumerate(metrics)
                if m["calm"]["route"] in ("side_memory", "verify_rollback")]
    reject_rows = [i for i, m in enumerate(metrics)
                   if m["calm"]["route"] == "gate_reject"]

    def _rate(rows: List[int], key: str = "ok") -> Optional[float]:
        if not rows:
            return None
        return round(sum(final_rows[i][key] for i in rows) / len(rows), 4)

    immediate = [1 if metrics[i]["post"].get("rewrite_ok") else 0 for i in param_rows]
    flush_hist = [{"size": f["flush_size"], "groups": f.get("groups"),
                   "refine_passes": f.get("refine_passes"),
                   "immediate_hits": f["immediate_hits"],
                   "seconds": f["seconds"]} for f in extra.get("deferred_flushes", [])]

    summary = {
        "mode": "calm",
        "defer_window": int(defer_window),
        "replay_cache": hp.cf_replay_cache,
        "group_keys": hp.defer_group_keys,
        "chunk": hp.defer_chunk,
        "delta_budget": hp.defer_delta_budget,
        "refine_passes": hp.defer_refine_passes,
        "layers": hp.cf_layers,
        "delta_rank": hp.cf_delta_rank,
        "edits": len(requests),
        "routes": {
            "parameter": len(param_rows),
            "batched_route_fallback": len(fallback_rows),
            "side_memory": len(mem_rows),
            "gate_reject": len(reject_rows),
        },
        # in-gate numbers -- always reported together with their scope
        "immediate_hit": (round(sum(immediate) / len(immediate), 4) if immediate else None),
        "immediate_hit_scope": f"n={len(immediate)} in-gate parameter writes, W={defer_window}",
        "final_state_hit": _rate(param_rows),
        "final_state_hit_scope": f"n={len(param_rows)} in-gate parameter writes, after {len(requests)} edits",
        "final_state_hit_all_edits": round(sum(r["ok"] for r in final_rows) / max(1, len(final_rows)), 4),
        "param_coverage": round(len(param_rows) / max(1, len(final_rows)), 4),
        "param_hit_total": round(sum(final_rows[i]["ok"] for i in param_rows) / max(1, len(final_rows)), 4),
        "survival": survival_table(checkpoints),
        "survival_checkpoints": [{k: v for k, v in cp.items() if k != "hits"} for cp in checkpoints],
        "degenerate_rate": round(sum(r["degenerate"] for r in final_rows) / max(1, len(final_rows)), 4),
        "nll_pre": round(nll_pre, 4),
        "nll_post": round(nll_post, 4),
        "nll_delta": round(nll_post - nll_pre, 4),
        "flush_hist": flush_hist,
        "writer_summary": editor.hypernetwork.summary(),
        "total_seconds": round(time.time() - t0, 1),
    }
    rows = []
    for i, (req, m, fr) in enumerate(zip(requests, metrics, final_rows)):
        rows.append({
            "slot": i,
            "subject": req["subject"],
            "target_new": req["target_new"],
            "route": m["calm"]["route"],
            "post_answer": m["post"].get("answer"),
            "immediate_ok": m["post"].get("rewrite_ok"),
            "final_answer": fr["answer"],
            "final_ok": fr["ok"],
        })
    return {"summary": summary, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="W6 deferred-write / survival@k runner")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "counterfact.json"))
    ap.add_argument("--data-offset", type=int, default=0,
                    help="take records[offset:offset+edits] for a held-out slice")
    ap.add_argument("--hparams", default=os.path.join(ROOT, "hparams", "CALM", "gpt2-small.yaml"))
    ap.add_argument("--edits", type=int, default=100)
    ap.add_argument("--defer-window", default="1,10,25",
                    help="comma list of window sizes to sweep, e.g. 1,10,25,50")
    ap.add_argument("--layers", default="4,5,6,7,8")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--l2-mode", default="rel", choices=["rel", "abs"])
    ap.add_argument("--z-clamp", type=float, default=1.0)
    ap.add_argument("--contexts", type=int, default=10)
    ap.add_argument("--v-steps", type=int, default=20)
    ap.add_argument("--delta-rank", type=int, default=0)
    ap.add_argument("--replay-cache", action="store_true")
    ap.add_argument("--group-keys", action="store_true",
                    help="split a window into key-disjoint sub-batches")
    ap.add_argument("--defer-chunk", type=int, default=0,
                    help="solve a window in contiguous sub-batches of this "
                         "size (0 = one joint solve over the whole window)")
    ap.add_argument("--refine-passes", type=int, default=0,
                    help="re-solve passes for window members still missing")
    ap.add_argument("--delta-budget", type=float, default=0.0,
                    help="max probe-NLL increase a single window may cause (0 disables)")
    ap.add_argument("--guard-scale", type=float, default=0.5,
                    help="delta multiplier retried once before rolling a window back")
    ap.add_argument("--write-budget", type=int, default=-1)
    ap.add_argument("--survival-every", type=int, default=10,
                    help="recheck all writes every N edits (0 disables)")
    ap.add_argument("--out-dir", default=os.path.join(
        REPO, "FONG CHIN WAI", "w6_deferred"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    requests = load_requests(args.data, args.edits, offset=args.data_offset)

    results = {}
    for w in [int(x) for x in args.defer_window.split(",") if x.strip()]:
        print(f"\n=== defer_window={w} (n={len(requests)}, replay={args.replay_cache}) ===")
        # fresh model per window size: weights must not leak across modes
        out = run_one(args, w, requests)
        tag = f"W{w}{'_replay' if args.replay_cache and w > 1 else ''}"
        results[tag] = out["summary"]
        with open(os.path.join(args.out_dir, f"results_defer_{tag}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        s = out["summary"]
        print(f"[{tag}] immediate={s['immediate_hit']} final={s['final_state_hit']} "
              f"survival={s['survival']} NLL {s['nll_pre']}->{s['nll_post']} "
              f"degen={s['degenerate_rate']} ({s['total_seconds']}s)")

    with open(os.path.join(args.out_dir, "results_defer_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("\n" + json.dumps({k: {kk: v[kk] for kk in
                                 ("defer_window", "replay_cache", "immediate_hit",
                                  "final_state_hit", "survival", "nll_delta",
                                  "degenerate_rate")}
                            for k, v in results.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
