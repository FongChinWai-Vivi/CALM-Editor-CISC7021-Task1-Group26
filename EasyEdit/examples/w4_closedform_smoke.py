"""
w4_closedform_smoke.py -- W4 skeleton runner for the MEMIT/AlphaEdit closed-form writer.

Purpose
-------
Measure the *writer* in isolation (no gate, no memory) so the W4 question
"does the closed-form delta actually land target_new in the model?" is answered
before any routing policy is layered on top.  This mirrors the 128-write audit
discipline: report per-edit ``pre_answer`` / ``post_answer`` / ``rewrite_ok``
plus the delta norms, never a single aggregate number.

Modes
-----
direct : load gpt2-small, build ClosedFormEditGenerator, for each edit
         compute -> apply -> measure.  Isolates writer quality.
calm   : build CALMEditor with ``hypernet_alg=closedform`` so the gate routing
         runs too (end-to-end smoke of the swap inside the production path).

Usage (repo root, venv active) ::

    .venv\\Scripts\\python.exe EasyEdit\\examples\\w4_closedform_smoke.py --edits 16
    .venv\\Scripts\\python.exe EasyEdit\\examples\\w4_closedform_smoke.py --mode calm --edits 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # EasyEdit/
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # CALM yaml paths are relative to the EasyEdit root


def load_requests(data_path: str, max_edits: int) -> List[Dict[str, Any]]:
    with open(data_path, encoding="utf-8") as fh:
        records = json.load(fh)[:max_edits]
    requests: List[Dict[str, Any]] = []
    for rec in records:
        rr = rec["requested_rewrite"]
        prompt = rr["prompt"].replace("{}", rr["subject"])
        target_new = rr["target_new"]["str"] if isinstance(rr["target_new"], dict) else rr["target_new"]
        target_true = rr["target_true"]["str"] if isinstance(rr["target_true"], dict) else rr["target_true"]
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
        neighbor: List[str] = []
        nval = rec.get("neighborhood_prompts")
        if isinstance(nval, str):
            try:
                nval = json.loads(nval.replace("\\'", "'"))
            except Exception:
                nval = [nval]
        if isinstance(nval, list):
            neighbor.extend([v for v in nval if isinstance(v, str) and v.strip()])
        requests.append({
            "prompt": prompt,
            "subject": rr["subject"],
            "relation": rr.get("relation_id", "fact"),
            "target_new": target_new,
            "ground_truth": target_true,
            "case_id": rec.get("case_id"),
            "rephrase_prompts": rephrase,
            "neighborhood_prompts": neighbor,
        })
    return requests


@torch.no_grad()
def greedy_answer(model, tok, prompt: str, max_new_tokens: int = 5) -> str:
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=64).input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = out[0, ids.size(1):]
    return tok.decode(gen, skip_special_tokens=True).strip()


def rewrite_ok(answer: str, target: str) -> bool:
    return target.strip().lower() in answer.lower()


# --------------------------------------------------------------------------
# locality / fluency probes
# --------------------------------------------------------------------------
# Fixed neutral English prose (never edited, never factual targets of this
# stream).  Mean token NLL over these sentences is the cheapest honest proxy
# for "did the sequential writes damage the model?"; the absolute value does
# not matter, only its drift from the pre-edit baseline.
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

DEGEN_MARKERS = ("\u200b",)


def mean_nll(model, tok, texts):
    """Mean per-token negative log-likelihood over ``texts`` (lower = better)."""
    total_tok, total_nll = 0, 0.0
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
        ids = enc.input_ids.to(model.device)
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


def heldout_accuracy(model, tok, items) -> float:
    """Fraction of *untouched* facts the model still answers correctly."""
    if not items:
        return float("nan")
    ok = sum(rewrite_ok(greedy_answer(model, tok, it["prompt"]), it["ground_truth"])
             for it in items)
    return ok / len(items)


def top1_snapshot(model, tok, prompts) -> Dict[str, tuple]:
    """Next-token top-1 id and its logprob for each prompt (no generation)."""
    snap: Dict[str, tuple] = {}
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True,
                      max_length=64).input_ids.to(model.device)
            logits = model(input_ids=ids).logits[0, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            top = int(logp.argmax())
            snap[p] = (top, float(logp[top]))
    return snap


def locality_drift(before: Dict[str, tuple], after: Dict[str, tuple]) -> Dict[str, object]:
    """Top-1 stability on neighbourhood prompts.

    The earlier locality metric compared held-out *exact-match* accuracy, whose
    pre-edit baseline is 0.0 on this stream (the 5-token greedy answer never
    equals the ground truth), so ``after/before`` was undefined and the number
    was meaningless.  Top-1 stability is defined at every point of the stream
    and reads 1.0 on an untouched model, so any drop is real collateral damage.
    """
    keys = [p for p in before if p in after]
    if not keys:
        return {"n": 0, "top1_unchanged": None, "mean_logprob_delta": None}
    same = sum(1 for p in keys if before[p][0] == after[p][0])
    dp = [after[p][1] - before[p][1] for p in keys]
    return {
        "n": len(keys),
        "top1_unchanged": round(same / len(keys), 4),
        "mean_top1_logprob_delta": round(sum(dp) / len(dp), 4),
    }


def first_token_accuracy(model, tok, items) -> float:
    """Fraction of held-out facts whose next token equals the true first token."""
    if not items:
        return float("nan")
    ok = 0
    with torch.no_grad():
        for it in items:
            ids = tok(it["prompt"], return_tensors="pt", truncation=True,
                      max_length=64).input_ids.to(model.device)
            top = int(model(input_ids=ids).logits[0, -1].float().argmax())
            want = tok(" " + it["ground_truth"].strip(),
                       add_special_tokens=False).input_ids
            if want and top == want[0]:
                ok += 1
    return ok / len(items)


def is_degenerate(answer: str) -> bool:
    """Repeated-token / empty-output detector for the 5-token greedy answer."""
    toks = [t for t in answer.replace("\\n", " ").split() if t.strip()]
    if not toks:
        return True
    if any(m in answer for m in DEGEN_MARKERS):
        return True
    # >=4 identical consecutive tokens, e.g. "Lisbon Lisbon Lisbon Lisbon"
    run = 1
    for prev, cur in zip(toks, toks[1:]):
        run = run + 1 if cur.lower() == prev.lower() else 1
        if run >= 4:
            return True
    return False


def run_direct(args, requests) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from easyeditor.models.calm.calm_hparams import CALMHyperParams
    from easyeditor.models.calm.closed_form_writer import ClosedFormEditGenerator

    hp = CALMHyperParams.from_hparams(args.hparams)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(hp.model_name).to(device).eval()
    tok = AutoTokenizer.from_pretrained(hp.tokenizer_name or hp.model_name)
    tok.pad_token = tok.eos_token

    writer = ClosedFormEditGenerator(
        model, tok,
        inner_params=hp.inner_params,
        layers=[int(x) for x in args.layers.split(",")],
        l2=args.l2,
        l2_mode=args.l2_mode,
        delta_scale=args.scale,
        resid_ref=args.resid_ref,
        z_clamp=args.z_clamp,
        max_contexts=args.contexts,
        v_num_grad_steps=args.v_steps,
        v_lr=args.v_lr,
        v_weight_decay=(args.v_decay if args.v_decay is not None else 0.5),
        kv_cache=args.kv_cache,
        null_space=args.null_space,
        proj_mix=args.proj_mix,
        proj_l2=args.proj_l2,
        delta_rank=args.delta_rank,
        delta_cap=args.delta_cap,
        replay_cache=args.replay_cache,
        device=device,
        verbose=args.verbose,
    )

    heldout = []
    if args.locality_n:
        heldout = load_requests(args.data, args.edits + args.locality_n)[args.edits:]
    heldout_before = heldout_accuracy(model, tok, heldout) if heldout else float("nan")
    heldout_top1_before = first_token_accuracy(model, tok, heldout) if heldout else float("nan")

    # Neighbourhood prompts: the edited facts' own neighbourhood sentences plus
    # every prompt of the held-out facts.  None of them is ever edited, so a
    # stable top-1 on them is the definition of locality.
    nb_prompts: List[str] = []
    for r in list(requests) + list(heldout):
        for pr in (r.get("neighborhood_prompts") or [])[:2]:
            if pr not in nb_prompts:
                nb_prompts.append(pr)
    nb_before = top1_snapshot(model, tok, nb_prompts) if nb_prompts else {}
    if nb_prompts:
        print(f"[locality] snapshot of {len(nb_prompts)} neighbourhood prompts taken")

    nll_before = mean_nll(model, tok, PROBE_TEXTS)
    print(f"[probe] pre-edit mean NLL = {nll_before:.4f}")
    probe_curve = [{"after_edits": 0, "probe_nll": round(nll_before, 4), "delta": 0.0}]

    rows: List[Dict[str, Any]] = []
    t_start = time.time()
    batch = max(1, int(args.batch_solve))
    cursor = 0
    while cursor < len(requests):
        chunk = requests[cursor: cursor + batch]
        pres = [greedy_answer(model, tok, r["prompt"]) for r in chunk]
        t0 = time.time()
        deltas = writer.generate_update(chunk[0]) if len(chunk) == 1 \
            else writer.generate_update_batch(chunk)
        writer.apply_update(deltas)
        apply_stats = writer.last_apply_stats
        for j, req in enumerate(chunk):
            i = cursor + j
            pre = pres[j]
            post = greedy_answer(model, tok, req["prompt"])
            row = {
                "index": i,
                "case_id": req.get("case_id"),
                "subject": req["subject"],
                "prompt": req["prompt"],
                "target_new": req["target_new"],
                "pre_answer": pre,
                "post_answer": post,
                "rewrite_ok": rewrite_ok(post, req["target_new"]),
                "pre_ok": rewrite_ok(pre, req["target_new"]),
                "degenerate": is_degenerate(post),
                "seconds": round((time.time() - t0) / len(chunk), 2),
                "writer": writer.summary(),
                "apply_stats": apply_stats,
            }
            rows.append(row)
            print(f"[{i + 1}/{len(requests)}] {req['subject']!r} -> {req['target_new']!r} | "
                  f"pre={pre!r} post={post!r} ok={row['rewrite_ok']} "
                  f"degen={row['degenerate']} ({row['seconds']}s)")
        cursor += len(chunk)

        if args.probe_every and cursor % args.probe_every == 0 and cursor < len(requests):
            cur = mean_nll(model, tok, PROBE_TEXTS)
            probe_curve.append({"after_edits": cursor, "probe_nll": round(cur, 4),
                                "delta": round(cur - nll_before, 4)})
            print(f"[probe] after {cursor} edits: NLL={cur:.4f} (delta {cur - nll_before:+.4f})")

    nll_after = mean_nll(model, tok, PROBE_TEXTS)
    probe_curve.append({"after_edits": len(rows), "probe_nll": round(nll_after, 4),
                        "delta": round(nll_after - nll_before, 4)})
    print(f"[probe] post-edit mean NLL = {nll_after:.4f} (delta {nll_after - nll_before:+.4f})")

    recheck = {}
    if args.recheck_n:
        old = requests[:min(args.recheck_n, len(requests))]
        still = sum(rewrite_ok(greedy_answer(model, tok, r["prompt"]), r["target_new"]) for r in old)
        recheck = {"n": len(old), "still_ok": still, "retention": round(still / len(old), 4)}
        print(f"[recheck] first {len(old)} edits still answered correctly: {still}/{len(old)}")

    locality: Dict[str, Any] = {}
    if args.locality_n:
        heldout_after = heldout_accuracy(model, tok, heldout)
        heldout_top1_after = first_token_accuracy(model, tok, heldout)
        nb_after = top1_snapshot(model, tok, nb_prompts) if nb_prompts else {}
        drift = locality_drift(nb_before, nb_after) if nb_before else {}
        sample = requests[:min(args.locality_n, len(requests))]
        gen_hit = gen_tot = 0
        for r in sample:
            for pr in (r.get("rephrase_prompts") or [])[:2]:
                gen_tot += 1
                gen_hit += rewrite_ok(greedy_answer(model, tok, pr), r["target_new"])
        locality = {
            "heldout_n": len(heldout),
            "heldout_acc_before": round(heldout_before, 4),
            "heldout_acc_after": round(heldout_after, 4),
            "heldout_retention": round(heldout_after / heldout_before, 4) if heldout_before else None,
            "heldout_first_token_acc_before": round(heldout_top1_before, 4),
            "heldout_first_token_acc_after": round(heldout_top1_after, 4),
            "neighborhood": drift,
            "generality_n": gen_tot,
            "generality": round(gen_hit / gen_tot, 4) if gen_tot else None,
        }
        print(f"[locality] neighbourhood top1_unchanged={drift.get('top1_unchanged')} "
              f"dlogp={drift.get('mean_top1_logprob_delta')} | "
              f"heldout first-token acc {heldout_top1_before:.3f} -> {heldout_top1_after:.3f} | "
              f"paraphrase generality {locality['generality']}")

    summary = {
        "mode": "direct",
        "locality": locality,
        "recheck": recheck,
        "hparams": args.hparams,
        "edits": len(rows),
        "writer_summary": writer.summary(),
        "rewrite_ok_rate": sum(r["rewrite_ok"] for r in rows) / max(1, len(rows)),
        "pre_ok_rate": sum(r["pre_ok"] for r in rows) / max(1, len(rows)),
        "degenerate_rate": sum(r["degenerate"] for r in rows) / max(1, len(rows)),
        "probe_nll_before": round(nll_before, 4),
        "probe_nll_after": round(nll_after, 4),
        "probe_nll_delta": round(nll_after - nll_before, 4),
        "probe_curve": probe_curve,
        "mean_seconds": sum(r["seconds"] for r in rows) / max(1, len(rows)),
        "total_seconds": round(time.time() - t_start, 1),
    }
    return {"rows": rows, "summary": summary}


def run_calm(args, requests) -> Dict[str, Any]:
    from easyeditor.editors.calm_editor import CALMEditor
    from easyeditor.models.calm.calm_hparams import CALMHyperParams

    hp = CALMHyperParams.from_hparams(args.hparams)
    hp.hypernet_alg = "closedform"
    hp.cf_layers = [int(x) for x in args.layers.split(",")]
    hp.cf_l2 = args.l2
    hp.cf_l2_mode = args.l2_mode
    hp.cf_delta_scale = args.scale
    hp.cf_resid_ref = args.resid_ref
    hp.cf_z_clamp = args.z_clamp
    hp.cf_max_contexts = args.contexts
    hp.cf_v_num_grad_steps = args.v_steps
    hp.cf_v_lr = args.v_lr
    if args.v_decay is not None:
        hp.cf_v_weight_decay = args.v_decay
    hp.cf_kv_cache = args.kv_cache
    hp.cf_null_space = args.null_space
    hp.cf_proj_mix = args.proj_mix
    hp.cf_proj_l2 = args.proj_l2
    hp.cf_verbose = args.verbose

    # ---- side-memory channel (teammate Fong Hio Lam's modules) -------- #
    # The dual-channel attribution in the W4/W6 reports requires the side
    # memory to be *live*: without these two paths CALMEditor has no store
    # and every ROUTE/dual-write decision silently degrades to "no channel",
    # which would make the per-channel numbers meaningless.
    if args.side_memory_dir:
        sm_dir = args.side_memory_dir
        hp.side_memory_path = os.path.join(sm_dir, "side_memory.py")
        hp.conflict_gate_path = os.path.join(sm_dir, "conflict_gate.py")
    hp.dual_write = bool(args.dual_write)
    hp.write_budget = int(args.write_budget)
    hp.verify_commit = bool(args.verify_commit)

    editor = CALMEditor.from_hparams(hp)
    t_start = time.time()
    metrics, _model, extra = editor.edit_requests(requests, sequential_edit=True, verbose=False)

    # Re-score every edit with the exact same 5-token greedy protocol used by
    # --mode direct.  CALM's internal evaluator uses long-form generations with
    # its own matching rule, so its numbers are not comparable to the direct
    # stream; this makes the two modes directly comparable.
    strict_rows = []
    for req in requests:
        ans = greedy_answer(_model, editor.tok, req["prompt"])
        strict_rows.append({"rewrite_ok": rewrite_ok(ans, req["target_new"]),
                            "degenerate": is_degenerate(ans), "answer": ans})

    rows: List[Dict[str, Any]] = []
    for i, (req, m) in enumerate(zip(requests, metrics)):
        rows.append({
            "index": i,
            "case_id": req.get("case_id"),
            "subject": req["subject"],
            "prompt": req["prompt"],
            "target_new": req["target_new"],
            "pre_answer": m["pre"]["answer"],
            "post_answer": m["post"]["answer"],
            "rewrite_ok": bool(m["post"]["rewrite_ok"]),
            "pre_ok": bool(m["pre"]["rewrite_ok"]),
            "route": m["calm"]["route"],
            "decision": m["calm"]["decision"],
            "memory_updated": bool(m["calm"].get("memory_updated")),
            "strict_answer": strict_rows[i]["answer"],
            "strict_ok": strict_rows[i]["rewrite_ok"],
        })
        print(f"[{i + 1}/{len(rows)}] {req['subject']!r} -> {req['target_new']!r} | "
              f"route={m['calm']['route']} ok={rows[-1]['rewrite_ok']}")

    writes = [r for r in rows if r["route"] == "hypernetwork"]

    # ---- per-channel attribution (W6 requirement) --------------------- #
    # A "channel" is where the answer came from: the parameters (hypernetwork
    # WRITE, optionally mirrored into side memory by dual_write) or the side
    # memory (ROUTE, and verify_commit rollbacks which end up as ROUTE).
    # Reporting a single aggregate rewrite rate across both channels is
    # exactly the conflation this split exists to prevent.
    def _block(rs):
        if not rs:
            return {"n": 0, "ok": 0, "rate": None}
        ok = sum(1 for r in rs if r["strict_ok"])
        return {"n": len(rs), "ok": ok, "rate": round(ok / len(rs), 4)}

    by_route = {}
    for k in sorted({r["route"] for r in rows}):
        by_route[k] = _block([r for r in rows if r["route"] == k])

    param_rows = [r for r in rows if r["route"] == "hypernetwork"]
    mem_rows = [r for r in rows if r["route"] in ("side_memory", "verify_rollback")]
    channels = {
        "parameter_only": _block([r for r in param_rows if not r.get("memory_updated")]),
        "parameter_dual_write": _block([r for r in param_rows if r.get("memory_updated")]),
        "side_memory": _block(mem_rows),
        "unrouted": _block([r for r in rows
                            if r["route"] not in ("hypernetwork", "side_memory",
                                                  "verify_rollback")]),
        "dual_write_enabled": bool(args.dual_write),
        "side_memory_live": bool(hp.side_memory_path),
    }

    summary = {
        "mode": "calm",
        "hparams": args.hparams,
        "edits": len(rows),
        "writer_summary": editor.hypernetwork.summary(),
        "routes": {k: sum(1 for r in rows if r["route"] == k)
                   for k in sorted({r["route"] for r in rows})},
        "write_real_hit_rate": (sum(r["rewrite_ok"] for r in writes) / len(writes)) if writes else None,
        "rewrite_ok_rate": sum(r["rewrite_ok"] for r in rows) / max(1, len(rows)),
        "gate_summary": extra.get("gate_summary"),
        "memory_stats": extra.get("memory_stats"),
        "by_route": by_route,
        "channels": channels,
        "strict_greedy_hit_rate": sum(r["rewrite_ok"] for r in strict_rows) / max(1, len(strict_rows)),
        "strict_greedy_degenerate_rate": sum(r["degenerate"] for r in strict_rows) / max(1, len(strict_rows)),
        "total_seconds": round(time.time() - t_start, 1),
    }
    return {"rows": rows, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description="W4 closed-form writer smoke/stream runner")
    ap.add_argument("--mode", choices=["direct", "calm"], default="direct")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "counterfact.json"))
    ap.add_argument("--hparams", default=os.path.join(ROOT, "hparams", "CALM", "gpt2-small.yaml"))
    ap.add_argument("--edits", type=int, default=16)
    ap.add_argument("--layers", default="4,5,6,7,8")
    ap.add_argument("--l2", type=float, default=1.0,
                    help="ridge lambda, i.e. MEMIT mom2_update_weight (NOT a small L2)")
    ap.add_argument("--l2-mode", default="rel", choices=["rel", "abs"],
                    help="rel: lambda = l2 * mean(diag(K K^T)); abs: lambda = l2")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply solved deltas")
    ap.add_argument("--resid-ref", default="z_layer", choices=["z_layer", "cur_layer"])
    ap.add_argument("--z-clamp", type=float, default=1.0, help="cap |z| at k*|h_z| (0 disables)")
    ap.add_argument("--contexts", type=int, default=10,
                    help="number of prompt templates for the key/residual solve (1 = rank-1, unstable)")
    ap.add_argument("--v-steps", type=int, default=20)
    ap.add_argument("--v-lr", type=float, default=0.5)
    ap.add_argument("--v-decay", type=float, default=None)
    ap.add_argument("--null-space", default="none", choices=["none", "keys"],
                    help="keys: AlphaEdit-lite projection onto the null space of already-written keys")
    ap.add_argument("--proj-mix", type=float, default=1.0,
                    help="0..1 blend between the raw update and its null-space projection")
    ap.add_argument("--proj-l2", type=float, default=-1.0,
                    help="ridge inside the projector (larger = weaker projection; -1 reuses --l2)")
    ap.add_argument("--kv-cache", action="store_true",
                    help="AlphaEdit-style running key statistics in the solve")
    ap.add_argument("--delta-rank", type=int, default=0,
                    help="SVD-truncate each solved delta to this rank (0 = full rank)")
    ap.add_argument("--delta-cap", type=float, default=-1.0,
                    help="cap ||delta||_F at k*||W||_F per layer (<=0 disables)")
    ap.add_argument("--batch-solve", type=int, default=1,
                    help="joint closed-form solve over chunks of N edits (1 = per-edit, MEMIT default)")
    ap.add_argument("--replay-cache", action="store_true",
                    help="re-impose every previously written chunk's constraints in later solves")
    ap.add_argument("--probe-every", type=int, default=16,
                    help="measure the neutral-prose NLL probe every N edits (0 = only before/after)")
    ap.add_argument("--locality-n", type=int, default=0,
                    help="held-out facts + paraphrase sample used for locality/generality")
    ap.add_argument("--recheck-n", type=int, default=0,
                    help="re-query the first N edited prompts at the end of the stream")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the z-value Adam (default: layer-dependent, as in MEMIT)")
    ap.add_argument("--side-memory-dir", default="",
                    help="dir holding side_memory.py + conflict_gate.py "
                         "(teammate modules); empty = CALM runs without a memory channel")
    ap.add_argument("--dual-write", action="store_true",
                    help="also store WRITE edits in side memory (dual channel)")
    ap.add_argument("--write-budget", type=int, default=-1,
                    help="max parameter WRITEs over the stream (-1 = unlimited)")
    ap.add_argument("--verify-commit", action="store_true",
                    help="roll back a WRITE that misses target_new and store it in memory")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        REPO, "FONG CHIN WAI", "w4_model_level_improve", "results",
        f"closedform_smoke_{args.mode}_n{args.edits}_L{args.layers.replace(',', '-')}_l2{args.l2:g}",
    )
    os.makedirs(out_dir, exist_ok=True)

    requests = load_requests(args.data, args.edits)
    result = run_direct(args, requests) if args.mode == "direct" else run_calm(args, requests)

    with open(os.path.join(out_dir, "per_edit.jsonl"), "w", encoding="utf-8") as fh:
        for row in result["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(result["summary"], fh, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"\nartifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
