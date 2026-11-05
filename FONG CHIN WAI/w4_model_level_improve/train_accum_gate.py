# W4 model-level improve: accumulation-aware hypernetwork meta-training
# with GATE-DISTRIBUTION sampling.
#
# Extends train_accum.py: within a segment every edit is sampled as a
# parameter WRITE with probability p (default 0.7, matching the observed
# gate decision distribution WRITE=351/500=0.70) and otherwise simulated as
# ROUTE/REJECT (no parameter change, no loss contribution).  This makes the
# training accumulation depth distribution closer to the real stream, where
# hypernetwork WRITEs are interleaved with side-memory ROUTEs and REJECTs.
#
# Usage (from repo root, inside .venv):
#   .venv\\Scripts\\python.exe w4_model_level_improve\\train_accum_gate.py \
#       --train-steps 501 --batch-size 16 --scale 5.0 --write-prob 0.7
#
# Outputs:
#   w4_model_level_improve/ckpt/<--ckpt-name>  (checkpoint)
#   w4_model_level_improve/logs/train_accum_gate.log  (training log)
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

def _find_repo_root(start: Path) -> Path:
    """Walk upwards from ``start`` until a directory containing .git is found."""
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start.parent


REPO = str(_find_repo_root(Path(__file__).resolve().parent))  # repo root
EASYEDIT = os.path.join(REPO, "EasyEdit")
os.chdir(EASYEDIT)
sys.path.insert(0, EASYEDIT)

from easyeditor.editors.calm_editor import CALMEditor
from easyeditor.models.calm.calm_hparams import CALMHyperParams
from easyeditor.util import nethook

HPARAMS = os.path.join(EASYEDIT, "hparams/CALM/gpt2-small.yaml")
COUNTERFACT = os.path.join(EASYEDIT, "data/counterfact.json")
IMPROVE_DIR = str(Path(__file__).resolve().parent)
CKPT_DIR = os.path.join(IMPROVE_DIR, "ckpt")
LOG_DIR = os.path.join(IMPROVE_DIR, "logs")


def build_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W4 accumulation-aware + gate-distribution training")
    parser.add_argument("--train-steps", type=int, default=501)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="segment length: how many edits are drawn per "
                             "segment (WRITE-sampled subset accumulates)")
    parser.add_argument("--write-prob", type=float, default=0.7,
                        help="per-edit probability of a parameter WRITE "
                             "(match gate decision distribution); the rest "
                             "simulate ROUTE/REJECT (no parameter change)")
    parser.add_argument("--sanity-accum-length", type=int, default=10,
                        help="accumulation-aware sanity: sequentially apply "
                             "this many deltas WITHOUT restoring, count how "
                             "many positions still improve (0 disables)")
    parser.add_argument("--max-train-cases", type=int, default=512)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--locality-lambda", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-name", type=str, default="hypernet_accum_gate_gpt2-small.pt")
    return parser.parse_args()


def load_counterfact_subset(path: str, max_cases: int):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    cases = []
    for rec in records:
        rr = rec.get("requested_rewrite", {})
        prompt = rr.get("prompt", "")
        subject = rr.get("subject", "")
        target_new = rr.get("target_new", {}).get("str", "")
        target_true = rr.get("target_true", {}).get("str", "")
        if prompt and target_new and target_true and "{}" in prompt:
            cases.append({
                "prompt": prompt.replace("{}", subject),
                "subject": subject,
                "target_new": target_new,
                "ground_truth": target_true,
            })
        if len(cases) >= max_cases:
            break
    return cases


def set_param(model: nn.Module, name: str, value) -> None:
    """Assign a weight holder for ``name``. Plain tensors are registered as
    buffers so the autograd graph stays alive through the generated delta
    (nn.Parameter(t) would detach it)."""
    parts = name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    attr = parts[-1]
    if attr in mod._parameters:
        del mod._parameters[attr]
    if attr in mod._buffers:
        del mod._buffers[attr]
    setattr(mod, attr, value)


def get_param(model: nn.Module, name: str) -> nn.Parameter:
    return nethook.get_parameter(model, name)


def compute_nll(model, tok, prompt: str, target: str, device: str,
                max_length: int = 64) -> torch.Tensor:
    full_text = f"{prompt} {target}".strip()
    enc = tok(full_text, return_tensors="pt", truncation=True,
              max_length=max_length)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    labels = input_ids.clone()
    prompt_enc = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=max_length)
    n_prompt = int(prompt_enc.input_ids.shape[1])
    labels[:, : n_prompt - 1] = -100
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels)
    return out.loss


def build_original_snapshot(model, inner_params):
    return {name: get_param(model, name) for name in inner_params}


def restore_originals(model, originals):
    for name, orig in originals.items():
        set_param(model, name, orig)


def main():
    args = build_config()
    torch.manual_seed(args.seed)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_lines = []
    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[ACCUM-TRAIN] loading CALM editor from {HPARAMS} ...")
    hparams = CALMHyperParams.from_hparams(HPARAMS)
    editor = CALMEditor.from_hparams(hparams)
    model = editor.model
    tok = editor.tok
    hypernet = editor.hypernetwork.hypernet
    hypernet.train()
    if args.scale != hypernet.scale:
        log(f"[ACCUM-TRAIN] setting hypernetwork scale to {args.scale} "
            f"(was {hypernet.scale})")
        hypernet.scale = args.scale

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log("[ACCUM-TRAIN] WARNING: CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    for p in model.parameters():
        p.requires_grad = False
    for p in hypernet.parameters():
        p.requires_grad = True

    inner_params = list(editor.hparams.inner_params)
    log(f"[ACCUM-TRAIN] inner params ({len(inner_params)}): "
        f"{[p for p in inner_params if p.endswith('weight')][:8]} ...")

    log(f"[ACCUM-TRAIN] loading counterfact subset (max {args.max_train_cases}) ...")
    cases = load_counterfact_subset(COUNTERFACT, args.max_train_cases)
    log(f"[ACCUM-TRAIN] loaded {len(cases)} train cases")

    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=args.lr)
    originals = build_original_snapshot(model, inner_params)

    # ---- accumulation-aware training loop (gate-distribution sampled) --- #
    log(f"[ACCUM-TRAIN] starting {args.train_steps} steps, segment_len="
        f"{args.batch_size}, write_prob={args.write_prob} (WRITE-sampled "
        f"deltas accumulate within segment, restore at end) ...")
    t_start = time.time()
    step_losses = []
    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad()
        idxs = torch.randperm(len(cases))[: args.batch_size].tolist()
        restore_originals(model, originals)      # clean start of segment
        current = {name: originals[name].data for name in inner_params}

        seg_edit = 0.0
        seg_loc = 0.0
        seg_loss = torch.tensor(0.0, device=args.device)
        first_edit = last_edit = 0.0
        first_loc = last_loc = 0.0
        n_write = 0
        for j, i in enumerate(idxs):
            # Gate-decision sampling: WRITE with prob p, else ROUTE/REJECT.
            do_write = (torch.rand(1).item() < args.write_prob)
            if not do_write:
                continue
            case = cases[i]
            feature = editor.hypernetwork.edit_feature(
                case["prompt"], case["target_new"])

            loc_before = 0.0
            if args.locality_lambda > 0.0:
                with torch.no_grad():
                    loc_before = compute_nll(
                        model, tok, case["prompt"], case["ground_truth"],
                        args.device, args.max_length).item()

            deltas = hypernet(feature)
            for name, delta in deltas.items():
                base = current[name]
                new_w = base.data + delta.to(base.device, base.dtype)
                set_param(model, name, new_w)
                current[name] = new_w.detach()

            edit_loss = compute_nll(model, tok, case["prompt"],
                                    case["target_new"], args.device,
                                    args.max_length)
            seg_edit += float(edit_loss.item())

            loc_loss = torch.tensor(0.0, device=args.device)
            if args.locality_lambda > 0.0:
                after = compute_nll(model, tok, case["prompt"],
                                    case["ground_truth"], args.device,
                                    args.max_length)
                loc_loss = torch.clamp(after - loc_before, min=0.0)
                seg_loc += float(loc_loss.item())

            seg_loss = seg_loss + edit_loss + args.locality_lambda * loc_loss
            n_write += 1

            if n_write == 1:
                first_edit = float(edit_loss.item())
                first_loc = float(loc_loss.item())
            last_edit = float(edit_loss.item())
            last_loc = float(loc_loss.item())

        # One optimizer step per segment (skip empty segments defensively).
        if n_write > 0:
            seg_loss = seg_loss / n_write
            seg_loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
            optimizer.step()
        restore_originals(model, originals)
        step_losses.append(float(seg_loss.item()))

        if step % 10 == 0 or step == 1:
            log(f"  step {step:4d}/{args.train_steps}  avg_loss="
                f"{float(seg_loss.item()):.4f}  writes={n_write}/{args.batch_size} "
                f"edit[first->last]={first_edit:.4f}->{last_edit:.4f}  "
                f"loc[first->last]={first_loc:.4f}->{last_loc:.4f}")

    t_train = time.time() - t_start
    log(f"[ACCUM-TRAIN] training done in {t_train:.1f}s "
        f"({t_train / args.train_steps:.3f}s/step)")
    if step_losses:
        log(f"[ACCUM-TRAIN] loss first={step_losses[0]:.4f} "
            f"last={step_losses[-1]:.4f} min={min(step_losses):.4f}")

    # ---- sanity: NLL(target_new) before vs after patch on 20 cases ------- #
    log("[ACCUM-TRAIN] sanity check: NLL(target_new) before vs after patch "
        "on 20 cases (clean baseline)")
    hypernet.eval()
    improved = 0
    checked = 0
    for case in cases[:20]:
        with torch.no_grad():
            before = compute_nll(model, tok, case["prompt"],
                                 case["target_new"], args.device,
                                 args.max_length).item()
            feature = editor.hypernetwork.edit_feature(case["prompt"],
                                                       case["target_new"])
            deltas = hypernet(feature)
            for name, delta in deltas.items():
                base = originals[name]
                set_param(model, name, base.data + delta.to(base.device, base.dtype))
            after = compute_nll(model, tok, case["prompt"], case["target_new"],
                                args.device, args.max_length).item()
            restore_originals(model, originals)
        checked += 1
        if after < before:
            improved += 1
        log(f"    case {checked}: nll {before:.3f} -> {after:.3f} "
            f"({'improved' if after < before else 'worse'})")
    log(f"[ACCUM-TRAIN] sanity: {improved}/{checked} cases improved after patch")

    # ---- accumulation-aware sanity -------------------------------------- #
    if args.sanity_accum_length > 0:
        log(f"[ACCUM-TRAIN] accum sanity: sequentially apply "
            f"{args.sanity_accum_length} deltas without restore ...")
        hypernet.eval()
        restore_originals(model, originals)
        accum_current = {name: originals[name].data for name in inner_params}
        pos_improved = []
        for j in range(args.sanity_accum_length):
            case = cases[j]
            with torch.no_grad():
                before = compute_nll(model, tok, case["prompt"],
                                     case["target_new"], args.device,
                                     args.max_length).item()
                feature = editor.hypernetwork.edit_feature(case["prompt"],
                                                           case["target_new"])
                deltas = hypernet(feature)
                for name, delta in deltas.items():
                    base = accum_current[name]
                    set_param(model, name,
                              base.data + delta.to(base.device, base.dtype))
                    accum_current[name] = base.data + delta.to(base.device, base.dtype)
                after = compute_nll(model, tok, case["prompt"],
                                    case["target_new"], args.device,
                                    args.max_length).item()
            ok = after < before
            pos_improved.append(int(ok))
            log(f"    pos {j + 1:2d}: nll {before:.3f} -> {after:.3f} "
                f"({'improved' if ok else 'worse'})")
        restore_originals(model, originals)
        log(f"[ACCUM-TRAIN] accum sanity: improved positions "
            f"{sum(pos_improved)}/{args.sanity_accum_length} "
            f"pattern={pos_improved}")

    # ---- save checkpoint ------------------------------------------------ #
    ckpt_path = os.path.join(CKPT_DIR, args.ckpt_name)
    ckpt = {
        "meta": {
            "rank": hypernet.rank,
            "hidden": hypernet.backbone[0].out_features,
            "scale": hypernet.scale,
            "feature_dim": hypernet.feature_dim,
            "inner_params": list(inner_params),
            "train_steps": args.train_steps,
            "lr": args.lr,
            "max_train_cases": len(cases),
            "seed": args.seed,
            "training": "accumulation-aware+gate-distribution",
            "segment_len": args.batch_size,
            "write_prob": args.write_prob,
            "locality_lambda": args.locality_lambda,
        },
        "state_dict": hypernet.state_dict(),
    }
    torch.save(ckpt, ckpt_path)
    log(f"[ACCUM-TRAIN] checkpoint saved: {ckpt_path}")

    log_path = os.path.join(LOG_DIR, "train_accum_gate.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"[ACCUM-TRAIN] log written: {log_path}")

    log("[ACCUM-TRAIN] done")


if __name__ == "__main__":
    main()
