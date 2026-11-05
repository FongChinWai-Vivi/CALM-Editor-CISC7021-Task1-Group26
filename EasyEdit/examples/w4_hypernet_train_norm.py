# W4 hypernetwork meta-training with norm-growth regularization (Kimi plan).
#
# Extends w3_hypernet_train.py with three training-side fixes targeting the
# sequential (lifelong) edit regime diagnosed in W4:
#
#   1. --norm-tau: differentiable norm cap.  Emitted deltas are scaled down
#      (with a differentiable multiplier) whenever ||delta||_F exceeds
#      tau * ||W_orig||_F, so the hypernetwork is forced to fit edits into
#      compact deltas instead of the ~1000x blow-up observed in the
#      500-edit stream (raw relative-Frobenius penalty explodes numerically
#      because base norms are tiny; a cap is stable and directly targets the
#      runaway cumulative weight growth).
#
#   2. --pollute-max:  randomized pollution depth.  Each step first applies
#      a random number d of *no-grad* edits (d drawn Uniform over the
#      depth ladder) to the inner params, mimicking an already-edited model
#      state; the target edit is then applied on top.  The hypernetwork
#      learns to write into a polluted state instead of a clean one.
#
#   3. --retention-lambda: in-segment retention.  When d > 0, after the
#      target edit we re-measure the NLL of the *first* polluted edit's
#      target_new and penalize an increase (the classic "old edit gets
#      overwritten" failure mode of sequential streams).
#
# Base model stays frozen; gradients flow only into the hypernetwork.
# The original parameter snapshot is restored after every step.
#
# Usage (from the EasyEdit root, inside the project .venv):
#   python examples/w4_hypernet_train_norm.py --train-steps 800 --norm-lambda 1e-3 \
#       --retention-lambda 0.2 --pollute-max 96 \
#       --init-ckpt <best W3 ckpt> --ckpt-dir <out>
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

EASYEDIT = str(Path(__file__).resolve().parent.parent)
os.chdir(EASYEDIT)
sys.path.insert(0, EASYEDIT)

from easyeditor.editors.calm_editor import CALMEditor
from easyeditor.models.calm.calm_hparams import CALMHyperParams
from easyeditor.util import nethook

HPARAMS = os.path.join(EASYEDIT, "hparams/CALM/gpt2-small.yaml")
COUNTERFACT = os.path.join(EASYEDIT, "data/counterfact.json")

# Kimi's suggested pollution-depth ladder.
POLLUTE_DEPTHS = [0, 16, 32, 48, 96]


def build_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W4 hypernetwork meta-training")
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-cases", type=int, default=512)
    parser.add_argument("--scale", type=float, default=10.0,
                        help="global scale of generated deltas during training "
                             "(amplified so the low-rank factors measurably "
                             "affect the frozen base model)")
    parser.add_argument("--locality-lambda", type=float, default=0.1,
                        help="weight of the ground-truth NLL hinge penalty")
    parser.add_argument("--norm-tau", type=float, default=2.0,
                        help="differentiable norm cap: ||delta||_F <= tau * "
                             "||W_orig||_F; try 0.5..4.0 (was norm-lambda)")
    parser.add_argument("--retention-lambda", type=float, default=0.2,
                        help="weight of in-segment retention penalty "
                             "(re-measure first polluted edit after target write)")
    parser.add_argument("--pollute-max", type=int, default=96,
                        help="max pollution depth; depths drawn from ladder "
                             "[0,16,32,48,96] clipped to <= pollute-max")
    parser.add_argument("--init-ckpt", type=str, default="",
                        help="optional warm-start checkpoint (best W3 ckpt)")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--ckpt-dir", type=str,
                        default=os.path.join(EASYEDIT, "results/models/CALM"))
    parser.add_argument("--log-out", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_counterfact_subset(path: str, max_cases: int):
    """Return a list of edit cases from counterfact.json (requested_rewrite)."""
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
    """Assign a weight holder for ``name`` keeping the autograd graph.

    - ``value`` is a Parameter -> registered back into ``_parameters``.
    - ``value`` is a plain tensor -> registered as a buffer, which is what
      makes gradients flow from the forward pass back to the hypernetwork.
    """
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
    """NLL (as a loss tensor) of ``target`` conditioned on ``prompt``."""
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
    """Save the original Parameter objects so they can be restored verbatim."""
    return {name: get_param(model, name) for name in inner_params}


def apply_deltas_cur(model, deltas, cur_params):
    """Apply deltas on top of the current (possibly polluted) param values.

    ``cur_params`` maps param name -> current tensor; it is updated in place.
    The replacement keeps the autograd graph: gradients flow through the
    generated delta back to the hypernetwork (the base model is frozen).
    """
    for name, delta in deltas.items():
        cur = cur_params[name]
        new_w = cur + delta.to(cur.device, cur.dtype)
        set_param(model, name, new_w)
        cur_params[name] = new_w


def scale_down_deltas(deltas, originals, tau):
    """Differentiable norm cap: keep ||delta||_F <= tau * ||W_orig||_F.

    Returns (scaled_deltas, ratios) where ratios[name] is the pre-cap
    relative norm ||delta||/||W|| for monitoring.
    """
    scaled = {}
    ratios = {}
    for name, delta in deltas.items():
        orig_w = originals[name].data
        base_norm = orig_w.norm(p="fro")
        delta_norm = delta.norm(p="fro")
        ratios[name] = float(delta_norm.item()) / (float(base_norm.item()) + 1e-12)
        factor = torch.clamp(tau * base_norm / (delta_norm + 1e-12), max=1.0)
        scaled[name] = delta * factor
    return scaled, ratios


def restore_originals(model, originals):
    for name, orig in originals.items():
        set_param(model, name, orig)


def main():
    args = build_config()
    torch.manual_seed(args.seed)

    log_lines = []
    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[W4-TRAIN] loading CALM editor from {HPARAMS} ...")
    hparams = CALMHyperParams.from_hparams(HPARAMS)
    editor = CALMEditor.from_hparams(hparams)
    model = editor.model
    tok = editor.tok
    hypernet = editor.hypernetwork.hypernet
    hypernet.train()
    if args.init_ckpt:
        data = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        state = data.get("state_dict", data)
        hypernet.load_state_dict(state)
        # Warm start: adopt the checkpoint's trained scale unless the user
        # explicitly overrides --scale (avoids silently doubling deltas).
        ckpt_scale = data.get("meta", {}).get("scale")
        if ckpt_scale is not None and args.scale == 10.0:
            log(f"[W4-TRAIN] warm start: adopting ckpt scale {ckpt_scale}")
            hypernet.scale = float(ckpt_scale)
        log(f"[W4-TRAIN] warm-started from {args.init_ckpt}")
    if args.scale != hypernet.scale and not (args.init_ckpt and args.scale == 10.0 and ckpt_scale is not None):
        log(f"[W4-TRAIN] setting hypernetwork scale to {args.scale} "
            f"(was {hypernet.scale})")
        hypernet.scale = args.scale

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log("[W4-TRAIN] WARNING: CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    # Freeze the base model entirely: only the hypernetwork parameters train.
    for p in model.parameters():
        p.requires_grad = False
    for p in hypernet.parameters():
        p.requires_grad = True

    inner_params = list(editor.hparams.inner_params)
    log(f"[W4-TRAIN] inner params ({len(inner_params)}): "
        f"{[p for p in inner_params if p.endswith('weight')][:8]} ...")

    log(f"[W4-TRAIN] loading counterfact subset (max {args.max_train_cases}) ...")
    cases = load_counterfact_subset(COUNTERFACT, args.max_train_cases)
    log(f"[W4-TRAIN] loaded {len(cases)} train cases")

    depths = [d for d in POLLUTE_DEPTHS if d <= args.pollute_max]
    log(f"[W4-TRAIN] pollution depth ladder: {depths}")

    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=args.lr)
    originals = build_original_snapshot(model, inner_params)

    # ---- training loop -------------------------------------------------- #
    log(f"[W4-TRAIN] starting {args.train_steps} training steps "
        f"(batch={args.batch_size}, lr={args.lr}, "
        f"locality_lambda={args.locality_lambda}, norm_tau={args.norm_tau}, "
        f"retention_lambda={args.retention_lambda}) ...")
    t_start = time.time()
    step_losses = []
    ratio_hist = []
    ret_pen_hist = []
    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=args.device)
        n_batch = 0
        step_ratio = 0.0
        step_ret_pen = 0.0
        # Randomly sample one batch per step.
        idxs = torch.randperm(len(cases))[: args.batch_size].tolist()
        for i in idxs:
            case = cases[i]
            feature = editor.hypernetwork.edit_feature(
                case["prompt"], case["target_new"])

            # Locality baseline: ground-truth NLL on the *unpatched* model.
            loc_before = 0.0
            if args.locality_lambda > 0.0:
                with torch.no_grad():
                    loc_before = compute_nll(
                        model, tok, case["prompt"], case["ground_truth"],
                        args.device, args.max_length).item()

            # --- Phase 1: random pollution (no-grad sequential edits) ----
            depth = int(torch.tensor(depths)[torch.randint(0, len(depths), (1,))].item())
            polluters = []
            cur_params = {name: get_param(model, name).data for name in originals}
            if depth > 0:
                pidxs = torch.randperm(len(cases))[: depth].tolist()
                polluters = [cases[j] for j in pidxs]
                with torch.no_grad():
                    for pc in polluters:
                        pfeat = editor.hypernetwork.edit_feature(
                            pc["prompt"], pc["target_new"])
                        pdeltas, _ = scale_down_deltas(hypernet(pfeat), originals, args.norm_tau)
                        apply_deltas_cur(model, pdeltas, cur_params)

            # --- Phase 2: target edit on top of polluted state ------------
            deltas, ratios = scale_down_deltas(hypernet(feature), originals, args.norm_tau)
            apply_deltas_cur(model, deltas, cur_params)
            step_ratio += sum(ratios.values()) / max(len(ratios), 1)

            edit_loss = compute_nll(model, tok, case["prompt"],
                                    case["target_new"], args.device,
                                    args.max_length)

            # Locality penalty (ground truth should not get worse).
            loc_loss = torch.tensor(0.0, device=args.device)
            if args.locality_lambda > 0.0:
                after = compute_nll(model, tok, case["prompt"],
                                    case["ground_truth"], args.device,
                                    args.max_length)
                loc_loss = torch.clamp(after - loc_before, min=0.0)

            # Retention penalty: first polluted edit's target NLL after write.
            ret_loss = torch.tensor(0.0, device=args.device)
            if args.retention_lambda > 0.0 and polluters:
                first = polluters[0]
                ret_loss = compute_nll(model, tok, first["prompt"],
                                       first["target_new"], args.device,
                                       args.max_length)
                step_ret_pen += float(ret_loss.item())

            restore_originals(model, originals)

            case_loss = (edit_loss
                         + args.locality_lambda * loc_loss
                         + args.retention_lambda * ret_loss)
            total_loss = total_loss + case_loss
            n_batch += 1

        total_loss = total_loss / n_batch
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
        optimizer.step()
        step_losses.append(float(total_loss.item()))
        ratio_hist.append(step_ratio / n_batch)
        ret_pen_hist.append(step_ret_pen / n_batch)

        if step % 10 == 0 or step == 1:
            log(f"  step {step:4d}/{args.train_steps}  loss={float(total_loss.item()):.4f}  "
                f"edit_nll={float(edit_loss.item()):.4f}  "
                f"loc_pen={float(loc_loss.item()):.4f}  "
                f"ratio={ratio_hist[-1]:.3f}  ret_pen={ret_pen_hist[-1]:.4f}")

    t_train = time.time() - t_start
    log(f"[W4-TRAIN] training done in {t_train:.1f}s "
        f"({t_train / args.train_steps:.3f}s/step)")
    if step_losses:
        log(f"[W4-TRAIN] loss first={step_losses[0]:.4f} "
            f"last={step_losses[-1]:.4f} "
            f"min={min(step_losses):.4f}")
        log(f"[W4-TRAIN] ratio first={ratio_hist[0]:.3f} "
            f"last={ratio_hist[-1]:.3f}  "
            f"ret_pen first={ret_pen_hist[0]:.4f} last={ret_pen_hist[-1]:.4f}")

    # ---- save checkpoint ------------------------------------------------ #
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.ckpt_dir,
        f"hypernet_tau{args.norm_tau}_ret{args.retention_lambda}_"
        f"poll{args.pollute_max}_steps{args.train_steps}_gpt2-small.pt",
    )
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
            "norm_tau": args.norm_tau,
            "retention_lambda": args.retention_lambda,
            "pollute_max": args.pollute_max,
            "seed": args.seed,
            "init_ckpt": args.init_ckpt,
        },
        "state_dict": hypernet.state_dict(),
    }
    torch.save(ckpt, ckpt_path)
    log(f"[W4-TRAIN] checkpoint saved: {ckpt_path}")

    if args.log_out:
        with open(args.log_out, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        log(f"[W4-TRAIN] log written: {args.log_out}")

    log("[W4-TRAIN] done")


if __name__ == "__main__":
    main()
