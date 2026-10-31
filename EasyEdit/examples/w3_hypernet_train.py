# W3 hypernetwork meta-training pipeline (MEND-style).
#
# Meta-trains the existing LowRankHypernetwork (easyeditor/models/calm/
# hypernetwork.py) on a counterfact subset.  For each sampled edit request
# (prompt, subject, target_new, ground_truth) the generator emits a low-rank
# delta that is TEMPORARILY applied to the gpt2-small inner MLP parameters
# (last 3 layers' c_fc / c_proj, aligned with W1 MEND).  The edit loss is the
# NLL of target_new under the patched model; an optional lightweight locality
# term penalises an increase in the NLL of ground_truth after the patch.
# Gradients flow only into the hypernetwork parameters -- the base model is
# never updated -- and the original parameter snapshot is restored after every
# step.
#
# Usage (from the EasyEdit root, inside the project .venv):
#   python examples/w3_hypernet_train.py --train-steps 501 --batch-size 2
#
# Outputs:
#   EasyEdit/results/models/CALM/hypernet_gpt2-small.pt   (checkpoint)
#   stdout training log (also written to --log-out, optional)
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


def build_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W3 hypernetwork meta-training")
    parser.add_argument("--train-steps", type=int, default=501)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-cases", type=int, default=512)
    parser.add_argument("--scale", type=float, default=10.0,
                        help="global scale of generated deltas during training "
                             "(amplified so the low-rank factors measurably "
                             "affect the frozen base model)")
    parser.add_argument("--locality-lambda", type=float, default=0.1,
                        help="weight of the ground-truth NLL hinge penalty")
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
      makes gradients flow from the forward pass back to the hypernetwork
      (``nn.Parameter(t)`` would detach the graph, see W3 diagnosis).
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
    # Only the target span is supervised; the final prompt token predicts the
    # first target token.
    labels[:, : n_prompt - 1] = -100
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels)
    return out.loss


def build_original_snapshot(model, inner_params):
    """Save the original Parameter objects so they can be restored verbatim."""
    return {name: get_param(model, name) for name in inner_params}


def apply_deltas(model, deltas, originals):
    """Temporarily replace inner params with ``orig.data + delta``.

    The replacement keeps the autograd graph: gradients flow through the
    generated delta back to the hypernetwork (the base model is frozen).
    """
    for name, delta in deltas.items():
        orig = originals[name]
        new_w = orig.data + delta.to(orig.device, orig.dtype)
        set_param(model, name, new_w)


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

    log(f"[W3-TRAIN] loading CALM editor from {HPARAMS} ...")
    hparams = CALMHyperParams.from_hparams(HPARAMS)
    editor = CALMEditor.from_hparams(hparams)
    model = editor.model
    tok = editor.tok
    hypernet = editor.hypernetwork.hypernet
    hypernet.train()
    if args.scale != hypernet.scale:
        log(f"[W3-TRAIN] setting hypernetwork scale to {args.scale} "
            f"(was {hypernet.scale})")
        hypernet.scale = args.scale

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log("[W3-TRAIN] WARNING: CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    # Freeze the base model entirely: only the hypernetwork parameters train.
    for p in model.parameters():
        p.requires_grad = False
    for p in hypernet.parameters():
        p.requires_grad = True

    inner_params = list(editor.hparams.inner_params)
    log(f"[W3-TRAIN] inner params ({len(inner_params)}): "
        f"{[p for p in inner_params if p.endswith('weight')][:8]} ...")

    log(f"[W3-TRAIN] loading counterfact subset (max {args.max_train_cases}) ...")
    cases = load_counterfact_subset(COUNTERFACT, args.max_train_cases)
    log(f"[W3-TRAIN] loaded {len(cases)} train cases")

    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=args.lr)
    originals = build_original_snapshot(model, inner_params)
    loss_fn = nn.CrossEntropyLoss()

    # ---- training loop -------------------------------------------------- #
    log(f"[W3-TRAIN] starting {args.train_steps} training steps "
        f"(batch={args.batch_size}, lr={args.lr}, "
        f"locality_lambda={args.locality_lambda}) ...")
    t_start = time.time()
    step_losses = []
    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=args.device)
        n_batch = 0
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

            deltas = hypernet(feature)          # dict {name: delta [out,in]}
            apply_deltas(model, deltas, originals)

            edit_loss = compute_nll(model, tok, case["prompt"],
                                    case["target_new"], args.device,
                                    args.max_length)

            loc_loss = torch.tensor(0.0, device=args.device)
            if args.locality_lambda > 0.0:
                after = compute_nll(model, tok, case["prompt"],
                                    case["ground_truth"], args.device,
                                    args.max_length)
                loc_loss = torch.clamp(after - loc_before, min=0.0)

            restore_originals(model, originals)

            case_loss = edit_loss + args.locality_lambda * loc_loss
            total_loss = total_loss + case_loss
            n_batch += 1

        total_loss = total_loss / n_batch
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
        optimizer.step()
        step_losses.append(float(total_loss.item()))

        if step % 10 == 0 or step == 1:
            avg_case = float(edit_loss.item())
            avg_loc = float(loc_loss.item())
            log(f"  step {step:4d}/{args.train_steps}  loss={float(total_loss.item()):.4f}  "
                f"edit_nll={avg_case:.4f}  loc_pen={avg_loc:.4f}")

    t_train = time.time() - t_start
    log(f"[W3-TRAIN] training done in {t_train:.1f}s "
        f"({t_train / args.train_steps:.3f}s/step)")
    if step_losses:
        log(f"[W3-TRAIN] loss first={step_losses[0]:.4f} "
            f"last={step_losses[-1]:.4f} "
            f"min={min(step_losses):.4f}")

    # ---- sanity check: edit NLL before/after on 20 held-out train cases --- #
    log("[W3-TRAIN] sanity check: NLL(target_new) before vs after patch "
        "on 20 sampled cases")
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
            apply_deltas(model, deltas, originals)
            after = compute_nll(model, tok, case["prompt"], case["target_new"],
                                args.device, args.max_length).item()
            restore_originals(model, originals)
        checked += 1
        if after < before:
            improved += 1
        log(f"    case {checked}: nll {before:.3f} -> {after:.3f} "
            f"({'improved' if after < before else 'worse'})")
    log(f"[W3-TRAIN] sanity: {improved}/{checked} cases improved after patch")

    # ---- save checkpoint ------------------------------------------------ #
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "hypernet_gpt2-small.pt")
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
        },
        "state_dict": hypernet.state_dict(),
    }
    torch.save(ckpt, ckpt_path)
    log(f"[W3-TRAIN] checkpoint saved: {ckpt_path}")

    if args.log_out:
        with open(args.log_out, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        log(f"[W3-TRAIN] log written: {args.log_out}")

    log("[W3-TRAIN] done")


if __name__ == "__main__":
    main()
