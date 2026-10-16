from copy import deepcopy
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.device import normalize_device
from .dpo_hparams import DPOHyperParams

def apply_dpo_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DPOHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    """
    weights_copy = {}
    if copy:
        # If you need to copy the model, handle it here
        pass  # Avoid deep copying to save memory

    device = normalize_device(getattr(hparams, "device", None))
    print(f"Using device: {device}")

    # Configure LoRA
    Config = LoraConfig

    peft_config = Config(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules
    )
    # Add LoRA modules to the model
    peft_model = get_peft_model(model, peft_config)

    # Manually set only LoRA parameters to be trainable
    for name, param in peft_model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if getattr(model, "hf_device_map", None) is None:
        peft_model.to(device)
    else:
        peft_model.is_parallelizable = True
        peft_model.model_parallel = True

    # Execute the DPO algorithm
    edited_model = execute_dpo(peft_model, tok, requests, hparams)

    return edited_model, weights_copy


def execute_dpo(
        peft_model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DPOHyperParams,
        **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the DPO algorithm for the specified updates.
    """
    peft_model.train()
    device = next(peft_model.parameters()).device

    # Define the optimizer
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    loss_meter = AverageMeter()

    # Prepare data
    texts = [r["prompt"] for r in requests]
    targets_pos = [r["target_new"] for r in requests]  # Positive samples
    targets_neg = [r["target_neg"] for r in requests]  # Negative samples

    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt_batch, tgt_pos_batch, tgt_neg_batch in zip(
                chunks(texts, hparams.batch_size),
                chunks(targets_pos, hparams.batch_size),
                chunks(targets_neg, hparams.batch_size),
        ):
            mask_token = -100
            opt.zero_grad()

            # Build inputs with labels only on completion tokens.
            tokens_pos = build_completion_batch(tok, txt_batch, tgt_pos_batch, device, mask_token)
            tokens_neg = build_completion_batch(tok, txt_batch, tgt_neg_batch, device, mask_token)

            # Compute outputs with LoRA modules (current model)
            outputs_pos = peft_model(**tokens_pos)
            outputs_neg = peft_model(**tokens_neg)

            # Compute outputs for the reference model (disable LoRA modules)
            peft_model.eval()  # Switch to evaluation mode
            peft_model.disable_adapter_layers()  # Disable LoRA layers

            try:
                with torch.no_grad():
                    ref_outputs_pos = peft_model(**tokens_pos)
                    ref_outputs_neg = peft_model(**tokens_neg)
            finally:
                peft_model.enable_adapter_layers()  # Enable LoRA layers
                peft_model.train()  # Switch back to training mode

            # Compute losses
            lora_loss = outputs_pos.loss
            beta = hparams.beta

            policy_log_probs_pos = completion_log_probs(outputs_pos.logits, tokens_pos["labels"], mask_token)
            policy_log_probs_neg = completion_log_probs(outputs_neg.logits, tokens_neg["labels"], mask_token)
            ref_log_probs_pos = completion_log_probs(ref_outputs_pos.logits, tokens_pos["labels"], mask_token)
            ref_log_probs_neg = completion_log_probs(ref_outputs_neg.logits, tokens_neg["labels"], mask_token)

            dpo_advantage = beta * (
                (policy_log_probs_pos - ref_log_probs_pos) -
                (policy_log_probs_neg - ref_log_probs_neg)
            )
            dpo_loss = -F.logsigmoid(dpo_advantage).mean()

            # Total loss
            loss = hparams.alpha * lora_loss + (1 - hparams.alpha) * dpo_loss

            loss.backward()
            opt.step()

            bs = len(txt_batch)
            loss_meter.update(loss.item(), n=bs)

        print(f"Total loss {loss_meter.avg}")

    return peft_model


def build_completion_batch(tok: AutoTokenizer, prompts: List[str], targets: List[str], device, mask_token: int):
    """
    Tokenize prompt + completion pairs and mask labels outside completion spans.
    DPO positive and negative completions may have different token lengths, so
    the preference objective must operate on per-example sequence logprobs.
    """
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    old_padding_side = tok.padding_side
    tok.padding_side = "right"
    try:
        full_texts = [f"{prompt} {target}" for prompt, target in zip(prompts, targets)]
        prefix_texts = [f"{prompt} " for prompt in prompts]
        try:
            tokens = tok(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
            )
            offset_mapping = tokens.pop("offset_mapping")
        except (NotImplementedError, TypeError):
            tokens = tok(full_texts, return_tensors="pt", padding=True, truncation=True)
            offset_mapping = None
        labels = torch.full_like(tokens["input_ids"], mask_token)

        for row, prefix_text in enumerate(prefix_texts):
            seq_len = int(tokens["attention_mask"][row].sum().item())
            if offset_mapping is not None:
                prefix_len = len(prefix_text)
                for col in range(seq_len):
                    start, end = offset_mapping[row][col].tolist()
                    if end > prefix_len and end > start:
                        labels[row, col] = tokens["input_ids"][row, col]
            else:
                prefix_ids = tok(prefix_text, add_special_tokens=True, truncation=True)["input_ids"]
                completion_start = min(len(prefix_ids), seq_len)
                if completion_start >= seq_len:
                    prompt_ids = tok(prompts[row], add_special_tokens=True, truncation=True)["input_ids"]
                    completion_start = min(len(prompt_ids), max(seq_len - 1, 0))
                labels[row, completion_start:seq_len] = tokens["input_ids"][row, completion_start:seq_len]

        tokens["labels"] = labels
        return tokens.to(device)
    finally:
        tok.padding_side = old_padding_side


def completion_log_probs(logits: torch.Tensor, labels: torch.Tensor, mask_token: int) -> torch.Tensor:
    """Return summed log probability of the labeled completion tokens."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    label_mask = shift_labels.ne(mask_token)
    safe_labels = shift_labels.masked_fill(~label_mask, 0)
    token_log_probs = shift_logits.log_softmax(-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_log_probs * label_mask).sum(-1)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i:i + n]
