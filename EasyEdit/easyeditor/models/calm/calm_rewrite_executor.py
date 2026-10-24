"""
calm_rewrite_executor.py -- stateless CALM executor (ALG_DICT registration).

``ALG_DICT["CALM"]`` needs a callable with the standard EasyEdit signature
``apply_to_model(model, tok, requests, hparams, ...)`` so that the generic
``BaseEditor.__init__`` can instantiate CALMEditor.  CALMEditor overrides its
own edit flow (gate -> hypernetwork / side-memory / reject) in
``editors/calm_editor.py``; this executor is the minimal stateless fallback
that applies low-rank hypernetwork updates directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from ...util import nethook
from .calm_hparams import CALMHyperParams
from .hypernetwork import HypernetworkEditGenerator


class CALMRewriteExecutor:
    """Stateless CALM rewrite executor for the EasyEdit ALG registry."""

    def __init__(self):
        self.is_init = False

    def apply_to_model(
        self,
        model,
        tok,
        requests: List[Dict[str, Any]],
        hparams: Any,
        copy: bool = False,
        return_orig_weights: bool = False,
        keep_original_weight: bool = False,
        **kwargs,
    ):
        hp = hparams if isinstance(hparams, CALMHyperParams) else CALMHyperParams(**hparams)
        generator = HypernetworkEditGenerator(
            model, tok, hp.inner_params,
            rank=hp.hypernet_rank,
            hidden=hp.hypernet_hidden,
            scale=hp.hypernet_scale,
            seed=hp.seed,
            max_length=getattr(hp, "max_length", 64),
        )
        weights_copy: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for request in requests:
                deltas = generator.generate_update(request)
                for name, delta in deltas.items():
                    param = nethook.get_parameter(model, name)
                    if return_orig_weights and name not in weights_copy:
                        weights_copy[name] = param.detach().clone()
                    param.data.add_(delta.to(device=param.device, dtype=param.dtype))
        self.is_init = True
        return model, weights_copy
