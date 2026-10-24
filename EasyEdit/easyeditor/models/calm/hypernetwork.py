"""
hypernetwork.py -- CALM Hypernetwork Edit Generator (W3, Fong Chin Wai).

Implements the MEND/MALMEN-style low-rank weight update generator that the
CALMEditor invokes on the WRITE path.  Given a single
subject-relation-object edit request, the generator maps the edit text into a
low-rank factor pair (U, V) for each editable MLP parameter and returns the
additive delta ``scale * U @ V^T`` -- no per-edit gradient descent on the
full model is required.

Backends
--------
* ``lowrank`` (default): a self-contained MLP hypernetwork.  The edit feature
  is the mean token embedding of ``prompt + target_new`` through the base
  model's input embedding; a shared backbone + per-parameter linear head
  emits U [out, r] and V [in, r] for every inner parameter.
* ``mend``: delegates to EasyEdit's trained MEND archive (GradientTransform /
  IDMLP low-rank factors) via ``MendRewriteExecutor`` -- wired up in
  ``calm_editor.CALMEditor`` when ``hparams.hypernet_alg == "mend"``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from ...util import nethook
from ...util.device import get_model_device


def _matrix_shapes(model: nn.Module, inner_params: List[str]) -> List[Tuple[str, int, int]]:
    """Return ``(name, out_dim, in_dim)`` for each 2-D inner parameter."""
    shapes: List[Tuple[str, int, int]] = []
    for name in inner_params:
        param = nethook.get_parameter(model, name)
        w = param.detach()
        if w.dim() == 2:
            out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
            shapes.append((name, out_dim, in_dim))
        else:
            # 1-D biases are not low-rank factored; skip them.
            print(f"[CALM] skipping non-matrix inner parameter: {name} {tuple(w.shape)}")
    return shapes


class LowRankHypernetwork(nn.Module):
    """MLP hypernetwork that emits rank-r factor pairs for inner parameters.

    Architecture: a shared backbone (``feature -> hidden -> ReLU``) plus one
    linear head per inner parameter.  Each head produces
    ``out_dim * r + in_dim * r`` values that are reshaped into
    U [out, r] and V [in, r]; the generated delta is ``scale * U @ V^T``.
    """

    def __init__(self, feature_dim: int, inner_shapes: List[Tuple[str, int, int]],
                 rank: int = 4, hidden: int = 512, scale: float = 1.0,
                 seed: int = 42):
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank
        self.scale = scale
        self.inner_shapes = inner_shapes

        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
        )
        # nn.ModuleDict keys must not contain ".", so we use positional keys
        # and keep a name -> index map for the forward pass.
        self.heads = nn.ModuleDict()
        self._name_to_idx: Dict[str, int] = {}
        for i, (name, out_dim, in_dim) in enumerate(inner_shapes):
            self.heads[f"head_{i}"] = nn.Linear(hidden, out_dim * rank + in_dim * rank)
            self._name_to_idx[name] = i

        torch.manual_seed(seed)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map an edit feature ``[B, feature_dim]`` to per-parameter deltas."""
        h = self.backbone(feature)                       # [B, hidden]
        deltas: Dict[str, torch.Tensor] = {}
        for name, out_dim, in_dim in self.inner_shapes:
            idx = self._name_to_idx[name]
            raw = self.heads[f"head_{idx}"](h)           # [B, out*r + in*r]
            u = raw[:, : out_dim * self.rank].view(-1, out_dim, self.rank)
            v = raw[:, out_dim * self.rank:].view(-1, in_dim, self.rank)
            delta = self.scale * torch.bmm(u, v.transpose(1, 2))   # [B, out, in]
            deltas[name] = delta.squeeze(0) if delta.size(0) == 1 else delta
        return deltas

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HypernetworkEditGenerator:
    """MEND/MALMEN-style low-rank edit generator bound to a base model.

    Generates weight deltas from an edit request without per-edit gradient
    descent.  The feature representation is the mean token embedding of
    ``prompt + target_new`` through the base model's input embedding.
    """

    def __init__(self, model: nn.Module, tok, inner_params: List[str],
                 rank: int = 4, hidden: int = 512, scale: float = 1.0,
                 seed: int = 42, max_length: int = 64):
        self.model = model
        self.tok = tok
        self.inner_params = list(inner_params)
        self.max_length = max_length

        emb = model.get_input_embeddings()
        feature_dim = int(emb.weight.shape[1])
        shapes = _matrix_shapes(model, inner_params)
        if not shapes:
            raise ValueError("[CALM] hypernetwork: no 2-D inner parameters found.")

        self.hypernet = LowRankHypernetwork(
            feature_dim, shapes, rank=rank, hidden=hidden, scale=scale, seed=seed,
        )
        self.device = get_model_device(model)
        self.hypernet.to(self.device)

    # -- feature construction -------------------------------------------- #

    def edit_feature(self, prompt: str, target_new: str) -> torch.Tensor:
        text = f"{prompt} {target_new}".strip()
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_length).input_ids.to(self.device)
        with torch.no_grad():
            emb = self.model.get_input_embeddings()(ids)   # [1, L, d]
        return emb.mean(dim=1)                             # [1, d]

    # -- public API ------------------------------------------------------- #

    def generate_update(self, request: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Return ``{param_name: delta}`` for one edit request."""
        prompt = request.get("prompt", "")
        target_new = request.get("target_new", "")
        feature = self.edit_feature(prompt, target_new)
        with torch.no_grad():
            deltas = self.hypernet(feature)
        return deltas

    def apply_update(self, deltas: Dict[str, torch.Tensor]) -> None:
        """Add deltas in-place to the base model's inner parameters."""
        with torch.no_grad():
            for name, delta in deltas.items():
                param = nethook.get_parameter(self.model, name)
                param.data.add_(delta.to(device=param.device, dtype=param.dtype))

    def generate_and_apply(self, request: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        deltas = self.generate_update(request)
        self.apply_update(deltas)
        return deltas
