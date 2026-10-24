"""
calm_hparams.py -- CALM hyper-parameters (W3, Fong Chin Wai).

Mirrors EasyEdit's ``HyperParams`` convention (YAML/JSON backed dataclass)
and holds everything CALMEditor needs: model, hypernetwork branch, side
memory branch and conflict-gate branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from ...util.hparams import HyperParams, load_hparams_config, normalize_alg_name


@dataclass
class CALMHyperParams(HyperParams):
    """Hyper-parameters container for the CALM editor (BaseEditor subclass)."""

    alg_name: str = "CALM"
    model_name: str = ""
    model_class: str = "GPT2LMHeadModel"
    tokenizer_class: str = "GPT2Tokenizer"
    tokenizer_name: str = ""
    inner_params: List[str] = field(default_factory=list)
    device: Any = 0
    batch_size: int = 1
    max_length: int = 64

    # --- hypernetwork branch ------------------------------------------- #
    # "lowrank" (default): self-contained MLP low-rank generator.
    # "mend": wraps an EasyEdit-trained MEND archive via MendRewriteExecutor.
    hypernet_alg: str = "lowrank"
    hypernet_rank: int = 4          # rank r of U @ V^T factors
    hypernet_hidden: int = 512      # hidden width of the generator MLP
    hypernet_scale: float = 1.0     # global scaling of generated deltas
    mend_hparams_path: Optional[str] = None  # required iff hypernet_alg == "mend"

    # --- side memory branch (teammate Fong Hio Lam modules) ------------ #
    side_memory_path: str = ""      # path to FONG HIO LAM/side_memory_router/side_memory.py
    conflict_gate_path: str = ""    # path to FONG HIO LAM/side_memory_router/conflict_gate.py
    memory_n_shards: int = 64
    memory_max_per_shard: int = 200
    gate_n_recent: int = 50
    gate_write_threshold: float = 0.3
    gate_route_threshold: float = 0.6
    gate_similarity_threshold: float = 0.7

    # --- misc ----------------------------------------------------------- #
    results_dir: str = "./results"
    seed: int = 42
    max_output_length: Optional[int] = None
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None
    model_parallel: bool = False
    fp16: bool = False
    padding_side: Optional[str] = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str) -> "CALMHyperParams":
        config = load_hparams_config(hparams_name_or_path)
        config = normalize_alg_name(config, "CALM")
        return cls(**config)
