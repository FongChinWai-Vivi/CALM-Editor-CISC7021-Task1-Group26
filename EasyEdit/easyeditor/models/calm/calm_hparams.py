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
    # "closedform": MEMIT/AlphaEdit closed-form writer (W4 quality path);
    #               ignores rank/hidden/scale/ckpt, see the cf_* block below.
    hypernet_alg: str = "lowrank"
    hypernet_rank: int = 4          # rank r of U @ V^T factors
    hypernet_hidden: int = 512      # hidden width of the generator MLP
    hypernet_scale: float = 1.0     # global scaling of generated deltas
    hypernet_ckpt: Optional[str] = None  # optional trained LowRankHypernetwork
                                         # checkpoint to load at construction
                                         # (None keeps the untrained behaviour)
    mend_hparams_path: Optional[str] = None  # required iff hypernet_alg == "mend"

    # --- closed-form branch (MEMIT/AlphaEdit, W4) ----------------------- #
    # Used only when hypernet_alg == "closedform".  The writer solves
    #   delta = R K^T (K K^T + lambda I)^-1
    # on the MLP c_proj ("rewrite module") weights of ``cf_layers``.
    # NOTE: the last layer must be strictly < n_layer-1, otherwise the z value
    # gradient is zero (MEMIT needs downstream blocks to propagate the write).
    cf_layers: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 8])
    cf_l2: float = 10.0                  # ridge lambda (scale set by cf_l2_mode)
    cf_l2_mode: str = "rel"              # 'rel': lambda = cf_l2 * mean(diag(K K^T))
    cf_v_lr: float = 0.5                 # Adam lr for the z (value) vector
    cf_v_num_grad_steps: int = 20        # z optimisation steps per edit
    cf_v_loss_layer: int = -1            # -1 = last layer (auto-detected)
    cf_v_weight_decay: float = 0.5
    cf_clamp_norm_factor: float = 0.75   # L2 ball on the z delta
    cf_kl_factor: float = 0.0625
    cf_fact_token: str = "subject_last"
    cf_rewrite_module_tmp: str = "transformer.h.{}.mlp.c_proj"
    cf_layer_module_tmp: str = "transformer.h.{}"
    cf_ln_f_module: str = "transformer.ln_f"
    cf_lm_head_module: str = "lm_head"
    cf_kv_cache: bool = False            # AlphaEdit cache_c key statistics
    cf_null_space: str = "none"          # "none" | "stats" (stats not shipped)
    cf_max_contexts: int = 10            # prompt templates feeding keys/residual
    cf_delta_scale: float = 1.0          # multiply the solved delta
    cf_resid_ref: str = "z_layer"        # "z_layer" | "cur_layer" (ablation)
    cf_z_clamp: float = 1.0              # cap |z| at k * |h_z| (0 disables)
    cf_delta_rank: int = 0               # >0: keep top-r directions of delta
    cf_delta_cap: float = -1.0           # >0: cap ||delta||_F at k*||W||_F
    cf_proj_mix: float = 1.0             # null_space='keys' mixture weight
    cf_proj_l2: float = -1.0             # projection ridge (-1 = reuse cf_l2)
    # cf_replay_cache (W6): re-impose the (keys, residual) rows of every chunk
    # already written when solving the next chunk, so chunk k+1 cannot take
    # back the space chunk k occupies.  This is the cumulative re-solve half
    # of the deferred-write fix and is only meaningful with defer_window > 1.
    cf_replay_cache: bool = False
    cf_verbose: bool = False

    # --- deferred (windowed) writes, W6 -------------------------------- #
    # W=1 reproduces the sequential CALM loop: one WRITE per request, solved
    # from a single key set, applied before the next gate decision is taken.
    # W>1 buffers W WRITE requests and solves them jointly (one least-squares
    # problem, one delta per parameter), then re-runs the gate on the flushed
    # state.  Rationale: the sequential loop forbids joint solving, so edits
    # in the same window compete for the same key directions and the oldest
    # writes always lose (measured 0/50 retention at n=500).
    defer_window: int = 1
    # When a flushed edit still misses target_new, mirror it into side memory
    # instead of dropping it (a joint solve cannot roll back one member).
    defer_fallback_route: bool = True
    # defer_chunk: solve a window in contiguous sub-batches of this size
    # instead of one problem over the whole window (0 = one solve).  A single
    # least-squares problem over W keys spreads a fixed delta budget over W
    # constraints, so conditioning (and the hit rate) collapses as W grows:
    # measured in-window hits 30/50 on one slice vs 11/50 on another with the
    # same config.  Chunking gives every sub-batch its own solve and its own
    # fluency guard, which is what turns coverage from 0.23 into a number that
    # does not depend on the luck of the batch composition.
    defer_chunk: int = 0

    # --- side memory branch (teammate Fong Hio Lam modules) ------------ #
    side_memory_path: str = ""      # path to FONG HIO LAM/side_memory_router/side_memory.py
    conflict_gate_path: str = ""    # path to FONG HIO LAM/side_memory_router/conflict_gate.py
    memory_n_shards: int = 64
    memory_max_per_shard: int = 200
    gate_n_recent: int = 50
    gate_write_threshold: float = 0.3
    gate_route_threshold: float = 0.6
    gate_similarity_threshold: float = 0.7

    # --- lifelong-routing policy (W4 fix) ------------------------------- #
    # write_budget: max number of parameter WRITEs allowed over an edit
    # stream (-1 = unlimited).  Once exhausted, every further edit is routed
    # to side memory, preventing unbounded parameter pollution on
    # nearly-all-distinct counterfact streams (the W4 failure mode).
    write_budget: int = -1
    # dual_write: also store WRITE edits into side memory, so CALM's
    # end-to-end retention is backed by memory even for parameter edits.
    dual_write: bool = False
    # verify_commit (Claude #9): tentatively apply WRITE deltas and roll
    # back if the post-edit answer misses target_new, storing the edit in
    # side memory instead.  Prevents failed deltas from polluting weights.
    verify_commit: bool = False
    # --- W6 Rev3: in-loop memory READ ---------------------------------- #
    # memory_read: consult side memory before generation and let a resolved
    # note answer the edit.  Without this the memory channel is write-only:
    # ROUTE stores the truth and every answer still comes from the weights,
    # which is why pre-Rev3 "memory routes" were counted as retained while
    # end-to-end hit_total stayed flat.
    memory_read: bool = False
    # memory_read_guard: dual-channel failure guard.  A note may speak only
    # when the parameter channel fails on the same query and the note is not
    # stale (no parameter write landed after the note was stored).  A live
    # correct parameter answer is never overwritten by a note.
    memory_read_guard: bool = True
    # memory_read_max_new: decode budget for the guarded re-answer.
    memory_read_max_new: int = 5
    # memory_read_template: how the retrieved note is rendered in front of the
    # prompt.  "prefix" = "[s] [r] is [v]" (the recall-probe form, which a
    # heavily edited model ignores: measured 0/32 in-loop vs 0.72 clean),
    # "instruct" = "Answer with v.", "qa" = Q/A frame.
    memory_read_template: str = "instruct"
    # memory_read_arbitrate: when free generation ignores the note, decide the
    # slot by likelihood instead of leaving it unanswered.  The note answers
    # only if log P(note | note-context) - log P(param answer | prompt) exceeds
    # memory_read_arb_margin.  Counted as channel "memory_arb" so copy-style
    # retrieval is never reported as free-generation recall.
    memory_read_arbitrate: bool = True
    memory_read_arb_margin: float = 0.0
    # feature_mode (Claude #3): hypernetwork edit-feature representation.
    #   "mean-emb"  : mean token embedding of "prompt target_new" (original)
    #   "key-hidden": last-subject-token hidden state from the penultimate
    #                 transformer layer (key-conditioned input signal).
    # Training and inference must use the same mode; the ckpt meta stores it.
    feature_mode: str = "mean-emb"

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
