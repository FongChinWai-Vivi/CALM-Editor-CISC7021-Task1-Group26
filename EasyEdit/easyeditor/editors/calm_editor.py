"""
calm_editor.py -- CALMEditor: EasyEdit BaseEditor subclass (W3, Fong Chin Wai).

Implements the frozen W1 interface (``calm/calm_editor.py`` blueprint,
``docs/w1_alignment_decision.md`` §2) as a real EasyEdit ``BaseEditor``
subclass.  Every edit request is routed through the three CALM components:

    1. ConflictGate (teammate Fong Hio Lam)   -> WRITE / ROUTE / REJECT;
    2. HypernetworkEditGenerator (this work package) -> low-rank weight update;
    3. SideMemory (teammate Fong Hio Lam)     -> non-parametric routing store.

The gate and memory modules live in the ``FONG HIO LAM/side_memory_router/``
folder and are loaded at runtime from the absolute paths given in the CALM
hparams (``side_memory_path`` / ``conflict_gate_path``), so the EasyEdit code
base is never coupled to a hard-coded teammate directory.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .editor import BaseEditor
from ..models.calm.calm_hparams import CALMHyperParams
from ..models.calm.hypernetwork import HypernetworkEditGenerator
from ..util import nethook


def _load_module_from_path(name: str, path: str):
    """Import a Python module from an absolute file path (teammate code)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"[CALM] module path does not exist: {path}")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"[CALM] cannot load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # required for dataclass / forward ref resolution
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class CALMEditor(BaseEditor):
    """Unified CALM editing entry point (EasyEdit BaseEditor subclass)."""

    @classmethod
    def from_hparams(cls, hparams: CALMHyperParams):
        """EasyEdit-compatible constructor: receive a resolved HyperParams
        object (as ``BaseEditor.from_hparams`` does) and build the editor."""
        return cls(hparams)

    def __init__(self, hparams: CALMHyperParams):
        super().__init__(hparams)

        # ---- teammate components (loaded from source paths) ----
        side_memory_mod = _load_module_from_path("calm_side_memory", hparams.side_memory_path)
        conflict_gate_mod = _load_module_from_path("calm_conflict_gate", hparams.conflict_gate_path)
        self.SideMemory = side_memory_mod.SideMemory
        self.ConflictGate = conflict_gate_mod.ConflictGate
        self.Decision = conflict_gate_mod.Decision
        self.EditRecord = conflict_gate_mod.EditRecord

        self.memory = self.SideMemory(
            n_shards=hparams.memory_n_shards,
            max_per_shard=hparams.memory_max_per_shard,
            seed=hparams.seed,
        )
        self.gate = self.ConflictGate(
            n_recent=hparams.gate_n_recent,
            write_threshold=hparams.gate_write_threshold,
            route_threshold=hparams.gate_route_threshold,
            similarity_threshold=hparams.gate_similarity_threshold,
            seed=hparams.seed,
        )

        # ---- hypernetwork edit generator ----
        if hparams.hypernet_alg == "mend":
            from ..models.mend.mend_hparams import MENDHyperParams
            from ..models.mend.mend_main import MendRewriteExecutor
            if not hparams.mend_hparams_path:
                raise ValueError("[CALM] hypernet_alg='mend' requires mend_hparams_path.")
            self.mend_params = MENDHyperParams.from_hparams(hparams.mend_hparams_path)
            self.mend_executor = MendRewriteExecutor()
            self.hypernetwork = None
        elif hparams.hypernet_alg == "closedform":
            # W4 quality path: MEMIT/AlphaEdit closed-form writer.  Same
            # generate_update / apply_update contract as the low-rank
            # hypernetwork, so the routing logic below is unchanged.
            from ..models.calm.closed_form_writer import ClosedFormEditGenerator
            self.hypernetwork = ClosedFormEditGenerator(
                self.model, self.tok,
                inner_params=hparams.inner_params,
                layers=getattr(hparams, "cf_layers", None),
                l2=getattr(hparams, "cf_l2", 10.0),
                l2_mode=getattr(hparams, "cf_l2_mode", "rel"),
                delta_scale=getattr(hparams, "cf_delta_scale", 1.0),
                resid_ref=getattr(hparams, "cf_resid_ref", "z_layer"),
                z_clamp=getattr(hparams, "cf_z_clamp", 1.0),
                max_contexts=getattr(hparams, "cf_max_contexts", 10),
                v_lr=getattr(hparams, "cf_v_lr", 0.5),
                v_num_grad_steps=getattr(hparams, "cf_v_num_grad_steps", 20),
                v_loss_layer=getattr(hparams, "cf_v_loss_layer", -1),
                v_weight_decay=getattr(hparams, "cf_v_weight_decay", 0.5),
                clamp_norm_factor=getattr(hparams, "cf_clamp_norm_factor", 0.75),
                kl_factor=getattr(hparams, "cf_kl_factor", 0.0625),
                fact_token=getattr(hparams, "cf_fact_token", "subject_last"),
                rewrite_module_tmp=getattr(hparams, "cf_rewrite_module_tmp",
                                           "transformer.h.{}.mlp.c_proj"),
                layer_module_tmp=getattr(hparams, "cf_layer_module_tmp",
                                         "transformer.h.{}"),
                ln_f_module=getattr(hparams, "cf_ln_f_module", "transformer.ln_f"),
                lm_head_module=getattr(hparams, "cf_lm_head_module", "lm_head"),
                kv_cache=getattr(hparams, "cf_kv_cache", False),
                null_space=getattr(hparams, "cf_null_space", "none"),
                proj_mix=getattr(hparams, "cf_proj_mix", 1.0),
                proj_l2=getattr(hparams, "cf_proj_l2", -1.0),
                delta_rank=getattr(hparams, "cf_delta_rank", 0),
                delta_cap=getattr(hparams, "cf_delta_cap", -1.0),
                replay_cache=getattr(hparams, "cf_replay_cache", False),
                device=getattr(hparams, "device", None),
                verbose=getattr(hparams, "cf_verbose", False),
            )
            self.mend_executor = None
        else:
            self.hypernetwork = HypernetworkEditGenerator(
                self.model, self.tok, hparams.inner_params,
                rank=hparams.hypernet_rank,
                hidden=hparams.hypernet_hidden,
                scale=hparams.hypernet_scale,
                seed=hparams.seed,
                max_length=getattr(hparams, "max_length", 64),
                ckpt=getattr(hparams, "hypernet_ckpt", None),
                feature_mode=getattr(hparams, "feature_mode", "mean-emb"),
            )
            self.mend_executor = None

        # ---- W4 enhancement: entity-level first-write-then-route -------- #
        # Set of subjects already WRITTEN into the model parameters during
        # this editor's lifetime. Once a subject has been written, further
        # edits on the same entity are routed to side memory instead of
        # re-polluting the weights (conflict-aware lifelong editing).
        self._written_subjects: set = set()
        # ---- W4 fix: bounded parameter writes --------------------------- #
        # Count of parameter WRITEs performed; capped by hparams.write_budget.
        self._write_count: int = 0
        # ---- W6 Rev3: in-loop memory READ + dual-channel guard ----------- #
        # Before Rev3 the side memory was write-only: ROUTE stored the answer
        # and no generation step ever read it back, so "memory routes" were
        # counted as retained while every answer still came from the weights.
        # ``memory_read`` closes that loop; the guard decides when a retrieved
        # note is allowed to speak for the model.
        self.memory_read: bool = bool(getattr(hparams, "memory_read", False))
        self.memory_read_guard: bool = bool(
            getattr(hparams, "memory_read_guard", True)
        )
        self.memory_read_max_new: int = int(
            getattr(hparams, "memory_read_max_new", 5) or 5
        )
        self._build_memory_note = getattr(side_memory_mod, "build_memory_note", None)
        # Rev3: how the retrieved note is put in front of the prompt, and
        # whether a free-generation failure may be arbitrated by likelihood.
        self.memory_read_template = str(getattr(self.hparams, "memory_read_template", "instruct"))
        self.memory_read_arbitrate = bool(getattr(self.hparams, "memory_read_arbitrate", True))
        self.memory_read_arb_margin = float(getattr(self.hparams, "memory_read_arb_margin", 0.0))
        # Service mode of the memory channel.  ``inject`` is the Rev3 behaviour
        # (note goes in the prompt, the LM must copy it out); ``direct`` lets
        # the store answer the query itself; ``hybrid`` tries injection first
        # and falls back to a direct serve.  ``direct``/``hybrid`` answers are
        # attributed to their own channel (``memory_direct``) so a retrieval
        # serve can never be quoted as free-generation recall.
        self.memory_serve = str(getattr(self.hparams, "memory_serve", "inject") or "inject").lower()
        # Monotone write clocks: who touched (subject, relation) last, the
        # parameter channel or the memory channel?  Used by the staleness guard.
        self._mem_version: Dict[str, int] = {}
        self._param_version: Dict[str, int] = {}
        self._version_seq: int = 0
        self.read_stats: Dict[str, int] = {
            "memory_probe": 0,
            "memory_hit": 0,
            "memory_miss": 0,
            "memory_injected": 0,
            "memory_answered": 0,
            "memory_skipped_shadowed": 0,
            "memory_skipped_param_ok": 0,
            "memory_degenerate": 0,
            "memory_arbitrated": 0,
            "memory_served": 0,
            "param_answered": 0,
            "unanswered": 0,
        }
        # ---- Claude #9: verify-then-commit ------------------------------ #
        # When enabled, every WRITE is applied tentatively; if the post-edit
        # answer does not contain target_new, the parameter delta is rolled
        # back and the edit is stored in side memory instead (ROUTE-like).
        # This prevents failed low-rank deltas from polluting the weights
        # permanently (the dominant cause of cumulative drift on long
        # streams) at zero additional training cost.
        self.verify_commit: bool = bool(getattr(hparams, "verify_commit", False))

        # ---- W6: deferred (windowed) parameter writes -------------------- #
        # defer_window == 1 keeps the sequential CALM loop untouched.  >1
        # buffers WRITE decisions and solves them jointly, which is the only
        # way to escape the "one key set per solve" regime that caps in-gate
        # retention at 0.6% (each new solve re-uses the previous key
        # directions and wins by arrival order).
        self.defer_window: int = max(1, int(getattr(hparams, "defer_window", 1) or 1))
        self.defer_fallback_route: bool = bool(
            getattr(hparams, "defer_fallback_route", True)
        )
        # Key-disjoint sub-windowing: edits sharing (subject, relation) inside
        # one window fight over the same key directions, so they are solved in
        # successive sub-batches instead of a single compromised solve.
        self.defer_group_keys: bool = bool(
            getattr(hparams, "defer_group_keys", False)
        )
        # Contiguous sub-batching inside one window (0 = single joint solve).
        # A window solve spreads one delta budget over every constraint in the
        # window, so a large window underfits its own members (measured 11/50
        # on the held-out slice with otherwise identical settings).  Chunking
        # gives each sub-batch its own solve and its own guard.
        self.defer_chunk: int = max(0, int(getattr(hparams, "defer_chunk", 0) or 0))
        # Re-solve passes for window members that still miss target_new after
        # the joint solve, before giving up and falling back to side memory.
        self.defer_refine_passes: int = max(
            0, int(getattr(hparams, "defer_refine_passes", 0) or 0)
        )
        # Per-window fluency guard: a joint solve over a big window can hit
        # every target and still wreck the surrounding language model.  The
        # budget caps the probe-NLL increase a single window may cause; a
        # window that overshoots is retried at a smaller delta scale and,
        # failing that, rolled back and routed to side memory instead of
        # silently poisoning the weights for every later edit.
        self.defer_delta_budget: float = float(
            getattr(hparams, "defer_delta_budget", 0.0) or 0.0
        )
        self.defer_guard_scale: float = float(
            getattr(hparams, "defer_guard_scale", 0.5) or 0.5
        )
        self.last_guard: Dict[str, Any] = {}
        self._pending: List[Dict[str, Any]] = []      # buffered WRITE requests
        self.deferred_flushes: List[Dict[str, Any]] = []   # per-flush diagnostics
        # Optional per-edit callback: fn(edit_index, metrics_row, all_metrics).
        # Used by the W6 harness to take survival@k checkpoints mid-stream
        # without forking the routing loop.
        self.after_edit_hook = None

    # ------------------------------------------------------------------ #
    # Public API (EasyEdit-compatible)
    # ------------------------------------------------------------------ #

    def edit(
        self,
        prompts: Union[str, List[str]],
        target_new: Union[str, List[str]],
        ground_truth: Optional[Union[str, List[str]]] = None,
        subject: Optional[Union[str, List[str]]] = None,
        relation: Optional[Union[str, List[str]]] = None,
        rephrase_prompts: Optional[Union[str, List[str]]] = None,
        sequential_edit: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]:
        """Apply a (batch of) factual edit(s) through the CALM pipeline.

        Returns ``(all_metrics, edited_model, extra)`` with EasyEdit-style
        per-request metrics plus CALM routing information.
        """
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        target_new = [target_new] if isinstance(target_new, str) else list(target_new)
        n = len(prompts)
        ground_truth = self._as_list(ground_truth, n, "")
        subject = self._as_list(subject, n, None)
        relation = self._as_list(relation, n, "fact")
        rephrase = self._as_list(rephrase_prompts, n, None)

        if not (len(target_new) == len(ground_truth) == len(subject) == len(relation)):
            raise ValueError("[CALM] prompts / target_new / ground_truth / subject / relation "
                             "must have equal length")

        requests: List[Dict[str, Any]] = []
        for i in range(n):
            req = {
                "prompt": prompts[i],
                "target_new": target_new[i],
                "ground_truth": ground_truth[i],
                "subject": subject[i],
                "relation": relation[i],
            }
            if rephrase[i] is not None:
                req["rephrase_prompt"] = rephrase[i]
            requests.append(req)

        return self.edit_requests(requests, sequential_edit=sequential_edit,
                                  verbose=verbose)

    def edit_requests(
        self,
        requests: List[Dict[str, Any]],
        sequential_edit: bool = True,
        verbose: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]:
        """Apply a stream of edit requests one by one (sequential editing).

        Each request goes through: gate decision -> WRITE (hypernetwork) /
        ROUTE (side memory) / REJECT (skip).
        """
        all_metrics: List[Dict[str, Any]] = []
        for idx, request in enumerate(requests):
            if verbose:
                print(f"[CALM] edit {idx + 1}/{len(requests)} | "
                      f"subject={request.get('subject')!r} "
                      f"prompt={request['prompt']!r} -> {request['target_new']!r}")

            pre = self._measure(request)
            current_answer, current_confidence = self._current_answer_and_confidence(
                request["prompt"]
            )
            decision = self.gate.decide(
                subject=request.get("subject") or "",
                relation=request.get("relation") or "fact",
                target_new=request["target_new"],
                current_answer=current_answer,
                current_confidence=current_confidence,
                recent_edits=list(self.gate.recent_edits),
                retrieval_context=None,
            )
            # W4 fix: entity-level first-write-then-route.  A WRITE on a
            # subject that was already written into the parameters becomes a
            # ROUTE instead (side memory, no parameter change).  This prevents
            # repeated parameter pollution on the same entity, which is the
            # dominant cause of the W4 lifelong-retention failure.
            raw_decision = decision
            decision = self._first_write_guard(request, decision)
            first_write_route = (decision != raw_decision)
            if not first_write_route:
                decision = self._write_budget_guard(request, decision)
                first_write_route = (decision != raw_decision)

            if self._deferring(decision):
                # W6: buffer the WRITE; it is solved jointly with the rest of
                # the window in _flush_deferred().  Nothing is measured here
                # because the parameters have not moved yet -- reporting a
                # "post" answer now would be the pre-edit answer mislabelled
                # as an in-gate hit.
                self._pending.append({"index": len(all_metrics), "request": request})
                route, memory_updated = "deferred_write", False
                post = {"answer": None, "rewrite_acc": None, "rewrite_ok": None,
                        "pending": True}
            elif self.verify_commit and decision == self.Decision.WRITE:
                # Claude #9: tentative write -> measure -> commit/rollback.
                snapshot = self._snapshot_inner_params()
                route, memory_updated = self._apply_decision(request, decision)
                post = self._measure(request)
                if not post["rewrite_ok"]:
                    self._restore_inner_params(snapshot)
                    self._write_count -= 1
                    subject = request.get("subject") or ""
                    if subject in self._written_subjects:
                        self._written_subjects.discard(subject)
                    stored = self._mem_store(
                        subject,
                        request.get("relation") or "fact",
                        request["target_new"],
                    )
                    route = "verify_rollback"
                    memory_updated = bool(stored)
                    decision = self.Decision.ROUTE
            else:
                route, memory_updated = self._apply_decision(request, decision)
                post = self._measure(request)

            _last_log = self.gate._decision_log[-1] if self.gate._decision_log else {}
            gate_score = float(_last_log.get("conf_score", 0.0))
            metrics = {
                "pre": pre,
                "post": post,
                "calm": {
                    "route": route,
                    "decision": decision,
                    "gate_score": round(gate_score, 4),
                    "gate_p_write": _last_log.get("gate_p_write"),
                    "memory_updated": memory_updated,
                    "first_write_route": first_write_route,
                    "current_answer": current_answer,
                    "current_confidence": round(current_confidence, 4),
                },
            }
            all_metrics.append(metrics)
            if verbose:
                print(f"    -> route={route} decision={decision} "
                      f"pre={pre['answer']!r} post={post['answer']!r}")

            # W6: window boundary -> solve + apply the buffered writes, then
            # re-measure every member of the window (this is what makes the
            # "immediate hit" a real post-write number instead of a guess).
            if self._deferring() and len(self._pending) >= self.defer_window:
                self._flush_deferred(all_metrics)
                if verbose:
                    last = self.deferred_flushes[-1]
                    print(f"    -> flushed {last['flush_size']} writes: "
                          f"{sum(f['ok'] for f in last['finalized'])}/"
                          f"{last['flush_size']} immediate hits "
                          f"({last['seconds']}s)")

            hook = self.after_edit_hook
            if hook is not None:
                hook(idx, metrics, all_metrics)

        # Tail window (stream length not divisible by defer_window).
        if self._pending:
            self._flush_deferred(all_metrics)

        extra = {
            "gate_summary": self.gate.get_decision_summary(),
            "memory_stats": self.memory.stats(),
            "defer_window": self.defer_window,
            "deferred_flushes": self.deferred_flushes,
        }
        return all_metrics, self.model, extra

    # ------------------------------------------------------------------ #
    # W6: deferred (windowed) writes
    # ------------------------------------------------------------------ #

    def _deferring(self, decision: Optional[str] = None) -> bool:
        """True when WRITE decisions must be buffered instead of applied now.

        Requires a writer that can solve a whole window at once: the
        closed-form writer exposes ``generate_update_batch``; the low-rank
        hypernetwork and MEND do not, so they keep the sequential path
        (silently deferring there would just re-apply single solves later).
        """
        if self.defer_window <= 1 or self.hypernetwork is None:
            return False
        if not hasattr(self.hypernetwork, "generate_update_batch"):
            return False
        if decision is None:
            return True
        return decision == self.Decision.WRITE


    # ------------------------------------------------------------------ #
    # Fluency guard helpers
    # ------------------------------------------------------------------ #
    PROBE_TEXTS = [
        "The river flows slowly through the wide valley before it reaches the sea.",
        "She opened the window and listened to the rain falling on the roof.",
        "A committee of local residents met on Tuesday to discuss the new library.",
        "The company reported a modest increase in revenue for the third quarter.",
        "Many species of birds migrate south during the colder months of the year.",
        "He studied the map carefully before choosing a route through the mountains.",
        "The museum's collection includes paintings, sculptures, and ancient coins.",
        "After the long discussion, the council agreed to postpone the final vote.",
    ]

    def _probe_nll(self) -> float:
        """Mean token-level NLL of the fixed probe sentences."""
        total, count = 0.0, 0
        with torch.no_grad():
            for text in self.PROBE_TEXTS:
                ids = self.tok(text, return_tensors="pt", truncation=True,
                               max_length=getattr(self.hparams, "max_length", 64)
                               ).input_ids.to(self.device)
                if ids.size(1) < 2:
                    continue
                logits = self.model(ids).logits[:, :-1, :]
                tgt = ids[:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                    reduction="sum",
                )
                total += float(loss.item())
                count += int(tgt.numel())
        return total / max(1, count)

    def _snapshot_params(self, names):
        state = {}
        for name in names:
            param = nethook.get_parameter(self.model, name)
            state[name] = param.detach().clone()
        return state

    def _restore_params(self, state) -> None:
        with torch.no_grad():
            for name, tensor in state.items():
                param = nethook.get_parameter(self.model, name)
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

    def _apply_with_guard(self, batch, budget: float = None):
        """Apply a window solve, then keep it only if fluency stays in budget.

        ``budget`` overrides ``defer_delta_budget`` for one apply (used by the
        flush loop to enforce the budget per *window* rather than per
        sub-solve: a window that spends its allowance on the first solve and
        then re-solves its stragglers used to slip through twice).

        Returns (rolled_back, applied_scale).
        """
        deltas = self.hypernetwork.generate_update_batch(batch)
        budget = self.defer_delta_budget if budget is None else float(budget)
        if budget <= 0:
            self.hypernetwork.apply_update(deltas)
            return False, 1.0
        snapshot = self._snapshot_params(deltas.keys())
        pre_nll = self._probe_nll()
        for scale in (1.0, self.defer_guard_scale):
            scaled = deltas if scale == 1.0 else {k: v * scale for k, v in deltas.items()}
            self.hypernetwork.apply_update(scaled)
            post_nll = self._probe_nll()
            if post_nll - pre_nll <= budget:
                self.last_guard = {"window_pre_nll": round(pre_nll, 4),
                                   "window_post_nll": round(post_nll, 4),
                                   "applied_scale": scale, "rolled_back": False,
                                   "budget": round(budget, 4)}
                return False, scale
            self._restore_params(snapshot)
        self.last_guard = {"window_pre_nll": round(pre_nll, 4),
                           "applied_scale": 0.0, "rolled_back": True,
                           "budget": round(budget, 4)}
        return True, 0.0

    def _flush_deferred(self, all_metrics: List[Dict[str, Any]]) -> None:
        """Solve the buffered window jointly, apply it, then re-measure it.

        The solve is a single least-squares problem over every (keys, residual)
        row in the window (plus the replay cache of earlier windows when
        ``cf_replay_cache`` is on), so the edits in one window are simultaneous
        constraints instead of competing arrivals.  A joint solve cannot be
        rolled back per member, so members that still miss ``target_new`` are
        mirrored into side memory (``defer_fallback_route``) rather than being
        silently counted as parameter hits.
        """
        if not self._pending:
            return
        t0 = time.time()
        pending, self._pending = self._pending, []
        groups = self._split_by_key(pending)
        if self.defer_chunk > 0:
            # Contiguous sub-batching: keep arrival order, but solve (and
            # guard) every chunk on its own so one hard sub-batch cannot drag
            # the rest of the window down with it.
            groups = [g[i:i + self.defer_chunk]
                      for g in groups
                      for i in range(0, len(g), self.defer_chunk)]

        rec: Dict[str, Any] = {
            "flush_index": len(self.deferred_flushes),
            "flush_size": len(pending),
            "first_slot": pending[0]["index"],
            "last_slot": pending[-1]["index"],
            "groups": [len(g) for g in groups],
            "refine_passes": 0,
            "delta_stats": getattr(self.hypernetwork, "last_apply_stats", {}),
            "finalized": [],
        }
        rec["guard"] = []
        # Fluency allowance is spent per window, not per sub-solve: without
        # this counter the first solve and the straggler re-solve each get a
        # full budget and one window can drift twice as far as configured
        # (measured +0.82 total on a budget of 0.5).
        spent = 0.0
        for group in groups:
            batch = [p["request"] for p in group]
            slots = [p["index"] for p in group]
            budget_left = None
            if self.defer_delta_budget > 0:
                budget_left = max(0.0, self.defer_delta_budget - spent)
            rolled_back, applied_scale = self._apply_with_guard(batch, budget_left)
            rec["guard"].append(dict(self.last_guard))
            if not rolled_back:
                spent += max(0.0, self.last_guard["window_post_nll"]
                             - self.last_guard["window_pre_nll"])
            if rolled_back:
                # The window would have wrecked fluency: leave the slots
                # un-written so the fallback pass mirrors them into memory.
                for slot in slots:
                    calm = all_metrics[slot]["calm"]
                    calm["flush_ok"] = False
                    calm["guard_rollback"] = True
                continue
            self._write_count += len(batch)
            rec["applied_scale"] = applied_scale
            # Stamp the parameter clock for every written key: a note stored
            # earlier for the same query is now stale and must not be injected.
            for request in batch:
                self._param_write_done(request)
            for slot, request in zip(slots, batch):
                self._finalize_slot(rec, all_metrics, slot, request)

            # Second-pass re-solve: a joint solution can still leave members
            # short when their key directions are near-collinear.  Re-solving
            # only the stragglers (their residuals re-measured against the
            # current weights) is cheap and removes them from the fallback
            # bucket instead of spending side-memory on a solvable row.
            for _ in range(max(0, self.defer_refine_passes)):
                missing = [(slot, req) for slot, req in zip(slots, batch)
                           if not all_metrics[slot]["calm"].get("flush_ok")]
                if not missing:
                    break
                # The refine pass must go through the same fluency guard as
                # the first solve: an unguarded re-solve is exactly how a
                # window turns into +1.79 NLL / 25% degenerate answers while
                # still reporting a better hit rate.
                budget_left = None
                if self.defer_delta_budget > 0:
                    budget_left = max(0.0, self.defer_delta_budget - spent)
                rolled_back, _ = self._apply_with_guard(
                    [r for _, r in missing], budget_left)
                rec["guard"].append(dict(self.last_guard))
                if rolled_back:
                    break
                spent += max(0.0, self.last_guard["window_post_nll"]
                             - self.last_guard["window_pre_nll"])
                self._write_count += len(missing)
                rec["refine_passes"] += 1
                for slot, request in missing:
                    self._finalize_slot(rec, all_metrics, slot, request,
                                        refine=rec["refine_passes"])

        for item in pending:
            slot, request = item["index"], item["request"]
            calm = all_metrics[slot]["calm"]
            if not calm.get("flush_ok") and self.defer_fallback_route:
                stored = self._mem_store(
                    request.get("subject") or "",
                    request.get("relation") or "fact",
                    request["target_new"],
                )
                calm["route"] = "batched_route_fallback"
                calm["memory_updated"] = bool(stored)
        rec["immediate_hits"] = sum(1 for f in rec["finalized"] if f["ok"])
        rec["seconds"] = round(time.time() - t0, 2)
        self.deferred_flushes.append(rec)

    def _finalize_slot(self, rec: Dict[str, Any], all_metrics: List[Dict[str, Any]],
                       slot: int, request: Dict[str, Any],
                       refine: int = 0) -> None:
        """Re-measure one buffered edit after its window landed."""
        post = self._measure(request)
        ok = bool(post["rewrite_ok"])
        # Rev3: the fallback decision belongs to the parameter channel.  With
        # the read path on, ``rewrite_ok`` is end-to-end, so using it here
        # would tell the fallback pass "already answered" for rows that only
        # the memory channel answered - and those rows would then never be
        # mirrored into memory (they would vanish on the next stream).
        param_ok = bool(post.get("param_ok", ok))
        row = all_metrics[slot]
        row["post"] = post
        calm = row["calm"]
        calm["route"] = "hypernetwork_batched"
        calm["pending"] = False
        calm["flush_index"] = rec["flush_index"]
        calm["flush_ok"] = param_ok
        calm["e2e_ok"] = ok
        calm["answered_by"] = post.get("channel", "parameter" if param_ok else "none")
        calm["refine"] = refine
        rec["finalized"].append({"slot": slot, "ok": ok,
                                 "answer": post["answer"], "refine": refine})

    def _split_by_key(self, pending: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split a window into key-disjoint sub-windows.

        Two edits that share ``(subject, relation)`` want the same key
        directions to move in different directions; a single least-squares
        solve then satisfies neither, and both fall back to side memory.
        Grouping the window by key (first occurrence wins, order preserved)
        turns those conflicts into successive solves, which is still batched
        but no longer self-defeating.
        """
        if not self.defer_group_keys:
            return [pending]
        seen: Dict[tuple, int] = {}
        groups: List[List[Dict[str, Any]]] = []
        for item in pending:
            req = item["request"]
            key = ((req.get("subject") or "").lower(),
                   (req.get("relation") or "fact"))
            if key in seen:
                groups[seen[key]].append(item)
            else:
                seen[key] = len(groups)
                groups.append([item])
        return groups


    def _first_write_guard(self, request: Dict[str, Any], decision: str) -> str:
        """Entity-level first-write-then-route guard (W4 enhancement).

        If ``decision`` is WRITE and the request's subject was already written
        into the parameters earlier in this stream, downgrade to ROUTE so the
        edit is stored in side memory instead of re-polluting the weights.
        Empty subjects keep their original decision.
        """
        if decision != self.Decision.WRITE:
            return decision
        subject = request.get("subject") or ""
        if not subject:
            return decision
        if subject in self._written_subjects:
            return self.Decision.ROUTE
        self._written_subjects.add(subject)
        return decision

    def _write_budget_guard(self, request: Dict[str, Any], decision: str) -> str:
        """Bounded-parameter-writes guard (W4 fix).

        Once ``hparams.write_budget`` parameter WRITEs have been performed in
        this stream, every further WRITE is downgraded to ROUTE (side memory).
        This stops unbounded parameter pollution on long streams composed of
        almost-all-distinct entities (the W4 failure mode: gate conflict
        detection degenerates, ~100% WRITE, retention collapses to 0).
        """
        if decision != self.Decision.WRITE:
            return decision
        budget = int(getattr(self.hparams, "write_budget", -1) or -1)
        if budget >= 0 and self._write_count >= budget:
            return self.Decision.ROUTE
        self._write_count += 1
        return decision

    def _snapshot_inner_params(self) -> Dict[str, torch.Tensor]:
        """Clone all inner parameters for tentative-write rollback."""
        snapshot: Dict[str, torch.Tensor] = {}
        for name in self.hparams.inner_params:
            param = nethook.get_parameter(self.model, name)
            snapshot[name] = param.data.clone()
        return snapshot

    def _restore_inner_params(self, snapshot: Dict[str, torch.Tensor]) -> None:
        """Restore inner parameters from a snapshot (verify-commit rollback)."""
        with torch.no_grad():
            for name, data in snapshot.items():
                param = nethook.get_parameter(self.model, name)
                param.data.copy_(data)

    def _apply_decision(self, request: Dict[str, Any], decision: str):
        """Execute a gate decision; returns (route, memory_updated)."""
        if decision == self.Decision.REJECT:
            return "gate_reject", False

        if decision == self.Decision.ROUTE:
            stored = self._mem_store(
                request.get("subject") or "",
                request.get("relation") or "fact",
                request["target_new"],
            )
            return "side_memory", bool(stored)

        # WRITE
        if self.hypernetwork is not None:
            deltas = self.hypernetwork.generate_update(request)
            self.hypernetwork.apply_update(deltas)
        elif self.mend_executor is not None:
            # This EasyEdit MEND revision expects plain-string targets
            # (``request["target_new"][0]``), not the {str, id} dict form.
            mend_request = [{
                "prompt": request["prompt"],
                "subject": request.get("subject") or "",
                "relation_id": request.get("relation") or "fact",
                "target_new": request["target_new"],
                "target_true": request.get("ground_truth", ""),
            }]
            self.mend_executor.apply_to_model(
                self.model, self.tok, mend_request, self.mend_params
            )
        else:
            raise RuntimeError("[CALM] no hypernetwork backend configured")
        # Rev3: the weights now hold this key, so any older note for the same
        # query must stop being injected (dual-channel staleness guard).
        self._param_write_done(request)
        # W4 fix: dual-write.  A parameter WRITE is also recorded in side
        # memory, so CALM's end-to-end retention is backed by memory even for
        # parameter edits (memory is part of the CALM architecture).
        if getattr(self.hparams, "dual_write", False):
            stored = self._mem_store(
                request.get("subject") or "",
                request.get("relation") or "fact",
                request["target_new"],
            )
            # Report it honestly: the W4/W6 per-channel tables split parameter
            # WRITEs into "parameter only" vs "parameter + memory mirror", so a
            # swallowed False here would silently move every dual-written edit
            # into the wrong bucket.
            return "hypernetwork", bool(stored)
        return "hypernetwork", False

    # ------------------------------------------------------------------ #
    # Lightweight measurement helpers
    # ------------------------------------------------------------------ #

    def _current_answer_and_confidence(self, prompt: str, max_new_tokens: int = 5):
        """Greedy-decode the model's current answer and its mean probability."""
        ids = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=getattr(self.hparams, "max_length", 64)
                       ).input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
            full_logits = self.model(out).logits
        gen = out[0, ids.size(1):]
        if gen.numel() == 0:
            return "", 0.0
        start = ids.size(1) - 1
        logits = full_logits[0, start:start + gen.numel()]
        probs = logits.softmax(dim=-1)
        chosen = probs.gather(1, gen.unsqueeze(-1)).squeeze(-1)
        text = self.tok.decode(gen, skip_special_tokens=True)
        return text, float(chosen.mean().item())

    # ------------------------------------------------------------------ #
    # W6 Rev3: memory READ path (the half that never existed before)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mem_key(subject: Any, relation: Any) -> str:
        """Version-clock key. Identity is the query only, never the answer."""
        s = " ".join(str(subject or "").lower().split())
        r = " ".join(str(relation or "fact").lower().split())
        return f"{s}|{r}"

    def _mem_store(self, subject: str, relation: str, answer: str) -> bool:
        """Store and stamp the memory channel's clock for this key."""
        stored = self.memory.store(subject, relation, answer)
        if stored:
            self._version_seq += 1
            self._mem_version[self._mem_key(subject, relation)] = self._version_seq
        return stored

    def _param_write_done(self, request: Dict[str, Any]) -> None:
        """Stamp the parameter channel's clock after a successful weight write."""
        self._version_seq += 1
        self._param_version[
            self._mem_key(request.get("subject"), request.get("relation"))
        ] = self._version_seq

    def memory_probe(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ``(subject, relation)`` and decide whether the note may speak.

        Reasons returned in ``info['reason']``:
          ``disabled``              read path off (pre-Rev3 behaviour)
          ``memory_miss``           query resolved to nothing
          ``shadowed_by_param``     parameters were written after this note
          ``ok``                    note may be injected
        """
        info: Dict[str, Any] = {"note": None, "used": False, "reason": "disabled",
                                "key": None, "shadowed": False}
        if not self.memory_read:
            return info
        subject = request.get("subject") or ""
        relation = request.get("relation") or "fact"
        key = self._mem_key(subject, relation)
        info["key"] = key
        self.read_stats["memory_probe"] += 1
        note = self.memory.lookup(subject, relation)
        if note is None:
            info["reason"] = "memory_miss"
            self.read_stats["memory_miss"] += 1
            return info
        self.read_stats["memory_hit"] += 1
        info["note"] = note
        if self.memory_read_guard:
            # Staleness guard: a parameter write that landed after this note
            # supersedes it.  Injecting the older note would resurrect a fact
            # the weights have already been corrected away from.
            if self._param_version.get(key, 0) > self._mem_version.get(key, 0):
                info["reason"] = "shadowed_by_param"
                info["shadowed"] = True
                self.read_stats["memory_skipped_shadowed"] += 1
                return info
        info["used"] = True
        info["reason"] = "ok"
        return info

    @staticmethod
    def _degenerate(text: str) -> bool:
        """True for empty / looping output that no longer carries a fact."""
        toks = (text or "").split()
        if not toks:
            return True
        if len(toks) >= 6 and len(set(toks)) <= max(1, len(toks) // 4):
            return True
        return False

    def _memory_prompt(self, subject: str, relation: str, note: str, prompt: str) -> str:
        """Render the retrieved note in front of the prompt.

        ``prefix`` is the original ``[s] [r] is [v]`` form used by the recall
        probe; ``instruct`` adds an explicit "Answer with v." instruction,
        which survives a drifted model better; ``qa`` uses a Q/A frame.
        """
        tmpl = (self.memory_read_template or "instruct").lower()
        if tmpl == "prefix":
            if self._build_memory_note is not None:
                return self._build_memory_note(subject, relation, note) + prompt
            return f"[{subject}] [{relation}] is [{note}]\n" + prompt
        if tmpl == "qa":
            return (f"Q: what is the {relation} of {subject}?\nA: {note}\n"
                    + prompt)
        return f"Answer with {note}.\n" + prompt

    def _mean_logp(self, prompt: str, text: str) -> float:
        """Mean per-token log-probability of ``text`` continued after ``prompt``."""
        text = (text or "").strip()
        if not text:
            return -1e9
        cap = 2 * int(getattr(self.hparams, "max_length", 64) or 64)
        pid = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=cap).input_ids.to(self.device)
        tid = self.tok(text, return_tensors="pt", add_special_tokens=False
                       ).input_ids.to(self.device)
        ids = torch.cat([pid, tid], dim=1)
        with torch.no_grad():
            logits = self.model(ids).logits
        start = pid.size(1) - 1
        lp = logits[0, start:start + tid.size(1)].log_softmax(dim=-1)
        picked = lp.gather(1, tid[0].unsqueeze(-1)).squeeze(-1)
        return float(picked.mean().item())

    def answer_with_memory(self, prompt: str, request: Optional[Dict[str, Any]] = None,
                           max_new_tokens: Optional[int] = None):
        """Answer ``prompt`` with the memory channel in the loop.

        Dual-channel order (guard semantics): the parameter channel answers
        first; memory is consulted only for the edits the weights fail on.  A
        live correct parameter answer is never overwritten by a note, and a
        degenerate note-answer never replaces a clean parameter answer.

        Returns ``(answer, info)`` where ``info['channel']`` is one of
        ``parameter`` / ``memory`` / ``none`` — the attribution the pre-Rev3
        tables were missing.
        """
        req: Dict[str, Any] = dict(request or {})
        max_new = int(max_new_tokens or self.memory_read_max_new)

        base_ans, _ = self._current_answer_and_confidence(prompt, max_new_tokens=max_new)
        target = str(req.get("target_new") or "").strip().lower()
        base_ok = bool(target) and target in base_ans.lower()

        info: Dict[str, Any] = {
            "channel": "parameter" if base_ok else "none",
            "base_answer": base_ans,
            "base_ok": base_ok,
            "injected": False,
            "note": None,
            "reason": "disabled" if not self.memory_read else "no_injection",
            "final_answer": base_ans,
            "final_ok": base_ok,
        }

        if base_ok:
            self.read_stats["param_answered"] += 1
            if self.memory_read:
                probe = self.memory_probe(req)  # probe regardless, keeps n honest
                info["note"], info["reason"] = probe["note"], probe["reason"]
                self.read_stats["memory_skipped_param_ok"] += int(probe["used"])
            return base_ans, info

        if not self.memory_read:
            self.read_stats["unanswered"] += 1
            return base_ans, info

        probe = self.memory_probe(req)
        info["note"], info["reason"] = probe["note"], probe["reason"]
        if not probe["used"]:
            self.read_stats["unanswered"] += 1
            return base_ans, info

        subject = req.get("subject") or ""
        relation = req.get("relation") or "fact"
        note = str(probe["note"])

        # Direct serve: a non-parametric memory is allowed to answer from the
        # store.  Requiring the LM to re-emit the note it was just shown
        # measures the model's copy fidelity, not the memory channel — the
        # measured copy rate on the revised backbone is ~0.14-0.38, which is
        # why the routed rows looked dead (8 rows, 1 hit on slice o700).
        if self.memory_serve == "direct":
            info.update({"injected": False, "served": True,
                         "memory_answer": note, "memory_ok": True,
                         "channel": "memory_direct", "final_answer": note,
                         "final_ok": True})
            self.read_stats["memory_served"] += 1
            self.read_stats["memory_answered"] += 1
            return note, info

        mem_ctx = self._memory_prompt(subject, relation, note, prompt)
        mem_ans, _ = self._current_answer_and_confidence(
            mem_ctx, max_new_tokens=max_new
        )
        mem_ok = bool(target) and target in mem_ans.lower()
        degenerate = self._degenerate(mem_ans)
        info.update({"injected": True, "memory_answer": mem_ans,
                     "memory_ok": mem_ok, "memory_degenerate": degenerate})
        if mem_ok and not degenerate:
            info.update({"channel": "memory", "final_answer": mem_ans,
                         "final_ok": True})
            self.read_stats["memory_injected"] += 1
            self.read_stats["memory_answered"] += 1
            return mem_ans, info

        # Free generation ignored the note.  A heavily edited model follows a
        # bare "[s] [r] is [v]" prefix far worse than a clean one (measured:
        # 0.00 vs 0.72 recall), so arbitrate by likelihood: the note wins the
        # slot only when log P(note | note-context) beats log P(param answer |
        # prompt) by the configured margin.  Counted as its own channel so a
        # copy-style read can never be reported as free-generation recall.
        if self.memory_read_arbitrate and not degenerate:
            lp_note = self._mean_logp(mem_ctx, note)
            lp_base = self._mean_logp(prompt, (base_ans or "").strip())
            margin = lp_note - lp_base
            info["arb_margin"] = round(margin, 4)
            if margin > self.memory_read_arb_margin:
                info.update({"channel": "memory_arb", "final_answer": note,
                             "final_ok": True})
                self.read_stats["memory_injected"] += 1
                self.read_stats["memory_arbitrated"] += 1
                self.read_stats["memory_answered"] += 1
                return note, info

        # Hybrid: injection failed, so serve the note from the store instead of
        # returning an answer the channels both failed.  Kept as its own branch
        # (and its own channel label) so the mixed-mode row can be separated
        # from pure copy behaviour in the tables.
        if self.memory_serve == "hybrid":
            info.update({"served": True, "memory_answer": note,
                         "channel": "memory_direct", "final_answer": note,
                         "final_ok": True})
            self.read_stats["memory_injected"] += 1
            self.read_stats["memory_served"] += 1
            self.read_stats["memory_answered"] += 1
            return note, info

        # Note injected but did not deliver: keep the parameter answer, and
        # count the failure so the read path cannot hide behind hit_total.
        self.read_stats["memory_injected"] += 1
        self.read_stats["memory_degenerate"] += int(degenerate)
        info.update({"channel": "none", "final_answer": base_ans, "final_ok": False})
        self.read_stats["unanswered"] += 1
        return base_ans, info

    def _measure(self, request: Dict[str, Any]) -> Dict[str, Any]:
        target = str(request.get("target_new") or "")
        if not self.memory_read:
            # Pre-Rev3 behaviour, byte-for-byte: parameter channel only.
            answer, _ = self._current_answer_and_confidence(request["prompt"])
            ok = target.strip().lower() in answer.lower()
            return {"answer": answer, "rewrite_acc": float(ok), "rewrite_ok": bool(ok)}

        # Rev3: answer end-to-end (both channels), but keep the parameter-only
        # verdict in ``param_ok`` so in-gate tables stay comparable and the
        # memory-provided rows can be counted separately.
        answer, info = self.answer_with_memory(request["prompt"], request)
        return {
            "answer": answer,
            "rewrite_acc": float(info["final_ok"]),
            "rewrite_ok": bool(info["final_ok"]),
            "param_ok": bool(info["base_ok"]),
            "param_answer": info["base_answer"],
            "channel": info["channel"],
            "mem_note": info.get("note"),
            "mem_reason": info.get("reason"),
            "mem_injected": bool(info.get("injected")),
            "mem_arb_margin": info.get("arb_margin"),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_list(value, n: int, default):
        if value is None:
            return [default] * n
        return [value] if isinstance(value, str) else list(value)
