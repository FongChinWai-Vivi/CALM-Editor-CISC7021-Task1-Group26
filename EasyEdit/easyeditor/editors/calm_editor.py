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
        # ---- Claude #9: verify-then-commit ------------------------------ #
        # When enabled, every WRITE is applied tentatively; if the post-edit
        # answer does not contain target_new, the parameter delta is rolled
        # back and the edit is stored in side memory instead (ROUTE-like).
        # This prevents failed low-rank deltas from polluting the weights
        # permanently (the dominant cause of cumulative drift on long
        # streams) at zero additional training cost.
        self.verify_commit: bool = bool(getattr(hparams, "verify_commit", False))

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

            if self.verify_commit and decision == self.Decision.WRITE:
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
                    stored = self.memory.store(
                        subject,
                        request.get("relation") or "fact",
                        request["target_new"],
                        request["target_new"],
                    )
                    route = "verify_rollback"
                    memory_updated = bool(stored)
                    decision = self.Decision.ROUTE
            else:
                route, memory_updated = self._apply_decision(request, decision)
                post = self._measure(request)

            gate_score = float(self.gate._decision_log[-1]["conf_score"]) \
                if self.gate._decision_log else 0.0
            metrics = {
                "pre": pre,
                "post": post,
                "calm": {
                    "route": route,
                    "decision": decision,
                    "gate_score": round(gate_score, 4),
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

        extra = {
            "gate_summary": self.gate.get_decision_summary(),
            "memory_stats": self.memory.stats(),
        }
        return all_metrics, self.model, extra

    # ------------------------------------------------------------------ #
    # CALM routing
    # ------------------------------------------------------------------ #

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
            stored = self.memory.store(
                request.get("subject") or "",
                request.get("relation") or "fact",
                request["target_new"],
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
        # W4 fix: dual-write.  A parameter WRITE is also recorded in side
        # memory, so CALM's end-to-end retention is backed by memory even for
        # parameter edits (memory is part of the CALM architecture).
        if getattr(self.hparams, "dual_write", False):
            self.memory.store(
                request.get("subject") or "",
                request.get("relation") or "fact",
                request["target_new"],
                request["target_new"],
            )
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

    def _measure(self, request: Dict[str, Any]) -> Dict[str, Any]:
        answer, _ = self._current_answer_and_confidence(request["prompt"])
        target = request["target_new"]
        ok = target.strip().lower() in answer.lower()
        return {"answer": answer, "rewrite_acc": float(ok), "rewrite_ok": bool(ok)}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_list(value, n: int, default):
        if value is None:
            return [default] * n
        return [value] if isinstance(value, str) else list(value)
