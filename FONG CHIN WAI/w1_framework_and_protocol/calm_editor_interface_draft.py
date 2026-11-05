"""
calm_editor.py -- CALM BaseEditor interface draft (W1)

CALM (Conflict-Aware Lightweight Model editor) exposes a unified editing
entry point that combines three mechanisms:

  1. Hypernetwork editing (MEND / MALMEN style): a learned hypernetwork or
     low-rank generator produces weight deltas for the edited MLP layers,
     trained on counterfactual / zsRE style meta-edits.
  2. Side memory: per-edit key-value storage (e.g. MEND-style caching or a
     MALMEN-like associative memory) that records edit signatures and can be
     queried for fast retrieval without touching base weights.
  3. Conflict gating: a classifier (the "conflict gate") decides whether an
     incoming request edits a fresh fact, conflicts with an existing memory,
     or should be routed to the side memory only.

The public API intentionally mirrors EasyEdit's BaseEditor
(``BaseEditor.from_hparams`` / ``edit`` / ``batch_edit``) so that CALM can be
dropped into existing EasyEdit pipelines as a subclass or drop-in companion.

This file is a design draft for the W1 interface. Method bodies are stubs
that document intended behavior and signatures; concrete implementations are
scheduled for later work packages (W2+).
"""

from __future__ import annotations

import abc
import dataclasses
import typing as T

if T.TYPE_CHECKING:
    from easyeditor.editors import BaseEditor as EasyBaseEditor


# --------------------------------------------------------------------------- #
# Shared data containers
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class EditRequest:
    """A single factual edit request, aligned with EasyEdit's edit() args."""

    prompt: str
    ground_truth: str
    target_new: str
    subject: str
    # Optional semantic tags for the conflict gate / side memory indexing.
    tags: T.Dict[str, T.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class EditResult:
    """Returned per edit: metrics (EasyEdit-compatible) plus CALM extras."""

    metrics: T.Dict[str, T.Any]
    # Post-edit model handle (same semantics as EasyEdit edit()'s 2nd return).
    edited_model: T.Any
    # Which CALM path was taken: 'hypernetwork' | 'side_memory' | 'gate_reject'.
    route: str
    # Conflict-gate raw score (higher => more confident of a conflict).
    gate_score: float
    # Whether a side-memory slot was created / updated for this edit.
    memory_updated: bool = False


@dataclasses.dataclass
class CALMHyperParams:
    """Hyper-parameters container.

    Mirrors EasyEdit's ``HyperParams`` role: one object holding everything
    needed to build the editor. ``from_hparams`` reads YAML/JSON or a
    dataclass dict, same convention as EasyEdit.
    """

    model_name: str
    device: str = "cuda:0"
    # --- hypernetwork branch ---
    hypernet_alg: T.Literal["MEND", "MALMEN", "SERAC"] = "MEND"
    hypernet_lr: float = 1e-4
    hypernet_rank: int = 512
    edited_layers: T.Optional[T.List[int]] = None  # e.g. [9, 10, 11]
    # --- side memory branch ---
    memory_capacity: int = 10_000
    memory_retrieval_topk: int = 5
    # --- conflict gate branch ---
    gate_ckpt: T.Optional[str] = None  # path to a trained gate checkpoint
    gate_threshold: float = 0.5

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str) -> "CALMHyperParams":
        """Load hyper-parameters from a YAML/JSON file or dotted name.

        Follows EasyEdit's convention: if ``.yaml`` / ``.json`` is not in the
        path, it is treated as a registered hparams directory and the default
        ``calm.yaml`` is looked up inside it.
        """
        raise NotImplementedError("W1 draft -- to be implemented in W2.")

    def to_dict(self) -> T.Dict[str, T.Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Sub-components
# --------------------------------------------------------------------------- #

class ConflictGate(abc.ABC):
    """Binary classifier over (prompt, target_new, side-memory context).

    Decision space:
      - NEW:   no conflict with existing memory -> full hypernetwork edit.
      - UPDATE: conflicts with a stored fact -> overwrite / route to memory.
      - REJECT: unsafe or contradictory request -> gate blocks the edit.
    """

    @abc.abstractmethod
    def score(self, request: EditRequest, memory_snapshot: T.Any) -> float:
        """Return a conflict score in [0, 1]; higher means more conflict."""

    @abc.abstractmethod
    def decide(self, request: EditRequest, memory_snapshot: T.Any) -> str:
        """Return one of 'NEW' | 'UPDATE' | 'REJECT'."""


class SideMemory(abc.ABC):
    """External key-value memory for edited facts (MALMEN-like associative
    store or a simple hash-indexed edit cache). Keeps base weights untouched."""

    @abc.abstractmethod
    def write(self, key: str, payload: T.Any) -> None:
        """Store a single edit record keyed by a canonical fact signature."""

    @abc.abstractmethod
    def lookup(self, query: str, topk: int = 5) -> T.List[T.Any]:
        """Retrieve the most relevant stored edit records for a query."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Reset the memory (used between experiments)."""


class HypernetworkEditor(abc.ABC):
    """Wraps MEND / MALMEN / SERAC-style editors behind a common interface."""

    @abc.abstractmethod
    def edit_one(self, request: EditRequest) -> EditResult:
        """Apply a single edit through the hypernetwork and return metrics."""


# --------------------------------------------------------------------------- #
# CALM unified editor
# --------------------------------------------------------------------------- #

class CALMEditor:
    """Unified editing entry point for CALM.

    Usage (EasyEdit-compatible style)::

        hparams = CALMHyperParams.from_hparams("hparams/CALM/gpt2-small.yaml")
        editor = CALMEditor.from_hparams(hparams)
        metrics, edited_model = editor.edit(
            prompts=[...], ground_truth=[...], target_new=[...], subject=[...],
            keep_original_weight=True,
        )

    The editor internally routes every request through:

        1. conflict gate  -> decides NEW / UPDATE / REJECT;
        2. hypernetwork   -> weight-delta generator (if NEW or UPDATE);
        3. side memory    -> persist / retrieve edit records;
        4. EasyEdit-style metrics evaluation on the edited model.
    """

    def __init__(
        self,
        hparams: CALMHyperParams,
        hypernetwork: T.Optional[HypernetworkEditor] = None,
        memory: T.Optional[SideMemory] = None,
        gate: T.Optional[ConflictGate] = None,
    ):
        """
        :param hparams:      resolved CALM hyper-parameters.
        :param hypernetwork: MEND/MALMEN-style editor; built from ``hparams``
                             if not provided.
        :param memory:       side-memory backend; defaults to an in-memory
                             hash store if not provided.
        :param gate:         conflict gate; if ``None``, all requests are
                             treated as NEW (gate disabled for warm-up).
        """
        self.hparams = hparams
        self.hypernetwork = hypernetwork
        self.memory = memory
        self.gate = gate
        # Delegation hook: when CALM wraps an EasyEdit editor, store it here.
        self._easyedit_backend: T.Optional["EasyBaseEditor"] = None

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_hparams(
        cls,
        hparams_name_or_path: str,
        use_easyedit_backend: bool = True,
    ) -> "CALMEditor":
        """Build a CALMEditor from hyper-parameters.

        :param hparams_name_or_path: path to a YAML/JSON hparams file, or a
            registered hparams name (e.g. ``gpt2-small``).
        :param use_easyedit_backend: if True, construct an underlying
            EasyEdit ``BaseEditor`` (via ``BaseEditor.from_hparams``) for
            metric computation and model management. CALM-specific components
            (gate / memory) are layered on top.
        :return: configured ``CALMEditor``.
        """
        hparams = CALMHyperParams.from_hparams(hparams_name_or_path)
        # W1 draft: wire sub-components from hparams here in W2.
        raise NotImplementedError("W1 draft -- to be implemented in W2.")

    # -- core editing ------------------------------------------------------ #

    def edit(
        self,
        prompts: T.List[str],
        ground_truth: T.List[str],
        target_new: T.List[str],
        subject: T.List[str],
        keep_original_weight: bool = False,
        **kwargs: T.Any,
    ) -> T.Tuple[T.Dict[str, T.Any], T.Any]:
        """Apply a batch of factual edits and return (metrics, edited_model).

        Signature-compatible with EasyEdit ``BaseEditor.edit``.

        :param prompts:               edit prompts (one per case).
        :param ground_truth:          original answers (pre-edit).
        :param target_new:            desired new answers (post-edit).
        :param subject:               subject strings for each case.
        :param keep_original_weight:  if True, the underlying model weights
                                      are kept untouched (delta applied at
                                      inference) -- default follows EasyEdit.
        :param kwargs:                forwarded to the backend editor.
        :return: ``(metrics, edited_model)`` where ``metrics`` is an
            EasyEdit-compatible dict (rewrite_acc / rephrase_acc / locality /
            portability) possibly extended with CALM gate statistics.
        """
        requests = self._build_requests(prompts, ground_truth, target_new, subject)
        results = [self.edit_single(r, keep_original_weight=keep_original_weight, **kwargs)
                   for r in requests]
        # Aggregate metrics into one dict (list of per-case dicts).
        raise NotImplementedError("W1 draft -- to be implemented in W2.")

    def batch_edit(
        self,
        requests: T.Sequence[EditRequest],
        **kwargs: T.Any,
    ) -> T.List[EditResult]:
        """Apply a sequence of ``EditRequest`` objects, one by one.

        Intended for online / streaming settings where each request updates
        the side memory before the next one is processed.

        :param requests: ordered edit requests.
        :return: per-request ``EditResult`` list.
        """
        results: T.List[EditResult] = []
        for req in requests:
            results.append(self.edit_single(req, **kwargs))
        return results

    def edit_single(
        self,
        request: EditRequest,
        keep_original_weight: bool = False,
        **kwargs: T.Any,
    ) -> EditResult:
        """Route one edit through gate -> hypernetwork -> side memory.

        :param request: the single edit to apply.
        :return: ``EditResult`` containing EasyEdit-style metrics plus the
            CALM routing information (route, gate_score, memory_updated).
        """
        # 1) Conflict gate (if enabled)
        gate_score = 0.0
        route = "hypernetwork"
        if self.gate is not None:
            snapshot = self.memory.lookup(request.prompt, topk=self.hparams.memory_retrieval_topk) if self.memory else []
            gate_score = self.gate.score(request, snapshot)
            decision = self.gate.decide(request, snapshot)
            if decision == "REJECT":
                return EditResult(
                    metrics={"rewrite_acc": 0.0, "gate_rejected": True},
                    edited_model=None,
                    route="gate_reject",
                    gate_score=gate_score,
                    memory_updated=False,
                )
            route = "side_memory" if decision == "UPDATE" else "hypernetwork"

        # 2) Hypernetwork / side-memory edit
        result = self.hypernetwork.edit_one(request) if self.hypernetwork else None
        edited_model = result.edited_model if result is not None else self._easyedit_backend

        # 3) Persist to side memory (if a memory backend exists)
        memory_updated = False
        if self.memory is not None:
            key = self._canonical_key(request)
            self.memory.write(key, request)
            memory_updated = True

        metrics = result.metrics if result is not None else {}
        metrics.setdefault("gate_score", gate_score)
        return EditResult(
            metrics=metrics,
            edited_model=edited_model,
            route=route,
            gate_score=gate_score,
            memory_updated=memory_updated,
        )

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _build_requests(
        prompts: T.List[str],
        ground_truth: T.List[str],
        target_new: T.List[str],
        subject: T.List[str],
    ) -> T.List[EditRequest]:
        """Zip parallel lists into ``EditRequest`` objects (length-checked)."""
        n = len(prompts)
        assert len(ground_truth) == n and len(target_new) == n and len(subject) == n, \
            "prompts / ground_truth / target_new / subject must have equal length"
        return [
            EditRequest(prompt=p, ground_truth=g, target_new=t, subject=s)
            for p, g, t, s in zip(prompts, ground_truth, target_new, subject)
        ]

    @staticmethod
    def _canonical_key(request: EditRequest) -> str:
        """Canonical fact signature used as side-memory key."""
        return f"{request.subject.strip().lower()} || {request.prompt.strip().lower()}"
