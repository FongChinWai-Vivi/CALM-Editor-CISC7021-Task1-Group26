"""_arm_common.py -- shared plumbing for the W4 gate ablation arms.

Owner: TANG I CHOI (MC631517).

Every arm must expose the three names that ``easyeditor.editors.calm_editor``
imports from the module referenced by ``hparams.conflict_gate_path``:

    Decision, EditRecord, ConflictGate

``Decision`` / ``EditRecord`` are taken from the teammate gate module
(``FONG HIO LAM/side_memory_router/conflict_gate.py``) so that an arm is a
drop-in replacement and no teammate file has to be modified.  Only ``decide``
differs between arms, which is exactly what an ablation needs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FONG_GATE_PATH = os.path.join(REPO, "FONG HIO LAM", "side_memory_router",
                              "conflict_gate.py")
W3_DIR = os.path.join(REPO, "TANG I CHOI",
                      "w3_conflict_gate_training_and_classifier")
GATE_MODEL = os.path.join(W3_DIR, "gate_model", "gate_classifier.joblib")


def load_module(name: str, path: str):
    """Import a module from an absolute path under a private name."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REF = load_module("tang_fong_gate_reference", FONG_GATE_PATH)
Decision = _REF.Decision
EditRecord = _REF.EditRecord


class BaseArm:
    """Common bookkeeping: recent-edit window, decision log, summary."""

    name = "base"

    def __init__(self, n_recent: int = 50, write_threshold: float = 0.3,
                 route_threshold: float = 0.6, similarity_threshold: float = 0.7,
                 seed: int = 0):
        self.n_recent = n_recent
        self.write_threshold = write_threshold
        self.route_threshold = route_threshold
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        self.recent_edits: List[Any] = []
        self._decision_log: List[Dict[str, Any]] = []

    # -- to be overridden ------------------------------------------------ #
    def _decide(self, subject: str, relation: str, target_new: str,
                current_answer: str, current_confidence: float,
                recent_edits: List[Any],
                retrieval_context: Optional[List[str]]) -> str:
        raise NotImplementedError

    # -- public API used by CALMEditor ---------------------------------- #
    def decide(self, subject: str, relation: str, target_new: str,
               current_answer: str, current_confidence: float,
               recent_edits: Optional[List[Any]] = None,
               retrieval_context: Optional[List[str]] = None) -> str:
        decision = self._decide(subject, relation, target_new, current_answer,
                                current_confidence,
                                list(recent_edits or self.recent_edits),
                                retrieval_context)
        conf_score = self._confidence_score(current_answer, target_new)
        self._decision_log.append({
            "subject": subject,
            "relation": relation,
            "target_new": target_new,
            "current_answer": current_answer,
            "current_confidence": round(float(current_confidence), 3),
            "conf_score": round(conf_score, 3),
            # T4: probability the classifier compared against, when the arm has
            # one.  Logged per edit so a threshold can be swept offline.
            "gate_p_write": getattr(self, "_last_p_write", None),
            "decision": decision,
            "arm": self.name,
        })
        self.recent_edits.append(EditRecord(subject, relation, target_new,
                                           current_answer, current_confidence))
        if len(self.recent_edits) > self.n_recent:
            self.recent_edits.pop(0)
        return decision

    def get_decision_summary(self) -> Dict[str, int]:
        summary = {"WRITE": 0, "ROUTE": 0, "REJECT": 0}
        for entry in self._decision_log:
            d = entry.get("decision")
            if d in summary:
                summary[d] += 1
        return summary

    def clear_history(self) -> None:
        self._decision_log.clear()
        self.recent_edits.clear()

    @staticmethod
    def _confidence_score(current_answer: str, target_new: str) -> float:
        if current_answer.strip().lower() == target_new.strip().lower():
            return 1.0
        target_words = set(target_new.lower().split())
        if not target_words:
            return 0.0
        overlap = len(set(current_answer.lower().split()) & target_words)
        return min(overlap / len(target_words) * 2.0, 1.0)
