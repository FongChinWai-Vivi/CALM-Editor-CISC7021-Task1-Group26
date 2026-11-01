"""rule_gate.py -- W4 ablation arm: the hand-written W1 rule gate.

Owner: TANG I CHOI (MC631517).

This arm is the annotation manual's decision rules expressed as code, with the
same priority order as the manual (prior-edit conflict > context conflict >
pre-training conflict):

    prior edit on the same (subject, relation) with a different target -> ROUTE
    retrieved context that does not contain the new target                -> ROUTE
    the model already answers the new target                              -> REJECT
    the model is confident in a different answer                          -> REJECT
    otherwise                                                             -> WRITE

It shares the feature extractor with the learned W3 gate
(``gate_features.py``), so the comparison isolates "rules vs learned" and not
"different features".

Selected by ``hparams.conflict_gate_path`` -- no teammate file is modified.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _arm_common import (W3_DIR, BaseArm, Decision, EditRecord,  # noqa: F401,E402
                         load_module)

_FEAT = load_module("tang_w3_gate_features",
                    os.path.join(W3_DIR, "gate_features.py"))
FEATURE_NAMES = _FEAT.FEATURE_NAMES
extract_features = _FEAT.extract_features


def rule_decide(sample: Dict[str, Any]) -> str:
    """The W1 manual's rules, made executable."""
    F = {n: i for i, n in enumerate(FEATURE_NAMES)}
    v = extract_features(sample)
    if v[F["prior_same_sr"]] > 0 and v[F["prior_target_match"]] == 0:
        return Decision.ROUTE
    if v[F["has_context"]] > 0 and v[F["context_target_overlap"]] < 0.5:
        return Decision.ROUTE
    if v[F["answer_exact"]] > 0:
        return Decision.REJECT
    if v[F["confidence"]] >= 0.5:
        return Decision.REJECT
    return Decision.WRITE


class ConflictGate(BaseArm):
    """Rule-based gate (W1 heuristic)."""

    name = "rule_gate"

    def _decide(self, subject: str, relation: str, target_new: str,
                current_answer: str, current_confidence: float,
                recent_edits: List[Any],
                retrieval_context: Optional[List[str]]) -> str:
        sample = {
            "subject": subject,
            "relation": relation,
            "target_new": target_new,
            "current_answer": current_answer,
            "current_confidence": current_confidence,
            "prior_edits": [
                {"subject": e.subject, "relation": e.relation,
                 "target": e.target_new}
                for e in recent_edits
            ],
            "retrieval_context": retrieval_context or [],
        }
        return rule_decide(sample)
