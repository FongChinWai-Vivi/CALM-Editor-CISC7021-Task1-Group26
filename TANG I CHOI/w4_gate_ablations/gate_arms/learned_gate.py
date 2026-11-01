"""learned_gate.py -- W4 ablation arm: the trained W3 logistic gate.

Owner: TANG I CHOI (MC631517).

Loads ``gate_model/gate_classifier.joblib`` (StandardScaler + multinomial
logistic regression, trained on ``gate_datasets/gate_train.jsonl`` in W3) and
maps its three predicted actions onto CALM's WRITE / ROUTE / REJECT.  Features
come from the shared ``gate_features.py`` extractor, so this arm differs from
``rule_gate`` only in *how* the decision is made.

Caveat documented in the W4 report: ``CALMEditor.edit_requests`` currently
calls ``gate.decide(..., retrieval_context=None)``, so the four context
features are always zero inside the CALM stream.  The gate therefore runs on
the prior-edit and current-answer signals only; the context branch of the
classifier is exercised in W3's offline evaluation, not here.

Selected by ``hparams.conflict_gate_path`` -- no teammate file is modified.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _arm_common import (GATE_MODEL, W3_DIR, BaseArm, Decision,  # noqa: F401,E402
                         EditRecord, load_module)

_FEAT = load_module("tang_w3_gate_features",
                    os.path.join(W3_DIR, "gate_features.py"))
FEATURE_NAMES = _FEAT.FEATURE_NAMES
extract_features = _FEAT.extract_features

# class label used in W3 training data -> CALM decision
LABEL_TO_DECISION = {
    "WRITE": Decision.WRITE,
    "ROUTE": Decision.ROUTE,
    "REJECT": Decision.REJECT,
}


class ConflictGate(BaseArm):
    """Logistic-regression gate trained in W3."""

    name = "learned_gate"

    def __init__(self, n_recent: int = 50, write_threshold: float = 0.3,
                 route_threshold: float = 0.6, similarity_threshold: float = 0.7,
                 seed: int = 0, model_path: Optional[str] = None):
        super().__init__(n_recent=n_recent, write_threshold=write_threshold,
                         route_threshold=route_threshold,
                         similarity_threshold=similarity_threshold, seed=seed)
        import joblib  # imported lazily so the other arms do not need it
        path = model_path or GATE_MODEL
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[learned_gate] missing W3 classifier: {path}. "
                "Run train_gate_classifier.py first."
            )
        self.model_path = path
        self.model = joblib.load(path)

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
        x = [extract_features(sample)]
        label = str(self.model.predict(x)[0])
        return LABEL_TO_DECISION.get(label, Decision.WRITE)
