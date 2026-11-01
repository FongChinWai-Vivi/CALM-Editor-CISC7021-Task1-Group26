"""always_route_gate.py -- W4 ablation arm: never touch the parameters.

Owner: TANG I CHOI (MC631517).

Every edit goes to side memory.  The base model is never edited, so this arm
has perfect locality and zero parameter pollution, but no parametric knowledge
gain: its end-to-end retention is purely memory retrieval.  It is the reference
for "how much does the gate lose by writing at all".

Selected by ``hparams.conflict_gate_path`` -- no teammate file is modified.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _arm_common import (BaseArm, Decision, EditRecord,  # noqa: F401,E402
                         load_module)


class ConflictGate(BaseArm):
    """Every edit is routed to side memory."""

    name = "always_route"

    def _decide(self, subject: str, relation: str, target_new: str,
                current_answer: str, current_confidence: float,
                recent_edits: List[Any],
                retrieval_context: Optional[List[str]]) -> str:
        return Decision.ROUTE
