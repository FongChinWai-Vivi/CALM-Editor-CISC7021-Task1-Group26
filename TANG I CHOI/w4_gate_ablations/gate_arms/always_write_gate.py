"""always_write_gate.py -- W4 ablation arm: never gate, always WRITE.

Owner: TANG I CHOI (MC631517).

Lower-bound arm: every edit is pushed into the model parameters.  This is the
"no conflict-awareness" baseline that the CALM gate has to beat: it maximises
immediate rewrite success and minimises it after a long stream (parameter
pollution), and it destroys locality.

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
    """Every edit is written to the parameters."""

    name = "always_write"

    def _decide(self, subject: str, relation: str, target_new: str,
                current_answer: str, current_confidence: float,
                recent_edits: List[Any],
                retrieval_context: Optional[List[str]]) -> str:
        return Decision.WRITE
