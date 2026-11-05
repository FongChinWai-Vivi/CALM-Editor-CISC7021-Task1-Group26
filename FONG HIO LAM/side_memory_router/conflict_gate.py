"""ConflictGate — Write/Route/Reject decision engine for CALM.

This module implements the ConflictGate component of the CALM project (W3,
Fong Hio Lam). It decides, for each edit request, whether to:

    - WRITE to model parameters (direct parameter update),
    - ROUTE to side memory (non-parametric storage), or
    - REJECT the edit entirely.

The gate uses a scoring mechanism based on three signals:

    1. Current answer distribution — how well does the current model already
       answer the edit? If it already answers correctly, reject.
    2. Recent stored edits — are there conflicting edits in side memory?
       If so, route to avoid overwriting.
    3. Retrieval context similarity — how similar is this edit to recent ones?
       High similarity suggests a related fact that should be routed.

The module is fully self-contained with no third-party dependencies beyond
Python's standard library. A smoke test is included at the bottom.

Usage as a component inside CALMEditor::

    from conflict_gate import ConflictGate, Decision

    gate = ConflictGate(
        n_recent=50,
        write_threshold=0.3,
        route_threshold=0.6,
    )
    decision = gate.decide(
        subject="Spain", relation="capital", target_new="Madrid",
        current_answer="Madrid", current_confidence=0.95,
        recent_edits=[...], retrieval_context=[...],
    )
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------


class Decision:
    """Decision returned by ConflictGate.decide()."""
    WRITE = "WRITE"      # Write directly to model parameters
    ROUTE = "ROUTE"      # Route to side memory
    REJECT = "REJECT"    # Reject the edit

    @staticmethod
    def from_str(s: str) -> str:
        return s.upper()


# ---------------------------------------------------------------------------
# Edit record for tracking
# ---------------------------------------------------------------------------


@dataclass
class EditRecord:
    """A single edit entry tracked by the gate."""
    subject: str
    relation: str
    target_new: str
    current_answer: str
    confidence: float
    timestamp: float = field(default_factory=lambda: 0.0)

    def signature(self) -> str:
        return f"{self.subject}|{self.relation}|{self.target_new}"

    def conflicts_with(self, other: "EditRecord") -> bool:
        """Check if two edits target the same subject-relation pair."""
        return self.subject == other.subject and self.relation == other.relation


# ---------------------------------------------------------------------------
# ConflictGate
# ---------------------------------------------------------------------------


class ConflictGate:
    """Write/Route/Reject decision engine.

    Parameters
    ----------
    n_recent : int
        Number of recent edits to consider in the sliding window.
    write_threshold : float
        Confidence threshold below which we always WRITE. Lower values mean
        more aggressive writing.
    route_threshold : float
        Confidence threshold above which we prefer ROUTE. Higher values mean
        more conservative routing.
    similarity_threshold : float
        Cosine similarity threshold between edit signatures. Edits with
        similarity >= this value are considered related and routed.
    seed : int
        Random seed for reproducibility.

    Attributes
    ----------
    recent_edits : list[EditRecord]
        Sliding window of recent edit records.
    """

    def __init__(self, n_recent: int = 50, write_threshold: float = 0.3,
                 route_threshold: float = 0.6, similarity_threshold: float = 0.7,
                 reject_conf_threshold: float = 0.35, seed: int = 0):
        self.n_recent = n_recent
        self.write_threshold = write_threshold
        self.route_threshold = route_threshold
        # A REJECT is only issued when the model can be shown to already
        # answer the request, so the decision needs both an answer match and
        # a minimum token-level confidence behind that answer.
        self.reject_conf_threshold = reject_conf_threshold
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        self.rng = random.Random(seed)
        self.recent_edits: List[EditRecord] = []
        self._decision_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_answer(text: str) -> str:
        """Lower-case, strip punctuation and leading articles for matching."""
        if not text:
            return ""
        cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " "
                          for ch in text)
        tokens = [t for t in cleaned.split() if t]
        while tokens and tokens[0] in ("the", "a", "an"):
            tokens.pop(0)
        return " ".join(tokens)

    @classmethod
    def _confidence_score(cls, current_answer: str, target_new: str) -> float:
        """Strict answer-match score in [0, 1].

        The previous version scaled any partial word overlap by 2.0, so a
        single shared token already produced a score above the reject
        threshold and triggered spurious REJECTs.  The score is now a plain
        token-level F1 in [0, 1], and 1.0 is reserved for a normalised exact
        match.
        """
        cur = cls._normalise_answer(current_answer)
        tgt = cls._normalise_answer(target_new)
        if not cur or not tgt:
            return 0.0
        if cur == tgt:
            return 1.0
        # Containment, i.e. the same "exact match or target contained in the
        # generation" rule that the W1 metrics use when scoring free
        # generation.  A greedy generation usually keeps talking after the
        # answer (" French. She is a native of ..."), so an exact-equality
        # test alone would score a correct answer as a miss and the REJECT
        # branch could never fire on generated text.
        if tgt in f" {cur} ":
            return 1.0
        words_cur = set(cur.split())
        words_tgt = set(tgt.split())
        if not words_tgt:
            return 0.0
        precision = len(words_cur & words_tgt) / len(words_cur)
        recall = len(words_cur & words_tgt) / len(words_tgt)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _similarity_score(edit_a: EditRecord, edit_b: EditRecord) -> float:
        """Similarity between two edit records, in [0, 1].

        Relation labels are only coarse strings (e.g. "capital of"), so a
        shared relation alone must not push two unrelated edits over the
        conflict threshold.  Sharing the subject accounts for half of the
        score, sharing the relation another 0.3, and the remaining 0.2 comes
        from target overlap.
        """
        subject_overlap = 0.5 if edit_a.subject == edit_b.subject else 0.0
        relation_overlap = 0.3 if edit_a.relation == edit_b.relation else 0.0

        words_a = set(edit_a.target_new.lower().split())
        words_b = set(edit_b.target_new.lower().split())
        if not words_a or not words_b:
            target_sim = 0.0
        else:
            target_sim = len(words_a & words_b) / len(words_a | words_b)

        return subject_overlap + relation_overlap + 0.2 * target_sim

    # ------------------------------------------------------------------
    # Main decision logic
    # ------------------------------------------------------------------

    def decide(self, subject: str, relation: str, target_new: str,
               current_answer: str, current_confidence: float,
               recent_edits: Optional[List[EditRecord]] = None,
               retrieval_context: Optional[List[str]] = None,
               pre_edit_probe_answer: Optional[str] = None,
               pre_edit_probe_match: Optional[bool] = None) -> str:
        """Decide whether to WRITE, ROUTE, or REJECT an edit.

        Parameters
        ----------
        subject : str
            The entity being edited (e.g., "Spain").
        relation : str
            The attribute being edited (e.g., "capital").
        target_new : str
            The desired new answer.
        current_answer : str
            What the model currently answers.
        current_confidence : float
            Model's confidence in its current answer [0, 1].
        recent_edits : list[EditRecord], optional
            Recent edit records from side memory.
        retrieval_context : list[str], optional
            Retrieved context snippets for additional signal.
        pre_edit_probe_answer : str, optional
            Answer the model gives to a paraphrased probe prompt **before**
            this edit is applied.  This is the pre-edit probe used to justify
            a REJECT: if the untouched model already returns ``target_new``
            for a differently worded prompt, the edit is redundant.
        pre_edit_probe_match : bool, optional
            Whether the pre-edit probe answer matches ``target_new``.  When
            it is ``True`` the REJECT is grounded in a verified model
            behaviour instead of a string heuristic.

        Returns
        -------
        str
            One of Decision.WRITE, Decision.ROUTE, or Decision.REJECT.
        """
        # --- Signal 1: pre-edit probe ---
        # A REJECT must be verifiable: either the model already answers the
        # edit prompt with target_new *and* is reasonably confident, or a
        # paraphrased pre-edit probe already returns target_new.  A bare
        # partial word overlap is no longer enough.
        conf_score = self._confidence_score(current_answer, target_new)
        direct_answer_match = conf_score >= 1.0
        probe_match = bool(pre_edit_probe_match)
        if pre_edit_probe_match is None and pre_edit_probe_answer is not None:
            probe_match = self._confidence_score(
                pre_edit_probe_answer, target_new) >= 1.0
        already_satisfied = probe_match or (
            direct_answer_match
            and current_confidence >= self.reject_conf_threshold)

        # --- Signal 2: Conflicts in recent edits ---
        conflict_count = 0
        max_similarity = 0.0
        edit = EditRecord(subject, relation, target_new, current_answer, current_confidence)
        if recent_edits:
            for rec in recent_edits:
                # Same subject-relation pair is always a potential conflict
                same_sr = (edit.subject == rec.subject and edit.relation == rec.relation)
                sim = self._similarity_score(edit, rec)
                if sim >= self.similarity_threshold or same_sr:
                    conflict_count += 1
                    max_similarity = max(max_similarity, sim)

        has_conflict = conflict_count > 0

        # --- Signal 3: Retrieval context relevance ---
        context_relevance = 0.0
        if retrieval_context:
            target_words = set(target_new.lower().split())
            relevant_snippets = sum(
                1 for ctx in retrieval_context
                if any(w in ctx.lower() for w in target_words)
            )
            context_relevance = relevant_snippets / len(retrieval_context)

        # --- Decision synthesis ---
        if already_satisfied:
            decision = Decision.REJECT
            reason = ("pre_edit_probe_already_correct" if probe_match
                      else "current_answer_matches_target")
        elif has_conflict:
            decision = Decision.ROUTE
            reason = f"conflicts_in_memory({conflict_count})"
        elif current_confidence < self.write_threshold:
            decision = Decision.WRITE
            reason = "low_model_confidence"
        elif context_relevance > 0.5:
            decision = Decision.ROUTE
            reason = "high_retrieval_relevance"
        else:
            # Default: moderate confidence → route to be safe
            decision = Decision.WRITE if current_confidence < self.route_threshold \
                else Decision.ROUTE
            reason = "default_moderate"

        # Log decision
        log_entry = {
            "subject": subject,
            "relation": relation,
            "target_new": target_new,
            "current_answer": current_answer,
            "current_confidence": round(current_confidence, 3),
            "conf_score": round(conf_score, 3),
            "pre_edit_probe_answer": pre_edit_probe_answer,
            "pre_edit_probe_match": probe_match,
            "already_satisfied": already_satisfied,
            "has_conflict": has_conflict,
            "max_similarity": round(max_similarity, 3),
            "context_relevance": round(context_relevance, 3),
            "decision": decision,
            "reason": reason,
        }
        self._decision_log.append(log_entry)

        # Update recent edits
        rec = EditRecord(subject, relation, target_new, current_answer,
                         current_confidence)
        self.recent_edits.append(rec)
        if len(self.recent_edits) > self.n_recent:
            self.recent_edits.pop(0)

        return decision

    def get_decision_summary(self) -> Dict[str, int]:
        """Return counts of each decision type."""
        summary = {"WRITE": 0, "ROUTE": 0, "REJECT": 0}
        for entry in self._decision_log:
            d = entry["decision"]
            if d in summary:
                summary[d] += 1
        return summary

    def clear_history(self) -> None:
        """Clear all decision history."""
        self._decision_log.clear()
        self.recent_edits.clear()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke():
    """Self-contained smoke test demonstrating WRITE/ROUTE/REJECT decisions."""
    print("=" * 60)
    print("ConflictGate Smoke Test")
    print("=" * 60)

    gate = ConflictGate(n_recent=10, write_threshold=0.3,
                        route_threshold=0.6, similarity_threshold=0.7)

    # Test case 1: Current answer matches target → REJECT
    d1 = gate.decide("Spain", "capital", "Madrid", "Madrid", 0.95)
    print(f"\nTest 1 (answer matches): {d1}")
    assert d1 == Decision.REJECT, f"Expected REJECT, got {d1}"

    # Test case 2: Low confidence → WRITE
    d2 = gate.decide("France", "capital", "Paris", "Lyon", 0.1)
    print(f"Test 2 (low confidence): {d2}")
    assert d2 == Decision.WRITE, f"Expected WRITE, got {d2}"

    # Test case 3: Moderate confidence, no conflict → default
    d3 = gate.decide("Germany", "capital", "Berlin", "Munich", 0.5)
    print(f"Test 3 (moderate confidence): {d3}")

    # Test case 4: With conflicting recent edit → ROUTE
    recent = [EditRecord("Spain", "capital", "Barcelona", "Madrid", 0.8)]
    d4 = gate.decide("Spain", "capital", "Madrid", "Paris", 0.5,
                     recent_edits=recent)
    print(f"Test 4 (conflicting edit): {d4}")
    assert d4 == Decision.ROUTE, f"Expected ROUTE, got {d4}"

    # Test case 5: High retrieval relevance → ROUTE
    ctx = ["Madrid is the capital of Spain", "Spanish language"]
    d5 = gate.decide("Italy", "capital", "Rome", "Florence", 0.5,
                     retrieval_context=ctx)
    print(f"Test 5 (high retrieval relevance): {d5}")

    # Summary
    summary = gate.get_decision_summary()
    print(f"\nDecision summary: {summary}")
    print("=" * 60)
    print("Smoke test passed.")


if __name__ == "__main__":
    run_smoke()
