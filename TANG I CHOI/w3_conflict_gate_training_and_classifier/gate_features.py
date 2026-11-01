"""gate_features.py -- feature extraction for the CALM conflict gate classifier.

W3 (Tang I Choi, MC631517).  Owner: TANG I CHOI.

The gate classifier sees exactly the signals that the CALM architecture makes
available at edit time (README "Conflict Gate": current answer distribution,
closest previously stored edit, retrieved context).  Nothing that is only known
to the annotation team (the four-way conflict label, the rationale, the source
split) is exposed, so the classifier cannot read the answer off a leaked field.

Feature groups
--------------
A. Current answer distribution (signal 1)
   ``answer_exact``, ``answer_contains``, ``target_in_answer_tokens``,
   ``confidence``, ``answer_len``, ``target_len``

B. Closest previously stored edits / side memory (signal 2)
   ``n_prior_edits``, ``prior_same_subject``, ``prior_same_sr``,
   ``prior_target_match``, ``prior_target_jaccard``

C. Retrieved context (signal 3)
   ``has_context``, ``n_context``, ``context_target_overlap``,
   ``context_prior_overlap``, ``context_mentions_subject``

D. Edit-form metadata (cheap, no leakage)
   ``subject_len``, ``relation_len``, ``relation_is_id``

Usage::

    from gate_features import FEATURE_NAMES, extract_features
    x = extract_features(sample)          # list[float], len == len(FEATURE_NAMES)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

FEATURE_NAMES: List[str] = [
    # A. current answer distribution
    "answer_exact",
    "answer_contains",
    "target_in_answer_tokens",
    "confidence",
    "answer_len",
    "target_len",
    # B. side memory / recent edits
    "n_prior_edits",
    "prior_same_subject",
    "prior_same_sr",
    "prior_target_match",
    "prior_target_jaccard",
    # C. retrieved context
    "has_context",
    "n_context",
    "context_target_overlap",
    "context_prior_overlap",
    "context_mentions_subject",
    # D. edit form
    "subject_len",
    "relation_len",
    "relation_is_id",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def _tokens(text: Any) -> List[str]:
    return _TOKEN_RE.findall(_norm(text))


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _overlap(a: str, b: str) -> float:
    """Fraction of b's tokens that also appear in a."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not tb:
        return 0.0
    return len(ta & tb) / len(tb)


def extract_features(sample: Dict[str, Any]) -> List[float]:
    """Map one gate sample to the fixed feature vector.

    ``sample`` must provide ``target_new``, ``current_answer``,
    ``current_confidence`` and may provide ``subject``, ``relation``,
    ``prior_edits`` (list of dicts: subject/relation/target) and
    ``retrieval_context`` (list of strings).
    """
    target = str(sample.get("target_new") or "")
    subject = str(sample.get("subject") or "")
    relation = str(sample.get("relation") or "")
    answer = str(sample.get("current_answer") or "")
    conf = float(sample.get("current_confidence") or 0.0)
    prior_edits = sample.get("prior_edits") or []
    context = sample.get("retrieval_context") or []

    # --- A. current answer distribution ------------------------------- #
    answer_exact = float(_norm(answer) == _norm(target) and _norm(target) != "")
    answer_contains = float(_norm(target) != "" and _norm(target) in _norm(answer))
    target_in_answer_tokens = _overlap(answer, target)
    answer_len = float(len(_tokens(answer)))
    target_len = float(len(_tokens(target)))

    # --- B. prior edits / side memory --------------------------------- #
    n_prior = float(len(prior_edits))
    prior_same_subject = 0.0
    prior_same_sr = 0.0
    prior_target_match = 0.0
    prior_target_jaccard = 0.0
    if prior_edits:
        last = prior_edits[-1]
        if _norm(last.get("subject")) == _norm(subject):
            prior_same_subject = 1.0
        same_sr = [
            e for e in prior_edits
            if _norm(e.get("subject")) == _norm(subject)
            and _norm(e.get("relation")) == _norm(relation)
        ]
        if same_sr:
            prior_same_sr = 1.0
            prior_target_match = float(
                _norm(same_sr[-1].get("target")) == _norm(target))
            prior_target_jaccard = _jaccard(same_sr[-1].get("target"), target)

    # --- C. retrieved context ----------------------------------------- #
    has_context = float(bool(context))
    n_context = float(len(context))
    ctx_join = " ".join(str(c) for c in context)
    context_target_overlap = _overlap(ctx_join, target) if context else 0.0
    prior_values = " ".join(str(e.get("target") or "") for e in prior_edits)
    context_prior_overlap = _overlap(ctx_join, prior_values) if (context and prior_values) else 0.0
    subj_tokens = set(_tokens(subject))
    context_mentions_subject = float(
        bool(context) and bool(subj_tokens & set(_tokens(ctx_join))))

    # --- D. edit form -------------------------------------------------- #
    subject_len = float(len(_tokens(subject)))
    relation_len = float(len(_tokens(relation)))
    relation_is_id = float(bool(re.fullmatch(r"[PpQq]\d+|[A-Za-z]+\d+", relation.strip())))

    return [
        answer_exact, answer_contains, target_in_answer_tokens,
        conf, answer_len, target_len,
        n_prior, prior_same_subject, prior_same_sr,
        prior_target_match, prior_target_jaccard,
        has_context, n_context, context_target_overlap,
        context_prior_overlap, context_mentions_subject,
        subject_len, relation_len, relation_is_id,
    ]


def extract_matrix(samples: List[Dict[str, Any]]) -> List[List[float]]:
    return [extract_features(s) for s in samples]
