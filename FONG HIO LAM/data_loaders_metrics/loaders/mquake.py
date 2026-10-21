"""MQuAKE data loader.

Supports two input shapes:

1. **EasyEdit-aligned** records with an explicit ``portability_prompts``
   list holding the multi-hop questions entailed by the edit::

       {
         "case_id": 0,
         "request": {"prompt": "...", "subject": "...",
                     "target_new": {"str": "..."}, "target_true": {"str": "..."}},
         "paraphrasess": ["..."],
         "locality_prompts": {"neighborhood_prompts": [...],
                              "attribute_prompts": [...]},
         "portability_prompts": ["multi-hop q1", "multi-hop q2"]
       }

2. **Raw MQuAKE** records with ``requested_rewrite`` + ``questions``; the
   loader extracts every ``question`` string into ``portability_prompts``
   so the metric pipeline can score the multi-hop ripple effect::

       {
         "case_id": 1,
         "requested_rewrite": {"prompt": "...", "subject": "...",
                               "target_new": {"str": "..."},
                               "target_true": {"str": "..."}},
         "questions": [{"question": "...", "answer": "...", "reason": "..."}],
         "locality_prompts": {...}
       }
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseDatasetLoader, EditCase, _as_list, _as_text


class MQuaKELoader(BaseDatasetLoader):
    """Loader for MQuAKE multi-hop editing records (both shapes above)."""

    dataset_name = "mquake"

    def _parse_record(self, record: Dict[str, Any], case_id: int) -> EditCase:
        # Accept both the EasyEdit "request" key and the raw "requested_rewrite".
        request = record.get("request", record.get("requested_rewrite", record))
        locality = record.get("locality_prompts", {}) or {}

        paraphrasess = _as_list(
            record.get("paraphrasess", record.get("paraphrase_prompts", []))
        )
        neighborhood = _as_list(
            locality.get("neighborhood_prompts",
                         record.get("neighborhood_prompts", []))
        )
        attribute = _as_list(
            locality.get("attribute_prompts",
                         record.get("attribute_prompts", []))
        )

        # Portability: prefer an explicit list, otherwise fall back to the
        # raw MQuAKE "questions" chain.
        portability = _as_list(record.get("portability_prompts", []))
        if not portability:
            questions = record.get("questions", [])
            portability = _as_list(
                [q.get("question") for q in questions
                 if isinstance(q, dict) and q.get("question")]
            )

        return EditCase(
            case_id=case_id,
            prompt=_as_text(request.get("prompt", "")),
            subject=_as_text(request.get("subject", "")),
            target_new=_as_text(request.get("target_new", "")),
            target_true=_as_text(request.get("target_true", "")),
            paraphrasess=paraphrasess,
            neighborhood_prompts=neighborhood,
            attribute_prompts=attribute,
            portability_prompts=portability,
        )
