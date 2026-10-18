"""CounterFact data loader.

Expects EasyEdit-aligned CounterFact JSON records::

    {
      "case_id": 0,
      "request": {
        "prompt": "What is the official language of Spain?",
        "subject": "Spain",
        "target_new": {"str": "French", "id": "Q150"},
        "target_true": {"str": "Spanish", "id": "Q1321"}
      },
      "paraphrasess": ["...", "..."],          # or "paraphrase_prompts"
      "locality_prompts": {
        "neighborhood_prompts": ["..."],
        "attribute_prompts": ["..."]
      }
    }

A list of such records is expected; wrapper dicts (``{"data": [...]}``)
are tolerated by the base loader.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseDatasetLoader, EditCase, _as_list, _as_text


class CounterFactLoader(BaseDatasetLoader):
    """Loader for CounterFact single-fact editing records."""

    dataset_name = "counterfact"

    def _parse_record(self, record: Dict[str, Any], case_id: int) -> EditCase:
        request = record.get("request", record)
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
        return EditCase(
            case_id=case_id,
            prompt=_as_text(request.get("prompt", "")),
            subject=_as_text(request.get("subject", "")),
            target_new=_as_text(request.get("target_new", "")),
            target_true=_as_text(request.get("target_true", "")),
            paraphrasess=paraphrasess,
            neighborhood_prompts=neighborhood,
            attribute_prompts=attribute,
        )
