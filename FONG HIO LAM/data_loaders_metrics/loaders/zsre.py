"""zsRE data loader.

Expects EasyEdit-aligned zsRE JSON records. zsRE items follow the same
shape as CounterFact (``request`` + ``paraphrasess`` +
``locality_prompts``), which is exactly the schema used by the MEND /
ROME / MEMIT pipelines inside EasyEdit::

    {
      "case_id": 0,
      "request": {
        "prompt": "What is the nationality of Albert Einstein?",
        "subject": "Albert Einstein",
        "target_new": {"str": "Chinese", "id": "Q..."},
        "target_true": {"str": "German", "id": "Q..."}
      },
      "paraphrasess": ["..."],
      "locality_prompts": {
        "neighborhood_prompts": ["..."],
        "attribute_prompts": ["..."]
      }
    }
"""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseDatasetLoader, EditCase, _as_list, _as_text


class ZsreLoader(BaseDatasetLoader):
    """Loader for zsRE zero-shot relation-extraction editing records."""

    dataset_name = "zsre"

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
