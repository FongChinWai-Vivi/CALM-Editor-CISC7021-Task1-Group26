"""Base classes and shared helpers for the CALM W1 data loaders.

All loaders output a uniform :class:`EditCase` structure aligned with the
EasyEdit dataset convention: ``request`` (prompt / subject / target_new /
target_true), ``paraphrasess`` for generalisation, ``locality_prompts``
(neighborhood + attribute) for locality, and ``portability_prompts`` for
multi-hop ripple (MQuAKE). Target values may appear either as a plain
string or as a dict with a ``"str"`` key (EasyEdit style).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _as_text(value: Any) -> str:
    """Extract the text of a target that may be ``str`` or ``{"str": ...}``."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("str"), str):
            return value["str"]
        if isinstance(value.get("text"), str):
            return value["text"]
    return str(value)


def _as_list(value: Any) -> List[str]:
    """Normalise a field into a list of strings (tolerates None/str/list)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):  # locality_prompts dict handled by callers
        return [_as_text(value)]
    return [_as_text(v) for v in value]


@dataclass
class EditCase:
    """A single knowledge-edit evaluation case (EasyEdit-aligned schema).

    Attributes:
        case_id: stable index within the dataset file.
        prompt: the edit request prompt (subject + relation).
        subject: the subject string of the edit.
        target_new: the new (edited) answer.
        target_true: the original (pre-edit) answer.
        paraphrasess: paraphrase prompts of the request (generalisation).
        neighborhood_prompts: unrelated prompts near the subject (locality).
        attribute_prompts: unrelated prompts about other attributes (locality).
        portability_prompts: multi-hop questions entailed by the edit (MQuAKE).
    """

    case_id: int
    prompt: str
    subject: str
    target_new: str
    target_true: str
    paraphrasess: List[str] = field(default_factory=list)
    neighborhood_prompts: List[str] = field(default_factory=list)
    attribute_prompts: List[str] = field(default_factory=list)
    portability_prompts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to the EasyEdit-aligned JSON shape."""
        return {
            "case_id": self.case_id,
            "request": {
                "prompt": self.prompt,
                "subject": self.subject,
                "target_new": {"str": self.target_new},
                "target_true": {"str": self.target_true},
            },
            "paraphrasess": self.paraphrasess,
            "locality_prompts": {
                "neighborhood_prompts": self.neighborhood_prompts,
                "attribute_prompts": self.attribute_prompts,
            },
            "portability_prompts": self.portability_prompts,
        }


class BaseDatasetLoader:
    """Base class for dataset loaders.

    Subclasses set :attr:`dataset_name` and implement
    :meth:`_parse_record`. The public API (:meth:`load`, :meth:`sample`,
    :attr:`size`) is shared and safe to call repeatedly (results are
    cached).
    """

    dataset_name: str = "base"

    def __init__(self, data_path: Optional[str] = None, limit: Optional[int] = None):
        self.data_path = data_path
        self.limit = limit
        self._cases: Optional[List[EditCase]] = None

    # -- record IO ------------------------------------------------------

    def _load_records(self) -> List[Dict[str, Any]]:
        """Read and validate the raw JSON file into a list of records."""
        if self.data_path is None:
            raise ValueError(
                f"{self.dataset_name}: data_path is required. Pass --data-path "
                "or use the bundled sample data (--mock)."
            )
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"data file not found: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):  # tolerate wrappers like {"data": [...]}
            raw = raw.get("data", raw.get("instances", raw.get("records", [])))
        if not isinstance(raw, list):
            raise ValueError(
                f"{self.dataset_name}: expected a JSON list, got {type(raw).__name__}"
            )
        return raw

    def _parse_record(self, record: Dict[str, Any], case_id: int) -> EditCase:
        """Subclass hook: turn one raw record into an :class:`EditCase`."""
        raise NotImplementedError

    # -- public API ------------------------------------------------------

    def load(self) -> List[EditCase]:
        """Load all cases (optionally capped by ``limit``)."""
        if self._cases is None:
            records = self._load_records()
            if self.limit is not None:
                records = records[: self.limit]
            self._cases = [
                self._parse_record(rec, idx) for idx, rec in enumerate(records)
            ]
        return self._cases

    def sample(self, n: int = 2) -> List[EditCase]:
        """Return the first ``n`` cases (convenience for smoke tests)."""
        return self.load()[:n]

    @property
    def size(self) -> int:
        """Number of loaded cases."""
        return len(self.load())
