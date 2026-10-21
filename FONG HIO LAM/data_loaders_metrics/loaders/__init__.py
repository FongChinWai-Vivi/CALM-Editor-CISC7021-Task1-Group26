"""Dataset loaders for the CALM W1 data pipelines.

Loaders convert raw dataset JSON (CounterFact / zsRE / MQuAKE) into the
uniform :class:`EditCase` schema that is aligned with EasyEdit's
``easyeditor/dataset`` convention, so the same metric pipeline and the
future ``CALMEditor`` evaluation flow can consume all three datasets.
"""

from .base import BaseDatasetLoader, EditCase
from .counterfact import CounterFactLoader
from .zsre import ZsreLoader
from .mquake import MQuaKELoader

LOADER_REGISTRY = {
    "counterfact": CounterFactLoader,
    "zsre": ZsreLoader,
    "mquake": MQuaKELoader,
}


def build_loader(dataset_name, data_path=None, limit=None):
    """Instantiate the loader registered under ``dataset_name``.

    Args:
        dataset_name: one of "counterfact", "zsre", "mquake".
        data_path: optional path to the dataset JSON file.
        limit: optional cap on the number of cases to load (smoke tests).
    """
    if dataset_name not in LOADER_REGISTRY:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; expected one of "
            f"{sorted(LOADER_REGISTRY)}"
        )
    return LOADER_REGISTRY[dataset_name](data_path=data_path, limit=limit)


__all__ = [
    "BaseDatasetLoader",
    "EditCase",
    "CounterFactLoader",
    "ZsreLoader",
    "MQuaKELoader",
    "build_loader",
    "LOADER_REGISTRY",
]
