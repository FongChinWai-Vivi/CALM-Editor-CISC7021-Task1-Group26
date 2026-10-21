"""CLI entry point for the CALM W1 data pipeline.

Runs a dataset loader and the metric battery end-to-end without touching
the EasyEdit code base, using the exact EasyEdit-aligned data schema.

Quick usage::

    # Smoke test on the bundled sample data (no model required)
    python evaluate_pipeline.py --dataset counterfact --mock --limit 3
    python evaluate_pipeline.py --dataset zsre      --mock --limit 3
    python evaluate_pipeline.py --dataset mquake    --mock --limit 3

    # Score precomputed generations (no model required)
    python evaluate_pipeline.py --dataset counterfact \\
        --data-path data/counterfact.json --predictions-json preds.json

For a real model, import the pipeline from Python and pass a generator::

    from evaluate_pipeline import run_with_generator
    def generate(prompt: str) -> str:
        ...  # e.g. model.generate(...) and decode
    print(run_with_generator("counterfact", "data/counterfact.json", generate))

See README.md for the input/output formats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loaders import EditCase, build_loader  # noqa: E402
from metrics import (  # noqa: E402
    compute_paraphrase_acc,
    compute_locality_acc,
    compute_portability_acc,
    compute_rewrite_acc,
)

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_sample_path(dataset_name: str) -> str:
    return os.path.join(SAMPLE_DIR, f"{dataset_name}_sample.json")


def _as_generation_list(value: Union[str, List[str]]) -> List[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def make_mock_generator(cases: Sequence[EditCase]) -> Callable[[str], str]:
    """Rule-based generator used only to verify the pipeline runs end-to-end.

    For every prompt seen in the dataset it returns the expected answer
    (target_new for edit/paraphrase/portability prompts, target_true for
    locality prompts), so all metrics are expected to be 1.0 on sample
    data. It is NOT a model and must never be used for reported numbers.
    """
    mapping: Dict[str, str] = {}
    for case in cases:
        mapping.setdefault(case.prompt, case.target_new)
        for p in case.paraphrasess:
            mapping.setdefault(p, case.target_new)
        for p in case.neighborhood_prompts:
            mapping.setdefault(p, case.target_true)
        for p in case.attribute_prompts:
            mapping.setdefault(p, case.target_true)
        for p in case.portability_prompts:
            mapping.setdefault(p, case.target_new)

    def generate(prompt: str) -> str:
        return mapping.get(prompt, "mock-unknown-answer")

    return generate


def load_predictions(path: str) -> Dict[str, Dict[str, List[str]]]:
    """Load precomputed generations.

    Expected JSON shape (case_id keys are strings)::

        {
          "rewrite":       {"0": ["gen"], ...},
          "paraphrase":    {"0": ["gen", "gen"], ...},
          "neighborhood":  {"0": ["gen"], ...},
          "attribute":     {"0": ["gen"], ...},
          "portability":   {"0": ["gen", "gen"], ...}
        }

    Only keys that appear are scored; missing groups contribute 0.0 and
    are reported with ``null`` so the caller can see they were absent.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    result: Dict[str, Dict[str, List[str]]] = {}
    for group in ("rewrite", "paraphrase", "neighborhood", "attribute",
                  "portability"):
        group_raw = raw.get(group, {})
        result[group] = {
            str(cid): _as_generation_list(gens)
            for cid, gens in group_raw.items()
        }
    return result


def score_from_predictions(cases: Sequence[EditCase],
                           preds: Dict[str, Dict[str, List[str]]]) -> Dict[str, float]:
    """Score a case list against precomputed predictions (no model call)."""
    by_id = {str(case.case_id): case for case in cases}

    rewrite_gens: List[str] = []
    rewrite_tgts: List[str] = []
    para_groups: List[List[str]] = []
    para_tgts: List[str] = []
    neigh_groups: List[List[str]] = []
    neigh_tgts: List[str] = []
    attr_groups: List[List[str]] = []
    attr_tgts: List[str] = []
    port_groups: List[List[str]] = []
    port_tgts: List[str] = []

    has_rewrite = has_para = has_neigh = has_attr = has_port = False

    for cid, case in by_id.items():
        if cid in preds["rewrite"]:
            has_rewrite = True
            rewrite_gens.extend(preds["rewrite"][cid])
            rewrite_tgts.extend([case.target_new] * len(preds["rewrite"][cid]))
        if cid in preds["paraphrase"]:
            has_para = True
            para_groups.append(preds["paraphrase"][cid])
            para_tgts.append(case.target_new)
        if cid in preds["neighborhood"]:
            has_neigh = True
            neigh_groups.append(preds["neighborhood"][cid])
            neigh_tgts.append(case.target_true)
        if cid in preds["attribute"]:
            has_attr = True
            attr_groups.append(preds["attribute"][cid])
            attr_tgts.append(case.target_true)
        if cid in preds["portability"]:
            has_port = True
            port_groups.append(preds["portability"][cid])
            port_tgts.append(case.target_new)

    return {
        "rewrite_acc": compute_rewrite_acc(rewrite_gens, rewrite_tgts)
        if has_rewrite else 0.0,
        "paraphrase_acc": compute_paraphrase_acc(para_groups, para_tgts)
        if has_para else 0.0,
        "locality_neighborhood_acc": compute_locality_acc(neigh_groups, neigh_tgts)
        if has_neigh else 0.0,
        "locality_attribute_acc": compute_locality_acc(attr_groups, attr_tgts)
        if has_attr else 0.0,
        "portability_acc": compute_portability_acc(port_groups, port_tgts)
        if has_port else 0.0,
        "n_cases": len(cases),
    }


def run_with_generator(dataset_name: str,
                       data_path: str,
                       generate_fn: Callable[[str], str],
                       limit: Optional[int] = None) -> Dict[str, float]:
    """Load ``dataset_name`` from ``data_path`` and score with a model callable."""
    from metrics import evaluate_cases

    loader = build_loader(dataset_name, data_path=data_path, limit=limit)
    cases = loader.load()
    return evaluate_cases(cases, generate_fn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CALM W1 dataset loader + metric pipeline."
    )
    parser.add_argument(
        "--dataset", choices=["counterfact", "zsre", "mquake"],
        default="counterfact",
        help="dataset to evaluate",
    )
    parser.add_argument(
        "--data-path", default=None,
        help="path to the dataset JSON (EasyEdit-aligned). Defaults to the "
             "bundled sample file when --mock is used.",
    )
    parser.add_argument(
        "--predictions-json", default=None,
        help="path to precomputed generations to score without a model.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only load the first N cases (small-scale smoke test).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="use the bundled sample data and a rule-based generator to "
             "verify the pipeline end-to-end.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, float]:
    """Execute the pipeline according to the CLI args and return metrics."""
    if args.data_path is None:
        if not args.mock:
            raise SystemExit(
                "provide --data-path (or --mock to use the bundled samples)."
            )
        data_path = _default_sample_path(args.dataset)
    else:
        data_path = args.data_path

    loader = build_loader(args.dataset, data_path=data_path, limit=args.limit)
    cases = loader.load()

    if args.predictions_json:
        preds = load_predictions(args.predictions_json)
        return score_from_predictions(cases, preds)

    if args.mock:
        return run_with_generator(
            args.dataset, data_path, make_mock_generator(cases),
            limit=args.limit,
        )

    raise SystemExit(
        "no generation source given: use --mock or --predictions-json, or "
        "call run_with_generator() from Python with a real model callable."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metrics = run(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
