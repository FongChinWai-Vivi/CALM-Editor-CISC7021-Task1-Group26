"""Evaluation metrics for the CALM W1 pipeline.

Implements the metrics locked in the team's evaluation protocol
(``docs/evaluation_protocol.md``), following the EasyEdit conventions:

- ``rewrite_acc`` : reliability -- the edited prompt recalls ``target_new``;
- ``paraphrase_acc`` : generalisation -- paraphrases recall ``target_new``;
- ``locality_neighborhood_acc`` / ``locality_attribute_acc`` : locality --
  unrelated prompts keep ``target_true`` after the edit;
- ``portability_acc`` : multi-hop ripple -- entailed questions answer with
  the edited fact (``target_new``), used for MQuAKE.

A generation counts as a match if, after light normalisation (lowercase,
strip punctuation, collapse whitespace), the target answer is contained in
or equals the generation. This is the same exact-or-containment protocol
used by EasyEdit's generation-based evaluation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

try:
    from .loaders import EditCase  # package-relative import
except ImportError:                # standalone import (evaluate_pipeline)
    from loaders import EditCase

# ---------------------------------------------------------------------------
# Matching primitives
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def match(generation: str, answer: str) -> bool:
    """Normalised exact-or-containment match of a generation against an answer."""
    gen = normalize(generation)
    ans = normalize(answer)
    if not ans or not gen:
        return False
    return ans == gen or ans in gen


def accuracy(generations: Sequence[str], answers: Sequence[str]) -> float:
    """Fraction of matched (generation, answer) pairs (0.0 on empty input)."""
    if not generations:
        return 0.0
    hits = sum(1.0 for g, a in zip(generations, answers) if match(g, a))
    return hits / len(generations)


# ---------------------------------------------------------------------------
# Metric functions (list-level; each metric maps 1:1 generations to answers)
# ---------------------------------------------------------------------------


def compute_rewrite_acc(generations: Sequence[str],
                        targets: Sequence[str]) -> float:
    """Reliability: fraction of edit prompts whose generation matches target_new."""
    return accuracy(generations, targets)


def compute_paraphrase_acc(generations_per_case: Sequence[Sequence[str]],
                           targets: Sequence[str]) -> float:
    """Generalisation: all paraphrase generations vs. each case's target_new.

    The flattened protocol matches EasyEdit, where every paraphrase prompt
    of a case is scored against that case's new answer.
    """
    flat_gens: List[str] = []
    flat_tgts: List[str] = []
    for group, target in zip(generations_per_case, targets):
        flat_gens.extend(group)
        flat_tgts.extend([target] * len(group))
    return accuracy(flat_gens, flat_tgts)


def compute_locality_acc(generations_per_case: Sequence[Sequence[str]],
                         targets: Sequence[str]) -> float:
    """Locality: all locality generations vs. each case's target_true."""
    flat_gens: List[str] = []
    flat_tgts: List[str] = []
    for group, target in zip(generations_per_case, targets):
        flat_gens.extend(group)
        flat_tgts.extend([target] * len(group))
    return accuracy(flat_gens, flat_tgts)


def compute_portability_acc(generations_per_case: Sequence[Sequence[str]],
                            targets: Sequence[str]) -> float:
    """Portability: multi-hop generations vs. each case's target_new."""
    flat_gens: List[str] = []
    flat_tgts: List[str] = []
    for group, target in zip(generations_per_case, targets):
        flat_gens.extend(group)
        flat_tgts.extend([target] * len(group))
    return accuracy(flat_gens, flat_tgts)


# ---------------------------------------------------------------------------
# End-to-end scoring over EditCase lists
# ---------------------------------------------------------------------------


def evaluate_cases(cases: Sequence[EditCase],
                   generate_fn: Callable[[str], str]) -> Dict[str, float]:
    """Run the full metric battery over a list of cases.

    ``generate_fn(prompt) -> str`` returns the model's generation for a
    prompt. It is invoked once per prompt (request, each paraphrase, each
    locality prompt, each portability prompt), so no batching is required
    for the smoke-test path. Wrap a batched generator with an LRU cache
    when used at scale.

    Returns a dict with the four metric families plus ``n_cases``.
    """
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

    for case in cases:
        rewrite_gens.append(generate_fn(case.prompt))
        rewrite_tgts.append(case.target_new)

        if case.paraphrasess:
            para_groups.append([generate_fn(p) for p in case.paraphrasess])
            para_tgts.append(case.target_new)

        if case.neighborhood_prompts:
            neigh_groups.append(
                [generate_fn(p) for p in case.neighborhood_prompts]
            )
            neigh_tgts.append(case.target_true)

        if case.attribute_prompts:
            attr_groups.append([generate_fn(p) for p in case.attribute_prompts])
            attr_tgts.append(case.target_true)

        if case.portability_prompts:
            port_groups.append(
                [generate_fn(p) for p in case.portability_prompts]
            )
            port_tgts.append(case.target_new)

    return {
        "rewrite_acc": compute_rewrite_acc(rewrite_gens, rewrite_tgts),
        "paraphrase_acc": compute_paraphrase_acc(para_groups, para_tgts)
        if para_groups else 0.0,
        "locality_neighborhood_acc": compute_locality_acc(neigh_groups, neigh_tgts)
        if neigh_groups else 0.0,
        "locality_attribute_acc": compute_locality_acc(attr_groups, attr_tgts)
        if attr_groups else 0.0,
        "portability_acc": compute_portability_acc(port_groups, port_tgts)
        if port_groups else 0.0,
        "n_cases": len(cases),
    }
