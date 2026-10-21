#!/usr/bin/env python3
"""AlphaEdit baseline reproduction (CALM W2, Fong Hio Lam).

This script extends the MEMIT reproduction in ``memit_repro.py`` with the
null-space constrained editing mechanism of *AlphaEdit: Null-Space
Constrained Knowledge Editing for Language Models* (Chen et al.).

Highlights
----------
* Reuses the toy transformer, edit-case builder and evaluation battery
  from :mod:`memit_repro` so the two baselines are directly comparable.
* Adds a *null-space projector* built from the context-key covariance:
  the delta written by MEMIT is projected onto the orthogonal complement
  of the top principal directions of the model's own feature space.  This
  hard constraint preserves unrelated facts more aggressively than the
  soft covariance regulariser of plain MEMIT.
* Reports a MEMIT-vs-AlphaEdit side-by-side comparison on the W1 metric
  battery (rewrite / paraphrase / locality / portability), which is the
  empirical evidence needed for Gate 2.

Run the self-contained smoke test with::

    python alphaedit_repro.py --mock --cases 3 --seed 0 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from memit_repro import (  # reuse the toy model + protocol from MEMIT
    EDIT_LAYER,
    HIDDEN,
    MiniLM,
    build_smoke_env,
    cache_keys,
    context_covariance,
    compute_layer_update,
    desired_residuals,
    evaluate_cases,
)


# ---------------------------------------------------------------------------
# Null-space machinery (AlphaEdit core mechanism)
# ---------------------------------------------------------------------------


def null_space_projector(context_keys: np.ndarray,
                         n_components: int = 32) -> np.ndarray:
    """Build the null-space projector ``P = I - V V^T``.

    ``V`` contains the top ``n_components`` right-singular vectors of the
    context-key matrix (rows = context activations at the edit layer).
    These directions capture the model's typical feature space; projecting
    an update away from them keeps the edit from disturbing unrelated
    facts (locality), which is exactly the AlphaEdit design:

        delta_W_alpha = delta_W_memit @ P
    """
    if n_components <= 0 or n_components >= HIDDEN:
        raise ValueError("n_components must satisfy 0 < n_components < HIDDEN")
    u, s, vt = np.linalg.svd(context_keys, full_matrices=False)
    del u, s  # only the right singular vectors are needed
    V = vt[:n_components].T                     # (HIDDEN, n_components)
    return np.eye(HIDDEN) - V @ V.T             # (HIDDEN, HIDDEN)


# ---------------------------------------------------------------------------
# AlphaEdit editor (EasyEdit BaseEditor-style surface)
# ---------------------------------------------------------------------------


class AlphaEditor:
    """AlphaEdit extension of the MEMIT batch editor.

    Usage::

        editor = AlphaEditor.from_hparams(model=model, layer=EDIT_LAYER,
                                          lam=1e-3, n_components=64)
        editor.batch_edit(cases)
        metrics = editor.evaluate(cases)
    """

    def __init__(self, model: MiniLM, layer: int = EDIT_LAYER,
                 lam: float = 1e-3, margin: float = 1.0,
                 context_scale: float = 5.0, n_components: int = 8,
                 rescale: bool = True, context_seed: int = 0):
        self.model = model
        self.layer = layer
        self.lam = lam
        self.margin = margin
        self.context_scale = context_scale
        self.n_components = n_components
        self.rescale = rescale
        self.context_seed = context_seed
        self.backup: Optional[List[np.ndarray]] = None

    @classmethod
    def from_hparams(cls, model: MiniLM, layer: int = EDIT_LAYER,
                     lam: float = 1e-3, margin: float = 1.0,
                     context_scale: float = 5.0, n_components: int = 8,
                     rescale: bool = True) -> "AlphaEditor":
        """Create an editor from hyper-parameters (EasyEdit convention)."""
        return cls(model=model, layer=layer, lam=lam, margin=margin,
                   context_scale=context_scale, n_components=n_components,
                   rescale=rescale)

    def build_projector(self, n_context: int = 512) -> np.ndarray:
        """Compute the null-space projector from context activations."""
        C = context_covariance(self.model, n_context=n_context,
                               seed=self.context_seed)
        eigvals, eigvecs = np.linalg.eigh(C)
        # SVD of the context-key matrix is equivalent to the eigen
        # decomposition of C = A^T A; take the top directions of C.
        order = np.argsort(eigvals)[::-1][:self.n_components]
        V = eigvecs[:, order]                    # (HIDDEN, n_components)
        return np.eye(HIDDEN) - V @ V.T

    def batch_edit(self, cases: Sequence[Dict]) -> Dict:
        """Write all edits in one AlphaEdit batch update.

        The MEMIT update is computed first and then hard-projected onto
        the null space of the model's feature directions; an optional
        norm-rescaling restores the magnitude lost by the projection.
        """
        if self.backup is None:
            self.backup = self.model.deepcopy_weights()
        delta_memit = compute_layer_update(
            self.model, cases, lam=self.lam, margin=self.margin,
            context_scale=self.context_scale, context_seed=self.context_seed)
        P = self.build_projector()
        delta_alpha = delta_memit @ P
        if self.rescale and np.linalg.norm(delta_alpha) > 1e-12:
            delta_alpha *= (np.linalg.norm(delta_memit)
                            / np.linalg.norm(delta_alpha))
        old = self.model.get_ffn_weight(self.layer)
        self.model.set_ffn_weight(self.layer, old + delta_alpha)
        return {
            "method": "AlphaEdit",
            "layer": self.layer,
            "n_edits": len(cases),
            "n_components": self.n_components,
            "rescaled": self.rescale,
            "delta_norm_memit": float(np.linalg.norm(delta_memit)),
            "delta_norm_after_projection": float(np.linalg.norm(delta_alpha)),
            "post_edit_answers": [
                "t" + str(case["prompt"][0]) + " -> t"
                + str(self.model.generate(case["prompt"]))
                for case in cases
            ],
        }

    def evaluate(self, cases: Sequence[Dict]) -> Dict[str, float]:
        """Run the W1 metric battery on the edited model."""
        return evaluate_cases(self.model, cases)


# ---------------------------------------------------------------------------
# Side-by-side comparison driver
# ---------------------------------------------------------------------------


def compare_baselines(n_cases: int = 3, seed: int = 0,
                      lam: float = 1e-3, margin: float = 1.0,
                      context_scale: float = 5.0, n_components: int = 8,
                      rescale: bool = True) -> Dict:
    """Fit one toy model, then edit it once with MEMIT and once with
    AlphaEdit (restored from the same backup) and compare the metrics.

    Using the same initial model for both baselines keeps the comparison
    fair: only the editing mechanism differs.
    """
    model, cases = build_smoke_env(seed=seed, n_cases=n_cases)

    memit_editor = MemitEditorFacade(model, layer=EDIT_LAYER, lam=lam,
                                     margin=margin,
                                     context_scale=context_scale)
    memit_summary = memit_editor.batch_edit(cases)
    memit_metrics = memit_editor.evaluate(cases)

    # Restore the pristine weights before the AlphaEdit run.
    model.set_ffn_weight(EDIT_LAYER, memit_editor.backup[EDIT_LAYER])

    alpha_editor = AlphaEditor(model, layer=EDIT_LAYER, lam=lam, margin=margin,
                               context_scale=context_scale,
                               n_components=n_components, rescale=rescale)
    alpha_summary = alpha_editor.batch_edit(cases)
    alpha_metrics = alpha_editor.evaluate(cases)

    return {
        "memit": {"summary": memit_summary, "metrics": memit_metrics},
        "alphaedit": {"summary": alpha_summary, "metrics": alpha_metrics},
    }


# Small facade so the comparison does not duplicate MemitEditor's import.
class MemitEditorFacade:
    """Thin wrapper around the MEMIT update used by the comparison."""

    def __init__(self, model: MiniLM, layer: int = EDIT_LAYER,
                 lam: float = 1e-3, margin: float = 1.0,
                 context_scale: float = 5.0):
        self.model = model
        self.layer = layer
        self.lam = lam
        self.margin = margin
        self.context_scale = context_scale
        self.backup = None

    def batch_edit(self, cases: Sequence[Dict]) -> Dict:
        if self.backup is None:
            self.backup = self.model.deepcopy_weights()
        delta_W = compute_layer_update(self.model, cases, lam=self.lam,
                                       margin=self.margin,
                                       context_scale=self.context_scale)
        old = self.model.get_ffn_weight(self.layer)
        self.model.set_ffn_weight(self.layer, old + delta_W)
        return {
            "method": "MEMIT",
            "layer": self.layer,
            "n_edits": len(cases),
            "delta_norm": float(np.linalg.norm(delta_W)),
            "post_edit_answers": [
                "t" + str(case["prompt"][0]) + " -> t"
                + str(self.model.generate(case["prompt"]))
                for case in cases
            ],
        }

    def evaluate(self, cases: Sequence[Dict]) -> Dict[str, float]:
        return evaluate_cases(self.model, cases)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_smoke(n_cases: int = 3, seed: int = 0, lam: float = 1e-3,
              margin: float = 1.0, context_scale: float = 5.0,
              n_components: int = 8, rescale: bool = True,
              verbose: bool = False) -> Dict:
    """End-to-end AlphaEdit smoke test with a MEMIT side-by-side."""
    result = compare_baselines(n_cases=n_cases, seed=seed, lam=lam,
                               margin=margin, context_scale=context_scale,
                               n_components=n_components, rescale=rescale)
    if verbose:
        print("[smoke] MEMIT summary:", json.dumps(result["memit"]["summary"],
                                                   indent=2))
        print("[smoke] MEMIT metrics:",
              json.dumps(result["memit"]["metrics"], indent=2))
        print("[smoke] AlphaEdit summary:",
              json.dumps(result["alphaedit"]["summary"], indent=2))
        print("[smoke] AlphaEdit metrics:",
              json.dumps(result["alphaedit"]["metrics"], indent=2))
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaEdit baseline reproduction (CALM W2, "
                    "null-space constrained extension of MEMIT)."
    )
    parser.add_argument("--mock", action="store_true",
                        help="run the smoke test on the built-in toy model.")
    parser.add_argument("--cases", type=int, default=3,
                        help="number of toy edit cases (smoke test).")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for reproducibility.")
    parser.add_argument("--lam", type=float, default=1e-3,
                        help="ridge regularisation weight (MEMIT solve).")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="safety margin for the new answer logit.")
    parser.add_argument("--context-scale", type=float, default=5.0,
                        help="weight of the context covariance term.")
    parser.add_argument("--n-components", type=int, default=8,
                        help="top feature directions removed by the "
                             "null-space projection (0 < k < HIDDEN).")
    parser.add_argument("--no-rescale", action="store_true",
                        help="disable the norm rescaling after projection.")
    parser.add_argument("--verbose", action="store_true",
                        help="print both edit summaries and metrics.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.mock:
        print(__doc__)
        return 0
    result = run_smoke(n_cases=args.cases, seed=args.seed, lam=args.lam,
                       margin=args.margin, context_scale=args.context_scale,
                       n_components=args.n_components,
                       rescale=not args.no_rescale, verbose=args.verbose)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

