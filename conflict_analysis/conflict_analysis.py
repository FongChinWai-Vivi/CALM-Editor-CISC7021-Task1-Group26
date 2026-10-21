#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conflict_analysis.py -- Conflict-type error analysis for CALM W5 (Fong Hio Lam).

Given the W4 evaluation outputs (MQuAKE / WikiBigEdit) of an edited model,
this self-contained script performs a per-sample error analysis. Each sample
is labelled with exactly one of the four conflict tags defined in the W0
taxonomy and used by the CALM ConflictGate design:

    * no-conflict              -- the edit does not collide with any known knowledge
    * pre-training-conflict    -- the edit contradicts knowledge in the model's pretraining
    * prior-edit-conflict      -- the edit contradicts an earlier edit applied in the same run
    * context-conflict         -- the edit is inconsistent with the context/supporting facts

For every tag the script reports:
    * number of samples, correct predictions, errors
    * error rate (errors / samples)
    * share of all errors contributed by each tag (error distribution)

It then renders two figures:
    * conflict_error_rates.png      -- grouped bar chart of error rate per tag
    * conflict_error_distribution.png -- pie chart of error share per tag

A small built-in smoke test (synthetic samples covering all four tags) is
provided so the pipeline can be verified without the full evaluation outputs.

Dependencies (all listed in README.md):
    * Python >= 3.9
    * numpy, matplotlib
    * (optional, only for smoke test) nothing else

Usage:
    python conflict_analysis.py --results <json> [--output-dir <dir>]
    python conflict_analysis.py --smoke-test [--output-dir <dir>]

Input format (JSON):
    A JSON object with the structure produced by the W4 evaluation scripts:

    {
      "metadata": {
          "task": "mquake" | "wikibigedit",
          "model": "Llama-3-8B",
          ...
      },
      "per_sample_results": [
          {
              "sample_id": 0,
              "conflict_tag": "pre-training-conflict",
              "prediction": "answer string or null",
              "expected": "gold answer string",
              "correct": true|false
          },
          ...
      ]
    }

    Every sample in "per_sample_results" must carry a valid "conflict_tag";
    the "correct" field is used as the ground truth for error accounting.
"""

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    plt = None
    _MPL_IMPORT_ERROR = exc
else:
    _MPL_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Conflict taxonomy (locked in W0 / used by W3 ConflictGate)
# ---------------------------------------------------------------------------
CONFLICT_TAGS = [
    "no-conflict",
    "pre-training-conflict",
    "prior-edit-conflict",
    "context-conflict",
]

TAG_COLORS = {
    "no-conflict": "#2e7d32",            # green
    "pre-training-conflict": "#c62828",  # red
    "prior-edit-conflict": "#f9a825",    # amber
    "context-conflict": "#1565c0",       # blue
}


# ---------------------------------------------------------------------------
# Data loading / validation
# ---------------------------------------------------------------------------
class ConflictAnalysisError(Exception):
    """Raised for invalid evaluation result files."""


def load_results(path):
    """Load and validate a W4-style evaluation JSON file.

    Args:
        path (str): Path to the JSON results file.

    Returns:
        tuple (metadata, samples): (dict, list of dict)

    Raises:
        ConflictAnalysisError: if the file is not valid JSON or lacks the
            required per-sample structure.
    """
    if not os.path.isfile(path):
        raise ConflictAnalysisError(f"Results file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConflictAnalysisError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConflictAnalysisError("Top-level JSON must be an object.")
    if "per_sample_results" not in data:
        raise ConflictAnalysisError(
            "Missing key 'per_sample_results' in results file."
        )

    samples = data["per_sample_results"]
    if not isinstance(samples, list):
        raise ConflictAnalysisError("'per_sample_results' must be a list.")
    if not samples:
        raise ConflictAnalysisError("'per_sample_results' is empty.")

    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ConflictAnalysisError(f"Sample {i} is not an object.")
        missing = [k for k in ("sample_id", "conflict_tag", "correct") if k not in sample]
        if missing:
            raise ConflictAnalysisError(
                f"Sample {i} missing required keys: {', '.join(missing)}"
            )
        tag = sample["conflict_tag"]
        if tag not in CONFLICT_TAGS:
            raise ConflictAnalysisError(
                f"Sample {i} has unknown conflict_tag: {tag!r}. "
                f"Expected one of {CONFLICT_TAGS}."
            )
        if not isinstance(sample["correct"], bool):
            raise ConflictAnalysisError(
                f"Sample {i} field 'correct' must be a boolean."
            )

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, samples


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(samples):
    """Compute per-conflict-type error statistics.

    Args:
        samples (list of dict): Validated per-sample evaluation records.

    Returns:
        OrderedDict: tag -> dict with keys:
            samples, correct, errors, error_rate, error_share
    """
    total = len(samples)
    totals_by_tag = Counter(s["conflict_tag"] for s in samples)
    errors_by_tag = Counter(
        s["conflict_tag"] for s in samples if not s["correct"]
    )
    correct_by_tag = Counter(
        s["conflict_tag"] for s in samples if s["correct"]
    )
    total_errors = sum(errors_by_tag.values())

    stats = OrderedDict()
    for tag in CONFLICT_TAGS:
        n = totals_by_tag.get(tag, 0)
        err = errors_by_tag.get(tag, 0)
        corr = correct_by_tag.get(tag, 0)
        error_rate = (err / n) if n else 0.0
        error_share = (err / total_errors) if total_errors else 0.0
        stats[tag] = {
            "samples": n,
            "correct": corr,
            "errors": err,
            "error_rate": error_rate,
            "error_share": error_share,
        }
    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_error_rates(stats, output_path):
    """Render a grouped bar chart of error rate per conflict tag.

    Args:
        stats (OrderedDict): output of :func:`analyze`.
        output_path (str): Destination PNG path.
    """
    if plt is None:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting; install it with "
            "`pip install matplotlib`."
        )
    tags = list(stats.keys())
    rates = [stats[t]["error_rate"] * 100.0 for t in tags]
    counts = [stats[t]["errors"] for t in tags]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    bars = ax.bar(
        tags,
        rates,
        color=[TAG_COLORS[t] for t in tags],
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, rate, count in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"{rate:.1f}% (n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Error rate (%)")
    ax.set_xlabel("Conflict type")
    ax.set_title("Per-conflict-type error rate of the edited model")
    ax.set_ylim(0, max(rates) * 1.25 + 2.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_error_distribution(stats, output_path):
    """Render a pie chart of the error distribution across conflict tags.

    Args:
        stats (OrderedDict): output of :func:`analyze`.
        output_path (str): Destination PNG path.
    """
    if plt is None:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting; install it with "
            "`pip install matplotlib`."
        )
    errors = [stats[t]["errors"] for t in CONFLICT_TAGS]
    labels = [t for t in CONFLICT_TAGS]
    colors = [TAG_COLORS[t] for t in CONFLICT_TAGS]

    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=150)
    total_errors = sum(errors)
    if total_errors == 0:
        ax.text(
            0.5, 0.5, "No errors found",
            ha="center", va="center", fontsize=13, transform=ax.transAxes,
        )
        ax.set_title("Error distribution across conflict types (no errors)")
    else:
        wedges, _texts, autotexts = ax.pie(
            errors,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%",
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title(
            f"Error distribution across conflict types (total errors: {total_errors})"
        )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report(metadata, stats, sample_count):
    """Produce a plain-text analysis report.

    Args:
        metadata (dict): metadata block from the results file.
        stats (OrderedDict): output of :func:`analyze`.
        sample_count (int): total number of samples.

    Returns:
        str: Human-readable report.
    """
    lines = []
    task = metadata.get("task", "unknown")
    model = metadata.get("model", "unknown")
    lines.append("=" * 74)
    lines.append("CONFLICT-TYPE ERROR ANALYSIS -- CALM W5")
    lines.append("=" * 74)
    lines.append(f"task      : {task}")
    lines.append(f"model     : {model}")
    lines.append(f"samples   : {sample_count}")
    lines.append("")
    header = (
        f"{'conflict tag':<24}{'samples':>9}{'correct':>9}"
        f"{'errors':>9}{'error rate':>12}{'err share':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for tag, s in stats.items():
        lines.append(
            f"{tag:<24}{s['samples']:>9}{s['correct']:>9}{s['errors']:>9}"
            f"{s['error_rate'] * 100:>11.1f}%{s['error_share'] * 100:>11.1f}%"
        )
    lines.append("-" * len(header))
    total_errors = sum(s["errors"] for s in stats.values())
    lines.append(f"{'TOTAL':<24}{sample_count:>9}{sample_count - total_errors:>9}"
                 f"{total_errors:>9}"
                 f"{(total_errors / sample_count) * 100:>11.1f}%"
                 f"{'100.0%':>12}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def build_smoke_samples():
    """Construct small synthetic samples covering all four conflict tags.

    Returns:
        dict: A results-file-shaped dictionary ready for :func:`analyze`.
    """
    synthetic = [
        # no-conflict: neutral edits that do not collide with known knowledge
        {"sample_id": 0, "conflict_tag": "no-conflict",
         "prediction": "Paris", "expected": "Paris", "correct": True},
        {"sample_id": 1, "conflict_tag": "no-conflict",
         "prediction": "Berlin", "expected": "Berlin", "correct": True},
        {"sample_id": 2, "conflict_tag": "no-conflict",
         "prediction": "Rome", "expected": "Rome", "correct": True},
        # pre-training-conflict: contradicting factual knowledge
        {"sample_id": 3, "conflict_tag": "pre-training-conflict",
         "prediction": "Lisbon", "expected": "Madrid", "correct": False},
        {"sample_id": 4, "conflict_tag": "pre-training-conflict",
         "prediction": "Vienna", "expected": "Madrid", "correct": False},
        {"sample_id": 5, "conflict_tag": "pre-training-conflict",
         "prediction": "Madrid", "expected": "Madrid", "correct": True},
        # prior-edit-conflict: colliding with an earlier edit
        {"sample_id": 6, "conflict_tag": "prior-edit-conflict",
         "prediction": "Athens", "expected": "Oslo", "correct": False},
        {"sample_id": 7, "conflict_tag": "prior-edit-conflict",
         "prediction": "Oslo", "expected": "Oslo", "correct": True},
        {"sample_id": 8, "conflict_tag": "prior-edit-conflict",
         "prediction": "Dublin", "expected": "Oslo", "correct": False},
        # context-conflict: inconsistent with supporting context
        {"sample_id": 9, "conflict_tag": "context-conflict",
         "prediction": "Tokyo", "expected": "Seoul", "correct": False},
        {"sample_id": 10, "conflict_tag": "context-conflict",
         "prediction": "Seoul", "expected": "Seoul", "correct": True},
        {"sample_id": 11, "conflict_tag": "context-conflict",
         "prediction": "Manila", "expected": "Seoul", "correct": False},
    ]
    return {
        "metadata": {"task": "smoke-test", "model": "synthetic", "note": "built-in W5 smoke test"},
        "per_sample_results": synthetic,
    }


def run_smoke_test(output_dir):
    """Run the built-in smoke test end to end and return its summary.

    Args:
        output_dir (str): Directory to store report and figures.

    Returns:
        str: One-line summary of the smoke-test outcome.
    """
    data = build_smoke_samples()
    metadata, samples = load_results_from_dict(data)
    stats = analyze(samples)
    report = format_report(metadata, stats, len(samples))

    report_path = os.path.join(output_dir, "smoke_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    rate_path = os.path.join(output_dir, "smoke_conflict_error_rates.png")
    dist_path = os.path.join(output_dir, "smoke_conflict_error_distribution.png")
    plot_error_rates(stats, rate_path)
    plot_error_distribution(stats, dist_path)

    total_errors = sum(s["errors"] for s in stats.values())
    return (
        f"SMOKE TEST PASSED: {len(samples)} synthetic samples, "
        f"{total_errors} errors; report and 2 figures written to {output_dir}"
    )


def load_results_from_dict(data):
    """Load and validate a results dict (used by the smoke test)."""
    if not isinstance(data, dict) or "per_sample_results" not in data:
        raise ConflictAnalysisError("Invalid results dictionary.")
    samples = data["per_sample_results"]
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ConflictAnalysisError(f"Sample {i} is not an object.")
        missing = [k for k in ("sample_id", "conflict_tag", "correct") if k not in sample]
        if missing:
            raise ConflictAnalysisError(
                f"Sample {i} missing required keys: {', '.join(missing)}"
            )
        tag = sample["conflict_tag"]
        if tag not in CONFLICT_TAGS:
            raise ConflictAnalysisError(
                f"Sample {i} has unknown conflict_tag: {tag!r}."
            )
        if not isinstance(sample["correct"], bool):
            raise ConflictAnalysisError(f"Sample {i} 'correct' must be bool.")
    metadata = data.get("metadata", {})
    return (metadata if isinstance(metadata, dict) else {}), samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Conflict-type error analysis for CALM W5."
    )
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to a W4 evaluation JSON file (MQuAKE/WikiBigEdit).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for the report and figures (default: cwd).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run the built-in synthetic smoke test instead of a real file.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point of the conflict analysis pipeline."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if _MPL_IMPORT_ERROR is not None:  # pragma: no cover
        print(
            f"WARNING: matplotlib not importable ({_MPL_IMPORT_ERROR}); "
            "figures will be skipped. Run `pip install matplotlib`.",
            file=sys.stderr,
        )

    out_dir = args.output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke_test:
        summary = run_smoke_test(out_dir)
        print(summary)
        return 0

    if not args.results:
        print(
            "Error: provide --results <json> or use --smoke-test.",
            file=sys.stderr,
        )
        return 2

    metadata, samples = load_results(args.results)
    stats = analyze(samples)
    report = format_report(metadata, stats, len(samples))
    print(report)

    report_path = os.path.join(out_dir, "conflict_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    if plt is not None:
        plot_error_rates(
            stats, os.path.join(out_dir, "conflict_error_rates.png")
        )
        plot_error_distribution(
            stats, os.path.join(out_dir, "conflict_error_distribution.png")
        )
        print(f"\nFigures written to {out_dir}")
    else:  # pragma: no cover
        print("\nFigures skipped (matplotlib unavailable).")

    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
