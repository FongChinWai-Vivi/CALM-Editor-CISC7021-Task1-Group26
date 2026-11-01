"""W5 (Tang I Choi) - figures for the results section.

Reads the artefacts produced by W3/W4 and writes three PNGs into ``figures/``:

  fig1_decision_mix.png  - share of WRITE / ROUTE / REJECT per ablation arm
  fig2_retention.png     - model-only vs end-to-end retention per arm
  fig3_gate_gap.png      - gate accuracy offline vs inside the CALM stream

All labels are English so that no CJK font is required on a fresh machine.

Usage::

    python make_w5_figures.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
W4_RESULTS = os.path.join(REPO, "TANG I CHOI", "w4_gate_ablations", "results")
W3_RESULTS = os.path.join(REPO, "TANG I CHOI",
                          "w3_conflict_gate_training_and_classifier", "results")
OUT = os.path.join(HERE, "figures")
ARMS = ["always_write", "always_route", "rule_gate", "learned_gate"]
DECISIONS = ["WRITE", "ROUTE", "REJECT"]
COLORS = {"WRITE": "#2c7fb8", "ROUTE": "#41ab5d", "REJECT": "#c23b22"}


def load(name: str, base: str) -> Dict:
    path = os.path.join(base, name)
    if not os.path.exists(path):
        raise SystemExit(f"[w5-fig] missing {path}; run W4/W3 scripts first")
    with open(path) as f:
        return json.load(f)


def fig_decision_mix(stats: Dict) -> str:
    rows = {r["arm"]: r for r in stats["summary"]}
    arms = [a for a in ARMS if a in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottom = np.zeros(len(arms))
    for dec in DECISIONS:
        vals = np.array([rows[a][dec.lower()] / rows[a]["n_edits"] for a in arms])
        ax.bar(arms, vals, bottom=bottom, label=dec, color=COLORS[dec], width=0.62)
        bottom += vals
    ax.set_ylabel("share of edit requests")
    ax.set_ylim(0, 1)
    ax.set_title("Gate decision mix per ablation arm "
                 f"(n={rows[arms[0]]['n_edits']} edits)")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT, "fig1_decision_mix.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_retention(stats: Dict) -> str:
    rows = {r["arm"]: r for r in stats["summary"]}
    arms = [a for a in ARMS if a in rows]
    x = np.arange(len(arms))
    model = [rows[a]["model_retention"] for a in arms]
    e2e = [rows[a]["e2e_retention"] for a in arms]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - 0.19, model, width=0.36, label="model-only (weights)",
           color="#9ecae1")
    ax.bar(x + 0.19, e2e, width=0.36, label="end-to-end (weights + side memory)",
           color="#2c7fb8")
    for i, (m, e) in enumerate(zip(model, e2e)):
        ax.text(i - 0.19, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.19, e + 0.02, f"{e:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, arms)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("retention after the whole stream")
    ax.set_title("Retention of every edit at the end of the stream")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT, "fig2_retention.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_gate_gap(gap: Dict, stats: Dict) -> str:
    gates = ["rule_gate", "learned_gate"]
    offline = [gap["gates"][g]["offline"]["accuracy"] for g in gates]
    online = [gap["gates"][g]["online"]["accuracy"] for g in gates]
    stream = {r["arm"]: r["gate_acc"] for r in stats["summary"]}
    in_stream = [stream.get(g, np.nan) for g in gates]

    x = np.arange(len(gates))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bars = (
        ("offline (context + history given)", offline, "#bdbdbd", -0.26),
        ("in-stream (real history, context=None)", in_stream, "#2c7fb8", 0.0),
        ("cold start (no history, no context)", online, "#e6a020", 0.26),
    )
    for label, values, colour, dx in bars:
        ax.bar(x + dx, values, width=0.24, label=label, color=colour)
        for i, v in enumerate(values):
            if not np.isnan(v):
                ax.text(i + dx, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, ["rule-based gate", "learned (W3) gate"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy vs. gold action")
    ax.set_title("Gate accuracy: offline features vs. the CALM stream path")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT, "fig3_gate_gap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    stats = load("w4_statistics.json", W4_RESULTS)
    gap = load("gate_online_gap.json", W3_RESULTS)
    for path in (fig_decision_mix(stats), fig_retention(stats),
                 fig_gate_gap(gap, stats)):
        print(f"[w5-fig] wrote {path}")


if __name__ == "__main__":
    main()
