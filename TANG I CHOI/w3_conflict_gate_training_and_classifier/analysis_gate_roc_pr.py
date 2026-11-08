"""analysis_gate_roc_pr.py -- T3/T4 audit of the learned gate.

Owner: Tang I Choi (MC631517).  Work package: W3 classifier, used by the W4
ablation and by the W6 self-audit of the gate acceptance criteria.

Why this script exists
----------------------
The report used to place the offline number (0.877 accuracy) next to the
in-stream number (0.424 agreement, 0.340 cold start) without saying *why* they
differ.  Accuracy alone cannot answer "is the ranking broken, or did the
distribution move?".  This script answers that with four quantities computed on
the same 835-record CounterFact test split:

  * AUC and AP for WRITE-vs-rest (ranking quality),
  * TPR at the decision threshold (does the gate ever let a write through?),
  * the realised write share (how many of 100 edits it admits), and
  * the 3x3 confusion matrix against the gold action.

Each quantity is computed twice: on the split as labelled (the offline
distribution, where the context features are populated) and with
``retrieval_context`` emptied for every record.  The second run reproduces what
the classifier actually receives inside the CALM stream, because
``CALMEditor.edit_requests`` calls ``gate.decide(..., retrieval_context=None)``
(``EasyEdit/easyeditor/editors/calm_editor.py:203``).  The difference between
the two columns is the covariate shift; whatever is left is the ranking.

It also sweeps the WRITE probability threshold and reports, per threshold, how
many edits out of 100 would be admitted and how many of those are gold WRITE
(the closest available proxy for ``param_hit_total``; the fluency half of the
double objective, ``nll_delta``, is *not* collected by the current runner and is
reported as missing rather than faked).

Usage::

    python analysis_gate_roc_pr.py            # writes results/gate_roc_pr.{json,md}
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             roc_auc_score)

from gate_features import extract_matrix  # noqa: E402

ACTIONS = ["WRITE", "ROUTE", "REJECT"]
MODEL_PATH = os.path.join(HERE, "gate_model", "gate_classifier.joblib")
TEST_PATH = os.path.join(HERE, "gate_datasets", "gate_test.jsonl")
W4_RESULTS = os.path.join(HERE, "..", "w4_gate_ablations", "results")
OUT_JSON = os.path.join(HERE, "results", "gate_roc_pr.json")
OUT_MD = os.path.join(HERE, "results", "gate_roc_pr.md")


class _SoftmaxView:
    """Version-independent view over the persisted StandardScaler+LogReg pipeline.

    The checkpoint was written with scikit-learn 1.9.1 while the local runtime
    ships 1.5.1, whose ``LogisticRegression.predict_proba`` cannot read the newer
    object (it looks for the removed ``multi_class`` attribute).  The fitted
    parameters -- ``mean_``, ``scale_``, ``coef_``, ``intercept_`` -- unpickle
    fine, so the forward pass is evaluated by hand below.  This changes no
    number in the model, only the code path that reads it.
    """

    def __init__(self, pipeline):
        scaler, clf = pipeline.steps[0][1], pipeline.steps[-1][1]
        self.mean = np.asarray(scaler.mean_, dtype=float)
        self.scale = np.asarray(scaler.scale_, dtype=float)
        self.coef = np.asarray(clf.coef_, dtype=float)
        self.intercept = np.asarray(clf.intercept_, dtype=float)
        self.classes_ = [str(c) for c in clf.classes_]

    def predict_proba(self, X):
        Z = (np.asarray(X, dtype=float) - self.mean) / self.scale
        logits = Z @ self.coef.T + self.intercept
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


def load_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def strip_prior_edits(rows):
    """Drop the side-memory signal, reproducing the first edit of a stream."""
    out = []
    for row in rows:
        clone = dict(row)
        clone["prior_edits"] = []
        out.append(clone)
    return out


def strip_context(rows):
    out = []
    for row in rows:
        clone = dict(row)
        clone["retrieval_context"] = []
        out.append(clone)
    return out


def score(model, rows, tag):
    X = extract_matrix(rows)
    proba = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]
    p_write = proba[:, classes.index("WRITE")]
    gold = [str(r["action"]) for r in rows]
    pred = [classes[int(i)] for i in np.argmax(proba, axis=1)]
    y_write = np.array([1 if g == "WRITE" else 0 for g in gold])

    auc = float(roc_auc_score(y_write, p_write))
    ap = float(average_precision_score(y_write, p_write))
    hard = [1 if p >= 0.5 else 0 for p in p_write]
    tp = sum(1 for h, y in zip(hard, y_write) if h and y)
    fn = sum(1 for h, y in zip(hard, y_write) if not h and y)
    tpr = float(tp / (tp + fn)) if (tp + fn) else 0.0
    write_share = float(sum(1 for p in pred if p == "WRITE") / len(pred))
    hit_total = float(sum(1 for p, g in zip(pred, gold) if p == "WRITE" and g == "WRITE"))
    admitted = sum(1 for p in pred if p == "WRITE")
    precision = float(hit_total / admitted) if admitted else 0.0
    cm = confusion_matrix(gold, pred, labels=ACTIONS).tolist()
    return {
        "tag": tag,
        "n": len(rows),
        "gold_action_distribution": {a: gold.count(a) for a in ACTIONS},
        "auc_write_vs_rest": round(auc, 4),
        "average_precision_write": round(ap, 4),
        "tpr_at_0.5": round(tpr, 4),
        "argmax_write_share": round(write_share, 4),
        "argmax_admitted_per_100": round(write_share * 100, 1),
        "argmax_admitted_gold_write_precision": round(precision, 4),
        "confusion_matrix_gold_rows": {"labels": ACTIONS, "matrix": cm},
        "n_context_features_nonzero": int(sum(
            1 for r in rows if (r.get("retrieval_context") or []))),
    }


def sweep(model, rows, thresholds):
    X = extract_matrix(rows)
    proba = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]
    p_write = proba[:, classes.index("WRITE")]
    gold = [str(r["action"]) for r in rows]
    out = []
    for t in thresholds:
        admitted = [1 if p >= t else 0 for p in p_write]
        n_adm = sum(admitted)
        gold_write_admitted = sum(1 for a, g in zip(admitted, gold)
                                  if a and g == "WRITE")
        out.append({
            "threshold": round(t, 2),
            "admitted_per_100": round(n_adm / len(rows) * 100, 1),
            "admitted_gold_write_precision": round(
                gold_write_admitted / n_adm, 4) if n_adm else None,
            "gold_write_recall": round(
                gold_write_admitted / max(gold.count("WRITE"), 1), 4),
        })
    return out



def plot_sweeps(sweep_rows, sweep_cold, out_png):
    """T4 wants a curve, not a table row: admission vs. precision per threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    panels = [(axes[0], sweep_rows, "stream path (retrieval_context emptied)"),
              (axes[1], sweep_cold, "cold start (context + prior history emptied)")]
    for axis, rows, title in panels:
        t = [r["threshold"] for r in rows]
        admitted = [r["admitted_per_100"] for r in rows]
        precision = [100.0 * (r["admitted_gold_write_precision"] or 0.0) for r in rows]
        recall = [100.0 * r["gold_write_recall"] for r in rows]
        axis.plot(t, admitted, marker="o", lw=1.8, label="admitted per 100 edits")
        axis.plot(t, precision, marker="s", lw=1.2, label="precision of admitted (%)")
        axis.plot(t, recall, marker="^", lw=1.2, ls="--", label="gold WRITE recall (%)")
        axis.axhline(50, color="grey", ls=":", lw=1.0)
        axis.annotate("coverage target 0.50", xy=(t[0], 51), fontsize=7, color="grey")
        axis.set_xlabel("p_WRITE threshold")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("edits per 100 / percent")
    fig.suptitle("T4 - single-objective threshold sweep at W=50 "
                 "(hit side only; nll_delta is not collected)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return out_png


def stream_control():
    """What the four arms actually admitted in the 191-edit stream."""
    control = {}
    for arm in ["always_write", "always_route", "rule_gate", "learned_gate"]:
        path = os.path.join(W4_RESULTS, f"w4_ablation_{arm}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        per_edit = data.get("per_edit") or []
        n = len(per_edit) or 1
        write = sum(1 for r in per_edit if r.get("decision") == "WRITE")
        model_hit = sum(1 for r in per_edit if r.get("final_model_hit"))
        e2e_hit = sum(1 for r in per_edit if r.get("final_e2e_hit"))
        gold_write = sum(1 for r in per_edit if r.get("gold_action") == "WRITE")
        control[arm] = {
            "n_edits": len(per_edit),
            "write": write,
            "write_per_100": round(write / n * 100, 1),
            "gold_write": gold_write,
            "model_hit_total": model_hit,
            "e2e_hit_total": e2e_hit,
            "model_hit_per_write": round(model_hit / write, 4) if write else None,
        }
    return control


def main():
    model = _SoftmaxView(joblib.load(MODEL_PATH))
    rows = load_rows(TEST_PATH)

    offline = score(model, rows, "offline_test_counterfact (as labelled)")
    noctx = score(model, strip_context(rows),
                  "stream_simulation (retrieval_context emptied)")
    cold = score(model, strip_prior_edits(strip_context(rows)),
                 "cold_start_simulation (retrieval_context and prior_edits emptied)")
    thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    sweep_rows = sweep(model, strip_context(rows), thresholds)
    sweep_cold = sweep(model, strip_prior_edits(strip_context(rows)), thresholds)

    result = {
        "owner": "Tang I Choi (MC631517)",
        "purpose": "T3 attribution (ranking vs distribution shift) and T4 threshold sweep",
        "model": MODEL_PATH,
        "test_split": TEST_PATH,
        "splits": {"offline": offline, "stream_simulation": noctx,
                   "cold_start_simulation": cold},
        "delta_offline_minus_stream": {
            k: round(offline[k] - noctx[k], 4)
            for k in ["auc_write_vs_rest", "average_precision_write",
                      "tpr_at_0.5", "argmax_write_share"]
        },
        "delta_offline_minus_cold_start": {
            k: round(offline[k] - cold[k], 4)
            for k in ["auc_write_vs_rest", "average_precision_write",
                      "tpr_at_0.5", "argmax_write_share"]
        },
        "threshold_sweep_on_stream_simulation": sweep_rows,
        "threshold_sweep_on_cold_start_simulation": sweep_cold,
        "stream_control_191_edits": stream_control(),
        "missing_metric": {
            "nll_delta": "not collected by run_w4_gate_ablation.py; the double "
                         "objective (hit_total, nll_delta) is therefore reported "
                         "as half-missing, never as a single-objective pass",
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    md = []
    md.append("# Learned gate: ranking vs distribution shift (T3) and threshold sweep (T4)\n")
    md.append("Generated by `analysis_gate_roc_pr.py` on the 835-record CounterFact "
              "test split. All columns use the same trained classifier and the "
              "same records; they differ only in which signals are emptied, so the "
              "gap between them isolates the distribution shift from the ranking.\n")
    md.append("- **offline** — the split as labelled (context and side memory populated).\n"
              "- **stream simulation** — `retrieval_context` emptied for every record, "
              "because `CALMEditor.edit_requests` calls `gate.decide(..., "
              "retrieval_context=None)` (`EasyEdit/easyeditor/editors/calm_editor.py:203`).\n"
              "- **cold start** — `retrieval_context` *and* `prior_edits` emptied, which "
              "is the state of the gate before the first edit of a stream.\n")
    md.append("| quantity | offline | stream simulation | cold start |")
    md.append("|---|---|---|---|")
    keys = [("auc_write_vs_rest", "AUC, WRITE vs rest"),
            ("average_precision_write", "AP, WRITE"),
            ("tpr_at_0.5", "TPR @ p>=0.5"),
            ("argmax_write_share", "realised write share (argmax)"),
            ("argmax_admitted_per_100", "admitted per 100 edits"),
            ("argmax_admitted_gold_write_precision", "precision of admitted (gold WRITE)")]
    for key, label in keys:
        md.append(f"| {label} | {offline[key]} | {noctx[key]} | {cold[key]} |")
    md.append("")
    md.append("### Confusion matrices (rows = gold, columns = WRITE / ROUTE / REJECT)\n")
    for tag, block in [("offline", offline), ("stream simulation", noctx),
                       ("cold start", cold)]:
        md.append(f"**{tag}**\n")
        md.append("| gold \\ pred | WRITE | ROUTE | REJECT |")
        md.append("|---|---|---|---|")
        for label, rowcm in zip(ACTIONS, block["confusion_matrix_gold_rows"]["matrix"]):
            md.append(f"| {label} | {rowcm[0]} | {rowcm[1]} | {rowcm[2]} |")
        md.append("")
    md.append("### Threshold sweep on the stream simulation\n")
    md.append("| p_WRITE threshold | admitted per 100 | precision (gold WRITE) | gold WRITE recall |")
    md.append("|---|---|---|---|")
    for r in sweep_rows:
        md.append(f"| {r['threshold']:.2f} | {r['admitted_per_100']} | "
                  f"{r['admitted_gold_write_precision']} | {r['gold_write_recall']} |")
    md.append("")
    md.append("### Threshold sweep on the cold start\n")
    md.append("| p_WRITE threshold | admitted per 100 | precision (gold WRITE) | gold WRITE recall |")
    md.append("|---|---|---|---|")
    for r in sweep_cold:
        md.append(f"| {r['threshold']:.2f} | {r['admitted_per_100']} | "
                  f"{r['admitted_gold_write_precision']} | {r['gold_write_recall']} |")
    md.append("")
    md.append("### Control: what the 191-edit stream actually admitted\n")
    md.append("| arm | admitted / 100 | gold WRITE | model hit | E2E hit | model hit per write |")
    md.append("|---|---|---|---|---|---|")
    for arm, c in result["stream_control_191_edits"].items():
        md.append(f"| `{arm}` | {c['write_per_100']} | {c['gold_write']} | "
                  f"{c['model_hit_total']} | {c['e2e_hit_total']} | "
                  f"{c['model_hit_per_write']} |")
    md.append("")
    md.append("**Missing half of the double objective.** `nll_delta` is not "
              "recorded by `run_w4_gate_ablation.py`, so no row above pairs a hit "
              "count with a fluency delta. Treat every admission number here as "
              "coverage-only evidence, not as a pass on the (hit, NLL) criterion.\n")
    png = os.path.join(os.path.dirname(OUT_JSON), "fig_threshold_sweep.png")
    plot_sweeps(sweep_rows, sweep_cold, png)
    md.append("\n![T4 threshold sweep](fig_threshold_sweep.png)\n")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print(json.dumps({"offline": {k: offline[k] for k in
                                  ["auc_write_vs_rest", "average_precision_write",
                                   "tpr_at_0.5", "argmax_write_share"]},
                      "stream": {k: noctx[k] for k in
                                 ["auc_write_vs_rest", "average_precision_write",
                                  "tpr_at_0.5", "argmax_write_share"]},
                      "cold_start": {k: cold[k] for k in
                                     ["auc_write_vs_rest", "average_precision_write",
                                      "tpr_at_0.5", "argmax_write_share"]}},
                     indent=2))
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
