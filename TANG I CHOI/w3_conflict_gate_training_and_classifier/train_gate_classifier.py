"""train_gate_classifier.py -- W3 conflict-gate classifier.

Owner: TANG I CHOI (MC631517).

Trains the conflict gate that CALMEditor calls instead of the hand-written
``ConflictGate.decide`` heuristic.  The gate reads only what the editor knows at
decision time (current answer/confidence, side-memory state, retrieved context,
edit shape) and predicts one of three actions:

    WRITE   -- commit the edit to the parameters
    ROUTE   -- do not touch the parameters; write to side memory
    REJECT  -- drop the edit, it is redundant or conflicts with the model

Data
----
``gate_datasets/gate_train.jsonl``   -- CounterFact-derived situations (fit)
``gate_datasets/gate_dev.jsonl``     -- held-out situations (model selection)
``gate_datasets/gate_test.jsonl``    -- held-out CounterFact subjects (in-distribution test)
``gate_datasets/gate_test_pilot100.jsonl`` -- W1 human-labelled pilot (out-of-distribution test)

The pilot-100 file keeps its *human* four-way labels, so it is reported
separately and is never used for fitting or model selection.

Artifacts
---------
``gate_model/gate_classifier.joblib``  -- fitted pipeline (scaler + logistic
                                          regression), loadable by CALMEditor
``gate_model/gate_classifier.json``    -- coefficients, feature names, metrics,
                                          confusion matrices, git-friendly summary
``gate_model/metrics.md``              -- human-readable metrics tables

Usage::

    python "TANG I CHOI/w3_conflict_gate_training_and_classifier/train_gate_classifier.py"
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gate_features import FEATURE_NAMES, extract_matrix  # noqa: E402

DATA_DIR = os.path.join(HERE, "gate_datasets")
MODEL_DIR = os.path.join(HERE, "gate_model")
ACTIONS = ("WRITE", "ROUTE", "REJECT")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def xy(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = extract_matrix(rows)
    y = np.array([r["action"] for r in rows])
    return X, y


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, int]]:
    matrix = {t: {p: 0 for p in ACTIONS} for t in ACTIONS}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def per_class_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for cls in ACTIONS:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        stats[cls] = {
            "support": int((y_true == cls).sum()),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    return stats


def evaluate(model, rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    X, y = xy(rows)
    pred = model.predict(X)
    return {
        "split": name,
        "n": int(len(rows)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro",
                                        labels=list(ACTIONS), zero_division=0)), 4),
        "per_class": per_class_stats(y, pred),
        "confusion": confusion(y, pred),
        "majority_class_acc": round(
            float(Counter(y).most_common(1)[0][1]) / max(len(y), 1), 4),
    }


def baseline_rules(rows: List[Dict[str, Any]]) -> List[str]:
    """The W1 heuristic, kept as a reference arm.

    Same priority order as the annotation manual: prior-edit conflict, then
    context conflict, then the pre-training (answer/confidence) check.
    """
    F = {n: i for i, n in enumerate(FEATURE_NAMES)}
    preds = []
    for row in rows:
        v = extract_matrix([row])[0]
        if v[F["prior_same_sr"]] > 0 and v[F["prior_target_match"]] == 0:
            preds.append("ROUTE")
        elif v[F["has_context"]] > 0 and v[F["context_target_overlap"]] < 0.5:
            preds.append("ROUTE")
        elif v[F["answer_exact"]] > 0:
            preds.append("REJECT")
        elif v[F["confidence"]] >= 0.5:
            preds.append("REJECT")
        else:
            preds.append("WRITE")
    return preds


def rule_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    _, y = xy(rows)
    pred = np.array(baseline_rules(rows))
    return {
        "n": int(len(rows)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro",
                                        labels=list(ACTIONS), zero_division=0)), 4),
        "confusion": confusion(y, pred),
    }


def md_table(title: str, matrix: Dict[str, Dict[str, int]]) -> str:
    lines = [f"**{title}**", "", "| true \\ pred | " + " | ".join(ACTIONS) + " |",
             "| --- | " + " | ".join("---" for _ in ACTIONS) + " |"]
    for t in ACTIONS:
        lines.append(f"| {t} | " + " | ".join(str(matrix[t][p]) for p in ACTIONS) + " |")
    return "\n".join(lines)


def main() -> int:
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    os.makedirs(MODEL_DIR, exist_ok=True)

    train = load_jsonl(os.path.join(DATA_DIR, "gate_train.jsonl"))
    dev = load_jsonl(os.path.join(DATA_DIR, "gate_dev.jsonl"))
    test = load_jsonl(os.path.join(DATA_DIR, "gate_test.jsonl"))
    pilot_path = os.path.join(DATA_DIR, "gate_test_pilot100.jsonl")
    pilot = load_jsonl(pilot_path) if os.path.exists(pilot_path) else []

    print(f"train={len(train)} dev={len(dev)} test={len(test)} pilot={len(pilot)}")

    X_train, y_train = xy(train)
    best = None
    # small, explicit grid; model selection only on dev
    for C in (0.1, 0.3, 1.0, 3.0):
        for cw in (None, "balanced"):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C, class_weight=cw,
                                           max_iter=5000)),
            ])
            pipe.fit(X_train, y_train)
            dev_pred = pipe.predict(*[xy(dev)[0]])
            from sklearn.metrics import f1_score
            score = f1_score(xy(dev)[1], dev_pred, average="macro",
                             labels=list(ACTIONS), zero_division=0)
            print(f"  C={C:<4} class_weight={str(cw):<9} dev macro-F1={score:.4f}")
            if best is None or score > best[0]:
                best = (score, C, cw, pipe)

    _, best_C, best_cw, model = best
    print(f"selected C={best_C} class_weight={best_cw} "
          f"(dev macro-F1={best[0]:.4f})")

    reports = [evaluate(model, dev, "dev"),
               evaluate(model, test, "test_counterfact")]
    if pilot:
        reports.append(evaluate(model, pilot, "test_pilot100_human_labels"))
    reports.append(dict(rule_metrics(dev), split="dev_rule_baseline"))
    reports.append(dict(rule_metrics(test), split="test_counterfact_rule_baseline"))
    if pilot:
        reports.append(dict(rule_metrics(pilot), split="test_pilot100_rule_baseline"))

    clf = model.named_steps["clf"]
    coefs = {
        cls: {name: round(float(w), 4)
              for name, w in zip(FEATURE_NAMES, clf.coef_[i])}
        for i, cls in enumerate(clf.classes_)
    }

    artifact = {
        "owner": "Tang I Choi (MC631517)",
        "work_package": "W3 -- conflict-gate classifier",
        "model": "StandardScaler + multinomial LogisticRegression",
        "hyperparameters": {"C": best_C, "class_weight": best_cw,
                            "max_iter": 5000, "selection": "dev macro-F1"},
        "actions": list(clf.classes_),
        "feature_names": FEATURE_NAMES,
        "coefficients": coefs,
        "train_size": len(train),
        "label_action_mapping": {
            "no-conflict + model knows target": "REJECT",
            "no-conflict + model cannot answer": "WRITE",
            "pre-training-conflict": "REJECT",
            "prior-edit-conflict": "ROUTE",
            "context-conflict": "ROUTE",
        },
        "reports": reports,
        "caveats": [
            "Trained on constructed CounterFact situations; labels follow the "
            "W1 annotation manual's decision rules, not human judgement.",
            "test_pilot100 keeps human labels and is out-of-distribution (its "
            "prompt format and relations differ from CounterFact).",
            "The gate cannot see the true value of a relation, only the model's "
            "answer and confidence, so pre-training conflicts are recoverable "
            "only through confidence.",
        ],
    }
    dump(model, os.path.join(MODEL_DIR, "gate_classifier.joblib"))
    with open(os.path.join(MODEL_DIR, "gate_classifier.json"), "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, ensure_ascii=False, indent=2)

    md = ["# W3 conflict-gate classifier -- metrics", "",
          f"Owner: Tang I Choi (MC631517). Train n={len(train)}, "
          f"dev n={len(dev)}, CounterFact test n={len(test)}, "
          f"pilot-100 test n={len(pilot)}.", "",
          f"Model: StandardScaler + multinomial LogisticRegression "
          f"(C={best_C}, class_weight={best_cw}), selected on dev macro-F1.", "",
          "| split | n | accuracy | macro-F1 | majority-class acc |",
          "| --- | --- | --- | --- | --- |"]
    for r in reports:
        md.append(f"| {r['split']} | {r['n']} | {r.get('accuracy')} | "
                  f"{r.get('macro_f1')} | {r.get('majority_class_acc', '-')} |")
    md.append("")
    for r in reports:
        md.append(md_table(f"{r['split']} confusion", r["confusion"]))
        md.append("")
    md.append("## Most influential features (logistic-regression weights)")
    md.append("")
    md.append("| feature | " + " | ".join(clf.classes_) + " |")
    md.append("| --- | " + " | ".join("---" for _ in clf.classes_) + " |")
    for name in FEATURE_NAMES:
        md.append(f"| {name} | " + " | ".join(
            f"{coefs[c][name]:+.3f}" for c in clf.classes_) + " |")
    with open(os.path.join(MODEL_DIR, "metrics.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    for r in reports:
        print(f"{r['split']:34s} n={r['n']:5d} acc={r.get('accuracy')} "
              f"macroF1={r.get('macro_f1')}")
    print(f"artifacts written to {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
