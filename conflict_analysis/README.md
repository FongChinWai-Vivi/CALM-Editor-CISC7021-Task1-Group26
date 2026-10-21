---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 3948123ca1e8ab71efaa265617bdd19e_33ef3d47cd4b11f1898e525400248c00
    ReservedCode1: nvLZc+xh80MS8y4ub5UfnmgKIUkfU1ibKI2viB2fgWvpOjYv8OyTE3l7SmFKWvPp0IqQiJH26U5FnN8HCMR1iVGzt1f0R0fDUW7ErlsbIqsxFAMBG5PcMhE0el4oSH9wfDOd6+9v2IvZzU5IwC2e62knbjFNYWE2B/kRxbIpJh63Szc56U0dst5jszM=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 3948123ca1e8ab71efaa265617bdd19e_33ef3d47cd4b11f1898e525400248c00
    ReservedCode2: nvLZc+xh80MS8y4ub5UfnmgKIUkfU1ibKI2viB2fgWvpOjYv8OyTE3l7SmFKWvPp0IqQiJH26U5FnN8HCMR1iVGzt1f0R0fDUW7ErlsbIqsxFAMBG5PcMhE0el4oSH9wfDOd6+9v2IvZzU5IwC2e62knbjFNYWE2B/kRxbIpJh63Szc56U0dst5jszM=
---

# CALM W5 -- Conflict-Type Error Analysis

**Author:** Fong Hio Lam (MC654953)
**Group:** CISC7021 Group26 -- CALM
**Deliverable:** W5 work package

This work package provides a self-contained Python pipeline that analyses the
W4 evaluation outputs (MQuAKE / WikiBigEdit) of the edited CALM model and
quantifies **where the model fails** in terms of the four conflict types
defined in the W0 taxonomy and used by the W3 `ConflictGate` design.

The output is a per-conflict-type error analysis (report) plus two figures:
a grouped bar chart of error rates and a pie chart of the error distribution.

---

## 1. Conflict taxonomy

Every evaluation sample is labelled with exactly one of the four tags:

| Tag                     | Meaning                                                              |
|-------------------------|----------------------------------------------------------------------|
| `no-conflict`           | The edit does not collide with any known knowledge.                  |
| `pre-training-conflict` | The edit contradicts knowledge stored in the model's pretraining.    |
| `prior-edit-conflict`   | The edit contradicts an earlier edit applied in the same editing run.|
| `context-conflict`      | The edit is inconsistent with the context / supporting facts.        |

These tags mirror the conflict signals handled by `ConflictGate.decide`
(`Write` / `Route` / `Reject`) from W3.

---

## 2. Requirements

- Python >= 3.9
- `numpy`
- `matplotlib` (used only for chart rendering; the pipeline prints a warning
  and skips charts if it is missing, but the report is still produced)

Install dependencies:

```bash
pip install numpy matplotlib
```

No other external packages are required; the script is self-contained and
does **not** depend on the CALM editing framework.

---

## 3. Files

| File                  | Purpose                                                       |
|-----------------------|---------------------------------------------------------------|
| `conflict_analysis.py`| Main script: loading, validation, analysis, plotting, smoke test. |
| `README.md`           | This document.                                                |

---

## 4. Usage

### 4.1 Analyse a real evaluation results file

```bash
python conflict_analysis.py --results <path-to-results.json> --output-dir <out-dir>
```

Example:

```bash
python conflict_analysis.py --results results/mquake_eval_results.json --output-dir ./analysis
```

What the script does:

1. Loads and validates the JSON file (structure described in Section 5).
2. Counts samples / correct predictions / errors per conflict tag.
3. Computes the error rate and the error share for every tag.
4. Writes `conflict_analysis_report.txt`.
5. Renders `conflict_error_rates.png` and `conflict_error_distribution.png`
   (skipped if matplotlib is unavailable).

If `--output-dir` is omitted the current working directory is used. The
directory is created automatically when it does not exist.

### 4.2 Smoke test

```bash
python conflict_analysis.py --smoke-test --output-dir <out-dir>
```

The smoke test builds 12 synthetic samples covering all four conflict tags
(3 samples per tag, 6 correct / 6 incorrect), runs the full pipeline and
writes:

- `smoke_test_report.txt`
- `smoke_conflict_error_rates.png`
- `smoke_conflict_error_distribution.png`

Expected output on a healthy installation ends with a line similar to:

```
SMOKE TEST PASSED: 12 synthetic samples, 6 errors; report and 2 figures written to ...
```

---

## 5. Data input format

The script consumes the JSON structure produced by the W4 evaluation scripts
(`mquake_eval.py` / `wikibigedit_eval.py`). The file must be a JSON object:

```json
{
  "metadata": {
    "task": "mquake",
    "model": "Llama-3-8B",
    "run_id": "w4-final",
    "note": "optional free-form metadata"
  },
  "per_sample_results": [
    {
      "sample_id": 0,
      "conflict_tag": "pre-training-conflict",
      "prediction": "answer string or null",
      "expected": "gold answer string",
      "correct": false
    },
    {
      "sample_id": 1,
      "conflict_tag": "no-conflict",
      "prediction": "Paris",
      "expected": "Paris",
      "correct": true
    }
  ]
}
```

Field requirements:

| Field              | Type    | Required | Notes                                        |
|--------------------|---------|----------|----------------------------------------------|
| `metadata`         | object  | optional | free-form; `task` / `model` are shown in report |
| `per_sample_results`| array  | **yes**  | must be non-empty                            |
| `sample_id`        | int     | **yes**  | per-sample identifier                        |
| `conflict_tag`     | string  | **yes**  | one of the four tags in Section 1            |
| `prediction`       | string / null | optional | model output for reference             |
| `expected`         | string  | optional | gold answer for reference                    |
| `correct`          | bool    | **yes**  | ground-truth correctness flag                |

Validation failures (missing keys, unknown tags, non-boolean `correct`,
empty list, unparsable JSON) abort with an explicit `ConflictAnalysisError`
message. If a results file contains samples without a conflict label, label
them first (e.g. via the W0 taxonomy rules or the W3 `ConflictGate`
decision trace) before running this script.

---

## 6. Outputs

### 6.1 `conflict_analysis_report.txt`

A plain-text table with one row per conflict tag:

```
conflict tag              samples  correct   errors  error rate   err share
---------------------------------------------------------------------------
no-conflict                     3        3        0        0.0%        0.0%
...
TOTAL                          12        6        6       50.0%      100.0%
```

- **error rate** = errors / samples for that tag.
- **err share**  = errors of that tag / total errors (the error distribution).

### 6.2 `conflict_error_rates.png`

Grouped bar chart: x-axis = four conflict tags, y-axis = error rate (%), with
the absolute error count printed above each bar.

### 6.3 `conflict_error_distribution.png`

Pie chart: share of all errors contributed by each conflict tag. If there are
no errors, the figure shows a "No errors found" placeholder instead.

---

## 7. Integrating with the official evaluation results

To run this analysis on the official W4 outputs:

1. Run `mquake_eval.py` and `wikibigedit_eval.py` from the W4 work package
   (they write their results to the configured `results` directory).
2. Ensure every sample in `per_sample_results` carries a `conflict_tag`
   field. If the W4 scripts do not yet emit the tag, extend them to attach
   the label at scoring time (the label comes from the W0 taxonomy rules,
   e.g. compare the edit target against pretraining facts, the edit history,
   and the context facts).
3. Run this script once per results file:
   ```bash
   python conflict_analysis.py --results results/mquake_eval_results.json --output-dir ./analysis
   python conflict_analysis.py --results results/wikibigedit_eval_results.json --output-dir ./analysis
   ```
4. Use the generated charts and report in the team report / presentation to
   discuss which conflict type drives the residual errors of the edited model
   and where the `ConflictGate` routing deserves attention.

---

## 8. Notes and limitations

- The script performs **error accounting only**; it does not re-run model
  inference.
- Correctness is taken from the `correct` field supplied by the evaluation;
  the script does not re-derive it from `prediction` vs `expected`.
- Chart rendering is headless (`matplotlib.use("Agg")`), safe for
  server / CI execution.
- All deliverable content is in English by team convention.
