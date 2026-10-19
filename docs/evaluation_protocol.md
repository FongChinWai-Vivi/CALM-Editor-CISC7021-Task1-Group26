---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_2598d2cec98511f18b7f525400393706
    ReservedCode1: CwilRLs5GcDVDxFg4Dk89JI4CjWd6yNPNq10ueYKHJ6qBXz7+1Te53nIV+zwmgxtPsyd3x8gIacISp139iOHyKSMayxkErQWccZ4s6siB8PBr2x46BiP2h6s2/yf6pYsfyZxwL4ZpEAZ026IJkk/imjHHMZFBCACqTcx7Z0SvRh9Ly5DHDx8lWPVhiw=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_2598d2cec98511f18b7f525400393706
    ReservedCode2: CwilRLs5GcDVDxFg4Dk89JI4CjWd6yNPNq10ueYKHJ6qBXz7+1Te53nIV+zwmgxtPsyd3x8gIacISp139iOHyKSMayxkErQWccZ4s6siB8PBr2x46BiP2h6s2/yf6pYsfyZxwL4ZpEAZ026IJkk/imjHHMZFBCACqTcx7Z0SvRh9Ly5DHDx8lWPVhiw=
---

# CALM Unified Evaluation Protocol (W1 Draft)

Status: Draft v0.1 · Owner: Fong Chin Wai (MC653748) · Last updated: 2026-10-16

This document defines the **unified evaluation protocol** for the CALM
(Conflict-Aware Lightweight Model editor) project. CALM is built on top of
[EasyEdit](https://github.com/zjunlp/EasyEdit), so the protocol reuses
EasyEdit's built-in metrics where possible and extends them with
CALM-specific signals (conflict gating and side-memory routing).

The goal is that **every algorithm we integrate (ROME, MEMIT, MEND, MALMEN,
SERAC, and CALM itself) is scored under the same dataset split, the same
metric definitions, and the same pass/fail criteria**, making W1..W6 results
comparable across work packages and team members.

---

## 1. Scope

The protocol covers evaluation of:

1. **Single edit quality** — how well a fact is rewritten after one edit.
2. **Generalization** — does the edit transfer to paraphrases of the prompt.
3. **Locality / specificity** — unrelated facts and neighborhood prompts
   should stay unchanged after the edit.
4. **Portability** — does the edit transfer to prompts about the same
   subject in new contexts.
5. **CALM-specific routing** — conflict-gate decisions and side-memory
   retrievals (extra section, not part of the EasyEdit baseline).

We intentionally do **not** define training convergence targets here; those
live in the W1 hypernetwork pipeline verification notes.

---

## 2. Evaluation Datasets

All experiments use **fixed, versioned splits** so that numbers are
reproducible. Unless a work package states otherwise, use:

| Split | Source | Format | Size | Use |
|---|---|---|---|---|
| `counterfact-train.json` | EasyEdit `data/counterfact-train.json` | CounterFact JSON | full | hypernetwork meta-training (MEND/MALMEN) |
| `counterfact-val.json` | EasyEdit `data/counterfact-val.json` | CounterFact JSON | full | early stopping / checkpoint selection |
| `zsre_mend_train.json` | EasyEdit `data/zsre_mend_train.json` | zsRE JSON | full | MEND-style training |
| `zsre_mend_eval.json` | EasyEdit `data/zsre_mend_eval.json` | zsRE JSON | full | MEND-style eval |
| `eval_ds.json` (CALM) | curated from CounterFact + zsRE | CALM JSON | ≥ 100 cases | official CALM benchmark (TBD in W3) |

For **W1 smoke tests**, a small deterministic subset (e.g. the first 6 cases
of CounterFact, or a hand-written 6-case ROME demo) is allowed; results must
be labeled `smoke=1` in the results report.

### 2.1 Required record fields

Every evaluation record **must** contain at least:

```json
{
  "prompt": "What university did Watts Humphrey attend?",
  "ground_truth": "Illinois Institute of Technology",
  "target_new": "University of Michigan",
  "subject": "Watts Humphrey",
  "rephrase_prompt": "Watts Humphrey was a student at which university?",
  "locality": {
    "nq": {"prompt": "nq question: ...", "ground_truth": "..."},
    "sentence": {"prompt": "...", "ground_truth": "..."}
  },
  "portability": {
    "p1": {"prompt": "...", "ground_truth": "..."}
  }
}
```

Missing locality / portability fields are reported as `N/A`, never silently
omitted.

---

## 3. Metric Definitions

All metrics follow EasyEdit's evaluator
(`easyeditor/evaluate/evaluate.py`) and are reported as **fractions in
[0, 1]** unless stated otherwise.

### 3.1 Core edit metrics

| Metric key | Definition | Scorer |
|---|---|---|
| `rewrite_acc` | Proportion of edit cases where the edited model completes `prompt` with `target_new` (exact match on generated tokens). | `test_prediction_acc` (LM) / `test_seq2seq_batch_prediction_acc` (T5) |
| `rephrase_acc` | Same as `rewrite_acc` but on `rephrase_prompt`. Measures paraphrase generalization. | same as above |
| `rewrite_ppl` | Perplexity of `target_new` given `prompt` after editing (lower is better). | `compute_rewrite_quality` |
| `token_accuracy` (a.k.a. `token_em`) | Token-level exact-match accuracy used internally by `test_prediction_acc`; reported alongside generation-level accuracy for debugging. | token-level EM scorer |

### 3.2 Locality metrics (specificity)

For each locality slice `key` in `record["locality"]`:

| Metric key | Definition |
|---|---|
| `locality.{key}_acc` | Fraction of neighborhood prompts whose output is **unchanged** by the edit: `mean(post_neighbor == pre_neighbor)` per case, then averaged over cases. |
| `locality.{key}_ppl` | Perplexity of the locality ground truth before/after the edit (reported as `nll/pre` and `nll/post` in training logs). |

Default slices: `nq` (natural-question neighborhood) and `sentence`
(paraphrase neighborhood), matching EasyEdit's `CounterFactDataset`.

### 3.3 Portability metrics

| Metric key | Definition |
|---|---|
| `portability.{key}_acc` | Fraction of portability prompts (same subject, different relation/context) that now yield `target_new` after the edit. |

### 3.4 CALM-specific metrics (added, optional in W1)

| Metric key | Definition | Status |
|---|---|---|
| `gate_accuracy` | Conflict gate decision accuracy on a labeled conflict/neutral set. | W2 |
| `memory_hit_rate` | Fraction of edit queries that retrieve the intended stored fact from side memory. | W2 |
| `route_ratio` | Fraction of edits routed to `hypernetwork` vs `side_memory` vs `gate_reject`. | W2 |

CALM reports the EasyEdit baseline metrics **plus** these, never instead of
them.

---

## 4. Evaluation Script Entry Points

### 4.1 EasyEdit built-in entry

Pre-trained / edited-model evaluation uses EasyEdit's built-in flow:

```bash
# From the EasyEdit repo root (with .venv active)
python edit.py            # uncomment the test_<ALG>() of interest
```

The edit functions in `edit.py` (e.g. `test_ROME()`, `test_MEND()`) call
`BaseEditor.from_hparams(...)` then `editor.edit(...)` and print a metrics
dict containing `rewrite_acc`, `rephrase_acc`, `locality` and `portability`.

### 4.2 Recommended script layout (CALM convention)

Every algorithm work package SHOULD provide a reproducible script:

```
EasyEdit/examples/run_<ALG>_eval.py
```

Contract for such scripts:

1. Read hyper-parameters from `hparams/<ALG>/<model>.yaml`.
2. Load the evaluation split from `data/` (see Section 2).
3. Build the editor, apply edits, then evaluate using
   `easyeditor.evaluate.evaluate` utilities.
4. Print a single machine-readable JSON block at the end:

```json
{
  "alg": "ROME", "model": "gpt2-xl", "split": "counterfact-val",
  "n": 100,
  "rewrite_acc": 0.93, "rephrase_acc": 0.86,
  "locality": {"nq_acc": 0.98, "sentence_acc": 0.97},
  "portability": {"p1_acc": 0.55},
  "smoke": 0, "gpu": "RTX4070-8GB", "elapsed_s": 1234
}
```

5. Save the JSON to `results/eval/<ALG>_<model>_<date>.json`.

### 4.3 Training-time metrics (MEND / MALMEN)

Hypernetwork training logs use EasyEdit's `EditTrainer` info dict:

| Key | Meaning |
|---|---|
| `loss/edit` | NLL of `target_new` after edit (should decrease). |
| `loss/loc` | KL locality loss vs. base model (should stay low). |
| `edit/acc` | token-level accuracy on the edit itself. |
| `acc/pre`, `acc/post` | locality accuracy before / after edit. |
| `nll/pre`, `nll/post` | locality NLL before / after edit. |

---

## 5. Pass / Fail Criteria

These are the **W1 baseline gates** (may be tightened in later work
packages; do not loosen without team agreement).

| Metric | Gate | Applies to |
|---|---|---|
| `rewrite_acc` | ≥ 0.80 (smoke ≥ 0.50) | all editing methods |
| `rephrase_acc` | ≥ 0.70 (smoke ≥ 0.50) | all editing methods |
| `locality.nq_acc` | ≥ 0.90 | all editing methods |
| `locality.sentence_acc` | ≥ 0.90 | all editing methods |
| `portability.p1_acc` | ≥ 0.30 (informational in W1) | methods with portability records |
| Training runs (MEND/MALMEN) | `loss/edit` decreases monotonically over the run; `edit/acc` ≥ 0.80 at the end | hypernetwork training pipelines |

A method **fails** if it misses a gate on the official (non-smoke) split.
Smoke runs (`smoke=1`) may miss stricter gates but must still satisfy the
smoke thresholds.

### 5.1 Reproducibility requirements

- Fixed seed: record `seed` (default 42).
- Fixed hardware: report GPU model and VRAM in every results JSON.
- Fixed data: report split name and `n`.
- Every result JSON is committed to `results/eval/` (no ad-hoc files).

---

## 6. Reporting Template

Each work package that touches editing MUST append a section to the shared
`docs/results.md` (or its own `docs/W<k>_results.md` linked from README) with:

1. Algorithm + model + hparams file path.
2. Dataset split + sample count.
3. The metrics JSON block (Section 4.2).
4. Pass/fail verdict per Section 5.
5. Any deviation from this protocol (with reason).

---

## 7. Open Items (W1)

- [ ] Fix the official CALM benchmark split (`eval_ds.json`, ≥100 cases) — W3.
- [ ] Decide whether `portability` gates become mandatory — W4.
- [ ] Define conflict-gate labeled evaluation set for `gate_accuracy` — W2.
- [ ] Standardize GPU / seed reporting fields across team scripts — W2.

---

## Appendix A — References

- EasyEdit evaluator: `easyeditor/evaluate/evaluate.py`, `metric_meta.py`.
- ROME demo (W1): `EasyEdit/examples/w1_rome_demo.py`.
- CALM interface draft: `calm/calm_editor.py`.
- EasyEdit datasets: `easyeditor/dataset/counterfact.py`, `easyeditor/dataset/zsre.py`.
*（内容由AI生成，仅供参考）*
