# CALM W1 - Fong Hio Lam (MC654953): Data Loaders and Evaluation Metrics

This work package delivers the W1 data pipeline of the CISC7021 Group 26 CALM
project, covering data loaders for CounterFact, zsRE and MQuAKE, together with
a metrics script for rewrite / paraphrase / locality / portability evaluation.

The package is self-contained with no third-party runtime dependency, aligns
with the field conventions and generation-based evaluation protocol of
EasyEdit (`easyeditor/dataset`), and supports small-scale smoke tests to
satisfy Gate 1: "all three data pipelines and the metrics script runnable".

## 1. Directory Structure

- `loaders/` - dataset loaders
  - `__init__.py` - loader registry + `build_loader()`
  - `base.py` - `EditCase` dataclass + `BaseDatasetLoader` base class
  - `counterfact.py` - CounterFact loader
  - `zsre.py` - zsRE loader
  - `mquake.py` - MQuAKE loader (compatible with the original MQuAKE format)
- `metrics.py` - rewrite_acc / paraphrase / locality / portability metrics
- `evaluate_pipeline.py` - CLI end-to-end pipeline (loader + metrics)
- `sample_data/` - example data
  - `counterfact_sample.json` - CounterFact sample (3 cases)
  - `zsre_sample.json` - zsRE sample (3 cases)
  - `mquake_sample.json` - MQuAKE sample (2 cases, one per input form)
- `real_eval_w1.py` - real-model evaluation driver (gpt2-xl)
- `w1_counterfact_gpt2xl.json` - real CounterFact results (gpt2-xl)
- `w1_zsre_gpt2xl.json` - real zsRE results (gpt2-xl)

## 2. Requirements

- Python 3.9+ (standard library only, no third-party dependency).
- To score with a real model, install model runtime dependencies such as
  `transformers` and `torch` (see Section 5.3).

## 3. Data Format (aligned with EasyEdit)

The unified intermediate structure is `EditCase` (`loaders/base.py`). Raw JSON
is a list of records whose fields follow EasyEdit conventions:

| Field | Meaning | Notes |
|---|---|---|
| `request.prompt` | Edit request prompt | e.g. `What is the official language of Spain?` |
| `request.subject` | Edit subject | e.g. `Spain` |
| `request.target_new` | Desired new answer | string or `{"str": ..., "id": ...}` |
| `request.target_true` | Original answer | same as above |
| `paraphrasess` | Paraphrase prompt list | backward-compatible with `paraphrase_prompts` |
| `locality_prompts.neighborhood_prompts` | Nearby unrelated prompts | used for locality evaluation |
| `locality_prompts.attribute_prompts` | Other-attribute unrelated prompts | used for locality evaluation |
| `portability_prompts` | Multi-hop derived questions (MQuAKE) | optional, see below |

**MQuAKE compatibility**: `MQuaKELoader` accepts two input forms:

1. EasyEdit style (explicit `portability_prompts`);
2. Original MQuAKE style (`requested_rewrite` + `questions` list); the loader
   automatically extracts `questions[].question` as `portability_prompts`.

Top-level wrapper objects such as `{"data": [...]}` or `{"instances": [...]}`
are unwrapped automatically.

## 4. Metric Definitions (evaluation protocol)

All metrics use normalized exact/containment matching (`metrics.normalize`:
lowercase, strip punctuation, collapse whitespace). A generation that contains
the target answer counts as a hit, matching the EasyEdit
exact-or-containment protocol.

| Metric | Definition | Judgement target |
|---|---|---|
| `rewrite_acc` | Edit reliability: generation on `request.prompt` | hits `target_new` |
| `paraphrase_acc` | Generalization: generations on all `paraphrasess` | hits `target_new` |
| `locality_neighborhood_acc` | Locality: generations on `neighborhood_prompts` | hits `target_true` |
| `locality_attribute_acc` | Locality: generations on `attribute_prompts` | hits `target_true` |
| `portability_acc` | Portability (multi-hop): generations on `portability_prompts` | hits `target_new` |

Formula: `accuracy = hits / generations`; a group without data returns `0.0`
(`evaluate_cases` also returns `n_cases` for reference).

## 5. Usage

### 5.1 Smoke test (no model required)

```bash
python evaluate_pipeline.py --mock --dataset counterfact --limit 3
python evaluate_pipeline.py --mock --dataset zsre --limit 3
python evaluate_pipeline.py --mock --dataset mquake --limit 2
```

`--mock` uses the built-in sample data with a rule-based generator; all three
datasets should produce plausible output. The rule-based generator is a
format/linkage self-check only and must not be used as official evaluation.

### 5.2 Score pre-generated predictions (no model required)

Produce predictions in the following JSON structure with any model:

```json
{
  "rewrite":      {"0": ["<generated text>"]},
  "paraphrase":   {"0": ["<paraphrase gen 1>", "<paraphrase gen 2>"]},
  "neighborhood": {"0": ["<neighborhood gen>"]},
  "attribute":    {"0": ["<attribute gen>"]},
  "portability":  {"0": ["<multi-hop gen>"]}
}
```

Then run:

```bash
python evaluate_pipeline.py --predictions-json preds.json --dataset counterfact
```

`--limit N` loads only the first N records for small-scale runs; absent groups
are counted as `0.0`.

### 5.3 Real-model evaluation driver

```bash
python real_eval_w1.py --dataset counterfact --model-key gpt2-xl --limit 100
python real_eval_w1.py --dataset zsre --model-key gpt2-xl --limit 100
```

`real_eval_w1.py` loads a Hugging Face GPT-2/GPT-J family model, generates
answers for every prompt group, and writes the metrics JSON used in this
folder. `generate_fn` is a pure callback that can be swapped for the edited
model output of CALMEditor, enabling unified pre-/post-edit scoring.

### 5.4 Use loaders directly

```python
from loaders import build_loader
loader = build_loader("counterfact", path="sample_data/counterfact_sample.json")
cases = loader.load(limit=10)
```

## 6. Integration with EasyEdit / CALMEditor

- **Data layer**: `EditCase` fields match the JSON conventions used by
  EasyEdit `easyeditor/dataset` (`rome_datasets.py` / `mend_datasets.py`).
  Per `w1_alignment_decision.md`, the team evaluates with `Llama-3-8B` and
  `GPT-J-6B` officially and uses `gpt2-xl/gpt2-small` for fast validation.
- **Metrics layer**: metrics use the native evaluate layer without touching
  EasyEdit internals; `metrics.py` can be reused by EasyEdit's `evaluate`
  logic (same normalized matching criterion).
- **Zero intrusion**: this package does not modify EasyEdit source, serving
  only as an external data pipeline providing a unified `EditCase` and scoring
  functions for the W2-W6 CALMEditor evaluation loop.

## 7. Gate 1 Acceptance Mapping

| Item | Location | Status |
|---|---|---|
| CounterFact loader runnable | `loaders/counterfact.py` + sample | OK |
| zsRE loader runnable | `loaders/zsre.py` + sample | OK |
| MQuAKE loader runnable | `loaders/mquake.py` + sample | OK |
| Metrics runnable (rewrite_acc/paraphrase/locality/portability) | `metrics.py` | OK |
| End-to-end smoke test | `evaluate_pipeline.py --mock/--predictions-json` | OK |
| Usage documentation | this README | OK |
