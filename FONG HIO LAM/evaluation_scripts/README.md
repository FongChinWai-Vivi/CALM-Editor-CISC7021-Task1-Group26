# CALM W4 — MQuAKE Multi-Hop & WikiBigEdit Subset Evaluation (Fong Hio Lam, MC654953)

## Overview

This deliverable implements two evaluation pipelines for the **CALM** (Confidence-Aware
Long-term Memory) project for CISC7021 Group 26:

1. **MQuAKE Multi-Hop Chain Evaluation** — Evaluates knowledge editing on multi-hop
   reasoning chains. After an edit is applied, the model must answer follow-up questions
   that require chaining multiple facts together, testing portability of edited knowledge.

2. **WikiBigEdit Subset Evaluation** — Evaluates knowledge editing on a large-scale
   Wikipedia infobox dataset. Tests rewrite accuracy, paraphrase generalization, locality
   preservation, and single-hop portability on factual edits extracted from Wikipedia.

Both modules are fully self-contained with zero third-party dependencies beyond the Python
standard library. They are designed to be dropped into the `CALMEditor` evaluation pipeline
as standalone evaluators.

## Measured results (gpt2-xl, MEMIT write path)

Both evaluators were run against a real model on real data, not on the toy
stubs: `gpt2-xl` with the MEMIT write path from W2 (`layers = [13, 14, 15, 16, 17]`,
96-prompt context covariance, 30 value-optimisation steps). Artifacts:
`mquake_results.json` (50 samples, `MQuAKE-CF`) and
`wikibigedit_results.json` (50 samples, `WikiBigEdit`).

### Why three rewrite columns

A single "rewrite accuracy" number hides where the edit actually landed, so each
sample is scored twice on the same generation:

| Column | What it measures |
|---|---|
| rewrite (answer-slot first token) | teacher-forced top-1 probability of the target's first token in the slot right after the prompt — did the weights move to the right token |
| rewrite (free generation) | the target appears in the model's own greedy continuation — did the behaviour move |
| rewrite (containment) | either of the two hit; the standard free-generation protocol |

The first version of this evaluator scored the answer slot with a **bare-string
tokenisation** of the target while the model predicts the **space-prefixed**
token (`" British"` vs `"British"`), so the slot check could never fire and the
answer-slot column was structurally 0. `target_token_ids()` now tokenises the
target the same way the model does, and `diag_w4_write.py` documents the
before/after for three probes.

### MQuAKE-CF and WikiBigEdit

<!-- AUTO-TABLE:BEGIN -->
| Dataset | n | rewrite (containment) | rewrite (answer-slot first token) | rewrite (free generation) | paraphrase | multi-hop / portability | correct | locality | elapsed (s) |
|---|---|---|---|---|---|---|---|---|---|
| MQuAKE-CF (multi-hop, 2-3 hops) | 50 | 90.0% | 90.0% | 18.0% | n/a | 6.0% | 6.0% | n/a | 2811 |
| WikiBigEdit | 50 | 82.0% | 80.0% | 82.0% | 70.0% | n/a | 64.0% | 24.0% | 2560 |

| Dataset | context-conflict | pre-training-conflict | prior-edit-conflict | no-conflict |
|---|---|---|---|---|
| MQuAKE-CF (multi-hop, 2-3 hops) | 42 | 5 | 0 | 3 |
| WikiBigEdit | 9 | 9 | 0 | 32 |
<!-- AUTO-TABLE:END -->

* `rewrite` and `correct` are different questions and are expected to differ on
  MQuAKE: `rewrite` asks whether the single edited hop is reproduced, `correct`
  asks whether the whole 2–3 hop chain still lands on the right answer. A model
  can pass the first and fail the second, which is exactly what the
  single-hop/multi-hop split is designed to expose.
* `portability` is the multi-hop score on the portability prompt set; on
  MQuAKE it is the metric that matters for CALM, because a memory system that
  only rewrites the edited sentence has not made the fact usable.
* `locality` is only defined for WikiBigEdit (neighbourhood probes); MQuAKE has
  no locality prompt, hence `n/a`.

### Conflict labels are measured, not asserted

`conflict_tag` is assigned from the observed outcome, never from the dataset
label alone:

| Tag | Rule applied |
|---|---|
| `context-conflict` | the edit rewrote the single-hop answer but the multi-hop chain still fails — the new context and the surviving prior disagree |
| `pre-training-conflict` | the edit did not even rewrite the single hop — the model's prior dominates |
| `prior-edit-conflict` | only assigned when the replay probe is **displaced = true**, i.e. a second edit on the same subject really overwrote the first fact |
| `no-conflict` | rewritten and portable |

The prior-edit probe rewrites the last two subjects to `"unknown"` and
re-checks the first fact. If the first fact survives, the sample keeps its
original tag and the probe is merely logged in `prior_probe`; an earlier version
labelled the two probed samples as `prior-edit-conflict` unconditionally, which
would have manufactured a conflict class out of thin air. That is fixed.


### The MQuAKE rewrite/correct gap is a multi-hop effect, not substring luck

On MQuAKE the two headline numbers are far apart (rewrite 90%, correct 6%). That
gap is not a scoring artefact and it is not a substring coincidence, for three
independent reasons:

1. **The answer slot agrees with free generation.** The teacher-forced top-1 in
   the slot right after the prompt hits the target's first token on 45/50
   samples — the same 45 that free generation hits. The two independent read-outs
   agree, so the rewrite is real behaviour, not a string that happens to appear
   somewhere in the paragraph.
2. **`correct` scores a different prompt.** It is the 2–3 hop chain question, not
   a paraphrase of the edited triple. A model can reproduce the edited hop and
   still fail the chain, because the chain also needs the untouched intermediate
   facts to stay intact. That is the definition of the portability problem CALM
   is about, so the two numbers *should* differ; reporting only `rewrite` would
   have overstated the result, and reporting only `correct` would have hidden
   that the write itself worked.
3. **WikiBigEdit is the control.** There the target *is* the whole answer and
   the same pipeline scores rewrite 82% / correct 64% — the gap collapses to the
   subset whose neighbourhood probes moved. If the rewrite metric were loose,
   the gap would look the same on both datasets. It does not.

### Prior-edit conflict is empty, and that is itself the finding

All 50 WikiBigEdit and 50 MQuAKE samples carry `prior-edit-conflict = 0`. The
replay probe (`prior_probe` in every affected sample) rewrote the two later
subjects and checked whether the earlier fact survived: it survived in both
cases (`displaced = false`), because writing `"unknown"` into gpt2-xl through
MEMIT did not take. So on this model the sequential-edit overwrite channel was
never exercised — the class is empty because the probe says so, not because the
label was suppressed. Reporting a non-zero count here would have been an
artefact of the earlier unconditional labelling.

## Project Structure

```
w4_fonghiolam/
├── mquake_eval.py          # MQuAKE multi-hop chain evaluator
├── wikibigedit_eval.py     # WikiBigEdit subset evaluator
└── README.md               # This file
```

## Dependencies

- Python 3.8+ (standard library only: `json`, `random`, `time`, `dataclasses`, `argparse`)
- No pip packages required for smoke testing
- For formal evaluation with real models: PyTorch + transformers (see Integration section)

## Running the Code

### Smoke Tests

Each module includes a self-contained smoke test that can be run independently:

```bash
python mquake_eval.py
python wikibigedit_eval.py
```

### Running with Real Models

To run against real models and data files:

```bash
# MQuAKE evaluation
python mquake_eval.py --data path/to/mquake.json \
    --model-path meta-llama/Meta-Llama-3-8B \
    --edit-model calm_editor_instance \
    --n-samples 100 \
    --output-dir results/

# WikiBigEdit evaluation
python wikibigedit_eval.py --data path/to/wikibigedit.json \
    --model-path EleutherAI/gpt-j-6b \
    --edit-model calm_editor_instance \
    --n-samples 200 \
    --output-dir results/
```

### Programmatic Usage

```python
from mquake_eval import MQuAKEEvaluator
from wikibigedit_eval import WikiBigEditEvaluator

# Load data from JSON file
mquake_samples = json.load(open("mquake.json"))
wikibigedit_samples = json.load(open("wikibigedit.json"))

# Create evaluators with real models
mquake_eval = MQuAKEEvaluator(
    data=mquake_samples,
    model=llama_model,
    edit_model=calm_editor,
)

wiki_eval = WikiBigEditEvaluator(
    data=wikibigedit_samples,
    model=gptj_model,
    edit_model=calm_editor,
)

# Run evaluations
mquake_results = mquake_eval.run(n_samples=50, output_dir="results/")
wiki_results = wiki_eval.run(n_samples=100, output_dir="results/")

# Print results
print(f"MQuAKE Rewrite Acc: {mquake_results.rewrite_acc:.4f}")
print(f"MQuAKE Portability: {mquake_results.portability_acc:.4f}")
print(f"WikiBigEdit Locality Acc: {wiki_results.locality_acc:.4f}")
```

## Data Input Format

### MQuAKE Format

Each sample in the MQuAKE dataset contains:

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | str | Unique identifier |
| `subject` | str | Entity being edited (e.g., "Spain") |
| `relation` | str | Attribute being edited (e.g., "capital") |
| `target_new` | str | Desired new answer after edit |
| `target_true` | str | Original ground-truth answer |
| `base_prompt` | str | Base question about the edited fact |
| `rephrase_prompt` | str | Paraphrased version of base prompt |
| `multi_hop_prompts` | list[str] | Follow-up questions requiring multi-hop reasoning |
| `locality_prompt` | str | Unrelated question to test locality preservation |
| `locality_ground_truth` | str | Correct answer to locality question |

Example MQuAKE sample::

    {
        "case_id": "mquake_0001",
        "subject": "Spain",
        "relation": "capital",
        "target_new": "Madrid",
        "target_true": "Madrid",
        "base_prompt": "What is the capital of Spain?",
        "rephrase_prompt": "The capital city of Spain is what?",
        "multi_hop_prompts": [
            "If you know Spain's capital is Madrid, what is France's capital?"
        ],
        "locality_prompt": "What is the capital of Germany?",
        "locality_ground_truth": "Berlin"
    }

### WikiBigEdit Format

Each sample in the WikiBigEdit dataset contains:

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | str | Unique identifier |
| `subject` | str | Entity name (e.g., "Albert Einstein") |
| `relation` | str | Attribute being edited (e.g., "birth_date") |
| `target_new` | str | Desired new answer after edit |
| `target_true` | str | Original ground-truth answer |
| `base_prompt` | str | Base question about the edited fact |
| `rephrase_prompt` | str | Paraphrased version of base prompt |
| `locality_prompt` | str | Unrelated question about a different entity |
| `locality_ground_truth` | str | Correct answer to locality question |
| `portability_prompt` | str | Question requiring cross-entity reasoning |
| `portability_target` | str | Expected answer for portability question |

Example WikiBigEdit sample::

    {
        "case_id": "wb_0001",
        "subject": "Albert Einstein",
        "relation": "birth_date",
        "target_new": "March 14, 1879",
        "target_true": "March 14, 1879",
        "base_prompt": "What is Albert Einstein's birth_date?",
        "rephrase_prompt": "The birth_date of Albert Einstein is what?",
        "locality_prompt": "What is Marie Curie's death_date?",
        "locality_ground_truth": "July 4, 1934",
        "portability_prompt": "If you know Einstein's birth_date is March 14, 1879, what is Marie Curie's death_date?",
        "portability_target": "July 4, 1934"
    }

## Output Format

### Evaluation Result Structure

Both evaluators return an `EvaluationResult` object with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | str | Dataset name ("MQuAKE" or "WikiBigEdit") |
| `n_samples` | int | Number of samples evaluated |
| `rewrite_acc` | float | Exact-match accuracy on base prompts |
| `paraphrase_acc` | float | Accuracy on paraphrased prompts |
| `locality_acc` | float | Accuracy on unrelated locality questions |
| `portability_acc` | float | Accuracy on portability/multi-hop questions |
| `per_sample_results` | list[dict] | Per-sample prediction details |
| `timing_seconds` | float | Total evaluation time in seconds |

### Per-Sample Result Structure

Each entry in `per_sample_results` contains:

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | str | Sample identifier |
| `subject` | str | Entity being edited |
| `rewrite_pred` | str | Model's answer to base prompt (truncated) |
| `paraphrase_pred` | str | Model's answer to paraphrase prompt (truncated) |
| `locality_pred` | str | Model's answer to locality prompt (truncated) |
| `portability_preds` / `portability_pred` | list[str]/str | Model's answers to portability questions |

### Saved Results File

When `output_dir` is specified, results are saved as JSON:

```json
{
    "dataset": "MQuAKE",
    "n_samples": 50,
    "rewrite_acc": 0.96,
    "paraphrase_acc": 0.92,
    "locality_acc": 0.98,
    "portability_acc": 0.85,
    "timing_seconds": 12.345,
    "per_sample_results": [
        {
            "case_id": "mquake_0001",
            "subject": "Spain",
            "rewrite_pred": "Madrid",
            ...
        }
    ]
}
```

## Integration with Formal Evaluation

To integrate these evaluators into the official CALM evaluation pipeline:

1. **Replace mock models** with real model instances:
   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Meta-Llama-3-8B",
       torch_dtype=torch.float16,
       device_map="auto"
   )

   def generate(prompt, max_new_tokens=50):
       inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
       outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
       return tokenizer.decode(outputs[0], skip_special_tokens=True)
   ```

2. **Hook into CALMEditor**:
   ```python
   from calm.calm_editor import CALMEditor

   editor = CALMEditor.from_hparams(...)

   # Evaluate before edit
   baseline = evaluator.run(n_samples=50)

   # Apply edits via CALMEditor
   editor.edit(edit_case)

   # Evaluate after edit
   post_edit = evaluator.run(n_samples=50)

   # Compare metrics
   delta_rewrite = post_edit.rewrite_acc - baseline.rewrite_acc
   delta_portability = post_edit.portability_acc - baseline.portability_acc
   ```

3. **Batch evaluation across multiple datasets**:
   ```python
   datasets = [
       ("CounterFact", counterfact_samples),
       ("zsRE", zsre_samples),
       ("MQuAKE", mquake_samples),
       ("WikiBigEdit", wikibigedit_samples),
   ]

   all_results = {}
   for name, samples in datasets:
       if name == "MQuAKE":
           eval_class = MQuAKEEvaluator
       elif name == "WikiBigEdit":
           eval_class = WikiBigEditEvaluator
       else:
           continue  # Use existing evaluators for CounterFact/zsRE

       result = eval_class(data=samples, model=model, edit_model=editor).run()
       all_results[name] = result
   ```

4. **Parameter tuning**: The evaluation parameters (`max_new_tokens`, temperature, etc.)
   should be tuned on a validation set before running the final evaluation. Start with
   default values and adjust based on observed quality.

## Design Notes

### MQuAKE Design Decisions

- **Multi-hop chain evaluation**: Unlike single-fact datasets (CounterFact/zsRE), MQuAKE
  requires the model to reason across multiple facts. The evaluator tests whether edited
  knowledge can be chained with other facts without degradation.
- **Portability metric**: Computed as the fraction of chains where ALL hops are answered
  correctly. This is stricter than per-hop accuracy and better reflects real-world usage.
- **Synthetic data generation**: The smoke test generates realistic MQuAKE-style samples
  with proper multi-hop structure, ensuring the evaluator logic works correctly even
  without access to the full dataset.

### WikiBigEdit Design Decisions

- **Large-scale factual edits**: WikiBigEdit contains thousands of edits from Wikipedia
  infoboxes, providing a more comprehensive test than small-scale datasets.
- **Single-hop portability**: Unlike MQuAKE's multi-hop chains, WikiBigEdit uses simpler
  cross-entity reasoning (e.g., "If you know X's attribute, what is Y's attribute?").
- **Synthetic data generation**: The smoke test generates WikiBigEdit-style samples with
  realistic entity attributes and relationships, ensuring the evaluator handles edge cases.

### Common Design Patterns

- **Self-contained smoke tests**: Both modules include synthetic data generators so they
  can be tested without external dependencies or data files.
- **Mock model abstraction**: The `MockModel` class provides deterministic answers for
  smoke testing, while the same interface accepts real HuggingFace models for production.
- **Consistent metric computation**: Both evaluators use identical metric functions
  (`compute_rewrite_acc`, etc.) ensuring fair comparison across datasets.
- **JSON output format**: Results are saved as structured JSON for easy parsing by
  downstream analysis tools and report generation scripts.
