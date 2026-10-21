---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 3948123ca1e8ab71efaa265617bdd19e_445409d4cd4911f1b307525400393706
    ReservedCode1: JmXVlS4oZEDzikc6Hhf6G7A6dLKXM8MqwO2F2qZlCT3ik9AQRPYrJwNS7xerFdcPkKC9IRhkMLj3ySeprfSlGbgTLCVRR9mhP0RZuYwOGRw2P3wzfcIF7DXrkkwY5vly/4yedusW4+0kiUW0n6sLOe5u097QHKcf8ntYxv5ryyrS8fqyhxnWn1+IPHs=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 3948123ca1e8ab71efaa265617bdd19e_445409d4cd4911f1b307525400393706
    ReservedCode2: JmXVlS4oZEDzikc6Hhf6G7A6dLKXM8MqwO2F2qZlCT3ik9AQRPYrJwNS7xerFdcPkKC9IRhkMLj3ySeprfSlGbgTLCVRR9mhP0RZuYwOGRw2P3wzfcIF7DXrkkwY5vly/4yedusW4+0kiUW0n6sLOe5u097QHKcf8ntYxv5ryyrS8fqyhxnWn1+IPHs=
---

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
