---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 3948123ca1e8ab71efaa265617bdd19e_d349baabcaca11f196f9525400248c00
    ReservedCode1: 9EMyjCU3yfK6zmRV/m+NHr1cuv5MiGw89eXJOI9WMehIpyu2/dtqALmNdy6eHzLyoHgUhtD2YQuQ2v7VLgiVJDpX5XvzD7l7WnxvnE9UC+B7JiqWZ+yv7QJopExfj2BwtLaI8C9VLsU2fSYWyLs7wFLG0ZCTnW1mkVN4YnNwXgfAGZoHCRMo9uyq2fQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 3948123ca1e8ab71efaa265617bdd19e_d349baabcaca11f196f9525400248c00
    ReservedCode2: 9EMyjCU3yfK6zmRV/m+NHr1cuv5MiGw89eXJOI9WMehIpyu2/dtqALmNdy6eHzLyoHgUhtD2YQuQ2v7VLgiVJDpX5XvzD7l7WnxvnE9UC+B7JiqWZ+yv7QJopExfj2BwtLaI8C9VLsU2fSYWyLs7wFLG0ZCTnW1mkVN4YnNwXgfAGZoHCRMo9uyq2fQ=
---

# CALM W1 - Fong Hio Lam (MC654953): Data Loaders and Evaluation Metrics

This work package delivers the **data pipeline** assigned to Fong Hio Lam in
stage W1 of the CISC7021 Group26 CALM project: data loaders for the
CounterFact, zsRE and MQuAKE datasets, together with a suite of metric
scripts (rewrite_acc / paraphrase / locality / portability).
The code is self-contained, has no third-party runtime dependencies, and
follows the field conventions and generative evaluation protocol of EasyEdit
`easyeditor/dataset`. It supports small-scale smoke tests and satisfies the
Gate 1 acceptance requirement that "the three data pipelines and the metric
scripts are all runnable".

---

## 1. Directory Structure

```
w1_fonghiolam/
├── README.md                    # This documentation
├── loaders/
│   ├── __init__.py              # Loader registry + build_loader()
│   ├── base.py                  # EditCase dataclass + BaseDatasetLoader base class
│   ├── counterfact.py           # CounterFactLoader
│   ├── zsre.py                  # ZsreLoader
│   └── mquake.py                # MQuaKELoader (compatible with the original MQuAKE format)
├── metrics.py                   # Metric implementations: rewrite_acc/paraphrase/locality/portability
├── evaluate_pipeline.py         # CLI end-to-end pipeline (loader + metrics)
└── sample_data/
    ├── counterfact_sample.json  # CounterFact sample data (3 records)
    ├── zsre_sample.json         # zsRE sample data (3 records)
    └── mquake_sample.json       # MQuAKE sample data (2 records, one per input form)
```

## 2. Environment Requirements

- Python 3.9+ (standard library only, no third-party dependencies).
- To score with a real model, install the model runtime dependencies
  yourself (e.g. `transformers`, `torch`); see the example in Section 6.

## 3. Data Format (Aligned with EasyEdit)

The unified intermediate structure is `EditCase` (`loaders/base.py`). The
raw JSON is a list of records, and each record follows the EasyEdit
conventions:

| Field | Meaning | Notes |
|---|---|---|
| `request.prompt` | Edit request prompt | e.g. `What is the official language of Spain?` |
| `request.subject` | Edit subject | e.g. `Spain` |
| `request.target_new` | New answer after the edit | A string or `{"str": ..., "id": ...}` |
| `request.target_true` | Original answer before the edit | Same as above |
| `paraphrasess` | List of paraphrase prompts | Legacy field name `paraphrase_prompts` is also accepted |
| `locality_prompts.neighborhood_prompts` | Unrelated neighborhood prompts | Used for locality evaluation |
| `locality_prompts.attribute_prompts` | Unrelated attribute prompts | Used for locality evaluation |
| `portability_prompts` | Multi-hop derived questions (MQuAKE) | Optional; see below |

**MQuAKE compatibility**: `MQuaKELoader` supports both input forms:

1. EasyEdit style (with explicit `portability_prompts`);
2. Original MQuAKE style (`requested_rewrite` + `questions` list), where the
   loader automatically extracts `questions[].question` as
   `portability_prompts`.

Top-level wrapper objects such as `{"data": [...]}` / `{"instances": [...]}`
are also unwrapped automatically.

## 4. Metric Definitions (Evaluation Protocol)

All metrics use **normalized exact/containment matching**
(`metrics.normalize`: lowercase, strip punctuation, collapse whitespace); a
generated text counts as a hit if it contains the target answer. This is
consistent with the EasyEdit generative evaluation
(exact-or-containment) protocol.

| Metric | Definition | Target |
|---|---|---|
| `rewrite_acc` | Edit reliability: generation for the edit prompt `request.prompt` | hits `target_new` |
| `paraphrase_acc` | Generalization: generation for all `paraphrasess` | hits `target_new` |
| `locality_neighborhood_acc` | Locality: generation for `neighborhood_prompts` | hits `target_true` |
| `locality_attribute_acc` | Locality: generation for `attribute_prompts` | hits `target_true` |
| `portability_acc` | Portability (multi-hop): generation for `portability_prompts` | hits `target_new` |

Formula: `accuracy = hits / generated`; if a group has no data, the metric
returns `0.0` (`evaluate_cases` also returns `n_cases` for reference).

## 5. How to Run

### 5.1 Smoke test (no model required; verifies the pipeline runs)

```bash
python evaluate_pipeline.py --dataset counterfact --mock --limit 3
python evaluate_pipeline.py --dataset zsre      --mock --limit 3
python evaluate_pipeline.py --dataset mquake    --mock --limit 3
```

With `--mock`, the built-in sample data (`sample_data/`) and a rule-based
generator are used; all three datasets should output something like:

```json
{
  "rewrite_acc": 1.0,
  "paraphrase_acc": 1.0,
  "locality_neighborhood_acc": 1.0,
  "locality_attribute_acc": 1.0,
  "portability_acc": 1.0,
  "n_cases": 3
}
```

> Note: the rule-based generator of `--mock` is only a format/link
> self-check and **must not** be used as official evaluation results.

### 5.2 Scoring pre-generated predictions (no model required)

First produce predictions with a model (or any generator) in the following
JSON structure:

```json
{
  "rewrite":      {"0": ["<generated text>"]},
  "paraphrase":   {"0": ["<paraphrase 1 generation>", "<paraphrase 2 generation>"]},
  "neighborhood": {"0": ["<neighborhood prompt generation>"]},
  "attribute":    {"0": ["<attribute prompt generation>"]},
  "portability":  {"0": ["<multi-hop question generation>"]}
}
```

Then:

```bash
python evaluate_pipeline.py --dataset counterfact \
    --data-path data/counterfact.json --predictions-json preds.json
```

`--limit N` loads only the first N records for a small-scale smoke run;
groups that are absent count as `0.0` in the results.

### 5.3 Integrating a real model in Python

```python
from evaluate_pipeline import run_with_generator

def generate(prompt: str) -> str:
    # Example: load Llama-3-8B / GPT-J-6B with transformers (W1 locked model list)
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # out = model.generate(**inputs, max_new_tokens=32)
    # return tokenizer.decode(out[0], skip_special_tokens=True)
    ...

result = run_with_generator(
    dataset_name="counterfact",
    data_path=r"C:\path\to\counterfact.json",
    generate_fn=generate,
    limit=100,
)
print(result)
```

`generate_fn` is a plain functional callback; it can be replaced directly
with the generation interface of the edited CALMEditor model to score
pre-edit and post-edit outputs uniformly.

### 5.4 Using the loaders directly

```python
from loaders import build_loader

loader = build_loader("counterfact", data_path="counterfact.json", limit=100)
cases = loader.load()          # list[EditCase]
print(loader.size, cases[0].target_new)
```

## 6. Integration Notes with EasyEdit / CALMEditor

- **Data layer**: the `EditCase` fields match the JSON conventions used by
  EasyEdit `easyeditor/dataset` (`rome_datasets.py` / `mend_datasets.py`,
  etc.): `request` / `paraphrasess` / `locality_prompts` /
  `portability_prompts`. Following `w1_alignment_decision.md`, the team uses
  `Llama-3-8B` and `GPT-J-6B` for official evaluation; `gpt2-xl/gpt2-small`
  are only for quick verification.
- **Metric layer**: the metrics use the native evaluate layer and do not
  intrude into EasyEdit components; `metrics.py` can be reused directly by
  EasyEdit's `evaluate` logic (same normalized matching criterion).
- **Zero intrusion**: this package does not modify EasyEdit source code; it
  only provides a unified `EditCase` and scoring functions as an external
  data pipeline for the CALMEditor evaluation loop in W2-W6.

## 7. Gate 1 Acceptance Checklist

| Acceptance item | Delivery location | Status |
|---|---|---|
| CounterFact data loader runs | `loaders/counterfact.py` + sample | ✅ |
| zsRE data loader runs | `loaders/zsre.py` + sample | ✅ |
| MQuAKE data loader runs | `loaders/mquake.py` + sample | ✅ |
| Metric scripts run (rewrite_acc/paraphrase/locality/portability) | `metrics.py` | ✅ |
| End-to-end smoke test (small scale) | `evaluate_pipeline.py --mock/--predictions-json` | ✅ |
| Usage instructions (how to run, input/output formats) | This README | ✅ |
*(Content generated by AI, for reference only)*
