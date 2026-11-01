# W3 -- Conflict-gate training data + gate classifier (Tang I Choi, MC631517)

This work package builds the *supervised* conflict gate that the W4 ablations
compare against the rule-based gate. It has two halves:

1. **Training data** (`build_gate_training_data.py`) -- turns the W1 four-label
   annotation scheme into a supervised dataset on top of CounterFact, with a
   cached model-answer signal so that training does not re-run the LM.
2. **Classifier** (`train_gate_classifier.py`, `gate_features.py`) -- a
   scaled multinomial logistic regression that maps an edit request to one of
   CALM's three actions: `WRITE` / `ROUTE` / `REJECT`.

Nothing in this folder writes outside `TANG I CHOI/w3_conflict_gate_training_and_classifier/`.

## 1. Label scheme (from W1)

Four labels, decided with the priority `context > prior-edit > pre-training`
(see the W1 manual in `conflict annotation manual + 100 pilot annotations/`):

| label | meaning | mapped action |
|---|---|---|
| `no-conflict` | the new fact agrees with the model; nothing to fix | `REJECT` if the model already answers the new target, else `WRITE` |
| `pre-training-conflict` | the new fact contradicts what the model memorised | `REJECT` (the parametric channel is what is broken) |
| `prior-edit-conflict` | the new fact contradicts an edit already applied earlier in this stream | `ROUTE` (side memory holds the authoritative value) |
| `context-conflict` | the fact at inference time contradicts retrieved context | `ROUTE` (context wins at inference) |

`no-conflict` is the only label that depends on the model's own state, hence
the cached answer signal below.

## 2. Dataset construction

`build_gate_training_data.py` performs, per CounterFact record:

1. assigns a situation (`true_fact_unknown`, `false_fact`, `context_override`,
   `context_supported`, `repeat_edit`) following the W1 manual;
2. derives the four-label annotation `label4` from that situation;
3. measures the *current answer* and mean token confidence of `gpt2-xl` for
   the prompt (`model_answer_cache.json`, so the LM is queried once);
4. derives the gold action with the table above;
5. writes `prompt / subject / relation / target_new / current_answer /
   current_confidence / prior_edits / retrieval_context / label4 / action /
   rationale`.

Splits (`seed=42`, grouped by subject so no subject crosses a boundary):

| split | n | WRITE | ROUTE | REJECT |
|---|---|---|---|---|
| train | 2584 | 1101 | 1117 | 366 |
| dev | 645 | 277 | 283 | 85 |
| test (CounterFact) | 835 | 350 | 360 | 125 |
| test (human pilot-100) | 100 | 38 | 35 | 27 |

Full distribution per label and per split: `gate_datasets/dataset_card.json`.
The pilot split keeps Tang's W1 human labels and is **never** used for training
or model selection; it is the honest out-of-distribution test set.

## 3. Features

`gate_features.py` (shared by training and all W4 arms) emits 19 numeric
features in three blocks:

- **answer signal** (6): exact match, substring match, target tokens inside the
  answer, confidence, answer length, target length;
- **prior-edit signal** (5): number of prior edits, same subject, same
  subject+relation, target match, target Jaccard;
- **retrieval-context signal** (4): has context, context size, target overlap,
  prior-edit overlap, subject mention;
- plus subject/relation shape features.

## 4. Classifier

`StandardScaler + multinomial LogisticRegression(C=1.0, class_weight="balanced")`,
selected on dev macro-F1. Full numbers in `gate_model/metrics.md`.

| split | accuracy | macro-F1 | rule-gate baseline accuracy |
|---|---|---|---|
| dev | 0.893 | 0.856 | 0.845 |
| test (CounterFact, 835) | 0.877 | 0.843 | 0.840 |
| test (human pilot-100) | 0.650 | 0.645 | 0.670 |

The learned gate beats the rules on the constructed CounterFact split and
**loses to them on the human pilot labels**, which is the honest headline of
this work package: the supervised gate mostly learns the dataset's construction
rules, and it does not generalise to human judgement at n=100.

Most influential weights (see `metrics.md`): `n_prior_edits` /
`prior_same_subject` / `has_context` push towards `ROUTE`, high confidence and
longer targets push towards `REJECT`, and `context_target_overlap` is the only
feature separating `context-conflict` (`ROUTE`) from `WRITE`.

## 5. Offline vs. CALM-stream gap (`evaluate_gate_online_gap.py`)

`CALMEditor.edit_requests` calls the gate with `retrieval_context=None`, so the
four context features are always zero inside the stream, and at the start of a
stream the prior-edit history is empty. Re-scoring both gates under those
conditions on the same 835 records (`results/gate_online_gap.json`):

| gate | offline (context + history) | in-stream (real history, no context) | cold start (no history, no context) |
|---|---|---|---|
| rule gate | 0.840 | 0.613 (W4 measurement) | 0.408 |
| learned gate | 0.877 | 0.424 (W4 measurement) | 0.340 |

The learned gate loses 0.45 accuracy between the offline data sheet and the
stream (exact McNemar, learned vs rule: p=0.0008 in the cold-start condition).
This is **the** finding that motivates the W4 analysis: the gate is trained on
features that the stream never provides.

## 6. Reproduce

```bash
cd "TANG I CHOI/w3_conflict_gate_training_and_classifier"
python build_gate_training_data.py     # needs EasyEdit/hugging_cache/gpt2-xl
python train_gate_classifier.py        # writes gate_model/
python evaluate_gate_online_gap.py     # writes results/gate_online_gap.json
```

Or run everything (W3 to W5) with
`bash "TANG I CHOI/w5_report_and_reproducibility/reproduce_all.sh"`.
