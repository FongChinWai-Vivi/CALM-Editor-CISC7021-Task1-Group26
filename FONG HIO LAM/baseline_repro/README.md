---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 3948123ca1e8ab71efaa265617bdd19e_0c2d22c3cd4711f1898e525400248c00
    ReservedCode1: sHRqgp0v78/NjqN1kAaamOzBzIsIktmiNzOBWv1SOQfA1d3s7tbYc7yA+Gf/6qLUDzW4sgAczoH0Q0wtpKIytNhZ7Yuzw8kwA06QbRPqo2Db6jo2rX+XC6Xg9GYhA9xdENOA1M55M9jUfHW0Pz2fpZ7K3G2YEycVWy7PSS8YP04892wyOMMHQCCP87M=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 3948123ca1e8ab71efaa265617bdd19e_0c2d22c3cd4711f1898e525400248c00
    ReservedCode2: sHRqgp0v78/NjqN1kAaamOzBzIsIktmiNzOBWv1SOQfA1d3s7tbYc7yA+Gf/6qLUDzW4sgAczoH0Q0wtpKIytNhZ7Yuzw8kwA06QbRPqo2Db6jo2rX+XC6Xg9GYhA9xdENOA1M55M9jUfHW0Pz2fpZ7K3G2YEycVWy7PSS8YP04892wyOMMHQCCP87M=
---

# W2 Reproduction Package — MEMIT & AlphaEdit Baselines

**Author:** Fong Hio Lam (MC654953) · CALM Project, Group 26 · CISC7021

This package reproduces the two batch knowledge-editing baselines required
by Work Package 2 (W2) of the CALM project:

| Baseline | Script | Mechanism |
| --- | --- | --- |
| **MEMIT** | `memit_repro.py` | Batch residual editing: compute desired residuals at the edit layer, solve a least-squares weight update over the context covariance, apply all edits in one step. |
| **AlphaEdit** | `alphaedit_repro.py` | Null-space constrained extension of MEMIT: the MEMIT delta is projected onto the orthogonal complement of the top feature directions of the model, protecting unrelated knowledge (locality) with a hard constraint. |

Both scripts follow the locking decisions in
`../w1/w1_alignment_decision.md`:

* **Model checklist** — Llama-3-8B (official evaluation), GPT-J-6B (early
  validation), gpt2-xl / gpt2-small (quick checks only, not scored).
  The toy smoke environment stands in for these models and implements the
  same EasyEdit-style interface (`from_hparams` / `batch_edit` /
  `evaluate`).
* **Interface style** — EasyEdit `BaseEditor` conventions: editor created
  with `from_hparams(model, **hparams)`, edits applied with
  `batch_edit(cases)`, metrics returned by `evaluate(cases)`.
* **Metric battery** — `rewrite_acc`, `paraphrase_acc`,
  `locality_neighborhood_acc`, `locality_attribute_acc`,
  `portability_acc`, following the W1 alignment decision and the EasyEdit
  generative evaluation protocol.
* **Gate 2 evidence** — the smoke tests below produce the
  reliability / generalisation / locality numbers required to evaluate
  the reproduced baselines before moving to the full models.

---

## 1. Requirements

* Python 3.8+ (tested on Windows 11 / Python 3.12)
* `numpy` (>= 1.21)

No other third-party packages are required: the toy model, the editing
machinery, the evaluation battery, and the smoke-test data are fully
self-contained.

Install dependencies if needed:

```powershell
pip install numpy
```

## 2. Running the smoke tests

Run the MEMIT reproduction (built-in toy model + synthetic edit cases):

```powershell
python memit_repro.py --mock --cases 3 --seed 0 --verbose
```

Run the AlphaEdit reproduction (side-by-side comparison against MEMIT on
the same fitted model):

```powershell
python alphaedit_repro.py --mock --cases 3 --seed 0 --verbose
```

Both commands print a JSON document to stdout containing the edit summary
and the metric battery. Reference results (seed 0, 3 cases):

| Metric | MEMIT | AlphaEdit (n_components=8) |
| --- | --- | --- |
| rewrite_acc | 1.000 | 1.000 |
| paraphrase_acc | 0.500 | 0.333 |
| locality_neighborhood_acc | 1.000 | 1.000 |
| locality_attribute_acc | 1.000 | 1.000 |
| portability_acc | 0.667 | 0.667 |

*Reliability* is captured by `rewrite_acc` (the edited fact is recalled
after editing), *generalisation* by `paraphrase_acc` (rephrased prompts
still retrieve the new fact), *locality* by the two `locality_*` metrics
(unrelated prompts keep their pre-edit answers). In this toy setup the
locality scores are already saturated by the orthogonal prompt design;
AlphaEdit's null-space projection becomes visible as the
rewrite/locality trade-off controlled by `--n-components`.

### Smoke-test options

| Option | Default | Meaning |
| --- | --- | --- |
| `--cases` | 3 | number of synthetic edit cases |
| `--seed` | 0 | random seed (reproducibility) |
| `--lam` | 1e-3 | ridge regularisation of the least-squares solve |
| `--margin` | 1.0 | safety margin of the new-answer logit |
| `--context-scale` | 5.0 | weight of the context covariance term |
| `--n-components` (AlphaEdit) | 8 | top feature directions removed by the null-space projection |
| `--no-rescale` (AlphaEdit) | off | disable the norm rescaling after projection |
| `--verbose` | off | print edit summaries and metrics |

## 3. Files

```
w2_fonghiolam/
├── README.md            (this file)
├── memit_repro.py       MEMIT batch-editing reproduction
├── alphaedit_repro.py   AlphaEdit null-space extension of MEMIT
└── smoke_outputs/
    ├── memit_smoke.json         reference MEMIT smoke output
    └── alphaedit_smoke.json     reference AlphaEdit smoke output
```

## 4. How the reproduction works

### 4.1 Toy model (`MiniLM` in `memit_repro.py`)

A small single-block transformer with:

* embedding table `E` (`VOCAB x HIDDEN`),
* a non-linear FFN block with four linear layers acting on the hidden
  state `h` (activation is linear so the block is a linear map, which
  keeps the ROME-style least-squares update exact),
* an output head `W_out / b_out` fitted by ridge least squares over a
  knowledge set that covers the vocabulary exactly once.

`fit_output()` guarantees that every knowledge prompt answers with its
pre-edit target, matching a real pretrained LM whose output head spans
the vocabulary.

### 4.2 Edit cases

`make_toy_cases()` builds cases with the EasyEdit shape
`{prompt, target_new, target_true, ...}` plus `subject`-structured
paraphrase / portability prompts and locality prompts (neighborhood =
unrelated fact prompt, attribute = unrelated attribute probe). Each
category captures its own pre-edit answer so the metrics are computed
against the true initial state.

### 4.3 MEMIT update (ROME-style locate-then-edit)

`MemitEditor.batch_edit()`:

1. `cache_keys` — collect hidden states at the edit layer for the edit
   prompts and for sampled context prompts.
2. `desired_residuals` — for each edit, compute the residual
   `r_i = W_out^+ (onehot(target_new) - b_out) - h_0` so that
   `h_1 = h_0 + r_i` answers `target_new`.
3. `context_covariance` — build `C = K K^T` from context keys (with
   optional `context_scale`).
4. `compute_layer_update` — solve
   `W_new = W_old + delta_W` with
   `delta_W = (R K^T) (C + lam I)^{-1}`, the ROME/MEMIT closed form.
5. Apply the single batch update to the FFN weight at the edit layer.

### 4.4 AlphaEdit null-space constraint

`AlphaEditor.batch_edit()` reuses the MEMIT delta and then:

1. `build_projector` — eigen-decompose the context covariance and keep
   the top `n_components` directions `V`.
2. `P = I - V V^T` — the null-space projector of the typical feature
   space.
3. `delta_alpha = delta_memit @ P` — drop the component of the update
   that lies along the model's dominant feature directions.
4. optional norm rescaling restores the magnitude lost by the
   projection.

This is the AlphaEdit principle (null-space constrained editing): the
update is only allowed to act in directions the model rarely uses, so
unrelated facts are preserved by construction. Larger `n_components`
means a stronger constraint (better locality, weaker rewrite); smaller
means a softer constraint (closer to plain MEMIT).

## 5. Connecting to the real model evaluation

The toy scripts are intended as executable, verifiable baselines. To run
the same pipelines on the locked model list (Llama-3-8B official,
GPT-J-6B early validation, gpt2-xl / gpt2-small quick checks), port the
interface points as follows:

1. **Keep the EasyEdit-style surface** — swap `MiniLM` for a
   HuggingFace `AutoModelForCausalLM`; implement `embed_prompt`,
   `forward_from`, `get/set_ffn_weight` against the chosen transformer
   (use the EasyEdit hooks at layer `EDIT_LAYER`).
2. **Use the EasyEdit dataset** — replace `make_toy_cases()` with the
   `Counterfact` / `ZsRE` splits from the EasyEdit repository so the
   metrics are comparable with published MEMIT and AlphaEdit numbers.
3. **Keep the same metric battery** — `evaluate_cases()` already follows
   the W1 evaluation protocol; adapt `generate()` to model.generate with
   greedy decoding.
4. **Full-model smoke** — run `--mock` locally for CI, then repeat the
   exact same calls on `gpt2-small -> gpt2-xl -> GPT-J-6B -> Llama-3-8B`
   and record the metrics into the W2 evidence table for Gate 2.

## 6. Notes

* All code is self-contained, English-only, and free of private
  third-party dependencies.
* Results are deterministic for a fixed seed (`--seed`).
* AlphaEdit and MEMIT are compared on the *same* initial model (MEMIT
  run first, then the weights are restored from backup before the
  AlphaEdit run), so the difference comes from the editing mechanism
  alone.
