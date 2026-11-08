# W4 gate ablation - results table

Model: `./hugging_cache/gpt2-xl` | hparams: `gpt2-xl.yaml` | edits: 191 over 40 subjects | held-out locality probes: 60

## 1. Ablation table

| arm | WRITE | ROUTE | REJECT | gate acc. | immediate rewrite | model-only retention | E2E retention | locality | memory entries | s/edit |
|---|---|---|---|---|---|---|---|---|---|---|
| `always_write` | 191 | 0 | 0 | 0.398 | 0.110 | 0.110 | 0.110 | 1.000 | 0 | 0.543 |
| `always_route` | 0 | 191 | 0 | 0.419 | 0.110 | 0.110 | 1.000 | 1.000 | 126 | 0.533 |
| `rule_gate` | 36 | 151 | 4 | 0.613 | 0.110 | 0.110 | 1.000 | 1.000 | 120 | 0.538 |
| `learned_gate` | 1 | 190 | 0 | 0.424 | 0.110 | 0.110 | 1.000 | 1.000 | 126 | 0.529 |

> **Metric caveat.** `immediate rewrite` and `model-only retention` are scored against the *base model's* answer for the same prompt, so they fire only on edits whose subject the untouched model already answers. The identical value in all four arms is therefore expected: the gate cannot change it. On this 191-row stream (40 distinct prompts) it equals the base-model hit rate; on a deduped stream the same four arms report 0.325 (= 13/40 base hits) and the writer-attributable rewrite after correction is 0.000. Diagnosis: `docs/w4_immediate_rewrite_diagnosis.md`; report section 3.4.

> **Table footnotes — table 1 — ablation.** (i) *Backbones are not comparable.* `gpt2-small` is run under the 8 GB profile and `gpt2-xl` from the full 30 GB cache; rows from the two backbones are printed side by side for reading only and are never pooled, differenced or averaged. (ii) *Closed-form arm: not run.* `closed_form_quickfix` does not consume a gate decision and is scored on the 5 K fluent-edit stream, so it cannot be an arm of a gate ablation. Arms without a closed-form counterpart are labelled "not run" instead of being left blank. (iii) *`nll_delta` is not collected.* `run_w4_gate_ablation.py` logs hit counts but no fluency delta, so no row here pairs `param_hit_total` with `nll_delta`. Every hit count in this table is coverage-only and is never reported as a pass on a (hit, NLL) criterion.

## 2. Paired comparisons (exact McNemar + paired bootstrap)

`b` = arm A hits / arm B misses, `c` = the opposite; delta = P(A) - P(B).

| metric | A vs B | b | c | p | delta | delta 95% CI |
|---|---|---|---|---|---|---|
| E2E retention | `always_write` vs `always_route` | 0 | 170 | 0.0000 | -0.890 | [-0.932, -0.843] |
| model-only retention | `always_write` vs `always_route` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| immediate rewrite | `always_write` vs `always_route` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| E2E retention | `always_write` vs `rule_gate` | 0 | 170 | 0.0000 | -0.890 | [-0.932, -0.843] |
| model-only retention | `always_write` vs `rule_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| immediate rewrite | `always_write` vs `rule_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| E2E retention | `always_write` vs `learned_gate` | 0 | 170 | 0.0000 | -0.890 | [-0.932, -0.843] |
| model-only retention | `always_write` vs `learned_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| immediate rewrite | `always_write` vs `learned_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| E2E retention | `always_route` vs `rule_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| model-only retention | `always_route` vs `rule_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| immediate rewrite | `always_route` vs `rule_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| E2E retention | `always_route` vs `learned_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| model-only retention | `always_route` vs `learned_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| immediate rewrite | `always_route` vs `learned_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| E2E retention | `rule_gate` vs `learned_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| model-only retention | `rule_gate` vs `learned_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |
| immediate rewrite | `rule_gate` vs `learned_gate` | 0 | 0 | 1.0000 | +0.000 | [+0.000, +0.000] |

> **Table footnotes — table 2 — paired comparisons.** (i) *Backbones are not comparable.* `gpt2-small` is run under the 8 GB profile and `gpt2-xl` from the full 30 GB cache; rows from the two backbones are printed side by side for reading only and are never pooled, differenced or averaged. (ii) *Closed-form arm: not run.* `closed_form_quickfix` does not consume a gate decision and is scored on the 5 K fluent-edit stream, so it cannot be an arm of a gate ablation. Arms without a closed-form counterpart are labelled "not run" instead of being left blank. (iii) *`nll_delta` is not collected.* `run_w4_gate_ablation.py` logs hit counts but no fluency delta, so no row here pairs `param_hit_total` with `nll_delta`. Every hit count in this table is coverage-only and is never reported as a pass on a (hit, NLL) criterion.


## 3. Gate quality (learned gate versus rules and baselines)

| comparison | b | c | p | kappa(A) | kappa(B) | agreement(A) |
|---|---|---|---|---|---|---|
| learned vs always_write | 80 | 75 | 0.7481 | 0.009 | 0.000 | 0.424 |
| learned vs always_route | 1 | 0 | 1.0000 | 0.009 | 0.000 | 0.424 |
| learned vs rule_gate | 0 | 36 | 0.0000 | 0.009 | 0.343 | 0.424 |

> **Table footnotes — table 3 — gate quality.** (i) *Backbones are not comparable.* `gpt2-small` is run under the 8 GB profile and `gpt2-xl` from the full 30 GB cache; rows from the two backbones are printed side by side for reading only and are never pooled, differenced or averaged. (ii) *Closed-form arm: not run.* `closed_form_quickfix` does not consume a gate decision and is scored on the 5 K fluent-edit stream, so it cannot be an arm of a gate ablation. Arms without a closed-form counterpart are labelled "not run" instead of being left blank. (iii) *`nll_delta` is not collected.* `run_w4_gate_ablation.py` logs hit counts but no fluency delta, so no row here pairs `param_hit_total` with `nll_delta`. Every hit count in this table is coverage-only and is never reported as a pass on a (hit, NLL) criterion.


## 4. Breakdown by conflict type

### `learned_gate`

| situation | n | E2E retention (95% CI) | decision correct (95% CI) |
|---|---|---|---|
| no-conflict | 85 | 1.000 [0.957, 1.000] | 0.012 [0.002, 0.064] |
| pre-training-conflict | 26 | 1.000 [0.871, 1.000] | 0.000 [0.000, 0.129] |
| prior-edit-conflict | 40 | 1.000 [0.912, 1.000] | 1.000 [0.912, 1.000] |
| context-conflict | 40 | 1.000 [0.912, 1.000] | 1.000 [0.912, 1.000] |

> **Table footnotes — table 4 — conflict-type breakdown, `learned_gate`.** (i) *Backbones are not comparable.* `gpt2-small` is run under the 8 GB profile and `gpt2-xl` from the full 30 GB cache; rows from the two backbones are printed side by side for reading only and are never pooled, differenced or averaged. (ii) *Closed-form arm: not run.* `closed_form_quickfix` does not consume a gate decision and is scored on the 5 K fluent-edit stream, so it cannot be an arm of a gate ablation. Arms without a closed-form counterpart are labelled "not run" instead of being left blank. (iii) *`nll_delta` is not collected.* `run_w4_gate_ablation.py` logs hit counts but no fluency delta, so no row here pairs `param_hit_total` with `nll_delta`. Every hit count in this table is coverage-only and is never reported as a pass on a (hit, NLL) criterion.

### `rule_gate`

| situation | n | E2E retention (95% CI) | decision correct (95% CI) |
|---|---|---|---|
| no-conflict | 85 | 1.000 [0.957, 1.000] | 0.435 [0.335, 0.541] |
| pre-training-conflict | 26 | 1.000 [0.871, 1.000] | 0.000 [0.000, 0.129] |
| prior-edit-conflict | 40 | 1.000 [0.912, 1.000] | 1.000 [0.912, 1.000] |
| context-conflict | 40 | 1.000 [0.912, 1.000] | 1.000 [0.912, 1.000] |

> **Table footnotes — table 4 — conflict-type breakdown, `rule_gate`.** (i) *Backbones are not comparable.* `gpt2-small` is run under the 8 GB profile and `gpt2-xl` from the full 30 GB cache; rows from the two backbones are printed side by side for reading only and are never pooled, differenced or averaged. (ii) *Closed-form arm: not run.* `closed_form_quickfix` does not consume a gate decision and is scored on the 5 K fluent-edit stream, so it cannot be an arm of a gate ablation. Arms without a closed-form counterpart are labelled "not run" instead of being left blank. (iii) *`nll_delta` is not collected.* `run_w4_gate_ablation.py` logs hit counts but no fluency delta, so no row here pairs `param_hit_total` with `nll_delta`. Every hit count in this table is coverage-only and is never reported as a pass on a (hit, NLL) criterion.

### `always_write`

| situation | n | E2E retention (95% CI) | decision correct (95% CI) |
|---|---|---|---|
| no-conflict | 85 | 0.247 [0.168, 0.348] | 0.894 [0.811, 0.943] |
| pre-training-conflict | 26 | 0.000 [0.000, 0.129] | 0.000 [0.000, 0.129] |
| prior-edit-conflict | 40 | 0.000 [0.000, 0.088] | 0.000 [0.000, 0.088] |
| context-conflict | 40 | 0.000 [0.000, 0.088] | 0.000 [0.000, 0.088] |

> **Table footnotes — table 4 — conflict-type breakdown, `always_write`.** (i) *Backbones are not comparable.* `gpt2-small` is run under the 8 GB profile and `gpt2-xl` from the full 30 GB cache; rows from the two backbones are printed side by side for reading only and are never pooled, differenced or averaged. (ii) *Closed-form arm: not run.* `closed_form_quickfix` does not consume a gate decision and is scored on the 5 K fluent-edit stream, so it cannot be an arm of a gate ablation. Arms without a closed-form counterpart are labelled "not run" instead of being left blank. (iii) *`nll_delta` is not collected.* `run_w4_gate_ablation.py` logs hit counts but no fluency delta, so no row here pairs `param_hit_total` with `nll_delta`. Every hit count in this table is coverage-only and is never reported as a pass on a (hit, NLL) criterion.

### `always_route`

| situation | n | E2E retention (95% CI) | decision correct (95% CI) |
|---|---|---|---|
| no-conflict | 85 | 1.000 [0.957, 1.000] | 0.000 [0.000, 0.043] |
| pre-training-conflict | 26 | 1.000 [0.871, 1.000] | 0.000 [0.000, 0.129] |
| prior-edit-conflict | 40 | 1.000 [0.912, 1.000] | 1.000 [0.912, 1.000] |
| context-conflict | 40 | 1.000 [0.912, 1.000] | 1.000 [0.912, 1.000] |

> **Table footnotes — table 4 — conflict-type breakdown, `always_route`.** (i) *Backbones are not comparable.* `gpt2-small` is run under the 8 GB profile and `gpt2-xl` from the full 30 GB cache; rows from the two backbones are printed side by side for reading only and are never pooled, differenced or averaged. (ii) *Closed-form arm: not run.* `closed_form_quickfix` does not consume a gate decision and is scored on the 5 K fluent-edit stream, so it cannot be an arm of a gate ablation. Arms without a closed-form counterpart are labelled "not run" instead of being left blank. (iii) *`nll_delta` is not collected.* `run_w4_gate_ablation.py` logs hit counts but no fluency delta, so no row here pairs `param_hit_total` with `nll_delta`. Every hit count in this table is coverage-only and is never reported as a pass on a (hit, NLL) criterion.

