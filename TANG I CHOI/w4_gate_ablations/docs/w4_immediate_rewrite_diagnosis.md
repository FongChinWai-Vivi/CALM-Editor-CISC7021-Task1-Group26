# W4 `immediate_rewrite` metric: diagnosis and fix

Owner: Tang I Choi (MC631517) · W4 gate ablations · last run 2026-11-05.

## TL;DR

`immediate_rewrite_rate = 0.1099` for `always_write` was **not** a measurement of
the writer. It was the base model's hit rate on the stream, re-weighted by
duplicated prompts. Three facts settle it:

| check | observation | meaning |
|---|---|---|
| four arms, legacy 191-row stream | `always_write` 0.1099, `always_route` 0.1099, `rule_gate` 0.1099, `learned_gate` 0.1099 | a writer metric cannot be identical when the WRITE share moves from 0% to 100% |
| four arms, deduplicated 40-row stream | all four arms 0.325 = 13/40 | 0.325 is exactly the fraction of prompts whose *base* answer already contains `target_new` |
| write-effect hook (`delta_norm`, answers) | deltas are applied (mean ‖Δ‖ = 0.0041), `answer_changed_by_write_rate = 0.000` | the WRITE lands in the weights but never moves greedy decoding |

After removing the baseline contamination the honest number is
**`immediate_rewrite = 0.000`** on all four arms, and the reason is that the
hypernetwork was never trained (`hparams/CALM/gpt2-xl.yaml` sets no
`hypernet_ckpt`, so `HypernetworkEditGenerator` keeps its random initialisation).

## Evidence 1 — the metric does not move with the gate

Legacy stream, 191 edits / 40 subjects (`results/w4_ablation_*.json`):

| arm | WRITE / ROUTE / REJECT | immediate_rewrite |
|---|---|---|
| always_write | 191 / 0 / 0 | 0.1099 |
| always_route | 0 / 191 / 0 | 0.1099 |
| rule_gate | 36 / 151 / 4 | 0.1099 |
| learned_gate | 1 / 190 / 0 | 0.1099 |

Four different decision mixes, one identical fidelity number. An `always_route`
arm writes nothing to the parameters and still "rewrites" 10.99% of the stream.

## Evidence 2 — `rewrite_ok` is baseline-contaminated

`EasyEdit/easyeditor/editors/calm_editor.py::_measure` returns
`{"rewrite_ok": bool(ok)}` where `ok` compares the post-edit greedy answer with
`target_new` — it never looks at the pre-edit answer. A row whose base answer
already contains `target_new` therefore reports `rewrite_ok = True` although the
edit contributed nothing. On the deduplicated stream:

| arm | reported | baseline hit | post hit | corrected (`post_hit ∧ ¬pre_hit`) |
|---|---|---|---|---|
| always_write | 0.325 | 0.325 | 0.325 | **0.000** |
| always_route | 0.325 | 0.325 | 0.325 | **0.000** |
| rule_gate | 0.325 | 0.325 | 0.325 | **0.000** |
| learned_gate | 0.325 | 0.325 | 0.325 | **0.000** |

Reported rate = baseline rate to the fourth decimal in every arm.

## Evidence 3 — the write is applied but ineffective

`run_w4_gate_ablation.py` now patches `editor.hypernetwork.apply_update` at
runtime (no teammate file is touched) and records the Frobenius norm of every
delta plus the answer before/after the edit:

| arm | write share | Δ applied | mean ‖Δ‖_F | answer changed by WRITE |
|---|---|---|---|---|
| always_write | 1.000 | 100% | 0.003912 | 0.000 |
| rule_gate | 0.900 | 100% | 0.004086 | 0.000 |
| learned_gate | 0.025 | 100% | 0.004723 | 0.000 |
| always_route | 0.000 | 0% | 0.000 | 0.000 |

Root cause: `EasyEdit/hparams/CALM/gpt2-xl.yaml` carries `hypernet_alg:
lowrank`, `hypernet_rank: 2`, `hypernet_scale: 1.0` but **no `hypernet_ckpt`**.
`HypernetworkEditGenerator` documents this branch explicitly ("when `ckpt` is
`None` the generator keeps its untrained random initialisation"), so every WRITE
adds a random rank-2 delta of norm ≈ 4e-3 to six GPT-2 XL MLP matrices. That is
far too small and too unstructured to change a greedy token.

This is consistent with the team-level finding in `output/w4_calm_fix_report.md`
and `w4_model_level_improve/` (model-level write quality 0.08% even with a
trained MEND hypernetwork), and it is the reason the W4 fix moved the system
target from *parameter-level* retention to *memory-backed end-to-end* retention.

## Evidence 4 — the 191-row stream double-counts prompts

`load_stream` groups up to 6 CounterFact situations per subject, and those
situations reuse a single prompt with **contradictory** targets (e.g.
`Winchester railway station was named for` → `Winchester` / `Alps` / `Cairo`).
No editor can satisfy two contradictory edits on one prompt, so the ceiling of
`immediate_rewrite` on that stream is `1 / edits_per_prompt ≈ 0.21`.

Replaying the legacy 191 rows against the per-subject baseline of the
deduplicated run explains the number exactly:

* 13 of the 40 subjects are base-model hits;
* 21 legacy rows report `immediate_rewrite = True`;
* **21 / 21** of them sit on those 13 baseline-hit subjects (`legacy_true_rows_explained` in
  `results/immediate_rewrite_diagnosis.json`).

0.1099 = baseline hits counted once per duplicated prompt.

## Fix applied (all inside `TANG I CHOI/w4_gate_ablations/`)

1. `run_w4_gate_ablation.py`
   * new `dedupe_by_prompt()` + `--dedupe-prompts`: one edit per distinct prompt,
     so contradictory targets cannot cancel each other;
   * new per-edit fields `answer_pre`, `answer_post`, `delta_norm` (via the
     `apply_update` hook);
   * new per-run fields `stream_mode`, `n_stream_rows_before_dedupe`,
     `n_distinct_prompts`, `hypernet_ckpt`, `write_delta_applied_rate`,
     `mean_delta_norm`, `answer_changed_by_write_rate`.
   * default behaviour is unchanged (dedupe off), so the earlier numbers stay
     reproducible.
2. `diagnose_immediate_rewrite.py` — reproduces every table above and writes
   `results/immediate_rewrite_diagnosis.json`.
3. `results/w4_ablation_*_deduped.json` — the four arms re-run on the
   deduplicated stream (40 edits / 40 prompts, gpt2-xl, greedy).
4. `analyze_w4_statistics.py` now reports `baseline_hit_rate` and
   `rewrite_after_baseline_correction` next to the legacy rate.

## Recommended reporting rule

| metric | use it for | do **not** use it for |
|---|---|---|
| `immediate_rewrite_rate` (legacy) | reproducing the old table only | comparing gate arms — it is the base-model hit rate |
| `rewrite_after_baseline_correction` | measuring the writer | — |
| `decisions` / `gate_accuracy_vs_gold` / Cohen's κ | comparing gate arms | measuring the writer |

With the correction, the W4 conclusion stands but must be stated precisely: the
gate layer is comparable across arms (see `results/w4_statistics.json`), while
the **writer contributes nothing at model level** because the hypernetwork in
this ablation was never trained; all end-to-end retention in W4 is memory-backed.
