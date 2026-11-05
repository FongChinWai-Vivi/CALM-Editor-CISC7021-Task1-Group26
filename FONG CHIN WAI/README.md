---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_902d27a2d91d11f19280525400393706
    ReservedCode1: HfMRZ7+uKSxcd6UixbXAy1joIHpbMEx0VyGxcjX5m/UHWy5pi5AX3AfeBIzSVRYdl7pATrqXYVqcAhIQtx6SXbIcMHIei525Gc2tmP5hUbuwO7DAJWop4fh3/62BZa8eGR62UfRCoRoR2mftZVWUTOGbJ4wPLq51hqyGo5+6uUeAJen51Uw/E8hGwjQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_902d27a2d91d11f19280525400393706
    ReservedCode2: HfMRZ7+uKSxcd6UixbXAy1joIHpbMEx0VyGxcjX5m/UHWy5pi5AX3AfeBIzSVRYdl7pATrqXYVqcAhIQtx6SXbIcMHIei525Gc2tmP5hUbuwO7DAJWop4fh3/62BZa8eGR62UfRCoRoR2mftZVWUTOGbJ4wPLq51hqyGo5+6uUeAJen51Uw/E8hGwjQ=
---

# FONG CHIN WAI -- W1-W5 (MC653748)

Personal work package archive for Group 26's CALM-Editor Task 1. This folder
collects the six work packages owned by Fong Chin Wai:

* **W1** -- the shared experiment framework and the unified evaluation protocol;
* **W2** -- the ROME single-fact baseline;
* **W3** -- the hypernetwork edit generator and the first 1K-edit sequential stream;
* **W4** -- the 5K sequential/lifelong stream, the bounded-write fix, and the
  model-level write-quality campaign;
* **W5** -- the assembled results tables, figures and the audited numbers.

The shared EasyEdit code base is **not** moved into this folder; it stays at
`EasyEdit/` (see *Where the code lives* below). The repository-root `README.md`
keeps the cross-member narrative and links into the paths in this folder.

## Map

| folder | package | one-line summary |
|---|---|---|
| `w1_framework_and_protocol/` | W1 | unified evaluation protocol (reliability / generalisation / locality), the W1 alignment decision, the small-scale MEND verification, and the frozen CALM editor interface draft |
| `w2_rome_baseline/` | W2 | ROME reproduction on gpt2-xl (CounterFact and zsRE, n=50) plus the MQuAKE portability probe and the GPT-J-6B Colab notebooks |
| `w3_hypernetwork_stream/` | W3 | hypernetwork edit generator validated end-to-end on a 1K-edit stream (route distribution, side-memory usage, sampled retention) |
| `w4_sequential_stream_and_fix/` | W4 (first half) | the 5K-edit sequential stream, and the bounded-write fix (`write_budget=128` + dual-write) that makes the system-level stream succeed |
| `w4_model_level_improve/` | W4 (second half) | accumulation-aware hypernetwork training campaign, diagnosis report, per-checkpoint evaluations, audit JSONs, checkpoints and training logs |
| `w5_results_and_figures/` | W5 | main results table, results-section draft, the audited-versus-raw write-hit figure and the audited write-hit figure |

## Contents and findings

### `w1_framework_and_protocol/` -- framework and protocol

| file | what it is |
|---|---|
| `evaluation_protocol.md` | the unified evaluation protocol: reliability = `rewrite_acc`, generalisation = `rephrase_acc`, locality = `neighborhood_acc` / `nq_acc`, portability as a separate split; defines the per-edit record schema shared by all three members |
| `w1_alignment_decision.md` | locked model roster (gpt2-small / gpt2-xl / GPT-J-6B), frozen CALM editor interface, Gate 1 checklist |
| `w1_hypernetwork_verification.md` | small-scale MEND training check: pipeline runs end to end, loss converges, edit effect visible (pipeline feasibility only, no published-metric claim) |
| `calm_editor_interface_draft.py` | the frozen W1 interface blueprint (formerly `calm/calm_editor.py`, moved here) that later became `EasyEdit/easyeditor/editors/calm_editor.py` |
| `W1_Team_Notice.docx` | the W1 team notice circulated with the alignment decision |

Key point: the protocol is what makes every number below comparable -- all
methods report under the same three axes, and the portability probe is reported
separately rather than folded into generalisation.

### `w2_rome_baseline/` -- ROME single-fact baseline

| file | what it is |
|---|---|
| `w2_rome_reproduction.md` | ROME reproduction report (method, data, numbers, local-hardware analysis) |
| `w2_rome_gptj6b_colab_v5.ipynb`, `w2_rome_gptj6b_colab_v6.ipynb` | the GPT-J-6B runs executed on Colab T4 (v6 is the final one) |

Findings: on gpt2-xl (n=50) ROME reaches **CounterFact 1.000 / 0.700 / 0.842**
and **zsRE 0.997 / 0.807 / 0.966** (rewrite / rephrase / locality). Portability
on MQuAKE multi-hop questions is weak (**p1 = 0.069**, n=50), consistent with the
ROME paper. GPT-J-6B was validated end-to-end on Colab T4 (zsRE n=10:
1.000 / 0.950 / 0.610) and is **not feasible on the local RTX 4070 8GB**
(weights plus activations exceed physical VRAM; ~570 s/case with system-memory
fallback).

### `w3_hypernetwork_stream/` -- hypernetwork edit generator, 1K stream

| file | what it is |
|---|---|
| `w3_calm_stream_1k.json` | machine-readable run record for the 1000-edit stream |
| `w3_calm_stream_verification.md` | verification report: route distribution, side-memory statistics, sampled retention |

Findings: with a meta-trained hypernetwork the 1K stream distributes
**WRITE 85.5% / ROUTE 14.1% / REJECT 0.4%**, end-to-end immediate rewrite is
**17.2%** (baseline without the hypernetwork: 15.1%), while model-level rewrite
is **0.4%** (baseline 1.0%). Sampled retention from the middle and the end of
the stream is **0/6** in both configurations: the hypernetwork gives a small
system-level gain but no lifelong retention.

### `w4_sequential_stream_and_fix/` -- 5K stream and the bounded-write fix

| file | what it is |
|---|---|
| `w4_sequential_5k_results.json`, `w4_sequential_5k_verification.md` | the 5K-edit stream under the default gate settings, both with and without the trained hypernetwork |
| `w4_calm_fix_5k_results.json`, `w4_calm_fix_report.md` | the same 5K stream with the deployment guards (`write_budget=128`, first-write guard, dual-write) |

Findings: extending the stream from 1K to 5K erases the meta-training advantage
-- trained **14.78%** end-to-end versus baseline **14.86%**, model-level
0.08% versus 0.76%, retention **0/9** in both. The tuned 1K advantage does not
survive a longer stream, and side memory is far from saturated (~11 entries per
shard against a capacity of 200) yet still not recalled. The fix changes the
*system-level* picture only: with `write_budget=128` + dual-write, 97.4% of the
5K stream is routed to memory, end-to-end rewrite rises to **97.48%** and
retention becomes memory-backed (**9/9**), while model-level quality stays at
0.08% -- parameter writes are bounded, not improved.

### `w4_model_level_improve/` -- model-level write-quality campaign

| item | what it is |
|---|---|
| `w4_model_level_improve_diagnosis.md` | root-cause diagnosis: inference-scale sweep and cumulative-pollution probe |
| `train_accum.py`, `train_accum_gate.py`, `train_accum_warm.py` | accumulation-aware meta-training (segment without snapshot restore; gate-distribution sampling; warm-started segments) |
| `debug_gate_ns.py` | gate / null-space diagnostic for a checkpoint (kept locally, untracked) |
| `w4_accum_training_report.md` | the campaign report: all configurations, per-configuration numbers, gate-threshold sweep |
| `results_analysis.ipynb` | analysis notebook over the `eval_*/` outputs |
| `norm_log_*.json`, `normlog_scale10.json`, `result_cap395623.json` | per-run normalisation traces and a captured run record |
| `eval_*/` (27 configurations) | per-configuration evaluation JSONs (budget sweeps, coherence penalties, multi-seed, rank/hidden variants, null-space probes) |
| `results/` | the W5 audit JSONs (`audit_baseline.json`, `audit_coh_l0.1.json`, `audit_coh_v2_l0.3_c0.6.json`, `audit_v2_seed101.json`), force-added because `results/` is gitignored |
| `ckpt/` | 33 checkpoints (~6.4 GB). **Gitignored, present on the local machine only**; regenerate with the training scripts below |
| `logs/`, `temp_diag/`, `_diag_warm.py`, `_inspect*.py`, `*.log`, `debug_ns*.txt` | raw training logs and throw-away diagnostic dumps from the campaign (gitignored or untracked, present on the local machine only) |

Findings: the W3 MEND-style training restores the parameter snapshot every
step, so the hypernetwork only learns to emit effective deltas on a *clean*
base; on an already-polluted stack it stops producing observable rewrites
(clean-baseline 5/10 versus cumulative 2/10, every delta after step 2 invisible
in greedy decoding). The fix is a different training paradigm, not a larger
delta (`scale=10` in-stream hurts: 2/129 versus 4/131). Raw model-level results
of the campaign:

| training configuration | model-level immediate (raw) |
|---|---|
| W3 old checkpoint (single-edit training) | 4/500 (0.8%) |
| accum, segment_len=4 | 8/500 (1.6%) |
| accum, segment_len=16 | 51/500 (10.2%) |
| accum16 + gate-distribution sampling | 32/500 (6.4%) |
| accum, segment_len=32 | 37/500 (7.4%) |
| accum16 + warm-start 32 | 56/500 (11.2%) |
| accum16 + warm-start 48 | 85/500 (17.0%) |
| accum16 + warm-start 64 | 35/500 (7.0%) |
| accum16 + warm-start 48, 1000 steps | 111/500 (22.2%) raw -> **4/128 (3.1%) audited** |

Warm-start depth follows a single-peaked curve: 48 is optimal because most of
the 128 real-stream WRITEs occur at pollution depth 16-64, while depth 64
over-trains the hypernetwork into small conservative deltas that miss on the
shallower real stream.

### `w5_results_and_figures/` -- results, figures, audit

| file | what it is |
|---|---|
| `w4_w5_main_results.md` | main results table (baselines, streams, model-level campaign, audit) with the evidence paths |
| `results_draft.md` | drop-in results section draft |
| `w5_audited_real_write_hit.png` | per-configuration audited real WRITE hit rate (128-write audit) |
| `w5_raw_vs_audited_real_write_hit.png` | raw versus audited write-hit rate, the figure that shows the inflation |

## Results in one table

| what | number |
|---|---|
| W2 ROME, gpt2-xl, CounterFact (n=50) | 1.000 / 0.700 / 0.842 (rewrite / rephrase / locality) |
| W2 ROME, gpt2-xl, zsRE (n=50) | 0.997 / 0.807 / 0.966 |
| W2 ROME portability, MQuAKE p1 (n=50) | 0.069 |
| W2 ROME, GPT-J-6B, Colab T4, zsRE (n=10) | 1.000 / 0.950 / 0.610 |
| W3 1K stream, trained hypernet | e2e 17.2%, model-level 0.4%, retention 0/6 |
| W3 1K stream, no hypernet | e2e 15.1%, model-level 1.0%, retention 0/6 |
| W4 5K stream, trained hypernet | e2e 14.78%, model-level 0.08%, retention 0/9 |
| W4 5K stream, no hypernet | e2e 14.86%, model-level 0.76%, retention 0/9 |
| W4 5K + `write_budget=128` + dual-write | e2e 97.48%, retention 9/9, model-level 0.08% |
| W4 best checkpoint, model-level immediate (raw) | 111/500 (22.2%) |
| W4 best checkpoint, **audited real WRITE hits** | **4/128 (3.1%)** |

The honest summary: **the system-level lifelong stream can be made to work
(97.48% with 9/9 retention), but only by routing almost everything to memory;
the parameter-write quality itself stays low (3.1% real hits, 124 of 128
parameter WRITEs still miss).** Long-horizon retention inside the parameters
remains unsolved.

## The W4 model-level metric: raw 22.2% versus audited 3.1%

The `22.2%` figure for the best checkpoint is the **pre-audit** convention
`model_level_ok / 500`, which counts every gate-level success in the 500-edit
stream, including edits the gate REJECTed because the model already answers
correctly. It is **inflated and must not be quoted as write quality**. A
per-edit 128-write audit of the same checkpoint gives:

* genuine parameter-write hits: **4/128 = 3.1%**;
* of the 111 raw hits, **107 are REJECT false hits** (`current_answer_matches_target`),
  4 are real parameter writes, and ROUTE contributes **0** (side-memory reads
  are not parameter-write quality);
* the audited improvement over the W3 checkpoint is therefore **0.08% -> 3.1%**,
  not 0.08% -> 22.2%.

Coherence-penalty and multi-seed routes audited under the same protocol do not
exceed this: `coh_l0.1` **5/128 (3.9%)**, `coh_v2_l0.3` **3/128 (2.3%)**,
`v2_seed101` **2/128 (1.6%)**, all with ROUTE hits 0 and 82 / 40 / 28 REJECT
false hits respectively. A gate-threshold sweep shows the default
`route_threshold=0.6` is already best for the model-level metric; lowering it to
0.3-0.35 raises the *system-level* rewrite rate to 96-100% at the cost of
parameter-write quality.

## Start here (reproduction)

Run from the repository root on Windows, with `.venv` present. The three
`train_accum*.py` scripts and the local `debug_gate_ns.py` locate the repository
root by walking upwards until they find `.git`, so they can be started from any
subfolder; `w3_calm_stream.py` and `audit_per_edit.py` live inside `EasyEdit/`
and resolve their paths relative to it.

```bat
REM 0) environment check
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"

REM 1) W3 / W4 end-to-end stream (gpt2-small + CounterFact). --max-edits 1000 for W3, 5000 for W4.
REM    Add --write-budget 128 and --dual-write for the W4 fix configuration.
.venv\Scripts\python.exe EasyEdit\examples\w3_calm_stream.py --max-edits 1000

REM 2) W4 accumulation-aware training, segment_len=16
.venv\Scripts\python.exe "FONG CHIN WAI\w4_model_level_improve\train_accum.py" --batch-size 16 --train-steps 501 --scale 5.0 --ckpt-name hypernet_accum16_gpt2-small.pt

REM 3) W4 warm-started training -- the best checkpoint (warm-start 48, 1000 steps)
.venv\Scripts\python.exe "FONG CHIN WAI\w4_model_level_improve\train_accum_warm.py" --warm-start 48 --train-steps 1000 --ckpt-name hypernet_accum16_warm48_steps1000_gpt2-small.pt

REM 4) W5 per-edit audit of a checkpoint under the 128-write protocol.
REM    audit_per_edit.py changes its working directory to EasyEdit\, so pass ABSOLUTE paths.
.venv\Scripts\python.exe EasyEdit\examples\audit_per_edit.py ^
    --ckpt "%CD%\FONG CHIN WAI\w4_model_level_improve\ckpt\hypernet_accum16_warm48_steps1000_gpt2-small.pt" ^
    --out "%CD%\FONG CHIN WAI\w4_model_level_improve\results\audit_baseline.json" ^
    --max-edits 500
```

If you only want the numbers: `w5_results_and_figures/w4_w5_main_results.md`
(tables) and `w5_results_and_figures/*.png` (figures). If you want the argument:
`w5_results_and_figures/results_draft.md`,
`w4_model_level_improve/w4_accum_training_report.md` and
`w4_model_level_improve/w4_model_level_improve_diagnosis.md`.

## Where the code lives

The runnable code base is shared by all three members and stays in `EasyEdit/`;
it is not duplicated here, so that there is exactly one implementation to
review:

| file | role |
|---|---|
| `EasyEdit/easyeditor/editors/calm_editor.py` | the CALM editor itself: hypernetwork write, conflict gate decision, side-memory routing, bounded-write guards (first-write guard, `write_budget`, dual-write) |
| `EasyEdit/easyeditor/models/calm/hypernetwork.py` | the hypernetwork edit generator (low-rank delta) |
| `EasyEdit/easyeditor/models/calm/calm_hparams.py` | CALM hyperparameters |
| `EasyEdit/examples/w3_calm_stream.py` | the sequential/lifelong stream driver used for the W3 and W4 stream numbers |
| `EasyEdit/examples/audit_per_edit.py` | the per-edit audit that separates real WRITE hits from ROUTE and REJECT hits |

`w1_framework_and_protocol/calm_editor_interface_draft.py` is the frozen W1
interface blueprint these implementations were written against; it is kept for
the record and is not imported by the running code.

## Conventions and boundaries

- Only `FONG CHIN WAI/` is written. `FONG HIO LAM/`, `TANG I CHOI/`, `EasyEdit/`
  and the remaining `docs/` entry (Tang's `docs/w2_ike_grace_reproduction.md`)
  are out of scope for this archive.
- The four scripts in `w4_model_level_improve/` derive the repository root by
  walking upwards to the directory containing `.git`, so moving this folder no
  longer breaks their `EasyEdit/` and `ckpt/` paths.
- Checkpoints (~6.4 GB in `ckpt/`) and the `logs/`, `temp_diag/` dumps are
  intentionally **not** tracked; every number in the documents can be
  regenerated from the scripts and the `eval_*/` JSONs.
- `results/` is gitignored upstream, so the W5 audit JSONs were force-added;
  they are the evidence behind the audited 3.1% figure.
- Documents keep their original language and encoding (the Chinese W2 report and
  the Chinese-annotated campaign report stay in UTF-8, unchanged).
