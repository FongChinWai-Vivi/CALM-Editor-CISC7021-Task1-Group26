# Reproducibility appendix (W5 deliverable, Tang I Choi)

> **Path note (2026-11-05).** Fong Chin Wai's artefacts were consolidated into `FONG CHIN WAI/` (see `FONG CHIN WAI/README.md` for the old-path to new-path mapping). References below have been updated; the bare directory names `output/` and `calm/` that appear in narrative text now live under `FONG CHIN WAI/`.


Everything the group needs to re-run the W3/W4/W5 experiments from a fresh
clone. The driver is
`TANG I CHOI/w5_report_and_reproducibility/reproduce_all.sh`.

## 1. Environment actually used

| item | value |
|---|---|
| machine | Apple M3 Max (Mac15,11, arm64), macOS 15.7.1, 36 GB unified memory |
| python | 3.11.9 |
| torch | 2.14.1 (MPS backend available, `torch.backends.mps.is_available() == True`) |
| scikit-learn | 1.9.1 |
| numpy / matplotlib | 2.4.6 / 3.11.2 |
| backbone (W3/W4) | `gpt2-xl` cached under `EasyEdit/hugging_cache/gpt2-xl` |
| backbone (team 1K/5K runs) | `gpt2-small` (Windows 11, see `FONG CHIN WAI/ run records (`w3_hypernetwork_stream/`, `w4_sequential_stream_and_fix/`)` `environment` block) |

The team 1K/5K stream runs in Table 1 were produced on a different machine
(`gpt2-small`), so absolute numbers are not comparable with the W4 `gpt2-xl`
ablation; only the direction of the effects is.

## 2. Seeds and determinism

| artefact | seed / determinism |
|---|---|
| dataset split (train/dev/test) | `seed=42`, grouped by subject |
| gate classifier | scikit-learn `random_state=0`, `C=1.0`, `class_weight="balanced"` |
| bootstrap CI | 5000 paired draws, `seed=20261101` |
| edit stream | deterministic: first 40 subjects of `gate_test.jsonl`, dataset order preserved, up to 6 edits per subject |
| locality probes | deterministic: first 60 held-out rows (subjects never in the stream) |
| LM generation | greedy (`max_new_tokens=5`), temperature 0 -> the answer signal is reproducible per backbone |

The edit stream and the locality probes are *not* re-shuffled between arms,
which is what makes the paired McNemar tests valid.

## 3. One-command reproduction

```bash
git clone https://github.com/FongChinWai-Vivi/CALM-Editor-CISC7021-Task1-Group26
cd CALM-Editor-CISC7021-Task1-Group26
bash "TANG I CHOI/w5_report_and_reproducibility/reproduce_all.sh"
```

Defaults: `PY=python`, `SUBJECTS=40`, `FORCE_TRAIN=0`. Steps 1-2 (dataset build
and classifier training) are skipped when the artefacts already exist, because
they need the CounterFact answer cache; `FORCE_TRAIN=1` rebuilds them.

Expected wall time on the machine above: data build ~6 min (cached answers
re-used), training ~20 s, gap evaluation ~5 s, four ablation arms ~7 min,
statistics ~5 s, figures ~3 s. Total under 15 min without `FORCE_TRAIN`.

## 4. What each log corresponds to

| log | command | produced |
|---|---|---|
| `run_logs/w4_ablation_gpt2xl_20261101.log` | `run_w4_gate_ablation.py --all-arms --subjects 40` | `w4_gate_ablations/results/w4_ablation_*.json` |
| `run_logs/w3_build_data.log` | `build_gate_training_data.py` | `gate_datasets/*.jsonl`, `dataset_card.json` |
| `run_logs/w3_train_gate.log` | `train_gate_classifier.py` | `gate_model/gate_classifier.joblib`, `metrics.md` |
| `run_logs/w3_gate_online_gap.log` | `evaluate_gate_online_gap.py` | `results/gate_online_gap.json` |
| `run_logs/w4_statistics.log` | `analyze_w4_statistics.py` | `results/w4_statistics.json`, `w4_ablation_table.md` |
| `run_logs/w5_figures.log` | `make_w5_figures.py` | `figures/fig{1,2,3}_*.png` |
| `run_logs/w5_tables.log` | `collect_w5_tables.py` | `report/results_tables.md` |

Logs are appended to by the driver; the ablation log is kept as a single
snapshot because it is the slowest step and reviewers usually want to see the
per-25-edit progress and the per-arm summaries in one file.

## 5. Known environment pitfalls

1. **Run the ablation from `EasyEdit/`-aware context only.** The script inserts
   `EasyEdit` into `sys.path` itself, but `hparams/CALM/gpt2-xl.yaml` paths are
   relative to `EasyEdit/`, so run it from
   `TANG I CHOI/w4_gate_ablations` (as the driver does) and not from the repo
   root.
2. **Module caching between arms.** Each arm loads a module named
   `conflict_gate`; the runner pops it from `sys.modules` before loading the
   next arm, otherwise `--all-arms` silently re-uses the first arm. Keep that
   `sys.modules.pop` if the file is refactored.
3. **MPS memory.** `gpt2-xl` at scale 1.0 fits in 36 GB; the runner calls
   `torch.mps.empty_cache()` between arms. On a 16 GB machine use
   `--subjects 20` and expect a noisier McNemar table.
4. **`EasyEdit/hugging_cache` is not in the repo** (size); the first run
   downloads `gpt2-xl`, which adds ~3 GB and several minutes.
5. **No teammate file is written.** Arms are injected via
   `hparams.conflict_gate_path`; the CALMEditor guards are disabled on the
   instance only. Re-running therefore cannot dirty the shared `calm/` or
   `EasyEdit/` trees.
