# W5 -- Report coordination, reproducible scripts, logs (Tang I Choi, MC631517)

This folder is the single entry point for the group's experiments: it renders
the figures, assembles the results tables, and records the logs of the runs the
tables are built from. It does not contain the report text itself (that lives
in the shared `docs/`), only the drop-in results section and the numbers behind
it.

## Contents

| path | what it is |
|---|---|
| `reproduce_all.sh` | one command that runs W3 -> W5 in order and tees every step into `run_logs/` |
| `make_w5_figures.py` | renders `figures/fig1_decision_mix.png`, `fig2_retention.png`, `fig3_gate_gap.png` from the W3/W4 JSON artefacts |
| `collect_w5_tables.py` | writes `report/results_tables.md`: stream results (team), gate ablation (W4), gate quality (W3), paired tests |
| `report/results_section.md` | drop-in draft of the report's results section (4.1-4.4) with the discussion and threats to validity |
| `report/reproducibility_appendix.md` | environment, seeds, per-command logs, known pitfalls |
| `report/results_tables.md` | generated tables (do not edit by hand) |
| `run_logs/` | last run of each step, so a reviewer can see what produced the numbers |
| `figures/` | the three results figures |

## How the figures are built

All three read machine-readable artefacts, never screenshots or hand-copied
numbers:

- `fig1_decision_mix.png` and `fig2_retention.png` <- `TANG I CHOI/w4_gate_ablations/results/w4_statistics.json`
- `fig3_gate_gap.png` <- `w3_conflict_gate_training_and_classifier/results/gate_online_gap.json`
  plus the in-stream gate accuracy from `w4_statistics.json`

If a script cannot find its input it exits with a message naming the missing
file, so a stale figure cannot silently survive a re-run.

## Quick start

```bash
bash "TANG I CHOI/w5_report_and_reproducibility/reproduce_all.sh"   # everything
cd "TANG I CHOI/w5_report_and_reproducibility" && python make_w5_figures.py   # figures only
cd "TANG I CHOI/w5_report_and_reproducibility" && python collect_w5_tables.py # tables only
```

## Boundaries

- This folder writes only inside `TANG I CHOI/`. The shared `output/` JSONs of
  the team stream runs are read, never modified.
- Table 1 mixes two machines (team `gpt2-small` 1K/5K runs vs. Tang's
  `gpt2-xl` ablation); the numbers are labelled accordingly and
  `report/results_section.md` states that they are not directly comparable.
