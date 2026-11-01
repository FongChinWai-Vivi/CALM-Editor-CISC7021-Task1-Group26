# TANG I CHOI -- W3-W6 (MC631517)

Owned work packages of Group 26's CALM-Editor Task 1. W1/W2 deliverables live
outside this folder (annotation manual, `docs/w2_*`), because they predate the
per-member layout.

## Map

| folder | package | one-line summary |
|---|---|---|
| `w3_conflict_gate_training_and_classifier/` | W3 | build the gate training data from CounterFact in the W1 four-label scheme, train the write/route/reject classifier, and measure how it degrades offline -> in-stream |
| `w4_gate_ablations/` | W4 | four gate arms (`always_write` / `always_route` / `rule` / `learned`) over one fixed stream, plus paired statistics and a per-conflict-type breakdown |
| `w5_report_and_reproducibility/` | W5 | three results figures, the assembled results tables, the drop-in results section, the reproducibility appendix, and `reproduce_all.sh` + run logs |
| `w6_cross_review/` | W6 | review findings for the other members' artefacts (filed, not applied) and the pre-submission checklist |

## Start here

```bash
bash "TANG I CHOI/w5_report_and_reproducibility/reproduce_all.sh"
```

Regenerates W3 metrics, the W3 offline/in-stream gap, the W4 ablation and
statistics, then the W5 figures and tables. Steps whose artefacts already exist
are skipped unless `FORCE_TRAIN=1`.

If you only want the numbers: `w5_report_and_reproducibility/report/results_tables.md`
(tables) and `figures/*.png`. If you want the argument:
`report/results_section.md` (draft) and `w4_gate_ablations/docs/w4_gate_ablation_report.md`.

## Results in one table

| what | number |
|---|---|
| W3 gate accuracy, offline (context + full history) | 0.877 (macro-F1 0.843, n=835) |
| W3 gate accuracy, same model in-stream (`context=None`) | 0.613 for the rules, 0.424 for the learned gate |
| W3 gate accuracy, cold start | 0.408 / 0.340 (McNemar p=0.0008 against the learned gate) |
| W4 end-to-end retention | 0.110 (`always_write`) vs 1.000 for the three memory-using arms (p=1.0 between them) |
| W4 gate accuracy vs gold action | 0.613 (rules) / 0.424 (learned) |
| human pilot (100 records) | learned 0.650, rules 0.670 |

The honest summary: **the gate does not affect retention in this setting, it
affects what gets written.** `always_write` is the only clearly harmful policy;
the learned gate silently degenerates to `always_route` because its context
features are empty inside the stream.

## Boundaries respected

- Only `TANG I CHOI/` is written. `calm/`, `EasyEdit/`, `FONG HIO LAM/` and
  `output/` are read-only for these scripts.
- The ablation injects its arms through `hparams.conflict_gate_path` and turns
  the CALMEditor guards off on the instance, so re-running cannot dirty the
  shared editor.
- Every number in the documents is generated from a JSON artefact by a script in
  the same folder; nothing is hand-copied.
- No absolute local paths, no model weights, no caches in the tree.
