# TANG I CHOI -- W0-W6 (MC631517)

Owned work packages of Group 26's CALM-Editor Task 1. All six of my packages are
indexed here. W1/W2 artefacts physically live outside this folder (annotation
manual, `docs/w2_*`, the IKE/GRACE notebook), because they predate the
per-member layout; `w1_w2_prework_index.md` maps them, and they were left in
place rather than relocated so that links from the root `README.md` and from
other members' documents keep working.

## Map

| folder / file | package | one-line summary |
|---|---|---|
| `w0_bibliography_verification.md` | W0 | the 12 papers, title-verified against arXiv; 4 wrong identifiers caught and corrected; open-access status; where each paper is used |
| `w1_w2_prework_index.md` | W1, W2 | index of the annotation manual + 100-edit pilot (W1) and the IKE/GRACE reproduction (W2), with owner evidence and the known gap in the label scheme |
| `w3_conflict_gate_training_and_classifier/` | W3 | build the gate training data from CounterFact in the W1 four-label scheme, train the write/route/reject classifier, and measure how it degrades offline -> in-stream |
| `w4_gate_ablations/` | W4 | four gate arms (`always_write` / `always_route` / `rule` / `learned`) over one fixed stream, plus paired statistics and a per-conflict-type breakdown; `docs/w4_immediate_rewrite_diagnosis.md` is the defect-1 diagnosis and `results/w4_ablation_table_deduped.md` the 40-prompt rerun that isolates it |
| `w5_report_and_reproducibility/` | W5 | three results figures, the assembled results tables, the drop-in results section, the reproducibility appendix, and `reproduce_all.sh` + run logs |
| `w6_cross_review/` | W6 | review findings for the other members' artefacts (filed, not applied), the pre-submission checklist, and `docs/w0_w6_completeness_audit.md` |

## Start here

```bash
bash "TANG I CHOI/w5_report_and_reproducibility/reproduce_all.sh"
```

Regenerates W3 metrics, the W3 offline/in-stream gap, the W4 ablation and
statistics, then the W5 figures and tables. Steps whose artefacts already exist
are skipped unless `FORCE_TRAIN=1`.

If you want to know whether every promised deliverable of W0-W6 is in the
repository and which file carries it: `w6_cross_review/docs/w0_w6_completeness_audit.md`
(matrix of week x member, reproduction status of every table, the four gaps left
open on purpose).

If you only want the numbers: `w5_report_and_reproducibility/report/results_tables.md`
(tables) and `figures/*.png`. If you want the argument:
`report/results_section.md` (draft) and `w4_gate_ablations/docs/w4_gate_ablation_report.md`.

## Results in one table

| what | number |
|---|---|
| W3 gate accuracy, offline (context + full history) | 0.877 (macro-F1 0.843, n=835) |
| W3 gate accuracy, same model in-stream (`context=None`) | 0.613 for the rules, 0.424 for the learned gate |
| W3 gate accuracy, cold start | 0.408 / 0.340 (McNemar p=0.0008 against the learned gate) |
| W4 end-to-end retention | 0.110 (`always_write`) vs 1.000 for the three memory-using arms (p=1.0 between them) -- see defect 1, 0.110 is not a writer measurement |
| W4 gate accuracy vs gold action | 0.613 (rules) / 0.424 (learned) |
| W4 learned gate, in-stream behaviour | 1 WRITE / 190 ROUTE / 0 REJECT, agreement 0.424, kappa 0.009 vs 0.343 for the rules, McNemar p<0.0001 -> it is `always_route` in disguise |
| W4 `pre-training-conflict` slice (n=26) | decision-correct 0.000 in **all four** arms; the gold action is REJECT and no arm ever rejects |
| W4 immediate rewrite, after the metric fix | 0.000 (deduped stream: 40 edits, 13/40 are base-model hits; all four arms report 0.325 before correction) |
| human pilot (100 records) | learned 0.650, rules 0.670 |

The honest summary: **the gate does not affect retention in this setting, it
affects what gets written.** `always_write` is the only clearly harmful policy;
the learned gate silently degenerates to `always_route` because its context
features are empty inside the stream.

## Three defects this revision answers

The W3-W6 numbers above are unchanged. What changed is that the three places
where a reader could take them for the editor's behaviour are now separated from
the places where they are artefacts of the measurement or of the label scheme.

| # | defect | where the answer is |
|---|---|---|
| 1 | immediate-rewrite / model-only retention was constant across the four arms (0.110 on the legacy stream), which would read as "the gate decides everything and the writer does nothing". Cause: the metric compares the post-write answer with the **base model's** answer, so it can only fire on edits whose subject the untouched model already answers -- and the legacy stream had 191 rows over 40 distinct prompts, so repeated prompts re-weighted that base rate. On a deduped stream all four arms report the same 0.325, equal to the 13/40 base hit rate; writer-attributable rewrite after correction is 0.000. The hypernetwork additionally has no checkpoint (`hypernet_ckpt: null`), so the delta it emits is an untrained direction: it is applied (delta norm > 0) and never moves greedy decoding. | `w4_gate_ablations/docs/w4_immediate_rewrite_diagnosis.md`, `docs/w4_gate_ablation_report.md` §3.4, `w5_report_and_reproducibility/report/results_section.md` §4.2.2 |
| 2 | the learned gate's in-stream degeneration was reported as a number without a cause | `w4_gate_ablations/docs/w4_gate_ablation_report.md` §3.5, `report/results_section.md` §4.2.1 (dead context block / covariate shift / late-arriving history) |
| 3 | `pre-training-conflict` sat in the training data and in the pilot but was missing from the per-type analysis, where its decision-correct is 0.000 for every arm | `docs/w4_gate_ablation_report.md` §3.3.1, `report/results_section.md` §4.2.3, `w1_w2_prework_index.md` (label coverage and its gap) |


## Boundaries respected

- Only `TANG I CHOI/` is written. `calm/`, `EasyEdit/`, `FONG HIO LAM/` and
  `output/` are read-only for these scripts.
- The ablation injects its arms through `hparams.conflict_gate_path` and turns
  the CALMEditor guards off on the instance, so re-running cannot dirty the
  shared editor.
- Every number in the documents is generated from a JSON artefact by a script in
  the same folder; nothing is hand-copied.
- No absolute local paths, no model weights, no caches in the tree.
