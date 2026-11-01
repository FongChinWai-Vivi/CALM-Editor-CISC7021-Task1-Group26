# W6 -- Cross-review findings (Tang I Choi, MC631517)

Scope: review of the artefacts owned by the other members plus my own, done by
reading the code and re-running the scripts, not by re-reading the reports.
Nothing outside `TANG I CHOI/` was edited: every item below is filed as a
finding for its owner to accept or reject.

Method:

1. re-ran every W3 -> W5 script individually (the driver is the same list of
   commands) and diffed the outputs;
2. read `EasyEdit/easyeditor/editors/calm_editor.py`, `calm/calm_editor.py`,
   `FONG HIO LAM/side_memory_router/{conflict_gate,side_memory}.py`;
3. cross-checked every number in the draft results section against
   `output/*.json` and my own `results/*.json`;
4. checked what is actually tracked by git (`git ls-files`, largest blobs).

Severity: **B** = blocks a claim in the report, **M** = must fix before
submission, **m** = nice to have.

---

## B1. `retrieval_context` is hard-coded to `None` in the editor

Evidence: `EasyEdit/easyeditor/editors/calm_editor.py:193` passes
`retrieval_context=None` on every `edit_requests` call, while
`ConflictGate.decide` (`FONG HIO LAM/side_memory_router/conflict_gate.py:168`)
already accepts the argument.

Impact (measured, not assumed): four of the nineteen gate features are constant
zero inside the stream, and the W3 classifier's strongest `ROUTE` weights are
exactly `has_context` / `n_prior_edits`. Re-scoring the same classifier under
stream conditions drops it from 0.877 to 0.340
(`TANG I CHOI/w3_conflict_gate_training_and_classifier/results/gate_online_gap.json`),
and in the W4 ablation the learned gate degenerates to 1 WRITE / 190 ROUTE
(`TANG I CHOI/w4_gate_ablations/docs/w4_gate_ablation_report.md`).

Recommendation: either pass retrieved snippets through the request payload, or
have the gate retrieve them itself. Until then the report must not claim that
the system detects context conflicts at runtime; the W1 context branch is
exercised offline only.

Owner: editor (W1/W2 owner).

## B2. The "frozen W1 interface" and the shipped editor have diverged

Evidence: `calm/calm_editor.py` (342 lines) declares
`decide(self, request: EditRequest, memory_snapshot)` plus `score` / `write`;
`EasyEdit/easyeditor/editors/calm_editor.py` (358 lines) instead calls
`self.gate.decide(subject=..., relation=..., ...)` and adds
`_first_write_guard` / `_write_budget_guard`. The file header still says it
implements the frozen W1 interface blueprint.

Impact: a reader (or the marker) who opens `calm/` to understand the interface
will read a different contract from the one that runs. Two `ConflictGate`
implementations now exist with different `decide` signatures.

Recommendation: keep one source of truth. Cheapest option: move the guards'
documentation into `EasyEdit/easyeditor/editors/calm_editor.py`, and mark
`calm/calm_editor.py` as the W1 design draft (or delete it) so the report can
cite a single implementation.

Owner: editor owner, with W1 owner's agreement.

## M1. `output/` has no index, and the two 5K files are not self-describing

Evidence: `output/` holds six files, no README. `w4_sequential_5k_results.json`
(baseline vs trained hypernet) and `w4_calm_fix_5k_results.json` (guards on)
differ only by file name; neither states which one supersedes the other.

Recommendation: add `output/README.md` with one line per file (what was run,
which machine, which backbone, whether it is superseded). Then the report can
point at one file per claim.

Owner: whoever produced each run.

## M2. Backbones differ between the team runs and the ablation

Evidence: `output/w3_calm_stream_1k.json` and `output/w4_sequential_5k_results.json`
were produced with `gpt2-small` on Windows; the W4 ablation used `gpt2-xl` on
Apple Silicon.

Impact: Table 1 and Table 2 in the draft results section are not comparable in
absolute terms. I labelled them, but the final report must repeat the caveat
wherever the two tables sit next to each other.

Recommendation: either rerun the ablation at `gpt2-small` for one comparable
column, or keep the two tables explicitly separate ("stream behaviour" vs
"gate ablation") and say so in the caption.

## M3. Machine-inserted AIGC front matter in tracked Markdown

Evidence: `docs/w2_ike_grace_reproduction.md` and
`FONG HIO LAM/side_memory_router/README.md` both start with an
`AIGC: / ContentProducer: ... / ReservedCode1: <base64>` block, and the W2 doc
repeats its closing `*（内容由AI生成，仅供参考）*` line twice.

Impact: it is noise in a deliverable that is supposed to read as engineering
documentation, and the duplicated line looks like a bad merge.

Recommendation: strip those blocks before submission. I did not touch them
because both files belong to other work packages.

## M4. Gate thresholds come from two different places

Evidence: the rule arm and the learned arm both receive
`write_threshold / route_threshold / similarity_threshold` from
`hparams/CALM/*.yaml` via `ConflictGate.__init__`, but the learned classifier
does not use them at all (`learned_gate.py:_decide` takes `predict` directly).

Impact: a reader may think the four arms share one threshold policy. They do
not.

Recommendation: one sentence in the ablation write-up; already stated in
`w4_gate_ablation_report.md` section 2, worth copying into the report.

## m1. Locality is saturated and cannot discriminate

All four arms score 1.000 because almost nothing is written to the weights.
Keep the metric, but say in the report that it only becomes informative once
the parametric channel actually writes (guards on, larger hypernet scale).

## m2. Model checkpoints stay out of git (verified, no action)

`git ls-files` largest blobs are upstream `EasyEdit/figs/*` (a 12 MB gif and
friends); `hugging_cache/` and `*.pt` are ignored. No action needed, recorded so
the next person does not re-check.

---

## Reproducibility check performed in this review

| step | result |
|---|---|
| `reproduce_all.sh` shell syntax + step wiring (`bash -n`, dry inspection) | PASS; every step was then executed individually (rows below), the driver itself was not run end-to-end in one go |
| `run_w4_gate_ablation.py --all-arms --subjects 40` | PASS, 4 arms, 0.53-0.54 s/edit, log in `w5_report_and_reproducibility/run_logs/` |
| `analyze_w4_statistics.py` on the four result JSONs | PASS, paired tests reproduce the table |
| `make_w5_figures.py` / `collect_w5_tables.py` | PASS, all inputs found from JSON artefacts |
| numbers in `report/results_section.md` vs artefacts | consistent (spot-checked every table cell) |

## My own files, for the reviewer

- I only wrote inside `TANG I CHOI/` (plus my earlier W1/W2 deliverables).
  `calm/`, `EasyEdit/`, `FONG HIO LAM/` and `output/` are untouched.
- Every number in my documents is generated from a JSON artefact by a script in
  the same folder; no hand-copied metrics.
- The ablation injects its arms through `hparams.conflict_gate_path` and
  disables the CALMEditor guards on the instance, so re-running cannot dirty
  the shared editor.
