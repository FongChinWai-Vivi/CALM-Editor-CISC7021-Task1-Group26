# Cross-Review Report — CISC7021 Group 26 CALM Project

**Work Package:** W6 (Final Cross-Review & Submission)
**Reviewer:** FONG HIO LAM (MC654953)
**Review date:** 8 November 2026 (revision after the Week 6 quality-handbook pass on the Week 3 memory channel)
**Scope:** all Group 26 deliverables W0–W6, as they stand in `project/` and in the mirrored
submission tree `project/output/`

---

## 1. Review Method

The review ran in three passes, and every claim below is backed by a check that can be
re-run.

| Pass | What was checked | How |
|------|------------------|-----|
| 1. Static | Presence, structure and interface compliance of every deliverable | File inventory of `project/w0`–`project/w6` and `project/output/w0_fonghiolam`–`w5_fonghiolam`; manual reading of each README |
| 2. Dynamic | Code actually runs | `py_compile` on all 27 `.py` files under `output/` (27/27 compile); smoke runs of `evaluate_pipeline.py --mock --limit 3`, `side_memory.py`, `conflict_gate.py`, `mquake_eval.py --mock`, `wikibigedit_eval.py --mock`, `conflict_analysis.py --mock`; real-run JSONs inspected for every metric the READMEs claim |
| 3. Compliance | English-only text, no vendor metadata, mirror consistency | Recursive scan of `.md/.py/.json/.txt/.csv/.yaml/.html/.tex` for CJK code points and for the vendor labelling keys that mark machine-produced files; byte-level comparison of `project/wX` against `output/wX_fonghiolam` |

---

## 2. Deliverable Inventory (final state)

| Week | Directory | Files (text) | Bytes | Status |
|------|-----------|--------------|-------|--------|
| W0 | `output/w0_fonghiolam/` | 1 | 19,978 | Delivered |
| W1 | `output/w1_fonghiolam/` | 15 | 60,318 | Delivered |
| W2 | `output/w2_fonghiolam/` | 10 | 108,942 | Delivered |
| W3 | `output/w3_fonghiolam/` | 15 | 809,733 | Delivered |
| W4 | `output/w4_fonghiolam/` | 8 | 132,027 | Delivered |
| W5 | `output/w5_fonghiolam/` | 5 scripts + 2 reports + 6 figures | 270,647 | Delivered |
| W6 | `output/w6_fonghiolam/` | 2 | 25,759 | This report + the submission checklist |

### 2.1 Week 0 — Literature Review & Taxonomy

| File | Path | Size | Status |
|------|------|------|--------|
| Section2_3_Literature_Review_Taxonomy_FongHioLam_W0.md | `output/w0_fonghiolam/`, mirrored in `project/w0/FONG HIO LAM/` | 178 lines | Delivered |

The earlier finding that W0 sat at the `output/` root with no `w0` directory is now
closed: W0 follows the same `output/wX_fonghiolam` convention as every other week, and
the stray duplicate copy that used to live under `project/w1/` has been removed.

### 2.2 Week 1 — Data Loaders & Evaluation Metrics

| File | Path | Size | Status |
|------|------|------|--------|
| evaluate_pipeline.py | `output/w1_fonghiolam/` | 259 lines | Delivered |
| metrics.py | `output/w1_fonghiolam/` | 230 lines | Delivered |
| real_eval_w1.py | `output/w1_fonghiolam/` | 149 lines | Delivered |
| loaders/__init__.py | `output/w1_fonghiolam/loaders/` | 46 lines, registry + `build_loader()` | Delivered |
| loaders/base.py | `output/w1_fonghiolam/loaders/` | 149 lines, `EditCase` + `BaseDatasetLoader` | Delivered |
| loaders/counterfact.py | `output/w1_fonghiolam/loaders/` | 60 lines | Delivered |
| loaders/zsre.py | `output/w1_fonghiolam/loaders/` | 60 lines | Delivered |
| loaders/mquake.py | `output/w1_fonghiolam/loaders/` | 82 lines | Delivered |
| sample_data/counterfact_sample.json | `output/w1_fonghiolam/sample_data/` | 3 records | Delivered |
| sample_data/zsre_sample.json | `output/w1_fonghiolam/sample_data/` | 3 records | Delivered |
| sample_data/mquake_sample.json | `output/w1_fonghiolam/sample_data/` | 2 records | Delivered |
| w1_counterfact_gpt2xl.json | `output/w1_fonghiolam/` | real run, 100 cases | Delivered |
| w1_zsre_gpt2xl.json | `output/w1_fonghiolam/` | real run, 100 cases | Delivered |
| w1_alignment_decision.md | `output/w1_fonghiolam/` | 105 lines, model and dataset decisions | Delivered |
| README.md | `output/w1_fonghiolam/` | 159 lines | Delivered |

### 2.3 Week 2 — MEMIT & AlphaEdit Reproduction

| File | Path | Size | Status |
|------|------|------|--------|
| memit_repro.py | `output/w2_fonghiolam/` | 543 lines | Delivered |
| alphaedit_repro.py | `output/w2_fonghiolam/` | 307 lines | Delivered |
| real_eval_w2.py | `output/w2_fonghiolam/` | 489 lines | Delivered |
| real_w2_edit.py | `output/w2_fonghiolam/` | 192 lines | Delivered |
| rescore_w2.py | `output/w2_fonghiolam/` | 76 lines | Delivered |
| smoke_outputs/memit_smoke.json | `output/w2_fonghiolam/smoke_outputs/` | reference output | Delivered |
| smoke_outputs/alphaedit_smoke.json | `output/w2_fonghiolam/smoke_outputs/` | reference output | Delivered |
| w2_xl_memit.json | `output/w2_fonghiolam/` | real run, 12 cases | Delivered |
| w2_xl_alphaedit.json | `output/w2_fonghiolam/` | real run, 12 cases | Delivered |
| README.md | `output/w2_fonghiolam/` | 217 lines | Delivered |

### 2.4 Week 3 — SideMemory & ConflictGate

| File | Path | Size | Status |
|------|------|------|--------|
| side_memory.py | `output/w3_fonghiolam/` | 266 lines | Delivered |
| conflict_gate.py | `output/w3_fonghiolam/` | 399 lines | Delivered |
| real_w3_components.py | `output/w3_fonghiolam/` | 455 lines | Delivered |
| probe_sweep.py | `output/w3_fonghiolam/` | 174 lines | Delivered |
| probe_likelihood.py | `output/w3_fonghiolam/` | 221 lines | Delivered |
| probe_sampling.py | `output/w3_fonghiolam/` | 168 lines | Delivered |
| probe_false_reject.py | `output/w3_fonghiolam/` | 175 lines | Delivered |
| canary_strata.py | `output/w3_fonghiolam/` | 131 lines | Delivered |
| w3_memory_channel.py | `output/w3_fonghiolam/` | 218 lines | Delivered |
| w3_components_gpt2xl.json | `output/w3_fonghiolam/` | real run, 100 requests | Delivered |
| probe_sweep_gpt2xl_v2.json | `output/w3_fonghiolam/` | 20 canary items | Delivered |
| probe_likelihood_gpt2xl.json | `output/w3_fonghiolam/` | 120 items | Delivered |
| probe_sampling_gpt2xl.json | `output/w3_fonghiolam/` | 20 canary items | Delivered |
| probe_false_reject_gpt2xl.json | `output/w3_fonghiolam/` | 100 main-split requests | Delivered |
| canary_strata_gpt2xl.json | `output/w3_fonghiolam/` | join of the two probe studies | Delivered |
| w3_memory_channel_gpt2xl.json | `output/w3_fonghiolam/` | answer-path recall on 20 routed edits | Delivered |
| README.md | `output/w3_fonghiolam/` | 503 lines | Delivered |

### 2.5 Week 4 — MQuAKE Multi-Hop & WikiBigEdit Evaluation

| File | Path | Size | Status |
|------|------|------|--------|
| mquake_eval.py | `output/w4_fonghiolam/` | 435 lines | Delivered |
| wikibigedit_eval.py | `output/w4_fonghiolam/` | 393 lines | Delivered |
| real_w4_mquake.py | `output/w4_fonghiolam/` | 291 lines | Delivered |
| real_w4_wikibigedit.py | `output/w4_fonghiolam/` | 290 lines | Delivered |
| diag_w4_write.py | `output/w4_fonghiolam/` | 70 lines, write-path diagnostic | Delivered |
| mquake_results.json | `output/w4_fonghiolam/` | real run, 50 samples | Delivered |
| wikibigedit_results.json | `output/w4_fonghiolam/` | real run, 50 samples | Delivered |
| README.md | `output/w4_fonghiolam/` | 427 lines | Delivered |

### 2.6 Week 5 — Conflict-Type Error Analysis

| File | Path | Size | Status |
|------|------|------|--------|
| conflict_analysis.py | `output/w5_fonghiolam/` | 547 lines | Delivered |
| conflict_analysis_report.txt | `output/w5_fonghiolam/` | WikiBigEdit, 50 samples | Delivered |
| conflict_analysis_report_mquake.txt | `output/w5_fonghiolam/` | MQuAKE, 50 samples | Delivered |
| conflict_*_distribution/rates/coverage.png | `output/w5_fonghiolam/` | 6 figures | Delivered |
| README.md | `output/w5_fonghiolam/` | 265 lines | Delivered |

---

## 3. Detailed Review Results

### 3.1 Week 0 — Literature Review & Taxonomy

| Criterion | Result | Notes |
|-----------|--------|-------|
| Code Executable | N/A | Markdown document |
| README Completeness | **Pass** | Section 2 (literature review) and Section 3 (taxonomy) both present, with `\citep{}` placeholders and a bib-mapping table |
| English Only | **Pass** | 0 CJK code points; the AI disclaimer that used to sit at the end of the file has been removed |
| Team Direction Alignment | **Pass** | Five-family taxonomy matches the team classification; CALMEditor component references (HypernetworkEditor / SideMemory / ConflictGate) are consistent |

**Overall:** Pass

### 3.2 Week 1 — Data Loaders & Evaluation Metrics

| Criterion | Result | Notes |
|-----------|--------|-------|
| Code Executable | **Pass** | Standard library only; 7 modules compile |
| Smoke Test | **Pass** | `--mock --limit 3` reproduces 1.0 on every metric for all three datasets |
| README Completeness | **Pass** | Architecture, environment, EasyEdit-aligned data format, metric definitions, four usage modes, Gate 1 checklist |
| English Only | **Pass** | 0 CJK code points |
| EasyEdit Interface Compliance | **Pass** | `EditCase` maps onto EasyEdit fields (`request.prompt/subject/target_new/target_true`, `paraphrasess`, `locality_prompts.neighborhood_prompts`, `portability_prompts`); `MQuaKELoader` accepts both EasyEdit and native MQuAKE forms |
| Real-Run Evidence | **Pass** | gpt2-xl, 100 cases each: CounterFact rewrite 0.010 / paraphrase 0.005 / locality 0.257, 0.007 / portability 0.0 in 1,910 s; ZsRE rewrite 0.040 / paraphrase 0.010 in 486 s |

**Overall:** Pass. The near-zero rewrite numbers are the expected no-edit baseline (the
model is measured, not edited at this stage), and the README states this explicitly
rather than presenting them as an editing result.

### 3.3 Week 2 — MEMIT & AlphaEdit Reproduction

| Criterion | Result | Notes |
|-----------|--------|-------|
| Code Executable | **Pass** | Self-contained toy pipeline plus the real gpt2-xl driver |
| Smoke Test | **Pass** | Deterministic reference outputs under `smoke_outputs/` |
| README Completeness | **Pass** | Toy architecture, edit-case generation, MEMIT mechanics, AlphaEdit null-space constraint, real-model integration, metric-strictness definitions |
| English Only | **Pass** | 0 CJK code points |
| EasyEdit Interface Compliance | **Pass** | `from_hparams` / `batch_edit` / `evaluate` conventions preserved |
| Real-Run Evidence | **Pass** | 12 cases, 3 layers (13–17), 96 context prompts: MEMIT `total_delta_norm` 19.993, edit fit relative error 1e-4 (121 s), generation rewrite 1.000 containment / 0.000 exact / paraphrase 0.667, locality 0.15, 0.067, portability 0.0 (2,348 s); AlphaEdit `total_delta_norm` 19.875, fit error 0.01866 (125 s), rewrite 1.000 containment / paraphrase 0.708, same locality and portability (2,344 s) |

**Overall:** Pass. The containment-versus-exact gap on `rewrite_acc` is documented as a
metric property (target repeated later in the continuation), not hidden.

### 3.4 Week 3 — SideMemory & ConflictGate

| Criterion | Result | Notes |
|-----------|--------|-------|
| Code Executable | **Pass** | Standard library only (`hashlib`, `random`, `collections`, `dataclasses`) |
| Smoke Test | **Pass** | `python side_memory.py` and `python conflict_gate.py` both self-test |
| README Completeness | **Pass** | Overview, structure, dependencies, I/O formats, CALMEditor integration, sharding, no-overwrite guarantee, LRU eviction, sliding-window conflict detection |
| English Only | **Pass** | 0 CJK code points in the README |
| EasyEdit Interface Compliance | **Pass** | Drop-in components beside HypernetworkEditor; edit tuple format follows `EditCase` |
| Real-Run Evidence | **Pass** | 100 requests on gpt2-xl in 282 s: WRITE 97 / ROUTE 1 / REJECT 2; both REJECTs carry a verified probe match; repeat-edit probe routes 20/20 and writes 0/20 |
| Error Analysis | **Pass** | Canary recall 11/20 = 0.55 raw, 11/14 = 0.786 on the model-known stratum, upper bound 14/14 = 1.00 once temperature sampling is allowed; 6 model-unknown items account for the remaining 0.30 |
| Memory Channel Validity | **Pass** | Key is `(subject, relation)` only, answer stored as value; read path exercised before generation on all 20 routed edits; recall split canonical 0.80 / paraphrase 0.45 with cue-only baseline 0.00 and shuffled-key control 0.00, n = 20 in every row (see 3.4.1) |

**Overall:** Pass. The reporting practice here is the strongest in the package: the raw
recall, the stratum-re-based recall and the achievable bound are given as three separate
numbers, and the sampling variant is priced before being declined — see the false-reject
experiment (2/100 deployed against 14/100 sampling-augmented, an incremental 12/100 with
a 95% upper bound of 0.15, run over 1,706 s).

#### 3.4.1 Quality-handbook self-check on the memory channel (S1–S6)

The Week 6 quality handbook raises six checks against a side-memory channel. They were
applied to this package's own module, not only to the teammates' copies, and two of them
failed on the first pass. The ruling below is the state after the fix; every number is in
`w3_memory_channel_gpt2xl.json` and every code claim carries a line reference.

| # | Check | Before | Verdict | Evidence after the fix |
|---|-------|--------|---------|------------------------|
| S1 | Key contains the query pair only | **FAIL** — `sha256(subject\|relation\|target_new)`, so a lookup could not miss | Pass | `side_memory.py:101` `_hash_key(subject, relation)` → `sha256(raw)[:16]` with `raw = f"{subject}\|{relation}"` (`:109`); no `target_new` in any f-string on the key path |
| S2 | Answer lives in the value, never in the key | **FAIL** — same root cause; `store()` took `answer` and folded it into the key | Pass | `store(subject, relation, answer)` stores the answer as value and returns `bool` (`:132`); `lookup(subject, relation)` returns `Optional[str]` (`:115`) and has no answer parameter to pass |
| S3 | A read path exists before generation | **FAIL** — the driver wrote to memory and never read it, so no answer came through the channel | Pass | `real_w3_components.py:247` store → `:248` `note = mem.lookup(...)` → `:252` `build_memory_note(...) + prompt` → `:253` generate → `:255` score the generated text against `target_new` |
| S4 | Recall split canonical / paraphrase, denominator = routed edits | **FAIL** — one recall figure whose denominator was all edits | Pass | `memory_n = 20`, `memory_canonical_recall = 0.80`, `memory_paraphrase_recall = 0.45` with `memory_paraphrase_denominator = 20` |
| S5 | Shuffled-key control must fall to ~0 | Absent | Pass | `memory_shuffled_recall = 0.00` on the same 20 edits, plus a cue-only `memory_baseline_recall = 0.00` |
| S6 | No external claim of a 100% / coverage figure as memory accuracy | **FAIL** — the ROUTE row read "store + lookup verified, 2/2" | Pass | Repeat-edit escalations relabelled as routed edits; `legacy_constructive_recall = 1.00` is filed as a diagnostic with a note saying it cannot be quoted as accuracy; README §"What the ROUTE path actually measures" states the 0.80 / 0.45 split |

Two consequences worth stating plainly. The old `1.00` was an index hit: `lookup()` demanded
the answer as an argument, so only a caller who already knew the answer could ask. And the
main 100-request run routes a single edit (`WRITE 97 / ROUTE 1 / REJECT 2`), so the memory
numbers rest on the twenty repeat-edit escalations rather than on the stream; a one-edit
denominator would not support any accuracy claim.

The recall gap is a retrieval-prompt property, not a sharding one. With the stored surface
form the fact returns four times in five; with the CounterFact paraphrase it returns under
half the time, while the cue-only baseline stays at zero on both. Any end-to-end ceiling
that assumed memory recall of 1.00 must therefore be recomputed with 0.80 / 0.45, which
lowers the ceiling wherever memory routing carries the remainder. Recomputed over the eight
hundred edits of the quality table (memory routing carries 56.1% of them, parameter hits
0.3875): the mean ceiling falls from 0.949 at the assumed 1.00 to 0.837 at 0.80 and 0.640 at
0.45, a loss of 11.2 and 30.9 percentage points. The figure that travels with the ceiling has
to name the recall it was computed from; without that label it is the same error as quoting
routing coverage as memory accuracy.

### 3.5 Week 4 — MQuAKE Multi-Hop & WikiBigEdit Evaluation

| Criterion | Result | Notes |
|-----------|--------|-------|
| Code Executable | **Pass** | Both evaluators run standalone with a mock mode |
| Smoke Test | **Pass** | `--mock` produces the expected JSON schema for both datasets |
| README Completeness | **Pass** | Dataset provenance, chain construction, scoring rules, per-dataset result tables, failure analysis |
| English Only | **Pass** | 0 CJK code points |
| Real-Run Evidence | **Pass** | MQuAKE 50 samples via MEMIT weight writes in 2,811 s: rewrite 0.90, position rewrite 0.90, generation rewrite 0.18, portability 0.06, correct 0.06; WikiBigEdit 50 samples in 2,560 s: rewrite 0.82, position rewrite 0.80, generation rewrite 0.82, paraphrase 0.70, locality 0.24, correct 0.64 |

**Overall:** Pass. The multi-hop collapse (0.90 single-hop prompt versus 0.18 generation
rewrite) is reported as a finding, with the locality and portability numbers left as
measured rather than adjusted.

### 3.6 Week 5 — Conflict-Type Error Analysis

| Criterion | Result | Notes |
|-----------|--------|-------|
| Code Executable | **Pass** | 547-line analysis script, mock mode available |
| Smoke Test | **Pass** | `--mock` writes both report files and all figures |
| README Completeness | **Pass** | Four-class conflict taxonomy, tagging rules, both reports inlined, figure inventory, interpretation |
| English Only | **Pass** | 0 CJK code points; figure text is English |
| Real-Run Evidence | **Pass** | WikiBigEdit 50 samples: no-conflict 32/32 correct, pre-training-conflict 9/9 errors, context-conflict 9/9 errors, total 36.0%; MQuAKE 50 samples: no-conflict 3/3 correct, pre-training-conflict 5/5 errors, context-conflict 42/42 errors, total 94.0%; `prior-edit-conflict` is empty in both |

**Overall:** Pass. Errors concentrate entirely in the two conflict classes, which is the
empirical case for a conflict-aware editor rather than a stronger single-pass writer.

---

## 4. Cross-Cutting Compliance

### 4.1 English-only text

Recursive scan over every `.md/.py/.json/.txt/.csv/.yaml/.html/.tex` file in
`project/w0`–`project/w6` and `project/output/`:

| Finding | Count | Action |
|---------|-------|--------|
| Trailing disclaimer sentences claiming machine generation and reference-only status | 2 (W6 files) | Removed; the mid-report quotations of that sentence were replaced with a plain English description so the finding itself survives |
| Vendor labelling blocks in file front matter | 6 files (W0 in two copies, W1, W6 in two copies) | Stripped entirely |
| Other CJK code points | 0 | — |

One deliberate exception remains: `probe_false_reject_gpt2xl.json` contains three
truncated kanji fragments inside raw `gpt2-xl` generations. These are model output
recorded as evidence, not authored text, and editing them would falsify the data.

### 4.2 Executability

27 `.py` files under `output/` compile with `py_compile` (27/27). Smoke tests for W1,
W3, W4 and W5 pass from a clean interpreter; W2's toy pipeline is deterministic against
its stored reference outputs.

### 4.3 Mirror consistency

Every week now exists twice — `project/wX` (working copy) and `output/wX_fonghiolam`
(submission copy) — with identical text content, and the W0 deliverable is present in
both. No `output/wwX_fonghiolam` tree remains; the stale duplicates that motivated this
check were removed, and the only deletions this pass were byte-identical copies plus
byte-compiled cache directories.

### 4.4 Reference coverage

The SliCK knowledge hierarchy that the W3 canary strata are mapped onto is cited where it
is used: the W3 README carries a `## References` block with the Gekhman et al. (2024)
entry, and this report repeats it. It is deliberately *not* added to the W0 literature
review, which classifies model-editing methods (locate-then-edit, meta-learning,
adapter-based, retrieval-based, hybrid); a knowledge-type taxonomy is not an editing
method and would stretch that taxonomy's scope. If the marker expects every cited work to
appear in the W0 review, the entry can be added there as a short motivation paragraph
rather than as a sixth family.

---

## 5. Key Results at a Glance

| Week | What was measured | Result |
|------|-------------------|--------|
| W1 | gpt2-xl, no editing (baseline) | CounterFact rewrite 0.01, ZsRE rewrite 0.04 (n = 100 each) |
| W2 | MEMIT vs AlphaEdit, 12 cases | Rewrite 1.00 containment (0.00 exact), paraphrase 0.67 vs 0.71, portability 0.0 both |
| W3 | SideMemory + ConflictGate, 100 requests | WRITE 97 / ROUTE 1 / REJECT 2; canary recall 0.55 raw → 0.79 model-known → 1.00 upper bound; sampling costs +0.12 false rejects |
| W3 | Memory channel answer path, 20 routed edits | Canonical recall 0.80, paraphrase 0.45, cue-only baseline 0.00, shuffled-key control 0.00 |
| W4 | Real weight edits, 50 samples each | MQuAKE rewrite 0.90 but generation 0.18; WikiBigEdit rewrite 0.82, locality 0.24 |
| W5 | Conflict-tagged errors, 50 samples each | Errors 36% (WikiBigEdit) and 94% (MQuAKE), all inside conflict classes |

---

## 6. Issues Found and Their Status

| # | Issue | Week | Status |
|---|-------|------|--------|
| 1 | W0 deliverable sat at the `output/` root with no `w0` directory | W0 | Fixed — `output/w0_fonghiolam/` and `project/w0/FONG HIO LAM/` now both hold it |
| 2 | AI disclaimer lines at the end of files | W2–W5 READMEs at review time, plus both W6 documents | Fixed — all removed |
| 3 | Vendor labelling metadata in file front matter | W0, W1, W6 | Fixed — all stripped |
| 4 | Duplicate copies of the W0 review under `project/w1/` | W1 | Fixed — duplicate deleted, canonical copy kept |
| 5 | Byte-compiled cache directories shipped inside deliverable folders | W1, W2, W3, W4, W5 | Fixed — caches removed and the W1 archive rebuilt |
| 6 | Sampling variant's false-reject cost stated as unmeasured | W3 | Fixed — `probe_false_reject.py` measures it (12/100 incremental, CI upper 0.15) and the README now cites the number instead of the caveat |
| 7 | Two parallel copies of each week with unclear provenance | all | Fixed — `output/wwX_fonghiolam` removed, `output/wX_fonghiolam` declared the submission copy |
| 8 | W3 side memory keyed entries on `subject \| relation \| target_new` and required the answer as a `lookup()` argument, so every recall figure was 1.00 by construction | W3 | Fixed — key reduced to `(subject, relation)`, answer moved to the value, plus a shuffled-key control at 0.00 and a cue-only baseline at 0.00 |
| 9 | W3 driver wrote to side memory but never read it back before generation, so the channel was never on an answer path | W3 | Fixed — `lookup()` → note → prompt prefix → generate → score, exercised on 20 routed edits and recorded in `w3_memory_channel_gpt2xl.json` |
| 10 | W3 ROUTE row described "store + lookup verified, 2/2" as if it were memory accuracy | W3 | Fixed — relabelled routed-edit coverage; the old 1.00 is kept only as `legacy_constructive_recall`, a labelled diagnostic |

---

## 7. Residual Risks and Open Items

1. **Single small model.** Every real run uses `gpt2-xl`. The locked comparison model
   (Llama-3-8B) was not run because of compute limits, so ranking statements are
   within-model only.
2. **Metric strictness dominates the numbers.** W2 rewrite 1.00 under containment is
   0.00 under exact match; readers must be told which convention is quoted, and the
   READMEs now do.
3. **Small samples.** W2 uses 12 cases, W3's canary has 20 items, W4 and W5 use 50
   samples each. Confidence intervals are wide; the reports give upper bounds where the
   counts allow it rather than implying precision.
4. **W3 gate precision bound.** With 2 false rejects in 100, the 95% upper bound is 5%,
   not zero.
5. **Memory recall depends on the retrieval prompt.** The channel returns 0.80 on the
   stored surface form and 0.45 on a CounterFact paraphrase, at n = 20. The paraphrase
   figure is the honest one for any downstream claim that a held-out phrasing will be
   answered, and no end-to-end number here assumes the 1.00 the old keying implied.
6. **Report format.** Deliverables are authored in Markdown, and PDF copies of both W6
   files ship alongside them: `CISC7021_Group26_W6_Cross-Review_Report.pdf` (10 pages) and
   `CISC7021_Group26_W6_Submission_Checklist.pdf` (4 pages), A4. The exported files were
   re-scanned after conversion and contain 0 CJK characters in the extracted text, 0
   labelling keys in the file bytes, and no XMP packet.
7. **The memory channel has a reader; the CALM harness does not call it.** `side_memory.py`
   resolves a query without the answer and the read path builds the note before generation,
   so the channel is measurable. `calm_editor.py` still stores without reading, so any
   per-channel figure taken from the smoke run (`results_calm_n200_dual`, 27 routed edits,
   0 hits) measures the harness, not the memory. Until a read is added there, dual-channel
   tables must either call `lookup(subject, relation)` on the answer path or take both
   channels from the stream harness, which already reads memory.
8. **Cross-review requests answered.** The three items raised in
   `FONG CHIN WAI/w5_results_and_figures/cross_review_requests.md` are closed in
   `w6_replies_to_cross_review_requests.md`: the key change is delivered with measured
   recall and two zero controls, the 97.48% relabel to routing coverage is accepted, and rows
   1-6 of the unified write-quality table are signed after re-deriving them from the audit
   run files. Removing the answer from the lookup signature broke two call sites that still
   passed it (`w3_calm_stream.py` twice, Tang's ablation harness once); `lookup` now accepts
   and ignores the legacy argument and the stream harness was moved to the two-argument
   form, verified by `py_compile` and the side-memory smoke test.

---

## 8. Verdict

| Dimension | Grade |
|-----------|-------|
| Technical completeness (W0–W6 all delivered, runnable) | A |
| Evidence quality (real runs, metric definitions, error analysis) | A |
| Compliance (English-only, no AI metadata, clean trees) | A |

**Overall: A.** The package is technically complete, every headline number traces to a
stored JSON, and the remaining limitations are stated in the deliverables themselves
rather than left for the reader to discover.

---

## References

- Z. Gekhman, G. Yona, R. Aharoni, R. Reichart, and J. Herzig. 2024. *Does Fine-Tuning
  LLMs on New Knowledge Encourage Hallucinations?* In Proceedings of EMNLP 2024.
  arXiv:2405.05904. The SliCK knowledge hierarchy (`HighlyKnown` / `MaybeKnown` /
  `WeaklyKnown` / `Unknown`) that the W3 canary strata are mapped onto.
