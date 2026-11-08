# Submission Checklist — CISC7021 Group 26 CALM Project

**Owner:** FONG HIO LAM (MC654953)
**Package:** weekly deliverables W0–W6
**Last verified:** 8 November 2026

---

## 1. Pre-Submission Gate

Tick every box before packaging. Each row names the check and how it was run at the
stated date, so it can be reproduced.

| # | Check | Command / Method | Result |
|---|-------|------------------|--------|
| 1 | Every week has a submission directory | `ls project/output/w*_fonghiolam` | Pass — `w0`–`w6` all present |
| 2 | Working copy and submission copy agree | byte comparison of `project/wX` against `project/output/wX_fonghiolam` | Pass |
| 3 | All Python compiles | `py_compile` over all `.py` under `output/` | Pass — 27/27 |
| 4 | Smoke tests pass | W1 `--mock --limit 3`; W3 `side_memory.py`, `conflict_gate.py`; W4 `--mock`; W5 `--mock` | Pass |
| 5 | English only | recursive CJK scan of `.md/.py/.json/.txt/.csv/.yaml/.html/.tex` | Pass — 0 CJK in authored text; one retained model-output exception, see §5 |
| 6 | No vendor labelling metadata | scan for the machine-generation labelling keys used by document exporters | Pass — 0 hits |
| 7 | No disclaimer lines | scan for sentences claiming machine generation or reference-only status | Pass — 0 hits |
| 8 | No byte-compiled caches shipped | scan for cache directories | Pass — removed, W1 archive rebuilt |
| 9 | Every README quotes the metric convention it uses | manual read of W1–W5 READMEs | Pass |
| 10 | Git history clean and pushed | `git status`, `git log`, `git ls-remote` in `project_repo` | Pass — every W0–W6 commit is on `origin/main`, working tree clean |
| 11 | W3 memory channel meets the six Week-6 handbook checks | handbook S1–S6 against `side_memory.py`, `real_w3_components.py`, `w3_memory_channel_gpt2xl.json` | Pass — key is `(subject, relation)`, answer in the value, read path before generation, recall split 0.80 / 0.45 on n = 20, shuffled control 0.00, no 1.00 quoted as accuracy |

---

## 2. Per-Week File Manifest

### Week 0 — Literature Review & Taxonomy
- [x] `output/w0_fonghiolam/Section2_3_Literature_Review_Taxonomy_FongHioLam_W0.md` (178 lines)
- [x] Mirrored at `project/w0/FONG HIO LAM/Section2_3_Literature_Review_Taxonomy_FongHioLam_W0.md`
- [x] Reference list present, `\citep{}` keys mapped to the bibliography table

### Week 1 — Data Loaders & Evaluation Metrics
- [x] `evaluate_pipeline.py` (259 lines) — mock, pre-scored and real-model modes
- [x] `metrics.py` (230 lines) — comprehensive metric battery
- [x] `real_eval_w1.py` (149 lines)
- [x] `w1_alignment_decision.md` (105 lines) — model and dataset decisions
- [x] `loaders/` (`__init__`, `base`, `counterfact`, `zsre`, `mquake`)
- [x] `sample_data/` — CounterFact, ZsRE, MQuAKE samples
- [x] Real runs: `w1_counterfact_gpt2xl.json` (100 cases), `w1_zsre_gpt2xl.json` (100 cases)
- [x] `README.md` (159 lines)

### Week 2 — MEMIT & AlphaEdit
- [x] `memit_repro.py` (543 lines), `alphaedit_repro.py` (307 lines)
- [x] `real_eval_w2.py` (489 lines), `real_w2_edit.py` (192 lines), `rescore_w2.py` (76 lines)
- [x] `smoke_outputs/memit_smoke.json`, `smoke_outputs/alphaedit_smoke.json`
- [x] Real runs: `w2_xl_memit.json`, `w2_xl_alphaedit.json` (12 cases, 3 layers)
- [x] `README.md` (217 lines) — states exact vs containment vs early-window rewrite

### Week 3 — SideMemory & ConflictGate
- [x] `side_memory.py` (266 lines), `conflict_gate.py` (399 lines)
- [x] `real_w3_components.py` (455 lines) — 100-request run driver
- [x] `w3_memory_channel.py` (218 lines) — answer-path recall study, cue-only baseline included
- [x] `probe_sweep.py` (174 lines), `probe_likelihood.py` (221 lines), `probe_sampling.py` (168 lines)
- [x] `probe_false_reject.py` (175 lines) — false-reject cost of the sampling variant
- [x] `canary_strata.py` (131 lines) — stratum join
- [x] Real runs: `w3_components_gpt2xl.json`, `probe_sweep_gpt2xl_v2.json`,
      `probe_likelihood_gpt2xl.json`, `probe_sampling_gpt2xl.json`,
      `probe_false_reject_gpt2xl.json`, `canary_strata_gpt2xl.json`,
      `w3_memory_channel_gpt2xl.json`
- [x] `README.md` (503 lines) — three canary denominators, the sampling cost table, and the memory-channel controls
- [x] Mirrored to the team repository (`side_memory_router`)

### Week 4 — MQuAKE & WikiBigEdit
- [x] `mquake_eval.py` (435 lines), `wikibigedit_eval.py` (393 lines)
- [x] `real_w4_mquake.py` (291 lines), `real_w4_wikibigedit.py` (290 lines), `diag_w4_write.py` (70 lines)
- [x] Real runs: `mquake_results.json` (50 samples), `wikibigedit_results.json` (50 samples)
- [x] `README.md` (427 lines)

### Week 5 — Conflict-Type Error Analysis
- [x] `conflict_analysis.py` (547 lines)
- [x] `conflict_analysis_report.txt` (WikiBigEdit), `conflict_analysis_report_mquake.txt`
- [x] Six figures, all generated from the two result files, all text English
- [x] `README.md` (265 lines)

### Week 6 — Cross-Review & Submission
- [x] `Cross-Review_Report.md` — inventory, per-week review, compliance, open items
- [x] `Submission_Checklist.md` — this file

---

## 3. Numbers to Quote in the Presentation

| Metric | Value | Source file |
|--------|-------|-------------|
| W1 CounterFact rewrite (no editing) | 0.01 | `w1_counterfact_gpt2xl.json` |
| W2 MEMIT rewrite (containment / exact) | 1.00 / 0.00 | `w2_xl_memit.json` |
| W2 AlphaEdit paraphrase | 0.708 | `w2_xl_alphaedit.json` |
| W3 decision split (WRITE / ROUTE / REJECT) | 97 / 1 / 2 | `w3_components_gpt2xl.json` |
| W3 memory recall, canonical cue | 16/20 = 0.80 | `w3_memory_channel_gpt2xl.json` |
| W3 memory recall, paraphrase cue | 9/20 = 0.45 | `w3_memory_channel_gpt2xl.json` |
| W3 memory recall, no note (baseline) | 0/20 = 0.00 | `w3_memory_channel_gpt2xl.json` |
| W3 memory recall, shuffled key (control) | 0/20 = 0.00 | `w3_memory_channel_gpt2xl.json` |
| W3 canary recall, raw | 11/20 = 0.55 | `w3_components_gpt2xl.json` |
| W3 canary recall, model-known stratum | 11/14 = 0.786 | `canary_strata_gpt2xl.json` |
| W3 canary recall, sampling upper bound | 14/14 = 1.00 | `canary_strata_gpt2xl.json` |
| W3 false rejects, deployed gate | 2/100 = 0.02 | `probe_false_reject_gpt2xl.json` |
| W3 false rejects, sampling-augmented gate | 14/100 = 0.14 | `probe_false_reject_gpt2xl.json` |
| W4 MQuAKE rewrite / generation | 0.90 / 0.18 | `mquake_results.json` |
| W4 WikiBigEdit rewrite / locality | 0.82 / 0.24 | `wikibigedit_results.json` |
| W5 WikiBigEdit error rate | 36.0% | `conflict_analysis_report.txt` |
| W5 MQuAKE error rate | 94.0% | `conflict_analysis_report_mquake.txt` |

---

## 4. Re-Run Commands

All commands assume the project root as working directory and the project virtual
environment on `PATH`.

```bash
# W1
python output/w1_fonghiolam/evaluate_pipeline.py --mock --limit 3

# W2
python output/w2_fonghiolam/memit_repro.py
python output/w2_fonghiolam/alphaedit_repro.py

# W3 (module self-tests)
python output/w3_fonghiolam/side_memory.py
python output/w3_fonghiolam/conflict_gate.py
# W3 (studies, need the gpt2-xl checkpoint)
python output/w3_fonghiolam/real_w3_components.py --out output/w3_fonghiolam/w3_components_gpt2xl.json
python output/w3_fonghiolam/canary_strata.py --run w3_components_gpt2xl.json \
    --sampling probe_sampling_gpt2xl.json --out canary_strata_gpt2xl.json

# W4
python output/w4_fonghiolam/mquake_eval.py --mock
python output/w4_fonghiolam/wikibigedit_eval.py --mock

# W5
python output/w5_fonghiolam/conflict_analysis.py --mock
```

---

## 5. Packaging

- [x] Seven week directories, `w0_fonghiolam` through `w6_fonghiolam`
- [x] One `README.md` per week, each with its own result tables inlined
- [x] No byte-compiled cache directories, no compiled `.pyc` files, no editor backup files
- [x] No Chinese text, no AI metadata, no disclaimers
- [x] PDF export of the W6 deliverables — both Markdown files converted with `pandoc`
      (GFM to HTML5) and printed by headless Edge to A4:
      `CISC7021_Group26_W6_Cross-Review_Report.pdf` (10 pages) and
      `CISC7021_Group26_W6_Submission_Checklist.pdf` (4 pages). Browser header and footer
      suppressed; the exported files were re-scanned afterwards and show 0 CJK characters
      in the extracted text, 0 labelling keys in the file bytes, and no XMP packet.
      Both PDFs are tracked in the team repository through an exception in `.gitignore`,
      which otherwise ignores `*.pdf` as a build artefact
- [x] Final push to the team repository — all W0–W6 files plus the two W6 PDFs tracked,
      `origin/main` in sync, working tree clean
- [x] Known exception recorded: `probe_false_reject_gpt2xl.json` keeps three CJK fragments
      (lines 340, 5299, 6443) inside model-generated text, retained as run evidence and
      excluded from the authored-text scan
