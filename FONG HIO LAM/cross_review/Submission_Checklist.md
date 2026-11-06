# Submission Checklist — CISC7021 Group 26 CALM Project

**Owner:** FONG HIO LAM (MC654953)
**Package:** weekly deliverables W0–W6
**Last verified:** 6 November 2026

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
| 10 | Git history clean and pushed | `git status`, `git log`, `git ls-remote` in `project_repo` | Pass — W3 `3d528a7` and W6 `f65f0d3` both on `origin/main`, working tree clean |

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
- [x] `side_memory.py` (225 lines), `conflict_gate.py` (399 lines)
- [x] `real_w3_components.py` (440 lines) — 100-request run driver
- [x] `probe_sweep.py` (174 lines), `probe_likelihood.py` (221 lines), `probe_sampling.py` (168 lines)
- [x] `probe_false_reject.py` (175 lines) — false-reject cost of the sampling variant
- [x] `canary_strata.py` (131 lines) — stratum join
- [x] Real runs: `w3_components_gpt2xl.json`, `probe_sweep_gpt2xl_v2.json`,
      `probe_likelihood_gpt2xl.json`, `probe_sampling_gpt2xl.json`,
      `probe_false_reject_gpt2xl.json`, `canary_strata_gpt2xl.json`
- [x] `README.md` (443 lines) — three canary denominators plus the sampling cost table
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
| W3 decision split (WRITE / ROUTE / REJECT) | 96 / 2 / 2 | `w3_components_gpt2xl.json` |
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
- [ ] PDF export of the W6 report — not applied; deliverables are Markdown, convert only if the marker requires PDF
- [x] Final push to the team repository — all W0–W6 files tracked, `origin/main` in sync
      (`f65f0d3` adds the W6 deliverables), working tree clean
- [x] Known exception recorded: `probe_false_reject_gpt2xl.json` keeps three CJK fragments
      (lines 340, 5299, 6443) inside model-generated text, retained as run evidence and
      excluded from the authored-text scan
