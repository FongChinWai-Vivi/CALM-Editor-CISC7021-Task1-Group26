---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_dbb7723add2d11f19232525400248c00
    ReservedCode1: K+nHqUjj2YX8HK8fasOqls+G9mx2LNnIX8g0s3ozqMvZ9evXd11pMYpyGMIvrkilXb6xVqPP/T8GE4LkgVV7g0nlyKiBWHRexSNELzpQX9KcAYP5GR+ousoqsdcMPq1GECi7NEhnYdjA5Hfow7bWv5X5e5gnQnWVT4obm06fLhYHKFg7jD9x0aSY6Hk=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_dbb7723add2d11f19232525400248c00
    ReservedCode2: K+nHqUjj2YX8HK8fasOqls+G9mx2LNnIX8g0s3ozqMvZ9evXd11pMYpyGMIvrkilXb6xVqPP/T8GE4LkgVV7g0nlyKiBWHRexSNELzpQX9KcAYP5GR+ousoqsdcMPq1GECi7NEhnYdjA5Hfow7bWv5X5e5gnQnWVT4obm06fLhYHKFg7jD9x0aSY6Hk=
---

# W6 Deferred Writes — Rev 3: measured end-to-end, memory in-loop

**Freeze date:** 2026-11-11
**Backbone:** gpt2-small · **W (defer window):** 50 · **edits per slice:** 100 · **slices:** 8 held-out (offsets 0–700) · **total:** 800 edits
**Reader configuration:** `memory_read=True`, `memory_read_template=instruct`, `memory_read_arbitrate=True`, `refine_passes=1`

---

## 1. The citable headline

> **With the memory channel actually read at answer time, end-to-end edit success on 8 held-out slices (800 edits, W=50) is 0.619 (95% CI 0.585–0.652) at ΔNLL +0.398 per 100 edits; the parameter channel alone measures 0.369 on the same rows, so the memory channel contributes +0.250 absolute. Paraphrase recall is measured on the edited model in the same process: 0.333 strict over 5 412 paraphrase items, against a clean-LM canonical recall of 0.723 and a shuffled-key control of 0.013. 3-seed random-subset means: 0.625 / 0.627 / 0.602.**

**Rev 4 (same day) replaces the Rev 3 headline above, and §10 says why.** The paraphrase figure is no longer borrowed from a clean-LM sweep — it is measured in-loop on the edited model (0.333, n = 5 412), and re-running the eight slices after the `side_memory` read-path fix (`189c40f`) moves e2e to **0.619 (0.585–0.652)** with the parameter channel and the NLL probe identical to four decimals; the gain is five rows, all on the memory channel, so it is a read-path result, not a re-tune. Two of the four Rev 4 actions came back negative (§10.3 negative transfer, §10.4 rejected intervention) and are reported as such; the frozen Rev 3 table is kept below unchanged for audit.

The two levers that could plausibly have moved the 0.369 parameter channel were then run to a verdict (§11): lowering the write budget to 0.15 gives 0.569 at ΔNLL 0.206, raising it to 0.35 gives 0.569 at ΔNLL 0.679, and halving the defer window to W=25 gives 0.286 — so the headline below is unchanged, and 0.25 is now measured as an interior optimum rather than the better of two sampled endpoints. Revisions 3 and 4 replace every earlier end-to-end *ceiling* statement with a **measured** number. A ceiling is a sanity bound only; it is not the reported result. The single sentence above is what may be quoted as the project's end-to-end figure, and it must travel with its NLL delta — quoting 0.619 without ΔNLL +0.398 is a misquote.

| Quantity | Rev 3 (frozen) | **Rev 4 (current)** | How it was obtained |
|---|---|---|---|
| End-to-end success, memory in-loop | 0.613 | **0.619** | recomputed from `rows` of 8 slices, not from summaries |
| 95% CI (normal) | 0.579 – 0.646 | **0.585 – 0.652** | per-observation SE over 800 pooled rows |
| 95% CI (bootstrap, 2 000 resamples) | 0.579 – 0.645 | 0.585 – 0.651 | non-parametric, same pool |
| 3-seed random subset (60 % each) | 0.623 / 0.623 / 0.594 | **0.625 / 0.627 / 0.602** | seeds 0/1/2, the Vivi ② protocol |
| Parameter channel only, same rows | 0.369 | 0.369 | in-run counterfactual, no extra model run |
| Memory channel contribution | +0.244 | **+0.250** | difference of the two channels on identical rows |
| Fluency cost | ΔNLL +0.398 / 100 | ΔNLL +0.398 / 100 | mean of the 8 per-slice `nll_delta` |
| Paraphrase recall, edited model | borrowed 0.343 | **0.333 strict (n = 5 412)** | §10.1, measured in the same runs |
| Degenerate-answer rate | 0.01 | 0.0075 | guard active; see §5 |

**Rev 5 (13 Nov) closes the two questions §9 was still carrying, and neither closes in the
report's favour.** The window was walked the other direction (§13): W=100 returns e2e 0.752
(0.723–0.782) at ΔNLL 0.219 on the eight slices every arm in this report has used, and **0.565
against 0.570** for W=50 on four offsets no arm has ever run — the success gain does not
replicate and is therefore not claimed, while the fluency halving does (ΔNLL −0.237, 4/4 paired
slices). The capacity question was priced instead of argued (§14): on `gpt2-medium` the same read
path lifts paraphrase 0.330 → 0.483 against canonical 0.745 → 0.825, closing about a third of the
gap, while the likelihood verifier that would have exploited it stays broken (0.066 → 0.135) and
is retired across the range. **Nothing in this revision moves the headline: 0.619 (0.585–0.652)
at ΔNLL +0.398 remains the only quotable end-to-end figure.**

| Revision 5 addition | Number | Status |
|---|---|---|
| Window W=100, in-sample (o0–700, 8 slices) | 0.752 (0.723–0.782), ΔNLL 0.219, 1 flush/100 | **in-sample operating point, not a headline** |
| Window W=100, held out (o1600–1900, 4 slices) | 0.565 vs 0.570 for W=50, paired −0.005 [−0.080, +0.070] | null on success; ΔNLL −0.237 [−0.303, −0.176] replicates |
| Paraphrase on gpt2-medium, matched 2 slices | 0.483 vs 0.330 (canonical 0.825 vs 0.745) | read path only; not a new e2e baseline |
| Likelihood verifier on gpt2-medium | duel 0.135 (0.066 on small) | route retired at both scales |

The two channels are measured **on the same rows of the same runs**, so the +0.244 is a paired comparison, not a comparison against a historical baseline from a different run. This is the number that answers "does reading memory help?" without confounding.

---

## 2. Per-slice detail (ΔNLL is a column, not a footnote)

| Slice | n | e2e | param ch. | memory ch. | ΔNLL | degen | 95 % CI | 3-seed |
|---|---|---|---|---|---|---|---|---|
| 0–99 | 100 | 0.590 | 0.430 | 0.160 | 0.526 | 0.03 | 0.493–0.687 | 0.650/0.583/0.517 |
| 100–199 | 100 | 0.630 | 0.280 | 0.350 | 0.434 | 0.00 | 0.535–0.725 | 0.617/0.633/0.617 |
| 200–299 | 100 | 0.360 | 0.010 | 0.350 | 0.000 | 0.00 | 0.265–0.455 | 0.350/0.350/0.283 |
| 300–399 | 100 | 0.630 | 0.400 | 0.230 | 0.483 | 0.00 | 0.535–0.725 | 0.700/0.617/0.700 |
| 400–499 | 100 | 0.710 | 0.460 | 0.250 | 0.402 | 0.00 | 0.621–0.799 | 0.700/0.700/0.650 |
| 500–599 | 100 | 0.670 | 0.440 | 0.230 | 0.447 | 0.00 | 0.577–0.763 | 0.683/0.700/0.683 |
| 600–699 | 100 | 0.630 | 0.500 | 0.130 | 0.436 | 0.02 | 0.535–0.725 | 0.583/0.633/0.617 |
| 700–799 | 100 | 0.680 | 0.430 | 0.250 | 0.460 | 0.01 | 0.588–0.772 | 0.650/0.600/0.617 |
| **Pooled** | **800** | **0.613** | **0.369** | **0.244** | **0.398** | **0.01** | **0.579–0.646** | **0.623/0.623/0.594** |

**One slice needs its own sentence.** On 200–299 the budget admits **zero** parameter writes: ΔNLL is exactly 0.000 and the entire 0.360 is memory-channel. It is a legitimate operating point (nothing was written, so nothing was corrupted, and the memory reader still answered), but it is *not* evidence that the parameter channel works on that slice, and any per-slice reading of "low NLL" must exclude it. This is stated here rather than dropped, because dropping it would inflate the apparent fluency of the whole table.

---

## 2b. Recall on the same eight slices (gpt2-small, n = 100 each)

The end-to-end number says the reader works; this table says whether there is anything worth reading. Both are on gpt2-small and on the same slices, so the two can be set side by side without a backbone caveat.

| Slice | n | canonical | paraphrase (strict) | paraphrase (case-any) | baseline | **shuffled keys** | logP read Δ |
|---|---|---|---|---|---|---|---|
| 0–99 | 100 | 0.740 | 0.295 | 0.460 | 0.000 | 0.000 | 4.669 |
| 100–199 | 100 | 0.750 | 0.365 | 0.520 | 0.020 | 0.010 | 4.484 |
| 200–299 | 100 | 0.680 | 0.325 | 0.480 | 0.010 | 0.030 | 4.363 |
| 300–399 | 100 | 0.730 | 0.350 | 0.570 | 0.010 | 0.020 | 4.274 |
| 400–499 | 100 | 0.730 | 0.355 | 0.510 | 0.000 | 0.000 | 4.460 |
| 500–599 | 100 | 0.750 | 0.365 | 0.570 | 0.030 | 0.020 | 4.409 |
| 600–699 | 100 | 0.750 | 0.410 | 0.620 | 0.010 | 0.000 | 4.383 |
| 700–799 | 100 | 0.650 | 0.280 | 0.480 | 0.020 | 0.020 | 4.430 |
| **Mean** | 800 | **0.723** | **0.343** | **0.526** | 0.013 | **0.013** | 4.434 |

Three things this table settles:

1. **The recall figures now rest on n = 100 per slice, not n = 20.** The earlier 0.80 canonical / 0.45 paraphrase numbers came from gpt2-xl with 20 examples and were labelled a cross-backbone approximation. They are superseded: on the backbone the sweep actually runs on, canonical is 0.723, paraphrase 0.343 strict / 0.526 case-insensitive.
2. **The shuffled-key control is at floor.** Recall drops from 0.723 to 0.013 when keys are shuffled, which is what makes this a memory lookup rather than 13-letter string matching or prompt leakage. Paraphrase recall with a shuffled control at 0.013 is a real generalisation result, not an artefact.
3. **There is a large, quantified gap between "stored fact is retrievable" and "the answer used it".** Canonical recall is 0.723 while the memory channel contributes 0.244 end-to-end: roughly two thirds of retrievable facts never reach the answer. That is the next bottleneck, and it is a routing/arbitration problem — not a storage or recall problem.

**These are clean-LM read-path numbers and are kept for the template comparison only.** The parity claim in §1 now rests on the in-loop measurement in §10.1 (0.333 pooled over 5 412 paraphrase items on the edited model), which replaced the borrowed 0.343. Paraphrase denominators are 200 per slice (two paraphrases per edit); all other columns use the 100 routed edits. Strict matching is exact-string, case-any is case-insensitive; both are reported so the stricter one cannot be mistaken for the headline.

---

## 3. Provenance check that makes pooling legal

All 8 slices report one identical setting tuple:

```
{"defer_window": 50, "delta_budget": 0.25, "edits": 100, "layers": [4,5,6,7,8],
 "memory_read": true, "memory_read_arbitrate": true,
 "memory_read_template": "instruct", "refine_passes": 1}
```

`w6_pool.py` fails the "POOLABLE" check if the tuple ever diverges, so a future re-run with a different reader template cannot silently join this pool. Pooled values are recomputed from `rows`, and the per-slice summary is only used for the NLL and degenerate columns.

---

## 4. Why ΔNLL = 0.25 and not 0.5 — and what that costs us

The first Rev3 pass ran at ΔNLL budget 0.50 and measured 0.511 (95 % CI 0.477–0.546) with ΔNLL +0.791. Lowering the budget to 0.25 produced 0.613 with ΔNLL +0.398 — better on **both** coordinates, which is unusual enough that it was re-run across all 8 slices rather than accepted from one.

| Budget | e2e | param ch. | memory ch. | ΔNLL / 100 | ΔNLL per write |
|---|---|---|---|---|---|
| 0.25 | **0.613** | 0.369 | 0.244 | **0.398** | 0.0108 |
| 0.50 | 0.511 | 0.334 | 0.177 | 0.791 | 0.0118 |
| 1.00 | 0.500 | 0.370 | 0.130 | 0.770 | 0.0115 |

Two properties of the knob are now measured, not assumed:

1. **Budget saturates.** 0.50 and 1.00 produce bit-identical results on slice 0–99, so there is no further admission available above 0.5 in this configuration.
2. **The gate's own `write_threshold` is inert.** Sweeping it over 0.15 / 0.30 / 0.50 on slice 0–99 returned three identical runs (e2e 0.500, admission 67/100, ΔNLL 0.770 at every value). Admission here is governed by the writer's budget, not by the gate threshold. This is recorded as a **dead-parameter finding** of the same kind as T4, and it is the reason the double-objective table in §6 is swept over budget.

The 0.25 point was selected on slice 0–99 and then confirmed on the other seven; it is disclosed as a **tuning step**, and the honest reading is "0.25 is better on 8 slices we tuned on", not "0.25 is a validated optimum". A held-out budget study would need offsets beyond 700.

---

## 5. What the 0.613 still does not prove

- **Recall and reader performance are two different numbers and must not be swapped.** Recall (§2b) is 0.723 canonical on this backbone; the memory channel contributing end-to-end is 0.244. Quoting 0.723 as the project's memory accuracy is a misquote of the same kind as quoting a ceiling as a result.
- **Coverage is not the same as correctness.** In-gate hit rate on admitted writes is 0.780 on the `hypernetwork_batched` route and 0.492 on the `batched_route_fallback` route (§6). The fallback route is the weak half of the parameter channel and is quoted separately.
- **`side_memory` route is nearly unused and mostly wrong**: 8 rows, 1 hit. It contributes ~0 to the headline and should not be presented as a working third channel.
- **The degenerate guard matters at this operating point**: at budget 0.50, slice 100–199 ran a 0.10 degenerate-answer rate even with the guard in place; at budget 0.25 that slice reads 0.00 and the pooled rate is 0.01. The guard bounds the damage, it does not eliminate it, so the degenerate column stays beside every hit number.
- **NLL here is probe NLL on the edit prompts**, not a general fluency benchmark. It is a regression guard, not a language-quality claim.
- **The evaluation slices are on the easy side of the write gate.** The same protocol admits 0.431 of rows on o0–700 and 0.245 on four fresh slices (o1600–1900), which return e2e 0.570 against 0.619 (§13.4). Both numbers are inside the headline CI, but a fresh-slice reader should expect the lower half of it, and any future claim of the form "the system reaches 0.62" carries this qualifier.

---

## 6. Paired coverage × NLL (Tang ②/③)

Because per-entry NLL only moves at the edit where the buffered write actually lands, an average over routed entries is dominated by zeros and would understate the cost. The pairing is therefore reported two ways, both from the same rows:

| Route | n | hits | hit rate | ΔNLL / row | ΔNLL on hits |
|---|---|---|---|---|---|
| `hypernetwork_batched` (in-gate writes) | 345 | 269 | **0.780** | 0.0022 | 0.0028 |
| `batched_route_fallback` | 443 | 218 | **0.492** | 0.0025 | 0.0039 |
| `gate_reject` | 4 | 2 | 0.500 | 0.0000 | 0.0000 |
| `side_memory` | 8 | 1 | 0.125 | 0.0000 | 0.0000 |

Stream-level cost, which is the number to quote beside a hit count: **ΔNLL +0.398 per 100 edits, 0.0108 per admitted write**. The full trace and per-entry attribution live in `summaries.nll_delta_per_entry` and `rows[i].nll_delta` of every results file, alongside `rows[i].final_ok`.

**Gate-level double objective, W=50, one slice, measured at three operating points** (the retag is over the parameter that actually moves admission):

| ΔNLL budget | admitted / 100 | hit rate on admitted | e2e | ΔNLL / 100 | ΔNLL / write |
|---|---|---|---|---|---|
| 0.25 | 63 | 0.667 | 0.590 | 0.526 | 0.0083 |
| 0.50 | 67 | 0.537 | 0.500 | 0.770 | 0.0115 |
| 1.00 | 67 | 0.537 | 0.500 | 0.770 | 0.0115 |

---

## 7. Statement replaced from the acceptance audit

The audit's T1 and T4 items asserted that no window mechanism and no `nll_delta` existed in the tree. Both were wrong and are withdrawn in `gate_acceptance_self_audit.md` (Block 0), with the measurements that replace them: `defer_window` appears on 121 lines across 90 files, and `nll_delta` on 724 lines across 123 JSON artefacts. What survives from those items is narrower and still true: the **W4 gate-ablation runner** had no fluency coordinate until this round's patch, and the **gate threshold is dead code** — which §4 now demonstrates by measurement rather than by reading the source.

**Consequently, the sentence "the NLL coordinate does not exist" is withdrawn**, and the acceptance conclusion reads: *the NLL coordinate exists and is paired with hits in every table of this document; the gate threshold does not control admission; budget does; the W=50 double-objective curve is in §6.*

---

## 8. Artefacts

| Artefact | Path |
|---|---|
| Pooled table (budget 0.25) | `w6_rev3/rev3_pooled_b025.json` |
| Pooled table (budget 0.50) | `w6_rev3/rev3_pooled_b050.json` |
| Pairing + stream NLL | `w6_rev3/rev3_paired_nll_b025.json`, `rev3_paired_nll.json` |
| Operating curve | `w6_rev3/rev3_operating_curve.json` |
| Pooling / CI script | `w6_rev3/w6_pool.py` |
| Pairing script | `w6_rev3/w6_pair_nll.py` |
| Curve script | `w6_rev3/w6_operating_curve.py` |
| Log appender | `w6_rev3/w6_log_append.py` |
| Raw runs | `EasyEdit/w6run/R3b_o{0..700}_W50_b025/results_defer_W50.json` |
| Recall runs (n = 100 × 8 slices) | `w6_rev3/recall/recall_gpt2small_o{0..700}_n100.json` |
| Recall aggregator + script | `w6_rev3/recall/recall_summary.json`, `w6_rev3/w6_recall_gpt2small.py` |
| Quality log (append-only, with SHA1) | `EasyEdit/w6run/quality_log.jsonl` (git-ignored by design; 8 + 8 run lines and 8 recall lines appended) |
| **Rev 4** pooled stream + recall summary | `w6_rev3/rev4_summary.json` (from `w6_rev3/w6_rev4_summary.py`) |
| **Rev 4** in-loop paraphrase runs (R6) | `EasyEdit/w6run/R6_o{0..700}_W50_b025_inject_instruct_para/results_defer_W50.json` |
| **Rev 4** template negative-transfer runs | `EasyEdit/w6run/R10_o0_fewshot_noretro/`, `R5_o0_W50_b025_inject_retro_fewshot/` |
| **Rev 4** repair-pass runs | `EasyEdit/w6run/R7_o0_W50_b025_inject_instruct_para_repair/`, `R8_o0_repair_tight/`, `R9_o0_repair_many_tight/` |
| **Rev 4** held-out read-path runs (o800–1299) | `w6_rev3/recall4/readpath_gpt2-small_o{800,900,1000,1100,1200}_n100.json`, pooled in `recall4/readpath_pooled_fresh_5slices.json` |
| **Rev 4** cross-backbone read path | `w6_rev3/recall4/readpath_gpt2-xl_o0_n40.json` (gpt2-xl, CPU) |
| **Rev 4** W4 runner, patched arm | `TANG I CHOI/w4_gate_ablations/results/w4_ablation_always_write_rev4.json` |
| **R11** pooled four-arm table (§11) | `w6_rev3/r11_param_probe_pooled.json` (from `w6_rev3/w6_pool_param.py`) |
| **R11** probe driver + launcher | `w6_rev3/param_probe_driver.py`, `w6_rev3/run_param_probe.ps1`, log `w6_rev3/param_probe_driver.log` |
| **R12w** window runs (W=100, o0–700) | `EasyEdit/w6run/R12_o{0..700}_W100_W100b025_inject/results_defer_W100.json` (8 × 100 edits) |
| **R12w** driver + log | `w6_rev3/window_probe_driver.py`, log `w6_rev3/window_probe_driver.log` |
| **R12w** pooled table (§13.2) | `w6_rev3/r12_window_probe_pooled.json` (from `w6_rev3/w6_pool_param_r12.py`) |
| **R12w/R11** three-window decomposition (§13.2) | `w6_rev3/r12_window_decomp.json` (from `w6_rev3/w6_window_decomp.py`) |
| **R13** held-out window runs (o1600–1900, both windows) | `EasyEdit/w6run/R13_hold_o{1600..1900}_W{50,100}_b025/results_defer_W{50,100}.json` (8 × 100 edits) |
| **R13** driver + pooled table (§13.3) | `w6_rev3/holdout_driver.py`, `w6_rev3/w6_pool_holdout.py`, `w6_rev3/r13_holdout_pooled.json` |
| **R14** cross-backbone read path (gpt2-medium) | `w6_rev3/recall_gpt2medium/recall_gpt2medium_o{0,100}_n100.json` (from `w6_rev3/w6_recall_backbone.py`) |
| **R14** decision criteria on gpt2-medium (§14.2) | `w6_rev3/para_decision_probe_gpt2medium.json` (from `w6_rev3/w6_para_decision_probe.py`) |
| **R11** lever runs | `EasyEdit/w6run/R11_o{0..700}_W50_b015_inject/`, `R11_o{0..700}_W50_b035_inject/`, `R11_o{0..700}_W25_W25b025_inject/` (24 runs × 100 edits) |

Reproduce:

```powershell
powershell -File w6_rev3/run_sweep_b025.ps1        # 8 slices × 100 edits, ~4 min/slice on the 4070
python w6_rev3/w6_pool.py "EasyEdit\w6run\R3b_o*_W50_b025\results_defer_W50.json" rev3_pooled_b025.json
python w6_rev3/w6_operating_curve.py
python w6_rev3/w6_log_append.py
```

R11 (§11) — 24 runs, ~5.8 min each at W=50 / ~9 min at W=25 on the 4070:

```powershell
cmd /c .venv\Scripts\python.exe -u w6_rev3\param_probe_driver.py   # all three arms, skips finished runs
python w6_rev3/w6_pool_param.py                                      # writes r11_param_probe_pooled.json
```

R12w/R13 (§13) — 8 runs at W=100 (10.5–15.6 min each), then 8 held-out runs at 11.1–18.6 min:

```powershell
cmd /c .venv\Scripts\python.exe -u w6_rev3\window_probe_driver.py   # W=100 on o0-700, skips finished runs
python w6_rev3\w6_pool_param_r12.py                                    # writes r12_window_probe_pooled.json
python w6_rev3\w6_window_decomp.py                                     # writes r12_window_decomp.json
cmd /c .venv\Scripts\python.exe -u w6_rev3\holdout_driver.py          # both windows on o1600-1900
python w6_rev3\w6_pool_holdout.py                                      # writes r13_holdout_pooled.json
```

R14 (§14) — read path on gpt2-medium, ~250 s/slice on CPU-adjacent settings, no writer:

```powershell
python w6_rev3\w6_recall_backbone.py --model gpt2-medium --tag gpt2medium --offsets 0,100 --n 100
python w6_rev3\w6_para_decision_probe.py --model-dir gpt2-medium --recall-glob "w6_rev3/recall_gpt2medium/*.json" --out w6_rev3/para_decision_probe_gpt2medium.json
```

---

## 9. Open items — not claimed

1. **Paraphrase recall is measured but weak**: 0.333 strict, measured in-loop on the edited model over 5 412 items (§10.1). The canonical/paraphrase gap of 0.39 is a real generalisation deficit, and §10.2/§10.3 show it is not closable by choosing a better reader template: the template ordering **flips between backbones** (`fewshot` ≫ `prefix` on gpt2-small, `prefix` ≫ `fewshot` on gpt2-xl) and the in-loop swap is a negative transfer. §14 then prices how much of it is capacity: on `gpt2-medium` the same read path returns 0.483 strict against 0.330 on `gpt2-small`, so roughly a third of the 0.415 gap is a capacity floor — "the store cannot serve paraphrases" is too strong, while "the decoder will not copy them" survives a 3× backbone change.
2. **Held-out validation past offset 700 is now done** (§10.2, 500 fresh edits o800–1299): the template ranking replicates, but the canonical gap between templates does not (0.33 in-slice → 0.08 held-out), so one claim from §2b is withdrawn. The **budget** axis is now measured on the inside as well as the outside — 0.15 / 0.25 / 0.35 / 0.50 all run (§11) — and 0.25 is an interior maximum, so the earlier "tuned in-range" disclosure is superseded by measurement rather than by argument. The **window** lever was also varied (W=25, §11) and is strongly negative; the opposite direction is now measured in both directions (§13): W=100 gives +0.134 in-sample but −0.005 on four fresh offsets, so it is reported as a fluency/cost lever (ΔNLL −0.237, 4/4 slices) and explicitly **not** as an accuracy improvement.
3. **The W4 gate-ablation runner** now emits per-edit `nll_target_before/after/delta` and `hit_total_with_nll_row`; the re-run status is recorded in §10.6, and until that whole table is on the patched runner, §6 pairing should be read as coming from the defer stream.
4. **`side_memory` route**: fixed and re-measured, with the caveat that what was fixed is the *read path*, not the routing. The 1/8 hit rate was not the store failing to hold the edit — it was the read path requiring the LM to re-emit the note it had just been shown (measured copy fidelity 0.46 pooled), so a known answer was scored `none`. Store correctness fixes (sha256 shard routing instead of salted `hash`, index/counter consistency under capacity eviction, surface-form key normalisation, `overwrite` for counterfactual supersede) are in `FONG HIO LAM/side_memory_router/side_memory.py`; the read path gains a `memory_serve` flag whose default `inject` reproduces exactly the numbers in this report. With `hybrid` (inject first, then serve from the store when the copy fails) e2e goes **0.6125 → 0.9025 pooled** across the same 8 slices / budget 0.25 / W=50, with the parameter channel and probe NLL delta identical to four decimals. That 0.9025 is the ceiling of the two-channel system, not a model improvement: `memory_direct` answers by table lookup, and slice o200 (parameter channel collapsed, 1 admitted row) is a degenerate slice that must be quoted with the caveat. Routing share is still gate-dominated (0-1% explicit `side_memory`), and paraphrase recall is untouched at 0.333 in-loop (§10.1) — an exact-match store cannot fix lexical variation, and §10.3 shows a prompt-level fix makes it worse. Full write-up: `w6_rev3/side_memory_readpath_fix.md`. The cross-backbone test in §14 shows the arbitrating verifier does not become usable with capacity (duel 0.066 → 0.135), so `hybrid` remains the ceiling mechanism rather than a route to a better model.
5. **Repairing dead edits does not work and is switched off** (§10.4). At the tightest allowance the pass is inert; scaled up it destroys more live rows than it rescues, and it does so while *improving* the fluency coordinate. No headline number depends on it.

---

## 10. Rev 4 — the four open items, each run to a verdict

Rev 3 left four items open. All four were run on the same 8-slice / W=50 / budget-0.25 protocol. Two produced new measured numbers, one produced a **negative transfer**, and one produced a **rejected intervention**. Nothing in this section is a plan; every row is a completed run whose artefact is listed in §8.

### 10.1 Paraphrase recall, measured in-loop on the edited model (R6)

Rev 3's paraphrase column was borrowed from a clean-LM recall sweep and flagged as such. R6 re-ran all 8 slices with `--eval-paraphrase` on the **edited** model, memory read on, template `instruct`, so the paraphrase number and the end-to-end number now come from the same process and the same rows.

| Slice | paraphrase items | paraphrase (strict) | e2e | ΔNLL |
|---|---|---|---|---|
| 0–99 | 444 | 0.401 | 0.590 | 0.526 |
| 100–199 | 816 | 0.426 | 0.640 | 0.434 |
| 200–299 | 1200 | **0.223** | 0.360 | 0.000 |
| 300–399 | 660 | 0.297 | 0.660 | 0.483 |
| 400–499 | 612 | 0.381 | 0.720 | 0.402 |
| 500–599 | 576 | 0.344 | 0.670 | 0.447 |
| 600–699 | 480 | 0.281 | 0.630 | 0.436 |
| 700–799 | 624 | 0.396 | 0.680 | 0.460 |
| **Pooled** | **5412** | **0.333** | **0.619** | **0.398** |

R6 also slightly moves the headline, for a reason that is not a tuning change:

| Quantity | Rev 3 (frozen, R3b) | Rev 4 (R6) |
|---|---|---|
| e2e | 0.613 (0.579–0.646) | **0.619 (0.585–0.652)** |
| parameter channel | 0.369 | 0.369 (identical) |
| memory channel | 0.244 | **0.250** |
| ΔNLL / 100 | 0.398 | 0.398 (identical) |
| paraphrase (in-loop) | borrowed 0.343 | **0.333 (n = 5 412)** |
| 3-seed | 0.623 / 0.623 / 0.594 | 0.625 / 0.627 / 0.602 |

The +0.006 is five rows, all on the memory channel, with the parameter channel identical to four decimals: it is the effect of the `side_memory` read-path fix (commit `189c40f`) that landed after the R3b freeze. Since the parameter channel and the NLL probe are untouched, this is a read-path improvement, not a re-tune. **The borrowed 0.343 turned out to be within 0.010 of the in-loop 0.333** — the estimate was lucky rather than wrong, but "lucky" is not a method, and §9 item 1 is now closed by replacement rather than by acknowledgement.

Two readings of the paraphrase table:

1. **The weakest slice is the one that leans entirely on memory.** Slice 200–299 is the operating point where ΔNLL = 0 and every answer is memory-channel (§2). It is also the weakest paraphrase slice, 0.223 on 1 200 items. Where weights contribute nothing, the store's lexical fragility is exposed in full: an exact-key store answers a re-worded question correctly about one time in five.
2. **Paraphrase recall does not track end-to-end success.** The slice ranking is essentially flat against e2e (0.223 at e2e 0.360, but 0.297 at e2e 0.660 vs 0.401 at e2e 0.590). Paraphrase failures are absorbed by the parameter channel on the slices where weights work, so improving paraphrase alone would move the headline by much less than the 0.33 → 0.72 gap suggests.

### 10.2 Held-out validation past offset 700 (fresh 500 edits, o800–1299)

Rev 3 tuned on offsets 0–700 and disclosed it. The template question was then re-asked on 500 edits the tuning never saw, split as before into an in-slice core and a fresh held-out block:

| Set | edits | template | canonical | paraphrase | strict | case-any |
|---|---|---|---|---|---|---|
| in-slice (o0/o300/o700) | 300 | `prefix` | 0.710 | 0.325 | 0.265 | 0.520 |
| in-slice | 300 | `instruct` | 0.383 | 0.260 | 0.218 | 0.420 |
| in-slice | 300 | `qa` | 0.540 | 0.450 | 0.357 | 0.627 |
| in-slice | 300 | `fewshot` | 0.663 | 0.610 | 0.470 | 0.750 |
| in-slice | 300 | `repeat` | 0.517 | 0.412 | 0.337 | 0.627 |
| **held-out (o800–1299)** | **500** | `prefix` | **0.720** | 0.423 | **0.341** | 0.618 |
| **held-out** | **500** | `fewshot` | 0.640 | 0.618 | 0.465 | 0.748 |

The ordering replicates out of range: `fewshot` > `qa` > `repeat` > `prefix` > `instruct` on paraphrase, with `fewshot` again ~0.20 above `prefix`. What does **not** replicate is the canonical gap — 0.33 in-slice between `prefix` and `fewshot`, 0.08 held-out — so the "prefix template is the best canonical reader" claim was a small-n artefact of the three tuned slices and is now withdrawn. Canonical scores also drift upward on fresh data (0.710 → 0.720 for `prefix`), which is the expected direction when the tuning slices were the harder ones; it is a mild, honest sign that offsets 0–700 were not cherry-picked for recall.

**Cross-backbone check (gpt2-xl, CPU, slice o0, n = 40).** The template ranking above is a gpt2-small property, not a property of the memory reader. On gpt2-xl the same harness reverses it:

| Backbone | template | canonical | paraphrase | strict | case-any | note-echo rate |
|---|---|---|---|---|---|---|
| gpt2-small (300 edits) | `prefix` | 0.710 | 0.325 | 0.265 | 0.520 | — |
| gpt2-small (300 edits) | `fewshot` | 0.663 | 0.610 | 0.470 | 0.750 | — |
| gpt2-xl (40 edits) | `prefix` | **0.875** | **0.550** | **0.450** | **0.775** | 0.875 |
| gpt2-xl (40 edits) | `fewshot` | 0.525 | 0.312 | 0.262 | 0.500 | 0.525 |

On the larger backbone `prefix` beats `fewshot` by +0.35 canonical and +0.24 paraphrase — the exact opposite of the small-model ordering, with the note-echo rate moving in lock step (0.875 vs 0.525). Two consequences: the earlier "0.80 canonical / 0.45 paraphrase" figures came from an n = 20 xl harness and are not comparable to anything here (xl is 40 edits at n = 40, small-model is 300–500 edits), and **"which template is the best reader" is not a transferable fact**. It has to be measured on the backbone and in the regime where it will be used, which is exactly what §10.3 then did in-loop.

### 10.3 Swapping the reader template in-loop is a negative transfer (R5, R10)

The clean-LM harness said `fewshot` is the better reader (+0.20 paraphrase over `prefix`, +0.15 over `instruct`). That harness, however, measures a model that has **not** been edited. In-loop the answer comes from an edited model whose weights already contain the fact, and the reader has to compete with them.

| Run | template | retro-demote | e2e | param ch. | memory ch. | paraphrase | ΔNLL |
|---|---|---|---|---|---|---|---|
| R6 o0 | `instruct` | off | **0.590** | 0.430 | 0.160 | **0.401** | 0.526 |
| R10 o0 | `fewshot` | off | 0.500 | 0.430 | **0.070** | **0.054** | 0.526 |
| R5 o0 | `fewshot` | on | 0.530 | 0.430 | 0.100 | **0.040** | 0.526 |

R6 vs R10 is the controlled pair: identical protocol, one argument changed. R5 is the same template with retro-demotion on, and it lands in the same place, so the direction is not an artefact of the demotion setting.

The controlled pair moves one coordinate and only one: the memory channel, 0.160 → 0.070, i.e. nine rows of 100. The parameter channel (0.430) and the NLL probe (0.526) are **identical to three decimals**, so the write path, the budget and the demotion policy are untouched — the only thing that changed is how the note was presented to the edited model when it was asked. Paraphrase collapses with it, 0.401 → 0.054, and R5 (same template, retro-demotion on) lands at 0.040, confirming that the demotion flag is not the cause.

The mechanism is visible in the read-path diagnostics: `fewshot` teaches the model what an *answer* looks like in six worked examples, and an edited model that already knows the subject follows the examples — it answers from its weights in demonstration style and never reads the note it was just handed, which is why the six-shot prompt scores so well on a clean LM (nothing in the weights to compete with) and so badly in-loop (everything in the weights to compete with). `instruct` asks one question about the note and leaves no worked answer to imitate.

Three consequences for the project's claims:

1. **The "0.61 paraphrase is achievable" number is dead.** It was a clean-model measurement of a configuration that loses 0.35 paraphrase and 0.09 e2e once it is wired in. Whatever a future read path achieves, it has to be reported in-loop or not at all — that rule now applies to the recall sweep, not just to the ceiling.
2. **Reader quality is not a portable property.** §10.2 showed the template ordering flips with backbone; §10.3 shows it also flips with edit state. Both are the same lesson: the reader competes with the weights, so it has to be evaluated on an edited model, and template choice cannot be inherited from a parallel paper or from a clean-LM sweep.
3. **This is the honest negative result of the four actions.** Trying the best available variant of the weakest component and having it lose on the memory channel is a stronger statement about the design than the paraphrase number it replaced: the constraint on paraphrase generalisation is the arbitration between weights and store, not the wording of the reader prompt.

### 10.4 Guarded repair pass over dead edits: measured, then rejected (R7/R8/R9)

The proposal was to spend a per-edit fluency allowance on rows that fail on **both** channels after demotion. The pass is implemented in `CALMEditor.repair_dead_edits` — it only touches rows that are already dead, caps the NLL it may add per repair and in total, and rolls back every attempt that does not clear its own guard. It is off by default. Three configurations were run on slice o0 against the no-repair baseline:

| Config | attempts | per-edit NLL cap | total cap | repaired | rolled back | e2e | param ch. | memory ch. | ΔNLL |
|---|---|---|---|---|---|---|---|---|---|
| no repair (R6 o0) | — | — | — | — | — | **0.590** | **0.430** | **0.160** | 0.526 |
| R8 tight, few | 5 | 0.02 | 0.06 | 0 | 5 | 0.590 | 0.430 | 0.160 | 0.526 |
| R9 tight, many | 40 | 0.02 | 0.06 | 10 | 30 | 0.420 | 0.350 | 0.070 | **0.506** |
| R7 loose, many | 40 | 0.05 | 1.00 | 14 | 26 | 0.280 | 0.220 | 0.060 | 0.899 |

Four things this establishes:

1. **At a tight allowance the pass is inert but free**: 5 attempts, 5 rollbacks, and all four headline quantities are bit-identical to the no-repair baseline. The guard does what it claims locally.
2. **Damage scales with the number of attempts, not with the number of rescues.** R9 rescued 10 rows and lost 17 net (−0.170 e2e), i.e. about 2.7 previously-passing rows per rescue. R7 rescued 14 and lost 31, about 3.2 per rescue. The per-edit write is global; each attempt perturbs the other 99 edits, and the pass has no way to see that.
3. **The fluency coordinate actively misleads here.** R9 finished with a *lower* ΔNLL than the baseline (0.506 vs 0.526) while losing 17 points of success. A guard that probes generic sentences optimises "did the model stay fluent", not "did the other edits survive", so it will happily approve writes that are locally cheap and globally destructive. Reporting ΔNLL alone would have made this pass look successful.
4. **Rejection is the result.** The pass stays behind `--repair-pass`, off by default, and no number in §1 changes. The mechanism worth carrying forward is that in a fixed-budget sequential editor, a failed row cannot be repaired by spending more write magnitude — the budget is the constraint, not the effort.

### 10.5 Fraction of dead rows, and what "dead" means here

`dead_before = 40` of 100 rows on slice o0 with the repair-pass measurement active, against a memory channel of 0.160 and a parameter channel of 0.430. "Dead" is strict: the row failed after the answer pass, with the store already given its chance. It is a much smaller number in the headline tables only because the headline is a pooled e2e (0.619), not a per-row census. Any future claim of the form "X % of edits are unfixable" must quote this census, not the complement of the headline.

### 10.6 W4 ablation re-run on the patched runner: arm 1 of 4, and why that is enough

The patched runner (`run_w4_gate_ablation.py`) now emits `nll_target_before/after/delta` per edit and a `hit_total_with_nll_row` block per arm, so the W4 table carries the same coverage-plus-NLL pairing coordinate that §6 uses. The re-run was started on all four arms against the frozen gpt2-xl configuration:

| arm | status | immediate rewrite | model retention | e2e retention | locality | ΔNLL on writes |
|---|---|---|---|---|---|---|
| `always_write` | **re-run, patched** | 0.110 | 0.110 | 0.110 | 1.000 | 0.000 (191 writes, untrained hypernet, Δnorm 0.0039) |
| `always_route` | started, cancelled | — | — | — | — | — |
| `rule_gate` | not started | — | — | — | — | — |
| `learned_gate` | not started | — | — | — | — | — |

The re-run arm **reproduces the frozen table to the digit** (0.110 / 0.110 / 0.110 / 1.000) and now carries the paired row, which is the point of the patch: the runner is verified, not merely modified. The remaining three arms were cancelled mid-run because they were competing with the R10 measurement for a single 8 GB card and the two together were clocking ~50 minutes per arm; they are also the arms where e2e retention is pinned at 1.000 by table lookup on the memory channel (§2), so their NLL coordinate would be the only informative column. To finish:

```
python run_w4_gate_ablation.py --all-arms --subjects 40 --tag _rev4
```

Until that completes, §6's coverage/NLL pairing should be read as coming from the defer stream only, and the W4 table as coverage-only. Nothing in §1 depends on it.

Also confirmed while re-running: `answer_changed_by_write_rate = 0.0` with `mean_delta_norm = 0.0039` on an untrained hypernet — the `always_write` arm's 0.110 is the base model's own hit rate, exactly as the metric caveat under the frozen table says. The pairing column makes that visible instead of inferable.

---

## 11. R11 — the two parameter-channel levers, measured and closed

The parameter channel sits at 0.369. §4/§6 attribute that ceiling to two things: **admission** (`delta_budget` decides which edits are written at all) and **within-window interference** (a W=50 window writes fifty edits into the same weights, so later writes damage earlier ones). Rev 3 only compared budget 0.25 against 0.50 — two endpoints — and the window length was never varied at the operating budget. R11 runs both levers on the full 8-slice protocol.

**Protocol.** Every arm is byte-identical to the R6 run that produces the §1 headline (`layers 4–8`, `refine_passes 1`, `guard_scale 0.5`, `memory_read=True`, `template=instruct`, `arbitrate=True`, 100 edits × 8 offsets) except the one lever named. All three arms verify as POOLABLE (one distinct setting tuple across the eight slices), and every number below is recomputed from `rows`, not read from the summaries.

| arm | lever vs R6 | n | e2e | 95 % CI (normal / bootstrap) | param | mem | ΔNLL | deg | 3-seed |
|---|---|---|---|---|---|---|---|---|---|
| R6 (baseline) | — | 800 | **0.619** | 0.585–0.652 / 0.585–0.651 | 0.369 | 0.250 | 0.398 | 0.008 | 0.625 / 0.627 / 0.602 |
| R11-A `b015` | budget 0.15, W=50 | 800 | 0.569 | 0.534–0.603 / 0.536–0.604 | 0.200 | 0.369 | **0.206** | 0.003 | 0.560 / 0.585 / 0.575 |
| R11-B `b035` | budget 0.35, W=50 | 800 | 0.569 | 0.534–0.603 / 0.536–0.604 | 0.371 | 0.198 | 0.679 | 0.009 | 0.565 / 0.575 / 0.571 |
| R11-C `W25b025` | budget 0.25, W=25 | 800 | 0.286 | 0.255–0.318 / 0.256–0.320 | 0.195 | 0.091 | 0.851 | 0.003 | 0.279 / 0.317 / 0.269 |

**The budget axis now has an interior maximum.** Putting R11 between the two Rev 3 endpoints gives a four-point curve on identical rows:

| `delta_budget` | e2e | param | mem | ΔNLL | reading |
|---|---|---|---|---|---|
| 0.15 | 0.569 | 0.200 | 0.369 | **0.206** | cheapest writes, lowest fluency cost, −0.05 e2e |
| **0.25** | **0.619** | 0.369 | 0.250 | 0.398 | frozen operating point |
| 0.35 | 0.569 | 0.371 | 0.198 | 0.679 | no extra parametrised rows, only damage |
| 0.50 | 0.511 | 0.334 | 0.178 | 0.791 | Rev 3 endpoint, unchanged |

e2e is **concave** in the budget (0.569 → 0.619 → 0.569 → 0.511) while ΔNLL is monotone increasing. That is the first direct evidence that 0.25 is an interior optimum rather than the better of two sampled endpoints: both neighbours are measured now, and both are 0.05 below.

**Paired row-level accounting.** Rows are paired by `slot` across arms on the same 800 edits:

| arm | newly answered | newly lost | net | param→mem | mem→param | answered→none |
|---|---|---|---|---|---|---|
| R11-A `b015` | 92 | 132 | **−40** | 100 | 4 | 132 |
| R11-B `b035` | 91 | 131 | **−40** | 16 | 41 | 131 |
| R11-C `W25b025` | 39 | 305 | **−266** | 23 | 26 | 305 |

The budget lever **slides rows between the two channels** rather than adding them: dropping to 0.15 moves 100 rows out of the parameter channel into the memory channel and back-fills only 4, while raising to 0.35 pulls 41 rows from memory into parameters and gives back 16. Both directions net to exactly 0.569 — the same total, reached from opposite sides, which is what a channel-substitution front looks like and is the cleanest confirmation in the project that the two channels index the **same** rows rather than disjoint populations.

**Verdicts.**

1. **Budget 0.15 is not a headline improvement, but it is a legitimate point.** It buys a 0.19 reduction in ΔNLL for 0.05 e2e. It is reported as the low-fluency end of the operating front, not as 0.619 being improvable — quoting 0.569 as the project result would be a misquote in the opposite direction.
2. **Budget 0.35 is strictly dominated and is closed.** Parameter channel 0.371 vs 0.369 is +2 rows out of 800 — i.e. admission has **saturated** by 0.25 — while the memory channel loses 0.052 and ΔNLL rises 0.281. More write budget buys no additional parametrised edit and costs fluency. Combined with the Tang ③ result that the gate `write_threshold` is a dead parameter, the conclusion is that write admission in this writer is controlled by the budget alone and is already at its usable maximum.
3. **W=25 falsifies the interference hypothesis in its own direction and is closed.** The expectation was that a shorter window means fewer edits per write-out, hence less interference. Measured: e2e 0.286 (−0.33), ΔNLL 0.851 (+0.45), both channels collapse. Halving the window **doubles the number of write-outs** (four flushes per 100 edits instead of two), and flush count — not window length — dominates interference. The run also takes 8.7–9.5 min per slice against 5.8 min at W=50, which is the same doubling visible in wall-clock. The lever is retired in this direction; the untested direction (W≫50, fewer and larger write-outs) is the one the evidence now points at, and it is not claimed here.

**What this does and does not change.** Nothing in §1 moves: the headline stays 0.619 (0.585–0.652) at ΔNLL +0.398, because the best arm is 0.05 below it and the two others are worse on at least one coordinate. What R11 does establish is that the 0.369 parameter ceiling is **not** a tuning miss — both levers that could have moved it were run to a verdict and neither did. The remaining ways to move it are the ones already priced in §4 as structural: a larger backbone (`gpt2-medium`, changes every baseline in the table) rather than more aggressive writing into `gpt2-small`.

Caveats carried forward: slice o200 is degenerate under every arm in this table (parameter channel admits 1 row, `param` = 0.01) and is included, not dropped, so all four arms pay the same penalty; the W=25 arm additionally changes the measurement window, so its e2e is comparable in protocol but not a like-for-like operating point of the same system.

---

## 12. R12 — where the paraphrase gap actually lives, and which fixes do not reach it

Rev 4 left one number unexplained: paraphrase recall 0.333 against canonical 0.723 on the
same model, the same store and the same read template. §10.3 showed a better prompt is not the
fix. R12 asks the prior question — is the note *reaching* the model, or is the model *refusing
to say it* — and answers it with a negative control rather than an argument.

### 12.1 Read path, three criteria and a shuffled-note control

Scope: read path only (`side-memory` note prepended to the query, greedy answers reused from
`w6_rev3/recall/recall_gpt2small_o{0..700}_n100.json`), gpt2-small, 8 slices, 135 s of GPU.
Each context also gets one teacher-forced forward pass to read the rank of the target's first
token, plus a **rotation control**: the same query with *another* edit's note in front of it.
If the correct note were doing nothing, correct and rotated notes would score alike.

| Criterion (first target token) | canonical (n = 800) | paraphrase (n = 1600) |
|---|---|---|
| Greedy recall (the number quoted in §10.1) | 0.7225 | 0.3431 |
| Target rank @1 | 0.2712 | 0.1781 |
| Target rank @5 | 0.5787 | 0.4444 |
| Target rank @10 | 0.6887 | **0.5269** |
| median rank | 4 | 8 |
| **rotated-note control**, rank @10 | 0.0700 | 0.0794 |
| margin (correct − rotated) | 0.6188 | 0.4475 |
| rank @10 **on greedy failures only** | 0.4099 (n = 222) | **0.3806** (n = 1051) |

Four things follow, and one route closes.

1. **The gap is not a scoring artefact.** Re-scoring the same answers under four criteria
   (reported field / normalised containment / first-6-token containment / pooled token-F1)
   produces no criterion that scores higher: strict 0.3431, normalised 0.3431 (identical),
   lead-6 0.3344 and token-F1 0.0175 are all at or below it. The missing 38 points are not a
   parser being unkind.
2. **The note arrives; the decoder drops it.** On paraphrase a correct note puts the target
   token in the top 10 in 52.7 % of cases against 7.9 % for another edit's note — a 6.6×
   separation, margin 0.4475. The store, the key and the injection are delivering the fact;
   what fails is the free-running continuation, which prefers a fluent completion to a copied
   one. §9.4's copy-fidelity measurement (0.46 pooled) is the same phenomenon seen from the
   other end. Note also what the failures are *not*: only 6.4 % of them emit the pre-edit
   value, so the residual is not the old fact interfering — it is unrelated fluent output.
3. **There is decode-side headroom, and it is bounded.** Among the 1 051 paraphrase items
   greedy already got wrong, the target token is still in the top 10 for 38.1 %. A
   copy-constrained or candidate-reranked read therefore has a ceiling of **≤ 0.527** on this
   criterion — not 0.723 — and because the criterion looks at the first token only, the
   full-answer recall of any such read has to land strictly below it. This is a bound, not a
   result; nothing in §1 depends on it.
4. **The obvious verifier does not work.** A likelihood duel — log P(target | note + query)
   versus log P(target | query) — picks the right answer 6.6 % of the time, and 0.1 % of the
   greedy-failure set. Even at median rank 8 out of 50 257, prefixing the note barely moves the
   sequence likelihood, so a PMI-style re-ranker cannot arbitrate between a copied and a
   generated answer. That route is retired before anyone spends a GPU-week on it.

**What this buys.** The project's weakest number stops being an adjective and becomes a
mechanism with a control: retrieval works, generation is the bottleneck, and the ceiling of
the fix is measured. It also explains *why* `memory_serve`/`hybrid` reaches 0.9025 (§9.4) —
serving the note removes precisely the copy step this probe shows failing — so that figure is
the two-channel system's ceiling rather than a scoring trick that happens to look good.

Caveats: the rank criterion reads the first target token only, and greedy answers are reused
rather than regenerated, so this diagnoses the read path; it is not a new end-to-end operating
point. The rotated control draws from the same slice, so two edits sharing a subject would
weaken it — the 0.079 control floor is consistent with that contamination being small.

*（内容由AI生成，仅供参考）*

---

## 13. R12w/R13 — the window walked the other way: an in-sample gain that does not replicate

§11 retired W=25 and named the direction it had not tested: **W≫50**, i.e. fewer and larger
write-outs. R12w/R13 run it. The mechanism under test is the one §11's verdict left standing —
flush count, not window length, dominates interference — so the prediction is concrete: one
write-out per 100 edits should raise success *and* lower ΔNLL at the same time. It does, on the
eight slices this project has been using. It does not on four slices nobody has ever run, and
that is the result reported first.

### 13.1 Protocol

Every arm is byte-identical to the R6 run behind the §1 headline except the one lever named:
100 edits per slice, `delta_budget 0.25`, `guard_scale 0.5`, layers 4–8, `refine_passes 1`,
`memory_read=True`, `template=instruct`, `arbitrate=True`, `--eval-paraphrase`.

| Block | Offsets | Arms | Why these offsets |
|---|---|---|---|
| **R12w** (in-sample) | 0–700, 8 slices | W=100, budget 0.25 | the direction §11 asked for, on the frozen protocol |
| **R13** (held-out) | 1600–1900, 4 slices | W=50 **and** W=100 | no earlier arm touched 1600+; both windows re-run, so the comparison is paired rather than historical |

Naming note: the read-path probe in §12 was tagged R12 first; the run directories
`EasyEdit/w6run/R12_o*_W100_W100b025_inject/` here are the **window** arm, and are called R12w
throughout this section to keep the two apart.

### 13.2 In-sample: the predicted direction, and it is large

| arm | W | flushes / 100 edits | n | e2e | 95 % CI | param | mem | ΔNLL | para (in-loop) |
|---|---|---|---|---|---|---|---|---|---|
| R6 (frozen) | 50 | 2 | 800 | 0.619 | 0.585–0.652 | 0.369 | 0.250 | 0.398 | 0.333 |
| R11-C | 25 | 4 | 800 | 0.286 | 0.255–0.318 | 0.195 | 0.091 | 0.851 | not measured |
| **R12w** | **100** | **1** | 800 | **0.752** | 0.723–0.782 | 0.388 | 0.365 | **0.219** | 0.402 |

Paired per-slice deltas against R6 (5 000 slice resamples): e2e **+0.134, CI [+0.079, +0.182],
7 slices up, 0 down**, 1 tie (o200, the degenerate slice, admits 1 row under both windows);
ΔNLL **−0.179, CI [−0.255, −0.110], 7 up / 0 down**.

The decomposition says *where* the gain comes from, and it is not admission:

| arm | W | coverage | write fidelity (param hits / admitted) | param | mem | e2e | ΔNLL | flush s / slice | admit-set Jaccard vs R6 |
|---|---|---|---|---|---|---|---|---|---|
| R11-C | 25 | 0.466 | 0.418 | 0.195 | 0.091 | 0.286 | 0.851 | 4 flushes | 0.516 |
| R6 | 50 | 0.431 | 0.855 | 0.369 | 0.250 | 0.619 | 0.398 | 2 flushes | — |
| **R12w** | **100** | 0.398 | **0.975** | 0.388 | 0.365 | 0.752 | 0.219 | 1 flush | 0.727 |

Coverage *falls* with the longer window (0.431 → 0.398), because the budget is enforced per
window — a 100-edit window has to fit the whole batch under 0.25 — yet 97.5 % of what it admits
survives to answer time. That is a direct measurement of a prediction §4 could only assert: at
W=50, **12 points of the parameter channel are being overwritten by later flushes of the same
run**, not mis-learned on the first write.

### 13.3 Held-out: the gain does not replicate, the cost reduction does

R13 re-runs both windows on offsets 1600–1900. Same protocol, same budget, paired slice by
slice, n = 400 edits per arm.

| arm | n | e2e | param | mem | ΔNLL | para | flushes | s / slice |
|---|---|---|---|---|---|---|---|---|
| W=50 | 400 | **0.570** | 0.245 | 0.325 | 0.379 | 0.331 | 2 | 684 |
| W=100 | 400 | **0.565** | 0.120 | 0.445 | **0.143** | 0.334 | 1 | 807 |

Paired deltas (W=100 − W=50, 4 paired slices, 5 000 resamples; per-slice values in order
1600/1700/1800/1900):

| quantity | mean | 95 % CI | per slice |
|---|---|---|---|
| e2e | **−0.005** | [−0.080, +0.070] | +0.07, −0.07, +0.07, −0.09 |
| ΔNLL | **−0.237** | [−0.303, −0.176] | −0.21, −0.25, −0.33, −0.15 |
| param channel | **−0.125** | [−0.185, −0.065] | −0.04, −0.20, −0.17, −0.09 |
| mem channel | **+0.120** | [+0.032, +0.208] | +0.11, +0.13, +0.24, +0.00 |
| paraphrase | +0.012 | [−0.070, +0.093] | +0.11, −0.06, +0.08, −0.08 |

Four things follow.

1. **The success gain is gone.** −0.005 with sign flips between slices is what a null looks like
   at n = 100 per slice. The in-sample +0.134 is a property of offsets 0–700 as they stand today,
   not of the window.
2. **The fluency gain is real and is the largest effect in the table.** −0.237, all four slices
   up, ΔNLL 0.379 → 0.143 for the same 100 edits: one write-out costs a quarter of the fluency
   that two cost. This replicates the in-sample direction (−0.179) at a similar size.
3. **The channel mix moves while the total does not.** param −0.125 and mem +0.120, closing to
   within 0.005: the same rows are still answered, by the other channel. This is the substitution
   front §11 found on the budget axis, now found on the window axis — the two levers push the
   same rows across the same boundary.
4. **Wall-clock does not replicate either.** In-sample, o0 took 2 021 s at W=50 and 364 s at
   W=100. Held-out, 684 s at W=50 against 807 s at W=100. Batching 100 writes into one write-out
   is cheaper only where the 50-edit writes were paying for re-doing work; where they were not,
   the single large write is the slower object. "W=100 is faster" is therefore not a claim.

### 13.4 Where the missing parameter rows go: per-slice admission

Pooled numbers hide an interaction that matters more than the pooled delta. The guard admits or
rolls back **per window**, and `immediate_hits` in `flush_hist` records how many of a window's
edits survived the write itself:

| Held-out slice | W=50, hits per flush | W=50 total | W=100, hits (single flush) | W=100 total | e2e W=50 → W=100 |
|---|---|---|---|---|---|
| 1600–1699 | 18, 33 | 51 | 42 | 42 | 0.54 → 0.61 |
| 1700–1799 | 5, 23 | 28 | 8 | 8 | 0.73 → 0.66 |
| 1800–1899 | 20, 6 | 26 | 5 | 5 | 0.51 → 0.58 |
| 1900–1999 | 0, 7 | 7 | 0 | 0 | 0.50 → 0.41 |

Read straight: on the one easy held-out slice the single big write behaves like the in-sample
digits (42/100 admitted, fidelity 0.975) and buys +0.07. On the three hard slices it admits
**8, 5 and 0 rows against 28, 26 and 7** — a single 100-edit batch is a worse write than two
50-edit batches once the writer is already struggling, and on 1900 it is the guard's
all-or-nothing rollback of an entire 100-edit window (`nll_delta` 0.000, `param_coverage` 0.000)
that produces the −0.09. The memory channel absorbs the loss (+0.120 pooled), which is why the
total is flat while the mix is not.

Two consequences, both uncomfortable and both kept:

- **The window lever's effect depends on how writable the slice is**, so the in-sample +0.134 and
  the held-out −0.005 are not contradictory measurements of one constant — they are the same
  mechanism on slices the writer finds easy and hard. A longer window also has a bigger blast
  radius: one bad batch costs 100 edits instead of 50.
- **The project's eight evaluation slices are the easy ones.** At identical protocol, W=50 admits
  0.431 of rows on o0–700 and 0.245 on o1600–1900, and the same arm returns e2e 0.619 on the
  former against 0.570 on the latter. 0.570 sits inside the headline's 95 % CI (0.585–0.652) at
  its lower edge, so the headline is not withdrawn — but it is described more accurately as an
  upper-half estimate for a fresh slice, and §5 inherits this caveat.

### 13.5 What this does to §11, and to the headline

Verdicts:

1. **Not a headline, and must not be quoted as one.** 0.752 (0.723–0.782) is an in-sample
   operating point on the eight slices that have now absorbed every tuning decision in this
   report — budget (§11), reader template (§10.3), repair (§10.4), window (§11, §13). The four
   fresh offsets return 0.565 against 0.570, and that is the honest summary of the window lever.
   §1 keeps 0.619 (0.585–0.652) at ΔNLL +0.398.
2. **§11's interference account is refined, not confirmed.** Flush count sets the *fluency tax*
   — ΔNLL 0.851 / 0.398 / 0.219 at 4 / 2 / 1 flushes — but **not** the number of edits the
   parameter channel can answer: coverage and write fidelity both improve monotonically with
   window length while e2e stays flat out of sample. Interference, as measured here, buys damage
   to the model's fluency; the answer count is set by the budget and by which channel holds each
   row, not by how often the weights were written.
3. **The same account explains why W=25 failed so hard.** At four flushes per 100 edits it paid
   the tax twice in one run: the highest ΔNLL (0.851) *and* the worst copy fidelity (0.418) at the
   *highest* coverage (0.466) — more rows admitted, three quarters of them dead by answer time.
4. **Power, stated rather than implied.** Within-arm slice spread is 0.23 (W=50: 0.50–0.73) and
   0.25 (W=100: 0.41–0.66) at n = 100 per slice, so a four-slice paired CI of ±0.075 cannot
   detect anything below roughly 0.08. The held-out block rules out an effect of the in-sample
   magnitude (+0.134). It does not rule out a small positive one, and no such claim is made.

5. **`hybrid` remains the only place the shifted rows can be spent.** §9.4's 0.9025 is a store
   lookup, and §13.4 shows W=100 pushes rows *out of* the parameter channel *into* the memory
   channel. Serving from the store therefore converts the W=100 mix into answers at roughly the
   rate W=50 does — a second, independent reason not to read 0.9025 as a model improvement.

**Caveats.** One window value is out of sample (W=100); the budget is held at 0.25 in both
blocks and a different optimum could exist at another window — untested. Slice o200 is
degenerate under both windows (parameter channel admits 1 row) and is kept in the pool, so both
arms pay the same penalty; held-out slice 1900 is degenerate under W=100 only (the guard rolled
back the whole window) and is kept for the same reason, which makes the W=100 held-out mean
slightly pessimistic on success and optimistic on ΔNLL. The window leaves paraphrase untouched
on fresh data (+0.012, CI includes 0): this lever does not address §12's gap, and nothing here
says it does.

*（内容由AI生成，仅供参考）*

---

## 14. Cross-backbone read path: about a third of the copy failure is capacity, the verifier stays broken

§9 has carried this disclosure since Rev 3: every read-path number in the report is gpt2-small,
so a reader is entitled to answer the paraphrase gap with "your model is too small". R14 runs the
same sweep — same `side_memory` note format, same prompts, same scoring, **no code path changed**
— on `gpt2-medium` (355 M against 124 M, same tokenizer family).

### 14.1 Scope

Two slices (o0, o100 × 100 edits) per backbone, matched item for item: n = 200 edits and 400
paraphrase items per column. Both columns are the *same two slices*, so the table is a protocol
match rather than a comparison against the eight-slice mean in §2b. Wall-clock is deliberately not
tabulated: the two recall blocks were captured with other runs in flight, so their timings are not
comparable and the difference would read as a backbone effect when it is a queue effect.

| Backbone | canonical | paraphrase (strict) | paraphrase (case-any) | shuffled-key control | gap (canon − para) | logP read Δ | note echo |
|---|---|---|---|---|---|---|---|
| gpt2-small (124 M) | 0.745 | 0.330 | 0.490 | 0.015 | **0.415** | 4.58 | 0.745 |
| gpt2-medium (355 M) | 0.825 | **0.483** | 0.675 | 0.015 | **0.343** | 5.97 | 0.825 |

### 14.2 The decision criteria on each backbone

| Criterion (first target token) | gpt2-small, paraphrase (n = 1 600) | gpt2-medium, paraphrase (n = 400) |
|---|---|---|
| Greedy recall | 0.3431 | **0.4825** |
| Target rank @1 | 0.1781 | 0.2600 |
| Target rank @5 | 0.4444 | 0.5575 |
| Target rank @10 | 0.5269 | 0.6400 |
| median rank | 8 | 4 |
| mean rank | 386.1 | 139.8 |
| rotated-note control, @10 | 0.0794 | 0.0550 |
| margin (correct − rotated) | 0.4475 | 0.5850 |
| likelihood duel win | 0.0663 | 0.1350 |
| @10 on greedy failures | 0.3806 (n = 1 051) | 0.4541 (n = 207) |

*Note on scope:* the recall rows above are matched two-slice comparisons; the criterion columns
are not (small is eight slices, medium is two). The same direction appears on both, and the small
column is quoted only to show what the verdict was before the backbone changed.

Four things follow.

1. **Capacity closes about a third of the gap.** Canonical +0.080, paraphrase +0.153, gap
   0.415 → 0.343 — 37 % of the deficit disappears over a 3× parameter range with no change to the
   store, the key or the prompt. So part of what §9 called a generalisation deficit is a capacity
   floor, and saying so is more accurate than saying the store "cannot" serve paraphrases.
2. **The rest is not capacity.** At 355 M the model still reads a paraphrase 0.48 of the time
   against 0.83 on a canonical prompt, while the target sits in the top 10 for 64 % of paraphrase
   items. The note is arriving and the decoder is still preferring a fluent continuation to a
   copied one — §12's mechanism survives the backbone change; only its size shrinks.
3. **The likelihood verifier does not improve in any usable way.** Duel win 0.066 → 0.135, median
   rank 8 → 4. Even at 3× capacity, prefixing the note barely moves the sequence likelihood, so a
   PMI-style re-ranker still cannot arbitrate a copied against a generated answer. §12 item 4
   retires that route at one scale; this retires it across the range we can afford to test.
4. **The control stays clean and the ceiling moves.** Rotated-note top10 is 0.055 on medium
   against 0.079 on small, so the margin is entirely the correct note. The measured ceiling for
   any copy-constrained read rises 0.527 → 0.640: that is the number a re-ranker would have to
   beat, and the verifier above does not.

**What this deliberately does not buy.** A gpt2-medium *end-to-end* number. Changing the backbone
replaces every baseline in §1–§13 — writer inner layers, gate thresholds, the budget optimum, the
window verdict — each of which was derived on gpt2-small, so a medium e2e cell would be a new
project dressed as a robustness check. The read-path probe is the bounded substitute: it prices
the capacity question where it is cheap to price, and it leaves the end-to-end system where it was
measured rather than where it was guessed.

Caveats: two slices, not eight; a 3× parameter range is a range, not a law — the paraphrase gap
may keep shrinking on a 10× backbone, which §10.3's cross-backbone template flip already hints at
(template ordering reverses between gpt2-small and gpt2-xl, so prompt-level conclusions must not
be carried across backbones); and the medium column is a clean-LM read path, not an edited model.

*（内容由AI生成，仅供参考）*

---

## 15. Claim ledger — every number this report has revised, and where it stands

Five revisions have each changed something. This table is the audit trail: what was claimed,
whether it still holds, and the section that carries it now. A reader who only has time for one
table should read this one before quoting anything.

| Claim | Status | Where |
|---|---|---|
| e2e 0.613 (Rev 3) | **superseded** by 0.619 after the read-path fix | §10.1, §1 |
| e2e **0.619 (0.585–0.652) at ΔNLL +0.398** | **current headline**, must travel with its NLL | §1 |
| Parameter channel 0.369 / memory +0.250 | unchanged across all five revisions | §1 |
| Paraphrase 0.343, borrowed from a clean-LM sweep | **superseded** by in-loop 0.333 (n = 5 412) | §10.1 |
| Paraphrase 0.333 / canonical 0.723, gap 0.39 | current; mechanism diagnosed in §12 | §10.1, §12 |
| `hybrid` 0.9025 as the two-channel ceiling | current, and only with the o200 caveat attached | §9.4 |
| Template ranking (prefix > instruct > fewshot) | **partly withdrawn** — canonical gap 0.33 → 0.08 held out | §9 item 2, §10.2 |
| In-loop `retro_fewshot` template swap | **rejected** (negative transfer, para 0.426 → 0.040) | §10.3 |
| Guarded repair pass | **rejected and switched off** (three tests, all negative) | §10.4, §9 item 5 |
| Budget 0.15 as an improvement | **not a headline** — 0.569 at ΔNLL 0.206, low-cost operating point | §11 verdicts |
| Budget 0.35 as an improvement | **closed** — strictly dominated, admission saturated at 0.25 | §11 verdicts |
| Budget 0.25 as tuned-in-range | **upgraded** to a measured interior optimum (four points) | §11 |
| W=25 as less interference | **falsified and closed** — flush count, not window length, drives interference | §11 verdicts |
| W=100 as a success gain | **in-sample only** (0.752); does not replicate (0.565 vs 0.570) — not claimed | §13.2, §13.3 |
| W=100 as a fluency lever | **current** — ΔNLL −0.237 [−0.303, −0.176], 4/4 paired slices | §13.3 |
| The eight evaluation slices as representative | **qualified** — 0.431 admission vs 0.245 on fresh slices | §13.4, §5 |
| "The store cannot serve paraphrases" | **softened** — ~37 % of the gap is a capacity floor | §14.2 |
| "The decoder will not copy them" | **survives** at 3× capacity (para 0.483 vs canonical 0.825) | §14.2 |
| Likelihood duel as a re-ranker | **retired at both scales** (0.066 → 0.135) | §12.1 item 4, §14.2 |
| Copy-constrained read ceiling | **measured**: ≤ 0.527 small, ≤ 0.640 medium — a bound, not a result | §12.1 item 3, §14.2 |
| gpt2-medium end-to-end number | **never run, not claimed** | §14 closing |

*（内容由AI生成，仅供参考）*
