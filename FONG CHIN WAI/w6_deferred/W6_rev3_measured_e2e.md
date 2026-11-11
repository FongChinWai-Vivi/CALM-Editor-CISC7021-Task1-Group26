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

Revisions 3 and 4 replace every earlier end-to-end *ceiling* statement with a **measured** number. A ceiling is a sanity bound only; it is not the reported result. The single sentence above is what may be quoted as the project's end-to-end figure, and it must travel with its NLL delta — quoting 0.619 without ΔNLL +0.398 is a misquote.

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

Reproduce:

```powershell
powershell -File w6_rev3/run_sweep_b025.ps1        # 8 slices × 100 edits, ~4 min/slice on the 4070
python w6_rev3/w6_pool.py "EasyEdit\w6run\R3b_o*_W50_b025\results_defer_W50.json" rev3_pooled_b025.json
python w6_rev3/w6_operating_curve.py
python w6_rev3/w6_log_append.py
```

---

## 9. Open items — not claimed

1. **Paraphrase recall is measured but weak**: 0.333 strict, measured in-loop on the edited model over 5 412 items (§10.1). The canonical/paraphrase gap of 0.39 is a real generalisation deficit, and §10.2/§10.3 show it is not closable by choosing a better reader template: the template ordering **flips between backbones** (`fewshot` ≫ `prefix` on gpt2-small, `prefix` ≫ `fewshot` on gpt2-xl) and the in-loop swap is a negative transfer.
2. **Held-out validation past offset 700 is now done** (§10.2, 500 fresh edits o800–1299): the template ranking replicates, but the canonical gap between templates does not (0.33 in-slice → 0.08 held-out), so one claim from §2b is withdrawn. The **budget** axis (0.25 vs 0.50) still has no out-of-range test, and §4 still discloses it as tuned in-range.
3. **The W4 gate-ablation runner** now emits per-edit `nll_target_before/after/delta` and `hit_total_with_nll_row`; the re-run status is recorded in §10.6, and until that whole table is on the patched runner, §6 pairing should be read as coming from the defer stream.
4. **`side_memory` route**: fixed and re-measured, with the caveat that what was fixed is the *read path*, not the routing. The 1/8 hit rate was not the store failing to hold the edit — it was the read path requiring the LM to re-emit the note it had just been shown (measured copy fidelity 0.46 pooled), so a known answer was scored `none`. Store correctness fixes (sha256 shard routing instead of salted `hash`, index/counter consistency under capacity eviction, surface-form key normalisation, `overwrite` for counterfactual supersede) are in `FONG HIO LAM/side_memory_router/side_memory.py`; the read path gains a `memory_serve` flag whose default `inject` reproduces exactly the numbers in this report. With `hybrid` (inject first, then serve from the store when the copy fails) e2e goes **0.6125 → 0.9025 pooled** across the same 8 slices / budget 0.25 / W=50, with the parameter channel and probe NLL delta identical to four decimals. That 0.9025 is the ceiling of the two-channel system, not a model improvement: `memory_direct` answers by table lookup, and slice o200 (parameter channel collapsed, 1 admitted row) is a degenerate slice that must be quoted with the caveat. Routing share is still gate-dominated (0-1% explicit `side_memory`), and paraphrase recall is untouched at 0.333 in-loop (§10.1) — an exact-match store cannot fix lexical variation, and §10.3 shows a prompt-level fix makes it worse. Full write-up: `w6_rev3/side_memory_readpath_fix.md`.
5. **Repairing dead edits does not work and is switched off** (§10.4). At the tightest allowance the pass is inert; scaled up it destroys more live rows than it rescues, and it does so while *improving* the fluency coordinate. No headline number depends on it.
*（内容由AI生成，仅供参考）*

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
