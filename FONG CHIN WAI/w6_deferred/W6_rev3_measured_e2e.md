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

> **With the memory channel actually read at answer time, end-to-end edit success on 8 held-out slices (800 edits, W=50) is 0.613 (95% CI 0.579–0.646) at ΔNLL +0.398 per 100 edits; the parameter channel alone measures 0.369 on the same rows, so the memory channel contributes +0.244 absolute. The stored facts behind that channel are themselves measured on the same 8 slices at n=100 each: canonical recall 0.723, paraphrase recall 0.343 (case-insensitive 0.526), against a shuffled-key control of 0.013. 3-seed random-subset means: 0.623 / 0.623 / 0.594.**

Revision 3 replaces every earlier end-to-end *ceiling* statement with a **measured** number. A ceiling is now a sanity bound only; it is not the reported result. The single sentence above is what may be quoted as the project's end-to-end figure, and it must travel with its NLL delta — quoting 0.613 without ΔNLL +0.398 is a misquote.

| Quantity | Value | How it was obtained |
|---|---|---|
| End-to-end success, memory in-loop | **0.613** | recomputed from `rows` of 8 slices, not from summaries |
| 95% CI (normal) | 0.579 – 0.646 | per-observation SE over 800 pooled rows |
| 95% CI (bootstrap, 2 000 resamples) | 0.579 – 0.645 | non-parametric, same pool |
| 3-seed random subset (60 % each) | 0.623 / 0.623 / 0.594 | seeds 0/1/2, the Vivi ② protocol |
| Parameter channel only, same rows | 0.369 | in-run counterfactual, no extra model run |
| Memory channel contribution | **+0.244** | difference of the two channels on identical rows |
| Fluency cost | ΔNLL **+0.398** / 100 edits | mean of the 8 per-slice `nll_delta` |
| Degenerate-answer rate | 0.01 | guard active; see §5 |

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

Paraphrase denominators are 200 per slice (two paraphrases per edit); all other columns use the 100 routed edits. Strict matching is exact-string, case-any is case-insensitive; both are reported so the stricter one cannot be mistaken for the headline.

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

Reproduce:

```powershell
powershell -File w6_rev3/run_sweep_b025.ps1        # 8 slices × 100 edits, ~4 min/slice on the 4070
python w6_rev3/w6_pool.py "EasyEdit\w6run\R3b_o*_W50_b025\results_defer_W50.json" rev3_pooled_b025.json
python w6_rev3/w6_operating_curve.py
python w6_rev3/w6_log_append.py
```

---

## 9. Open items — not claimed

1. **Paraphrase recall is measured but weak**: 0.343 strict. The canonical/paraphrase gap of 0.38 is a real generalisation deficit; the case-any column (0.526) shows roughly a fifth of that gap is surface formatting rather than semantics.
2. **Held-out budget validation** beyond offsets 700 is not done; §4 discloses the tuning.
3. **The W4 gate-ablation runner** now emits per-edit `nll_target_before/after/delta` and `hit_total_with_nll_row`, but that arm has **not been re-run**; the pairing in §6 comes from the defer stream, not from the ablation table.
4. **`side_memory` route**: fixed and re-measured, with the caveat that what was fixed is the *read path*, not the routing. The 1/8 hit rate was not the store failing to hold the edit — it was the read path requiring the LM to re-emit the note it had just been shown (measured copy fidelity 0.46 pooled), so a known answer was scored `none`. Store correctness fixes (sha256 shard routing instead of salted `hash`, index/counter consistency under capacity eviction, surface-form key normalisation, `overwrite` for counterfactual supersede) are in `FONG HIO LAM/side_memory_router/side_memory.py`; the read path gains a `memory_serve` flag whose default `inject` reproduces exactly the numbers in this report. With `hybrid` (inject first, then serve from the store when the copy fails) e2e goes **0.6125 → 0.9025 pooled** across the same 8 slices / budget 0.25 / W=50, with the parameter channel and probe NLL delta identical to four decimals. That 0.9025 is the ceiling of the two-channel system, not a model improvement: `memory_direct` answers by table lookup, and slice o200 (parameter channel collapsed, 1 admitted row) is a degenerate slice that must be quoted with the caveat. Routing share is still gate-dominated (0-1% explicit `side_memory`), and paraphrase recall 0.343 is untouched — an exact-match store cannot fix lexical variation. Full write-up: `w6_rev3/side_memory_readpath_fix.md`.
*（内容由AI生成，仅供参考）*
