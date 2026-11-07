---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_cc609412d79611f1a216525400248c00
    ReservedCode1: cy3f46rs3HCXIPdgn+ckMR1r8FiBuAKT2uwzSIsRUpvG8wNUuqfbpDOoAicC0+vEAvcCU6ZyJJBh32zjQfE1C6covFho9B8MstZvjnegx5Zh9Yy2PJJkzbsVFmlbsIAm5vza9thMJP19y6q4oeNNjpovMxM7A1lSzeazOxk4gKITokOrSf3EgJ6xxiU=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_cc609412d79611f1a216525400248c00
    ReservedCode2: cy3f46rs3HCXIPdgn+ckMR1r8FiBuAKT2uwzSIsRUpvG8wNUuqfbpDOoAicC0+vEAvcCU6ZyJJBh32zjQfE1C6covFho9B8MstZvjnegx5Zh9Yy2PJJkzbsVFmlbsIAm5vza9thMJP19y6q4oeNNjpovMxM7A1lSzeazOxk4gKITokOrSf3EgJ6xxiU=
---

# Main Results Summary (W2–W4, with W5 audit correction)

> Compiled 2026-11-03 for the W4→W5 handover.
> All numbers come from the repository artifacts listed in [Data sources](#data-sources).
> The W5 per-edit audit corrects the previously reported W4 model-level WRITE-hit numbers; the raw (pre-audit) figures are kept for transparency and marked as such.

## 1. W2 — Single-fact editing baselines (gpt2-xl, n=50)

Same evaluation protocol (`FONG CHIN WAI/w1_framework_and_protocol/evaluation_protocol.md`): reliability (`rewrite_acc`), generalisation (`rephrase_acc`), locality (`neighborhood_acc` on CounterFact / `nq_acc` on zsRE), portability (`p1_acc` on MQuAKE).

| Method | Dataset | Reliability | Generalisation | Locality | Portability |
|---|---|---|---|---|---|
| ROME (parametric) | CounterFact | 1.000 | 0.700 | 0.842 | — |
| ROME (parametric) | zsRE | 0.997 | 0.807 | 0.966 | — |
| ROME (parametric) | MQuAKE | 0.947 | — | — | 0.069 |
| IKE (memory-based) | CounterFact | 1.000 | 0.720 | 0.188 | — |
| IKE (memory-based) | zsRE | 0.993 | 0.985 | 0.670 | — |
| GRACE (memory-based) | CounterFact | 1.000 | 0.000 | 1.000 | — |
| GRACE (memory-based) | zsRE | 1.000 | 0.000 | 1.000 | — |

Notes:
- ROME GPT-J-6B smoke (zsRE n=10, Colab T4 16GB): rewrite 1.000 / rephrase 0.950 / nq 0.610; full 6B evaluation is infeasible on the local 8GB RTX 4070.
- GRACE rephrase 0.000 is a method property (keyed on the last prompt token), not a bug.

## 2. W3/W4 — CALM sequential editing streams (gpt2-small, CounterFact)

| Stream | Variant | WRITE | ROUTE | REJECT | e2e immediate | model-level immediate | retention |
|---|---|---|---|---|---|---|---|
| W3 1K | trained hypernet | ~85.5% | ~14.1% | ~0.4% | 17.2% | 0.4% | 0/9 |
| W3 1K | baseline (no hypernet) | ~85.1% | ~14.1% | ~0.8% | 15.1% | 1.0% | 0/9 |
| W4 5K | trained hypernet | 85.2% | 14.7% | 0.1% | 14.78% | 0.08% | 0/9 |
| W4 5K | baseline (no hypernet) | 85.6% | 14.2% | 0.2% | 14.86% | 0.76% | 0/9 |
| W4 5K | **+ write_budget=128, dual-write (fix)** | 2.56% (128) | 97.4% (4870) | 2 | **97.48%** | 0.08% | **9/9** |

Key findings:
- Scaling 1K→5K degrades end-to-end immediate rewrite for both configs (trained 17.2%→14.78%, baseline 15.1%→14.86%); at 5K the meta-trained hypernetwork no longer beats the baseline.
- The bounded-write fix makes the *system-level* lifelong stream succeed: 97.48% end-to-end rewrite, 9/9 memory-backed retention.

## 3. W4 — Model-level write-quality improvement (500-edit stream, write_budget=128, dual-write)

### 3.1 Training-configuration sweep (raw, pre-audit)

| Training configuration | Model-level immediate | WRITE hit rate | Audited real WRITE hit (128-write audit) |
|---|---|---|---|
| W3 old ckpt (single-edit) | 4/500 (0.8%) | 3.1% | un-audited |
| accum, segment_len=4 | 8/500 (1.6%) | 6.3% | un-audited |
| accum, segment_len=16 | 51/500 (10.2%) | 39.8% | un-audited |
| accum16 + gate-distribution sampling | 32/500 (6.4%) | 25.0% | un-audited |
| accum, segment_len=32 | 37/500 (7.4%) | 28.9% | un-audited |
| accum16 + warm-start 32 | 56/500 (11.2%) | 43.8% | un-audited |
| accum16 + warm-start 48 (best) | 85/500 (17.0%) | 66.4% | un-audited |
| accum16 + warm-start 64 | 35/500 (7.0%) | 27.3% | un-audited |
| **accum16 + warm-start 48, 1000 steps** | **111/500 (22.2%)** | **86.7%** | **4/128 (3.1%)** |

### 3.2 W5 per-edit audit (128-write audit on real stream WRITEs)

| Checkpoint | Raw model-level ok | WRITE.rewrite_ok | Real WRITE hit rate | REJECT.rewrite_ok | ROUTE.rewrite_ok | Audit JSON |
|---|---|---|---|---|---|---|
| baseline (accum16+warm48·1000) | 111 | 4 | **3.1% (4/128)** | 107 | 0 | `FONG CHIN WAI/w4_model_level_improve/results/audit_baseline.json` |
| coh_l0.1_c0.6 | 87 | 5 | **3.9% (5/128)** | 82 | 0 | `FONG CHIN WAI/w4_model_level_improve/results/audit_coh_l0.1.json` |
| coh_v2_l0.3_c0.6 | 31 | 3 | **2.3% (3/128)** | 28 | 0 | `FONG CHIN WAI/w4_model_level_improve/results/audit_coh_v2_l0.3_c0.6.json` |
| v2_seed101 | 42 | 2 | **1.6% (2/128)** | 40 | 0 | `FONG CHIN WAI/w4_model_level_improve/results/audit_v2_seed101.json` |

Audit conclusion:
- The pre-audit "WRITE hit rate" (`model_level_ok/128`) counts every gate-level success, including REJECTed edits where the model already answers correctly. For the best checkpoint, 107 of the 111 raw hits are REJECT false hits and ROUTE contributes 0; the genuine parameter-write hit rate is only **4/128 (3.1%)**, i.e. 124/128 real WRITEs still miss.
- The audited improvement is therefore **0.08% → 3.1%** on real WRITEs (not 0.08% → 22.2%).
- Coherence-penalty and multi-seed routes do not exceed the baseline's real write quality (5/128, 3/128, 2/128 vs 4/128); all are dominated by REJECT false hits.

![Audited real WRITE hit rate per checkpoint](w5_audited_real_write_hit.png)

![Raw model-level vs audited real WRITE hit](w5_raw_vs_audited_real_write_hit.png)

### 3.3 Writer replacement: closed-form (MEMIT-style) generator

The audit above measures the *hypernetwork* route. A second route was built in parallel:
replace the hypernetwork edit generator with a MEMIT/AlphaEdit-style closed-form solver that
takes one joint least-squares over the whole edit stream instead of one edit at a time.
Full method and negative results: `w4_model_level_improve/closedform_writer_v3.md`.

Metric convention: `final_acc` re-prompts **every** edited case after the stream ends. The
earlier `immediate_hit` convention (probe right after each write) is not a stream-quality
metric — at batch=1 it reads 100% while the final-state accuracy is 0%.

| configuration | immediate_hit | final_acc | recheck(50) | ΔNLL | degen | nbhd top1 | generality |
|---|---|---|---|---|---|---|---|
| layers 6-8, batch=1 (v2 convention) | 1.000 | 0.000 | 0/50 | +2.501 | 10.9% | 0.400 | 0.00 |
| layers 6-8, batch=32 | 0.867 | — | 7/50 | +1.141 | 0.8% | 0.536 | 0.10 |
| layers 6-8, batch=64 | 0.766 | — | 21/50 | +0.971 | 0.8% | 0.461 | 0.28 |
| layers 6-8, one joint solve (n=128) | 0.625 | 0.625 | 34/50 | +0.618 | 1.6% | 0.492 | 0.40 |
| layers 6-8, one joint solve (n=500) | — | 0.424 | 17/50 | +1.060 | 0.4% | 0.530 | 0.10 |
| **layers 4-8, one joint solve (n=500)** | — | **0.788** | **39/50** | **+0.555** | 4.6% | **0.569** | **0.40** |
| layers 4-8, joint solve + `delta_cap=0.02` (n=500) | — | 0.010 | 0/50 | −0.042 | 0.0% | 0.974 | — |

The `delta_cap` row is a falsification, not a tuning result: 0.02·`||W||_F` ≈ 3–4 against a
solved delta of ~70–80 per layer, so the cap erases the write. Final accuracy 1.0% sits at the
pre-edit baseline (0.8%), retention is 0/50, and the NLL *improves* because nothing was written.
A degenerate rate of 0% under a cap that writes nothing carries no information — the degenerate
cases (23/500, all verbatim repetition of `target_new`) need a per-request control, not a
stream-wide norm budget.

Reading:
- The hypernetwork's limiting factor was the *solve scope*, not the conditioning: solving the
  whole stream jointly lifts long-horizon retention from 0/50 to 39/50 and cuts the fluency
  cost from +2.501 to +0.555 NLL.
- Widening from three c_proj layers (6-8) to five (4-8) is what raises final accuracy
  (0.424 → 0.788); the capacity probe (effective writable rank ≈75/3072, same-relation key
  directions overlapping at 0.63) says more bandwidth, not better regularisation, is the lever.
- Two routes were falsified and are documented so they are not retried: `delta_rank=16`
  truncation (final accuracy 0.094 — a rank-16 Δ cannot satisfy 640 constraints) and
  `replay_cache` (retention stays 7/50, ΔNLL worsens to +1.958 — historical residuals drift
  once the weights move).
- **Scope caveat, do not drop it:** the 78.8% is a *direct writer, gate-bypassed*
  measurement. The audited 3.1% above is *end-to-end inside CALM*. The two are answers to
  different questions; the unified table with per-row protocols is
  `w6_unified_write_quality_table.md`.

#### End-to-end check: the writer inside the gate

The gate-bypassed figure has now been paired with a gate-included run of the same writer
(layers 4-8, dual-write on, side memory live, n=200, `EasyEdit/results_calm_n200_dual`):

| scope | n | result | degen |
|---|---|---|---|
| direct writer, joint solve (gate bypassed) | 500 | 78.8% | 4.6% |
| **inside CALM, per-request solve (gate included)** | 200 | **0.5%** (parameter channel 1/173 = 0.6%) | **12.0%** |

Reason it is not a tuning gap: the gate's decision depends on the model's post-write answer,
so each WRITE must be solved and applied before the next decision — the writer never sees more
than a batch of one, which is the regime that measured 100% immediate / 0% final in §3.3. The
joint solve is a stream-level optimisation the sequential architecture forbids. Writer
replacement therefore does not fix end-to-end CALM; batched/deferred writes would.

## 4. Bottom line

- **System level:** the bounded-write + dual-write fix solves lifelong retention end-to-end (97.48% rewrite, 9/9 retention on 5K CounterFact).
- **Model level:** the earlier 22.2% / 86.7% claims were inflated by REJECT false hits; real parameter-WRITE quality is ~1.6–3.9% across audited checkpoints. Improving it remains an open problem (further warm-started training, lr decay, longer steps).
- **Writer level (new):** the failure was in the writer's *solve scope*, not in the training recipe. A closed-form joint solve over the stream lifts parameter-only model-level rewrite to **78.8% final-state accuracy with 39/50 retention** and a **+0.555 NLL** cost; the same writer at batch=1 shows 100% immediate and 0% final, which is exactly the inflation the audit was written to catch.
- **Architecture limit (new):** the closed-form writer drops to **0.6% inside the CALM gate**, because the gate forces one-edit-at-a-time solving. The writer is not the end-to-end bottleneck; the sequential write loop is. Batched/deferred writes are the only route to bringing the joint solve inside CALM.
- **Channel accounting (new):** the dual-channel table cannot be completed as things stand — `side_memory.py` keys entries by a hash containing `target_new`, so a memory hit is tautologically correct, and the harness never reads memory on the answer path. The 97.48% system figure is a routing-coverage number, not a memory-retrieval number.

## Reproduction

`w4_w5_results_reproduce.ipynb` is the single entry point for every number above:

- **Part A** rebuilds the tables in this file and both figures straight from
  `FONG CHIN WAI/w4_model_level_improve/results/audit_*.json` and the
  `eval_*/w3_calm_stream_*.json` run records. Seconds, no GPU.
- **Part B** re-executes the three pipelines locally against `./.venv`
  (RTX 4070 Laptop; 5 training steps + 16 edits, ~75 s total) to confirm the scripts still
  resolve their repo root, `EasyEdit/data` and `EasyEdit/hparams` after the
  `FONG CHIN WAI/` consolidation. Part B is a plumbing check and yields no reportable
  metrics - only Part A carries the numbers.

The notebook ships with its outputs stored and drives everything through `subprocess`, so it
runs on any Python 3 kernel.

## Data sources

- `FONG CHIN WAI/w2_rome_baseline/w2_rome_reproduction.md`, `docs/w2_ike_grace_reproduction.md` (W2 baselines)
- `FONG CHIN WAI/w3_hypernetwork_stream/w3_calm_stream_1k.json`, `FONG CHIN WAI/w4_sequential_stream_and_fix/w4_sequential_5k_results.json`, `FONG CHIN WAI/w4_sequential_stream_and_fix/w4_calm_fix_5k_results.json` (W3/W4 streams)
- `FONG CHIN WAI/w4_model_level_improve/results/audit_*.json` (W5 audit)
- `FONG CHIN WAI/w4_model_level_improve/closedform_writer_v3.md` and
  `EasyEdit/results_b*/summary.json` (closed-form writer, §3.3)
- `FONG CHIN WAI/w5_results_and_figures/w6_unified_write_quality_table.md` (all write-quality
  numbers on one table, with the measurement protocol of each row)
- `README.md` "W4 — Model-level write quality improvement"
