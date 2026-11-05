---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_f1e9d14cd59b11f180d5525400248c00
    ReservedCode1: B/wc3/MJGc/4Gv6Z2/+fl60yS5fH6fAdz6qHyX2IbO1PXYPWi/Q71FhuRCEDkRRCMEBzxg9kor/kouuNWrXrfVSty8OyMrr/njq3ucY133jEHx5ZnLxDwnQGzT2k5EoFwFZmmyyPXTdw+pLX/KCGUc2LlRhSMG70w/gZl3RcG1GxMMauUnN20dh/aLw=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_f1e9d14cd59b11f180d5525400248c00
    ReservedCode2: B/wc3/MJGc/4Gv6Z2/+fl60yS5fH6fAdz6qHyX2IbO1PXYPWi/Q71FhuRCEDkRRCMEBzxg9kor/kouuNWrXrfVSty8OyMrr/njq3ucY133jEHx5ZnLxDwnQGzT2k5EoFwFZmmyyPXTdw+pLX/KCGUc2LlRhSMG70w/gZl3RcG1GxMMauUnN20dh/aLw=
---

# CALM Model-Level Rewrite Improvement — Diagnostic Report

Date: 2026-11-01
Context: W4 5K stream model-level immediate rewrite rate stuck at 0.08%
(4/5000; only 4 of 128 parameter WRITEs observable in greedy decoding).
Question: can this be improved?

## 1. Scale sweep (inference-time override, no retraining)

60 CounterFact cases, single-edit patch on a clean baseline, `hypernet.scale`
overridden at inference (ckpt default = 5.0):

| scale | hit (target_new in greedy answer) | avg delta L2 norm |
|---|---|---|
| 0.05 | 1/60 (1.7%) | 1.28 |
| 0.5  | 2/60 (3.3%) | 12.77 |
| 1.0  | 3/60 (5.0%) | 25.53 |
| 2.0  | 12/60 (20.0%) | 51.06 |
| 5.0  | 23/60 (38.3%) | 127.66 |
| **10.0** | **27/60 (45.0%)** | 255.32 |
| 20.0 | 18/60 (30.0%) | 510.63 |
| 50.0 | 0/60 (0.0%) | 1276.58 |

Single-edit visibility can be improved by scaling (peak at scale=10, 45%),
at the cost of locality (20-case check: scale=5 → 3/20 leakage; scale=10 →
5/20 leakage).

## 2. Sequential-stream comparison (500 edits, write_budget=128 + dual_write)

| scale | model-level immediate (all 500) | within WRITE edits | e2e immediate |
|---|---|---|---|
| 5.0 | 4/500 (0.80%) | 4/131 (3.05%) | 373/500 (74.6%) |
| 10.0 | 2/500 (0.40%) | 2/129 (1.55%) | 373/500 (74.6%) |

In the stream, higher scale does **not** help; it hurts slightly. The
single-edit advantage disappears under accumulated deltas.

## 3. Cumulative-pollution diagnostic (root cause)

25/10 CounterFact edits, same ckpt, scale=5:

- Clean-baseline patch (restore to original params before each edit):
  **5/10 hit (50%)**.
- Cumulative-stream patch (never restore; each delta applied on top of all
  previous deltas): **2/10 hit (20%)**, with hits only at steps 0-1; from
  step 2 onward every new delta is unobservable in greedy decoding.

Conclusion: after ~2 accumulated parameter edits, newly generated deltas stop
producing observable rewrites. The trained hypernetwork assumes a clean base
model (training restores the snapshot every step, MEND-style single-edit
view), so it never learns to emit compensatory deltas on an already-modified
parameter stack. This fully explains the 5K stream result: the 128 WRITEs are
front-loaded, the first few hit, the rest miss → 0.08% overall.

## 4. Verdict and improvement path

- Scale / hyperparameter tuning alone: **cannot fix it** (experimentally
  falsified in §2).
- Root-cause fix: **change the training paradigm to simulate accumulation** —
  in `w3_hypernet_train.py`, do not restore the snapshot every step; instead
  accumulate deltas (or periodically restart) so the hypernetwork learns to
  generate effective deltas on an already-polluted parameter stack. Keep
  edit loss (target_new NLL) + locality loss (ground_truth NLL hinge).
- Expected effect: WRITE-internal hit rate could rise from ~3% toward the
  clean-baseline ~50% level; with 128 WRITEs / 5000 that would lift the
  model-level 5K metric from 0.08% to roughly 1.3%+; with a higher write
  budget, further.
- Side constraints: locality must be monitored (accumulation training tends
  to worsen ground-truth drift); retention probes stay e2e-backed.

## 5. Evidence files

- `temp/w4_scale_sweep.py` + output (scale sweep)
- `temp/w4_stream_scale_cmp.py` + output (500-edit stream comparison)
- `temp/w4_accum_diag.py` + output (cumulative-pollution diagnostic)
*（内容由AI生成，仅供参考）*
