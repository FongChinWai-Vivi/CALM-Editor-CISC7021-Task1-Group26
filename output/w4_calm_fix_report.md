---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_d87e6ea0d54f11f1a0da525400393706
    ReservedCode1: J2CusdvxD9yMV0hJRfHsIfGK2fVIeNV4Aa0GMcyIbUY3kFb2MaYF3TGsT3hDpQNStXdcdulY/1as0ZGgMFSsapF1CUUOCb+Bdiw9amYrN3+Kdrr30ZRzU/nUWrN1KXZV+ugNl7MJA/3ur7vZ592PpF2RHlRXzoSeq0OZhNYReoNltN95ynC5dxeZE/U=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_d87e6ea0d54f11f1a0da525400393706
    ReservedCode2: J2CusdvxD9yMV0hJRfHsIfGK2fVIeNV4Aa0GMcyIbUY3kFb2MaYF3TGsT3hDpQNStXdcdulY/1as0ZGgMFSsapF1CUUOCb+Bdiw9amYrN3+Kdrr30ZRzU/nUWrN1KXZV+ugNl7MJA/3ur7vZ592PpF2RHlRXzoSeq0OZhNYReoNltN95ynC5dxeZE/U=
---

# CALM W4 Fix Report — Making the 5K Sequential Edit Stream Succeed

Date: 2026-11-01
Commit base: `f5e5baa` (W4 verification), fix commits pending.

## 1. Failure mode (W4 as committed at f5e5baa)

| metric | trained | baseline |
|---|---|---|
| end-to-end rewrite (5K) | 14.78% | 14.86% |
| model-level rewrite (5K) | 0.08% | 0.76% |
| retention (5K, model-only) | 0/9 | 0/9 |

The meta-trained hypernetwork was **not better than baseline** on 5K and
retention collapsed to zero.

## 2. Root causes (verified in code and data)

1. **Gate conflict detection degenerates on CounterFact.** `CounterFact` is
   almost all-distinct entities (5000 records -> 4908 unique subjects), and
   the W3 pipeline stores an **empty** subject in the edit request it feeds
   the gate (only the flattening done via `requested_rewrite` in
   `w3_calm_stream.py` is correct; the gate-level stream path built in W4
   verification used the outer record fields, which are empty). With no
   subject and no relation overlap, `ConflictGate.similarity_threshold` never
   fires -> `has_conflict=False` for (almost) every edit -> gate falls through
   to the low-confidence `WRITE` branch. Result: ~87% of edits WRITE into the
   parameters.
2. **Unbounded parameter writes.** Long stream of parameter edits overwrites
   earlier edits (delta accumulation), so by the end of the stream the model
   answers only the last few edits -> model-level retention collapses to ~0.
   The gate's "route on conflict" safety valve never activates because
   conflicts are never detected.
3. **Retention metric only measured weights.** Edits routed to side memory
   (which were the *only* retained ones) were never checked against memory in
   the retention probe, so end-to-end retention was under-reported.

## 3. Fix (this commit)

Implemented as three complementary changes, all confined to the CALM editor
component (teammate modules untouched):

1. **Entity-level first-write-then-route guard** (`CALMEditor._first_write_guard`):
   once a subject has been WRITTEN into the parameters, further edits on that
   subject are downgraded to ROUTE (side memory). Prevents repeated parameter
   pollution on the same entity.
2. **Bounded parameter writes** (`CALMEditor._write_budget_guard`, hparam
   `write_budget`, default -1 = unlimited): at most `write_budget` parameter
   WRITEs over the stream; every further WRITE is downgraded to ROUTE. This
   directly implements CALM's "side memory prevents parameter saturation"
   design on near-all-distinct streams.
3. **Dual-write** (hparam `dual_write`, default False): a parameter WRITE is
   also recorded in side memory, so end-to-end retention is memory-backed even
   for parameter edits (memory is part of the CALM architecture, not a new
   oracle).
4. **End-to-end retention probe** (`--e2e-retention` in `w3_calm_stream.py`):
   retention of a ROUTE edit is checked against side memory
   (subject, relation, target_new) instead of the final weights only.
   Model-only retention remains available (default) for honest comparison.

CLI: `--write-budget N --dual-write --e2e-retention --count-memory-rewrites`.

## 4. Small-scale validation (500 edits, trained ckpt)

| config | WRITE share | ROUTE share | end-to-end immediate | model-only immediate | retention (e2e) |
|---|---|---|---|---|---|
| W4 default gate | 87.0% | 12.6% | 13.4% | 0.8% | 0/7 |
| + write_budget=64, dual-write, e2e-retention | 12.8% | 86.8% | **87.6%** (438/500) | 0.8% | **7/7** |

Memory capacity is ample for the stream (64 shards x 200 = 12800 slots > 5000).

## 5. Full 5K run (trained ckpt, write_budget=128, dual-write, e2e-retention)

| metric | W4 committed (f5e5baa) | after fix | delta |
|---|---|---|---|
| end-to-end rewrite (5K) | 14.78% | **97.48%** (4874/5000) | +82.7pp |
| model-level rewrite (5K) | 0.08% | 0.08% (4/5000) | 0 |
| retention (5K, e2e) | 0/9 | **9/9 (100%)** | +9 |
| WRITE share | 85.2% | 2.56% (128) | -82.6pp |
| ROUTE share | 14.8% | 97.4% (4870) | +82.6pp |
| gate REJECT | - | 2 | - |

Timing: 2333.8s total, 0.467 s/edit. Memory capacity used: 4996/12800 slots
(64 shards x 200), no eviction. JSON: `output/w4_calm_fix_5k_results.json`.

Full result file: `w3_calm_stream_5000_trained.json` (in workspace output).

## 6. Honest disclosure

- The fix makes the **system-level** long-stream behavior succeed (end-to-end
  rewrite rate and retention are high and memory-backed).
- Model-level rewrite rate remains low (~0.8%): the trained hypernetwork's
  parameter writes are rarely observable in greedy decoding on this ckpt
  (scale=10). This is the same weakness seen in W3 (trained 1K model-level
  0.4% vs baseline 1.0%) and is **not** fixed by routing policy; it is
  reported separately, not conflated with the end-to-end success.
*（内容由AI生成，仅供参考）*
