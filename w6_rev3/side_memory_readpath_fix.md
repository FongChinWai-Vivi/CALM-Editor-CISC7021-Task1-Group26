---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_6ba588b6dd6f11f19232525400248c00
    ReservedCode1: WriMl0WvucxBJg2lzYemjz5XnqQrmiBBFMPpkCkiMvAjJqO2EkUtHNKd0KnuARok6tbL41MxXDh+0UHSVvfCcufloPgrpFUF9ui+4snyS1fnAI5xQQh49GmihqglvCBYjbGvcH92UYtWGWJnbJWso8raADI3vaPrEexbMcxN8kxKYuYglkVd2OEbZ24=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_6ba588b6dd6f11f19232525400248c00
    ReservedCode2: WriMl0WvucxBJg2lzYemjz5XnqQrmiBBFMPpkCkiMvAjJqO2EkUtHNKd0KnuARok6tbL41MxXDh+0UHSVvfCcufloPgrpFUF9ui+4snyS1fnAI5xQQh49GmihqglvCBYjbGvcH92UYtWGWJnbJWso8raADI3vaPrEexbMcxN8kxKYuYglkVd2OEbZ24=
---

# Side-memory read path: what was broken, what changed, what it is worth

Scope: the memory channel as it enters the Rev3 end-to-end number. Two files,
one store module (`FONG HIO LAM/side_memory_router/side_memory.py`) and one read
path (`EasyEdit/easyeditor/editors/calm_editor.py` + its driver
`EasyEdit/examples/w6_defer_stream.py`). Everything here is measured on the
same 8 slices, budget 0.25, W=50 as the frozen R3b sweep, so any delta below is
a one-flag difference, not a different operating point.

## 1. Why the routed rows looked dead

Rev3 reported `side_memory` routing as effectively unusable: across the eight
slices only a handful of edits landed on `side_memory` (1 row on o700, 0 on o0)
and of those almost none ended in a correct answer. The 8-rows-1-hit reading is
real but it was read as "the memory channel does not work". It is two separate
facts stacked on top of each other:

1. **Routing is dominated by the parameter channel.** With `write_budget`/
   `delta_budget` at 0.25 the gate admits ~0.43-0.63 of edits into the weights
   and the rest fall to `batched_route_fallback`; the explicit `side_memory`
   branch only sees the residue. So there is very little to measure there in the
   first place.
2. **A stored hit was not an answered question.** This is the actual defect. The
   read path put the note into the prompt (`memory_read_template=instruct`,
   "Answer with {note}."), asked the LM to re-emit it, and only counted a hit if
   the LM's own output matched. Measured copy fidelity on gpt2-small is ~0.44
   (16/36 on o0), so more than half of the rows where the store knew the exact
   answer were scored `none`. The memory channel was being judged on the
   backbone's copy behaviour, not on whether it held the edit.

Fact 2 is a measurement bug with a mechanical fix. Fact 1 is a design property
and is not "fixed" here — it is reported as its own column.

## 2. Store fixes (`side_memory.py`)

| # | Defect | Effect | Fix |
|---|---|---|---|
| S1 | `route()` used builtin `hash()` | `hash(str)` is salted per process (`PYTHONHASHSEED`), so a subject maps to a different shard between the run and the audit; shard LRU eviction therefore claims different entries on replay | SHA-256 digest routing (`int(digest[:16],16) % n_shards`) |
| S2 | shard-capacity eviction dropped entries only inside `_Shard.put` | the global `_index` and `total_stored` kept the evicted key, so `lookup()` could return a shard entry that no longer existed and a later `evict()` decremented for absent keys (`total_stored` could go negative) | eviction performed in `store()` before the write, index and counter updated with it; `evictions` counter exposed |
| S3 | key built from raw strings | `("Spain ", " Capital ")` and `("spain","capital")` were different keys, while the editor's version clock (`_mem_key`) already lower-cases and collapses whitespace — the same query could be a hit for the clock and a miss for the store | shared `_norm()` (whitespace collapse + lower) applied inside `_hash_key` and `route()` |
| S4 | a second edit on the same query was rejected as a duplicate | the store kept serving the superseded answer while the caller believed the edit landed (counterfactual-supersede hole) | `store(..., overwrite=False)`; default keeps the duplicate-rejection contract every existing table was measured under |

Smoke test (`python side_memory.py`) now asserts all four: case/space
invariance, sha256 route stability, `overwrite` semantics (False → refuse,
True → supersede), and `index_consistent=True` with `evictions=3` after
overflowing a 3-entry shard. It passed.

Note on S1-S4: none of them change the *hit rate* of a single-process run. They
change whether the number can be reproduced and whether the store's own counters
mean anything. They are correctness fixes, not score fixes — stated plainly so
nobody reads the table below as their effect.

## 3. Read-path fix (`calm_editor.py`, `w6_defer_stream.py`)

New hparam/flag `memory_serve` with three values; **default is `inject`, i.e.
Rev3 behaviour is unchanged and its numbers stay reproducible**.

| mode | behaviour | channel label |
|---|---|---|
| `inject` (default) | note goes in the prompt, LM must copy it; failure keeps the parameter answer | `memory` / `memory_arb` / `none` |
| `direct` | the store answers the query | `memory_direct` |
| `hybrid` | try injection, fall back to a direct serve when the LM fails to copy | `memory` / `memory_arb` / `memory_direct` |

`memory_direct` is a separate label on purpose: a retrieval serve must never be
quotable as free-generation recall. `read_stats` gains `memory_served`, and
`answered_by` gains `memory_direct`, so the mix is visible per slice.

Honest framing for the report: `direct`/`hybrid` is a non-parametric memory
answering from its own store — standard for a memory-backed editor, but it
measures *coverage and routing*, not the LM. The gap between `inject` and
`hybrid` is exactly the part of the ceiling that was being lost to copy failure.

## 4. Measured effect (8 slices, budget 0.25, W=50)

Filled from `w6run/R3c_o*_W50_b025_hybrid` against the frozen
`w6run/R3b_o*_W50_b025`.

| slice | R3b e2e (inject) | R3c e2e (hybrid) | Δ | R3b NLL Δ | R3c NLL Δ |
|---|---|---|---|---|---|
| o0 | 0.590 | 0.790 | +0.200 | 0.5258 | 0.5258 |
| o100 | 0.630 | 0.940 | +0.310 | 0.4336 | 0.4336 |
| o200 | 0.360 | 1.000 | +0.640 | 0.0000 | 0.0000 |
| o300 | 0.630 | 0.920 | +0.290 | 0.4831 | 0.4831 |
| o400 | 0.710 | 0.910 | +0.200 | 0.4017 | 0.4017 |
| o500 | 0.670 | 0.900 | +0.230 | 0.4470 | 0.4470 |
| o600 | 0.630 | 0.850 | +0.220 | 0.4360 | 0.4360 |
| o700 | 0.680 | 0.910 | +0.230 | 0.4595 | 0.4595 |
| **pooled** | **0.6125** | **0.9025** | **+0.290** | **0.3983** | **0.3983** |

Channel decomposition, pooled: parameter 0.3688 → 0.3688 (untouched), memory
0.2438 → 0.5337, `none` 0.387 → 0.094. Probe NLL delta is identical to four
decimals on every slice — the fallback answers questions the weights already
failed, it never touches the weights.

Measured copy fidelity (`memory_answered - memory_served` over `injected`) is
**0.35-0.53 per slice, 0.46 pooled** — i.e. the LM ignored the note it was
shown roughly half the time. That half is the whole delta above.

Per-slice detail on o0: parameter channel unchanged at 0.43 (43 rows), memory
channel 0.16 → 0.36 (16 → 36 rows, of which 20 are `memory_direct`), `none`
41 → 21.

**Read o200 before quoting it.** It is the slice where the parameter channel
collapsed (1 row admitted, 0.01 param share) and 99 edits fell to the fallback
route, so almost every row is served from the store: e2e 1.000 is the
architecture working, not a model doing anything. It is the clearest evidence
for the diagnosis (a store that knows the edit was being scored as a failure)
and the weakest evidence for "the editor is good". Report the pooled number and
this caveat together, or drop o200 and say so.

Honest framing for the report: `hybrid` raises the column because a
non-parametric memory is allowed to answer from its own store — standard for a
memory-backed editor (SERAC/IKE-style retrieval serving), but it measures
*coverage and routing*, not the LM. The 0.6125 `inject` number stays the
headline (it is what the parametric path does); 0.9025 is the ceiling of the
two-channel system, and the +0.290 gap is a quantified *diagnosis of the read
path*, not a model improvement. Do not print 0.9025 without 0.6125 next to it.

## 5. What remains open

- **Routing share is still gate-dominated.** `side_memory` retains ~0-1% of
  edits at budget 0.25 because `batched_route_fallback` absorbs the overflow.
  Serving those rows is what `hybrid` fixes; moving the *share* would need a
  gate/budget change, which is a different experiment.
- **Paraphrase recall (0.343) is untouched.** The store is an exact-match table
  on `(subject, relation)`; S3 removes surface noise (case/space) but not
  lexical variation. A normalisation or alias layer is the next lever, and it is
  a modelling change, not a bug fix — not claimed here.
- **Arbitration is conditioned on the parameter answer being present**
  (`memory_read_arbitrate` compares the injected answer against the base
  answer). When the base answer is already degenerate the comparison is
  meaningless; the current default is unchanged, flagged as a known limit.
*（内容由AI生成，仅供参考）*
