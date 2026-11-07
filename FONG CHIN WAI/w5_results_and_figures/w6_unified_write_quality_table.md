# W6 input — one table for every "write quality" number in the project

**Purpose.** The project currently reports two numbers that look contradictory to a reader
who has not followed the W5 audit: **3.1%** (hypernetwork, real WRITE) and **78.8%**
(closed-form writer, final state). This file puts every write-quality figure in the project
on a single table with the measurement protocol attached to each row, so the W6
cross-review and the slides can quote them without mixing conventions.

**Rule of use.** Never quote a number from this table without its row's *protocol* and
*scope* columns. Rows with different `scope` values are not comparable and must not be
averaged, ranked, or placed in the same "improvement" claim.

**Row ownership (confirm or correct before submission).** Sonia (MC654953): rows 1-6 (audited
hypernetwork checkpoints), the system-level figures in §3, and the side-memory / router
evidence behind rows 15-16 and §4. Vivi (MC653748): rows 7-14 (closed-form writer, direct and
in-gate) and §4.1. Tang (MC631517): gate ablation rows. The ownership line is a request, not a
fact — `cross_review_requests.md` asks each owner to sign their own rows, because a table
written by one author and checked by nobody is not a cross-review.

---

## 1. Master table

| # | route / checkpoint | scope | protocol (what is measured) | n | result | retention | locality | fluency ΔNLL | evidence |
|---|---|---|---|---|---|---|---|---|---|
| 1 | W3 checkpoint (single-edit training) | CALM, gate included | per-edit audit: gate must route WRITE, write must land, model must answer | 128 real WRITEs of the 500 stream | **3.1% → 0.8%** (0.8% at W3 = 4/500 stream; audited W3-quality is the floor row of §2.1) | 0/9 | — | — | `w4_model_level_improve/results/audit_baseline.json` (baseline row) |
| 2 | accum16 + warm-start 48, 1000 steps (best hypernetwork) | CALM, gate included | same 128-write audit | 128 | **3.1% (4/128)** | 0/9 | — | — | `results/audit_baseline.json` |
| 3 | accum16 + warm48, coherence λ=0.1 | CALM, gate included | same 128-write audit | 128 | 3.9% (5/128) | — | — | — | `results/audit_coh_l0.1.json` |
| 4 | accum16 + warm48, coherence v2 (λ=0.3, c=0.6) | CALM, gate included | same 128-write audit | 128 | 2.3% (3/128) | — | — | — | `results/audit_coh_v2_l0.3_c0.6.json` |
| 5 | accum16 + warm48, v2 seed 101 | CALM, gate included | same 128-write audit | 128 | 1.6% (2/128) | — | — | — | `results/audit_v2_seed101.json` |
| 6 | any hypernetwork checkpoint, **raw** | CALM, gate included | `model_level_ok / 500` — counts REJECT false hits | 500 | 22.2% (111/500) | — | — | — | **inflated, do not quote** (§2.2) |
| 7 | closed-form writer, layers 6-8, batch=1 | direct writer, no gate | immediate per-write probe only | 128 | 100% immediate / **0% after re-prompt** | 0/50 | 0.400 | +2.501 | `results_base_b1_n128_L678` |
| 8 | closed-form writer, layers 6-8, one joint solve | direct writer, no gate | final-state re-prompt of all edited cases | 128 | 62.5% | 34/50 | 0.492 | +0.618 | `results_b128_n128_L678` |
| 9 | closed-form writer, layers 6-8, one joint solve | direct writer, no gate | final-state re-prompt, W4 scale | 500 | 42.4% | 17/50 | 0.530 | +1.060 | `results_b500_n500_L678` |
| 10 | **closed-form writer, layers 4-8, one joint solve** | direct writer, no gate | final-state re-prompt, W4 scale | 500 | **78.8%** | **39/50** | **0.569** | **+0.555** | `results_b500_n500_L45678` |
| 11 | closed-form writer, layers 4-8, `delta_cap=0.02` | direct writer, no gate | final-state re-prompt, W4 scale | 500 | **1.0%** (cap erases the write — negative result) | 0/50 | 0.974 | −0.042 | `results_b500_n500_L45678_cap002` |
| 12 | closed-form writer, `delta_rank=16` | direct writer, no gate | final-state re-prompt | 128 | 9.4% | 3/50 | 0.515 | +0.672 | `results_r16_b64_n128_L678` — negative result |
| 13 | closed-form writer + `replay_cache` | direct writer, no gate | final-state re-prompt | 128 | 85.9% immediate / 7/50 retention | 7/50 | 0.448 | +1.958 | `results_b32_replay_n128_L678` — negative result |
| 14 | MEMIT / AlphaEdit reproduction (reference, different model size) | single-fact, no stream | ROME/EasyEdit single-edit eval | 12 | exact rewrite 0.00–0.30 depending on judge level | — | 0.15–0.24 | — | `FONG HIO LAM/baseline_repro/` |
| 15 | closed-form writer, layers 4-8, **inside CALM** (per-request solve, dual-write on) | CALM, gate included | end-to-end: gate decides, then re-prompt every edit at the end of the stream | 200 | 0.5% (1/200); parameter channel 0.6% (1/173) | — | — | — | `EasyEdit/results_calm_n200_dual` |

Rows 1-6 are hypernetwork; rows 7-13 and 15 are the closed-form writer; row 14 is the
reference reproduction and is **not** part of the CALM stream at all.

**Row 15 is the answer to the "never run through the gate" gap in §4, and it is bad news:**
the same writer that scores 78.8% gate-bypassed scores 0.6% inside CALM. The cause is
structural, not a tuning issue — inside the gate loop each WRITE has to be solved and applied
before the next routing decision can be taken (the gate reads the model's post-write answer),
so the writer can only ever see a batch of one. The joint solve that produces 78.8% is exactly
the thing the sequential architecture forbids.

---

## 2. Why the headline pair (3.1% vs 78.8%) is not a contradiction

### 2.1 They measure different access paths

| | hypernetwork 3.1% | closed-form 78.8% |
|---|---|---|
| writer | hypernetwork emits a low-rank delta | closed-form least squares over the whole stream |
| gate | **inside CALM**: must route WRITE | bypassed: every edit is written |
| side memory | present, and ROUTE/REJECT answers are *excluded* from the W5 audit | absent |
| what a "hit" means | the model answers `target_new` **after** gate + write | the model answers `target_new` after write alone |
| denominator | 128 real WRITEs from the 500-edit stream | 500 edited cases |

So 3.1% answers *"how good is CALM end-to-end at model-level rewriting"* and 78.8% answers
*"how good is the closed-form writer, isolated"*. Both are true; they are answers to
different questions.

### 2.2 The 22.2% raw figure must never be quoted

Of the 111 raw hits in row 6, 107 are REJECT false hits (`current_answer_matches_target`,
i.e. the gate refused the write because the model already answered correctly) and ROUTE
contributes 0. Only 4 are parameter writes. Any slide that shows 22.2% as write quality is
inventing an improvement that does not exist.

### 2.3 The 90.8% v2 figure is superseded

`closedform_writer_v2.md` reported 90.8% immediate hits at layers 6-8. Row 7 shows why that
convention is unusable: at batch=1 the immediate hit is 100% and the final-state accuracy is
**0%**. v3 (`closedform_writer_v3.md`) replaces it with final-state accuracy; the
corresponding number is **78.8%**, not 90.8%.

---

## 3. Sentence to use in the report / slides

> Replacing the hypernetwork generator with a closed-form writer raises parameter-only
> model-level rewrite quality from **3.1% (4/128 audited real WRITEs) to 78.8% final-state
> accuracy with 39/50 long-horizon retention** on the same 500-edit CounterFact stream,
> at a fluency cost of +0.555 probe NLL and a 4.6% degenerate-output rate. The two figures
> differ in scope: 3.1% is measured inside the CALM gate, 78.8% with the gate bypassed.

If a single number is required for a slide, use **78.8% final-state accuracy, gate bypassed,
n=500** and put the 3.1% audited figure immediately next to it with its scope label.

---

## 4. Dual-channel attribution (parameter channel vs side memory)

The W4/W6 reports attribute results to two channels: the **parameters** (a WRITE, optionally
mirrored into side memory by `dual_write`) and the **side memory** (a ROUTE, and any
`verify_commit` rollback, which ends as a ROUTE). The smoke harness now reports both, per
edit, from `results_calm_n200_dual/summary.json`:

| quantity | value |
|---|---|
| edits in stream | 200 |
| gate decisions | WRITE 173 · ROUTE 27 · REJECT 0 |
| parameter channel (dual-write mirror on) | 173 edits, **1 hit = 0.6%** |
| parameter channel (no mirror) | 0 edits (dual-write was on for the whole run) |
| side-memory channel | 27 edits, 0 hits |
| aggregate end-to-end | 1/200 = 0.5%, degenerate 12.0% |
| memory entries stored | 200 (173 mirrors + 27 routes, 64 shards) |
| wall time | 419 s |

`memory_updated` is now returned truthfully per edit (`calm_editor._apply_decision` used to
discard the store result for WRITEs, which put every dual-written edit in the wrong bucket);
`w4_closedform_smoke.py` gained `--side-memory-dir / --dual-write / --write-budget /
--verify-commit` and emits `by_route` + `channels` blocks.

**Two things this table must not be used to claim.**

1. The CLOSED-FORM writer was never compared to the hypernetwork on the *same* 200-edit CALM
   slice with identical seeds; row 15 vs row 2 is n=200 vs n=128 and is indicative only.
2. The 27 side-memory edits read 0 hits because **this smoke harness has no memory reader** —
   `calm_editor` calls `memory.store(...)` in three places and never looks anything up on the
   answer path, so a ROUTEd fact is written and then never consulted. The memory channel's
   accuracy is therefore *not measurable here*; the system-level 97.48% / 9-of-9-retention
   number comes from the team's 5K harness, which does read memory back. Quoting "memory
   channel = 0%" from this run would be a harness artifact, not a finding.

Fix required before the dual-channel claim goes in the report: either plug `memory.lookup`
into the answer path of the smoke harness (a reader), or take both channels from the 5K
harness and use the smoke run only for the parameter channel.

### 4.1 Blocking design defect: the side memory's key contains the answer

`FONG HIO LAM/side_memory_router/side_memory.py` keys every entry on
`sha256(subject|relation|target_new)` and `lookup(subject, relation, target_new)` resolves that
same key:

```python
def _hash_key(subject, relation, target_new):
    raw = f"{subject}|{relation}|{target_new}"     # answer inside the key
def lookup(self, subject, relation, target_new):
    key = self._hash_key(subject, relation, target_new)
    return self._index.get(key, (-1, None))[1]     # hit == answer already known
```

Consequences, and they need to be stated before anyone quotes a memory-channel number:

1. **A memory hit is tautologically correct.** The only way `lookup` returns non-`None` is if
   the caller already passed the exact `target_new` that was stored. Any "memory channel
   accuracy" computed through this API would be 100% by construction, independent of the model.
2. **It is not retrieval.** There is no way to ask the memory "what do you know about
   *subject* / *relation*" — the query must already contain the answer. So the channel cannot
   be used by a reader that does not know the answer in advance.
3. Therefore the system-level 97.48% / 9-of-9 retention figure is a **routing-coverage**
   number (how many edits did CALM route instead of writing), not a memory-retrieval number,
   and the W5 audit's "ROUTE contributes 0" is a consequence of the answer path never
   consulting memory — not of memory being useless.

This is a side-memory (owner: Hoi Lam) item, not a W4 writer item, but it is **blocking** for
any per-channel table: fixing it means re-keying on `(subject, relation)` and making the
reader produce the answer span against the model's prompt, so a ROUTE answer can be wrong.

---

## 5. Residual risks this table does not fix

- Row 15 closes the "never run through the gate" gap, and the result is negative: the
  closed-form advantage is a *stream-level joint solve* that the sequential gate loop
  structurally forbids (§1, row 15 note). Writer replacement alone does not fix CALM.
- Degenerate rate behaves in the opposite direction under per-request solving (0.4–4.6% at
  joint solve, **12%** inside CALM), and the global `delta_cap` control is falsified (row 11).
  Only a per-request guard remains available.
- The dual-channel numbers are **parameter channel only** on the memory side (see the reader
  caveat above). Until a reader exists, any "model-level rewrite" claim must say
  "parameter-only".
- The MEMIT/AlphaEdit reproduction (row 14) is on gpt2-xl with n=12-50 and cannot be
  compared to the gpt2-small streams.
