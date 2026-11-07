# W4 model-level write quality: closed-form (MEMIT-style) writer, v3

**Scope.** Replace CALM-Editor's hypernetwork edit generator with a MEMIT/AlphaEdit-style
closed-form writer, and close the three limitations recorded in `closedform_writer_v1.md`
/ `v2.md`: (i) the locality metric was unmeasurable, (ii) long-horizon retention collapsed
to 0/50, (iii) fluency cost was prohibitive.

**Headline.** On the W4 scale (500 sequential edits, CounterFact, gpt2-small, direct writer
mode), the closed-form writer now reaches **78.8% final-state rewrite accuracy** with
**39/50 early-edit retention** and a **+0.555 probe-NLL cost**, against **3.1% (4/128)**
audited real-write quality for the hypernetwork route. All three limitations are closed;
what remains open is the trade-off, not the metric.

---

## 1. Metric definitions

The v2 report quoted a 90.8% "hit rate". That number was measured **immediately after each
individual write**, which is structurally inflated: a later edit can overwrite an earlier
one and the immediate probe never sees it. v3 reports the metric W4 actually needs and keeps
both numbers so the difference is auditable.

| metric | definition | why it is reported |
|---|---|---|
| `immediate_hit` | after each write, 5-token greedy decode of that edit's prompt; hit if the answer contains `target_new` | comparable to the v2 / hypernetwork per-write convention, but **not** a stream-quality metric |
| `final_acc` (`rewrite_ok_rate`) | after the whole stream, re-prompt **every** edited case; fraction whose answer contains `target_new` | the W4 model-level metric; this is what "the weights remember" means |
| `recheck(50)` | the same re-prompt restricted to the first 50 edits | long-horizon retention |
| `degen` | empty answer, zero-width characters, or ≥4 repeated tokens | detects "hit but broken" outputs |
| `probe NLL` | mean token NLL over 12 fixed neutral English prose passages (pre-edit 3.5687) | fluency cost of sequential writing |
| `neighbourhood top1_unchanged` | over CounterFact `neighborhood_prompts` (2 per case, 500 cases → 1000 probes), fraction whose next-token top-1 is unchanged | locality; replaces the broken held-out metric |
| `mean Δlogp` | mean change in the logprob of the neighbourhood prompts' original top-1 token | graded locality signal |
| `generality` | fraction of a held-out paraphrase set answered with `target_new` | transfer of the written fact |

### 1.1 Why the old locality metric was void

The `heldout` probe was scored by exact match against `target_new`. Unedited facts never
answer with `target_new`, so the pre-edit baseline is **0/50** and any "retention" ratio has
a zero denominator. The metric was reporting an artifact, not locality. v3 replaces it with
the neighbourhood top-1 criterion above, which has a real pre-edit baseline (1.000 by
construction) and a real post-edit signal.

---

## 2. Recommended configuration

```
writer       = closedform           # MEMIT / AlphaEdit-style closed form
layers       = 4,5,6,7,8            # five c_proj layers (was 6,7,8)
l2 / mode    = 1.0 / rel            # lambda = 1.0 * trace(KK^T)/n_ctx
z_clamp      = 0.6                  # |z| <= 0.6 * mean|h_z|
resid_ref    = z_layer              # MEMIT convention, distributed 1/(N-i)
batch_solve  = full stream          # one joint least-squares over all requests
delta_rank   = 0                    # low-rank truncation off (see §6.1)
delta_cap    = off                  # Frobenius cap off unless used as an ablation (§5.3)
replay_cache = off                  # see §6.2
```

`batch_solve` is the single most consequential knob: it is the difference between an
edit-at-a-time least squares (which re-occupies the rank direction of the previous edit) and
one joint solve over all constraints in the stream.

---

## 3. Implementation

`EasyEdit/easyeditor/models/calm/closed_form_writer.py`
- `generate_update_batch`: joint solve over a whole batch of requests — shared keys/residuals
  across layers, one least-squares per layer, chunked to bound memory.
- `_limit_delta` with `delta_rank` (SVD truncation) and `delta_cap`
  (`||Δ||_F ≤ cap·||W||_F`).
- `replay_cache`: carries historical chunk (keys, residual) pairs into later solves.

`EasyEdit/easyeditor/editors/calm_editor.py`
- `hypernet_alg: closedform` branch. Same interface as the hypernetwork generator
  (`generate_update` / `apply_update` / `generate_and_apply` / `summary` /
  `last_apply_stats`), so routing, budget and verify-commit logic are reused unchanged.

`EasyEdit/examples/w4_closedform_smoke.py`
- `top1_snapshot` / `locality_drift` / `first_token_accuracy` (new locality metrics),
  the batch-solve loop, and the `--batch-solve`, `--replay-cache`, `--delta-rank`,
  `--delta-cap`, `--locality-n`, `--recheck-n` flags.

---

## 4. Results — batch size sweep (n=128, layers 6/7/8)

| batch_solve | immediate_hit | final_acc | recheck(50) | ΔNLL | degen | nbhd top1 | generality |
|---|---|---|---|---|---|---|---|
| 1 (v2 baseline) | 1.000 | 0.000 | 0/50 | +2.501 | 10.9% | 0.400 | 0.00 |
| 32 | 0.867 | — | 7/50 | +1.141 | 0.8% | 0.536 | 0.10 |
| 64 | 0.766 | — | 21/50 | +0.971 | 0.8% | 0.461 | 0.28 |
| 128 (full) | 0.625 | 0.625 | 34/50 | +0.618 | 1.6% | 0.492 | 0.40 |

The sweep is monotone in the direction that matters: the larger the solve, the better the
retention, fluency and generality, and the lower the *immediate* hit. The immediate hit
falling is expected and harmless — it is the inflated convention. `final_acc` for batch=1 is
0.000, i.e. **the v2 90.8% number does not survive a single re-prompt.**

---

## 5. Results — 500-edit stream (W4 scale)

### 5.1 Layer depth

| config | final_acc | recheck(50) | ΔNLL | degen | nbhd top1 | generality | wall time |
|---|---|---|---|---|---|---|---|
| layers 6-8, batch=500 | 0.424 | 17/50 | +1.060 | 0.4% | 0.530 | 0.10 | 632 s |
| **layers 4-8, batch=500 (recommended)** | **0.788** | **39/50** | **+0.555** | 4.6% | **0.569** | **0.40** | 816 s |

Widening from three to five layers is the only lever that actually raised final accuracy,
and it follows from the capacity probe: the effective writable rank of the c_proj stack is
only ~75 of 3072, and same-relation key directions already overlap at 0.63, so *more
bandwidth*, not better conditioning, is what buys accuracy.

### 5.2 Against the hypernetwork route

| route | metric | value | note |
|---|---|---|---|
| hypernetwork (best checkpoint, warm48·1000) | audited real WRITE hit, 128-write protocol | **3.1% (4/128)** | `results/audit_baseline.json` |
| closed-form, layers 6-8 | final_acc, 500-edit stream | 42.4% | same protocol family, direct writer mode |
| closed-form, layers 4-8 | final_acc, 500-edit stream | **78.8%** | recommended configuration |

The two routes are **not** measured on the same access path: the hypernetwork number is
audited inside the CALM gate (so a real WRITE must also be routed to WRITE and survive the
gate), while the closed-form number is a direct-writer measurement with no gate. Both are
parameter-only writes, neither has side-memory help. §5.4 states the consequence plainly.

### 5.3 `delta_cap` ablation (degenerate-rate control)

The only defect that grew with the layer widening is the degenerate rate (0.4% → 4.6%).
`--delta-cap 0.02` was run as the control: it caps `||Δ||_F` at 2% of the layer's `||W||_F`.

| config | final_acc | recheck(50) | ΔNLL | degen | nbhd top1 | wall time |
|---|---|---|---|---|---|---|
| layers 4-8, no cap | 0.788 | 39/50 | +0.555 | 4.6% | 0.569 | 816 s |
| layers 4-8, `--delta-cap 0.02` | **0.010** | **0/50** | **−0.042** | **0.0%** | **0.974** | 915 s |

**This is a falsification, not a fix.** The cap turns the writer into a no-op: `||W||_F` for
these `c_proj` layers is 140–208, so 0.02·`||W||_F` ≈ 3–4 while the natural solved delta is
~70–80 per layer. Capping at ~5% of the delta's own magnitude removes the write itself —
hence final accuracy 1.0% (against a 0.8% pre-edit baseline), forgets everything (0/50), and
leaves the model untouched (neighbourhood top-1 unchanged at 97.4%, NLL *below* the pre-edit
value). The degenerate rate going to 0% here is an artifact of writing nothing.

Conclusion: **a global Frobenius cap is the wrong instrument for the degenerate rate.** The
degenerate cases are a per-request over-promotion phenomenon — 23/500 cases where the model
repeats `target_new` to the 5-token limit (e.g. `'Vienna Vienna Vienna Vienna Vienna'`) —
so the control has to be per-request (bound the promoted logit gap, or shrink that one
request's delta until the repetition disappears), not a stream-wide norm budget.

### 5.4 What the two numbers mean together

A reader who sees "3.1%" and "78.8%" side by side will assume a contradiction. They are not
the same quantity:

* 3.1% is *end-to-end inside CALM*: the gate must route to WRITE, the write must land, and
  the model must answer. It also includes the hypernetwork's own generator limitations.
* 78.8% is *the writer alone, no gate*: every edit is written and then re-prompted.

So the honest reading is: **the closed-form writer removes the writer as the bottleneck
(78.8% final state, 39/50 retention), and the remaining gap to end-to-end success is not the
writer.** §6 settles what it actually is: the sequential gate loop can only ever ask the
writer for a batch of one, so the joint solve that earns the 78.8% is unavailable inside CALM.
Quoting 78.8% as "the project's model-level improvement" without the gate caveat would repeat
exactly the inflation that the W5 audit was written to correct.

---

## 6. Inside CALM: end-to-end run and per-channel attribution

The direct-mode numbers above never pass through the gate. §5.4 declared that gap; it is now
closed by running the same writer through `CALMEditor` (`--mode calm`, layers 4-8, dual-write
on, side memory live):

| quantity | value |
|---|---|
| edits | 200 |
| gate decisions | WRITE 173 · ROUTE 27 · REJECT 0 |
| parameter channel (dual-write mirror) | 173 edits → **1 hit (0.6%)** |
| side-memory channel | 27 edits → 0 hits (see caveat 2) |
| aggregate | 1/200 = 0.5%, degenerate **12.0%** |
| immediate hit rate (before re-prompt) | 16/173 = 9.3% |
| memory entries stored | 200, 64 shards, avg 3.1/shard |
| wall time | 419 s |

**Why 78.8% becomes 0.6%.** Inside the gate loop every WRITE must be solved and applied
*before* the next decision, because the gate reads the model's post-write answer. The writer
therefore only ever sees a batch of one — the exact regime that measured 100% immediate /
0% final in §4. The joint solve is not a better hyperparameter, it is a *stream-level
optimisation*, and the sequential architecture structurally forbids it. Writer replacement
alone cannot fix CALM; it would need batched/deferred writes (queue the WRITE decisions,
solve them as one joint problem, then re-check the gate), which changes the algorithm rather
than its parameters.

**Degenerate rate moves the other way.** 12.0% per-request versus 0.4–4.6% joint: a delta
solved against six contexts alone overshoots the target token far more often. This strengthens
the §5.3 conclusion — the degenerate rate is a per-request phenomenon and any control has to
be per-request.

**Even the immediate hit rate decays.** Direct mode at batch=1 scores 100% immediate (§4);
inside the gate the same per-request solve scores **9.3%** immediate. The difference is state:
in the gate loop the 173rd write is solved against a model already carrying 172 deltas, whereas
the direct batch=1 sweep always wrote onto a clean model. Both "the joint solve is what earns
the accuracy" and "the deltas do not compose" fall out of the same observation.

**Caveats, both of which forbid quoting the memory column.**

1. The side-memory channel is a *store*, not a *reader*, in this harness: `calm_editor` calls
   `memory.store(...)` three times and never consults memory on the answer path, so a ROUTEd
   fact is written and then ignored. 0/27 is a harness property, not a memory property.
2. Even with a reader the number would be meaningless: `side_memory.py` keys entries on
   `sha256(subject|relation|target_new)` and looks them up with the same key, so a hit implies
   the caller already knew the answer. Memory hits are tautologically correct. See
   `w5_results_and_figures/w6_unified_write_quality_table.md` §4.1.

Net: the parameter channel is what this run measures, and it says the closed-form writer does
not rescue end-to-end CALM rewriting. That is a real, load-bearing negative result, not a
missing experiment.

---

## 7. Negative results (do not re-run these)

1. **`delta_rank=16` truncation.** batch=64, n=128 → final accuracy collapses to **0.094**
   (3/50 retention). A rank-16 Δ cannot satisfy 64×10 = 640 constraints; the rank must grow
   with the constraint count. Low-rank truncation is only usable for small batches or as a
   fluency diagnostic.
2. **`replay_cache` (history replay in chunked solves).** n=128, batch=32: retention stays
   7/50 and ΔNLL worsens from +1.141 to **+1.958** (generality 0.10 → 0.03). Historical
   residuals are computed under the *old* weights, so as soon as the weights move the cached
   constraint points drift. **Chunking plus replay is not a substitute for one joint solve.**
3. **Tuning the old writer to fix long-horizon conflict.** The capacity probe (effective
   rank 75/3072, same-relation direction overlap 0.63) rules this out; the v2 sweep over
   λ / scale / norm-cap / null-space projection had already shown the same thing.

---

## 7. Open items

- **Per-request delta guard.** The degenerate rate is a per-request phenomenon (12.0% inside
  CALM, 4.6% at layers 4-8 joint, 0.4% at 6-8). The global `delta_cap` is falsified (§5.3), so
  the only live option is: after each write, probe the immediate target; if it repeats, shrink
  *that request's* delta (0.5×, then 0.25×). This is compatible with the sequential loop
  (where per-request deltas already exist) but not with the whole-stream joint solve.
- **Multiple refinement sweeps** (re-solve with failed constraints up-weighted, jointly with
  the full constraint set) is the principled next step for final accuracy — not more
  chunking.
- **Batched/deferred writes** are the only way to bring the joint solve inside CALM (§6):
  queue the WRITE decisions, solve them as one problem, then re-run the gate on the resulting
  state. This is an algorithm change, and it is the item with the largest expected payoff.
- **Side-memory reader + re-keying** (§6 caveat 2, `side_memory.py`): entries are keyed by a
  hash that contains `target_new`, so a lookup can only succeed when the caller already knows
  the answer. Until it is keyed on `(subject, relation)` and read on the answer path, no
  memory-channel accuracy is measurable and "dual channel" cannot be claimed. Owner: side
  memory (Hoi Lam), blocking for the W6 per-channel table.

---

## 8. Reproduction

```powershell
cd EasyEdit
# recommended configuration (500 edits, ~14 min on RTX 4070 Laptop)
..\.venv\Scripts\python.exe -u examples\w4_closedform_smoke.py --mode direct --edits 500 `
  --layers 4,5,6,7,8 --l2 1 --z-clamp 0.6 --batch-solve 500 `
  --locality-n 10 --recheck-n 50 --probe-every 250 --out-dir results_b500_n500_L45678

# batch-size sweep (n=128): --batch-solve 1 / 32 / 64 / 128, layers 6,7,8
# delta-cap control:          add --delta-cap 0.02
# low-rank negative result:   add --delta-rank 16 --batch-solve 64
# replay negative result:     add --replay-cache --batch-solve 32

# end-to-end through the CALM gate, dual channel on (§6, 200 edits, ~7 min)
..\.venv\Scripts\python.exe -u examples\w4_closedform_smoke.py --mode calm --edits 200 `
  --layers 4,5,6,7,8 --l2 1 --z-clamp 0.6 --locality-n 10 --recheck-n 50 `
  --side-memory-dir "..\FONG HIO LAM\side_memory_router" --dual-write `
  --out-dir results_calm_n200_dual
```

Artifact directories (all under `EasyEdit/`, one `summary.json` + `per_edit.jsonl` each):
`results_base_b1_n128_L678`, `results_b32_r0_n128_L678`, `results_b64_r0_n128_L678`,
`results_b128_n128_L678`, `results_b32_replay_n128_L678`, `results_r16_b64_n128_L678`,
`results_b500_n500_L678`, `results_b500_n500_L45678`, `results_b500_n500_L45678_cap002`,
`results_calm_n200_dual`.
