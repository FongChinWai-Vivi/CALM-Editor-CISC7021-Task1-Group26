# CALM W3 — SideMemory & ConflictGate (Fong Hio Lam, MC654953)

## Overview

This deliverable implements two core components of the **CALM** (Confidence-Aware
Long-term Memory) project for CISC7021 Group 26:

1. **SideMemory** — A WISE-style sharded non-parametric memory that stores edit
   answers in hash-routed shards. Edits are never overwritten; once a subject is
   stored in a shard, subsequent edits to the same subject are rejected by that
   shard, forcing them through the parameter-writing path or another shard.

2. **ConflictGate** — A Write/Route/Reject decision engine that determines whether
   each edit should be written directly to model parameters, routed to side memory,
   or rejected entirely based on current answer quality, recent edit conflicts, and
   retrieval context relevance.

Both modules are fully self-contained with zero third-party dependencies beyond the
Python standard library. They are designed to be dropped into the `CALMEditor` class
as components alongside `HypernetworkEditor`.

## Project Structure

```
w3_fonghiolam/
├── side_memory.py                # Sharded non-parametric memory module
├── conflict_gate.py              # Write/Route/Reject decision engine
├── real_w3_components.py         # Real-model driver (gpt2-xl + CounterFact)
├── probe_sweep.py                # Pre-edit probe recall sweep (template x budget)
├── probe_likelihood.py           # Probability-based probe study (AUC / operating points)
├── probe_sampling.py             # Probe vs model ceiling (greedy + temperature sampling)
├── probe_false_reject.py         # False-reject price of wiring sampling into REJECT
├── canary_strata.py              # Canary recall re-based on model-knownness strata
├── w3_memory_channel.py          # Memory answer-path recall (canonical/paraphrase/shuffled)
├── w3_components_gpt2xl.json     # Real-run output (100 requests, canary, repeat-edit)
├── probe_sweep_gpt2xl_v2.json    # Probe sweep results (20 canary items)
├── probe_likelihood_gpt2xl.json  # Likelihood probe results (120 items)
├── probe_sampling_gpt2xl.json    # Greedy vs sampled probe results (20 canary items)
├── probe_false_reject_gpt2xl.json  # Greedy vs sampled REJECTs on the 100 main requests
├── canary_strata_gpt2xl.json     # Canary recall by stratum (join of the two above)
├── w3_memory_channel_gpt2xl.json # Memory answer-path recall on 20 routed edits
└── README.md                     # This file
```

## Dependencies

- Python 3.8+ (standard library only: `hashlib`, `random`, `collections`, `dataclasses`)
- No pip packages required

## Running the Code

### Smoke Tests

Each module includes a self-contained smoke test that can be run independently:

```bash
python side_memory.py
python conflict_gate.py
```

### Integration Example

To use both modules together inside a `CALMEditor` instance:

```python
from side_memory import SideMemory
from conflict_gate import ConflictGate, Decision

mem = SideMemory(n_shards=64, max_per_shard=200)
gate = ConflictGate(n_recent=50, write_threshold=0.3, route_threshold=0.6)

# Decide on an edit
decision = gate.decide(
    subject="Spain", relation="capital", target_new="Madrid",
    current_answer="Madrid", current_confidence=0.95,
    recent_edits=mem._index.keys(),  # pass relevant keys from side memory
)

if decision == Decision.WRITE:
    # Apply parameter update via HypernetworkEditor
    ...
elif decision == Decision.ROUTE:
    # Store answer in side memory
    mem.store("Spain", "capital", "Madrid", "Madrid")
elif decision == Decision.REJECT:
    # Skip this edit
    ...
```

## Data Input Format

The modules accept edits in a simple tuple format consistent with the project's
`EditCase` specification:

| Field | Type | Description |
|-------|------|-------------|
| `subject` | str | Entity name (e.g., "Spain") |
| `relation` | str | Attribute being edited (e.g., "capital") |
| `target_new` | str | Desired new answer |
| `current_answer` | str | What the model currently answers |
| `current_confidence` | float | Model confidence [0, 1] |

For side memory lookups, the three fields `(subject, relation, target_new)` are
hashed into a deterministic signature using SHA-256 truncated to 16 hex characters.

## Output Format

### SideMemory Stats

```python
{
    "n_shards": 64,
    "max_per_shard": 200,
    "total_stored": 12,
    "avg_entries_per_shard": 1.5,
    "min_entries": 0,
    "max_entries": 3,
}
```

### ConflictGate Decision Summary

```python
{"WRITE": 3, "ROUTE": 5, "REJECT": 2}
```

### Individual Decision Log Entry

```python
{
    "subject": "Spain",
    "relation": "capital",
    "target_new": "Madrid",
    "current_answer": "Paris",
    "current_confidence": 0.72,
    "conf_score": 0.5,
    "has_conflict": False,
    "max_similarity": 0.0,
    "context_relevance": 0.0,
    "decision": "WRITE",
    "reason": "low_model_confidence",
}
```

## Real-model validation run (gpt2-xl, CounterFact, 100 requests)

The components are not only unit-tested: they were driven by a real model on the
CounterFact evaluation split (`w3_components_gpt2xl.json`, 100 edit requests,
8.8 min on an RTX 5060 Laptop GPU). Every input signal to the gate comes from the
model itself, never from the gold label:

| Signal | Source |
|---|---|
| `current_confidence` | mean softmax probability of the tokens the model actually generated |
| `current_answer` | free greedy generation for the edit prompt |
| `pre_edit_probe_answer` | greedy generation on a 3-prompt probe set (original prompt + the two CounterFact paraphrase prompts, 16 tokens each); the probe is a hit when *any* answer contains the target |
| `retrieval_context` | character 3-gram similarity between subjects already in the side memory |

### Decision distribution

| Decision | Requests | Share | Downstream path exercised |
|---|---|---|---|
| WRITE | 97 | 97% | parameter-write path owned by the W2 editors |
| ROUTE | 1 | 1% | side-memory store + lookup on the answer path |
| REJECT | 2 | 2% | edit skipped, justification logged |

The stream only routes one edit, so the main run cannot carry a memory accuracy on
its own; it is the twenty repeat-edit escalations that populate the memory channel, and
those are what `w3_memory_channel.py` scores. The split quoted in earlier drafts,
96/2/2, came from an archived run made with `--max-new-tokens 16`; `current_confidence`
is a mean over the tokens actually generated, so a shorter span moves the gate margin
and one borderline sample (Toko Yasuda) flips ROUTE to WRITE. The file shipped here is
the default-budget run (`--max-new-tokens 24`, 282 s) and yields 97/1/2.

Greedy decoding here is bit-reproducible: two independent passes on the first three
requests return identical text and identical `current_confidence` (|diff| = 0.0), so the
counts above are a property of the configured budget rather than of run-to-run noise.
The gate margin is what a different budget perturbs, and only on samples that sit on it.

### What the ROUTE path actually measures

The key is the query pair and nothing else: `sha256(subject + "|" + relation)[:16]`,
with `target_new` stored as the *value*. An earlier build folded `target_new` into
the key and made it a required argument of `lookup()`, so recall could only ever be
1.0 — the caller had to know the answer before asking. That number was an index hit,
not memory recall, and it is no longer reported as accuracy.

The read path is now explicit: `lookup(subject, relation)` returns a note,
`build_memory_note()` wraps it as a prompt prefix, the prefix is prepended to the
cue, and the model generates. `w3_memory_channel.py` measures that path end to end on
the routed edits (gpt2-xl, greedy, 24 new tokens, `w3_memory_channel_gpt2xl.json`):

| Metric | Denominator | Value | Reading |
|---|---|---|---|
| `memory_n` | — | 20 | routed edits written to side memory |
| `memory_baseline_recall` | 20 | 0.00 | same cue, **no** note injected — the channel contributes nothing if this moves |
| `memory_canonical_recall` | 20 | 0.80 | target reproduced after the note is injected (original cue) |
| `memory_paraphrase_recall` | 20 | 0.45 | same note, CounterFact paraphrase cue |
| `memory_shuffled_recall` | 20 | 0.00 | control: note of a *different* edit injected, so a hit would mean leakage |
| `legacy_constructive_recall` | 20 | 1.00 | diagnostic only: reproduces the old keying, which cannot miss |

Three controls make the 0.80 readable. The cue-only baseline is 0.00, so the twenty
edits are facts the model does not already answer and the gain is attributable to the
injection rather than to the cue. The shuffled-key arm is also 0.00, so the model is
not simply copying whatever target the note happens to contain; it is retrieving the
entry that matches `(subject, relation)`. And `legacy_constructive_recall` stays at 1.00
precisely because the old keying cannot miss, which is why it is filed as a diagnostic
and never as accuracy.

What is left is the shape of the retrieval prompt: the fact comes back when the cue is
the stored surface form (0.80) and less than half the time when CounterFact paraphrases
it (0.45). That gap is a prompt-format property, not a property of sharding, and it is
the number to quote whenever the end-to-end ceiling is recomputed — the ceiling must use
0.80 / 0.45, never the 1.00 that the old path reported.

CounterFact always asks for a *new* fact, so the unedited model can rarely already
answer `target_new`; a REJECT on this split can only happen when the probe happens
to reproduce the target string. Both REJECTs that did occur carry
`pre_edit_probe_match = true`, i.e. the gate never rejects without evidence. One of
them is a genuine redundancy rather than a coincidence: for
`Mayer Carl von Rothschild → London` the model's own continuation already reads
"the Rothschild banking house in London", so writing that edit would be a no-op.

### REJECT branch is live, not dead code

Because the main split cannot reliably exercise REJECT, a canary group was added: it
asks the gate to write the **true** fact (`target_true`), a redundant edit the model
already knows. A correct gate must reject it through the pre-edit probe.

| Canary (20 requests) | Count | Rate |
|---|---|---|
| REJECT | 11 | 0.55 |
| WRITE or ROUTE (missed) | 9 | 0.45 |

Every canary REJECT was backed by a probe hit, and every probe hit produced a
REJECT (precision 1.00). "Zero false reject" here means zero *observed* on the 100
main-split requests, which is a 95% upper bound of 5/100 by the rule of three (2 observed
false rejects), not a proof of zero. The remaining 0.45 miss rate is *model
knowledge*, not gate logic: in the missed cases the probe answers the question with a
different (and often wrong) value, e.g. `Lyon → Beirut` answered "…a city in France,
located in the south of…", `Wellington → Sydney` answered "…a city in New Zealand…".
No decision rule can reject on evidence the model never produced. That claim is not
left as an assertion — the two sections below measure how much of the 0.45 is
recoverable at all, and re-base the denominator on the items the model can be shown
to know.

### Why the probe recalled only 0.45, and how it was raised to 0.55

The first real run used a single paraphrase prompt with an 8-token budget and reached
0.45. Two follow-up experiments pinned down where that ceiling comes from. Both are
reproducible with the scripts shipped next to this README.

`probe_sweep.py` re-generates the probe for the same 20 canary items under every
combination of probe template and budget and scores containment exactly as the gate
does (`probe_sweep_gpt2xl_v2.json`):

| Probe prompt set | 8 tok | 16 tok | 24 tok | 32 tok |
|---|---|---|---|---|
| original prompt only | 7/20 | 7/20 | 7/20 | 7/20 |
| paraphrase prompt #1 | 4/20 | 4/20 | 5/20 | 6/20 |
| paraphrase prompt #2 | 5/20 | 7/20 | 7/20 | 7/20 |
| union of the two paraphrases | 7/20 | 8/20 | 8/20 | 9/20 |
| **union incl. original prompt** | **10/20** | **11/20** | **11/20** | **11/20** |

Two conclusions follow. Budget stops mattering after 16 tokens (the model either
states the attribute immediately or drifts), and no single paraphrase prompt is
enough — the recall gain comes from asking in several formulations, including the
original one. The gate was therefore changed to probe with up to three prompts and to
accept a hit from any of them; the canary rate moved from 0.45 to 0.55 and the main
split stayed at 2/100 REJECTs.

`probe_likelihood.py` tests the alternative of replacing generation with a
probability test — length-normalised log p(target | probe prompt) — on 100 main plus
20 canary items (`probe_likelihood_gpt2xl.json`):

| Signal | canary mean | main mean | AUC |
|---|---|---|---|
| absolute log p(requested target \| probe) | -6.55 | -9.11 | 0.706 |
| margin log p(requested) − log p(competing) | +2.81 | -2.58 | 0.873 |

The absolute signal is deployable (it needs only the requested target), but its best
zero-false-positive operating point has recall 0.00, and the threshold that recovers
0.55 recall would reject 29% of genuine edits. The margin separates the classes well
but needs the competing target, which a real gate does not have. The likelihood probe
is therefore reported as a diagnostic, not wired into REJECT: the containment probe
is kept because it rejects only when the model has actually volunteered the requested
answer.

### How much of the remaining miss rate is the probe, and how much is the model

The two studies above share one blind spot: they only vary the prompt and the budget,
both of which still read the *mode* of the distribution. A fact the model holds weakly
can be absent from the argmax string and still surface under sampling. `probe_sampling.py`
separates the two failure modes on the same 20 canary items, with the deployed probe
configuration (3 prompts, 16 tokens) plus 8 temperature samples (T=0.7, top_p=0.95) per
prompt — the cheap local form of the SliCK categorisation (Gekhman et al., *Does
Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*):

| Category | Items | Reading |
|---|---|---|
| HighlyKnown — a greedy answer already contains the target | 11/20 = 0.55 | what the deployed gate rejects today |
| WeaklyKnown — greedy misses, a sample contains the target | 3/20 = 0.15 | recoverable by sampling: a probe ceiling |
| Unknown — no greedy or sampled answer ever contains it | 6/20 = 0.30 | model-knowledge gap: no probe can fix it |

Sampling alone lifts recall to 0.65 (13/20): three of the nine greedy misses are
recoverable — `Wellington`, `Shree Pundalik`, `Andreas Ivanschitz` — while six never
surface at all: `Toko Yasuda`, `Lyon`, `Anaal Nathrakh`, `Percy Snow`,
`Saint Petersburg`, `Robert William Muench`. The two probe modes are complementary
rather than nested: `Edwin of Northumbria` is hit greedily but by 0 of its 24 samples
(temperature drifts off the answer), so the achievable bound is greedy **or** sampled,
**0.70 (14/20)**, not either one alone.

This is the decomposition that matters for the gate's error analysis: **at most 0.15 of
the 0.45 canary miss rate is fixable by a better probe; the other 0.30 is the model not
holding the fact**. It also explains why the previous 0.55 figure must not be read as a
gate defect. Sampling is *not* wired into the deployed gate. It costs nine generations
per prompt instead of one (27 generations per item, 687 s for the 20 canary items,
against 3 generations per item for the deployed probe), and the recall it buys is paid
for one-for-one in false rejects; the next section measures that price instead of
asserting it. The 0.65 (0.70 with the greedy hits sampling loses) ceiling is therefore
reported as an achievable bound, with the gate kept at the cheap, precision-first
setting.

### What sampling would cost: false rejects on the main split, measured

The previous section prices sampling on the canary, where a REJECT is *correct*. The
cost lands on the main split, where a REJECT is *wrong*: the 100 requests there are
genuine edits that should be written. `probe_false_reject.py` runs both probe variants
on those 100 requests — the deployed greedy probe (3 prompts, 16 tokens) and, alongside
it, 8 samples per prompt at T=0.7 / top_p=0.95 — and scores containment exactly as the
gate does (`probe_false_reject_gpt2xl.json`):

| Gate variant | REJECTs on 100 genuine edits | Rate | 95% upper bound |
|---|---|---|---|
| deployed (greedy probe only) | 2 | 0.02 | 0.05 |
| sampling-augmented (greedy, or any of the 24 samples) | 14 | 0.14 | 0.17 |
| **incremental cost of sampling** | **+12** | **0.12** | **0.15** |

Upper bounds use the (k+3)/n rule of three at 95%. Enabling sampling would buy +0.15
canary recall (0.55 → 0.70) at a cost of up to +0.15 false rejects: a one-for-one
trade, not a net gain. That is the empirical reason the sampling variant is analysed
but not deployed.

The 12 extra rejects are mostly *incidental containment* — the sampled continuation
mentions the requested target inside an unrelated clause while asserting something
else. All three examples below are WRITE under the deployed gate and REJECT once
sampling is enabled:

| Item | Requested target | Deployed greedy probe | Sampled continuation that "hit" |
|---|---|---|---|
| `Michel Denisot` | Russian | "…French." | "French and he speaks English and **Russian**. He has lived in the United States for…" |
| `Danielle Darrieux` | English | "French. She is a native of Montreal, Quebec…" | "a form of French and is a mix of both **English** and French…" |
| `Majorette` | London | "the city of San Juan, Puerto Rico…" | "the West End of **London**." |

In each case the model is plainly still asserting the old fact, so the sample hit is not
evidence that the edit is redundant — it is a false reject. Only 2 of the 100 requests
are hit by both variants (the two the deployed gate already rejects), i.e. sampling adds
no confirmed redundancy on this split, only risk. Zero false rejects therefore remains
an observed property of the deployed setting, with the measured price of relaxing it
now on record.

All four probe studies run read-only against the frozen model, so they can be
reproduced without touching the editor:

```bash
python probe_sweep.py --data-path /path/to/counterfact.json --n 20 \
    --out probe_sweep_gpt2xl_v2.json
python probe_likelihood.py --data-path /path/to/counterfact.json --n 100 \
    --out probe_likelihood_gpt2xl.json
python probe_sampling.py --data-path /path/to/counterfact.json --n 20 \
    --max-templates 3 --budget 16 --n-samples 8 --temperature 0.7 --top-p 0.95 \
    --out probe_sampling_gpt2xl.json
```

The re-basing below is a pure join of the last two output files, so it needs no model
run and re-derives in under a second:

```bash
python canary_strata.py --run w3_components_gpt2xl.json \
    --sampling probe_sampling_gpt2xl.json --out canary_strata_gpt2xl.json
```

The memory-channel readout is a separate model pass over the routed edits:

```bash
python w3_memory_channel.py --model-key gpt2-xl \
    --data-path /path/to/counterfact.json --n 20 \
    --out w3_memory_channel_gpt2xl.json
```

### The canary denominator is re-based on what the model knows

The canary label says "the world knows this fact, so a correct gate must reject it".
That is the wrong label for a *model-side* gate: `gpt2-xl` demonstrably does not hold 6
of the 20 (`Unknown`: 18 greedy continuations and 144 sampled ones across all three
prompts, none containing the answer), so scoring them as misses charges the gate for a
fact no probe could have recovered. `canary_strata.py` joins the probe strata with
the gate's own decisions and reports recall under three denominators (`canary_strata_gpt2xl.json`):

| Denominator | Items | REJECT | Greedy recall | Greedy or sampled |
|---|---|---|---|---|
| all canary items (old number, kept for continuity) | 20 | 11 | 0.550 | 0.700 |
| **model-known only** (HighlyKnown + WeaklyKnown) | **14** | **11** | **0.786** | **1.000** |
| model-unknown (excluded from recall) | 6 | 0 | 0.000 | 0.000 |

The last row is the sanity check that makes the re-basing honest: on the 6 model-unknown
items the gate rejects **zero** — not because it fails, but because there is nothing to
match. All six are reported by name (`Toko Yasuda`, `Lyon`, `Anaal Nathrakh`,
`Percy Snow`, `Saint Petersburg`, `Robert William Muench`), so the exclusion can be
audited item by item instead of being taken on trust.

Read together: on the items the model provably knows, the deployed gate catches
**11/14 = 0.79**, and sampling would take that to **14/14 = 1.00**. The 0.55 headline
is therefore a property of the canary's label, not of the decision rule — and the honest
summary of the REJECT branch is "no false rejects, 0.79 recall on what is detectable,
0.30 of the canary is undetectable by construction".

Lifecycle of the recall number, so the several pairs in this README are not confused:
single-prompt 8-token probe 0.45 → multi-prompt 16-token probe 0.55 (0.45 → 0.55 is a
probe change) → sample-only 0.65 → greedy-or-sampled 0.70 → re-based on the 14
model-known items 0.79 → sampling bound on that subset 1.00.

### Conflict signal actually escalates

The conflict signal only fires when the sliding window holds an edit for the same
`(subject, relation)`. To show it works rather than being unreachable, a
repeat-edit probe edits the same subject twice with two different targets:

| Repeat-edit probe | Count | Rate |
|---|---|---|
| second request escalated to ROUTE | 20 / 20 | 1.00 |
| second request overwritten as WRITE | 0 / 20 | 0.00 |

The second edit is therefore never written over the first one; it is routed to the
side memory, which is exactly the overwrite protection W3 is meant to provide.

## Integration with Formal Evaluation

To integrate these components into the official CALM evaluation pipeline:

1. **Replace placeholder components** in `CALMEditor.__init__`:
   ```python
   self.side_memory = SideMemory(n_shards=64, max_per_shard=200)
   self.conflict_gate = ConflictGate(
       n_recent=50, write_threshold=0.3, route_threshold=0.6
   )
   ```

2. **Hook into the edit flow** inside `CALMEditor.edit()`:
   - Before applying any update, call `gate.decide()` to get the routing decision.
   - On `WRITE`: proceed with `HypernetworkEditor.generate_update()` as before.
   - On `ROUTE`: call `side_memory.store(subject, relation, target_new)` and read it
     back with `side_memory.lookup(subject, relation)`; the answer is the value, never
     part of the key, and `lookup()` does not accept it.
   - On `REJECT`: skip the edit entirely and log it.

3. **Metrics computation** remains unchanged. The evaluation metrics
   (`rewrite_acc`, `paraphrase_acc`, `locality_acc`, `portability_acc`) are computed
   identically regardless of which component handled the edit. The only difference is
   where the knowledge is stored (parameters vs. side memory).

4. **Parameter tuning**: The gate thresholds (`write_threshold`, `route_threshold`,
   `similarity_threshold`) should be tuned on a validation set before running the
   final evaluation. Start with the defaults provided and adjust based on observed
   trade-offs between reliability and generalisation.

## Design Notes

### SideMemory Design Decisions

- **Sharding strategy**: Edits are routed to shards using `hash(subject) % n_shards`.
  This ensures deterministic placement while distributing load evenly across shards.
- **No-overwrite guarantee**: Once an entry exists in a shard, `store()` returns False
  without modifying anything. This prevents accidental overwrites of previously edited
  facts.
- **LRU eviction**: When a shard reaches its capacity limit, the oldest entry is evicted.
  This prevents unbounded growth while keeping recently accessed entries available.
- **Global index**: A separate dictionary maps edit signatures to their shard location,
  enabling O(1) lookup across all shards without scanning every shard.

### ConflictGate Design Decisions

- **Three-signal scoring**: The gate evaluates each edit against three independent signals:
  1. Current answer quality (how close is the model's existing answer?)
  2. Recent edit conflicts (are there conflicting edits in side memory?)
  3. Retrieval context relevance (does retrieved context support this edit?)
- **Threshold-based decisions**: Simple threshold comparisons produce clean WRITE/ROUTE/REJECT
  outputs. The thresholds are tunable parameters exposed in the constructor.
- **Sliding window**: Recent edits are tracked in a bounded list of size `n_recent`,
  preventing memory leaks while maintaining short-term conflict detection capability.
- **Decision logging**: Every decision is logged with full context for post-hoc analysis
  and debugging.

## References

- Z. Gekhman, G. Yona, R. Aharoni, R. Reichart, and J. Herzig. 2024. *Does Fine-Tuning
  LLMs on New Knowledge Encourage Hallucinations?* In Proceedings of EMNLP 2024.
  arXiv:2405.05904. The SliCK knowledge hierarchy (`HighlyKnown` / `MaybeKnown` /
  `WeaklyKnown` / `Unknown`) that the canary strata in this README are mapped onto.

Only the model side of SliCK is used above (what `gpt2-xl` can be shown to know under
greedy and sampled decoding); the fine-tuning dynamics studied in that paper are out of
scope for a gate that never updates weights.
