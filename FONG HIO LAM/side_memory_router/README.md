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
├── side_memory.py      # Sharded non-parametric memory module
├── conflict_gate.py    # Write/Route/Reject decision engine
└── README.md           # This file
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
5.0 min on an RTX 5060 Laptop GPU). Every input signal to the gate comes from the
model itself, never from the gold label:

| Signal | Source |
|---|---|
| `current_confidence` | mean softmax probability of the tokens the model actually generated |
| `current_answer` | free greedy generation for the edit prompt |
| `pre_edit_probe_answer` | a second greedy generation on the first paraphrase prompt (short budget, 8 tokens) |
| `retrieval_context` | character 3-gram similarity between subjects already in the side memory |

### Decision distribution

| Decision | Requests | Share | Downstream path exercised |
|---|---|---|---|
| WRITE | 98 | 98% | parameter-write path owned by the W2 editors |
| ROUTE | 1 | 1% | side-memory store + lookup verified |
| REJECT | 1 | 1% | edit skipped, justification logged |

CounterFact always asks for a *new* fact, so the unedited model can never already
answer `target_new`; a REJECT is therefore structurally impossible on this split
except when the probe happens to reproduce the target string. The single REJECT
that did occur carries `pre_edit_probe_match = true`, i.e. the gate never rejects
without evidence.

### REJECT branch is live, not dead code

Because the main split cannot exercise REJECT, a canary group was added: it asks
the gate to write the **true** fact (`target_true`), a redundant edit the model
already knows. A correct gate must reject it through the pre-edit probe.

| Canary (20 requests) | Count | Rate |
|---|---|---|
| REJECT | 9 | 0.45 |
| WRITE or ROUTE (missed) | 11 | 0.55 |

Every canary REJECT was backed by a probe hit, and every probe hit produced a
REJECT (precision 1.00). The 0.55 miss rate is a generation-capability limit of
gpt2-xl, not a logic error: the 8-token probe often drifts into an unrelated
clause, so the target string never appears even though the model knows the fact.

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
   - On `ROUTE`: call `side_memory.store(subject, relation, target_new, answer)` instead.
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
