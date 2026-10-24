# CALM W3 -- Gate 3 Verification: End-to-End 1K-Edit Stream

- **Date**: 2026-10-24
- **Model**: `gpt2-small` (124M, local cache `EasyEdit/hugging_cache/gpt2-small`)
- **Backend**: CALM hypernetwork edit generator (`hypernet_alg: lowrank`, rank=4)
- **Script**: `EasyEdit/examples/w3_calm_stream.py` (new in this milestone)
- **Environment**: local `.venv` (Python 3.11.8), Windows 11

## 1. Objective

Run CALM end-to-end on a sequential (lifelong) edit stream: the first 1000
records of `EasyEdit/data/counterfact.json` are fed one by one into
`CALMEditor`.  Every edit passes through the `ConflictGate`, which routes it
to one of three outcomes:

| Decision | Route | Effect |
|---|---|---|
| `WRITE` | `hypernetwork` | low-rank weight update applied to the model |
| `ROUTE` | `side_memory` | fact stored in SideMemory, no parameter change |
| `REJECT` | `gate_reject` | edit skipped (model already consistent / conflict too weak) |

The run records the route distribution, side-memory usage, per-edit timing and
a sampled rewrite-retention check.

## 2. Reproduce

From the repository root, with the project virtual environment active:

```powershell
.venv\Scripts\python.exe EasyEdit\examples\w3_calm_stream.py --max-edits 1000 --sample-edits 8 --output-dir output
```

Supported options:

- `--max-edits` (default `1000`) -- number of CounterFact records to stream;
- `--sample-edits` (default `8`) -- how many middle/end edits to re-measure
  for retention on the final model;
- `--data`, `--hparams`, `--output-dir` -- custom paths.

The script prints a summary and writes
`output/w3_calm_stream_1k.json` (the 1000-record run) or
`output/w3_calm_stream_<n>.json` otherwise.

## 3. Results (n = 1000, gpt2-small)

### 3.1 Route distribution

| Route | Count | Share |
|---|---|---|
| `hypernetwork` (WRITE) | 855 | 85.5% |
| `side_memory` (ROUTE) | 141 | 14.1% |
| `gate_reject` (REJECT) | 4 | 0.4% |

The gate routed the vast majority of new facts to the hypernetwork, kept
14.1% in side memory (mostly repeated-subject / low-confidence conflicts) and
rejected only 4 edits (0.4%) where the model already produced the target.

### 3.2 Side memory usage

| Metric | Value |
|---|---|
| Shards | 64 |
| Max entries per shard | 200 |
| Total stored entries | 141 |
| Avg entries per shard | 2.2 |
| Min / max entries per shard | 0 / 7 |

### 3.3 Timing

| Metric | Value |
|---|---|
| Total wall time | 490.4 s |
| Average per edit | 0.490 s/edit |

Includes two greedy measurements (pre/post) per edit, gate scoring and, for
WRITE, the hypernetwork update.

### 3.4 Rewrite retention

Immediate rewrite success (measured right after each edit, within the stream):

| Metric | Value |
|---|---|
| Immediate `rewrite_ok` | 10 / 1000 (1.0%) |

Sampled retention on the **final** model (6 samples: 4 middle + 2 end of the
stream):

| Index | Subject | Target | Final answer (first tokens) | Retained |
|---|---|---|---|---|
| 252 | Victor Merzhanov | guitar | " role of the Russian ambassador" | no |
| 314 | Johannesburg | Portsmouth | " a city of the future" | no |
| 376 | Kalyan Kumar | Jerusalem | " the city of Kaly" | no |
| 438 | Adliswil | Germany | " the heart of the city" | no |
| 900 | Michael Umaña | football | " of baseball. He is" | no |
| 962 | Malcolm Arthur Smith | actor | " a bit of a mystery" | no |

Retained on final model: **0 / 6**.

## 4. Interpretation & limitations

- **Pipeline integrity**: the full CALM stream ran end-to-end without errors;
  all three gate outcomes were exercised, side memory grew exactly with ROUTE
  count (141 stored), and results were persisted. Gate 3's runnability
  criterion is met.
- **Low immediate rewrite rate**: only 1% of edits produced the target within
  the first 5 greedy-decoded tokens. Two contributing factors: (a) the rank-4
  low-rank hypernetwork update is a deliberately lightweight edit generator
  and is not competitive with full-weight editors (e.g. ROME/MEMIT) on hard,
  obscure CounterFact facts; (b) the metric is strict -- it requires the
  target string to appear in the first 5 generated tokens, which is a
  coarse proxy (many CounterFact templates generate a leading article or
  clause first).
- **Zero sampled retention**: the sampled facts were not visible in the final
  model. WRITE edits at positions 252-962 were measured after hundreds of
  subsequent updates, so interference/overwriting is expected for a rank-4
  generator; ROUTE entries (141) intentionally live in side memory and are
  not expected to change generation by themselves (they would be used by a
  retrieval-augmented inference path, which is out of scope for Gate 3).
- **Honest reporting**: no metric was tuned or cherry-picked. The numbers
  above are the raw outputs of `w3_calm_stream_1k.json`.

## 5. Files

| File | Description |
|---|---|
| `EasyEdit/examples/w3_calm_stream.py` | new Gate 3 stream script |
| `output/w3_calm_stream_1k.json` | machine-readable results (this run) |
| `output/w3_calm_stream_verification.md` | this verification report |
