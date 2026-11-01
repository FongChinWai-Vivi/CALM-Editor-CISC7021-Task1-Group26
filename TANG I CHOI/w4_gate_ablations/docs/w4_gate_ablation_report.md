# W4 -- Gate ablations on the CALM edit stream (Tang I Choi, MC631517)

Owner: Tang I Choi. Scripts: `run_w4_gate_ablation.py` (arms + stream),
`analyze_w4_statistics.py` (tests and tables), arms in `gate_arms/`.
Artefacts: `results/w4_ablation_<arm>.json`, `results/w4_statistics.json`,
`results/w4_ablation_table.md`. No teammate file is modified: the arm is
injected through `hparams.conflict_gate_path` and the two CALMEditor routing
guards added earlier in this project are switched off on the editor instance,
so each arm measures the gate alone.

## 1. Question

W3 showed that a supervised conflict gate can reach 0.88 accuracy on its own
data sheet. This work package asks the question that actually matters for the
system: **does gating change what the editor remembers at the end of a stream,
and does the learned gate beat always-write / always-route / rule-based gating
by a margin worth the complexity?**

## 2. Design

| element | setting |
|---|---|
| model | `gpt2-xl` (hparams `EasyEdit/hparams/CALM/gpt2-xl.yaml`) |
| stream | `gate_test.jsonl`, first 40 subjects, up to 6 edits each -> 191 edits |
| locality probes | 60 held-out facts on subjects that never appear in the stream |
| arms | `always_write`, `always_route`, `rule_gate` (W1 manual), `learned_gate` (W3 classifier) |
| metrics | decision mix, immediate rewrite, final model retention, final E2E retention, locality, side-memory entries |
| tests | exact McNemar (two-sided), paired bootstrap (5000 draws, seed 20261101), Cohen's kappa, Wilson 95% CI |

Metric definitions:

- **immediate rewrite** -- the new target appears in the model's answer right
  after the edit request is processed (parametric channel success);
- **final model retention** -- share of *all* stream edits the parameters still
  answer correctly at the end of the stream (what parameter pollution
  destroys);
- **final E2E retention** -- the same, but counting a side-memory hit as a
  success (what the user sees);
- **locality** -- share of held-out facts whose answer is unchanged by the
  whole stream.

## 3. Results

| arm | WRITE | ROUTE | REJECT | gate acc. vs gold | immediate rewrite | model-only retention | E2E retention | locality | memory entries |
|---|---|---|---|---|---|---|---|---|---|
| `always_write` | 191 | 0 | 0 | 0.398 | 0.110 | 0.110 | 0.110 | 1.000 | 0 |
| `always_route` | 0 | 191 | 0 | 0.419 | 0.110 | 0.110 | **1.000** | 1.000 | 126 |
| `rule_gate` | 36 | 151 | 4 | **0.613** | 0.110 | 0.110 | **1.000** | 1.000 | 120 |
| `learned_gate` | 1 | 190 | 0 | 0.424 | 0.110 | 0.110 | **1.000** | 1.000 | 126 |

Throughput: 0.53-0.54 s/edit for every arm (gate cost is negligible next to
the LM forward pass).

### 3.1 Paired tests (end-to-end retention)

| A | B | b | c | McNemar p | delta | delta 95% CI |
|---|---|---|---|---|---|---|
| `always_write` | `always_route` | 0 | 170 | <0.0001 | -0.890 | [-0.932, -0.843] |
| `always_write` | `rule_gate` | 0 | 170 | <0.0001 | -0.890 | [-0.932, -0.843] |
| `always_write` | `learned_gate` | 0 | 170 | <0.0001 | -0.890 | [-0.932, -0.843] |
| `always_route` | `rule_gate` | 0 | 0 | 1.0 | 0.000 | - |
| `always_route` | `learned_gate` | 0 | 0 | 1.0 | 0.000 | - |
| `rule_gate` | `learned_gate` | 0 | 0 | 1.0 | 0.000 | - |

### 3.2 Agreement with the gold action

| comparison | kappa | agreement |
|---|---|---|
| learned gate vs gold action | 0.009 | 0.424 |
| rule gate vs gold action | 0.343 | 0.613 |
| learned gate vs rule gate | - | 0.424 (learned-only correct 0, rule-only correct 36, p<0.0001) |

### 3.3 Breakdown by conflict type (E2E retention / decision correct)

| label | n | always_write | always_route | rule_gate | learned_gate |
|---|---|---|---|---|---|
| no-conflict | 85 | 0.247 / 0.894 | 1.000 / 0.000 | 1.000 / 0.435 | 1.000 / 0.012 |
| pre-training-conflict | 26 | 0.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 |
| prior-edit-conflict | 40 | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| context-conflict | 40 | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

Reading the table: `no-conflict` edits are the only group where writing
parametrically beats routing (0.247 vs 1.000 for retention, but 0.894 vs 0.000
for decision correctness), and storing them costs nothing else. The two
conflict groups are pure ROUTE wins, and `pre-training-conflict` is the group
where **no** arm can write the new fact into the weights.

The 1.000 decision-correct figure that both gates reach on context-conflict is
an artefact of always routing: the gold action there is `ROUTE`, and a gate
that routes everything is trivially "right". The learned gate never actually
sees the context features (see below), so this must not be read as context
awareness.

## 4. What the numbers mean

1. **The gate is not where the retention comes from in this setup.** All four
   arms achieve the same 0.110 immediate rewrite and model-only retention. The
   untrained/lightly-trained hypernetwork writes very little into the
   parameters (matches the team 5K run: model-only rewrite 0.08%), so the
   jump from 0.110 to 1.000 E2E retention is produced entirely by the side
   memory, not by smarter routing.
2. **Gating still matters for cost and for correctness of the record,** not for
   this retention metric: `always_write` shows what a gate-less system looks
   like (E2E 0.110, delta -0.890, p<0.0001).
3. **The learned gate collapses to "always route" in the stream** (1 WRITE,
   190 ROUTE, 0 REJECT), which is exactly why it is indistinguishable from
   `always_route` (p=1.0 on all three metrics) and why its kappa is 0.009 while
   the rules reach 0.343.
4. **Cause (measured, not guessed).** `CALMEditor.edit_requests` calls
   `gate.decide(..., retrieval_context=None)`, so the four context features are
   constant zero, and the prior-edit features only become informative after the
   first few edits. Re-scoring the W3 classifier under stream conditions drops
   it from 0.877 (offline) to 0.340 (cold start), while the rules drop from
   0.840 to 0.408 -- see `w3_conflict_gate_training_and_classifier/results/gate_online_gap.json`.
5. **Locality is saturated (1.000) for every arm**, including `always_write`.
   With a write rate this low the stream simply does not perturb the held-out
   subjects, so locality cannot discriminate between arms here. It should be
   re-measured once the parametric channel actually writes (guards on, larger
   hypernet scale).

## 5. Limitations

- One backbone (`gpt2-xl`), one stream size (191 edits, 40 subjects). The
  paired CIs are correspondingly wide for anything but the 0.89 gap.
- `hypernet_scale=1.0` and the W4 guards disabled: this isolates the gate, but
  it deliberately keeps the parametric channel weak.
- The gold action for CounterFact situations is derived, not human-labelled;
  only the pilot-100 split is human. The 0.65 human-set accuracy in W3 is the
  realistic ceiling to keep in mind when reading gate accuracy here.
- `always_route` keeps an unbounded side memory; at 5K edits the team run shows
  4870 entries at 97.4% route share, so a cost-aware arm (budgeted ROUTE) is
  the natural next step, not another classifier.

## 6. Reproduce

```bash
cd "TANG I CHOI/w4_gate_ablations"
python run_w4_gate_ablation.py --all-arms --subjects 40   # ~7 min on M3 Pro/Max
python analyze_w4_statistics.py                           # tables + tests
```

Full driver (W3 -> W5): `bash "TANG I CHOI/w5_report_and_reproducibility/reproduce_all.sh"`.
