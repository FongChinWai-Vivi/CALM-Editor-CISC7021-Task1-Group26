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

> **The `immediate rewrite` and `model-only retention` columns are the base
> model's hit rate on this stream, not a measure of the writer** (identical
> 0.110 in all four arms, including `always_route`, which writes nothing).
> Baseline-corrected values are 0.000 everywhere; see §3.4 and
> `docs/w4_immediate_rewrite_diagnosis.md`. Only the decision columns, the
> E2E-retention column, locality and memory entries are arm-dependent.

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

### 3.3.1 `pre-training-conflict`: why `decision correct` is zero for every arm

`pre-training-conflict` (n = 26) is the only conflict type whose **gold action is
`REJECT`**: the edit contradicts knowledge the backbone already holds
(e.g. "the Eiffel Tower is in Berlin"), so the correct behaviour is to refuse
the write. All four arms therefore score exactly 0.000 on decision correctness
for this group: the constant is a property of the label, not a ranking of the
arms.

| arm | what it does on these 26 edits | gold = REJECT | decision correct |
|---|---|---|---|
| `always_write` | writes 26/26 | 0/26 | 0.000 (by construction: it never rejects) |
| `always_route` | routes 26/26 | 0/26 | 0.000 (routes instead of rejecting) |
| `rule_gate` | routes 26/26 | 0/26 | 0.000 (the W1 rules never reject pre-training conflicts) |
| `learned_gate` | routes 26/26 | 0/26 | 0.000 (the classifier predicts `ROUTE` here) |

Two consequences, both reported rather than hidden:

1. The all-zero row is a **label/action mismatch**: all four arms are scored
   against an action (`REJECT`) that none of them is able to emit on this
   group. It is kept in the table for completeness and must not be read as
   "all arms are equally bad at pre-training conflicts"; what the table does
   show is that *no* arm distinguishes this conflict type in stream.
2. The E2E-retention entry (1.000) is memory-backed for the three routing arms
   because the *original* answer still matches after a side-memory store; the
   retention metric does not reward rejecting the edit. A `REJECT`-aware
   retention metric (fraction of pre-training conflicts left unedited *and*
   still answered with the old fact) is the correct instrument here; the
   `delta` between `immediate rewrite` and `final answer` in
   `results/immediate_rewrite_diagnosis.json` is what a future run would need to
   log for it.

`pre-training-conflict` is therefore **covered by the breakdown, but not yet
discriminated by any arm** — an explicit limitation of the W4 ablation, carried
into `w5_report_and_reproducibility/report/results_section.md` §"What does not
work".

### 3.4 `immediate rewrite`: metric defect and corrected values

All four arms report `immediate_rewrite = 0.110` — including `always_route`,
which never writes. `CALMEditor._measure` sets `rewrite_ok` from a post-edit
answer / `target_new` comparison without looking at the pre-edit answer, so any
edit whose *base* answer already contained `target_new` counts as a success.
Re-running the four arms on a deduplicated stream (one edit per prompt, 40 rows)
makes the contamination explicit:

| arm | WRITE share | reported immediate rewrite | baseline hit | corrected (`post ∧ ¬pre`) |
|---|---|---|---|---|
| `always_write` | 1.000 | 0.325 | 0.325 | **0.000** |
| `rule_gate` | 0.900 | 0.325 | 0.325 | **0.000** |
| `learned_gate` | 0.025 | 0.325 | 0.325 | **0.000** |
| `always_route` | 0.000 | 0.325 | 0.325 | **0.000** |

The dedicated hook in `run_w4_gate_ablation.py` shows the writes are applied
(mean ‖Δ‖_F ≈ 0.004) but never change a greedy answer
(`answer_changed_by_write_rate = 0.000`), because `hparams/CALM/gpt2-xl.yaml`
defines no `hypernet_ckpt` and the generator stays randomly initialised. The
legacy 0.110 is fully explained: 13 of 40 subjects are base-model hits, 21
legacy rows report `rewrite_ok`, and 21/21 of those sit on baseline-hit subjects.
Details, reproducer and the recommended reporting rule:
`docs/w4_immediate_rewrite_diagnosis.md`. The rerun's own table and statistics
are `results/w4_ablation_table_deduped.md` and `results/w4_statistics_deduped.json`
(raw arms in `results/w4_ablation_*_deduped.json`, log in
`run_logs/w4_ablation_deduped_20261105.log`); `analyze_w4_statistics.py` keeps
the two streams apart with `--stream canonical|deduped`, so the 40-prompt rerun
can never overwrite the 191-edit table above.

The 1.000 decision-correct figure that both gates reach on context-conflict is
an artefact of always routing: the gold action there is `ROUTE`, and a gate
that routes everything is trivially "right". The learned gate never actually
sees the context features (see below), so this must not be read as context
awareness.

## 3.5 Conclusion: the learned gate degenerates in-stream, and why

This is the most important negative result of W4 and is stated here explicitly
rather than left implicit in the tables above.

**Conclusion.** The W3 learned conflict gate does not survive the transition from
offline evaluation to a live CALM stream. In the 191-edit ablation it emits
`WRITE / ROUTE / REJECT = 1 / 190 / 0` (0.5% / 99.5% / 0.0%), i.e. it collapses
into a constant `always_route` policy:

| evidence | learned gate | rule gate (W1 manual) | reading |
|---|---|---|---|
| decision mix | 1 / 190 / 0 | 36 / 151 / 4 | learned gate never writes, never rejects |
| agreement with gold action | 0.424 | 0.613 | below the majority-class baseline of the rules |
| Cohen's κ vs gold | 0.009 | 0.343 | essentially chance |
| learned-only / rule-only correct | 0 / 36 (McNemar p < 0.0001) | — | the classifier is never right where the rules are wrong |
| vs `always_route`, E2E retention | 1.000 vs 1.000, p = 1.0 | — | indistinguishable from the trivial policy |
| offline → in-stream accuracy | 0.877 → 0.424 | 0.840 → 0.613 | −0.453 drop, larger than the rules' −0.227 |
| cold-start accuracy (first edits) | 0.340 | 0.408 | worst stage for the learned gate |

Source of the accuracy rows:
`w3_conflict_gate_training_and_classifier/results/gate_online_gap.json`
(offline / in-stream / cold-start: learned 0.877 / 0.424 / 0.340, rule 0.840 /
0.613 / 0.408; McNemar p = 0.0008).

**Cause (measured, in order of impact).**

1. **The context features are structurally dead in the stream.**
   `EasyEdit/easyeditor/editors/calm_editor.py` calls
   `gate.decide(..., retrieval_context=None)` unconditionally. The four context
   features (`has_context`, `n_context`, `context_target_overlap`,
   `context_contradicts`) are therefore constant zero for every edit, and they
   carry the largest positive `ROUTE` weights in the trained classifier (W3
   feature-importance table). The gate is asked to decide with its strongest
   evidence permanently missing.
2. **Zero-vector inputs fall in a region the classifier never saw in training.**
   In the W3 training set (`build_gate_training_data.py`) the context block is
   populated from retrieved passages, so the all-zero context vector co-occurs
   almost exclusively with `ROUTE`/`REJECT` labels; at inference every row is
   that zero vector, and the model maps it to its majority answer for the
   region, `ROUTE`. This is covariate shift caused by our own training
   construction, not a fitting error: the offline macro-F1 stays 0.843 while the
   in-stream accuracy halves.
3. **The surviving features only become informative late.** `has_prior_edit`,
   `n_prior_edits`, `prior_same_subject` are all zero until the first edit has
   been stored, so even the reduced gate starts blind — hence the additional
   cold-start drop to 0.340. On a stream whose dominant action is `ROUTE`
   (memory entries 126/191), the early blindness locks the gate into the
   routing mode it never leaves.

**What this result does and does not license.**

* It is **not** evidence that learned gating is worse than rule gating in
  general — the classifier is being scored with one of its four feature blocks
  forcibly disabled. An honest comparison requires injecting a retrieval context
  into `CALMEditor.edit_requests` (a change to a teammate-owned file, therefore
  filed as a recommendation in `w6_cross_review/docs/w6_cross_review_findings.md`
  B1 and **not** applied here).
* It **is** evidence that the current CALM end-to-end path cannot exercise a
  context-aware gate, and that any W4/W5 number claiming "context awareness"
  from this gate is an artefact of the routing prior. The rule gate's higher
  κ (0.343) is the correct reference point until the context channel exists.
* Consequences for the report: the ablation's contribution is (a) a paired
  measurement of what routing is worth end-to-end (`always_write` loses 0.890,
  p < 0.0001), and (b) the identification of this defect chain, which is a
  reusable finding for anyone wiring a learned gate into EasyEdit.

## 4. What the numbers mean

1. **The gate is not where the retention comes from in this setup.** All four
   arms report the same immediate-rewrite value -- 0.110 on the legacy stream,
   0.325 on the deduped stream, both of which are the base-model hit rate rather
   than a writer measurement (§3.4). The gate therefore does not move the
   writer's behaviour at all. The untrained/lightly-trained hypernetwork writes
   very little into the parameters (matches the team 5K run: model-only rewrite
   0.08%), so the jump from 0.110 to 1.000 E2E retention is produced entirely by
   the side memory, not by smarter routing.
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
