# W4 -- Conflict gate ablations and statistics (Tang I Choi, MC631517)

Four gate policies over one fixed edit stream, then paired statistics on the
result. The point of the experiment is to answer "does the conflict gate earn
its keep?" rather than "which gate wins", because retention turned out to be
insensitive to the decision.

## Contents

| path | what it is |
|---|---|
| `run_w4_gate_ablation.py` | runs one or all arms over the stream; `--all-arms` is what produced `results/` |
| `gate_arms/always_write_gate.py` | decides `WRITE` for every edit (the failure baseline) |
| `gate_arms/always_route_gate.py` | decides `ROUTE` for every edit (side memory only) |
| `gate_arms/rule_gate.py` | the W3 rule policy: thresholded confidence + similarity (as in the frozen W1 blueprint) |
| `gate_arms/learned_gate.py` | the W3 logistic-regression classifier, loaded from `gate_model/gate_classifier.joblib` |
| `gate_arms/_arm_common.py` | shared plumbing: `sys.path` for `EasyEdit/`, one-arm injection via `hparams.conflict_gate_path`, MPS cache release, JSON dump |
| `analyze_w4_statistics.py` | paired tests (exact McNemar, Cohen's kappa, Wilson CI, bootstrap CI), per-conflict-type breakdown |
| `results/w4_ablation_<arm>.json` | per-arm output (per-edit decisions + metrics) |
| `results/w4_statistics.json` | the statistics table the report and the figures read |
| `results/w4_ablation_table.md` | markdown render of the same table |
| `docs/w4_gate_ablation_report.md` | full write-up: design, results, tests, per-type breakdown, limitations |

## Design in one paragraph

The stream is the first 40 subjects of `gate_test.jsonl` in dataset order, up to
6 edits per subject (191 edits), and the locality probes are the first 60 rows
whose subject never appears in the stream -- identical for the four arms, which
is what makes the paired tests valid. The backbone is `gpt2-xl` (EasyEdit cache),
greedy decoding. Arms are injected through `hparams.conflict_gate_path`; the
CALMEditor's own guards are disabled on the instance for the ablation, because
they were added later and would otherwise mask the arm's decision.

## Headline result

| arm | E2E retention | model-only | gate acc vs gold | WRITE/ROUTE/REJECT |
|---|---|---|---|---|
| always_write | 0.110 | 0.110 | 0.450 | 191/0/0 |
| always_route | 1.000 | 0.000 | 0.503 | 0/191/0 |
| rule_gate | 1.000 | 0.000 | 0.613 | 36/151/4 |
| learned_gate | 1.000 | 0.000 | 0.424 | 1/190/0 |

All pairwise exact McNemar tests on retention return p=1.0 among the three
memory-using arms; the learned gate is significantly worse than the rules on
the decision itself (p<0.0001, kappa 0.009 vs 0.343). The learned gate degrades
because `retrieval_context=None` in the editor makes four of its features
constant -- see W3 `results/gate_online_gap.json` and W6 finding B1.

Locality is 1.000 for every arm and therefore uninformative here: with ~0-2%
of edits actually written to the weights, there is almost nothing to leak.

## Re-run

```bash
cd "TANG I CHOI/w4_gate_ablations"
python run_w4_gate_ablation.py --all-arms --subjects 40      # ~7 min on M3 Max, MPS
python analyze_w4_statistics.py                              # seconds
```

`--subjects 20` gives a fast smoke run; results are appended per arm, so a
partial run is obvious from the missing JSONs. Requires
`EasyEdit/hugging_cache/gpt2-xl` (not tracked) and an installed `easyeditor`.
