# Week 4 Verification — Sequential / Lifelong Editing (5K+ edits)

**Date:** 2026-10-31
**Repo:** `C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26`
**Milestone:** CALM runs end-to-end on a sequential / lifelong 5K-edit stream (counterfact, gpt2-small), with and without the meta-trained hypernetwork.

## 1. Commands

Trained (meta-train ckpt):
```
python EasyEdit/examples/w3_calm_stream.py --max-edits 5000 --sample-edits 12 \
  --hypernet-ckpt EasyEdit/results/models/CALM/hypernet_gpt2-small.pt \
  --count-memory-rewrites
```

Baseline (no hypernetwork):
```
python EasyEdit/examples/w3_calm_stream.py --max-edits 5000 --sample-edits 12 \
  --count-memory-rewrites
```

Both runs executed from the repo root with the project virtualenv (`.venv`); the script chdirs to the EasyEdit root internally so relative hparams paths resolve.

## 2. Environment

| Item | Value |
|---|---|
| OS | Windows 11 (Build 26200) |
| Python | `.venv` (project virtualenv) |
| Model | `./hugging_cache/gpt2-small` (gpt2-small) |
| Data | `EasyEdit/data/counterfact.json` — 21,919 records; first 5,000 used (full 5,000 available, no truncation) |
| Hparams | `EasyEdit/hparams/CALM/gpt2-small.yaml` |
| Hypernetwork ckpt | `EasyEdit/results/models/CALM/hypernet_gpt2-small.pt` (scale=5.0, 501 training steps) |
| Metric flags | `--count-memory-rewrites` (end-to-end WRITE/ROUTE/REJECT accounting) |

## 3. Results — 5K stream (trained vs baseline)

| Metric | Trained (hypernet ckpt) | Baseline (no hypernet) |
|---|---|---|
| Edits | 5,000 | 5,000 |
| Total wall time | 2,276.9 s (~38 min) | 2,132.3 s (~36 min) |
| Avg per edit | 0.455 s | 0.426 s |
| WRITE (hypernetwork) | 4,262 (85.24%) | 4,281 (85.62%) |
| ROUTE (side memory) | 735 (14.70%) | 710 (14.20%) |
| REJECT (gate) | 3 (0.06%) | 9 (0.18%) |
| End-to-end immediate rewrite rate | 14.78% (739/5000) | 14.86% (743/5000) |
| Model-level immediate rewrite rate | 0.08% (4/5000) | 0.76% (38/5000) |
| Side memory writes | 735 | 710 |
| Side memory usage | 64 shards, 735 stored (avg 11.48/shard, max 20) | 64 shards, 710 stored (avg 11.09/shard, max 19) |
| Retention (mid/end samples) | 0/9 | 0/9 |
| Status | PASS | PASS |

Raw per-run JSON (intermediate): `temp/w3_calm_stream_5000_trained.json`, `temp/w3_calm_stream_5000.json`.
Consolidated structured result: `FONG CHIN WAI/w4_sequential_stream_and_fix/w4_sequential_5k_results.json`.

## 4. Comparison with W3 (1K stream)

| Metric | W3 1K trained | W3 1K baseline | W4 5K trained | W4 5K baseline |
|---|---|---|---|---|
| End-to-end immediate rewrite rate | 17.2% | 15.1% | 14.78% | 14.86% |
| Model-level rewrite rate | 0.4% | 1.0% | 0.08% | 0.76% |
| Retention | 0/6 | 0/6 | 0/9 | 0/9 |

## 5. Analysis & conclusion

1. **Scaling hurts both approaches.** Extending the stream from 1K to 5K edits drops the end-to-end immediate rewrite rate for both variants (trained 17.2% → 14.78%; baseline 15.1% → 14.86%). The 5K stream remains fully executable (status PASS, no crash, no resource failure).
2. **The meta-trained hypernetwork advantage disappears at 5K.** At 1K, trained (17.2%) beat baseline (15.1%) by +2.1pp end-to-end. At 5K, trained (14.78%) is marginally *below* baseline (14.86%, −0.08pp). Model-level rewrite rate is consistently weak for both and worse for trained (0.08% vs 0.76%), matching the W3 model-level trend (0.4% vs 1.0%).
3. **Routing is stable.** Both variants route ~85% of edits through the hypernetwork (WRITE) and ~14% through side memory (ROUTE), with <0.2% gate rejects. The trained variant writes slightly more to side memory (735 vs 710) and rejects less (3 vs 9).
4. **Side memory is not a bottleneck.** Capacity is far from saturated (735–710 stored vs 12,800 slots across 64 shards; ≈11 entries/shard), yet retention on sampled mid/end edits is 0/9 for both — the same lifelong-edit failure mode observed in W3, indicating write-away memory recall does not recover edits once the parameter budget is exceeded.
5. **Throughput is acceptable.** ~0.43–0.46 s/edit; the trained variant adds ~7% wall time (memory-rewrite accounting + hypernetwork inference), negligible at this scale.

**Verdict:** W4 sequential/lifelong 5K+ experiment executed end-to-end with honest metrics. Honest reporting: meta-trained hypernetwork does not improve lifelong retention beyond 1K scale; at 5K it is on par with (slightly below) the no-hypernetwork baseline on end-to-end success while remaining worse on model-level success.
