---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 8aaebaa239fa14c1cf5eb03fd99189ca_ce8d78fed79611f1a216525400248c00
    ReservedCode1: EuN2phG5a4lEsD89WKudx3mmhnbAeMKSZWEXWzAE3xhOpQkX4wa2hl/Ytv57psz+TlCGL8OULP+TYEoO8FGofqL9+BpfVCy9ltkuBnCeTw0t/G9jkd4iV/J/IIEb/p4z6amveRep8zo777cnm+NtprYrRTmCYDocgeS4IPDNTlqnDHpQR50cKeQ+vWg=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 8aaebaa239fa14c1cf5eb03fd99189ca_ce8d78fed79611f1a216525400248c00
    ReservedCode2: EuN2phG5a4lEsD89WKudx3mmhnbAeMKSZWEXWzAE3xhOpQkX4wa2hl/Ytv57psz+TlCGL8OULP+TYEoO8FGofqL9+BpfVCy9ltkuBnCeTw0t/G9jkd4iV/J/IIEb/p4z6amveRep8zo777cnm+NtprYrRTmCYDocgeS4IPDNTlqnDHpQR50cKeQ+vWg=
---

# Results (Draft)

> Draft English Results section for the final report, based on the audited repository results (W2–W4, corrected in W5). All figures trace to the data sources listed at the end.

## 4.1 Single-fact editing baselines (W2)

We reproduced three single-fact editing baselines on gpt2-xl under one evaluation protocol (reliability / generalisation / locality / portability), with n=50 per dataset:

- **ROME** (parametric, rank-one update): rewrite 1.000 / rephrase 0.700 / neighborhood 0.842 on CounterFact, and 0.997 / 0.807 / 0.966 on zsRE. On MQuAKE, rewrite reached 0.947 while multi-hop portability stayed at 0.069, confirming that a single-hop parameter edit transfers very little to multi-hop composition.
- **IKE** (memory-based, no weight change): rewrite 1.000 / rephrase 0.720 / neighborhood 0.188 on CounterFact, and 0.993 / 0.985 / 0.670 on zsRE. IKE is reliable and rephrase-strong, but its retrieval-context mechanism perturbs unrelated prompts: neighborhood accuracy drops to 0.188.
- **GRACE** (memory-based, single-layer gradient overwrite): rewrite 1.000 / locality 1.000 on both splits, but rephrase 0.000 — a method property, because the retrieval key is the hidden state of the last prompt token, so paraphrases miss the key.

Together the three baselines show that "memory-based" is not a single behaviour: locality is guaranteed only when the memory read provably leaves unrelated answer distributions unchanged, and generalisation depends on the retrieval-key alignment with the query, not on the method label.

## 4.2 CALM sequential editing: system-level lifelong stream (W3/W4)

On gpt2-small with CounterFact, we first ran the CALM sequential stream without write control. At 1K edits the end-to-end immediate rewrite rate was 17.2% (trained hypernetwork) vs 15.1% (baseline). Scaling to 5K degraded both (trained 14.78%, baseline 14.86%); at 5K the meta-trained hypernetwork no longer outperformed the baseline, and mid/end-of-stream retention was 0/9 for both configurations. Side-memory capacity was far from saturated (~11 entries/shard vs 200), so the bottleneck is not capacity.

We then added two bounded-write guards in the editor — a write budget (`write_budget=128`) that downgrades further parameter WRITEs to ROUTE once the budget is spent, and dual-write that mirrors every parameter write in side memory. On the same 5K stream this raised the system-level end-to-end rewrite rate to **97.48%** (4,870/5,000 edits routed to memory, 128 parameter WRITEs) with **9/9** retention on sampled mid/end-of-stream edits. The fix makes the *system-level* lifelong stream succeed: retention becomes memory-backed end-to-end, and parameter writes are bounded.

## 4.3 CALM model-level write quality (W4, audited in W5)

The remaining weakness was model-level: whether a *parameter write alone* makes greedy decoding output the target. The W4 campaign addressed the root cause — the W3 MEND-style training restored the parameter snapshot every step, so the hypernetwork only learned effective deltas on a clean base — by switching to accumulation-aware training with warm-started segments. The best configuration (`accum16 + warm-start 48, 1000 steps`) raised the raw model-level metric from 0.08% to 22.2% (111/500) and the pre-audit "WRITE hit rate" to 86.7%.

**Audit correction.** A per-edit audit of the 128 real-stream WRITEs (W5) showed that the raw numbers were inflated: the pre-audit convention counted every gate-level success, including edits the gate REJECTed because the model already answered correctly. For the best checkpoint, 107 of the 111 raw hits are REJECT false hits (`current_answer_matches_target`), ROUTE contributes 0, and the genuine parameter-write hit rate is only **4/128 (3.1%)** — i.e. 124/128 real WRITEs still miss. The audited model-level improvement is therefore **0.08% → 3.1%** on real WRITEs, not 0.08% → 22.2%.

We audited coherence-penalty and multi-seed routes under the same protocol; none exceeded the baseline's real write quality: `coh_l0.1` 5/128 (3.9%), `coh_v2_l0.3` 3/128 (2.3%), `v2_seed101` 2/128 (1.6%), all with ROUTE hits 0 and dominated by REJECT false hits (82/40/28 respectively).

The gate threshold sweep shows that the default `route_threshold=0.6` is already best for the model-level metric; lowering it to 0.3–0.35 shifts edits to side memory and raises the system-level rewrite rate to 96–100% at the cost of parameter-write quality.

## 4.4 Limitations and open problems

- **Parameter-write quality remains low.** Real WRITE hits are ~1.6–3.9% across audited checkpoints; accumulation-aware warm-start training improves the metric relative to W3 (0.08%) but does not yet make parameter writes reliable. Further warm-started training (e.g. lr decay, 1500+ steps) or side-memory-leaning gate settings are candidate next steps.
- **Long-stream retention is memory-backed, not parameter-backed.** The 9/9 retention and 97.48% end-to-end rewrite depend on side-memory reads; parameter-write retention itself remains an open bottleneck.
- **Scaling limits.** At 5K edits the meta-trained hypernetwork no longer beats the baseline end-to-end; the lifelong benefit of meta-training is not yet demonstrated at scale.
- **Hardware.** GPT-J-6B full evaluation is infeasible on the local 8GB RTX 4070 (validated only as a 10-case smoke on Colab T4 16GB); the official Llama-3-8B evaluation is planned on ≥16GB hardware.

## Data sources

- `FONG CHIN WAI/w2_rome_baseline/w2_rome_reproduction.md`, `docs/w2_ike_grace_reproduction.md`
- `FONG CHIN WAI/w3_hypernetwork_stream/w3_calm_stream_1k.json`, `FONG CHIN WAI/w4_sequential_stream_and_fix/w4_sequential_5k_results.json`, `FONG CHIN WAI/w4_sequential_stream_and_fix/w4_calm_fix_5k_results.json`
- `FONG CHIN WAI/w4_model_level_improve/results/audit_baseline.json`, `audit_coh_l0.1.json`, `audit_coh_v2_l0.3_c0.6.json`, `audit_v2_seed101.json`
- `FONG CHIN WAI/w5_results_and_figures/w4_w5_main_results.md` (consolidated main-results table and charts)
*（内容由AI生成，仅供参考）*
