# CALM: Conflict-Aware Lightweight Model Editor

A knowledge editing framework for large language models built on **EasyEdit**.

CALM (Conflict-Aware Lightweight Model Editor) examines the trade-off among **reliability**, **generalisation**, and **locality** in lifelong knowledge editing of large language models (LLMs). Existing editors degrade under a stream of sequential edits: weight-modifying methods such as ROME and MEMIT accumulate interference, while non-parametric methods such as GRACE and IKE are often claimed to preserve locality but generalise weakly. **These claims are conditional, not universal**: in our reproduction, GRACE indeed preserves locality but shows zero generalisation, whereas IKE generalises moderately on zsRE but *does not* preserve locality on CounterFact. The precise conditions are documented in the section *Memory-based methods and generalisation: claims and conditions* below, together with concrete experimental numbers. CALM adds two lightweight components to an existing editing method:

- **Hypernetwork Edit Generator** — maps a single subject–relation–object edit into a low-rank weight update, so that editing does not require per-edit gradient descent on the full model.
- **Conflict Gate** — scores a candidate edit against (a) the model's current answer distribution, (b) the closest previously stored edit, and (c) the retrieved context, deciding whether to write to parameters, route to a non-parametric side memory (WISE-style), or reject a genuine knowledge conflict.

This repository uses EasyEdit (https://github.com/zjunlp/EasyEdit) as its code base, located under `EasyEdit/` (its inner `.git` has been removed so that it becomes part of this repository). All three members work on the same code base.

## Team Division of Labour

| Member | Student ID | Role Track | Responsibilities |
| --- | --- | --- | --- |
| Fong Chin Wai | MC653748 | Framework & parametric editing | EasyEdit experiment framework and unified evaluation protocol; ROME single-fact baseline; hypernetwork edit generator; sequential/lifelong editing experiment; main results table and figures |
| Fong Hio Lam | MC654953 | Data pipeline & memory routing | Data loaders and metric scripts for CounterFact, zsRE, MQuAKE; MEMIT/AlphaEdit baseline reproduction; sharded side memory and write router; MQuAKE multi-hop ripple test and WikiBigEdit subset; error analysis by conflict type |
| Tang I Choi | MC631517 | Conflict annotation & gating classification | Bibliographic verification of the 12 papers; conflict-annotation manual and 100-edit pilot annotation; IKE/GRACE baseline reproduction; conflict-gate training data and gate classifier; gate ablations and statistical analysis; report coordination |

## Weekly Work Packages (W0–W6)

Work is divided evenly: each week assigns exactly one work package to each of the three members, so each member owns **6 work packages of comparable scope** (18 in total). The schedule spans W0–W6. Milestones are relative to the Task 1 deadline (10 October 2026, 23:55) and the intended project completion in mid-November 2026.

| Week | Fong Chin Wai | Fong Hio Lam | Tang I Choi |
| --- | --- | --- | --- |
| W0 (7–10 Oct) | Consolidate the seven-candidate comparison table and the selection rationale | Draft the literature review and the taxonomy (Sections 2–3) | Verify the bibliographic data and open-access links of the 12 papers |
| W1 (12–18 Oct) | Set up the EasyEdit experiment framework and the unified evaluation protocol | Build the data loaders and metric scripts for CounterFact, zsRE and MQuAKE | Write the conflict-annotation manual and pilot-label 100 edits |
| W2 (19–25 Oct) | Reproduce ROME (single-fact baseline) | Reproduce MEMIT (batch-editing baseline) and AlphaEdit (null-space baseline, folded into the MEMIT pipeline) | Reproduce IKE and GRACE (memory-based baselines) |
| W3 (26 Oct–1 Nov) | Implement the hypernetwork edit generator | Implement the sharded side memory and the write router | Build the conflict-gate training data and train the gate classifier |
| W4 (2–8 Nov) | Run the sequential/lifelong editing experiment (5K+ edits) | Run the MQuAKE multi-hop ripple test and the WikiBigEdit subset | Run the gate ablations (always-write / always-route / gated) and the statistical analysis |
| W5 (9–15 Nov) | Produce the main results table and figures; draft the results section | Error analysis by conflict type and the conflict-distribution figure | Coordinate the report, the reproducible scripts and the logs |
| W6 (16–17 Nov) | Final cross-review and submission | Final cross-review and submission | Final cross-review and submission (all three members) |

## Collaboration Guidelines

- **Shared code base (EasyEdit)** — All members work on the same code base. CALM is implemented as a new `BaseEditor` subclass in EasyEdit: the hypernetwork produces the update, the conflict gate decides the write destination, and the side memory is sharded. Interfaces are agreed up front; no member is assigned a lighter or heavier role, and code, experiments and writing are merged in the same way by all three members.
- **Branch strategy** — Feature development is carried out on separate branches and merged into `main` after review; the final cross-review and submission are done jointly by all members at W6.
- **Stage gates** — Progress is measured by what has actually been produced rather than by any particular setup:
  - **Gate 1 (end of W1):** the three data pipelines and the metric scripts all run.
  - **Gate 2 (end of W2):** the reproduced baselines (ROME; MEMIT + AlphaEdit; IKE and GRACE) produce reliability / generalisation / locality numbers under one protocol.
  - **Gate 3 (end of W3):** CALM runs end-to-end on a 1K-edit stream.
  - **Gate 4 (end of W4):** the results table and the ablation table are complete; if CALM does not beat the best baseline on retention, the ablation and the analysis are reported instead.
- **Fallback (degraded) plan** — If the hypernetwork cannot be meta-trained within the planned timeframe, CALM degrades to the conflict gate alone (a routing layer that chooses between ROME-style parametric writing and GRACE-style memory writing). This fallback is implemented and evaluated by all three members within the same schedule, so the even split is unaffected.
- **Early validation** — Because the meta-trained hypernetwork is the highest-risk component, its training pipeline is validated at small scale on GPT-J-6B during W1 (rather than first attempting it in W3), so that any problem is surfaced early and the conservative fallback can be triggered in time.

## Concrete Experimental Results

All runs below share one evaluation protocol (reliability = `rewrite_acc`, generalisation = `rephrase_acc`, locality = `neighborhood_acc` / `nq_acc`), as defined in `FONG CHIN WAI/w1_framework_and_protocol/evaluation_protocol.md`. Detailed logs are linked under each table.

### W2 — Single-fact baselines (gpt2-xl, n=50 per dataset)

| Method | Type | Data | rewrite | rephrase | locality |
|---|---|---|---|---|---|
| ROME (Fong Chin Wai) | parametric | CounterFact | 1.000 | 0.700 | 0.842 |
| ROME | parametric | zsRE | 0.997 | 0.807 | 0.966 |
| IKE (Tang I Choi) | memory-based | CounterFact | 1.000 | 0.720 | 0.188 |
| IKE | memory-based | zsRE | 0.993 | 0.985 | 0.670 |
| GRACE (Tang I Choi) | memory-based | CounterFact | 1.000 | 0.000 | 1.000 |
| GRACE | memory-based | zsRE | 1.000 | 0.000 | 1.000 |
| MEMIT (Fong Hio Lam) | parametric | toy smoke | 1.000 | 0.500 | 1.000 / 1.000 |
| AlphaEdit (Fong Hio Lam) | parametric | toy smoke | 1.000 | 0.333 | 1.000 / 1.000 |

ROME portability on MQuAKE multi-hop questions is weak (p1 = 0.069, n=50), consistent with the ROME paper. GPT-J-6B was validated end-to-end on Colab T4 (zsRE n=10: 1.000 / 0.950 / 0.610); it is **not feasible on the local RTX 4070 8GB** (weights + activations exceed physical VRAM; ~570 s/case with system-memory fallback), see `FONG CHIN WAI/w2_rome_baseline/w2_rome_reproduction.md` §6. Evidence: `FONG CHIN WAI/w2_rome_baseline/w2_rome_reproduction.md`, `docs/w2_ike_grace_reproduction.md`, `FONG HIO LAM/baseline_repro/README.md`.

### W3/W4 — CALM sequential editing streams (gpt2-small, CounterFact)

| Stream | Variant | WRITE | ROUTE | REJECT | e2e immediate | model-level immediate | retention |
|---|---|---|---|---|---|---|---|
| W3 1K | trained hypernet | 85.5% | 14.1% | 0.4% | 17.2% | 0.4% | 0/6 |
| W3 1K | baseline (no hypernet) | — | — | — | 15.1% | 1.0% | 0/6 |
| W4 5K | trained hypernet | 85.2% | 14.7% | 0.1% | 14.78% | 0.08% | 0/9 |
| W4 5K | baseline (no hypernet) | 85.6% | 14.2% | 0.2% | 14.86% | 0.76% | 0/9 |
| W4 5K | + write_budget=128, dual-write, e2e retention | 2.56% (128) | 97.4% (4870) | 2 | **97.48%** | 0.08% | **9/9** |

The last row is the fix that makes the *system-level* lifelong stream succeed: end-to-end rewrite rate rises to 97.48% and retention is memory-backed (9/9), because parameter writes are bounded (128 WRITEs) and every write is also mirrored in side memory. The trained hypernetwork's *model-level* write quality (whether a parameter write alone makes greedy decoding output `target_new`) was the remaining weakness (0.08%); the improvement campaign below addresses exactly that. Evidence: `FONG CHIN WAI/w3_hypernetwork_stream/w3_calm_stream_verification.md`, `FONG CHIN WAI/w4_sequential_stream_and_fix/w4_sequential_5k_verification.md`, `FONG CHIN WAI/w4_sequential_stream_and_fix/w4_calm_fix_report.md`.

### W4 — Model-level write quality improvement (500-edit stream, write_budget=128, dual-write)

Root-cause diagnosis (inference-scale sweep + cumulative-pollution probe, see `FONG CHIN WAI/w4_model_level_improve/w4_model_level_improve_diagnosis.md`): the W3 MEND-style training restores the parameter snapshot every step, so the hypernetwork only learns to emit effective deltas on a *clean* base; on an already-polluted parameter stack it stops producing observable rewrites (clean-baseline hit 5/10 vs cumulative 2/10, with every delta after step 2 invisible in greedy decoding). The fix is to change the training paradigm to **accumulation-aware training** (do not restore snapshots inside a segment; warm-start the segment on a pre-polluted stack), not to scale up the delta (scale=10 in-stream actually hurts: 2/129 vs 4/131).

| Training configuration | Model-level immediate | WRITE hit rate | Audited real WRITE hit (128-write audit) |
|---|---|---|---|
| W3 old ckpt (single-edit) | 4/500 (0.8%) | 3.1% | un-audited |
| accum, segment_len=4 | 8/500 (1.6%) | 6.3% | un-audited |
| accum, segment_len=16 | 51/500 (10.2%) | 39.8% | un-audited |
| accum16 + gate-distribution sampling | 32/500 (6.4%) | 25.0% | un-audited |
| accum, segment_len=32 | 37/500 (7.4%) | 28.9% | un-audited |
| accum16 + warm-start 32 | 56/500 (11.2%) | 43.8% | un-audited |
| **accum16 + warm-start 48 (best)** | **85/500 (17.0%)** | **66.4%** | un-audited |
| accum16 + warm-start 64 | 35/500 (7.0%) | 27.3% | un-audited |
| **accum16 + warm-start 48, 1000 steps** | **111/500 (22.2%)** | **86.7%** | **4/128 (3.1%)** |

Warm-start depth follows a single-peaked curve: 48 is optimal because most of the 128 real-stream WRITEs occur at pollution depth 16–64; depth 64 over-trains the hypernetwork to emit small conservative deltas that miss on the shallower real stream. **Audit correction (W5).** The "WRITE hit rate" column above is the pre-audit `model_level_ok/128` convention, which is inflated: it counts *every* gate-level success in the 500-edit stream, including edits that the gate REJECTed because the model already answers correctly. A per-edit 128-write audit of the best checkpoint (`accum16 + warm-start 48, 1000 steps`) shows that the real WRITE hit rate is only **4/128 (3.1%)**, not 86.7%: of the 111 raw model-level hits, 107 are REJECT false hits (gate redundancy, where `current_answer_matches_target`), 4 are genuine parameter-write hits, and ROUTE contributes 0 (side-memory reads are not parameter-write quality). The remaining 124/128 parameter WRITEs still miss. The audited improvement is therefore **0.08% → 3.1%** on real WRITEs (raw 111/500 = 22.2% is inflated by 107 REJECT false hits), not 0.08% → 22.2%. Coherence-penalty and multi-seed routes audited under the same protocol do not exceed this real write quality: `coh_l0.1` 5/128 (3.9%), `coh_v2_l0.3` 3/128 (2.3%), `v2_seed101` 2/128 (1.6%), all with ROUTE hits 0 and REJECT false hits (82/40/28 respectively). A gate threshold sweep (see §4i of the training report) shows the default `route_threshold=0.6` is already best for the model-level metric; lowering it to 0.3–0.35 shifts edits to side memory and raises the *system-level* rewrite rate to 96–100% at the cost of parameter-write quality. Evidence: `FONG CHIN WAI/w4_model_level_improve/w4_accum_training_report.md`; audit JSONs: `FONG CHIN WAI/w4_model_level_improve/results/audit_baseline.json`, `audit_coh_l0.1.json`, `audit_coh_v2_l0.3_c0.6.json`, `audit_v2_seed101.json`.

## Memory-based methods and generalisation: claims and conditions

The README's opening comparison is only valid under explicit conditions, which we verified empirically:

- **"Memory-based methods preserve locality"** — holds for *parameter-touching* memory methods only: GRACE rewrites a single MLP layer (200 gradient steps + contrastive early stopping) and leaves all other weights untouched, so neighborhood accuracy is perfect (1.000) by construction. It does **not** hold for IKE, which leaves weights untouched but changes the *retrieval context*: injecting k=16 retrieved facts into the prompt perturbs the whole answer distribution, and neighborhood accuracy drops to 0.188 on CounterFact. Preserving locality therefore requires either a localised weight overwrite (GRACE) or a memory read that provably does not change the distribution of unrelated prompts — neither is guaranteed by "memory-based" alone.
- **"Memory-based methods generalise weakly"** — is a *retrieval-key* condition, not a method-level constant. GRACE's rephrase=0.000 is explained by its key-value design: it uses the hidden state of the *last prompt token* as the retrieval key; any paraphrase that changes the final token misses the key and falls back to pre-edit knowledge. IKE's rephrase is high when the paraphrase shares surface form with the edited prompt (zsRE 0.985) and lower when it does not (CounterFact 0.720), because in-context rewriting depends on retrieved-fact alignment with the query.
- **Generalisation metrics are not interchangeable** — `rephrase_acc` (paraphrase of the same fact) and `portability` (multi-hop transfer) measure different abilities: ROME reaches 0.700 rephrase but only 0.069 MQuAKE p1. Any claim about "generalisation" must state which split is meant and under what retrieval/paraphrase conditions.

## Conflict gate: when it writes, routes, or rejects

The gate (`FONG HIO LAM/side_memory_router/conflict_gate.py`) scores each edit on three signals and decides in this fixed priority order (defaults: `write_threshold=0.3`, `route_threshold=0.6`, `similarity_threshold=0.7`):

1. **REJECT — the model already answers correctly.** If `conf_score(current_answer, target_new) > 0.9` (identical answer, or a word-overlap score ≥ 0.9), the edit is rejected as redundant (`current_answer_matches_target`).
2. **ROUTE — the edit conflicts with a recent edit.** If any record in the recent-edit window (size `n_recent=50`) shares the same subject+relation, or its similarity `0.6·subject/relation-overlap + 0.4·target-word-Jaccard ≥ similarity_threshold`, the edit is routed to side memory to avoid overwriting an earlier fact (`conflicts_in_memory`).
3. **WRITE — the model is unsure.** If `current_confidence < write_threshold`, the edit is written directly to parameters (`low_model_confidence`).
4. **ROUTE — the retrieval context supports the edit.** If > 50% of retrieved snippets contain target words, the edit is routed to side memory (`high_retrieval_relevance`).
5. **Default — moderate confidence.** Below `route_threshold` → WRITE, otherwise → ROUTE (`default_moderate`).

**Deployment-level guards.** On near-all-distinct streams such as CounterFact (5,000 records → 4,908 unique subjects), signal 2 almost never fires because subjects do not overlap, so ~85% of edits fall through to low-confidence WRITE and the parameter budget saturates early (W3/W4 rows above). CALM therefore adds two bounded-write guards in the editor: (a) a **first-write guard** — a second edit on an already-written subject is downgraded to ROUTE; (b) a **write budget** (hparam `write_budget`) — once `write_budget` parameter WRITEs have been issued, every further WRITE is downgraded to ROUTE; (c) **dual-write** — every parameter WRITE is mirrored in side memory so retention is memory-backed end-to-end. With `write_budget=128` + dual-write the 5K stream routes 97.4% of edits to memory and reaches 97.48% end-to-end rewrite with 9/9 retention.

## Current status and remaining gap

The model-level write-quality campaign improved the raw model-level metric from 0.08% → 22.2% on the same local RTX 4070 8GB machine, but the W5 per-edit audit shows this is inflated: the real WRITE hit rate at warm48·1000 is **4/128 (3.1%)**, i.e. 124 of 128 parameter WRITEs still miss; 107 of the 111 raw hits are REJECT false hits (the gate REJECTed edits where the model already answers correctly, `current_answer_matches_target`), and ROUTE contributes 0 hits (side-memory reads are not parameter-write quality). Coherence-penalty and multi-seed routes audited under the same protocol do not exceed this: `coh_l0.1` 5/128 (3.9%), `coh_v2_l0.3` 3/128 (2.3%), `v2_seed101` 2/128 (1.6%). The root cause was the *training paradigm* (single-edit snapshot restoration), not GPU capacity. The remaining gap (parameter WRITE quality below ~3–4% real hits) is addressable by further warm-started training (e.g. lr decay, 1500+ steps) or side-memory-leaning gate settings (route_threshold 0.3–0.35 raises system-level rewrite to 96–100%), none of which require a stronger GPU. Hardware constraints do appear, however, at the 6B scale: GPT-J-6B full evaluation is infeasible on 8GB VRAM (validated instead on Colab T4), and the official Llama-3-8B evaluation is planned on ≥16GB hardware. Evidence: `FONG CHIN WAI/w2_rome_baseline/w2_rome_reproduction.md` §6, `FONG CHIN WAI/w4_model_level_improve/w4_accum_training_report.md`, `FONG CHIN WAI/w4_model_level_improve/results/audit_*.json`, `FONG CHIN WAI/w5_results_and_figures/w4_w5_main_results.md`.

## Getting Started

```bash
# Enter the project root
cd CALM-Editor-CISC7021-Task1-Group26

# The EasyEdit code base lives under EasyEdit/
# See EasyEdit/README.md for dependencies and usage instructions
```
