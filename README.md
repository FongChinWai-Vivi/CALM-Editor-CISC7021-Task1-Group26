# CALM: Conflict-Aware Lightweight Model Editor

A knowledge editing framework for large language models built on **EasyEdit**.

CALM (Conflict-Aware Lightweight Model Editor) examines the trade-off among **reliability**, **generalisation**, and **locality** in lifelong knowledge editing of large language models (LLMs). Existing editors degrade under a stream of sequential edits: weight-modifying methods such as ROME and MEMIT accumulate interference, while non-parametric methods such as GRACE and IKE preserve locality but generalise weakly. CALM adds two lightweight components to an existing editing method:

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

## Getting Started

```bash
# Enter the project root
cd CALM-Editor-CISC7021-Task1-Group26

# The EasyEdit code base lives under EasyEdit/
# See EasyEdit/README.md for dependencies and usage instructions
```
