# W0 deliverable -- bibliographic verification of the 12 papers

> **Path note (2026-11-05).** Fong Chin Wai's artefacts were consolidated into `FONG CHIN WAI/` (see `FONG CHIN WAI/README.md` for the old-path to new-path mapping). References below have been updated; the bare directory names `output/` and `calm/` that appear in narrative text now live under `FONG CHIN WAI/`.


Owner: Tang I Choi (MC631517) · work package W0 (7-10 Oct) · **verification run
2026-11-05**, recorded here because W0's output had no file of its own in the
repository (it lived only in the review thread and in the group README).

Scope: verify that every paper the project actually builds on exists, that its
identifier is correct, and that it is openly accessible with a citable link. The
motivation is concrete: during W1/W2 review three commonly circulated arXiv
identifiers turned out to be wrong (table 2), and a wrong identifier in a
bibliography is indistinguishable from a fabricated citation.

## How each entry was verified

1. Open the `arxiv.org/abs/<id>` page (or query the arXiv API by title).
2. Confirm the **title** on the page matches the method the team cites.
3. Record the submission history (a stable v1 date indicates a real record).
4. Check that a full-text PDF is downloadable without a subscription.

Every row below was checked this way on 2026-11-05, not copied from a
bibliography file.

## Table 1 -- the 12 papers, verified

| # | method | title (as on the abs page) | arXiv | venue | open access |
|---|---|---|---|---|---|
| 1 | KnowledgeEditor | Editable Neural Networks | [2004.00345](https://arxiv.org/abs/2004.00345) | ICLR 2020 | abs + PDF, no paywall |
| 2 | MEND | Fast Model Editing at Scale | [2110.11309](https://arxiv.org/abs/2110.11309) | ICLR 2022 | abs + PDF, no paywall |
| 3 | ROME | Locating and Editing Factual Associations in GPT | [2202.05262](https://arxiv.org/abs/2202.05262) | NeurIPS 2022 | abs + PDF, no paywall |
| 4 | SERAC | Memory-Based Model Editing at Scale | [2206.06520](https://arxiv.org/abs/2206.06520) | ICML 2022 | abs + PDF, no paywall |
| 5 | MEMIT | Mass-Editing Memory in a Transformer | [2210.07229](https://arxiv.org/abs/2210.07229) | ICLR 2023 | abs + PDF, no paywall |
| 6 | GRACE | Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adapters | [2211.11031](https://arxiv.org/abs/2211.11031) | NeurIPS 2023 | abs + PDF, no paywall |
| 7 | IKE | Can We Edit Factual Knowledge by In-Context Learning? | [2305.12740](https://arxiv.org/abs/2305.12740) | EMNLP 2023 | abs + PDF, no paywall |
| 8 | MQuAKE | MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions | [2305.14795](https://arxiv.org/abs/2305.14795) | EMNLP 2023 | abs + PDF, no paywall |
| 9 | MALMEN | Massive Editing for Large Language Models via Meta Learning | [2311.04661](https://arxiv.org/abs/2311.04661) | arXiv preprint (venue unconfirmed at check time -- cite as preprint) | abs + PDF, no paywall |
| 10 | WISE | Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models | [2405.14768](https://arxiv.org/abs/2405.14768) | NeurIPS 2024 | abs + PDF, no paywall |
| 11 | AlphaEdit | AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models | [2410.02355](https://arxiv.org/abs/2410.02355) | ICLR 2025 | abs + PDF, no paywall |
| 12 | WikiBigEdit | Understanding the Limits of Lifelong Knowledge Editing in LLMs | [2503.05683](https://arxiv.org/abs/2503.05683) | arXiv preprint (2025) | abs + PDF, no paywall |

Note on #9 and #12: no proceedings version could be confirmed at check time, so
they are cited as preprints rather than assigned a venue. This is deliberate --
an unverified venue line is exactly the kind of error this work package exists to
catch.

## Table 2 -- identifiers corrected during verification

These are the failures the check actually found. Each was a plausible-looking
arXiv number that resolves to an unrelated paper, which is why they are recorded
rather than silently replaced.

| claimed / circulated | what that id actually is | correct id |
|---|---|---|
| `2110.00359` for MEND | "Finite Time Exact Quantized Average Consensus ... for Energy-Aware Networks" (control theory) | `2110.11309` (Fast Model Editing at Scale) |
| `2210.02167` for IKE | "Faster parameterized algorithms for modification problems to minor-closed classes" (graph algorithms) | `2305.12740` (Can We Edit Factual Knowledge by In-Context Learning?) |
| `2306.07842` for SERAC | "Progressive Segmentation-guided Scene Text Removal Network" (vision) | `2206.06520` (Memory-Based Model Editing at Scale) |
| `2406.04733` for WikiBigEdit | "Unsupervised representation learning with Hebbian synaptic and structural plasticity ..." (computational neuroscience) | `2503.05683` (Understanding the Limits of Lifelong Knowledge Editing in LLMs) |

All four wrong ids resolve successfully -- they simply point at a different
paper. That is the failure mode a title check catches and a spot-check of the
link does not.

## Where each paper is used in this repository

| paper | artefact that depends on it |
|---|---|
| KnowledgeEditor, MEND | `EasyEdit/easyeditor/models/` editor registry; the hypernetwork lineage that `EasyEdit/easyeditor/models/calm/hypernetwork.py` follows |
| ROME | `FONG CHIN WAI/w2_rome_baseline/w2_rome_gptj6b_colab_v*.ipynb`, `FONG CHIN WAI/w2_rome_baseline/w2_rome_reproduction.md` |
| MEMIT, AlphaEdit | `FONG HIO LAM/` batch-editing baselines |
| SERAC, WISE | side-memory design in `FONG HIO LAM/side_memory_router/` |
| GRACE, IKE | W2 memory-based reproduction, `docs/w2_ike_grace_reproduction.md`, `w2_ike_grace_gptj6b_colab_v1.ipynb` |
| MALMEN | the multi-edit hypernetwork idea discussed for W3 |
| MQuAKE, WikiBigEdit | `FONG HIO LAM/` multi-hop and lifelong subsets |
| GRACE (again) | the locality condition argued in the root `README.md` §"Memory-based methods and generalisation" |

## Status of this work package

| item | state |
|---|---|
| 12 papers identified and title-verified | done (2026-11-05) |
| Identifiers corrected (4) | done, table 2 |
| Open-access status confirmed for all 12 | done |
| Venue lines | 10 of 12 confirmed; MALMEN and WikiBigEdit left as preprints |
| Re-check before final submission | pending (W6 checklist, `w6_cross_review/docs/pre_submission_checklist.md`) |
