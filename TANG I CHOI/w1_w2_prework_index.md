# W0-W2 deliverables: where they live and why

> **Path note (2026-11-05).** Fong Chin Wai's artefacts were consolidated into `FONG CHIN WAI/` (see `FONG CHIN WAI/README.md` for the old-path to new-path mapping). References below have been updated; the bare directory names `output/` and `calm/` that appear in narrative text now live under `FONG CHIN WAI/`.


Owner: Tang I Choi (MC631517). This file is the single entry point for my three
pre-W3 work packages. W3-W6 have folders of their own under `TANG I CHOI/`;
W0-W2 were produced before the per-member layout existed, so their artefacts sit
in team-level paths. **They are not moved or copied here**, because
`docs/w2_ike_grace_reproduction.md` and the annotation fold-br are linked from
the root `README.md` and from other members' documents, and relocating them
would break those links and rewrite history that other people own.

Status: all three packages are complete; W0's output had no file of its own until
this commit (it existed only in the review thread), so it was written up in
`w0_bibliography_verification.md`.

## Index

| package | state | artefact(s) in the repository | what it contains | owner check |
|---|---|---|---|---|
| **W0** (7-10 Oct) | complete, written up now | `TANG I CHOI/w0_bibliography_verification.md` | 12 papers, title-verified against arXiv on 2026-11-05; 4 wrong identifiers caught and corrected; open-access status for all 12; where each paper is used in this repo | this file + arXiv abs pages cited in it |
| **W1** (12-18 Oct) | complete | `conflict annotation manual + 100 pilot annotations/` -- `README.md`, `docs/conflict_annotation_manual.md`, `data/conflict_pilot_100.jsonl`, `tools/build_pilot_annotations.py` | four-label conflict scheme (no-conflict / pre-training-conflict / prior-edit-conflict / context-conflict) with a fixed priority, coding rules with worked examples, 100 human-labelled pilot edits, and the builder script that turns them into a jsonl | root `README.md` W1 row assigns "conflict-annotation manual and pilot-label 100 edits" to MC631517; the folder was committed as `cc74418` |
| **W2** (19-25 Oct) | complete | `w2_ike_grace_gptj6b_colab_v1.ipynb`, `docs/w2_ike_grace_reproduction.md` | IKE and GRACE reproduction: config table (gpt2-xl on MPS, Colab T4 for GPT-J-6B), the k=16 retrieval setup, GRACE's single-layer 200-step overwrite, and the numbers that entered the root README (IKE CounterFact rewrite 1.000 / rephrase 0.720 / locality 0.188; GRACE rephrase 0.000 / locality 1.000) | commit `ed7dcdf` "W2: Tang I Choi IKE/GRACE memory-based baselines reproduction"; root `README.md` W2 row assigns IKE/GRACE to MC631517 |

Not mine, listed only so the folder is unambiguous: `FONG CHIN WAI/w2_rome_baseline/w2_rome_gptj6b_colab_v5.ipynb`,
`FONG CHIN WAI/w2_rome_baseline/w2_rome_gptj6b_colab_v6.ipynb`, `FONG CHIN WAI/w2_rome_baseline/w2_rome_reproduction.md` (Fong Chin Wai);
`FONG CHIN WAI/w1_framework_and_protocol/w1_alignment_decision.md`, `FONG CHIN WAI/w1_framework_and_protocol/w1_hypernetwork_verification.md`,
`FONG CHIN WAI/w1_framework_and_protocol/evaluation_protocol.md`, `FONG CHIN WAI/w1_framework_and_protocol/W1_Team_Notice.docx` (team-level W1
decisions, Fong Chin Wai / Fong Hio Lam). None of these were touched.

## What the W1 annotation manual already covers, and where W4/W5 picks it up

| label | definition (W1) | gold action | used by |
|---|---|---|---|
| `no-conflict` | the fact is absent from the model; no competing edit | WRITE (model cannot answer) / REJECT (model already answers) | W3 training data, W4 §3.3 |
| `pre-training-conflict` | the edit contradicts knowledge the backbone already holds | REJECT | W3 training data (325 of 2584), W4 §3.3.1, W5 §4.2.3 |
| `prior-edit-conflict` | the edit contradicts an edit already in the stream | ROUTE | W3, W4 §3.3 |
| `context-conflict` | the retrieved context contradicts the candidate edit | ROUTE | W3, W4 §3.3 |

Priority order, fixed in W1 and honoured throughout W3-W5:
`context-conflict > prior-edit-conflict > pre-training-conflict`. The 100 pilot
edits are the only human-labelled signal in the whole project, which is why every
gate number in W5 is reported next to the 0.65/0.67 human-set ceiling.

## Known gap in the pre-W3 material (recorded, not papered over)

- The pilot covers 100 edits with 25 labelled `pre-training-conflict`. That is
  enough to validate the coding rules, not enough to train a per-label
  discriminator, and W4 §3.3.1 shows no arm can yet act correctly on this label
  (gold `REJECT`, no arm rejects). The W3 training set expands the label to 325
  records from CounterFact construction, which is a rule-derived label, not a
  human one -- stated in `w3_conflict_gate_training_and_classifier/gate_datasets/dataset_card.json`.
- No second annotator, so no inter-annotator agreement figure exists for W1. The
  pilot annotations were re-read once for consistency but not double-coded; this
  is a limitation of the label scheme rather than of the gate experiments built
  on it.
