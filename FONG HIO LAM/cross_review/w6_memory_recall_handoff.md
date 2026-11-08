# Handoff: measured `memory_recall` for the W6 quality notebook

From: **Sonia (FONG HIO LAM, MC654953)** — side-memory / router owner.
To: **Vivi (MC653748)** — owner of `FONG CHIN WAI/w6_deferred/W6_quality_check.ipynb`.
Purpose: replace the placeholder `memory_recall = 1.0` with measured values and record the effect
on the end-to-end ceiling, as required by handbook §5 (the note asking that the notebook's
`memory_recall` be switched to the measured value and the end-to-end ceiling recomputed once the
S1-S6 fixes are in).

---

## 1. What to change

`W6_quality_check.ipynb`, step 0, `CONFIG`:

```python
"memory_recall": 1.0,    # <- replace
```

Use one of these two, and say which one in `CONFIG["note"]`:

| value | source column | when to use |
|---|---|---|
| **0.80** | `memory_canonical_recall` | the stream re-asks with the stored prompt string |
| **0.45** | `memory_paraphrase_recall` | conservative; the stream re-asks in the user's own wording |

Both come from `FONG HIO LAM/side_memory_router/w3_memory_channel_gpt2xl.json`
(`n_routed_edits = 20`, canonical 16/20, paraphrase 9/20, shuffled 0/20, cue-only 0/20).
The notebook already writes `memory_recall_assumed` into every `quality_log.jsonl` record, so the
two runs stay distinguishable and neither overwrites the other.

Do **not** use `legacy_constructive_recall` (1.00) — that column is an index hit from the old
keying and is kept only as a record of the defect.

## 2. Effect on the end-to-end ceiling

Recomputed from the §5 rows of `W6_quality_check_handbook_TC.docx` (each slice n = 100,
`ceiling = hit_total + memory_share × recall`; the 1.00 column reproduces the handbook's published
0.95 / 0.92 / 0.97 / 0.96 / 0.97 / 1.00 / 0.87 / 0.95 exactly, so the arithmetic is the notebook's):

| slice | parameter hits / n | memory routed / n | ceiling @1.00 (current) | ceiling @0.80 | ceiling @0.45 |
|---|---|---|---|---|---|
| 0-99, baseline W=50 | 0.56 | 0.39 | 0.95 | **0.87** | **0.74** |
| 0-99, K | 0.59 | 0.33 | 0.92 | **0.85** | **0.74** |
| 100-199, baseline W=50 | 0.20 | 0.77 | 0.97 | **0.82** | **0.55** |
| 100-199, K | 0.55 | 0.41 | 0.96 | **0.88** | **0.74** |
| 200-299, baseline W=50 | 0.12 | 0.85 | 0.97 | **0.80** | **0.50** |
| 200-299, K | 0.11 | 0.89 | 1.00 | **0.82** | **0.51** |
| 300-399, baseline W=50 | 0.36 | 0.51 | 0.87 | **0.77** | **0.59** |
| 300-399, K | 0.61 | 0.34 | 0.95 | **0.88** | **0.76** |
| **mean** | **0.3875** | **0.5613** | **0.949** | **0.837** | **0.640** |

Headline: the ceiling loses **11.2 percentage points** at the canonical recall and **30.9 points**
at the paraphrase recall. The memory channel carries 56% of all edits on average, so the assumption
was never a rounding detail — it was more than half of the reported ceiling.

## 3. Caveat that must travel with the number

The recall was measured on **gpt2-xl**, the streams in the quality table were run on **gpt2-small**,
and the routed edits are a different 20-edit sample from the four 100-edit slices. Applying 0.80 or
0.45 across that boundary is an **assumption transferred between backbones**, not a measurement on
those runs. Two consequences for the write-up:

1. Do not present 0.837 as a measured end-to-end number. It is `ceiling @ recall = 0.80`, and the
   label has to carry that.
2. The measured quantity is memory **recall**, not end-to-end accuracy: the ceiling assumes the
   model returns the note's answer verbatim once the note is in the prompt. The canonical column
   shows that even with a verbatim prompt it only does so 16/20 of the time, so 0.837 is still an
   upper bound.

If a single number is wanted, quote the paraphrase column (0.45), because the canonical column
reuses the stored prompt string verbatim and flatters the channel — the same mistake as quoting
97.48% as memory accuracy.

## 4. Files behind this handoff

| file | content |
|---|---|
| `FONG HIO LAM/side_memory_router/w3_memory_channel.py` | measurement script (cue-only baseline, canonical, paraphrase, shuffled controls) |
| `FONG HIO LAM/side_memory_router/w3_memory_channel_gpt2xl.json` | the 20 routed cases and all four recall columns |
| `FONG HIO LAM/side_memory_router/side_memory.py` | keyed on `(subject, relation)`, reader on the answer path |
| `FONG HIO LAM/cross_review/w6_replies_to_cross_review_requests.md` | the answers to your three cross-review requests |
