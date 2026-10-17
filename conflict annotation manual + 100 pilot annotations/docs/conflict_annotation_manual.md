---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: a00e5df06e93a5902bdea1ae35e297b8_8c6dc2d7ca0111f18b7f525400393706
    ReservedCode1: 8riEAgdm2IzlG8cKXvx35tY15WSMe5ii46RVt5NxqKm9tMhLpe+FhiTwCoisMiassXRD1JaGKz3R4rDB6lzOi1TKc0adww+pyVoMM85FczWfm2VgKhHFVqP28GdUDovXExKzlR3CYPq/E1PuDu6vBjiV3V4KstWtWYHOTd+6wPKdl+uCN7NxH2N8pYk=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: a00e5df06e93a5902bdea1ae35e297b8_8c6dc2d7ca0111f18b7f525400393706
    ReservedCode2: 8riEAgdm2IzlG8cKXvx35tY15WSMe5ii46RVt5NxqKm9tMhLpe+FhiTwCoisMiassXRD1JaGKz3R4rDB6lzOi1TKc0adww+pyVoMM85FczWfm2VgKhHFVqP28GdUDovXExKzlR3CYPq/E1PuDu6vBjiV3V4KstWtWYHOTd+6wPKdl+uCN7NxH2N8pYk=
---

# Conflict Annotation Manual — CALM Project (W1)

**Author**: Tang I Choi (MC631517)
**Status**: Frozen for W1 pilot annotation; input to W3 ConflictGate training data
**Date**: 2026-10-17
**Related docs**: `Group26_CALM_Task1_Review_Report_ACL.pdf` (Table 2), `docs/w1_alignment_decision.md` (Section 2.2, ConflictGate)

---

## 1. Purpose and Scope

This manual defines how to annotate a **knowledge edit** with a conflict label for the CALM
project. CALM adds a **ConflictGate** to existing model editors: given a candidate edit
`(subject, relation, target)`, a candidate parameter update, and optional retrieved context,
the gate decides `Write | Route | Reject`. To train that gate (W3), we need labelled edits.

The four labels come from the knowledge-conflict taxonomy of Xu et al. (2024)
(*Knowledge Conflicts for LLMs: A Survey*, arXiv:2403.08319) as adapted in the Group 26
Task 1 review report (Table 2):

| Label | Definition | Example |
|---|---|---|
| `no-conflict` | The edit is consistent with pretraining knowledge, prior edits, and retrieved context. | A genuinely new fact the model never saw (e.g. a post-training event). |
| `pre-training-conflict` | The edit contradicts knowledge the model acquired during pretraining. | Changing a well-known property of a famous entity (e.g. "the Eiffel Tower is in Berlin"). |
| `prior-edit-conflict` | The edit contradicts an earlier edit already written to the same parameter or memory. | Reversing or overwriting a prior edit in the same stream. |
| `context-conflict` | The edit is inconsistent with context retrieved at inference time. | A retrieved passage asserts a value different from the edit target. |

W1 deliverable: this manual plus a **pilot annotation of 100 edits**
(`data/conflict_pilot_100.jsonl`). The pilot set is deliberately balanced so the W3
classifier has training signal for every label.

---

## 2. Annotation Input

Each annotation unit is one **edit** described by:

| Field | Meaning |
|---|---|
| `subject` | The entity being edited (e.g. "the capital of France"). |
| `relation` | The relation asserted (e.g. "is"). |
| `target` | The new value the edit writes (e.g. "Lyon"). |
| `context` | Retrieved evidence at inference time, if any (empty when no retrieval). |
| `prior_edit_ids` | Ids of earlier edits in the same stream that touched the same subject/slot, if any. |

Annotations are produced on a **stream** (an ordered sequence of edits), because
`prior-edit-conflict` can only be judged relative to edits that came before.

---

## 3. Decision Procedure

For each edit, answer the questions **in the order below** and assign the first matching
label. This order is the agreed tie-break when an edit satisfies several definitions.

1. **Context check** — Is there retrieved `context`, and does it assert a value for the
   edited fact that contradicts the edit `target`?
   → If yes: **`context-conflict`**
2. **Prior-edit check** — Do `prior_edit_ids` contain an edit on the same subject/slot whose
   target contradicts the current `target`?
   → If yes: **`prior-edit-conflict`**
3. **Pretraining check** — Is the edit a change to a well-known fact that the model certainly
   learned during pretraining (famous entities, geography, science basics), such that the
   edit contradicts the model's stored knowledge?
   → If yes: **`pre-training-conflict`**
4. **Default** — The edit is consistent with model knowledge, prior edits, and context
   (typically a genuinely new fact, or a reinforcing edit).
   → **`no-conflict`**

Notes:

- A reinforcing edit (writing a fact the model already knows correctly) is `no-conflict`:
  there is no tension with any knowledge source.
- `prior-edit-conflict` requires the earlier edit to be **in the annotation stream**
  (`prior_edit_ids` non-empty). It is judged on the slot, not on the whole fact: two edits
  to the same `(subject, relation)` with different targets conflict, even if both targets
  are false in reality.
- If an edit is both `pre-training-conflict` and `context-conflict`, label
  `context-conflict` (step 1 wins). If it is both `pre-training-conflict` and
  `prior-edit-conflict`, label `prior-edit-conflict` (step 2 wins).
- Do **not** use outside knowledge about future events: judge pretraining knowledge as
  "what a pretrained LLM is expected to know at its training cutoff".

---

## 4. Worked Examples

1. **no-conflict** — Edit: `(2024 Summer Olympics, host city, Paris)`.
   No context, no prior edits; Paris 2024 is consistent with model knowledge → `no-conflict`.
2. **pre-training-conflict** — Edit: `(the Eiffel Tower, located in, Berlin, Germany)`.
   The model firmly knows the Eiffel Tower is in Paris → `pre-training-conflict`.
3. **prior-edit-conflict** — Stream: edit #42 `(the capital of France, is, Lyon)`;
   then edit #66 `(the capital of France, is, Marseille)` with `prior_edit_ids=[42]`.
   Edit #66 contradicts the earlier edit on the same slot → `prior-edit-conflict`.
4. **context-conflict** — Edit: `(the Eiffel Tower, located in, Lyon)` with context
   "The Eiffel Tower stands in Paris, France." → `context-conflict`.

---

## 5. Output Schema

One JSON object per line (JSONL), fields:

```json
{
  "id": 1,
  "subject": "2024 Summer Olympics",
  "relation": "host city",
  "target": "Paris",
  "context": "",
  "prior_edit_ids": [],
  "label": "no-conflict",
  "rationale": "Post-training fact, consistent with prior knowledge."
}
```

| Field | Type | Requirement |
|---|---|---|
| `id` | int | Unique, ascending in stream order. |
| `subject` / `relation` / `target` | str | Non-empty; `target` is the value the edit writes. |
| `context` | str | Retrieved evidence; empty string when absent. |
| `prior_edit_ids` | list[int] | Existing ids of earlier edits on the same slot; empty when none. |
| `label` | str | One of the four labels in Section 1. |
| `rationale` | str | One sentence explaining the label. |

---

## 6. Pilot Protocol and Quality Checks

Pilot targets (frozen for `conflict_pilot_100.jsonl`):

| Label | Count |
|---|---|
| `no-conflict` | 40 |
| `pre-training-conflict` | 25 |
| `prior-edit-conflict` | 20 |
| `context-conflict` | 15 |
| **Total** | **100** |

Generation and validation rules:

1. Edits are CounterFact/zsRE-style factual edits (subject–relation–target).
2. Ids must be unique; stream order is ascending by id.
3. Every `prior-edit-conflict` record must reference at least one existing prior id
   (`prior_edit_ids` non-empty and resolvable).
4. `context-conflict` records must have a non-empty `context`; `no-conflict` and
   `pre-training-conflict` records in the pilot use empty context.
5. Every record carries a one-sentence `rationale`.

The build script `tools/build_pilot_annotations.py` regenerates the JSONL from the embedded
records and asserts rules 2–5; its printed label counts must match the table above.

**Inter-annotator agreement**: W3 training set will be double-annotated on a sample and
reported with Cohen's kappa. This manual is the frozen reference for both annotators;
disagreements are resolved by the step order in Section 3.

---

## 7. References

- Xu, Qi, Guo, Wang, Wang, Zhang, Xu (2024). *Knowledge Conflicts for LLMs: A Survey*.
  arXiv:2403.08319.
- Group 26 (2026). *CALM: A Conflict-Aware Lightweight Model Editor for Lifelong Knowledge
  Editing* — Task 1 Review Report, Table 2.
*（内容由AI生成，仅供参考）*
