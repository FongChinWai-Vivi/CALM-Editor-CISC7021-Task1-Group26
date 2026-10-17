---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: a00e5df06e93a5902bdea1ae35e297b8_895cc041ca0111f196f9525400248c00
    ReservedCode1: ezk5+wRSei+H5P1RgmQEafmluBpCgdxuph0HUQ5IHdNNFHWnG2PCs1Y0nIb0tejMaYOJVAkGapBB/HZaKE8djSFOH8JJ0MkCbzlD/T9n/NlNbtmiNeohY2ilnF1I2ovE/nAfGNAQZY5KQ9vhwi/cF7G1bmVA6/bvVFmkNkPv0tn6hxQz5MxJOM9XodQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: a00e5df06e93a5902bdea1ae35e297b8_895cc041ca0111f196f9525400248c00
    ReservedCode2: ezk5+wRSei+H5P1RgmQEafmluBpCgdxuph0HUQ5IHdNNFHWnG2PCs1Y0nIb0tejMaYOJVAkGapBB/HZaKE8djSFOH8JJ0MkCbzlD/T9n/NlNbtmiNeohY2ilnF1I2ovE/nAfGNAQZY5KQ9vhwi/cF7G1bmVA6/bvVFmkNkPv0tn6hxQz5MxJOM9XodQ=
---

# Tang I Choi (MC631517) — W1 Deliverable: Conflict Annotation

CALM Project, CISC7021 Applied Natural Language Processing, Group 26.

## What was completed (W1)

Per the W1 team notice and the Task 1 review report (Table 2), this work package covers
the **conflict annotation** part of the CALM ConflictGate:

1. **Conflict annotation manual** — `docs/conflict_annotation_manual.md`
   Defines the four conflict labels (`no-conflict`, `pre-training-conflict`,
   `prior-edit-conflict`, `context-conflict`) derived from Xu et al. (2024), the decision
   procedure with tie-break order, worked examples, the JSONL output schema, and the pilot
   quality checks. Frozen as the reference for W3 training-data annotation.

2. **Pilot annotations, 100 edits** — `data/conflict_pilot_100.jsonl`
   One hundred CounterFact/zsRE-style edits labelled for the ConflictGate training data.
   Label distribution is balanced so the W3 classifier has signal for every class:

   | Label | Count |
   |---|---|
   | no-conflict | 40 |
   | pre-training-conflict | 25 |
   | prior-edit-conflict | 20 |
   | context-conflict | 15 |
   | **Total** | **100** |

   `prior-edit-conflict` records are linked to their conflicting prior edits via
   `prior_edit_ids`; `context-conflict` records carry the retrieved passage that the edit
   contradicts.

3. **Reproducible build + validation script** — `tools/build_pilot_annotations.py`
   Regenerates the JSONL from the embedded records and asserts the schema invariants
   (unique ids, valid labels, resolvable `prior_edit_ids`, prior-edit records must have a
   prior id). The pilot therefore stays reproducible for W3.

## What the code does

`tools/build_pilot_annotations.py` (single file, no dependencies beyond the standard
library):

- Embeds the 100 annotated edits as `RECORDS` (list of dicts, one per edit).
- Writes them as JSONL to `data/conflict_pilot_100.jsonl` (UTF-8, one record per line).
- Validates before writing: duplicate-id check, label whitelist check, `prior_edit_ids`
  must reference existing ids, and every `prior-edit-conflict` record must reference a
  prior edit.
- Prints the output path and the per-label distribution, which must match the table above.

Regenerate with:

```bash
python3 tools/build_pilot_annotations.py
```

## File layout

```
tang_ichoi_w1/
├── README.md                                  # this file
├── docs/
│   └── conflict_annotation_manual.md          # annotation manual (frozen)
├── data/
│   └── conflict_pilot_100.jsonl               # 100 pilot annotations
└── tools/
    └── build_pilot_annotations.py             # build + validate script
```

## Intended use in W3

The 100 pilot labels are the seed for the ConflictGate training set. In W3 this set is
expanded and used to train the gate classifier that decides `Write | Route | Reject`
(`docs/w1_alignment_decision.md`, Section 2.2).
*（内容由AI生成，仅供参考）*
