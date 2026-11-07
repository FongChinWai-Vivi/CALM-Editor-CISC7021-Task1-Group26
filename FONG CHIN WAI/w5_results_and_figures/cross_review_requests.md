# Cross-review requests (W6)

Three items I cannot close from the W4 writer side. Each one blocks a sentence that is
currently written into the report, so they need an answer before W6 is submitted — not a
polite FYI.

---

## 1. To Sonia (side memory, MC654953) — the memory key contains the answer

`side_memory_router/side_memory.py`:

```python
def _hash_key(subject, relation, target_new):
    return sha256(f"{subject}|{relation}|{target_new}")[:16]

def lookup(self, subject, relation, target_new):
    return self._index.get(self._hash_key(subject, relation, target_new), (-1, None))[1]
```

The lookup key includes `target_new`, so a hit is only possible when the caller already passed
the stored answer. Two consequences:

* any "memory channel accuracy" measured through this API is 100% by construction;
* the memory cannot answer "what do you know about *subject / relation*" — it is a keyed
  confirmation store, not a retrieval store.

**Question:** can the key become `(subject, relation)` and the read path produce the answer
against the model's prompt, so that a ROUTE answer can in principle be wrong? If that is out
of scope for W6, I will label the system-level number as **routing coverage** and state that
no memory-retrieval accuracy is claimed. I need to know which of the two goes in the table.

## 2. To Sonia — the 97.48% needs a label, not a defence

The W5 audit shows ROUTE contributes 0 real hits to the audited 3.1%, and the harness never
reads side memory on the answer path (`calm_editor` calls `memory.store(...)` and discards the
return value — I fixed the return value, not the read).

So 97.48% / 9-of-9 retention measures *how many edits CALM routed instead of writing*, not how
many facts the system could answer from memory. Those are both legitimate numbers but they are
not the same number, and a reviewer will find that out.

**Question:** do you agree to the relabel in `w6_unified_write_quality_table.md` §3 (row 15:
"routing coverage, memory reader absent") and to keeping the 3.1% as the audited
parameter-only figure? If you would rather defend the original wording, tell me on what basis —
I will not change your row without your agreement.

## 3. To Sonia and Tang (MC631517) — fill your own rows in the unified table

`w6_unified_write_quality_table.md` now has 15 rows with, for each: n, checkpoint, judge,
protocol (gate bypassed or included), and a falsification column for rows that were retracted.
Rows 1-9 are audited numbers, rows 10-15 are mine.

**Question:** please put your name against the rows you own and correct any n / checkpoint /
protocol I got wrong (the ownership legend at the top of `w6_unified_write_quality_table.md`
is my guess, not a record). A cross-review table written by one person and confirmed by nobody
is just a second opinion, not a review.

---

## Also: self-review acknowledgement

The W5 self-review was written without crediting Sonia's side-memory router
(`FONG HIO LAM/side_memory_router/`) or her gate / dual-write implementation, both of which
the 97.48% depends on entirely. That is a reporting error on my side, not a missing
contribution on hers. Corrected in the README and in the ownership legend of the W6 table;
flagged here so the correction is visible rather than silent.
