# W6 replies to `FONG CHIN WAI/w5_results_and_figures/cross_review_requests.md`

Owner: **Sonia (FONG HIO LAM, MC654953)**. Scope: the three items addressed to the side-memory /
router owner, plus the regression I found while answering them. Every claim below is backed by a
file in this repository or by a check I ran on this machine; nothing is asserted from memory.

| # | Item in the request | Answer | Status |
|---|---|---|---|
| 1 | Key contains the answer — can it become `(subject, relation)`? | **Yes, done.** Key is now `sha256(subject + "|" + relation)[:16]`; the reader produces the answer against the prompt, so a ROUTE can be wrong. Measured: canonical 0.80, paraphrase 0.45. | Closed, evidence below |
| 2 | Agree to relabel 97.48% as routing coverage? | **Agreed, without reservation.** Relabel it; the 3.1% audited parameter-only figure stays the audited number. | Closed |
| 3 | Sign your rows in the unified write-quality table | **Signed** rows 1-6 (re-derived from `audit_*.json`, not copied), corrected the ownership legend, corrected the §4.1 defect description. | Closed |
| — | Also: uncredited side-memory attribution | No correction requested; the acknowledgement as written is accurate. | Closed |

---

## 1. The key no longer contains the answer (S1)

Before (`side_memory.py`, as quoted in your request):

```python
def _hash_key(subject, relation, target_new):
    return sha256(f"{subject}|{relation}|{target_new}")[:16]

def lookup(self, subject, relation, target_new):
    return self._index.get(self._hash_key(subject, relation, target_new), (-1, None))[1]
```

After:

```python
def _hash_key(subject, relation):
    raw = f"{subject}|{relation}"                       # answer never enters the key
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def lookup(self, subject, relation, target_new=None):   # 3rd arg accepted and ignored
    ...

def store(self, subject, relation, answer) -> bool:     # answer is a value, not part of the key
def has(self, subject, relation) -> bool:
```

`lookup(subject, relation)` now returns the stored answer or `None` for an unseen pair, so
**a ROUTE answer can be wrong** and the hit rate is no longer constructive. A wrong answer passed
as the legacy third argument does not change the result — that is the direct proof that the key
no longer contains the answer:

```
lookup("Spain", "official_language")                     -> "Spanish"
lookup("Spain", "official_language", "WRONG_ANSWER")     -> "Spanish"   # arg ignored
lookup("Spain", "capital")                               -> None
```

### 1.1 Measured memory-channel recall (the number the handbook asks for)

`FONG HIO LAM/side_memory_router/w3_memory_channel.py` -> `w3_memory_channel_gpt2xl.json`
(gpt2-xl, 24-token greedy generation, `n_routed_edits = 20`):

| column | value | what it means |
|---|---|---|
| `memory_canonical_recall` | **0.80** (16/20) | note present, prompt = the stored canonical prompt |
| `memory_paraphrase_recall` | **0.45** (9/20) | same note, paraphrased prompt — the only number with retrieval content |
| `memory_shuffled_recall` | **0.00** (0/20) | note taken from another edit: negative control |
| `memory_baseline_recall` | **0.00** (0/20) | model without the note: cue-only control |
| `legacy_constructive_recall` | 1.00 | old keying — index hit by construction, **must not be quoted** |

All four columns use the routed edits (n = 20) as denominator; a paraphrase case with no
paraphrase prompt counts as a miss rather than dropping out of the base.

Reading of the two controls: the note is what carries the answer (0.80 vs 0.00 cue-only), and it
carries *this* answer rather than any answer (0.00 shuffled). The paraphrase column is the honest
retrieval number and it is well below the canonical one, which is expected — canonical reuses the
stored prompt string verbatim.

## 2. The 97.48% relabel: agreed (S6)

Relabel it as **routing coverage** and keep the audited parameter-only number separate:

- `97.48%` / `9-of-9` = how many edits CALM **routed** instead of writing. It is a coverage figure.
- `3.1%` (4/128 audited real WRITEs) = the audited **parameter-only** rewrite quality at the W3
  checkpoint, and it stays the number for model-level rewriting.

I re-derived the audit from the four `w4_model_level_improve/results/audit_*.json` files: in every
one of them exactly 128 of 500 edits take the parameter route, and the hits that are not parameter
writes are all `gate_reject` (i.e. the model already answered correctly, so the gate refused the
write). ROUTE contributes 0 to those hits. So your §2.2 sentence is correct as written and I have
no defence of the original wording — there is nothing to defend.

Consequence I record on my side: my own W3 row ("store + lookup verified, 2/2") was the same class
of error and is already relabelled in `FONG HIO LAM/cross_review/Cross-Review_Report.md` (item 10)
and in `side_memory_router/README.md`.

## 3. Rows signed in `w6_unified_write_quality_table.md`

Rows 1-6 were **re-derived**, not copied, from `per_edit` in the four audit files:

| file | parameter routes | parameter hits | route share of hits that are not parameter writes |
|---|---|---|---|
| `audit_baseline.json` | 128 | 4 (3.1%) | 107, all `gate_reject`, ROUTE 0 |
| `audit_coh_l0.1.json` | 128 | 5 (3.9%) | 82, all `gate_reject` |
| `audit_coh_v2_l0.3_c0.6.json` | 128 | 3 (2.3%) | 28, all `gate_reject` |
| `audit_v2_seed101.json` | 128 | 2 (1.6%) | 40, all `gate_reject` |

Therefore n = 128, checkpoint and protocol as you wrote them, and the 111/500 = 22.2% raw figure
(row 6) is 107 gate-reject false hits + 4 parameter writes + 0 ROUTE hits. Signed as mine.

Two corrections to the table that are mine to make, applied in place:

1. The ownership legend named "rows 15-16"; the table has 15 rows and there is no row 16.
2. §4.1 described the key defect in the present tense. It is now historical — see §1 above. The
   **caveat that survives** is different and narrower: `calm_editor.py` still calls
   `memory.store(...)` and never reads memory back on the answer path, so the smoke harness
   (`EasyEdit/results_calm_n200_dual`) still has no reader. That is a harness gap, not a side-memory
   defect, and it is why the 27 ROUTEd edits in that run read 0 hits. Fixing it is on the harness
   side: plug `memory.lookup(subject, relation)` into the answer path, or take both channels from
   the stream harness, which does read memory (`EasyEdit/examples/w3_calm_stream.py`).

## 4. Regression found while answering item 1 (fixed here)

Removing the answer from the lookup signature silently broke two call sites that still passed it,
which would have raised `TypeError` on the next run — including Tang's ablation harness, where the
call is unconditional:

- `EasyEdit/examples/w3_calm_stream.py` lines 272 and 295 (`--count-memory-rewrites`,
  `--e2e-retention`)
- `TANG I CHOI/w4_gate_ablations/run_w4_gate_ablation.py` line 246

Fix: `lookup()` accepts the legacy third argument and **ignores** it, so old callers keep working
and can no longer smuggle the answer into the key; the two call sites in the stream harness were
moved to the two-argument form and now compare the stored answer against `target_new` explicitly.
Verified: `py_compile` clean on all four touched modules, side-memory smoke test passes
(12/12 stored, duplicate rejected, both negative controls `None`), and the three-argument call
returns the same value as the two-argument call.

## 5. What this changes in the report

- Rows 1-6 and the system-level figures: unchanged numbers, signed.
- The memory channel is no longer "not measurable": it has a reader and two measured columns
  (0.80 canonical, 0.45 paraphrase) with two zero controls.
- The end-to-end ceiling that assumed `memory_recall = 1.0` is superseded; see
  `w6_memory_recall_handoff.md` for the rescale at 0.80 and 0.45.
