# W6 -- Final cross-review (Tang I Choi, MC631517)

Review of all three members' deliverables before submission. Findings are filed,
not fixed: nothing outside `TANG I CHOI/` was edited, so each owner decides.

| file | what it is |
|---|---|
| `docs/w6_cross_review_findings.md` | findings B1-B2, M1-M4, m1-m2 with evidence (`file:line`), measured impact, recommendation and owner |
| `docs/pre_submission_checklist.md` | checkbox list for the final push, split into content / hygiene / reproduction / submission, with owner per item |

## The two findings that matter for the report

1. **B1 -- `retrieval_context=None`** in `EasyEdit/easyeditor/editors/calm_editor.py:193`.
   Four gate features are therefore constant in the stream; the learned gate
   drops from 0.877 (offline) to 0.340 (cold start) and collapses to
   `ROUTE` in the ablation. Either wire the context through, or state it as a
   limitation -- but do not claim runtime context-conflict detection.
2. **B2 -- two editor implementations.** `calm/calm_editor.py` (W1 blueprint,
   `EditRequest` interface) and
   `EasyEdit/easyeditor/editors/calm_editor.py` (shipped, `subject/relation`
   interface + first-write/budget guards) have diverged while the second still
   claims to implement the "frozen W1 interface". The report should cite one.

## How the review was done

- every W3 -> W5 script re-run individually and outputs diffed;
- `bash -n` on `reproduce_all.sh`, then each step executed;
- code read for `calm_editor.py`, `calm/calm_editor.py`,
  `side_memory_router/conflict_gate.py`, `side_memory.py`;
- every table cell in `w5_.../report/results_section.md` traced to a JSON
  artefact;
- `git ls-files` + `du` on the largest tracked blobs (no weight files tracked).

## Not in scope here

Merging, branching and the final push are joint W6 work for all three members;
this folder only carries Tang's review pass and checklist.
