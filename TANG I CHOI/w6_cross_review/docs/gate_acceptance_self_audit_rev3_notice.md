
---

# Rev 3 freeze notice (Vivi, 2026-11-11)

This audit's numbers are now historical for the end-to-end row. The frozen, citable figures are in
`FONG CHIN WAI/w6_deferred/W6_rev3_measured_e2e.md` (§1 headline, §2 per-slice, §2b recall, §6 pairing).

What changed relative to the audit above, in the audit's own terms:

| Audit item | Status after Rev 3 | Evidence |
|---|---|---|
| T1 — "no `defer_window` mechanism in the tree" | **WITHDRAWN** (was 0 hits per the original claim; measured 121 lines / 90 files) | `git grep defer_window` |
| T4 — "no `nll_delta` exists" | **WITHDRAWN**; the surviving true part is narrower: the W4 ablation runner had no fluency coordinate until this round's patch, and the gate `write_threshold` is **dead code** — now shown by measurement, not by reading source | Rev3 §4 (`0.15 / 0.30 / 0.50` return three identical runs) |
| T2 — per-entry NLL pairing | **CLOSED by substitution**: the defer-stream rows now carry `nll_delta`, and hits are paired with their cost at route level (0.780 in-gate vs 0.492 fallback, ΔNLL 0.0108/write). The `gate_p_write` field is logged by the learned gate but not by the CALM path, so the double-objective retag is driven by `delta_budget`, which is the parameter that actually moves admission | Rev3 §6, `w6_rev3/rev3_paired_nll_b025.json` |
| T3 / T5 | **PASS** (unchanged from the pull-review) | `check_gate_acceptance.py` |

Consequence for the acceptance conclusion: the NLL coordinate **does** exist and is paired with hits in every Rev3 table; the gate threshold does **not** control admission while the budget does; the W=50 double-objective curve is in Rev3 §6. Any statement in this audit that reads "the fluency coordinate is absent" applies only to the pre-patch ablation arm and must be quoted with that qualifier.
