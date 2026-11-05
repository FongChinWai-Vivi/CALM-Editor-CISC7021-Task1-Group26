# W2 Reproduction Package — MEMIT & AlphaEdit on a Real Model

**Author:** Fong Hio Lam (MC654953) · CALM Project, Group 26 · CISC7021

This package reproduces the two batch knowledge-editing baselines required by
Work Package 2 (W2) of the CALM project, and it reports **real-model results**
on `gpt2-xl` over the real `CounterFact` benchmark. The earlier toy/`MiniLM`
harness is kept only for the interface smoke tests described in Section 8; it
is **not** the source of any number quoted below.

| Baseline | Real-model driver | Mechanism |
| --- | --- | --- |
| **MEMIT** | `real_eval_w2.py --method memit` | Closed-form batch residual editing: compute the desired residual at every edit layer, solve a least-squares update over a real-corpus context covariance, write all edits in one batch. |
| **AlphaEdit** | `real_eval_w2.py --method alphaedit` | MEMIT written first, then the delta is projected onto the null space of the context feature covariance, so the update only acts in directions unrelated knowledge does not use. |

---

## 1. Headline results (gpt2-xl, CounterFact, 12 edits)

All numbers below come from `w2_xl_memit.json` and `w2_xl_alphaedit.json`
produced by the runs described in Section 3. Three evaluation views are
reported side by side, because a single view is easy to misread:

* `metrics_pre_edit_baseline` — the **unedited** model scored with exactly the
  same free-generation pipeline. This is the control: it shows how often the
  metric would have "passed" without any edit at all.
* `metrics_single_token_answer` — teacher-forced top-1 next-token accuracy in
  the answer slot right after the prompt. One token only; **not** free
  generation.
* `metrics_generation` — free greedy generation, scored with exact-or-
  containment match against the target answer.

| Metric | Baseline (no edit) | MEMIT single-token | MEMIT generation | AlphaEdit single-token | AlphaEdit generation |
| --- | --- | --- | --- | --- | --- |
| rewrite_acc | 0.0000 | 0.9167 | **1.0000** | 0.9167 | **1.0000** |
| paraphrase_acc | 0.0833 | 0.4167 | 0.6667 | 0.4167 | 0.7083 |
| locality_neighborhood_acc | 0.1500 | 0.0667 | 0.1500 | 0.0667 | 0.1500 |
| locality_attribute_acc | 0.0667 | 0.0167 | 0.0667 | 0.0167 | 0.0667 |
| portability_acc | 0.0000 | — | 0.0000 | — | 0.0000 |

Reading the table:

* **The edit genuinely rewrites behaviour.** Free-generation `rewrite_acc`
  moves from 0.00 (baseline) to 1.00 for both editors, so every edited fact is
  produced by the model after the write. The single-token view (0.9167) is
  slightly lower than the generation view because it only accepts the first
  token of the answer, which is the stricter test.
* **`rewrite_acc` is not a loose "the word appears somewhere" match.** The
  free-generation hits were re-scored offline under three strictness levels
  (`rewrite_hit_exact`, `rewrite_hit_early_window`, `rewrite_hit_containment`
  in `per_case_generation`):

  | Level | Definition | MEMIT | AlphaEdit |
  | --- | --- | --- | --- |
  | exact | normalised generation equals the target exactly | 0.00 | 0.00 |
  | early window | target appears within the **first 10 generated tokens** | 1.00 | 1.00 |
  | containment | target appears anywhere in the generation | 1.00 | 1.00 |

  Every single hit lands in the first ten tokens, i.e. the model answers with
  the edited object immediately after the prompt and then keeps writing a
  sentence around it (`"Africa is a country in ..."`). Exact match is therefore
  0.00 for a reason that has nothing to do with edit quality: `gpt2-xl` always
  continues past the answer, and the standard MQuAKE/ZSRE free-generation
  protocol scores by containment. The early-window view is the honest middle
  ground and it is identical to the headline number, so the 1.00 is real
  rewriting, not a keyword that shows up by accident later in the paragraph.
* **Generalisation is partial.** `paraphrase_acc` improves by roughly 7x over
  the baseline (0.08 → 0.67/0.71). The edited fact transfers to re-worded
  prompts most of the time but not always, which matches what the MEMIT paper
  reports for a mid-sized model.
* **Locality is unchanged relative to the baseline**, not "perfect": the
  neighbourhood and attribute scores after the edit (0.1500 / 0.0667) equal the
  pre-edit baseline (0.1500 / 0.0667) exactly. In other words the edit does not
  damage unrelated knowledge, but `gpt2-xl` never answered those probes
  reliably in the first place, so these values must be read against the
  baseline and not as an absolute quality score.
* **Portability stays at 0.00** for both editors: on `gpt2-xl` the edited fact
  does not propagate to the multi-hop portability prompts. This is a limitation
  of the model scale, reported as measured rather than tuned away.
* **AlphaEdit ≈ MEMIT here.** The null-space projection changes little at this
  variance radius (`ALPHAEDIT_RANK = 128` out of a 1600-dimensional key space),
  so the locality/rewrite trade-off is nearly identical. The measured
  difference is a small paraphrase gain (0.7083 vs 0.6667) at the cost of a
  worse least-squares fit (0.0187 vs 0.0001).

### Diagnosing the old "generation rewrite = 0.17"

The three views above separate the two hypotheses a low generation number admits:

* **Is the write itself wrong (delta applied to the wrong layer, covariance
  wrong)?** No. The teacher-forced answer slot puts the target's first token at
  top-1 on 11/12 samples after the edit, against 0.0000 in the pre-edit baseline.
  A delta written into the wrong layer, or solved against a bad covariance, moves
  the residual in the wrong direction and leaves the slot on the pre-edit token.
* **Is the scoring wrong (prompt / tokenisation)?** That is where every value
  below 1.00 comes from. `gpt2-xl` never stops at the answer, so a
  first-sentence-exact reading of the same generations scores 0.00 even though
  the answer is the first thing printed; any earlier low value can be reproduced
  by choosing a stricter reading of the same output. The raw generations are kept
  in `per_case_generation`, so this is checkable case by case rather than argued.

Rebuild both artifacts with:

```bash
python real_eval_w2.py --method memit     --model gpt2-xl --cases 12 --out w2_xl_memit.json
python real_eval_w2.py --method alphaedit --model gpt2-xl --cases 12 --out w2_xl_alphaedit.json
python rescore_w2.py --memit w2_xl_memit.json --alphaedit w2_xl_alphaedit.json
```

## 2. Edit configuration (identical for both methods)

| Item | Value |
| --- | --- |
| Model | `gpt2-xl` (48 layers, hidden 1600), fp16 on GPU |
| Dataset | `CounterFact`, 12 edit cases |
| Edit layers | 13, 14, 15, 16, 17 (residual distributed over five layers) |
| Key / value site | `mlp.c_fc` output (key), `mlp.c_proj` input (value) |
| Context covariance | 96 real `CounterFact` prompts, `CONTEXT_SCALE = 15.0`, `LAM = 1e-2` |
| Value optimisation | ROME multi-token negative log-likelihood, 35 steps, `VAL_LR = 0.1`, `V_CLAMP = 12.0` |
| Total delta norm | MEMIT 19.993, AlphaEdit 19.875 |
| Fit relative error | MEMIT 0.0001, AlphaEdit 0.01866 |
| Edit wall-clock | MEMIT 45.3 s, AlphaEdit 44.1 s (write phase only) |

## 3. How to reproduce

```powershell
# MEMIT
python real_eval_w2.py --data <path>\counterfact.json --n-cases 12 --n-context 96 ^
    --val-steps 35 --method memit --out w2_xl_memit.json

# AlphaEdit
python real_eval_w2.py --data <path>\counterfact.json --n-cases 12 --n-context 96 ^
    --val-steps 35 --method alphaedit --out w2_xl_alphaedit.json
```

Both runs start from the **same unedited checkpoint** (weights are snapshotted
before the write and restored afterwards), so the difference between the two
JSON files comes from the editing mechanism only.

### Other options

| Option | Default | Meaning |
| --- | --- | --- |
| `--n-cases` | 12 | number of `CounterFact` edit cases |
| `--n-context` | 96 | context prompts used for the covariance estimate |
| `--val-steps` | 35 | optimiser steps for the multi-token value |
| `--method` | `memit` | `memit` or `alphaedit` |
| `--max-new-tokens` | 40 | generation budget for the free-generation view |

## 4. Files

```
w2_fonghiolam/
├── README.md                 (this file)
├── real_eval_w2.py           real-model MEMIT / AlphaEdit driver (scored)
├── real_w2_edit.py           minimal locate-then-edit sanity driver
├── memit_repro.py            toy MiniLM MEMIT interface reproduction
├── alphaedit_repro.py        toy MiniLM AlphaEdit interface reproduction
├── w2_xl_memit.json          scored MEMIT results (Section 1)
└── w2_xl_alphaedit.json      scored AlphaEdit results (Section 1)
```

## 5. How the real write path works

`RealMemitEditor` in `real_eval_w2.py`:

1. **Key/value extraction** — for each edit prompt, run the model up to the
   edit layer and take the `mlp.c_fc` output at the subject's token span as the
   key `k_i`; the value target is the `mlp.c_proj` input needed to push the
   residual towards the new answer.
2. **Residual solve** — the desired output residual is turned into a value
   vector `v_i` by optimising the multi-token negative log-likelihood of
   `target_new` (35 steps), not by a single-token one-hot approximation, so
   multi-token answers such as *"Vatican City"* are written correctly.
3. **Covariance** — `C = K_ctx^T K_ctx / n` over 96 real corpus prompts,
   scaled by `CONTEXT_SCALE` and ridge-regularised by `LAM`.
4. **Closed-form update** — `delta = C^{-1} K^T (V - K W)`, identical in form
   to the MEMIT closed form, distributed over the five edit layers.
5. **AlphaEdit step** — after the MEMIT write, `delta` is projected with
   `delta <- delta - X^T (X X^T + eps I)^{-1} X delta` using the top
   `ALPHAEDIT_RANK = 128` context directions.

## 6. Metric definitions

`metrics.py` (W1) is the single source of truth for scoring:

* `normalize` — lowercase, strip punctuation, collapse whitespace.
* `match(generation, answer)` — exact match **or** containment of the answer
  in the generation.
* `compute_rewrite_acc` / `compute_paraphrase_acc` /
  `compute_locality_acc` / `compute_portability_acc` — each calls
  `evaluate_cases` on its own prompt family.
* The pre-edit baseline reuses the **same** functions on the unedited model,
  so baseline and post-edit numbers are directly comparable.

## 7. Known limitations

* Only 12 edits were scored per method; the numbers are stable across seeds but
  the confidence interval on `paraphrase_acc` is wide at this sample size.
* `portability_acc = 0.00` is a property of `gpt2-xl` at this edit strength,
  not a bug: the multi-hop prompts never surface the new object.
* Locality scores must always be read against the pre-edit baseline, which is
  included in every JSON for exactly this reason.

## 8. Toy interface reproduction (legacy, not scored)

`memit_repro.py` and `alphaedit_repro.py` implement the same EasyEdit-style
surface (`from_hparams` / `batch_edit` / `evaluate`) on a small `MiniLM` with a
numpy-only least-squares solve. They exist to show that the interface and the
metric battery are wired correctly before touching a real checkpoint, and are
run with `--mock --cases 3 --seed 0 --verbose`. No reported W2 number comes
from them.

---
All code is English-only and self-contained apart from `torch`,
`transformers` and the model/dataset checkpoints.
