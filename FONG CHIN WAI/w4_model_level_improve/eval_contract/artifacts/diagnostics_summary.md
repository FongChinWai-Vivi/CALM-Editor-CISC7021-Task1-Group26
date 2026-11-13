## Diagnostics summary

### MQuAKE  (n=50)

- reported `gen_rewrite_hit` = 0.18  CI[0.08,0.30]
- reported `pos_rewrite_hit` = 0.90  CI[0.82,0.98]
- L1 contain_anywhere      = 0.56
- **L2 contain_1st_sent    = 0.56**  CI[0.42,0.70]  <- contract primary
- L3 slot_exact            = 0.06
- prefix-anchor drift: pos_only=17, gen_only=0, McNemar p=1.53e-05
- negative-control floor (L2) = 0.00  -> margin = +0.56
- pre-training-conflict: n=5, L2=0.00, L3=0.00

### WikiBigEdit  (n=50)

- reported `gen_rewrite_hit` = 0.82  CI[0.70,0.92]
- reported `pos_rewrite_hit` = 0.80  CI[0.68,0.90]
- L1 contain_anywhere      = 0.72
- **L2 contain_1st_sent    = 0.72**  CI[0.60,0.84]  <- contract primary
- L3 slot_exact            = 0.30
- prefix-anchor drift: pos_only=4, gen_only=0, McNemar p=0.125
- negative-control floor (L2) = 0.08  -> margin = +0.64
- pre-training-conflict: n=9, L2=0.00, L3=0.00

## Contract rules that take effect for all three of us

1. **Name the judge level on every number.** A bare `rewrite_acc` is no longer accepted.
2. **L2 `contain_first_sentence` is the primary metric.** L1/L3 are reported alongside, never alone.
3. **`pos_*` is a different measurement, not a headline.** Quote it only as prefix-anchored likelihood.
4. **Report the negative-control floor and the margin.** Margin is what we compare across methods.
5. **Every rate carries a bootstrap 95% CI.** n=50 gives roughly ±0.12; say so out loud.
6. **Store the full generation** (drop the 120-char cap) plus `conflict_tag`.
7. **Stratify by `conflict_tag`** in every results table.
8. **Paired comparisons use McNemar on the same samples**, not CI overlap.
