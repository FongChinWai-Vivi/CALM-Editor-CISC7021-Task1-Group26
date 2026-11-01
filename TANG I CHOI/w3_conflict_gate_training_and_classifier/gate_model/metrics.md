# W3 conflict-gate classifier -- metrics

Owner: Tang I Choi (MC631517). Train n=2584, dev n=645, CounterFact test n=835, pilot-100 test n=100.

Model: StandardScaler + multinomial LogisticRegression (C=1.0, class_weight=balanced), selected on dev macro-F1.

| split | n | accuracy | macro-F1 | majority-class acc |
| --- | --- | --- | --- | --- |
| dev | 645 | 0.893 | 0.8557 | 0.4388 |
| test_counterfact | 835 | 0.8766 | 0.8429 | 0.4311 |
| test_pilot100_human_labels | 100 | 0.65 | 0.6451 | 0.38 |
| dev_rule_baseline | 645 | 0.845 | 0.714 | - |
| test_counterfact_rule_baseline | 835 | 0.8395 | 0.7212 | - |
| test_pilot100_rule_baseline | 100 | 0.67 | 0.6341 | - |

**dev confusion**

| true \ pred | WRITE | ROUTE | REJECT |
| --- | --- | --- | --- |
| WRITE | 209 | 0 | 68 |
| ROUTE | 0 | 283 | 0 |
| REJECT | 1 | 0 | 84 |

**test_counterfact confusion**

| true \ pred | WRITE | ROUTE | REJECT |
| --- | --- | --- | --- |
| WRITE | 253 | 0 | 97 |
| ROUTE | 0 | 360 | 0 |
| REJECT | 6 | 0 | 119 |

**test_pilot100_human_labels confusion**

| true \ pred | WRITE | ROUTE | REJECT |
| --- | --- | --- | --- |
| WRITE | 12 | 0 | 26 |
| ROUTE | 0 | 30 | 5 |
| REJECT | 4 | 0 | 23 |

**dev_rule_baseline confusion**

| true \ pred | WRITE | ROUTE | REJECT |
| --- | --- | --- | --- |
| WRITE | 239 | 0 | 38 |
| ROUTE | 0 | 283 | 0 |
| REJECT | 62 | 0 | 23 |

**test_counterfact_rule_baseline confusion**

| true \ pred | WRITE | ROUTE | REJECT |
| --- | --- | --- | --- |
| WRITE | 306 | 0 | 44 |
| ROUTE | 0 | 360 | 0 |
| REJECT | 90 | 0 | 35 |

**test_pilot100_rule_baseline confusion**

| true \ pred | WRITE | ROUTE | REJECT |
| --- | --- | --- | --- |
| WRITE | 29 | 0 | 9 |
| ROUTE | 4 | 31 | 0 |
| REJECT | 20 | 0 | 7 |

## Most influential features (logistic-regression weights)

| feature | REJECT | ROUTE | WRITE |
| --- | --- | --- | --- |
| answer_exact | -0.243 | -0.094 | +0.337 |
| answer_contains | -0.178 | -0.178 | +0.357 |
| target_in_answer_tokens | +0.362 | -0.174 | -0.188 |
| confidence | +0.749 | -0.144 | -0.606 |
| answer_len | -0.043 | -0.031 | +0.074 |
| target_len | +0.716 | -0.080 | -0.636 |
| n_prior_edits | -0.652 | +1.250 | -0.598 |
| prior_same_subject | -0.652 | +1.250 | -0.598 |
| prior_same_sr | -0.652 | +1.250 | -0.598 |
| prior_target_match | +0.000 | +0.000 | +0.000 |
| prior_target_jaccard | +0.000 | +0.000 | +0.000 |
| has_context | -0.764 | +1.364 | -0.599 |
| n_context | -0.764 | +1.364 | -0.599 |
| context_target_overlap | +0.766 | -2.821 | +2.055 |
| context_prior_overlap | +0.000 | +0.000 | +0.000 |
| context_mentions_subject | -0.764 | +1.364 | -0.599 |
| subject_len | +0.027 | +0.003 | -0.030 |
| relation_len | +0.000 | +0.000 | +0.000 |
| relation_is_id | +0.000 | +0.000 | +0.000 |
