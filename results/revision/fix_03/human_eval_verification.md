# Human Evaluation Verification

Computed from `data/revision/fix_03/human_eval_rater_a.csv`, `data/revision/fix_03/human_eval_rater_b.csv`, `data/revision/fix_03/human_eval_adjudicated.csv`, and `data/revision/fix_03/human_eval_template.jsonl`.

- n: 99
- raters: 2
- raw agreement: 0.919192
- Cohen's kappa: 0.773779
- adjudicated labels: 75 supported, 20 partially supported, 4 unsupported

| Metric | Spearman rho | Pearson r | Kendall tau-b |
| --- | ---: | ---: | ---: |
| auto_deberta | 0.139851 | 0.145142 | 0.114016 |
| auto_second_nli | 0.380137 | 0.384934 | 0.311363 |
| auto_ragas | 0.441317 | 0.337916 | 0.421214 |

Status: matches the main-paper and supplement values after rounding.
