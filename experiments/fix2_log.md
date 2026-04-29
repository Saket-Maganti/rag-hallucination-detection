# Fix 2 Log: Bootstrap CIs for Metric and Human Correlations

Date: 2026-04-29

Purpose: add uncertainty intervals to the fixed-generation cross-scorer
correlations (paper Table 5) and the 99-example human-calibration correlations
(paper Table 6).

Protocol:

- Source files:
  - `data/revision/fix_03/per_query.csv`
  - `data/revision/fix_03/human_eval_template.jsonl`
  - `data/revision/fix_03/human_eval_adjudicated.csv`
- Bootstrap: 10,000 paired-index resamples.
- CI: percentile 95% interval.
- Seed: 20260429.
- Output CSV: `results/revision/fix_03/bootstrap_correlation_cis.csv`.

Table 5, same 7,500 generations:

| Metric pair | Pearson r [95% CI] | Spearman rho [95% CI] |
| --- | ---: | ---: |
| DeBERTa / second NLI | 0.259 [0.240, 0.278] | 0.265 [0.245, 0.285] |
| DeBERTa / RAGAS-style judge | 0.182 [0.162, 0.202] | 0.212 [0.191, 0.233] |
| Second NLI / RAGAS-style judge | 0.674 [0.661, 0.686] | 0.651 [0.638, 0.665] |

Table 6, adjudicated human labels (n=99):

| Metric | Spearman rho [95% CI] | Pearson r [95% CI] |
| --- | ---: | ---: |
| DeBERTa-v3 NLI | 0.140 [-0.047, 0.314] | 0.145 [-0.016, 0.296] |
| Second NLI | 0.380 [0.208, 0.534] | 0.385 [0.225, 0.523] |
| RAGAS-style judge | 0.441 [0.249, 0.624] | 0.338 [0.121, 0.542] |

Interpretation: the cross-scorer intervals are tight at n=7,500. The human
calibration intervals are much wider at n=99. DeBERTa remains weakly aligned,
while the second NLI and RAGAS-style judge are moderate in this sample; the
rank-alignment intervals overlap, so scorer ordering is suggestive rather than
definitive.
