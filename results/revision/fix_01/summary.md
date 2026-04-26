# Fix 1 - causal coherence intervention

This experiment tests whether HIGH-CCS contexts produce more faithful
answers than LOW-CCS contexts when mean per-passage query similarity is
matched within +/-0.02.

## Match diagnostics

| dataset   |   n_pairs_constructed |   mean_sim_high |   mean_sim_low |   mean_abs_sim_gap |   max_abs_sim_gap |   mean_ccs_high |   mean_ccs_low |   mean_ccs_gap |   min_ccs_gap |   mean_overlap |   max_overlap |   mean_bucket_size |
|:----------|----------------------:|----------------:|---------------:|-------------------:|------------------:|----------------:|---------------:|---------------:|--------------:|---------------:|--------------:|-------------------:|
| squad     |                   200 |        0.423588 |       0.421478 |           0.006351 |          0.018512 |         0.67006 |       0.137426 |       0.532634 |      0.264139 |          0.395 |             1 |             255.44 |

## Paired faithfulness test

(generation/NLI not run yet)

Decision rule: H1 is supported only when the one-sided paired Wilcoxon
p-value is < 0.05, Cohen's dz is > 0.2, the 10000-resample bootstrap
CI on the mean paired difference excludes 0, and the max similarity
gap remains <= 0.02 by construction.

If this rule fails, the paper must downgrade causal/mechanistic wording.
