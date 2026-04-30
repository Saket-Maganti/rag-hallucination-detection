# Source Trace

This file maps the review-version claims in `main.tex` and
`supplement.tex` to local result directories. It is intentionally
anonymized: public release, benchmark, repository, and archive links are
omitted from the review version and will be released upon acceptance.

## Main Claims

| Claim | Paper location | Source |
| --- | --- | --- |
| Matched CCS null: HIGH-LOW faithfulness = -0.002, p = 0.628, CI [-0.022, 0.017] | main Section 5, supplement Section 3 | `results/revision/fix_01/` |
| Scaled SQuAD/Mistral audit: 0.661 / 0.650 / 0.661 over 2,500 examples per condition | main Section 6, supplement Section 2 | `results/revision/fix_02/` |
| Mistral fixed-generation metric fragility: raw 0.011 / 0.032 / 0.140 and z-scored 0.071 / 0.231 / 0.337 | main Section 7, supplement Section 6 | `results/revision/fix_03/`, `results/revision/fix_03/standardized_scorer_fragility.csv`, `scripts/compute_standardized_scorer_fragility.py` |
| Human calibration: n = 99, kappa = 0.774, Spearman 0.140 / 0.380 / 0.441 | main Section 8, supplement Section 10 | `results/human_*/` |
| Threshold-transfer summary and full sweep | main Section 9, supplement Section 9 | `results/revision/fix_04/` |
| Coherence-preserving vs random retrieval noise slopes | main Section 10, supplement Section 8 | `results/revision/fix_05/` |
| Cost-aware CRAG / HCPC-v2 / RAPTOR-2L head-to-head with faithfulness/hallucination CIs | main Section 11 | `results/revision/fix_06/`, `results/revision/fix_06/h2h_summary_with_ci.csv`, `results/revision/fix_11/`, `scripts/compute_cost_headtohead_cis.py` |
| Answer-span/control diagnostic: span and CCS complementary in matched cell | main Section 5, supplement Section 3 | `results/revision/fix_12/` |
| Stronger retriever sanity check | supplement Section 4 | `results/revision/fix_13/`, `results/multi_retriever/` |
| Qwen2.5 fixed-generation metric-fragility replication: 0.015 / 0.046 / 0.134 | main Section 7, supplement Section 5 | `results/revision/fix_14/` |
| Long-form stress test is exploratory and supplement-only | supplement `supp:longform` | `results/revision/fix_15/` |
| Matched HIGH/LOW CCS quantiles and threshold-sensitive hallucination rates | supplement Section 2 | `results/revision/fix_01/matched_ccs_faithfulness_quantiles.csv`, `results/revision/fix_01/matched_ccs_threshold_rates.csv`, `scripts/compute_matched_ccs_distribution.py` |
| Targeted scorer-disagreement human annotation artifact (unlabeled, not used for paper claims) | internal revision artifact only | `results/human_disagreement_expansion/annotation_batch_disagreement_100.csv`, `results/human_disagreement_expansion/README.md`, `scripts/select_human_disagreement_cases.py`, `scripts/analyze_human_disagreement_labels.py` |

## Artifact Manifest

The review artifact contains per-query CSVs, matched-context pairs,
source tables, scripts, supplement traces, documentation, build outputs,
and an unlabeled targeted scorer-disagreement annotation batch for
post-submission human calibration. Public links and author-identifying
release names are withheld for double-blind review.
