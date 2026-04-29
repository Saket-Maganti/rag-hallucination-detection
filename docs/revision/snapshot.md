# Reviewer Fix Results And Remaining Work

This file summarizes the NeurIPS revision fixes from the actual result files
available locally, plus the verified Fix 3/Fix 4 Kaggle package:

- `/Users/saketmaganti/Downloads/fix3_4_t4x2_outputs.zip`

Do not treat a fix as reportable unless the listed CSV/result/log files exist
or the verified zip has been imported.

## Status Snapshot

| Fix | Status | Main artifact/result |
| --- | --- | --- |
| 1 | Done | `results/revision/fix_01/*`; H1 unsupported |
| 2 | Done | `data/revision/fix_02/per_query.csv`, 7500 rows |
| 3 | Done, verified zip | `data/revision/fix_03/per_query.csv`, 7500 rows |
| 4 | Done, verified zip | `data/revision/fix_04/per_query.csv`, 7500 rows |
| 5 | Done | `data/revision/fix_05/per_query.csv`, 1591 rows |
| 6 | Pending | Kaggle notebook/runner built; result not run yet |
| 7 | Budget-blocked | Independent 70B reproduction not run under zero-dollar plan |
| 8 | Pending paper integration | Theory reframe required because Fix 1 was null |
| 9 | Limited local run complete | No-control correlation survives; control columns absent |
| 10 | Pending paper integration | Deployment/scope rewrite |
| 11 | Done | `data/revision/fix_11/per_query.csv`, 300 rows |

## Fix 1: Causal Matched-Pair Test

Files:

- `data/revision/fix_01/matched_pairs.csv`
- `data/revision/fix_01/per_query.csv`
- `results/revision/fix_01/summary.md`
- `results/revision/fix_01/paired_wilcoxon.csv`
- `results/revision/fix_01/bootstrap_ci.csv`

Construction:

| dataset | pairs | mean_abs_sim_gap | max_abs_sim_gap | mean_ccs_high | mean_ccs_low | mean_ccs_gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 200 | 0.006351 | 0.018512 | 0.670060 | 0.137426 | 0.532634 |

Paired result:

| n_pairs | high faith | low faith | high-minus-low | Wilcoxon p greater | Cohen dz | bootstrap 95% CI | H1 supported |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 200 | 0.636195 | 0.638587 | -0.002392 | 0.628268 | -0.017086 | [-0.021651, 0.016819] | False |

Interpretation:

- The causal/mechanistic claim is not supported.
- Paper language must downgrade from "drives" or "causes" to predictive,
  diagnostic, or associated framing.

## Fix 2: Scaled Headline Cell

Files:

- `data/revision/fix_02/per_query.csv`
- `results/revision/fix_02/headline_table.csv`
- `results/revision/fix_02/paired_contrasts.csv`
- `results/revision/fix_02/summary.md`

Rows:

- `7500` total rows.
- SQuAD, `n=500`, seeds `41 42 43 44 45`.
- Conditions: baseline, HCPC-v1, HCPC-v2.

Headline table:

| condition | n_rows | faithfulness | faith CI95 lo | faith CI95 hi | hallucination_rate | retrieval_similarity | refine_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2500 | 0.660947 | 0.654902 | 0.666820 | 0.1460 | 0.532193 | 0.0000 |
| hcpc_v1 | 2500 | 0.650271 | 0.644680 | 0.655954 | 0.1472 | 0.568948 | 0.0000 |
| hcpc_v2 | 2500 | 0.661196 | 0.655259 | 0.667052 | 0.1428 | 0.532152 | 0.7048 |

Paired contrast highlights:

| contrast | seed notes | interpretation |
| --- | --- | --- |
| baseline_minus_v1 | Positive in all seeds; strongest seed 44 mean diff 0.020443, p=0.008163 | HCPC-v1 generally underperforms baseline in this scaled run. |
| v2_minus_v1 | Positive in all seeds; significant in seeds 41, 44, 45 | HCPC-v2 recovers much of the HCPC-v1 loss. |
| baseline_minus_v2 | Small and mixed across seeds | HCPC-v2 is close to baseline, not clearly above it. |

Interpretation:

- The n=30 pilot should be replaced by this scaled estimate.
- HCPC-v2 looks like a conservative recovery policy, not a sweeping
  improvement over baseline.

## Fix 3: Multi-Metric Faithfulness

Source:

- Verified zip: `/Users/saketmaganti/Downloads/fix3_4_t4x2_outputs.zip`

Expected files in verified package:

- `data/revision/fix_03/per_query.csv`
- `data/revision/fix_03/per_query_partial.csv`
- `data/revision/fix_03/human_eval_template.jsonl`
- `results/revision/fix_03/table1_multimetric.csv`
- `results/revision/fix_03/metric_correlations.csv`
- `results/revision/fix_03/summary.md`
- `logs/revision/fix_03_kaggle_t4x2.log`

Verification:

- Zip integrity passed.
- Final rows: `7500`.
- Missing metric scores: `0`.
- Second-NLI scorer errors: `0`.
- RAGAS judge errors: `0`.
- Log ends with `[Fix03] wrote 7500 scored rows`.

Condition means:

| dataset | condition | n | DeBERTa | second NLI | RAGAS |
| --- | --- | ---: | ---: | ---: | ---: |
| squad | baseline | 2500 | 0.660947 | 0.350109 | 0.729640 |
| squad | hcpc_v1 | 2500 | 0.650271 | 0.318418 | 0.590434 |
| squad | hcpc_v2 | 2500 | 0.661196 | 0.350878 | 0.727948 |

Metric correlations:

| metric pair | n | Pearson r | Pearson p | Spearman rho | Spearman p |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeBERTa vs second NLI | 7500 | 0.258666 | 6.2756e-115 | 0.265295 | 5.1802e-121 |
| DeBERTa vs RAGAS | 7500 | 0.181871 | 8.6476e-57 | 0.212055 | 5.3135e-77 |
| second NLI vs RAGAS | 7500 | 0.674177 | 0.0 | 0.651497 | 0.0 |

Human-eval template:

- `human_eval_template.jsonl` contains `99` items, not `100`, because it
  stratifies evenly across three conditions (`33 x 3`).
- Report as `n=99` unless it is manually topped up.

Interpretation:

- DeBERTa weakly agrees with the alternate metrics.
- Second NLI and RAGAS agree much more strongly with each other.
- HCPC-v2 is close to baseline across all three metrics, while HCPC-v1 drops
  especially under RAGAS.

## Fix 4: Tau Generalization

Source:

- Verified zip: `/Users/saketmaganti/Downloads/fix3_4_t4x2_outputs.zip`

Expected files in verified package:

- `data/revision/fix_04/per_query.csv`
- `results/revision/fix_04/tau_summary.csv`
- `results/revision/fix_04/tau_transfer_matrix.csv`
- `results/revision/fix_04/generalization_flags.csv`
- `results/revision/fix_04/summary.md`
- `logs/revision/fix_04_kaggle_t4x2.log`

Verification:

- Zip integrity passed.
- Final rows: `7500`.
- Datasets: hotpotqa, naturalqs, pubmedqa, squad, triviaqa.
- Each dataset has `1500` rows.
- Conditions: baseline, HCPC-v1, CCS gate, each `2500` rows.
- Tau values: `0.3 0.4 0.5 0.6 0.7`, each `1500` rows.
- Error rows: `0`.

Tau summary:

| dataset | tau | baseline | HCPC-v1 | CCS gate | recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | 0.3 | 0.632815 | 0.618484 | 0.616882 | -0.111786 |
| hotpotqa | 0.4 | 0.632815 | 0.618484 | 0.614315 | -0.290908 |
| hotpotqa | 0.5 | 0.632815 | 0.618484 | 0.611741 | -0.470518 |
| hotpotqa | 0.6 | 0.632815 | 0.618484 | 0.613263 | -0.364315 |
| hotpotqa | 0.7 | 0.632815 | 0.618484 | 0.614037 | -0.310306 |
| naturalqs | 0.3 | 0.674915 | 0.655548 | 0.672291 | 0.864512 |
| naturalqs | 0.4 | 0.674915 | 0.655548 | 0.677628 | 1.140084 |
| naturalqs | 0.5 | 0.674915 | 0.655548 | 0.682690 | 1.401456 |
| naturalqs | 0.6 | 0.674403 | 0.653204 | 0.662881 | 0.456484 |
| naturalqs | 0.7 | 0.670616 | 0.653522 | 0.658179 | 0.272435 |
| pubmedqa | 0.3 | 0.559148 | 0.562581 | 0.568466 | -1.714244 |
| pubmedqa | 0.4 | 0.559148 | 0.562581 | 0.557595 | 1.452374 |
| pubmedqa | 0.5 | 0.559148 | 0.563720 | 0.560686 | 0.663605 |
| pubmedqa | 0.6 | 0.559148 | 0.563720 | 0.563128 | 0.129484 |
| pubmedqa | 0.7 | 0.559148 | 0.563720 | 0.564482 | -0.166667 |
| squad | 0.3 | 0.704895 | 0.654248 | 0.695914 | 0.822675 |
| squad | 0.4 | 0.704895 | 0.654248 | 0.679029 | 0.489289 |
| squad | 0.5 | 0.700294 | 0.654248 | 0.669306 | 0.327021 |
| squad | 0.6 | 0.700294 | 0.654248 | 0.659994 | 0.124788 |
| squad | 0.7 | 0.700294 | 0.654248 | 0.652850 | -0.030361 |
| triviaqa | 0.3 | 0.672794 | 0.654134 | 0.661569 | 0.398446 |
| triviaqa | 0.4 | 0.672794 | 0.654134 | 0.662790 | 0.463880 |
| triviaqa | 0.5 | 0.672162 | 0.654134 | 0.659955 | 0.322887 |
| triviaqa | 0.6 | 0.672162 | 0.654134 | 0.656624 | 0.138118 |
| triviaqa | 0.7 | 0.672162 | 0.654134 | 0.653656 | -0.026514 |

Transfer matrix:

| tune dataset | tau | squad | pubmedqa | hotpotqa | naturalqs | triviaqa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pubmedqa | 0.4 | 0.489289 | 1.452374 | -0.290908 | 1.140084 | 0.463880 |
| naturalqs | 0.5 | 0.327021 | 0.663605 | -0.470518 | 1.401456 | 0.322887 |
| squad | 0.3 | 0.822675 | -1.714244 | -0.111786 | 0.864512 | 0.398446 |
| triviaqa | 0.4 | 0.489289 | 1.452374 | -0.290908 | 1.140084 | 0.463880 |
| hotpotqa | 0.3 | 0.822675 | -1.714244 | -0.111786 | 0.864512 | 0.398446 |

Generalization flags:

| tune dataset | diagonal recovery | off-diagonal mean recovery | diagonal minus off-diagonal | must flag |
| --- | ---: | ---: | ---: | --- |
| pubmedqa | 1.452374 | 0.450586 | 1.001788 | True |
| naturalqs | 1.401456 | 0.210749 | 1.190707 | True |
| squad | 0.822675 | -0.140768 | 0.963443 | True |
| triviaqa | 0.463880 | 0.697710 | -0.233830 | False |
| hotpotqa | -0.111786 | 0.092847 | -0.204633 | False |

Interpretation:

- Tau generalization is not uniformly stable.
- The paper must flag diagonal-vs-offdiagonal gaps for PubMedQA, NaturalQS,
  and SQuAD.
- HotpotQA is a weak/null case for the CCS gate under this grid.

## Fix 5: Coherence-Preserving Noise

Files:

- `data/revision/fix_05/per_query.csv`
- `data/revision/fix_05/per_query_partial.csv`
- `results/revision/fix_05/noise_summary.csv`
- `results/revision/fix_05/slope_response.csv`
- `results/revision/fix_05/summary.md`

Rows:

- `1591` rows.

Summary:

| condition | n_noise | n | faith | halluc | sim |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 200 | 0.680252 | 0.155000 | 0.535800 |
| coherent_uninformative_noise | 1 | 197 | 0.667338 | 0.126904 | 0.525405 |
| coherent_uninformative_noise | 2 | 197 | 0.658515 | 0.152284 | 0.501794 |
| coherent_uninformative_noise | 3 | 197 | 0.638522 | 0.137056 | 0.450318 |
| hcpc_v1_refinement | n/a | 200 | 0.679045 | 0.110000 | 0.566900 |
| random_noise | 1 | 200 | 0.674134 | 0.125000 | 0.402512 |
| random_noise | 2 | 200 | 0.637596 | 0.145000 | 0.254744 |
| random_noise | 3 | 200 | 0.628406 | 0.100000 | 0.081838 |

Slope response:

| condition | faith slope per noise rate | sim slope per noise rate | drop at full noise |
| --- | ---: | ---: | ---: |
| random_noise | -0.068592 | -0.481011 | 0.051845 |
| coherent_uninformative_noise | -0.043224 | -0.112631 | 0.041729 |
| hcpc_v1_refinement | n/a | n/a | 0.001206 |

Interpretation:

- Random noise damages similarity much more strongly than coherent
  uninformative noise.
- Coherent uninformative noise still reduces faithfulness, but with a smaller
  similarity collapse than random noise.
- HCPC-v1 refinement has little faithfulness drop in this specific table.

## Fix 6: Proper Baseline Head-to-Head

Status:

- Pending result.
- Notebook and runner have been built locally:
  - `notebooks/revision_fix6_kaggle_t4x2_fresh.ipynb`
  - `scripts/kaggle_fix6_t4x2.sh`
  - `scripts/kaggle_stream_fix6_t4x2.py`
- `experiments/fix_06_baseline_h2h_pareto.py` now writes periodic partial
  CSVs with `--save_every`.

Recommended run order:

1. Run no-Self-RAG Fix 6 first on Kaggle T4 x2.
2. Package and download immediately.
3. Optionally run Self-RAG smoke test.
4. Only run full Self-RAG if the smoke test passes.

Expected no-Self-RAG output:

- `data/revision/fix_06/per_query.csv`
- `results/revision/fix_06/h2h_summary.csv`
- `results/revision/fix_06/pareto_faithfulness_latency.pdf`

## Fix 7: Independent 70B Reproduction

Status:

- Budget-blocked under the zero-dollar plan.

Interpretation:

- Do not fake or imply an independent 70B reproduction.
- Disclose that the 70B reproduction is not completed unless genuinely free
  70B-capable compute becomes available.

## Fix 8: Information-Theory Reframe

Status:

- Paper patch exists but is not integrated into the main paper.
- Mandatory because Fix 1 was null.

Required action:

- Rename/soften the theory section.
- Make clear that the proposition/theorem is a consistency check, not proof
  of prevalence, magnitude, or necessity.
- Replace causal/mechanistic language with predictive/diagnostic language.

Patch location:

- `papers/arxiv_longform/sections/revision/fix_08_theory_reframe.tex`

## Fix 9: Self-Confidence Partial Correlations

Status:

- Local script has been run.
- Output exists:
  - `data/revision/fix_09/input_copy.csv`
  - `results/revision/fix_09/partial_correlations.csv`
  - `results/revision/fix_09/summary.md`
- Important limitation: the available input file
  `results/confidence_calibration/per_query.csv` does not contain
  `mean_retrieval_similarity` or `passage_redundancy`, so the run only reports
  the no-control association. This does not fully resolve the confounding
  weakness.

Existing confidence-calibration result before partial controls:

| dataset | n | Pearson r | Pearson p | Spearman rho | Spearman p |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 60 | 0.3600 | 0.0047 | 0.4815 | 0.0001 |
| pubmedqa | 30 | 0.0664 | 0.7273 | 0.0323 | 0.8654 |
| squad | 30 | 0.2563 | 0.1716 | 0.1295 | 0.4953 |

Fix 9 output from the current available CSV:

| n | controls | partial Pearson r | partial Pearson p | partial Spearman rho | partial Spearman p | survives |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 60 | none | 0.360029 | 0.004720 | 0.481454 | 9.844e-05 | True |

Command run:

```bash
python3 experiments/fix_09_partial_correlations.py \
  --input results/confidence_calibration/per_query.csv
```

Interpretation rule:

- Because the full-control partial correlation could not be computed from the
  available CSV, demote the confidence result to "suggestive" unless a later
  confidence-calibration run includes the required control columns.

## Fix 10: Deployment Scope Rewrite

Status:

- Paper patch exists but is not integrated into the main paper.

Required action:

- Scope claims to short-answer extractive QA.
- Promote long-form null/non-result into a subsection named "Scope of the
  Paradox."
- Broader impact should say HCPC-v2 is a conservative short-answer deployment
  policy that requires validation before long-form use.

Patch location:

- `papers/arxiv_longform/sections/revision/fix_10_scope_deployment.tex`

## Fix 11: RAPTOR Full Table

Files:

- `data/revision/fix_11/per_query.csv`
- `results/revision/fix_11/raptor_full_table.csv`
- `results/revision/fix_11/raptor_indexing_costs.csv`
- `results/revision/fix_11/summary.md`

Rows:

- `300` rows.
- Datasets: SQuAD, PubMedQA, HotpotQA.

RAPTOR full table:

| dataset | n | faithfulness | hallucination_rate | p50 latency ms | p99 latency ms | dense index s | RAPTOR index s | index size MB | clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | 100 | 0.616708 | 0.210000 | 1972.55 | 4274.46 | 7.140 | 99.740 | 16.594 | 6 |
| pubmedqa | 100 | 0.560011 | 0.290000 | 3900.97 | 6811.90 | 1.177 | 108.264 | 3.102 | 6 |
| squad | 100 | 0.789326 | 0.050000 | 1190.46 | 4942.80 | 8.333 | 161.203 | 1.832 | 6 |

Interpretation:

- RAPTOR table is now reportable with faithfulness, hallucination rate,
  latency, indexing cost, and index size.

## Remaining Work Order

1. Run Fix 6 no-Self-RAG on Kaggle T4 x2 and download the zip immediately.
2. Optionally attempt Fix 6 Self-RAG after a smoke test.
3. Decide whether to rerun confidence calibration with similarity/redundancy
   controls for Fix 9 or report the current confidence result as suggestive.
4. Integrate Fix 8 and Fix 10 paper patches.
5. Update paper tables and narrative for Fixes 1-6, 9, 10, and 11.
6. Disclose Fix 7 as budget-blocked unless real free 70B compute appears.
