# Claims Audit

This file is the claim ledger for the double-blind blended submission. The
paper's safe thesis is:

> RAG faithfulness claims are under-identified unless retrieval similarity,
> context-set structure, evaluator choice, human calibration, threshold
> transfer, and cost are separated.

## Allowed Claims

| Claim | Supporting paper item | Source files | Exact evidence |
| --- | --- | --- | --- |
| Retrieval quality and faithfulness can decouple. | Tables `tab:scaled`, `tab:metrics` | `results/revision/fix_02/headline_table.csv`; `results/revision/fix_03/table1_multimetric.csv` | In the scaled SQuAD/Mistral cell, HCPC-v1 has higher retrieval similarity than baseline (0.5689 vs. 0.5322) but lower DeBERTa faithfulness (0.650271 vs. 0.660947). |
| Local retrieval scores do not identify RAG faithfulness. | Sections 4-9 | `results/revision/fix_01/` through `fix_06/`; `results/revision/fix_11/` | Matched similarity gives HIGH--LOW = -0.002392; metric magnitudes vary from 0.011 to 0.140; threshold gaps vary from -0.234 to 1.191; baseline ordering changes by dataset/cost. |
| CCS is diagnostic but not causal in the tested matched-similarity setting. | Table `tab:causal`; Table `tab:noise` | `results/revision/fix_01/paired_wilcoxon.csv`; `results/revision/fix_05/slope_response.csv` | Fix 1: HIGH faith = 0.636195, LOW faith = 0.638587, HIGH--LOW = -0.002392, Wilcoxon p = 0.628268, CI [-0.021651, 0.016819]. Fix 5: coherent-noise slope -0.043 vs. random-noise slope -0.069. |
| Metric choice materially changes intervention magnitude. | Tables `tab:metrics`, `tab:metric_corr` | `results/revision/fix_03/table1_multimetric.csv`; `results/revision/fix_03/metric_correlations.csv`; `results/revision/fix_03/bootstrap_correlation_cis.csv` | Baseline--v1 magnitudes are DeBERTa 0.011, second NLI 0.032, RAGAS-style judge 0.140. DeBERTa/RAGAS Pearson r = 0.182 [0.162, 0.202]; second NLI/RAGAS Pearson r = 0.674 [0.661, 0.686]. |
| Human calibration shows weak DeBERTa alignment and moderate-at-best automatic metric alignment in the audited 99-example sample, with overlapping rank-correlation intervals. | Table `tab:human_calibration` | `data/revision/fix_03/human_eval_rater_a.csv`; `data/revision/fix_03/human_eval_rater_b.csv`; `data/revision/fix_03/human_eval_adjudicated.csv`; `results/revision/fix_03/human_eval_summary.csv`; `results/revision/fix_03/human_eval_correlations.csv`; `results/revision/fix_03/bootstrap_correlation_cis.csv`; `results/revision/fix_03/human_eval_verification.csv` | n = 99, two raters, raw agreement = 0.919192, Cohen's kappa = 0.773779. Spearman vs. adjudicated human labels: DeBERTa 0.140 [-0.047, 0.314], second NLI 0.380 [0.208, 0.534], RAGAS-style judge 0.441 [0.249, 0.624]. |
| Threshold transfer is uneven. | Table `tab:tauflags` | `results/revision/fix_04/generalization_flags.csv`; `results/revision/fix_04/tau_transfer_matrix.csv` | SQuAD, PubMedQA, and NaturalQuestions show diagonal-vs-off-diagonal gaps greater than 0.03. TriviaQA and HotpotQA do not flag. |
| No matched-harness tested method has one deployment ordering across dataset/cost surfaces. | Figure `fig:pareto_p99`; Tables `tab:h2h`, `tab:raptorcost` | `results/revision/fix_06/h2h_summary_full_selfrag.csv`; `data/revision/fix_06/per_query_full_selfrag.csv`; `results/revision/fix_11/raptor_full_table.csv` | SQuAD faithfulness: RAPTOR-2L 0.710022, HCPC-v2 0.708378, CRAG 0.698205. HotpotQA: CRAG 0.643152 beats HCPC-v2 0.632306 and RAPTOR-2L 0.630825 with lower hallucination and lower p99 latency. RAPTOR tree build is 99.740-161.203 s. Self-RAG is separated into a harness-mismatched supplement table and is not used for the matched-harness ordering claim. |
| ControlledRAG is a reusable audit protocol. | Table `tab:controlledrag_protocol` | `experiments/fix_01_log.md` through `experiments/fix_06_log.md`; `experiments/fix_11_log.md`; `docs/revision/runbook.md` | The protocol combines matched similarity, multi-seed scaling, fixed-generation multi-metric and human-calibrated scoring, threshold transfer, coherent-vs-random noise, and cost-aware baseline auditing. |

## Forbidden Claims

Do not claim any of the following in the main paper or supplement except as a
rejected hypothesis, limitation, or forbidden-language reminder:

- CCS drives faithfulness.
- CCS causes faithfulness.
- CCS proves faithfulness.
- HCPC-v2 solves hallucination.
- HCPC-v2 dominates.
- The DeBERTa paradox is large and stable at scale.
- Independent 70B reproduction was rerun in the revision.
- Self-RAG is globally weak.
- Human evaluation definitively settles faithfulness metric validity.
- The benchmark certifies faithful RAG outputs.
- This is submission-ready if style, checklist, and page count are not fixed.
