# Claims Audit

Guardrail for the NeurIPS research version. The governing thesis is:

> Local retrieval scores do not identify RAG faithfulness.

## Allowed Claims

| Claim | Supporting table/figure | Source CSV/result file | Exact numeric evidence |
| --- | --- | --- | --- |
| Local retrieval scores do not identify RAG faithfulness. | Table `tab:scaled`; Table `tab:causal` | `neuripsnewpaper/source_tables/headline_table.csv`; `results/revision/fix_02/headline_table.csv`; `neuripsnewpaper/source_tables/paired_wilcoxon.csv`; `results/revision/fix_01/paired_wilcoxon.csv` | HCPC-v1 raises retrieval similarity from 0.532193 to 0.568948 while DeBERTa faithfulness drops from 0.660947 to 0.650271; matched HIGH-CCS vs LOW-CCS at fixed similarity gives faithfulness 0.636195 vs 0.638587. |
| CCS is diagnostic but not causal. | Figure `fig:matched_ccs_null`; Table `tab:causal`; Table `tab:noise` | `neuripsnewpaper/source_tables/paired_wilcoxon.csv`; `results/revision/fix_01/paired_wilcoxon.csv`; `results/revision/fix_01/bootstrap_ci.csv`; `neuripsnewpaper/source_tables/slope_response.csv`; `results/revision/fix_05/slope_response.csv` | HIGH-LOW faithfulness is -0.002392, Wilcoxon p=0.628268, 95% CI [-0.021651, 0.016819]; HIGH-CCS hallucination is 16.5% vs LOW-CCS 9.0%. Coherent uninformative noise slope is -0.043224 vs random-noise slope -0.068592. |
| The scaled DeBERTa refinement effect is small and seed-sensitive. | Table `tab:scaled` | `neuripsnewpaper/source_tables/headline_table.csv`; `neuripsnewpaper/source_tables/paired_contrasts.csv`; `results/revision/fix_02/paired_contrasts.csv` | SQuAD/Mistral n=500 x 5 seeds: baseline 0.660947, HCPC-v1 0.650271, HCPC-v2 0.661196; only seed 44 is significant for baseline-v1. |
| Metric choice materially changes intervention magnitude. | Table `tab:metrics`; Table `tab:metric_corr` | `neuripsnewpaper/source_tables/table1_multimetric.csv`; `neuripsnewpaper/source_tables/metric_correlations.csv`; `results/revision/fix_03/table1_multimetric.csv`; `results/revision/fix_03/metric_correlations.csv` | Baseline-v1 effect is 0.011 under DeBERTa, 0.032 under second NLI, and 0.140 under local RAGAS-style judge; DeBERTa-RAGAS Pearson r=0.181871; second NLI-RAGAS Pearson r=0.674177. |
| Threshold transfer is uneven. | Table `tab:tauflags` | `neuripsnewpaper/source_tables/generalization_flags.csv`; `neuripsnewpaper/source_tables/tau_transfer_matrix.csv`; `results/revision/fix_04/generalization_flags.csv`; `results/revision/fix_04/tau_transfer_matrix.csv` | SQuAD, PubMedQA, and NaturalQuestions have diagonal-vs-off-diagonal recovery gaps of 0.963443, 1.001788, and 1.190707; TriviaQA and HotpotQA do not flag. |
| No tested method is uniformly best across dataset/cost surfaces. | Figure `fig:pareto_p99`; Table `tab:h2h`; Table `tab:raptorcost` | `neuripsnewpaper/source_tables/h2h_summary_full_selfrag.csv`; `results/revision/fix_06/h2h_summary_full_selfrag.csv`; `data/revision/fix_06/per_query_full_selfrag.csv`; `neuripsnewpaper/source_tables/raptor_full_table.csv`; `results/revision/fix_11/raptor_full_table.csv`; `neuripsnewpaper/source_tables/CHECKSUMS.md` | SQuAD: RAPTOR-2L 0.710022, HCPC-v2 0.708378, CRAG 0.698205, Self-RAG 0.574128. HotpotQA: CRAG 0.643152 beats HCPC-v2 0.632306 and RAPTOR-2L 0.630825, with lower hallucination and lower p99 latency. Full Self-RAG artifact has 1600 per-query rows and 8 summary rows. RAPTOR tree-build cost is 99.740-161.203 s versus dense indexing 1.177-8.333 s. |
| ControlledRAG is a reusable audit protocol. | Table `tab:controlledrag_protocol` | `neuripsnewpaper/source_tables/*`; `results/revision/fix_01/` through `fix_06/` and `fix_11/`; `experiments/fix_04_tau_generalization.py` for the threshold formula | The protocol combines six reusable controls: matched similarity, n=500 x 5 seed scale correction, fixed-generation scoring over three judges, 5x5 threshold transfer, coherent-vs-random noise slopes, and cost-aware Pareto baseline auditing. |
| Self-confidence association is suggestive only. | Appendix/source manifest | `neuripsnewpaper/source_tables/partial_correlations.csv`; `results/revision/fix_09/partial_correlations.csv` | No-control association: Pearson r=0.360029, p=0.004720; required similarity and redundancy controls were absent. |

## Forbidden Claims

- CCS drives, causes, proves, or guarantees faithfulness.
- Context coherence is the mechanism behind the refinement paradox.
- HCPC-v2 solves hallucination or dominates baselines.
- The original n=30 DeBERTa headline is the main result.
- The paradox is large and stable under DeBERTa at scale.
- A single automatic faithfulness metric identifies intervention effect size.
- A threshold tuned on SQuAD is broadly validated without transfer checks.
- The 70B result was independently rerun in the senior-review revision.
- Self-RAG is globally weak.
- The benchmark certifies faithful RAG outputs.
- RAG evaluation has been optimizing the wrong quantity, unless scoped as: standard reporting is insufficient without controls.

## Exact Baseline Verification

Full Self-RAG artifact, imported into repository-local files:

- `data/revision/fix_06/per_query_full_selfrag.csv`
- `results/revision/fix_06/h2h_summary_full_selfrag.csv`
- `neuripsnewpaper/source_tables/h2h_summary_full_selfrag.csv`
- checksums: `neuripsnewpaper/source_tables/CHECKSUMS.md`
- original archive sha256: `cf2b25a1b5c16e3430ca22ed4aeaac80dce23e9b570c4e8715be620c8ae45bb8`
- per-query rows: 1600
- summary rows: 8
- errors: 0
- queries per dataset/condition: 200
- wrapper status: `END stage=selfrag rc=0 elapsed=5071s`

| Dataset | CRAG | HCPC-v2 | RAPTOR-2L | Self-RAG |
| --- | ---: | ---: | ---: | ---: |
| SQuAD faithfulness | 0.698205 | 0.708378 | 0.710022 | 0.574128 |
| SQuAD hallucination | 0.120 | 0.125 | 0.125 | 0.390 |
| HotpotQA faithfulness | 0.643152 | 0.632306 | 0.630825 | 0.573900 |
| HotpotQA hallucination | 0.105 | 0.130 | 0.130 | 0.430 |

## Claim Audit Result

The paper may argue that RAG faithfulness is under-identified by local
relevance, CCS alone, single-metric evaluation, single-dataset threshold
tuning, and one-row baseline reporting. It may not argue that CCS is causal,
that HCPC-v2 is a deployment-dominant solution, or that the benchmark
certifies individual RAG outputs as faithful.
