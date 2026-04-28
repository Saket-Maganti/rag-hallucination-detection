# Revision Summary

This file tracks the senior reviewer's 10 weaknesses for the NeurIPS revision.
Fix 1 was the active gating item. Its preregistered causal intervention has now
completed and did not support the causal H1, so downstream paper language must
downgrade mechanistic/causal claims to diagnostic or predictive claims.

The canonical revision book is [`README.md`](README.md) (full info in
one document). Operational handoff, zero-dollar execution plan, and
session checklist live in [`codex.md`](codex.md). Exact execution
commands and runtime estimates live in [`runbook.md`](runbook.md).

## Current State

| # | Weakness | Fix | Status | Assessment |
| --- | --- | --- | --- | --- |
| 1 | Causal-vs-correlational CCS claim | Matched-similarity HIGH/LOW CCS intervention on SQuAD | Done | 200 matched pairs generated/scored. H1 not supported: HIGH-CCS mean faithfulness 0.636195 vs LOW-CCS 0.638587, HIGH-LOW -0.002392, Wilcoxon p=0.628268, bootstrap CI [-0.021651, 0.016819]. Causal language must be downgraded. |
| 2 | Headline cell sample size too small | SQuAD/Mistral n=500 x 5 seeds | Done | 7500 rows. Baseline faithfulness 0.660947, HCPC-v1 0.650271, HCPC-v2 0.661196. The old n=30 headline should be treated as a pilot. |
| 3 | Single faithfulness metric | Add local RAGAS-style judge + second NLI + optional human eval | Done | 7500 rows from verified Kaggle T4 x2 package. DeBERTa weakly agrees with alternate metrics; second NLI vs RAGAS has Pearson r=0.674177 and Spearman rho=0.651497. Human-eval template has 99 items. |
| 4 | Tau-tuning leakage | 5x5 tune-on/eval-on matrix | Done | 7500 rows. Tau generalization is uneven; diagonal-vs-offdiagonal gaps must be flagged for PubMedQA, NaturalQS, and SQuAD. |
| 5 | SQuAD noise-slope disclosure | Coherence-preserving same-topic noise injection | Done | 1591 rows. Coherent uninformative noise has smaller similarity slope than random noise, but still lowers faithfulness. |
| 6 | Baseline head-to-head weak | Self-RAG/CRAG/RAPTOR/HCPC table | Pending compute | Fresh Kaggle T4 x2 notebook and runner are built for no-Self-RAG first, with optional Self-RAG smoke/full run. |
| 7 | Single-backend 70B reliance | Together.ai Llama-3.3-70B reproduction | Budget-blocked | Script/log/LaTeX and Together wrapper written, but true 70B reproduction is not feasible under zero-dollar mode unless free 70B compute appears. |
| 8 | Information-theory overclaim | Rename/reframe as consistency check | Pending paper integration | Paper patch/log written. Fix 1 null result makes this mandatory. |
| 9 | Self-confidence confounding | Partial correlations controlling similarity/redundancy | Limited run complete | Available CSV lacks similarity/redundancy controls, so only no-control association was computed: Pearson r=0.360029, p=0.004720; Spearman rho=0.481454. Treat confidence as suggestive unless controls are regenerated. |
| 10 | Deployment scope oversold | Scope abstract and promote long-form non-result | Pending paper integration | Paper patch/log written; wire now and scope claims to short-answer extractive QA. |
| 11 | RAPTOR full table | Per-dataset/per-metric RAPTOR table | Done | 300 rows. Full table includes faithfulness, hallucination rate, p50/p99 latency, dense indexing cost, RAPTOR tree cost, and index size. |

## Fix 1 Artifacts

| Path | Purpose |
| --- | --- |
| [`runbook.md`](runbook.md) | Exact execution commands, sequencing, zero-dollar platform guidance, and runtime estimates for local/free-GPU runs. |
| `experiments/fix_01_log.md` | Pre-registration, Claude handoff audit, protocol, and honest-reporting commitment. |
| `experiments/fix_01_causal_matched_pairs.py` | Construction, generation, and analysis harness for matched-similarity HIGH/LOW CCS triples. |
| `data/revision/fix_01/` | Per-query and matched-pair CSV outputs with column documentation. |
| `results/revision/fix_01/` | Construction diagnostics, paired Wilcoxon, bootstrap CIs, and summary once generated. |
| `ragpaper/sections/revision/fix_01_causal_intervention.tex` | Paper-ready subsection with track-changes comments and pending result placeholders. |

## Code And Result Artifacts

| Fix | Script / Patch | Log | Paper section |
| --- | --- | --- | --- |
| 2 | `experiments/fix_02_scaled_headline_n500.py` | `experiments/fix_02_log.md` | `ragpaper/sections/revision/fix_02_scaled_headline.tex` |
| 3 | `experiments/fix_03_multimetric_faithfulness.py` | `experiments/fix_03_log.md` | `ragpaper/sections/revision/fix_03_multimetric.tex` |
| 4 | `experiments/fix_04_tau_generalization.py` | `experiments/fix_04_log.md` | `ragpaper/sections/revision/fix_04_tau_generalization.tex` |
| 5 | `experiments/fix_05_coherence_preserving_noise.py` | `experiments/fix_05_log.md` | `ragpaper/sections/revision/fix_05_noise_slope.tex` |
| 6 | `experiments/fix_06_baseline_h2h_pareto.py` | `experiments/fix_06_log.md` | `ragpaper/sections/revision/fix_06_baselines.tex` |
| 7 | `experiments/fix_07_together_70b_reproduction.py` | `experiments/fix_07_log.md` | `ragpaper/sections/revision/fix_07_together.tex` |
| 8 | paper-only | `experiments/fix_08_log.md` | `ragpaper/sections/revision/fix_08_theory_reframe.tex` |
| 9 | `experiments/fix_09_partial_correlations.py` | `experiments/fix_09_log.md` | `ragpaper/sections/revision/fix_09_partial_confidence.tex` |
| 10 | paper-only | `experiments/fix_10_log.md` | `ragpaper/sections/revision/fix_10_scope_deployment.tex` |
| 11 | `experiments/fix_11_raptor_full_table.py` | `experiments/fix_11_log.md` | `ragpaper/sections/revision/fix_11_raptor_full_table.tex` |

Key completed result paths:

| Fix | Main result files |
| --- | --- |
| 2 | `data/revision/fix_02/per_query.csv`; `results/revision/fix_02/headline_table.csv`; `results/revision/fix_02/paired_contrasts.csv` |
| 3 | `data/revision/fix_03/per_query.csv`; `results/revision/fix_03/table1_multimetric.csv`; `results/revision/fix_03/metric_correlations.csv` |
| 4 | `data/revision/fix_04/per_query.csv`; `results/revision/fix_04/tau_summary.csv`; `results/revision/fix_04/tau_transfer_matrix.csv`; `results/revision/fix_04/generalization_flags.csv` |
| 5 | `data/revision/fix_05/per_query.csv`; `results/revision/fix_05/noise_summary.csv`; `results/revision/fix_05/slope_response.csv` |
| 9 | `data/revision/fix_09/input_copy.csv`; `results/revision/fix_09/partial_correlations.csv` |
| 11 | `data/revision/fix_11/per_query.csv`; `results/revision/fix_11/raptor_full_table.csv`; `results/revision/fix_11/raptor_indexing_costs.csv` |

## Decision Rule

Fix 1 resolves Weakness 1 only if the primary SQuAD run reaches at least 200
matched pairs and satisfies all preregistered criteria:

- paired Wilcoxon on `faith_high - faith_low` gives `p < 0.05`;
- Cohen's `d_z > 0.2`;
- 10000-resample bootstrap 95% CI excludes 0;
- constructed pairs remain within the hard `0.02` mean-similarity tolerance.

If not, the central claim is downgraded from causal/mechanistic to predictive
and diagnostic throughout the paper.

## Fix 1 Result

The construction stage completed for the preregistered primary SQuAD cell:

| metric | value |
| --- | ---: |
| valid matched pairs | 200 |
| skipped queries | 0 |
| mean absolute similarity gap | 0.006351 |
| maximum absolute similarity gap | 0.018512 |
| mean CCS gap | 0.532634 |
| minimum CCS gap | 0.264139 |
| maximum HIGH/LOW passage overlap | 1 |

The full Mistral/DeBERTa generation and analysis stage then completed:

| metric | value |
| --- | ---: |
| generated rows | 400 |
| complete matched pairs | 200 |
| mean faithfulness, HIGH-CCS | 0.636195 |
| mean faithfulness, LOW-CCS | 0.638587 |
| HIGH minus LOW | -0.002392 |
| Wilcoxon p-value, HIGH > LOW | 0.628268 |
| Cohen's dz | -0.017086 |
| bootstrap 95% CI lower | -0.021651 |
| bootstrap 95% CI upper | 0.016819 |
| hallucination rate, HIGH-CCS | 0.165 |
| hallucination rate, LOW-CCS | 0.090 |
| H1 supported | false |

Interpretation: the construction constraints were satisfied, but the
intervention did not show that higher CCS causally improves faithfulness at
fixed mean query similarity. The paper should preserve CCS as a diagnostic or
predictive signal only unless a later, preregistered replication changes this
result.
