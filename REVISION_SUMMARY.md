# Revision Summary

This file tracks the senior reviewer's 10 weaknesses for the NeurIPS revision.
Fix 1 is the active gating item; downstream fixes should not be framed until
the causal intervention result is known.

Operational handoff, zero-dollar execution plan, and session checklist live in
`CODEX.md`.

## Current State

| # | Weakness | Fix | Status | Assessment |
| --- | --- | --- | --- | --- |
| 1 | Causal-vs-correlational CCS claim | Matched-similarity HIGH/LOW CCS intervention on SQuAD | Partial | Protocol and harness added; required 200 matched pairs constructed with zero skips and max similarity gap 0.018512. Full generation/NLI run still required before the paper can claim causal evidence. |
| 2 | Headline cell sample size too small | SQuAD/Mistral n=500 x 5 seeds | Code | Script/log/LaTeX written; execution pending. |
| 3 | Single faithfulness metric | Add local RAGAS-style judge + second NLI + optional human eval | Code | Script/log/LaTeX plus scorer wrappers written; execution pending after Fix 2. Under zero-dollar mode, GPT-4o-mini/Claude judging is replaced by local judging plus human eval. |
| 4 | Tau-tuning leakage | 5x5 tune-on/eval-on matrix | Code | Script/log/LaTeX written; execution pending. |
| 5 | SQuAD noise-slope disclosure | Coherence-preserving same-topic noise injection | Code | Script/log/LaTeX written; execution pending. |
| 6 | Baseline head-to-head weak | Self-RAG/CRAG/RAPTOR/HCPC table | Code | Script/log/LaTeX written; Self-RAG optional CUDA path included. |
| 7 | Single-backend 70B reliance | Together.ai Llama-3.3-70B reproduction | Budget-blocked | Script/log/LaTeX and Together wrapper written, but true 70B reproduction is not feasible under zero-dollar mode unless free 70B compute appears. |
| 8 | Information-theory overclaim | Rename/reframe as consistency check | Code | Paper patch/log written; should be wired after Fix 1 result. |
| 9 | Self-confidence confounding | Partial correlations controlling similarity/redundancy | Code | Script/log/LaTeX written; execution pending. |
| 10 | Deployment scope oversold | Scope abstract and promote long-form non-result | Code | Paper patch/log written; wire after Fix 1 result. |
| 11 | RAPTOR full table | Per-dataset/per-metric RAPTOR table | Code | Script/log/LaTeX written; execution pending. |

## Fix 1 Artifacts

| Path | Purpose |
| --- | --- |
| `REVISION_RUNBOOK.md` | Exact execution commands, sequencing, zero-dollar platform guidance, and runtime estimates for local/free-GPU runs. |
| `experiments/fix_01_log.md` | Pre-registration, Claude handoff audit, protocol, and honest-reporting commitment. |
| `experiments/fix_01_causal_matched_pairs.py` | Construction, generation, and analysis harness for matched-similarity HIGH/LOW CCS triples. |
| `data/revision/fix_01/` | Per-query and matched-pair CSV outputs with column documentation. |
| `results/revision/fix_01/` | Construction diagnostics, paired Wilcoxon, bootstrap CIs, and summary once generated. |
| `ragpaper/sections/revision/fix_01_causal_intervention.tex` | Paper-ready subsection with track-changes comments and pending result placeholders. |

## Code Artifacts For Remaining Fixes

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

## Decision Rule

Fix 1 resolves Weakness 1 only if the primary SQuAD run reaches at least 200
matched pairs and satisfies all preregistered criteria:

- paired Wilcoxon on `faith_high - faith_low` gives `p < 0.05`;
- Cohen's `d_z > 0.2`;
- 10000-resample bootstrap 95% CI excludes 0;
- constructed pairs remain within the hard `0.02` mean-similarity tolerance.

If not, the central claim is downgraded from causal/mechanistic to predictive
and diagnostic throughout the paper.

## Fix 1 Construction Result

The construction stage has completed for the preregistered primary SQuAD cell:

| metric | value |
| --- | ---: |
| valid matched pairs | 200 |
| skipped queries | 0 |
| mean absolute similarity gap | 0.006351 |
| maximum absolute similarity gap | 0.018512 |
| mean CCS gap | 0.532634 |
| minimum CCS gap | 0.264139 |
| maximum HIGH/LOW passage overlap | 1 |

Interpretation: the dataset construction routine satisfies the matching
constraints and is ready for the expensive Mistral/DeBERTa run. This is not
yet evidence for or against the causal hypothesis.
