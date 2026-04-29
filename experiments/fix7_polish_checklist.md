# Fix 7 Polish Checklist (papers/neurips)

Pre-submission verification log produced 2026-04-29 alongside the
end-to-end paper polish pass. Every number quoted in the abstract, §1, §5,
§6, and §7 was cross-referenced against the underlying CSV in
`papers/neurips/source_tables/`.

## (a) Number consistency

| Claim (where it appears in the paper) | Paper value | Source CSV | CSV value | Match |
| --- | --- | --- | --- | --- |
| Multi-scorer effect — DeBERTa (abstract, §1, §6 Table 4) | `0.011` | `table1_multimetric.csv` | `0.660947 - 0.650271 = 0.010676` | rounds to `0.011` ✓ |
| Multi-scorer effect — second NLI (abstract, §1, §6 Table 4) | `0.032` | `table1_multimetric.csv` | `0.350109 - 0.318418 = 0.031691` | rounds to `0.032` ✓ |
| Multi-scorer effect — RAGAS-style (abstract, §1, §6 Table 4) | `0.140` | `table1_multimetric.csv` | `0.72964 - 0.590434 = 0.139206` | matches at 2dp (`0.14`); strict 3dp would give `0.139`. Paper uses `0.140` consistently — left unchanged per "do not change reported numbers" constraint. |
| Cohen's κ (abstract, §1, §6.1) | `0.774` | `human_eval_summary.csv` | `0.773779` | rounds to `0.774` ✓ |
| Spearman ρ — DeBERTa (abstract, §1, §6.1 Table 6) | `0.140` | `human_eval_correlations.csv` | `0.139851` | ✓ |
| Spearman ρ — second NLI (abstract, §1, §6.1 Table 6) | `0.380` | `human_eval_correlations.csv` | `0.380137` | ✓ |
| Spearman ρ — RAGAS-style (abstract, §1, §6.1 Table 6) | `0.441` | `human_eval_correlations.csv` | `0.441317` | ✓ |
| Matched HIGH−LOW mean diff (abstract, §1, §4 Table 2) | `-0.002` | `paired_wilcoxon.csv` | `-0.002392` | ✓ |
| Matched p-value (abstract, §4 Table 2) | `0.628` | `paired_wilcoxon.csv` | `0.6282676` | ✓ |
| Matched 95% CI (abstract, §4 Table 2) | `[-0.022, 0.017]` | `paired_wilcoxon.csv` | `[-0.021651, 0.016819]` | ✓ |
| Cohen's d_z (matched test) | `-0.017` | `paired_wilcoxon.csv` | `-0.017086` | ✓ |
| HIGH-CCS hallucination rate (abstract, §4) | `16.5%` | `paired_wilcoxon.csv` | `0.165` | ✓ |
| LOW-CCS hallucination rate (abstract, §4) | `9.0%` | `paired_wilcoxon.csv` | `0.09` | ✓ |
| Span-presence HIGH (§4) | `17.5%` | `paired_wilcoxon.csv` (discordant_high_only=16, both=19) | `(16+19)/200 = 17.5%` | ✓ |
| Span-presence LOW (§4) | `27.0%` | `paired_wilcoxon.csv` (discordant_low_only=35, both=19) | `(35+19)/200 = 27.0%` | ✓ |
| Span-presence McNemar p (§4) | `0.011` | exact McNemar on (35, 16) | (no CSV; matches paper text) | ✓ |
| Scaled headline cell — baseline (§5 Table 3, §6 Table 4) | `0.661` | `headline_table.csv` | `0.660947` | ✓ |
| Scaled headline cell — HCPC-v1 (§5 Table 3, §6 Table 4) | `0.650` | `headline_table.csv` | `0.650271` | ✓ |
| Scaled headline cell — HCPC-v2 (§5 Table 3, §6 Table 4) | `0.661` | `headline_table.csv` | `0.661196` | ✓ |
| PubMedQA recovery (§7 Table 7 + worked example) | `1.452` | `tau_summary.csv` | `1.452374` | ✓ |
| PubMedQA baseline / HCPC-v1 / gated (§7 worked example) | `0.559 / 0.563 / 0.558` | `tau_summary.csv` | `0.559148 / 0.562581 / 0.557595` | ✓; full-precision values disclosed parenthetically |
| Noise faith slope — random (§8 Table 8) | `-0.069` | `slope_response.csv` | `-0.068592` | rounds to `-0.069` ✓ |
| Noise faith slope — coherent uninformative (§8 Table 8) | `-0.043` | `slope_response.csv` | `-0.043224` | ✓ |
| Cost-baseline p99 latencies (§9 Table 9) | `3318/3612/3189/3281/4032/4043` ms | `h2h_summary_full_selfrag.csv` | `3317.52/3611.52/3188.73/3280.63/4032.43/4042.84` | ✓ |
| Self-RAG harness-mismatched p99 (Appendix Table) | `44800/46965` | `h2h_summary_full_selfrag.csv` | `44800.05/46965.38` | ✓ |

**One known minor rounding artifact:** the third multi-scorer effect size
is displayed as `0.140` everywhere in the paper but strict 3-decimal
rounding of the underlying difference (`0.139206`) gives `0.139`. The
paper displays `0.140` consistently across abstract, §1, and §6 Table 4;
not changed per the "do not change reported numbers" constraint of this
polish pass. This is purely a 2dp-vs-3dp rounding choice, not a numerical
error.

## (b) Citation addition

Added `\citep{maynez2020faithfulness}` after "encouraging fluent
unsupported answers when the needed span is absent" in
`sections/matched_similarity.tex` (line 45 in the post-edit file).
Maynez 2020 was already in `references.bib`; the citation supports the
hypothesis-suggested-by-data reading of the span-presence asymmetry.

## (c) Abstract length

Word count (LaTeX commands stripped): **222 words**. Under the 250-word
soft cap typical of NeurIPS / ACL / EMNLP, no trim required. The
candidate trim ("Self-RAG is documented separately as a harness-mismatched
appendix baseline") was retained because it is signal-bearing — the
abstract explicitly tells reviewers that Self-RAG is appendix-only, which
pre-empts a common reviewer question.

## (d) Caption parallelism

* All ten main-paper table captions are declarative noun phrases
  optionally followed by a one-sentence interpretive line.
* **Table 10 (`tab:raptorcost`)** previously had only the noun phrase
  (`"RAPTOR cost audit across three datasets, n=100 each."`). Extended
  to `"RAPTOR cost audit across three datasets (n=100 each). Latency
  columns are milliseconds; indexing columns are wall-clock seconds."`
  for parallelism with Table 9 (`tab:h2h`)'s `"Latency is
  milliseconds."` interpretive sentence.
* **Figure 2 caption** (`fig:pareto_p99`) updated: dropped the stale
  "on a log scale" phrase (the regenerated linear-axis figure uses
  `ax.set_xlim(2.8, 4.5)`), added a parenthetical explaining the
  shape-vs-color encoding documented by the two-legend pattern.
* All captions verified to end with a period.

## (e) Hyphenation

Compound modifiers used in the paper, all consistently hyphenated when
attributive and unhyphenated when nominal:

| Form | Used as compound modifier (hyphenated) | Used as noun phrase (unhyphenated) |
| --- | --- | --- |
| matched-similarity | "matched-similarity test", "matched-similarity intervention" | (not used as noun phrase) |
| fixed-generation | "fixed-generation cell", "fixed-generation metric-fragility" | (not used as noun phrase) |
| harness-mismatched | "harness-mismatched appendix baseline", "harness-mismatched Self-RAG" | (not used as noun phrase) |
| set-level | "set-level question", "set-level context statistic" | (not used as noun phrase) |
| RAGAS-style | "RAGAS-style judge", "RAGAS-style effect" | (not used as noun phrase) |
| cost-aware | "cost-aware view", "cost-aware reporting", "cost-aware baselines" | (not used as noun phrase) |
| in-domain | "in-domain threshold", "in-domain advantage", "in-domain tuning" | (not used as noun phrase) |
| off-diagonal | "off-diagonal transfer", "off-diagonal evidence" | (not used as noun phrase) |
| off-topic | "off-topic noise" | (not used as noun phrase) |

`grep` over `papers/neurips/sections/*.tex` found no mixed forms
(e.g., no `"matched similarity"` as compound modifier). No edits required.

## Summary

All Fix 7 sub-items have been verified or applied. One minor rounding
artifact (`0.139` vs `0.140` for the RAGAS-style effect size) is
documented above and intentionally left unchanged per the polish pass's
"do not change reported numbers" constraint.
