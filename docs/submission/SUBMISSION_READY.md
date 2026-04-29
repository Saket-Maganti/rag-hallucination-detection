# Submission Ready Report — papers/neurips

Final pre-submission polish summary for *"When Retrieval Quality
Decouples from Faithfulness: A Pre-Registered Audit of RAG Evaluation"*.

Polish pass executed 2026-04-29. All eight fixes from the polish brief
are landed; the paper compiles to **10 pages** including a two-column
references section, with **0 LaTeX warnings** and **0 undefined
references**.

## Per-fix change log

| Fix | Scope | Files touched | Outcome |
| --- | --- | --- | --- |
| 1 | Regenerate Figure 2 cost-aware Pareto | new `scripts/plot_cost_pareto.py`; `papers/neurips/figures/figure2_pareto_p99_log.{pdf,png}` (overwritten); caption updated in `sections/cost_baselines.tex` | Two-legend layout (Method shape, Dataset color), explicit `xlim=[2.8,4.5]`, marker size 120 with black edge, adjustText label resolution, 300 DPI export. |
| 2 | Table 1 line-break artifacts | `sections/protocol.tex` | Switched from `tabularx` `X` column to explicit `p{2.6cm}p{3.6cm}p{4.4cm}p{1.8cm}` columns with locally raised `\arraystretch{1.15}`. Cells now wrap cleanly inside their column boundaries. |
| 3 | Table 7 worked example | `sections/threshold_transfer.tex` | Caption extended with full-precision worked example: baseline = 0.559148, HCPC-v1 = 0.562581, gated@τ=0.4 = 0.557595, recovery = (0.557595−0.562581)/(0.559148−0.562581) = 1.452. Math reproduces exactly. |
| 4 | §6 "much larger" sentence | `sections/metric_fragility.tex` | Replaced "small / larger / much larger" with "small / moderate / roughly an order of magnitude larger than DeBERTa's"; added forward reference to §6.1 human calibration so the open question is named in the paragraph itself. |
| 5 | Conclusion §13 redundancy | `sections/conclusion.tex` | Trimmed to three sentences. New closing sentence positions ControlledRAG as the operational deliverable rather than restating metric fragility a second time. |
| 6 | Related Work §11 rewrite | `sections/related_work.tex`; bib spacing in `main.tex` | Three-paragraph structure (retrieval/refinement, faithfulness evaluation, retrieval-time corrective methods). Restored citations: `nogueira2019passage` (later dropped during page-budget tightening), `xu2024recomp`, `honovich2022true`, `laban2022summac`, `min2023factscore`, `liu2023geval`. Bibliography rendered in two columns at footnotesize so the +6 entries fit on the same final page. |
| 7 | Polish checklist | `experiments/fix7_polish_checklist.md`; `sections/matched_similarity.tex`; `sections/cost_baselines.tex` | Number consistency verification (see below); Maynez 2020 citation added after "fluent unsupported answers"; Figure 2 caption no longer claims log-axis (the regenerated figure is linear); Table 10 caption extended for parallelism with Table 9. |
| 8 | Release-artifact consistency | new `SUBMISSION_CHECKLIST.md` (repo root) | Per-artifact inventory listing each releasable file, current visibility (anonymized / public / withheld), and whether it is ready for submission. Anonymization scan confirmed clean: no `Saket`, `maganti`, `freeopenapex`, or `github.com/Saket-Maganti` strings anywhere in `main.tex`, `sections/*.tex`, or the anonymized artifact package. |

## Number-consistency confirmation (Fix 7a)

All headline numbers in the abstract, §1, §5, §6, and §7 trace cleanly
to their CSV sources in
[`papers/neurips/source_tables/`](papers/neurips/source_tables/).
Full per-claim audit table is in
[`experiments/fix7_polish_checklist.md`](experiments/fix7_polish_checklist.md).
Summary:

* **Multi-scorer effect sizes** 0.011 / 0.032 / 0.140 trace to
  `table1_multimetric.csv` (the third value rounds strictly to 0.139 at
  3dp; paper consistently displays 0.140 — minor 2dp-vs-3dp rounding
  artifact, not changed per "do not change reported numbers"
  constraint).
* **Cohen's κ = 0.774** matches `human_eval_summary.csv` (0.773779).
* **Spearman ρ for human alignment** 0.140 / 0.380 / 0.441 matches
  `human_eval_correlations.csv` exactly at 3dp.
* **Matched HIGH−LOW = −0.002, p=0.628, CI [−0.022, 0.017], d_z=−0.017**
  matches `paired_wilcoxon.csv`.
* **HIGH/LOW hallucination 16.5% / 9.0%** matches `paired_wilcoxon.csv`.
* **Span-presence 17.5% / 27.0%, McNemar p=0.011** is internally
  consistent: from the four-way (35, 16, 19, 130) count, totals are
  HIGH = 35/200, LOW = 54/200.
* **Scaled headline cell 0.661 → 0.650 → 0.661** matches
  `headline_table.csv`.
* **PubMedQA recovery = 1.452** matches `tau_summary.csv` at full
  precision.

## Build statistics

| Metric | Value |
| --- | --- |
| Page count (`pdfinfo`) | **10 pages** |
| Abstract word count | **222 words** (under 250-word cap) |
| LaTeX warnings (filtered to non-Underfull, non-font) | **0** |
| Undefined references | **0** |
| Cited bibitems | 20 |
| Bibliography layout | two-column footnotesize, fits on page 10 |
| File size | `main.pdf` 449 kB |
| Builder | pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026) |

## Remaining TODOs (post-double-blind)

These are deliberately left for camera-ready de-anonymization and do not
block the current submission:

1. Drop `neurips_2026.sty` into `papers/neurips/` once the
   official venue package is released; the source already detects and
   uses it via `\IfFileExists`.
2. Insert the official NeurIPS 2026 reproducibility checklist when the
   venue publishes it.
3. Restore public-artifact links at camera-ready (currently withheld
   for double-blind; full inventory in
   [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) §B):
   * GitHub repository
   * pip package `context-coherence`
   * Zenodo DOI `10.5281/zenodo.19757291`
   * HuggingFace dataset `saketmgnt/context-coherence-bench`
   * HuggingFace Space demo
4. After de-anonymization, also revisit:
   * The `\anonymoustrue` toggle in `main.tex` (line 62) — flip to
     `\anonymousfalse`.
   * The placeholder language in `sections/appendix.tex` §`app:release`
     ("[anonymized repository]" → real URLs).

None of these are polish-pass blockers; they are post-decision actions.

## Files written or modified by this polish pass

```
scripts/plot_cost_pareto.py                                      (new)
SUBMISSION_CHECKLIST.md                                          (new)
SUBMISSION_READY.md                                              (this file)
experiments/fix7_polish_checklist.md                             (new)
papers/neurips/figures/figure2_pareto_p99_log.pdf        (regenerated)
papers/neurips/figures/figure2_pareto_p99_log.png        (regenerated)
papers/neurips/main.tex                                  (bib formatting)
papers/neurips/sections/protocol.tex                     (Fix 2)
papers/neurips/sections/threshold_transfer.tex           (Fix 3)
papers/neurips/sections/metric_fragility.tex             (Fix 4)
papers/neurips/sections/conclusion.tex                   (Fix 5)
papers/neurips/sections/related_work.tex                 (Fix 6)
papers/neurips/sections/matched_similarity.tex           (Fix 7b)
papers/neurips/sections/cost_baselines.tex               (Fig 2 + Table 10 captions)
papers/neurips/main.pdf                                  (final build)
papers/neurips/main.aux/.bbl/.blg/.log/.out              (build artifacts)
```

The polish pass did not run new experiments, did not change any reported
number, and did not soften the negative results. The paper still leads
with metric fragility on fixed generations as the central positive
contribution; the matched-similarity null, multi-seed scaling collapse,
threshold-transfer unevenness, and span-presence asymmetry all remain
explicit and unminimized.
