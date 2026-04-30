# Submission Checklist

- [ ] Official NeurIPS style status: `neurips_2026.sty` is not present in
  this repository. TODO: Replace fallback or temporary style with official
  `neurips_2026.sty` before final submission.
- [x] Main PDF compiles. Current build: `main.pdf`, 10 pages including
  references (rebuilt 2026-04-29 after the prescriptive ControlledRAG pass).
- [x] Supplement PDF compiles. Current build: `supplement.pdf`, 2 pages,
  appendix separated from the main paper.
- [x] Double-blind anonymization applied in `main.tex` and `supplement.tex`.
- [x] Double-blind source scan passed for submission sources. Build logs contain
  local TeX cache paths and are excluded from upload.
- [x] PDF metadata scan passed: main and supplement PDFs have blank Author,
  Subject, Keywords, and Title fields; no local path or username appears in
  metadata.
- [x] Author name, email, personal repository links, dataset username, DOI,
  acknowledgments, and thanks sections removed from the main submission.
- [x] Abstract is under 250 words. Current count: 247 words.
- [x] Major numeric claims trace to repository-local CSVs.
- [x] Human evaluation verified and traceable to raw rater/adjudicated files.
  Verification artifact: `results/revision/fix_03/human_eval_verification.md`.
- [x] Human-evaluation limitation stated.
- [x] Full Self-RAG artifact imported into repository-local files with checksums.
- [x] Self-RAG checksums verified against supplement Table 3.
- [x] No dangerous CCS/HCPC causal or dominance language in the paper except
  rejected-hypothesis and limitation contexts.
- [x] ControlledRAG framed as a prescriptive minimum reporting standard in
  abstract, introduction (§1 seven-item list), protocol (§3.1
  `sec:controlledrag_standard`), discussion (§10), and conclusion (§13).
- [x] Metric fragility named as the central positive result in abstract,
  introduction, and §6 metric-fragility section.
- [x] Human evaluation framed as calibration of metric disagreement, not
  validation of any scorer, in abstract, introduction, §6.1, and limitations.
- [x] Cost-aware baseline section opens and closes with identifiability framing
  rather than a method-ranking claim.
- [x] Matched CCS null is followed by an identifiability bridge clarifying
  that context structure is not irrelevant; coherence alone is insufficient.
- [x] Independent 70B rerun limitation disclosed.
- [x] No paid API claims in the revision evidence.
- [ ] NeurIPS checklist included. Add the official checklist when the venue
  package is available.
- [x] Artifact links anonymized or moved to supplement placeholders.
- [x] Main/supplement split decided; appendix is separated in `supplement.tex`.
- [x] Tables fit in the current build: Tables 1, 6, 9, 10, and supplement
  tables checked visually.
- [x] Figures readable in the current build; Figure 2 excludes the
  harness-mismatched Self-RAG rows and uses p99 latency on a log-scale x-axis.
- [x] Figure 2 regenerated with p99 latency log-axis and matched-harness
  methods only; Self-RAG moved to the supplement harness-mismatch table.
- [x] Unresolved-reference scan passes after build.
- [x] Bibliography complete in current build; no undefined citations.
- [x] No margin/font hacks added.
- [x] Bibliography compiles with current references.
- [x] Source trace file created: `SOURCE_TRACE.md`.
- [x] Claims audit file created: `CLAIMS_AUDIT.md`.
- [x] Anonymized artifact package prepared:
  `anonymized_artifact_package.zip`.

## Remaining Before OpenReview Upload

1. Add the official NeurIPS 2026 style and checklist when available.
2. Rebuild with the official style and re-check page count, overfull boxes, and
   anonymization.
3. Replace anonymous artifact placeholders only after the review period or in a
   separate post-review source.
