# Submission Checklist

Reviewer-facing checks before exporting the NeurIPS submission PDF.

- [ ] Claim audit passed against `neuripsnewpaper/CLAIMS_AUDIT.md`.
- [ ] All reported numbers trace to CSVs in `neuripsnewpaper/source_tables/` or `results/revision/fix_*/`.
- [ ] Abstract is <= 200 words.
- [ ] Human eval status explicit: no completed two-rater calibration is reported; staged templates are marked pending.
- [ ] Self-RAG artifact imported or TODO explicit; current import is `data/revision/fix_06/per_query_full_selfrag.csv` and `results/revision/fix_06/h2h_summary_full_selfrag.csv`.
- [ ] Self-RAG checksums recorded in `neuripsnewpaper/source_tables/CHECKSUMS.md`.
- [ ] No "drives" or "causes" language appears except as a rejected hypothesis or forbidden claim.
- [ ] No "causal", "mechanism", "proves", "dominates", "solves", or "guarantees" language is unqualified.
- [ ] No paid API claims or paid-compute results are introduced.
- [ ] Independent 70B reproduction limitation is disclosed as budget-blocked under zero-dollar constraints.
- [ ] Official NeurIPS 2026 style file (`neurips_2026.sty`) is used.
- [ ] TODO if absent: Replace local fallback with official neurips_2026.sty before submission.
- [ ] NeurIPS checklist is preserved or added from the official template before final submission.
- [ ] Anonymity checked: author block, acknowledgments, artifact links, repository links, and PDF metadata.
- [ ] Artifact links are anonymized or moved to supplementary material for double-blind submission.
- [ ] Main-vs-supplement split decided: `main.tex` builds the main paper; `supplement.tex` builds artifact/source-table notes.
- [ ] PDF compiles from a clean checkout.
- [ ] Figures are readable in grayscale and at printed size.
- [ ] Tables do not overflow the page or rely on unreadably small text.
- [ ] References are complete and free of hallucinated citations.
- [ ] No paid API claims appear in the paper, checklist, or artifact notes.
- [ ] No margin or font hacks are introduced.
- [ ] Claim audit passed after the final dangerous-language scan.
