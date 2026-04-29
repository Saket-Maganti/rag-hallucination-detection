# NeurIPS-Style Submission Paper

This is the active NeurIPS-style double-blind submission folder for:

**When Retrieval Quality Decouples from Faithfulness: A Pre-Registered Audit of RAG Evaluation**

This draft is the current submission source of truth. It was moved here from
the former `audit_submission_paper/` folder during the 2026-04-29 repository
cleanup.

## Purpose

The paper argues that RAG faithfulness claims are under-identified unless local
retrieval scores, context-set structure, metric choice, human calibration,
threshold transfer, and cost are audited separately. It does not claim that CCS
causes faithfulness or that HCPC-v2 is a deployment solution.

## Important Files

- `main.tex`: 10-page main paper source.
- `supplement.tex`: short supplement source.
- `main.pdf`: compiled main paper from the latest polish pass.
- `supplement.pdf`: compiled supplement from the latest polish pass.
- `main_submission.pdf` and `supplement_submission.pdf`: submission-oriented PDF copies.
- `references.bib`: bibliography.
- `sections/`: section sources.
- `figures/`: paper figures.
- `source_tables/`: CSVs and checksums used by the paper.
- `CLAIMS_AUDIT.md`: allowed claims and forbidden claims.
- `SOURCE_TRACE.md`: claim-to-source mapping.
- `SUBMISSION_CHECKLIST.md`: submission readiness checklist.
- `RESEARCH_FRAMING.md`: concise current framing.
- `anonymized_artifact_package/`: unzipped artifact package for double-blind review.
- `anonymized_artifact_package.zip`: local zip copy, ignored by default unless intentionally staged.

## Build

```bash
cd papers/neurips
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

pdflatex -interaction=nonstopmode supplement.tex
pdflatex -interaction=nonstopmode supplement.tex
```

Or from the repository root:

```bash
make paper-neurips
```

The official `neurips_2026.sty` file is not yet present. `main.tex` falls back
to local geometry so the draft remains buildable. Add the official style file
and official checklist before final OpenReview upload.

## Submission Status

Current status from the 2026-04-29 polish pass:

- Main paper compiled to 10 pages including references.
- Supplement compiled to 2 pages.
- Major numeric claims trace to local CSVs.
- Human evaluation values were recomputed from raw rater/adjudication files.
- Anonymization scans passed for the submission source.
- Public author-identifying artifact links remain withheld for double-blind review.

Remaining before final upload:

- Add official NeurIPS style/checklist when available.
- Rebuild with official style and re-check page count and references.
- Confirm final PDFs and metadata after the official style pass.

## Known Limitations

- Fix 1 rejected the strong causal CCS interpretation in a matched-similarity test.
- Fix 2 showed the DeBERTa headline effect is small and seed-sensitive at scale.
- Human calibration has 99 examples.
- The Self-RAG row is harness-mismatched and is treated as supplement context, not as a matched ranking claim.
- Fix 7 remains budget-blocked under zero-dollar constraints.
