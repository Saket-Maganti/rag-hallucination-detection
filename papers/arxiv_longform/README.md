# arXiv / Long-Form Paper

This is the active long-form manuscript folder. It was moved here from the
former `ragpaper/` directory during the 2026-04-29 repository cleanup.

## Purpose

The long-form paper preserves the broader manuscript, appendix sections,
revision-section material, and historical theory/methodology text that does not
fit in the 10-page NeurIPS-style audit submission.

Use this folder for:

- arXiv preparation;
- extended appendix writing;
- long-form claim review;
- background, benchmark, mechanistic, adversarial, and methodology details.

Use `papers/neurips/` for the active short submission.

## Important Files

- `main.tex`: long-form paper source.
- `main.pdf`: compiled long-form paper from before the cleanup.
- `references.bib`: bibliography.
- `sections/`: paper sections and appendices.
- `sections/revision/`: per-fix revision section drafts.
- `figures/`: figure PDFs and TeX wrappers.
- `ARXIV_CHECKLIST.md`: long-form release checklist.

The appendix is included directly in `main.tex`; there is no separate active
`supplement.tex` for this paper.

## Build

```bash
cd papers/arxiv_longform
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or from the repository root:

```bash
make paper-longform
```

## Current Caveat

This long-form source still contains older narrative layers. Before any arXiv
posting, review it against:

- `PROJECT_HISTORY.md`
- `papers/neurips/CLAIMS_AUDIT.md`
- `docs/revision/README.md`

In particular, downgrade causal language, make Fix 1 and Fix 2 visible, and
avoid describing HCPC-v2 as a deployed solution.
