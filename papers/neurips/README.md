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

## ControlledRAG: a prescriptive minimum standard

ControlledRAG is offered as the minimum standard of evidence for making RAG
faithfulness claims. A faithfulness claim that omits any of the following
axes is treated as under-identified:

1. Paired query-level contrasts with bootstrap intervals.
2. Local query-passage similarity and at least one set-level context statistic.
3. Multiple faithfulness scorers.
4. A small human calibration whenever automatic scorers disagree.
5. Threshold transfer across datasets.
6. Hallucination rates with Wilson intervals.
7. Mean latency, p99 latency, and offline indexing cost.

ControlledRAG operationalizes this standard. It does not certify faithful
outputs and it does not crown a single faithfulness metric.

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

## Where source data lives

- `source_tables/`: CSVs and checksums used directly by the paper tables.
- `../../results/revision/fix_*/`: per-fix scored outputs.
- `../../data/revision/fix_*/`: per-query inputs and human-eval rater files.

## Verify the human evaluation

```bash
python3 ../../scripts/verify_human_eval.py
```

The script recomputes the 99-example two-rater agreement, Cohen's kappa,
adjudicated label distribution, and metric/label correlations from the raw
rater and adjudication CSVs. The 2026-04-29 verification artifact is
`../../results/revision/fix_03/human_eval_verification.md`.

## Verify the artifact package

The double-blind artifact directory `anonymized_artifact_package/` ships with
`MANIFEST.txt` and `CHECKSUMS.sha256`. Re-verify with:

```bash
cd anonymized_artifact_package
shasum -a 256 -c CHECKSUMS.sha256
```

## What is anonymized for review

Author name, email, personal repository links, dataset username, public DOIs,
acknowledgments, and thanks sections are stripped from `main.tex` and
`supplement.tex`. Public author-identifying URLs are replaced with
`[anonymized repository]`, `[anonymized dataset]`, and `[anonymized DOI]` in
the supplement. Build logs are excluded from upload because they contain
local TeX cache paths.

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
