# Repository Reorganization Report

Date/time: 2026-04-29 19:57:01 IST

## Summary

The repository was reorganized into a cleaner research-project layout for
GitHub readers, future AI agents, paper builds, and NeurIPS/arXiv-style
submission work. No research history was intentionally destroyed. Old or
uncertain material was moved into `archive/`, generated package snapshots were
moved under `artifacts/`, and the two active paper lines now live under
`papers/`.

## Active Paper Locations

| Purpose | Active path | Source used |
|---|---|---|
| NeurIPS-style submission | `papers/neurips/` | Former `audit_submission_paper/` |
| arXiv / long-form paper | `papers/arxiv_longform/` | Former `ragpaper/` |

## Old-to-New Mapping

The detailed move manifest is in
`docs/audits/MOVE_MANIFEST_2026-04-29.md`. Major moves:

| Old location | New location |
|---|---|
| `audit_submission_paper/` | `papers/neurips/` |
| `ragpaper/` | `papers/arxiv_longform/` |
| `paper_longform/` | `archive/legacy_duplicates/paper_longform/` |
| `paper_neurips/` | `archive/legacy_papers/paper_neurips/` |
| `claudeneuroipspaper/` | `archive/legacy_papers/claudeneuroipspaper/` |
| `neuripsnewpaper/` | `archive/legacy_papers/neuripsnewpaper/` |
| `submission/` | `submission_packages/neurips/openreview/` |
| `SUBMISSION_READY.md` | `docs/submission/SUBMISSION_READY.md` |
| `SUBMISSION_CHECKLIST.md` | `docs/submission/SUBMISSION_ARTIFACT_INVENTORY.md` |
| `POLISH_SUMMARY.md` | `docs/submission/POLISH_SUMMARY.md` |
| `RUNBOOK.md` | `docs/project/RUNBOOK_legacy.md` |
| `chroma_db*/` | `archive/legacy_chroma_dbs/` |
| `logs/` | `archive/old_logs/logs_snapshot/` |
| `hf_dataset_staging/` | `artifacts/generated/hf_dataset_staging_snapshot/` |
| `space_deploy/` | `artifacts/generated/space_deploy_snapshot/` |

## Documentation Changes

- Rewrote the root `README.md` as the canonical GitHub entry point.
- Added `PROJECT_HISTORY.md` as the canonical project-history and AI-handoff
  document.
- Updated `AGENTS.md` and `CLAUDE.md` to point future tools at the new
  structure.
- Added paper-specific readmes for `papers/neurips/` and
  `papers/arxiv_longform/`.
- Added archive, artifact, submission-package, logs, and project-doc readmes.
- Added `.gitignore` rules for Python, Jupyter, LaTeX, logs, zips, and local
  vector DBs.

## Code and Path Updates

- Updated root paper references from `ragpaper/` and legacy paper folders to
  `papers/arxiv_longform/` or `papers/neurips/`.
- Updated generated Chroma/vector-store paths from top-level `chroma_db*` to
  `artifacts/generated/chroma_db*` where practical.
- Updated Makefile paper targets to build from the new paper folders.
- Updated source-trace and submission docs to reference the new layout.

## Archived Material

Archived material remains available locally under:

- `archive/legacy_papers/`
- `archive/legacy_duplicates/`
- `archive/legacy_chroma_dbs/`
- `archive/old_logs/`
- `archive/old_zips/`
- `archive/miscellaneous/`

The archive is intentionally preservation-oriented. Do not delete it unless a
human maintainer explicitly decides a historical artifact is no longer needed.

## Generated and Ignored Material

The following classes of material are intentionally ignored by git after the
cleanup:

- local Chroma/vector DB directories;
- temporary Python caches;
- temporary LaTeX build files;
- log files;
- zip/tar package outputs;
- generated snapshots under `artifacts/generated/`, except their README.

PDFs are not globally ignored because paper PDFs can be intentional submission
artifacts.

## Validation Results

Validation was run after the reorganization.

| Check | Result |
|---|---|
| `README.md` exists | Passed |
| `PROJECT_HISTORY.md` exists | Passed |
| `papers/neurips/` exists | Passed |
| `papers/arxiv_longform/` exists | Passed |
| `papers/neurips/README.md` exists | Passed |
| `papers/arxiv_longform/README.md` exists | Passed |
| `find . -maxdepth 3 -type d \| sort` | Passed; root layout contains the intended `papers/`, `archive/`, `docs/`, `submission_packages/`, and `artifacts/` directories. |
| `find papers -maxdepth 3 -type f \| sort` | Passed; both active paper folders contain source, figures/tables, PDFs, and README files. |
| `python3 -m pytest tests/ -v --tb=short` | Passed: 23 tests passed. |
| `python3 scripts/lint_paper.py` | Passed: 0 errors, 183 warnings, 8 notes. Warnings are pre-existing style/reference hygiene items in the long-form paper. |
| `make paper-neurips` | Passed; produced `papers/neurips/main.pdf` (10 pages). |
| `pdflatex supplement.tex` in `papers/neurips/` | Passed; produced `papers/neurips/supplement.pdf` (2 pages). |
| `make paper-longform` | Passed; produced `papers/arxiv_longform/main.pdf` (55 pages). Existing warning noise remains, including multiply-defined labels in the long-form appendix path. |

## Unresolved Issues and Risks

- The official NeurIPS 2026 style file is not present; the NeurIPS source has a
  fallback build path and should be rebuilt with the official style before
  final upload.
- The anonymized artifact package is preserved as a submission snapshot. Some
  internal paths inside that package may intentionally reflect its original
  standalone layout.
- `latexmk` was not available in the local environment, so validation used
  `pdflatex` and the repository Makefile targets.
- Archived Chroma DB folders are preserved for reproducibility, but are not
  considered active source-controlled project state.
- Generated zips are ignored by git by default. If a final zip package must be
  versioned, stage it intentionally and document why.

## Recommended Next Steps

1. Rebuild both papers from the new `papers/` paths after the official style
   files are available.
2. Re-run lightweight tests after dependency changes or before release tags.
3. Keep `PROJECT_HISTORY.md` updated after major experiments, paper framing
   changes, or submission milestones.
4. Use `submission_packages/` for final packaged uploads and `artifacts/` for
   generated staging outputs.
5. Avoid reintroducing top-level paper variants, Chroma DB folders, zip files,
   or ad hoc markdown status files.

## Git Metadata

This section is updated after commit/push.

| Field | Value |
|---|---|
| Cleanup commit | Pending |
| Final report commit | Pending |
| Branch | Pending |
| Remote | Pending |
| Push status | Pending |
| Intentionally untracked / ignored files | Pending |
