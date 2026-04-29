# Move Manifest - 2026-04-29

This manifest records the intentional repository moves made during the
2026-04-29 cleanup. It complements `BACKUP_MANIFEST_2026-04-29.md`, which was
captured before the reorganization as a broad file inventory.

## Active Paper Folders

| Old path | New path | Rationale |
|---|---|---|
| `audit_submission_paper/` | `papers/neurips/` | Latest NeurIPS-style audit submission source. |
| `ragpaper/` | `papers/arxiv_longform/` | Most complete long-form paper source, including revision sections and latest figures. |
| `ragpaper/main_neurips.tex` | `archive/legacy_papers/ragpaper_main_neurips_variant/main_neurips.tex` | Preserved old NeurIPS variant from the long-form folder. |
| `ragpaper/main_neurips.pdf` | `archive/legacy_papers/ragpaper_main_neurips_variant/main_neurips.pdf` | Preserved compiled PDF for the old variant. |

## Archived Paper Folders

| Old path | New path | Rationale |
|---|---|---|
| `paper_longform/` | `archive/legacy_duplicates/paper_longform/` | Older duplicate of the long-form paper. |
| `paper_neurips/` | `archive/legacy_papers/paper_neurips/` | Older NeurIPS-style draft. |
| `claudeneuroipspaper/` | `archive/legacy_papers/claudeneuroipspaper/` | Earlier NeurIPS-style draft line. |
| `neuripsnewpaper/` | `archive/legacy_papers/neuripsnewpaper/` | Earlier safer NeurIPS-style draft line. |

## Submission Materials

| Old path | New path | Rationale |
|---|---|---|
| `submission/` | `submission_packages/neurips/openreview/` | OpenReview metadata/checklist package. |
| `audit_submission_paper.zip` | `submission_packages/neurips/audit_submission_paper.zip` | Local package snapshot for the NeurIPS audit paper. Ignored by git by default. |
| `ragpaper.zip` | `archive/old_zips/ragpaper.zip` | Historical long-form package zip. Ignored by git by default. |
| `ragpaper 2.zip` | `archive/old_zips/ragpaper_2.zip` | Historical duplicate package zip. Ignored by git by default. |

## Documentation

| Old path | New path | Rationale |
|---|---|---|
| `SUBMISSION_READY.md` | `docs/submission/SUBMISSION_READY.md` | Preserved final audit readiness note under submission docs. |
| `SUBMISSION_CHECKLIST.md` | `docs/submission/SUBMISSION_ARTIFACT_INVENTORY.md` | Root artifact inventory moved out of the top-level directory. |
| `POLISH_SUMMARY.md` | `docs/submission/POLISH_SUMMARY.md` | Submission polish summary moved under submission docs. |
| `RUNBOOK.md` | `docs/project/RUNBOOK_legacy.md` | Historical runbook preserved under project docs. |
| `analysis.md` | `archive/miscellaneous/analysis_initial_auto.md` | Early miscellaneous analysis preserved in archive. |

## Generated and Local Artifacts

| Old path | New path | Rationale |
|---|---|---|
| `chroma_db*/` | `archive/legacy_chroma_dbs/` | Local Chroma/vector DB stores preserved for reproducibility, removed from the active root. |
| `logs/` | `archive/old_logs/logs_snapshot/` | Historical logs preserved; fresh root `logs/` kept for future runs. |
| `run_queue.log` | `archive/old_logs/run_queue.log` | Historical run queue log preserved. |
| `kaggle/` | `archive/old_logs/kaggle_working_snapshot/` | Kaggle working snapshot preserved. |
| `hf_dataset_staging/` | `artifacts/generated/hf_dataset_staging_snapshot/` | Generated staging output moved under artifacts. |
| `space_deploy/` | `artifacts/generated/space_deploy_snapshot/` | Generated HF Space deployment snapshot moved under artifacts. |

## Path Updates

- Paper-build references now point to `papers/neurips/` or
  `papers/arxiv_longform/`.
- New generated Chroma stores are directed to `artifacts/generated/chroma_db*`
  instead of top-level `chroma_db*`.
- OpenReview package references now point to
  `submission_packages/neurips/openreview/`.

