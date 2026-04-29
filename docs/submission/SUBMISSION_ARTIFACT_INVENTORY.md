# Submission Artifact Inventory — papers/neurips

Generated 2026-04-29 as part of the eight-fix pre-submission polish pass.
This document is **per-artifact** (each row is one releasable thing and
its visibility state). The process-level reviewer checklist remains at
[`papers/neurips/SUBMISSION_CHECKLIST.md`](papers/neurips/SUBMISSION_CHECKLIST.md);
this file is the artifact-side companion.

Status legend:

* **anonymized** — present and double-blind safe; ships in supplement now.
* **public** — already live on the open Internet; cannot be unmade public.
* **withheld** — exists internally but deliberately not in the submission.
* **TODO** — needs an action before submission can proceed.

## A. In-paper artifact references

| Artifact | Reference in `main.tex` | Status | Ready? | Notes |
| --- | --- | --- | --- | --- |
| `\method{}` (ControlledRAG) protocol code | §1 contribution: "Code and data are anonymized in the supplementary material and will be released upon acceptance." | anonymized | ✅ | Bundled in `papers/neurips/anonymized_artifact_package/experiments/`. No author identifiers in any CSV header or filename. |
| Per-query CSVs (Fix 1, 2, 3, 4, 5, 6, 11) | Appendix Table `tab:source_manifest` | anonymized | ✅ | Verified: column names are neutral (`question`, `answer`, `faithfulness_score`, …). No GitHub usernames in paths. |
| Human calibration adjudication file | Appendix §`app:human_eval`, Table `tab:human_label_distribution` | anonymized | ✅ | `human_eval_rater_a.csv`, `human_eval_rater_b.csv`, `human_eval_adjudicated.csv` carry no rater identity. SHA-256 documented in `source_tables/CHECKSUMS.md`. |
| Self-RAG audit (full Llama-2-7B run) | Appendix §`app:selfrag`, Tables `tab:selfrag_checksums`, `tab:selfrag_harness` | anonymized | ✅ | Per-query CSV (1600 rows, `aaf00af2…`) + summary CSV (8 rows, `5d325428…`). Imported repository-locally so the paper does not depend on an external path. |
| Threshold-transfer matrix | §7 Table `tab:tauflags` | anonymized | ✅ | `tau_transfer_matrix.csv` + `tau_summary.csv` in `papers/neurips/source_tables/`. |
| Cost-aware Pareto figure | §9 Figure `fig:pareto_p99` | anonymized | ✅ | `figures/figure2_pareto_p99_log.pdf` + `.png`, regenerated 2026-04-29 by `scripts/plot_cost_pareto.py` with two-legend layout and explicit `xlim`. |

## B. Public artifacts referenced from earlier paper drafts

These exist on the open Internet under the author's identifiable account,
and therefore must not be cited in the double-blind submission. They are
listed here as a deliberate exclusion record so a reviewer-aware re-add
after acceptance is straightforward.

| Artifact | URL (post-acceptance) | Status during review | Action at camera-ready |
| --- | --- | --- | --- |
| GitHub repository | `github.com/Saket-Maganti/rag-hallucination-detection` | withheld | restore in §1 contributions paragraph and §13 conclusion |
| pip package `context-coherence` (v2 publishable) | `pypi.org/project/context-coherence/` | withheld | mention in §3 release artifacts subsection at camera-ready |
| Zenodo DOI | `10.5281/zenodo.19757291` | withheld | restore in `references.bib` and `submission_packages/neurips/openreview/paper_metadata.yml` |
| HuggingFace dataset | `saketmgnt/context-coherence-bench` | withheld | restore in supplement docs only; do not pull into main paper |
| HuggingFace Space demo | `huggingface.co/spaces/saketmgnt/coherence-paradox-rag-demo` | withheld | optional camera-ready footnote |

The submission-time text already uses placeholders such as
`[anonymized repository]`, `[anonymized dataset]`, and `[anonymized DOI]`
in `sections/appendix.tex` (`app:release`). No accidental
de-anonymization observed in scans of `main.tex` or `sections/*.tex`.

## C. Anonymization scan results (2026-04-29)

```
$ grep -rE "Saket|saket|maganti|freeopenapex|@gmail|github\\.com/Saket" \\
    papers/neurips/main.tex papers/neurips/sections/*.tex \\
    papers/neurips/anonymized_artifact_package/
(no matches)
```

`main.tex` line 64 sets `\author{Anonymous Authors\\\textit{Submission
under double-blind review}}` and `\anonymoustrue` is the default. PDF
metadata for `main_submission.pdf` and `supplement_submission.pdf` was
previously verified blank by the upstream submission checklist; that
verification is preserved here by reference.

## D. Repository state (worktree-local edits made in this polish pass)

These files were written or edited by the 2026-04-29 polish pass and are
ready to commit alongside the paper:

* `scripts/plot_cost_pareto.py` (new — Fix 1)
* `papers/neurips/figures/figure2_pareto_p99_log.{pdf,png}` (regenerated — Fix 1)
* `papers/neurips/sections/protocol.tex` (Table 1 column widths — Fix 2)
* `papers/neurips/sections/threshold_transfer.tex` (Table 7 worked example — Fix 3)
* `papers/neurips/sections/metric_fragility.tex` (§6 paragraph rewrite — Fix 4)
* `papers/neurips/sections/conclusion.tex` (three-sentence rewrite — Fix 5)
* `papers/neurips/sections/related_work.tex` (three-paragraph §11 — Fix 6)
* `papers/neurips/sections/matched_similarity.tex` (Maynez 2020 citation — Fix 7b)
* `papers/neurips/sections/cost_baselines.tex` (Fig. 2 caption + Table 10 caption — Fix 7d)
* `experiments/fix7_polish_checklist.md` (Fix 7 verification log)
* `SUBMISSION_CHECKLIST.md` (this file — Fix 8)

## E. Outstanding TODOs (not blockers for the polish pass)

| TODO | Owner | Trigger |
| --- | --- | --- |
| Drop `neurips_2026.sty` into `papers/neurips/` | author | when venue package is released |
| Insert official NeurIPS reproducibility checklist | author | before OpenReview upload |
| Restore public-artifact links and acknowledgments | author | at camera-ready, post double-blind |
| Confirm 10-page page count after final compile | this pass | covered in `SUBMISSION_READY.md` |
