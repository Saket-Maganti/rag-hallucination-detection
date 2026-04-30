# Project History and AI Handoff

## 1. Project Identity

- Project name: RAG Hallucination Detection
- Current research theme: evaluation under-identification in retrieval-augmented generation faithfulness.
- Main paper direction: a conservative, pre-registered audit of RAG faithfulness claims.
- Current submission targets: NeurIPS-style submission first; arXiv / long-form draft second.
- Active NeurIPS paper path: `papers/neurips/`
- Active arXiv / long-form paper path: `papers/arxiv_longform/`

## 2. Executive Summary

This project studies when RAG evidence looks better locally but produces less
faithful answers. The initial story focused on the "refinement paradox": more
aggressive retrieval refinement can fragment retrieved context, and that
fragmentation can make a generator fill gaps with unsupported content. A later
senior-review revision substantially changed the interpretation. The matched
similarity CCS intervention was null, the original small-sample DeBERTa paradox
collapsed at larger scale, and the strongest current result is that different
faithfulness evaluators produce materially different conclusions on the same
generations. The repository should now be read as an audit-protocol artifact,
not as proof that CCS causes faithfulness.

## 3. Research Problem

Many RAG papers use local retrieval scores, a single automatic faithfulness
metric, and aggregate answer quality to justify claims about faithfulness.
Those signals are not enough when the retrieved set can be structurally
fragmented, when scorer choice changes effect size, when thresholds do not
transfer across datasets, and when methods have different latency and indexing
costs. This work asks what controls are needed before a RAG evaluation can
support a deployment-level claim.

## 4. Current Thesis / Core Claim

Current central claim:

> RAG faithfulness claims are under-identified unless retrieval similarity,
> context-set structure, evaluator choice, human calibration, threshold
> transfer, and cost are separated.

CCS remains a useful diagnostic and experimental variable. It should not be
described as a proven causal mechanism in the current evidence state. HCPC-v2
should be described as a controlled probe / audit condition, not a method that
solves hallucination or dominates baselines.

## 5. Historical Timeline

| Date / Period | Evidence | What Happened | Confidence |
|---|---|---|---|
| 2026-03-15 | `CHANGELOG.md` v1.0.0 entry | Initial RAG pipeline, HCPC retriever, multi-dataset evaluation, multi-model evaluation, mechanistic attention probe, and benchmark packaging existed. | High |
| 2026-04-23 | `AGENTS.md`, `CLAUDE.md`, professor-feedback block | Professor feedback emphasized that the paper could be attacked as metric based and dependent on one embedding model. Multi-retriever and multi-dataset/human-eval responses were prioritized. | High |
| 2026-04-24 | `RUNBOOK.md`, `analysis.md`, Phase 2.5 notes in old agent files | Additional rigor plans were staged: noise injection, prompt template ablation, RAG-vs-zero-shot, long-form evaluation, RAPTOR comparison, and deployment figure. | High |
| 2026-04-25 | `CHANGELOG.md` v2.0.0, `results/frontier_scale/`, `release/`, `space/` | Frontier-scale and release hardening landed: benchmark release, Zenodo DOI, HF dataset/space source, LangChain integration, Docker, CI, Makefile, tests, and multiple robustness figures. | High |
| 2026-04-25 to 2026-04-28 | `docs/revision/README.md`, `experiments/fix_*_log.md`, `results/revision/fix_*` | Senior-reviewer revision ran Fix 1 through Fix 11 under zero-dollar and pre-registration constraints. Several original claims were weakened. | High |
| 2026-04-28 | commits `7f1aca5e`, `0fb01489`, `efb6fbbc`; `docs/revision/` | Revision docs were consolidated into `docs/revision/`; the old agent handoff files were updated with current status. | High |
| 2026-04-29 | `papers/neurips/`, `docs/submission/SUBMISSION_READY.md`, `papers/neurips/SOURCE_TRACE.md` | A 10-page double-blind NeurIPS-style audit submission was polished. Human calibration and full Self-RAG artifacts were imported and traced. | High |
| 2026-04-29 | this cleanup; `docs/audits/BACKUP_MANIFEST_2026-04-29.md` | Repository structure was reorganized into active paper folders, docs, submission packages, artifacts, and archive while preserving legacy work. | High |
| Unknown / inferred | folder names and moved legacy papers | `claudeneuroipspaper/` appears to have been an earlier fresh NeurIPS draft; `neuripsnewpaper/` appears to have been a safer under-identification draft; `audit_submission_paper/` blended both. | Medium |

## 6. Paper Evolution

The paper evolved through several layers:

- Early long-form work: `ragpaper/` and `paper_longform/` held the long
  "When Better Retrieval Hurts" manuscript and appendix-heavy material.
- Early short NeurIPS draft: `paper_neurips/` compressed the same narrative
  into a body-only submission-style paper.
- Audit draft line: `claudeneuroipspaper/` developed a fresh pre-registered
  audit framing.
- Safer NeurIPS line: `neuripsnewpaper/` shifted toward under-identification
  and safer claims after the revision results.
- Final blended audit line: `audit_submission_paper/` combined the stronger
  audit framing, safer under-identification claims, human calibration, source
  trace, claims audit, and an anonymized artifact package.

After cleanup:

- `audit_submission_paper/` was moved to `papers/neurips/`.
- `ragpaper/` was moved to `papers/arxiv_longform/` because it was the
  long-form source with revision section material.
- `paper_longform/` was preserved in `archive/legacy_duplicates/paper_longform/`
  because it was essentially a duplicate of the long-form source without the
  additional revision-section material.
- `paper_neurips/`, `claudeneuroipspaper/`, and `neuripsnewpaper/` were moved
  to `archive/legacy_papers/`.
- The old `ragpaper/main_neurips.*` variant was preserved under
  `archive/legacy_papers/ragpaper_main_neurips_variant/`.

## 7. Current Active Paper Versions

Active NeurIPS paper path: `papers/neurips/`

Purpose: the current double-blind 10-page NeurIPS-style audit submission. It
contains `main.tex`, `supplement.tex`, `main.pdf`, `supplement.pdf`,
`references.bib`, `sections/`, `figures/`, `source_tables/`,
`CLAIMS_AUDIT.md`, `SOURCE_TRACE.md`, and `SUBMISSION_CHECKLIST.md`.

Active arXiv / long-form paper path: `papers/arxiv_longform/`

Purpose: the long-form manuscript with appendix sections and broader
historical material. It contains `main.tex`, `main.pdf`, `references.bib`,
`sections/`, `figures/`, and the per-fix revision section files under
`sections/revision/`.

## 8. Repository Map

- `src/`: reusable RAG, retrieval, scorer, generator, and metric modules.
- `experiments/`: experiment runners, per-fix revision scripts, and paper table/figure builders.
- `scripts/`: operational helpers for plotting, packaging, Kaggle streaming, linting, HF/Zenodo staging, and releases.
- `notebooks/`: Colab and Kaggle notebooks for revision and Self-RAG runs.
- `data/`: curated inputs, adversarial examples, benchmark data, and revision per-query outputs.
- `results/`: experiment outputs, aggregated summaries, figures, and paper tables.
- `docs/revision/`: current senior-reviewer revision book, status, runbook, and snapshot.
- `docs/submission/`: submission polish reports and artifact inventory moved from root.
- `docs/audits/`: repository-level cleanup and backup manifests.
- `docs/project/`: project operations, reorganization report, and legacy runbook.
- `papers/neurips/`: active NeurIPS-style submission.
- `papers/arxiv_longform/`: active long-form / arXiv-style paper.
- `submission_packages/`: OpenReview metadata/checklists and local upload packages.
- `artifacts/`: generated staging snapshots and non-source outputs.
- `archive/`: preserved historical material, old papers, logs, zips, and local vector stores.
- `release/`: benchmark release bundles used by existing package/upload scripts.
- `pip-package/`: standalone `context-coherence` package.
- `integrations/`: LangChain integration.
- `space/` and `leaderboard/`: HF Space app and leaderboard source.
- `tests/`: lightweight tests for builders, CCS math, and paper lint helpers.

## 9. Codebase Map

- `src/rag_pipeline.py`: primary RAG pipeline. Default local Chroma store is now under `artifacts/generated/chroma_db`.
- `src/coherence_metrics.py`: CCS and retrieval-level statistics.
- `src/hcpc_retriever.py`, `src/hcpc_v2_retriever.py`: HCPC-v1 / HCPC-v2 retrieval conditions.
- `src/crag_retriever.py`, `src/raptor_retriever.py`, `src/selfrag_wrapper.py`: baseline wrappers.
- `src/ragas_scorer.py`, `src/vectara_hem_scorer.py`, provider wrappers: metric and model adapters.
- `experiments/fix_01_*` through `experiments/fix_11_*`: senior-reviewer revision scripts and pre-registration logs.
- `experiments/build_*`: figure/table builders. Output paths now target `papers/arxiv_longform/` unless the builder is submission-specific.
- `scripts/plot_cost_pareto.py`: regenerates the NeurIPS audit Pareto figure in `papers/neurips/`.
- `scripts/lint_paper.py`: LaTeX lint for the long-form paper by default.
- `Makefile`: common recipes for tests, figures, papers, release helpers, and Docker.
- `Dockerfile`: reproducibility image for tests, figures, and paper build support.
- `main.py`: older entry point for demo/evaluation modes.

## 10. Data and Chroma DB History

The repository previously had many top-level `chroma_db*` folders. These are
local/generated Chroma vector stores, not canonical source code. They were moved
to `archive/legacy_chroma_dbs/` and ignored by git. They are preserved locally
for reproducibility investigation unless proven unnecessary.

Future experiment runs should write local vector stores under
`artifacts/generated/chroma_db*`. Existing scripts were updated from
`./chroma_db*` to `./artifacts/generated/chroma_db*` where practical.

Tracked historical Chroma files that were previously committed under
`chroma_db/` are removed from the active git tree in this cleanup; the local
copies remain under the archive path.

## 11. Experiments and Results

Major experiment families:

- Original ablations: chunk size, top-k, prompt strategy, multi-model,
  reranker, zero-shot, adaptive chunking, HCPC.
- Robustness and hardening: multi-retriever, cross-encoder sensitivity,
  quantization, temperature, top-k, long-form, frontier scale, deployment
  figure, confidence calibration.
- Senior-reviewer revision:
  - Fix 1: matched-similarity CCS intervention, null result.
  - Fix 2: n=500 x 5 seed scaled headline, small / seed-sensitive DeBERTa effect.
  - Fix 3: multi-metric scoring and human calibration.
  - Fix 4: threshold transfer.
  - Fix 5: coherent-vs-random noise.
  - Fix 6: baseline head-to-head including full Self-RAG artifact.
  - Fix 7: budget-blocked 70B rerun.
  - Fix 8: theory reframe.
  - Fix 9: confidence correlation, limited no-control result.
  - Fix 10: deployment scope reframe.
  - Fix 11: RAPTOR full table.

Important current result files:

- `results/revision/fix_01/paired_wilcoxon.csv`
- `results/revision/fix_02/headline_table.csv`
- `results/revision/fix_03/table1_multimetric.csv`
- `results/revision/fix_03/human_eval_summary.md`
- `results/revision/fix_04/generalization_flags.csv`
- `results/revision/fix_05/slope_response.csv`
- `results/revision/fix_06/h2h_summary_full_selfrag.csv`
- `results/revision/fix_11/raptor_full_table.csv`

## 12. Submission Packages

- `submission_packages/neurips/openreview/`: OpenReview checklist and metadata inherited from the older `submission/` folder.
- `submission_packages/neurips/audit_submission_paper.zip`: local copy of the current audit paper package. Zip files are ignored by git unless intentionally staged.
- `papers/neurips/anonymized_artifact_package/`: current unzipped anonymized artifact package.
- `papers/neurips/anonymized_artifact_package.zip`: zip package produced by the submission polish pass. It is kept locally and ignored by default.
- `archive/old_zips/`: older zip snapshots, including old `ragpaper` packages.
- `release/`: benchmark release bundles used by existing scripts, not a paper submission package.

## 13. Important Existing Documents

- `README.md`: GitHub-facing current overview.
- `PROJECT_HISTORY.md`: this canonical human and AI handoff.
- `AGENTS.md`: short instructions for Codex and other agentic tools.
- `CLAUDE.md`: short instructions for Claude Code style sessions.
- `docs/revision/README.md`: senior-reviewer revision book and detailed evidence map.
- `docs/revision/status.md`: per-fix scoreboard.
- `docs/revision/runbook.md`: exact execution commands.
- `docs/revision/codex.md`: old operational handoff, now subordinate to this file.
- `docs/submission/SUBMISSION_READY.md`: 2026-04-29 polish pass report.
- `docs/submission/SUBMISSION_ARTIFACT_INVENTORY.md`: artifact visibility and anonymization inventory.
- `docs/submission/POLISH_SUMMARY.md`: short polish summary.
- `papers/neurips/CLAIMS_AUDIT.md`: allowed and forbidden claims.
- `papers/neurips/SOURCE_TRACE.md`: paper claim-to-file trace.
- `papers/neurips/RESEARCH_FRAMING.md`: concise current framing.
- `docs/project/RUNBOOK_legacy.md`: older project execution runbook.
- `CHANGELOG.md`: chronological release and experiment notes.

## 14. Known Issues / Risks

- The NeurIPS folder still needs the official `neurips_2026.sty` and official checklist when available.
- The current NeurIPS PDF was built before the move; build validation should be repeated after any significant paper edit.
- The long-form paper still carries some older stronger language and must be edited before public arXiv release.
- Some archived papers contain old claims and old paths; treat them as history, not active truth.
- The local archived Chroma DBs are not tracked by git and may not exist on a fresh clone.
- Zip packages are preserved locally but ignored unless intentionally staged for a release.
- Fix 7 remains budget-blocked under zero-dollar constraints.
- Human calibration is useful but small: 99 adjudicated examples.
- Some source and revision docs mention old paths inside frozen anonymized artifact packages; those packages are preserved as produced.

## 15. Future Work Plan

Immediate cleanup:

- Re-run lightweight tests and paper lint after path updates.
- Rebuild `papers/neurips/main.pdf` and `papers/neurips/supplement.pdf` after adding the official style.
- Rebuild `papers/arxiv_longform/main.pdf` after long-form claim edits.

Paper polish:

- Keep the NeurIPS paper centered on under-identification and audit protocol.
- Keep Fix 1 null and Fix 2 scale collapse prominent.
- Remove or qualify any remaining "drives", "causes", "dominates", or
  "solves hallucination" language.

Experiments:

- Add a second matched-similarity domain or generator if zero-dollar compute allows.
- Add larger human calibration if budget / IRB constraints are resolved.
- Re-run confidence calibration with similarity and redundancy controls.

Reproducibility:

- Keep scripts writing generated DBs under `artifacts/generated/`.
- Keep release builders working with `release/`.
- Keep `PROJECT_HISTORY.md` and `papers/neurips/SOURCE_TRACE.md` updated after new results.

Long-term:

- Prepare an arXiv long-form version with all revision lessons integrated.
- Maintain benchmark and package docs separately from double-blind submission text.
- Add a clean arXiv packaging script once the long-form is stable.

## 16. Instructions for Future AI Agents

1. Read `PROJECT_HISTORY.md` first.
2. Then read `README.md`.
3. For submission work, read `papers/neurips/README.md`, `CLAIMS_AUDIT.md`, and `SOURCE_TRACE.md`.
4. For long-form work, read `papers/arxiv_longform/README.md`.
5. For revision evidence, read `docs/revision/README.md`.
6. Do not delete `archive/`.
7. Do not stage local Chroma DBs, logs, virtualenvs, caches, or accidental zips.
8. Use existing evidence before editing claims.
9. Preserve null results and reviewer-facing caveats.
10. Update this file after major paper, experiment, release, or structure changes.

## 17. Last Cleanup Summary

Cleanup run date: 2026-04-29.

What changed:

- Created canonical active paper folders:
  - `papers/neurips/`
  - `papers/arxiv_longform/`
- Moved old paper variants to:
  - `archive/legacy_papers/`
  - `archive/legacy_duplicates/`
- Moved old root submission docs into `docs/submission/`.
- Moved the OpenReview helper folder into `submission_packages/neurips/openreview/`.
- Moved generated staging snapshots into `artifacts/generated/`.
- Moved old logs and Kaggle working files into `archive/old_logs/`.
- Moved old zips into `archive/old_zips/`.
- Moved local Chroma DB folders into `archive/legacy_chroma_dbs/`.
- Removed cache and LaTeX intermediate files where safe.
- Updated script paths from root `ragpaper/` to `papers/arxiv_longform/` and from root Chroma paths to `artifacts/generated/chroma_db*`.
- Rewrote root `README.md`.
- Replaced `AGENTS.md` and `CLAUDE.md` with short current handoff files.
- Added `docs/project/REORGANIZATION_REPORT.md`.

See `docs/project/REORGANIZATION_REPORT.md` for validation status and git push metadata.

## 18. 2026-04-29 Prescriptive ControlledRAG Polish

Polish run date: 2026-04-29.

Goal: address senior-reviewer / professor feedback that the paper risked
being read as a collection of negative results. ControlledRAG was sharpened
into an explicit prescriptive minimum reporting standard for RAG faithfulness
audits, while preserving every verified value, rejected hypothesis, and
forbidden-claim rule.

What changed in `papers/neurips/`:

- `sections/protocol.tex`: added `\paragraph{ControlledRAG as a minimum
  reporting standard.}` (`\label{sec:controlledrag_standard}`) so the
  prescriptive framing has a named anchor used by the discussion and
  conclusion.
- `sections/matched_similarity.tex`: added an identifiability bridge
  clarifying that the matched CCS null does not imply context structure is
  irrelevant; it shows coherence alone is insufficient and points forward to
  the separable reporting axes.
- `sections/cost_baselines.tex`: opens with "this section is not a
  leaderboard" and closes with the identifiability message that the
  preferred matched-harness method changes once dataset and cost axes are
  included.
- `sections/discussion.tex`: extended the "what can and cannot be claimed"
  ledger with an explicit pointer to `sec:controlledrag_standard` for future
  RAG faithfulness papers.
- `sections/conclusion.tex`: final sentence promotes ControlledRAG from
  operational deliverable to prescriptive minimum standard of evidence.
- `sections/related_work.tex`: condensed the three-paragraph related-work
  block into one paragraph to free vertical space; every previously cited
  reference is preserved.
- `sections/limitations_ethics.tex`: merged short-answer scope and CCS scope,
  and merged Self-RAG and large-model reproduction limitations, freeing
  additional vertical space.
- `sections/metric_fragility.tex`, `sections/introduction.tex`: minor
  compression to keep the body+references inside the 10-page envelope.
- `CLAIMS_AUDIT.md`: added prescriptive ControlledRAG entry to allowed
  claims; added "ControlledRAG certifies faithful RAG outputs" and
  "ControlledRAG validates any single faithfulness scorer" to forbidden
  claims.
- `SOURCE_TRACE.md`: added a row for `sec:controlledrag_standard` mapping
  the seven-item minimum reporting standard to the section sources.
- `SUBMISSION_CHECKLIST.md`: added explicit ControlledRAG-prescriptive,
  metric-fragility-central, human-calibration-as-calibration, cost-as-
  identifiability, and matched-CCS-bridge items.
- `README.md`: added a "ControlledRAG: a prescriptive minimum standard"
  section, plus blocks describing where source data lives, how to verify
  the human evaluation, how to verify the artifact package, and what is
  anonymized for review.
- `main.pdf` and `supplement.pdf` rebuilt; `main_submission.pdf` and
  `supplement_submission.pdf` refreshed.

Verification status:

- Main PDF: 10 pages including references; PDF metadata blank for Title,
  Subject, Keywords, and Author.
- Supplement PDF: 2 pages; metadata blank.
- Dangerous-language scan: every match in §sections is a rejected hypothesis,
  limitation, descriptive method name, or table label.
- Double-blind scan: no author identifiers, no local paths, no
  acknowledgments, no thanks, no camera-ready text in `main.tex`,
  `supplement.tex`, `references.bib`, or any of `sections/*.tex`.
- All verified numbers preserved: matched CCS null (HIGH-LOW = -0.002,
  p = 0.628, 95% CI [-0.022, 0.017]), HIGH/LOW hallucination 16.5%/9.0%,
  span presence 27.0%/17.5% with McNemar p=0.011, intervention magnitudes
  0.011/0.032/0.140, human eval n=99 / kappa=0.774 / Spearman 0.140/0.380/
  0.441.
