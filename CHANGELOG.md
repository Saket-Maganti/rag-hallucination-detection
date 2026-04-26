# Changelog

All notable changes to this project. Versions follow semver-ish:
**major.minor.patch**, where major bumps reflect a research-narrative
shift, minor bumps add experiments or infrastructure, and patches are
fixes / docs.

## [Unreleased]

- Temperature sensitivity (Ollama backend) — running in background
- Confidence calibration — pending temperature
- Tag `v2.1.0` to be cut once temperature + confidence land
- NeurIPS senior-review revision started with Fix 1 causal intervention:
  `experiments/fix_01_causal_matched_pairs.py`,
  `experiments/fix_01_log.md`, `REVISION_SUMMARY.md`, and
  `ragpaper/sections/revision/fix_01_causal_intervention.tex`.
- `context-coherence` package version bumped to `0.2.0` for the revision
  artifact line while leaving the released API intact.
- Added runnable senior-review revision scaffolds for Fixes 2-11, including
  optional provider wrappers, a zero-cost RAGAS-style judge path, and
  second-NLI scoring wrappers.
- Added `REVISION_RUNBOOK.md` with execution commands, sequencing, zero-dollar
  platform guidance, and runtime estimates for M4 Air and free hosted GPU
  notebooks.

## [2.0.0] — 2026-04-25

The "frontier-scale + Phase 5 hardening" release. The paper and
codebase are submission-ready for NeurIPS 2026.

### Added

#### Phase 4 — ChatGPT-review hardening
- `experiments/run_topk_sensitivity.py` — top-k ablation
  (k ∈ {2, 3, 5, 10}) with bare CCS-gate baseline
- `experiments/build_disentanglement_figure.py` — fix similarity, vary CCS
- `experiments/build_coherence_heatmap.py` — pairwise sim heatmap (real or synthetic)
- `experiments/build_embedding_clusters.py` — t-SNE/UMAP per-condition layout
- `experiments/build_topk_table.py` — auto-populate `tab:topk` from CSV
- `src/ccs_gate_retriever.py` — bare CCS-gate baseline (HCPC-v$1.5$)
- New paper sections: §`sec:ccs_policy` (CCS as decision policy),
  §`sec:ccs_fails` (when CCS fails — 3 regimes), §`sec:rob:topk`
- New paper figures: `disentanglement.{pdf,tex}`, `coherence_heatmap.{pdf,tex}`,
  `embedding_clusters.{pdf,tex}`, `qualitative_cases.tex`
- Stronger abstract opener (ChatGPT's framing sentence)
- Anonymous-toggle author block (`\anonymousfalse` / `\anonymoustrue`)

#### Phase 5 — project hardening
- `tests/` — 30 tests across 4 files (CCS math, lint paper, builders, pip pkg)
  - 12 of those are in `pip-package/tests/` for the standalone metric
- `pip-package/` — standalone `pip install context-coherence` package
  (pure-numpy core, sentence-transformers optional). Distributes the
  `ccs(...)` metric and `CCSGate` decision policy.
- `scripts/push_to_hf_datasets.py` — push the benchmark to HF Datasets
  (5 configs: adversarial_drift / disjoint / contradict / control + coherence_paradox)
- `scripts/build_colab_tutorial.py` — generates `notebooks/colab_tutorial.ipynb`
- `experiments/run_quantization_sensitivity.py` — Q4/Q5/Q8 sensitivity
- `experiments/run_temperature_sensitivity.py` — T=0/0.3/0.7/1.0 (multi-backend)
- `experiments/run_crossencoder_sensitivity.py` — MiniLM-L6 / L12 / BAAI bge
- `experiments/run_confidence_calibration.py` — model self-confidence vs CCS
- `integrations/langchain/coherence_gated_retriever.py` — drop-in `BaseRetriever`
- `Makefile` — single-source recipe (`make help`)
- `Dockerfile` + `.dockerignore` — two-stage image (slim + full)
- `.github/workflows/ci.yml` — 3 jobs: tests, smoke, pip-package matrix

#### Phase 3 — pre-submission polish
- Headline figure (frontier-scale paradox vs scale)
- CCS calibration figure (distribution split + quintile bars)
- Qualitative paradox example (Super Bowl 50 / Santa Clara → SF Bay Area)
- `scripts/lint_paper.py` — pre-submission LaTeX lint (refs/cites/typos)
- `scripts/upload_to_zenodo.py` — Zenodo deposit + publish helper
- `submission/openreview_checklist.md` + `submission/paper_metadata.yml`

### Changed

- README rewritten with released-artifacts table at the top
- CITATION.bib: added `@dataset{maganti2026ccbench}` Zenodo entry
- `ragpaper/sections/abstract.tex`: opens with the ChatGPT framing sentence
- `theory.tex`: 13 placeholder labels auto-mapped to existing labels
- `space/requirements.txt`: added `audioop-lts` shim for Python 3.13 + Gradio

### Fixed

- `experiments/run_quantization_sensitivity.py`: `q4_0` mapping pointed
  at non-published Ollama tag `mistral:7b-instruct-q4_0`; corrected to
  `mistral:latest` (which IS the q4_0 build).
- `experiments/build_qualitative_example.py`: `string.Template` instead of
  `str.format` so LaTeX `{}` and `%` don't collide with Python interpolation.
- `experiments/build_embedding_clusters.py` + `run_topk_sensitivity.py`:
  `HCPCv2Retriever` doesn't take `top_k` kwarg (reads from `pipeline.top_k`).

### Permanent identifiers

- DOI: [10.5281/zenodo.19757291](https://doi.org/10.5281/zenodo.19757291)
- HF Dataset: <https://huggingface.co/datasets/saketmgnt/context-coherence-bench>
- HF Space: <https://huggingface.co/spaces/saketmgnt/sakkk>

### Headline numbers (for citation)

- **SQuAD refinement paradox**: 0.069 ± 0.004 across 3 seeds (signal/σ = 17×)
- **Frontier scale**: paradox magnitude exactly reproduced at Llama-3.3-70B
  (0.100 vs 0.100 at 7B); persists at GPT-OSS-120B (+0.030)
- **Top-k sensitivity**: paradox doubles from k=2 (0.063) to k=5 (0.124) on SQuAD
- **Quantization-agnostic**: SQuAD paradox = 0.088-0.095 across Q4/Q5/Q8
- **Reranker-agnostic**: SQuAD paradox = 0.071-0.113 across MiniLM-L6/L12/BAAI bge
- **HCPC-v2 recovery**: 0% hallucination on SQuAD with intervention rate 16.7%

## [v1.0.0] — 2026-03-15 (pre-paper)

- Initial RAG pipeline + HCPC retriever (v1)
- 5-dataset multi-dataset evaluation (SQuAD / PubMedQA / HotpotQA / NQ / TriviaQA)
- 3-generator multi-model evaluation (Mistral-7B / Llama-3-8B / Qwen-2.5-7B)
- Mechanistic attention probe
- ContextCoherenceBench v1 packaging
