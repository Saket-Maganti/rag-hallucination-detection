# RAG Hallucination Detection

This repository contains the research artifact for a controlled study of
faithfulness failures in retrieval-augmented generation (RAG). The current
paper direction is deliberately conservative: local retrieval scores, context
coherence, evaluator choice, threshold transfer, human calibration, and cost
must be separated before making strong claims about RAG faithfulness.

The project began as a study of the "refinement paradox", where more aggressive
per-passage retrieval refinement can reduce answer faithfulness by fragmenting
the context given to a generator. Later senior-reviewer revisions weakened the
original causal story: the strongest matched-similarity test was null, the
headline DeBERTa effect became small at scale, and the cleanest positive result
became metric divergence on fixed generations. The current submission therefore
frames the work as a pre-registered audit protocol and evidence that common RAG
faithfulness claims are under-identified.

## Research Motivation

RAG systems are often evaluated with local retrieval metrics, a single
faithfulness scorer, and aggregate answer quality. That can hide several
failure modes:

- retrieved passages can be individually relevant but mutually incoherent;
- faithfulness effects can be scorer-dependent;
- thresholds tuned on one dataset may not transfer;
- stronger retrieval methods may change latency and cost rankings;
- human calibration may not agree cleanly with automatic scorers.

The core research question is:

> What evidence is needed before a RAG paper can claim that a retrieval method
> improves faithfulness rather than only changing local retrieval scores or a
> particular metric?

## Contributions

The current evidence supports modest claims:

- A reusable audit protocol, called `ControlledRAG` in the NeurIPS-style draft,
  for separating matched similarity, scale, metric choice, threshold transfer,
  coherent-vs-random noise, human calibration, and baseline cost.
- A generator-free context statistic, CCS, that remains useful as a diagnostic,
  but is not presented as a proven causal mechanism.
- A fixed-generation multi-metric result showing materially different
  faithfulness magnitudes across DeBERTa, a second NLI scorer, and a local
  RAGAS-style judge.
- A 99-item two-rater human-calibration sample showing high inter-rater
  agreement but only weak-to-moderate alignment with automatic scorers.
- A cost-aware comparison of CRAG, HCPC-v2, RAPTOR-2L, and a harness-mismatched
  Self-RAG row, with no single method dominating across datasets and costs.

Do not describe the project as proving that CCS causes faithfulness, that
HCPC-v2 solves hallucination, or that the original refinement paradox is large
and stable at scale.

## Repository Layout

```text
.
├── README.md
├── PROJECT_HISTORY.md
├── AGENTS.md
├── CLAUDE.md
├── src/                         # reusable RAG, retrieval, scoring, and metric code
├── experiments/                 # experiment runners and revision fix scripts
├── scripts/                     # packaging, plotting, lint, and release helpers
├── notebooks/                   # Colab/Kaggle notebooks
├── data/                        # curated input data and revision raw outputs
├── results/                     # experiment outputs and paper tables
├── docs/                        # project, revision, submission, and audit docs
├── papers/
│   ├── neurips/                 # active NeurIPS-style submission
│   └── arxiv_longform/          # active long-form / arXiv-style draft
├── submission_packages/         # OpenReview and local submission package material
├── artifacts/                   # generated staging snapshots and non-source outputs
├── archive/                     # preserved legacy papers, zips, logs, and local DBs
├── release/                     # benchmark release bundles used by existing scripts
├── pip-package/                 # standalone context-coherence package
├── integrations/                # LangChain integration
├── space/ and leaderboard/      # HuggingFace Space and leaderboard apps
└── tests/                       # lightweight tests
```

Local Chroma vector stores have been moved out of the root into
`archive/legacy_chroma_dbs/` and are ignored by git. Future runs write new
stores under `artifacts/generated/chroma_db*`.

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -v --tb=short
```

Most experiment runners require local models through Ollama and may take from
minutes to hours. Do not run model experiments as part of routine cleanup or
documentation edits.

## Optional Model Setup

For local reproduction with Ollama:

```bash
ollama pull mistral
ollama pull llama3
bash scripts/setup_qwen.sh
```

The project operates under a zero-dollar research constraint unless the owner
explicitly approves otherwise. Existing wrappers for paid or quota-bound
providers are preserved, but the revision evidence should not be regenerated
with paid APIs unless the constraint changes.

## Reproducing Experiments

The full revision runbook is in `docs/revision/runbook.md`; the consolidated
revision book is `docs/revision/README.md`.

Representative commands:

```bash
# Fast checks
python -m pytest tests/ -v --tb=short
python scripts/lint_paper.py

# Revision examples
python experiments/fix_01_causal_matched_pairs.py --help
python experiments/fix_03_multimetric_faithfulness.py --help
python experiments/fix_06_baseline_h2h_pareto.py --help

# Figure/table rebuilds that do not require generator calls
python experiments/build_headline_figure.py
python experiments/build_ccs_calibration.py
python scripts/plot_cost_pareto.py
```

Long-running experiments are checkpointed by script-specific outputs in
`data/revision/`, `results/revision/`, and `artifacts/generated/chroma_db*`.

## Building Papers

Active paper locations:

- NeurIPS-style submission: `papers/neurips/`
- arXiv / long-form draft: `papers/arxiv_longform/`

Build commands:

```bash
make paper-neurips
make paper-longform
```

Direct commands:

```bash
cd papers/neurips
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

cd ../arxiv_longform
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The NeurIPS folder also has `supplement.tex`. The long-form paper keeps its
appendix inside `main.tex`.

## Data and Results

- `data/adversarial/`: adversarial/coherence cases and rejected cases.
- `data/revision/fix_*/`: per-fix raw or intermediate revision data.
- `results/revision/fix_*/`: per-fix aggregated results used by the current
  submission.
- `results/paper_tables/`: generated Markdown tables for paper assembly.
- `release/context_coherence_bench_v1/` and `release/context_coherence_bench_v2/`:
  benchmark release bundles used by the packaging scripts.

The released public artifact line includes:

- Python package: `context-coherence`
- HuggingFace dataset: `saketmgnt/context-coherence-bench`
- Zenodo DOI: `10.5281/zenodo.19757291`
- LangChain integration: `integrations/langchain/`
- HuggingFace Space app source: `space/`

Double-blind submission text withholds author-identifying links until
camera-ready.

## Submission Status

As of the 2026-04-29 cleanup:

- `papers/neurips/` is the active 10-page NeurIPS-style audit submission.
- `papers/neurips/SUBMISSION_CHECKLIST.md`, `CLAIMS_AUDIT.md`, and
  `SOURCE_TRACE.md` are the submission-specific ledgers.
- `papers/arxiv_longform/` is the active long-form paper source.
- Older paper variants are preserved in `archive/legacy_papers/` and
  `archive/legacy_duplicates/`.
- Local Chroma stores and old logs are preserved locally under `archive/` but
  ignored by git.

## Guidance for Future AI Agents

Read these in order:

1. `PROJECT_HISTORY.md`
2. `README.md`
3. `papers/neurips/README.md`
4. `papers/arxiv_longform/README.md`
5. `docs/revision/README.md`

Rules for future edits:

- do not delete `archive/`;
- do not restore strong causal or dominance claims unless new evidence supports
  them;
- use `papers/neurips/` and `papers/arxiv_longform/` as the only active paper
  folders;
- update `PROJECT_HISTORY.md` after major paper, experiment, or structure
  changes;
- keep generated vector stores and logs out of git.

## Known Limitations and TODOs

- The official NeurIPS 2026 style file and checklist are not yet included.
- Fix 7 remains budget-blocked under the zero-dollar constraint.
- The matched-similarity CCS test was null and must remain visible in the
  narrative.
- The scaled DeBERTa effect is small and seed-sensitive.
- Human calibration has 99 adjudicated examples, not a large crowdsourced study.
- Some legacy docs intentionally retain historical terminology; use
  `PROJECT_HISTORY.md` for the current interpretation.

## Citation

No final venue citation exists yet. Use the placeholder below only for internal
drafting:

```bibtex
@misc{maganti2026controlledrag,
  title  = {When Retrieval Quality Decouples from Faithfulness: A Pre-Registered Audit of RAG Evaluation},
  author = {Maganti, Saket},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Contact information appears in de-anonymized long-form sources and release
metadata. The active NeurIPS submission remains double-blind.
