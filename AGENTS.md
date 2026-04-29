# AGENTS.md

This repository has been reorganized. Future agent sessions should start here:

1. Read `PROJECT_HISTORY.md`.
2. Read `README.md`.
3. For the active NeurIPS-style paper, read `papers/neurips/README.md`,
   `papers/neurips/CLAIMS_AUDIT.md`, and `papers/neurips/SOURCE_TRACE.md`.
4. For the long-form paper, read `papers/arxiv_longform/README.md`.
5. For senior-reviewer revision evidence, read `docs/revision/README.md`.

Current active paper folders:

- `papers/neurips/`: 10-page double-blind NeurIPS-style audit submission.
- `papers/arxiv_longform/`: long-form / arXiv-style manuscript.

Repository rules:

- Do not delete `archive/`; it preserves legacy paper versions, old zips, old logs, and local Chroma DBs.
- Do not stage local Chroma DBs, virtualenvs, caches, temporary LaTeX files, logs, or accidental zip packages.
- New local vector stores should be generated under `artifacts/generated/chroma_db*`.
- Do not reintroduce unsupported claims that CCS causes faithfulness, that HCPC-v2 solves hallucination, or that the original DeBERTa refinement paradox is large and stable at scale.
- Preserve the Fix 1 null result and Fix 2 scale collapse in any paper edits.
- Update `PROJECT_HISTORY.md` after major changes to experiments, paper framing, submission state, or repository structure.

Operational constraints:

- Default to zero-dollar mode unless the owner explicitly approves paid compute or APIs.
- Treat `docs/revision/README.md` and `papers/neurips/SOURCE_TRACE.md` as evidence ledgers.
- Prefer `make tests`, `python scripts/lint_paper.py`, and paper build commands for validation.
