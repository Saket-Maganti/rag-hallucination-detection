# CLAUDE.md

Start with `PROJECT_HISTORY.md`. It is the canonical project-history and AI
handoff file for this repository.

Active papers:

- `papers/neurips/`: current double-blind NeurIPS-style audit submission.
- `papers/arxiv_longform/`: current long-form / arXiv-style manuscript.

Important supporting docs:

- `README.md`: GitHub-facing overview and quickstart.
- `papers/neurips/CLAIMS_AUDIT.md`: allowed and forbidden claims.
- `papers/neurips/SOURCE_TRACE.md`: claim-to-source trace for the submission.
- `docs/revision/README.md`: senior-reviewer revision book.
- `docs/project/REORGANIZATION_REPORT.md`: cleanup and validation record.

Rules for Claude Code sessions:

- Do not delete `archive/`.
- Do not stage local Chroma DBs, logs, virtualenvs, caches, temporary LaTeX files, or accidental zip packages.
- Generated vector stores now belong under `artifacts/generated/chroma_db*`.
- Keep causal and dominance language conservative unless new evidence supports a stronger claim.
- Preserve null results and reviewer caveats.
- Update `PROJECT_HISTORY.md` after major paper, experiment, release, or structure changes.

Useful commands:

```bash
python -m pytest tests/ -v --tb=short
python scripts/lint_paper.py
make paper-neurips
make paper-longform
```
