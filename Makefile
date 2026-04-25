# Makefile — single-source recipe for the project.
# Usage:
#   make help                — list targets
#   make figures             — regenerate every Phase 3/4 figure (~3 min)
#   make tests               — run the test suite
#   make paper               — pdflatex × bibtex × pdflatex × pdflatex
#   make all                 — figures + tests + paper
#   make topk                — re-run top-k ablation (Ollama, ~1 hr)
#   make zenodo              — upload bundle to Zenodo (needs ZENODO_TOKEN)
#   make hf-dataset          — push ContextCoherenceBench to HF (needs HF_TOKEN)
#   make hf-space            — re-deploy the Gradio Space (needs HF_TOKEN)
#   make pip-build           — build the standalone context-coherence wheel
#   make pip-test            — pytest the pip package
#   make clean               — remove caches + temporary build files
#   make docker-build        — build the reproducibility Docker image
#   make docker-run          — run a smoke check inside the container

PYTHON      ?= python3
RAGPAPER    := ragpaper
RESULTS_DIR := results
LOGS_DIR    := logs

.PHONY: help all figures tests paper topk zenodo hf-dataset hf-space \
        pip-build pip-test pip-publish clean docker-build docker-run

help:
	@grep -E '^# ' $(MAKEFILE_LIST) | grep -E 'make ' | sed 's/^# *//'
	@echo ""
	@echo "Examples:"
	@echo "  make figures && make paper          # rebuild figures + PDF"
	@echo "  make tests                          # ~20s; CI uses this target"
	@echo "  ZENODO_TOKEN=... make zenodo        # mint a versioned DOI"

# ── Test suite (fast, mocked) ────────────────────────────────────────

tests:
	$(PYTHON) -m pytest tests/ -v --tb=short

pip-test:
	cd pip-package && $(PYTHON) -m pytest tests/ -v --tb=short

# ── Figure regeneration (no Ollama needed) ───────────────────────────

figures:
	@echo "[make] regenerating Phase 3 + Phase 4 figures"
	$(PYTHON) experiments/build_headline_figure.py
	$(PYTHON) experiments/build_ccs_calibration.py
	$(PYTHON) experiments/build_qualitative_example.py --top 5
	$(PYTHON) experiments/build_disentanglement_figure.py
	$(PYTHON) experiments/build_coherence_heatmap.py
	$(PYTHON) experiments/build_embedding_clusters.py
	@echo "[make] figures done."

# ── Paper compile (LaTeX) ────────────────────────────────────────────

paper:
	cd $(RAGPAPER) && pdflatex -interaction=nonstopmode main.tex && \
		bibtex main && \
		pdflatex -interaction=nonstopmode main.tex && \
		pdflatex -interaction=nonstopmode main.tex
	@echo "[make] $(RAGPAPER)/main.pdf updated (longform, ~55 pages)."

paper-neurips:
	cd $(RAGPAPER) && pdflatex -interaction=nonstopmode main_neurips.tex && \
		bibtex main_neurips && \
		pdflatex -interaction=nonstopmode main_neurips.tex && \
		pdflatex -interaction=nonstopmode main_neurips.tex
	@echo "[make] $(RAGPAPER)/main_neurips.pdf updated (NeurIPS-tight, ~28 pages)."

papers: paper paper-neurips
	@echo "[make] both PDFs built:"
	@ls -lh $(RAGPAPER)/main.pdf $(RAGPAPER)/main_neurips.pdf

lint:
	$(PYTHON) scripts/lint_paper.py

# ── Top-k experiment (Ollama, ~1 hr) ─────────────────────────────────

topk:
	@mkdir -p $(LOGS_DIR)
	@curl -s -m 3 http://localhost:11434/api/version >/dev/null 2>&1 || \
		(echo "[make] Ollama not running. Start with: ollama serve" && exit 1)
	nohup $(PYTHON) -u experiments/run_topk_sensitivity.py \
		--k 2 3 5 10 \
		--datasets squad pubmedqa \
		--model mistral \
		--n_questions 30 \
		> $(LOGS_DIR)/topk_sensitivity.log 2>&1 &
	@echo "[make] top-k launched in background. tail -f $(LOGS_DIR)/topk_sensitivity.log"

topk-table:
	$(PYTHON) experiments/build_topk_table.py

# ── Zenodo / HF integrations ─────────────────────────────────────────

zenodo:
	@test -n "$(ZENODO_TOKEN)" || (echo "[make] export ZENODO_TOKEN" && exit 1)
	$(PYTHON) scripts/upload_to_zenodo.py

hf-dataset:
	$(PYTHON) scripts/push_to_hf_datasets.py --push

hf-space:
	$(PYTHON) scripts/prepare_hf_space.py --overwrite
	@echo "[make] space staged at space_deploy/. push manually to HF (see CLAUDE.md)."

# ── Standalone pip package ───────────────────────────────────────────

pip-build:
	cd pip-package && $(PYTHON) -m pip install --upgrade build && \
		$(PYTHON) -m build
	@echo "[make] wheels in pip-package/dist/"

pip-publish:
	@test -n "$(PYPI_TOKEN)" || (echo "[make] export PYPI_TOKEN" && exit 1)
	cd pip-package && $(PYTHON) -m twine upload \
		--username __token__ --password $(PYPI_TOKEN) dist/*

# ── Docker (reproducibility) ─────────────────────────────────────────

docker-build:
	docker build -t coherence-paradox:v2.0.0 .

docker-run:
	docker run --rm -it -v $(PWD):/workspace coherence-paradox:v2.0.0 \
		bash -c "cd /workspace && make tests"

# ── Aggregate ────────────────────────────────────────────────────────

all: figures tests paper

clean:
	rm -rf $(RAGPAPER)/main.aux $(RAGPAPER)/main.log $(RAGPAPER)/main.bbl \
		$(RAGPAPER)/main.blg $(RAGPAPER)/main.out $(RAGPAPER)/main.toc \
		$(RAGPAPER)/main.fls $(RAGPAPER)/main.fdb_latexmk
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
