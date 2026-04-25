# Coherence Paradox in RAG — reproducibility image (v2.0.0)
# ===========================================================
# A self-contained image that can run the test suite, regenerate every
# Phase 3+4 figure, and recompile the paper (via the texlive layer).
# Excludes Ollama: the long Ollama-backed runs (top-k, frontier-scale)
# are intentionally out-of-scope for the container — those use the
# host's Ollama via the network.
#
# Build:    docker build -t coherence-paradox:v2.0.0 .
# Test:     docker run --rm coherence-paradox:v2.0.0 make tests
# Figures:  docker run --rm -v $PWD/results:/app/results \
#                          -v $PWD/ragpaper:/app/ragpaper \
#               coherence-paradox:v2.0.0 make figures
#
# Image size is ~1.5 GB (Python + numpy + sentence-transformers).
# To shrink: build with --target=slim for the minimal pip-package only.

# ─── stage 1: minimal package-only target (≈ 200 MB) ────────────────
FROM python:3.11-slim AS slim

WORKDIR /app
COPY pip-package/ ./pip-package/
RUN pip install --no-cache-dir ./pip-package

CMD ["python3", "-c", "import context_coherence; print('OK', context_coherence.__version__)"]

# ─── stage 2: full reproducibility image (default target) ───────────
FROM python:3.11-slim AS full

# System deps for matplotlib + bibtex + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: install deps separately so code-only changes don't bust cache
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-cov

# Layer 2: standalone pip package (small, install-once)
COPY pip-package/ ./pip-package/
RUN pip install --no-cache-dir -e ./pip-package

# Layer 3: project code + scripts (the layer that actually changes)
COPY src/         ./src/
COPY experiments/ ./experiments/
COPY scripts/     ./scripts/
COPY tests/       ./tests/
COPY ragpaper/    ./ragpaper/
COPY data/        ./data/
COPY results/     ./results/
COPY release/     ./release/
COPY Makefile     CLAUDE.md  README.md  ./

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["make", "tests"]
