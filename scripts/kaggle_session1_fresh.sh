#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Saket-Maganti/rag-hallucination-detection.git}"
REPO_DIR="${REPO_DIR:-/kaggle/working/rag-hallucination-detection}"
MODEL="${MODEL:-mistral}"
OLLAMA_LOG="${OLLAMA_LOG:-/kaggle/working/ollama.log}"
export OLLAMA_LOG

START_TS=$(date +%s)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [session1] $*"; }
section() { echo; echo "[$(ts)] ===== $* ====="; }

row_count() {
  local path="$1"
  if [ -s "$path" ]; then
    python - "$path" <<'PYROW'
import pandas as pd, sys
try:
    print(len(pd.read_csv(sys.argv[1])))
except Exception:
    print("?")
PYROW
  else
    echo 0
  fi
}

run_with_heartbeat() {
  local label="$1"
  local csv_path="$2"
  local expected_rows="$3"
  shift 3
  local start
  start=$(date +%s)
  section "$label"
  "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    local now elapsed rows pct
    now=$(date +%s)
    elapsed=$((now - start))
    rows=$(row_count "$csv_path")
    pct=""
    if [ "$expected_rows" != "0" ] && [ "$rows" != "?" ]; then
      pct=" ($(( rows * 100 / expected_rows ))%)"
    fi
    log "$label heartbeat: elapsed=${elapsed}s rows=${rows}/${expected_rows}${pct}"
    tail -n 5 "$OLLAMA_LOG" 2>/dev/null | sed 's/^/[ollama-tail] /' || true
  done
  wait "$pid"
  log "$label finished in $(($(date +%s) - start))s"
}

cd /kaggle/working
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --branch main "${REPO_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" fetch origin main
  git -C "${REPO_DIR}" checkout main
  git -C "${REPO_DIR}" pull --ff-only origin main
fi

cd "${REPO_DIR}"
mkdir -p logs/revision

section "Environment"
log "commit: $(git rev-parse --short HEAD)"
nvidia-smi || true
python --version

section "System packages"
apt-get update -y
apt-get install -y zstd curl git zip

section "Python dependencies"
python -m pip install -U pip setuptools wheel
python -m pip install -e pip-package
python -m pip install \
  pandas numpy scipy scikit-learn matplotlib seaborn tabulate tqdm \
  datasets sentence-transformers transformers torch accelerate \
  langchain langchain-community langchain-core langchain-text-splitters langchain-ollama \
  chromadb

section "Ollama guard"
bash scripts/kaggle_ollama_guard.sh "${MODEL}"

section "Preflight"
python -m py_compile \
  experiments/fix_01_causal_matched_pairs.py \
  experiments/fix_05_coherence_preserving_noise.py \
  experiments/fix_11_raptor_full_table.py \
  experiments/revision_utils.py \
  src/ragas_scorer.py src/vectara_hem_scorer.py

if [ ! -s data/revision/fix_01/matched_pairs.csv ]; then
  section "Fix 1 construction"
  PYTHONUNBUFFERED=1 python -u experiments/fix_01_causal_matched_pairs.py \
    --stage construct \
    --dataset squad \
    --n_target 200 \
    --seed 42 \
    --max_contexts 400 \
    --candidate_limit 400 \
    --run_tag primary_n200 \
    2>&1 | tee logs/revision/fix_01_construct_session1.log
fi

bash scripts/kaggle_ollama_guard.sh "${MODEL}"
run_with_heartbeat "Fix 1 generation" "data/revision/fix_01/per_query.csv" 400 \
  bash -lc 'PYTHONUNBUFFERED=1 python -u experiments/fix_01_causal_matched_pairs.py \
    --stage generate \
    --backend ollama \
    --model "'"${MODEL}"'" \
    --resume \
    --save_every 2 \
    --progress_every 10 \
    2>&1 | tee logs/revision/fix_01_generate_session1.log'

section "Fix 1 analysis"
PYTHONUNBUFFERED=1 python -u experiments/fix_01_causal_matched_pairs.py \
  --stage analyze \
  2>&1 | tee logs/revision/fix_01_analyze_session1.log

bash scripts/kaggle_ollama_guard.sh "${MODEL}"
run_with_heartbeat "Fix 5 noise slope" "data/revision/fix_05/per_query_partial.csv" 1600 \
  bash -lc 'PYTHONUNBUFFERED=1 python -u experiments/fix_05_coherence_preserving_noise.py \
    --n 200 \
    --seed 42 \
    --backend ollama \
    --model "'"${MODEL}"'" \
    --max_contexts 300 \
    --n_noise 1 2 3 \
    --save_every 5 \
    2>&1 | tee logs/revision/fix_05_noise_slope_session1.log'

bash scripts/kaggle_ollama_guard.sh "${MODEL}"
run_with_heartbeat "Fix 11 RAPTOR" "data/revision/fix_11/per_query.csv" 300 \
  bash -lc 'PYTHONUNBUFFERED=1 python -u experiments/fix_11_raptor_full_table.py \
    --datasets squad pubmedqa hotpotqa \
    --n 100 \
    --backend ollama \
    --model "'"${MODEL}"'" \
    --max_contexts 150 \
    --raptor_clusters 6 \
    2>&1 | tee logs/revision/fix_11_raptor_full_table_session1.log'

section "Packaging outputs"
rm -f /kaggle/working/revision_session1_outputs.zip
zip -r /kaggle/working/revision_session1_outputs.zip \
  data/revision/fix_01 results/revision/fix_01 \
  data/revision/fix_05 results/revision/fix_05 \
  data/revision/fix_11 results/revision/fix_11 \
  logs/revision \
  CODEX.md REVISION_SUMMARY.md REVISION_RUNBOOK.md

ls -lh /kaggle/working/revision_session1_outputs.zip
log "done total_elapsed=$(($(date +%s) - START_TS))s"

