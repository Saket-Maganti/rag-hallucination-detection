#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-full}"
MODEL="${MODEL:-mistral}"
REPO_DIR="${REPO_DIR:-/kaggle/working/rag-hallucination-detection}"
LOG_DIR="${LOG_DIR:-/kaggle/working/fix5_11_t4x2_logs}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/kaggle/working/ollama_models}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"

FIX5_N="${FIX5_N:-200}"
FIX5_MAX_CONTEXTS="${FIX5_MAX_CONTEXTS:-300}"
FIX5_SAVE_EVERY="${FIX5_SAVE_EVERY:-5}"

FIX11_DATASETS="${FIX11_DATASETS:-squad pubmedqa hotpotqa}"
FIX11_N="${FIX11_N:-100}"
FIX11_MAX_CONTEXTS="${FIX11_MAX_CONTEXTS:-150}"
FIX11_SAVE_EVERY="${FIX11_SAVE_EVERY:-5}"
FIX11_RAPTOR_CLUSTERS="${FIX11_RAPTOR_CLUSTERS:-6}"

mkdir -p "${LOG_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [fix5-11-t4x2] $*"; }
section() { echo; log "===== $* ====="; }

row_count() {
  local path="$1"
  if [ -s "${path}" ]; then
    python - "$path" <<'PYROW'
import csv
import sys

try:
    with open(sys.argv[1], newline="") as f:
        rows = sum(1 for _ in csv.reader(f))
    print(max(0, rows - 1))
except Exception:
    print("?")
PYROW
  else
    echo 0
  fi
}

repo_setup() {
  cd /kaggle/working
  if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone --progress --branch main https://github.com/Saket-Maganti/rag-hallucination-detection.git "${REPO_DIR}"
  else
    git -C "${REPO_DIR}" fetch origin main
    git -C "${REPO_DIR}" checkout main
    git -C "${REPO_DIR}" pull --ff-only origin main
  fi
  cd "${REPO_DIR}"
  mkdir -p logs/revision data/revision/fix_05 data/revision/fix_11 results/revision/fix_05 results/revision/fix_11
}

install_deps() {
  section "Dependencies"
  apt-get update -y
  apt-get install -y zstd curl git zip procps
  python -m pip install -U pip setuptools wheel
  python -m pip install -e pip-package
  python -m pip install \
    pandas numpy scipy scikit-learn matplotlib seaborn tabulate tqdm \
    datasets sentence-transformers transformers torch accelerate \
    langchain langchain-community langchain-core langchain-text-splitters langchain-ollama \
    chromadb
}

is_live() {
  local port="$1"
  curl -fsS "http://127.0.0.1:${port}/api/tags" >/dev/null 2>&1
}

start_server() {
  local gpu="$1"
  local port="$2"
  local log_file="${LOG_DIR}/ollama_gpu${gpu}.log"

  if is_live "${port}"; then
    log "ollama already live gpu=${gpu} port=${port}"
    return
  fi

  log "starting ollama gpu=${gpu} port=${port} log=${log_file}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OLLAMA_HOST="127.0.0.1:${port}" \
    OLLAMA_MODELS="${OLLAMA_MODELS_DIR}" \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_NUM_PARALLEL=1 \
    "${OLLAMA_BIN}" serve >"${log_file}" 2>&1 &
  echo "$!" >"${LOG_DIR}/ollama_gpu${gpu}.pid"

  for _ in $(seq 1 90); do
    if is_live "${port}"; then
      log "ollama live gpu=${gpu} port=${port}"
      return
    fi
    sleep 2
  done

  log "ERROR: ollama failed gpu=${gpu} port=${port}"
  tail -n 120 "${log_file}" || true
  exit 1
}

ensure_ollama_binary() {
  section "Ollama binary"
  if ! command -v ollama >/dev/null 2>&1; then
    log "installing ollama"
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    log "ERROR: ollama is still not on PATH after install"
    log "PATH=${PATH}"
    ls -lh /usr/local/bin/ollama /usr/bin/ollama 2>/dev/null || true
    exit 1
  fi

  OLLAMA_BIN="$(command -v ollama)"
  export OLLAMA_BIN
  log "ollama binary: ${OLLAMA_BIN}"
}

stop_ollama() {
  section "Reset Ollama"
  pkill -x ollama >/dev/null 2>&1 || true
  sleep 5
}

pull_model() {
  section "Model"
  OLLAMA_HOST="127.0.0.1:11434" OLLAMA_MODELS="${OLLAMA_MODELS_DIR}" "${OLLAMA_BIN}" pull "${MODEL}"
  OLLAMA_HOST="127.0.0.1:11434" OLLAMA_MODELS="${OLLAMA_MODELS_DIR}" "${OLLAMA_BIN}" list
  OLLAMA_HOST="127.0.0.1:11435" OLLAMA_MODELS="${OLLAMA_MODELS_DIR}" "${OLLAMA_BIN}" list || true
}

watch_server() {
  local gpu="$1"
  local port="$2"
  local fails=0
  while true; do
    sleep 20
    if is_live "${port}"; then
      fails=0
      continue
    fi
    fails=$((fails + 1))
    log "watchdog: gpu=${gpu} port=${port} API miss ${fails}/3"
    if [ "${fails}" -ge 3 ]; then
      local pid_file="${LOG_DIR}/ollama_gpu${gpu}.pid"
      if [ -s "${pid_file}" ]; then
        kill "$(cat "${pid_file}")" >/dev/null 2>&1 || true
      fi
      start_server "${gpu}" "${port}"
      fails=0
    fi
  done
}

preflight() {
  section "Preflight"
  git log -1 --oneline
  nvidia-smi || true
  python --version
  python -m py_compile \
    experiments/fix_05_coherence_preserving_noise.py \
    experiments/fix_11_raptor_full_table.py \
    experiments/revision_utils.py
  curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
  curl -fsS http://127.0.0.1:11435/api/tags >/dev/null
  log "preflight OK"
}

setup_stage() {
  repo_setup
  install_deps
  ensure_ollama_binary
  stop_ollama
  mkdir -p "${OLLAMA_MODELS_DIR}"
  start_server 0 11434
  start_server 1 11435
  pull_model
  preflight
}

run_fix5() {
  cd "${REPO_DIR}"
  section "Run Fix 5 on GPU0"
  env CUDA_VISIBLE_DEVICES=0 \
    OLLAMA_HOST="http://127.0.0.1:11434" \
    OLLAMA_BASE_URL="http://127.0.0.1:11434" \
    PYTHONUNBUFFERED=1 \
    python -u experiments/fix_05_coherence_preserving_noise.py \
      --n "${FIX5_N}" \
      --seed 42 \
      --backend ollama \
      --model "${MODEL}" \
      --max_contexts "${FIX5_MAX_CONTEXTS}" \
      --n_noise 1 2 3 \
      --save_every "${FIX5_SAVE_EVERY}" \
      2>&1 | tee logs/revision/fix_05_kaggle_t4x2.log
}

run_fix11() {
  cd "${REPO_DIR}"
  section "Run Fix 11 on GPU1"
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES=1 \
    OLLAMA_HOST="http://127.0.0.1:11435" \
    OLLAMA_BASE_URL="http://127.0.0.1:11435" \
    PYTHONUNBUFFERED=1 \
    python -u experiments/fix_11_raptor_full_table.py \
      --datasets ${FIX11_DATASETS} \
      --n "${FIX11_N}" \
      --backend ollama \
      --model "${MODEL}" \
      --max_contexts "${FIX11_MAX_CONTEXTS}" \
      --raptor_clusters "${FIX11_RAPTOR_CLUSTERS}" \
      --save_every "${FIX11_SAVE_EVERY}" \
      2>&1 | tee logs/revision/fix_11_kaggle_t4x2.log
}

parallel_stage() {
  repo_setup
  ensure_ollama_binary
  start_server 0 11434
  start_server 1 11435
  preflight

  watch_server 0 11434 &
  watch0=$!
  watch_server 1 11435 &
  watch1=$!
  trap 'kill "${watch0}" "${watch1}" >/dev/null 2>&1 || true' EXIT

  run_fix5 &
  pid5=$!
  run_fix11 &
  pid11=$!

  while kill -0 "${pid5}" 2>/dev/null || kill -0 "${pid11}" 2>/dev/null; do
    sleep "${HEARTBEAT_SECONDS}"
    f5_rows=$(row_count "data/revision/fix_05/per_query.csv")
    f5_partial=$(row_count "data/revision/fix_05/per_query_partial.csv")
    f11_rows=$(row_count "data/revision/fix_11/per_query.csv")
    f11_squad=$(row_count "data/revision/fix_11/per_query_squad_partial.csv")
    f11_pubmed=$(row_count "data/revision/fix_11/per_query_pubmedqa_partial.csv")
    f11_hotpot=$(row_count "data/revision/fix_11/per_query_hotpotqa_partial.csv")
    log "heartbeat fix5_final=${f5_rows} fix5_partial=${f5_partial} fix11_final=${f11_rows} fix11_partials=squad:${f11_squad},pubmed:${f11_pubmed},hotpot:${f11_hotpot}"
    tail -n 3 "${LOG_DIR}/ollama_gpu0.log" 2>/dev/null | sed 's/^/[ollama0] /' || true
    tail -n 3 "${LOG_DIR}/ollama_gpu1.log" 2>/dev/null | sed 's/^/[ollama1] /' || true
  done

  set +e
  wait "${pid5}"
  rc5=$?
  wait "${pid11}"
  rc11=$?
  set -e

  kill "${watch0}" "${watch1}" >/dev/null 2>&1 || true
  trap - EXIT

  if [ "${rc5}" -ne 0 ] || [ "${rc11}" -ne 0 ]; then
    log "ERROR: Fix 5 rc=${rc5}, Fix 11 rc=${rc11}"
    tail -n 80 logs/revision/fix_05_kaggle_t4x2.log || true
    tail -n 80 logs/revision/fix_11_kaggle_t4x2.log || true
    exit 1
  fi

  log "Fix 5 and Fix 11 completed"
}

status_stage() {
  repo_setup
  section "Status"
  log "Fix5 final rows: $(row_count data/revision/fix_05/per_query.csv)"
  log "Fix5 partial rows: $(row_count data/revision/fix_05/per_query_partial.csv)"
  log "Fix11 final rows: $(row_count data/revision/fix_11/per_query.csv)"
  log "Fix11 squad partial rows: $(row_count data/revision/fix_11/per_query_squad_partial.csv)"
  log "Fix11 pubmed partial rows: $(row_count data/revision/fix_11/per_query_pubmedqa_partial.csv)"
  log "Fix11 hotpot partial rows: $(row_count data/revision/fix_11/per_query_hotpotqa_partial.csv)"
  find data/revision/fix_05 data/revision/fix_11 results/revision/fix_05 results/revision/fix_11 -maxdepth 2 -type f -print | sort
}

package_stage() {
  repo_setup
  section "Package"
  rm -f /kaggle/working/fix5_11_t4x2_outputs.zip
  zip -r /kaggle/working/fix5_11_t4x2_outputs.zip \
    data/revision/fix_05 data/revision/fix_11 \
    results/revision/fix_05 results/revision/fix_11 \
    logs/revision "${LOG_DIR}" \
    CODEX.md REVISION_RUNBOOK.md REVISION_SUMMARY.md
  ls -lh /kaggle/working/fix5_11_t4x2_outputs.zip
}

case "${STAGE}" in
  setup)
    setup_stage
    ;;
  fix5)
    repo_setup
    run_fix5
    ;;
  fix11)
    repo_setup
    run_fix11
    ;;
  parallel)
    parallel_stage
    ;;
  status)
    status_stage
    ;;
  package)
    package_stage
    ;;
  full)
    setup_stage
    parallel_stage
    package_stage
    ;;
  *)
    echo "Usage: $0 {setup|parallel|fix5|fix11|status|package|full}" >&2
    exit 2
    ;;
esac
