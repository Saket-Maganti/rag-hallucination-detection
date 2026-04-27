#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-full_no_selfrag}"
MODEL="${MODEL:-mistral}"
REPO_DIR="${REPO_DIR:-/kaggle/working/rag-hallucination-detection}"
LOG_DIR="${LOG_DIR:-/kaggle/working/fix6_t4x2_logs}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/kaggle/working/ollama_models}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"

FIX6_DATASETS="${FIX6_DATASETS:-squad hotpotqa}"
FIX6_N="${FIX6_N:-200}"
FIX6_MAX_CONTEXTS="${FIX6_MAX_CONTEXTS:-250}"
FIX6_RAPTOR_CLUSTERS="${FIX6_RAPTOR_CLUSTERS:-6}"
FIX6_SAVE_EVERY="${FIX6_SAVE_EVERY:-10}"
FIX6_CUDA_VISIBLE_DEVICES="${FIX6_CUDA_VISIBLE_DEVICES:-1}"
SELFRAG_MODEL="${SELFRAG_MODEL:-selfrag/selfrag_llama2_7b}"
SELFRAG_QUANT="${SELFRAG_QUANT:-8bit}"
SELFRAG_SMOKE_N="${SELFRAG_SMOKE_N:-5}"
SELFRAG_SMOKE_MAX_CONTEXTS="${SELFRAG_SMOKE_MAX_CONTEXTS:-50}"

mkdir -p "${LOG_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [fix6-t4x2] $*"; }
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
  mkdir -p logs/revision data/revision/fix_06 results/revision/fix_06
}

install_deps() {
  section "Dependencies"
  apt-get update -y
  apt-get install -y zstd curl git zip procps
  python -m pip install -U pip setuptools wheel
  python -m pip install -e pip-package
  python -m pip install \
    pandas numpy scipy scikit-learn matplotlib seaborn tabulate tqdm \
    datasets sentence-transformers transformers torch accelerate bitsandbytes \
    langchain langchain-community langchain-core langchain-text-splitters langchain-ollama \
    chromadb
}

is_live() {
  local port="$1"
  curl -fsS "http://127.0.0.1:${port}/api/tags" >/dev/null 2>&1
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
  tail -n 160 "${log_file}" || true
  exit 1
}

pull_model() {
  section "Model"
  OLLAMA_HOST="127.0.0.1:11434" OLLAMA_MODELS="${OLLAMA_MODELS_DIR}" "${OLLAMA_BIN}" pull "${MODEL}"
  OLLAMA_HOST="127.0.0.1:11434" OLLAMA_MODELS="${OLLAMA_MODELS_DIR}" "${OLLAMA_BIN}" list
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
    experiments/fix_06_baseline_h2h_pareto.py \
    experiments/revision_utils.py \
    src/selfrag_wrapper.py
  curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
  log "preflight OK"
}

setup_stage() {
  repo_setup
  install_deps
  ensure_ollama_binary
  stop_ollama
  mkdir -p "${OLLAMA_MODELS_DIR}"
  start_server 0 11434
  pull_model
  preflight
}

selfrag_quant_args() {
  case "${SELFRAG_QUANT}" in
    8bit)
      echo "--selfrag_8bit"
      ;;
    4bit)
      echo "--selfrag_4bit"
      ;;
    none)
      echo ""
      ;;
    *)
      log "ERROR: SELFRAG_QUANT must be one of 8bit, 4bit, none"
      exit 2
      ;;
  esac
}

run_fix6() {
  local mode="$1"
  local n_value="$2"
  local max_contexts="$3"
  local log_file="$4"
  shift 4

  cd "${REPO_DIR}"
  section "Run Fix 6 ${mode}"
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES="${FIX6_CUDA_VISIBLE_DEVICES}" \
    OLLAMA_HOST="http://127.0.0.1:11434" \
    OLLAMA_BASE_URL="http://127.0.0.1:11434" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    python -u experiments/fix_06_baseline_h2h_pareto.py \
      --datasets ${FIX6_DATASETS} \
      --n "${n_value}" \
      --backend ollama \
      --model "${MODEL}" \
      --max_contexts "${max_contexts}" \
      --raptor_clusters "${FIX6_RAPTOR_CLUSTERS}" \
      --save_every "${FIX6_SAVE_EVERY}" \
      "$@" \
      2>&1 | tee "${log_file}"
}

run_no_selfrag() {
  run_fix6 "without Self-RAG" "${FIX6_N}" "${FIX6_MAX_CONTEXTS}" \
    "logs/revision/fix_06_baselines_no_selfrag.log"
}

run_selfrag_smoke() {
  local qargs
  qargs="$(selfrag_quant_args)"
  # shellcheck disable=SC2086
  run_fix6 "Self-RAG smoke" "${SELFRAG_SMOKE_N}" "${SELFRAG_SMOKE_MAX_CONTEXTS}" \
    "logs/revision/fix_06_selfrag_smoke.log" \
    --include_selfrag --selfrag_model "${SELFRAG_MODEL}" ${qargs}
}

run_selfrag() {
  local qargs
  qargs="$(selfrag_quant_args)"
  # shellcheck disable=SC2086
  run_fix6 "with Self-RAG" "${FIX6_N}" "${FIX6_MAX_CONTEXTS}" \
    "logs/revision/fix_06_baselines_with_selfrag.log" \
    --include_selfrag --selfrag_model "${SELFRAG_MODEL}" ${qargs}
}

heartbeat_status() {
  local final squad hotpot
  final=$(row_count data/revision/fix_06/per_query.csv)
  squad=$(row_count data/revision/fix_06/per_query_squad_partial.csv)
  hotpot=$(row_count data/revision/fix_06/per_query_hotpotqa_partial.csv)
  log "heartbeat fix6_final=${final} fix6_partials=squad:${squad},hotpot:${hotpot}"
  tail -n 5 logs/revision/fix_06_baselines_no_selfrag.log 2>/dev/null | sed 's/^/[fix6-no-selfrag] /' || true
  tail -n 5 logs/revision/fix_06_baselines_with_selfrag.log 2>/dev/null | sed 's/^/[fix6-selfrag] /' || true
  tail -n 3 "${LOG_DIR}/ollama_gpu0.log" 2>/dev/null | sed 's/^/[ollama0] /' || true
}

run_stage_with_watchdog() {
  local runner="$1"
  repo_setup
  ensure_ollama_binary
  start_server 0 11434
  preflight

  watch_server 0 11434 &
  watch0=$!
  trap 'kill "${watch0}" >/dev/null 2>&1 || true' EXIT

  "${runner}" &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${HEARTBEAT_SECONDS}"
    heartbeat_status
  done

  set +e
  wait "${pid}"
  rc=$?
  set -e
  kill "${watch0}" >/dev/null 2>&1 || true
  trap - EXIT

  if [ "${rc}" -ne 0 ]; then
    log "ERROR: Fix 6 ${runner} rc=${rc}"
    tail -n 160 logs/revision/fix_06_*.log 2>/dev/null || true
    exit "${rc}"
  fi
  heartbeat_status
  log "Fix 6 ${runner} completed"
}

status_stage() {
  repo_setup
  section "Status"
  heartbeat_status
  find data/revision/fix_06 results/revision/fix_06 logs/revision -maxdepth 2 -type f \
    \( -path '*fix_06*' -o -path '*fix6*' -o -name 'per_query*.csv' -o -name 'h2h_summary.csv' \) \
    -print | sort
}

package_stage() {
  repo_setup
  section "Package"
  rm -f /kaggle/working/fix6_t4x2_outputs.zip /kaggle/working/AAA_FIX6_T4X2_OUTPUTS.zip
  zip -r /kaggle/working/fix6_t4x2_outputs.zip \
    data/revision/fix_06 \
    results/revision/fix_06 \
    logs/revision/fix_06_* \
    "${LOG_DIR}" \
    CODEX.md REVISION_RUNBOOK.md REVISION_SUMMARY.md
  cp /kaggle/working/fix6_t4x2_outputs.zip /kaggle/working/AAA_FIX6_T4X2_OUTPUTS.zip
  ls -lh /kaggle/working/fix6_t4x2_outputs.zip /kaggle/working/AAA_FIX6_T4X2_OUTPUTS.zip
}

case "${STAGE}" in
  setup)
    setup_stage
    ;;
  no_selfrag)
    run_stage_with_watchdog run_no_selfrag
    ;;
  smoke_selfrag)
    run_stage_with_watchdog run_selfrag_smoke
    ;;
  selfrag)
    run_stage_with_watchdog run_selfrag
    ;;
  status)
    status_stage
    ;;
  package)
    package_stage
    ;;
  full_no_selfrag)
    setup_stage
    run_stage_with_watchdog run_no_selfrag
    package_stage
    ;;
  full_selfrag)
    setup_stage
    run_stage_with_watchdog run_selfrag_smoke
    run_stage_with_watchdog run_selfrag
    package_stage
    ;;
  *)
    echo "Usage: $0 {setup|no_selfrag|smoke_selfrag|selfrag|status|package|full_no_selfrag|full_selfrag}" >&2
    exit 2
    ;;
esac
