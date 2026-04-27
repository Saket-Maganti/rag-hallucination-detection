#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-full}"
MODEL="${MODEL:-mistral}"
REPO_DIR="${REPO_DIR:-/kaggle/working/rag-hallucination-detection}"
LOG_DIR="${LOG_DIR:-/kaggle/working/fix3_4_t4x2_logs}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/kaggle/working/ollama_models}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"

FIX3_INPUT="${FIX3_INPUT:-data/revision/fix_02/per_query.csv}"
FIX3_SAVE_EVERY="${FIX3_SAVE_EVERY:-25}"
FIX3_SECOND_NLI_MODEL="${FIX3_SECOND_NLI_MODEL:-vectara/hallucination_evaluation_model}"
FIX3_LIMIT="${FIX3_LIMIT:-}"

FIX4_DATASETS="${FIX4_DATASETS:-squad pubmedqa hotpotqa naturalqs triviaqa}"
FIX4_TAUS="${FIX4_TAUS:-0.30 0.40 0.50 0.60 0.70}"
FIX4_N="${FIX4_N:-100}"
FIX4_MAX_CONTEXTS="${FIX4_MAX_CONTEXTS:-150}"
FIX4_SAVE_EVERY="${FIX4_SAVE_EVERY:-5}"
FIX4_EXPECTED_MIN_ROWS="${FIX4_EXPECTED_MIN_ROWS:-1}"

mkdir -p "${LOG_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [fix3-4-t4x2] $*"; }
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

sum_rows() {
  python - "$@" <<'PYSUM'
import csv
import sys
from pathlib import Path

total = 0
for arg in sys.argv[1:]:
    path = Path(arg)
    if not path.exists() or path.stat().st_size == 0:
        continue
    try:
        with path.open(newline="") as f:
            total += max(0, sum(1 for _ in csv.reader(f)) - 1)
    except Exception:
        pass
print(total)
PYSUM
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
  mkdir -p logs/revision data/revision/fix_02 data/revision/fix_03 data/revision/fix_04 \
    results/revision/fix_02 results/revision/fix_03 results/revision/fix_04
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

import_fix2_if_needed() {
  cd "${REPO_DIR}"
  if [ -s data/revision/fix_02/per_query.csv ]; then
    log "Fix 2 input already present: $(row_count data/revision/fix_02/per_query.csv) rows"
    return
  fi

  section "Import Fix 2 outputs"
  local candidates=(
    "/kaggle/working/AAA_FIX2_T4X2_OUTPUTS.zip"
    "/kaggle/working/fix2_t4x2_outputs.zip"
    "/kaggle/working/rag-hallucination-detection/AAA_FIX2_T4X2_OUTPUTS.zip"
    "/kaggle/working/rag-hallucination-detection/fix2_t4x2_outputs.zip"
  )

  local zip_path=""
  for path in "${candidates[@]}"; do
    if [ -s "${path}" ]; then
      zip_path="${path}"
      break
    fi
  done
  if [ -z "${zip_path}" ]; then
    zip_path="$(find /kaggle/input /kaggle/working -maxdepth 5 \( -name '*FIX2*.zip' -o -name '*fix2*.zip' \) 2>/dev/null | head -n 1 || true)"
  fi

  if [ -z "${zip_path}" ] || [ ! -s "${zip_path}" ]; then
    log "ERROR: Fix 2 zip not found. Upload/attach fix2_t4x2_outputs.zip or run Fix 2 first."
    exit 1
  fi

  log "using Fix 2 zip: ${zip_path}"
  unzip -o "${zip_path}" 'data/revision/fix_02/*' 'results/revision/fix_02/*'
  if [ ! -s data/revision/fix_02/per_query.csv ]; then
    log "ERROR: Fix 2 zip did not contain data/revision/fix_02/per_query.csv"
    exit 1
  fi
  log "Fix 2 imported: $(row_count data/revision/fix_02/per_query.csv) rows"
}

preflight() {
  section "Preflight"
  import_fix2_if_needed
  git log -1 --oneline
  nvidia-smi || true
  python --version
  python -m py_compile \
    experiments/fix_03_multimetric_faithfulness.py \
    experiments/fix_04_tau_generalization.py \
    experiments/revision_utils.py \
    src/ragas_scorer.py
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

fix3_input_path() {
  if [ -s data/revision/fix_03/per_query_partial.csv ] && [ ! -s data/revision/fix_03/per_query.csv ]; then
    echo "data/revision/fix_03/per_query_partial.csv"
  else
    echo "${FIX3_INPUT}"
  fi
}

run_fix3() {
  cd "${REPO_DIR}"
  section "Run Fix 3 on GPU0"
  local input_path
  input_path="$(fix3_input_path)"
  local args=(
    python -u experiments/fix_03_multimetric_faithfulness.py
    --input "${input_path}"
    --judge_backend ollama
    --judge_model "${MODEL}"
    --second_nli_model "${FIX3_SECOND_NLI_MODEL}"
    --save_every "${FIX3_SAVE_EVERY}"
    --build_human_eval
  )
  if [ -n "${FIX3_LIMIT}" ]; then
    args+=(--limit "${FIX3_LIMIT}")
  fi
  env CUDA_VISIBLE_DEVICES=0 \
    OLLAMA_HOST="http://127.0.0.1:11434" \
    OLLAMA_BASE_URL="http://127.0.0.1:11434" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    "${args[@]}" 2>&1 | tee logs/revision/fix_03_kaggle_t4x2.log
}

run_fix4() {
  cd "${REPO_DIR}"
  section "Run Fix 4 on GPU1"
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES=1 \
    OLLAMA_HOST="http://127.0.0.1:11435" \
    OLLAMA_BASE_URL="http://127.0.0.1:11435" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    python -u experiments/fix_04_tau_generalization.py \
      --datasets ${FIX4_DATASETS} \
      --taus ${FIX4_TAUS} \
      --n "${FIX4_N}" \
      --seed 42 \
      --backend ollama \
      --model "${MODEL}" \
      --max_contexts "${FIX4_MAX_CONTEXTS}" \
      --save_every "${FIX4_SAVE_EVERY}" \
      2>&1 | tee logs/revision/fix_04_kaggle_t4x2.log
}

partial_rows_fix4() {
  shopt -s nullglob
  local files=(data/revision/fix_04/*_partial.csv)
  if [ "${#files[@]}" -eq 0 ]; then
    echo 0
  else
    sum_rows "${files[@]}"
  fi
}

heartbeat_status() {
  local f2 f3 f3p f4 f4p
  f2=$(row_count data/revision/fix_02/per_query.csv)
  f3=$(row_count data/revision/fix_03/per_query.csv)
  f3p=$(row_count data/revision/fix_03/per_query_partial.csv)
  f4=$(row_count data/revision/fix_04/per_query.csv)
  f4p=$(partial_rows_fix4)
  log "heartbeat fix2_input=${f2} fix3_final=${f3} fix3_partial=${f3p} fix4_final=${f4} fix4_partial_sum=${f4p}"
  tail -n 4 logs/revision/fix_03_kaggle_t4x2.log 2>/dev/null | sed 's/^/[fix3] /' || true
  tail -n 4 logs/revision/fix_04_kaggle_t4x2.log 2>/dev/null | sed 's/^/[fix4] /' || true
  tail -n 3 "${LOG_DIR}/ollama_gpu0.log" 2>/dev/null | sed 's/^/[ollama0] /' || true
  tail -n 3 "${LOG_DIR}/ollama_gpu1.log" 2>/dev/null | sed 's/^/[ollama1] /' || true
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

  run_fix3 &
  pid3=$!
  run_fix4 &
  pid4=$!

  while kill -0 "${pid3}" 2>/dev/null || kill -0 "${pid4}" 2>/dev/null; do
    sleep "${HEARTBEAT_SECONDS}"
    heartbeat_status
  done

  set +e
  wait "${pid3}"
  rc3=$?
  wait "${pid4}"
  rc4=$?
  set -e

  kill "${watch0}" "${watch1}" >/dev/null 2>&1 || true
  trap - EXIT

  if [ "${rc3}" -ne 0 ] || [ "${rc4}" -ne 0 ]; then
    log "ERROR: Fix 3 rc=${rc3}, Fix 4 rc=${rc4}"
    tail -n 120 logs/revision/fix_03_kaggle_t4x2.log || true
    tail -n 120 logs/revision/fix_04_kaggle_t4x2.log || true
    exit 1
  fi

  local f3 f4
  f3=$(row_count data/revision/fix_03/per_query.csv)
  f4=$(row_count data/revision/fix_04/per_query.csv)
  if [ "${f3}" -lt 1 ]; then
    log "ERROR: Fix 3 produced no final rows"
    exit 1
  fi
  if [ "${f4}" -lt "${FIX4_EXPECTED_MIN_ROWS}" ]; then
    log "ERROR: Fix 4 final row count ${f4} below minimum ${FIX4_EXPECTED_MIN_ROWS}"
    exit 1
  fi
  log "Fix 3 and Fix 4 completed"
}

status_stage() {
  repo_setup
  import_fix2_if_needed
  section "Status"
  heartbeat_status
  find data/revision/fix_03 data/revision/fix_04 results/revision/fix_03 results/revision/fix_04 -maxdepth 2 -type f -print | sort
}

package_stage() {
  repo_setup
  section "Package"
  rm -f /kaggle/working/fix3_4_t4x2_outputs.zip /kaggle/working/AAA_FIX3_4_T4X2_OUTPUTS.zip
  zip -r /kaggle/working/fix3_4_t4x2_outputs.zip \
    data/revision/fix_02 data/revision/fix_03 data/revision/fix_04 \
    results/revision/fix_02 results/revision/fix_03 results/revision/fix_04 \
    logs/revision/fix_03_* logs/revision/fix_04_* \
    "${LOG_DIR}" \
    CODEX.md REVISION_RUNBOOK.md REVISION_SUMMARY.md
  cp /kaggle/working/fix3_4_t4x2_outputs.zip /kaggle/working/AAA_FIX3_4_T4X2_OUTPUTS.zip
  ls -lh /kaggle/working/fix3_4_t4x2_outputs.zip /kaggle/working/AAA_FIX3_4_T4X2_OUTPUTS.zip
}

case "${STAGE}" in
  setup)
    setup_stage
    ;;
  parallel)
    parallel_stage
    ;;
  fix3)
    repo_setup
    ensure_ollama_binary
    start_server 0 11434
    import_fix2_if_needed
    run_fix3
    ;;
  fix4)
    repo_setup
    ensure_ollama_binary
    start_server 1 11435
    import_fix2_if_needed
    run_fix4
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
    echo "Usage: $0 {setup|parallel|fix3|fix4|status|package|full}" >&2
    exit 2
    ;;
esac
