#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-full}"
MODEL="${MODEL:-mistral}"
REPO_DIR="${REPO_DIR:-/kaggle/working/rag-hallucination-detection}"
LOG_DIR="${LOG_DIR:-/kaggle/working/fix2_t4x2_logs}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/kaggle/working/ollama_models}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"

FIX2_DATASETS="${FIX2_DATASETS:-squad}"
FIX2_N="${FIX2_N:-500}"
FIX2_MAX_CONTEXTS="${FIX2_MAX_CONTEXTS:-600}"
FIX2_SAVE_EVERY="${FIX2_SAVE_EVERY:-5}"
FIX2_SEEDS_GPU0="${FIX2_SEEDS_GPU0:-41 42 43}"
FIX2_SEEDS_GPU1="${FIX2_SEEDS_GPU1:-44 45}"
FIX2_EXPECTED_ROWS="${FIX2_EXPECTED_ROWS:-7500}"

mkdir -p "${LOG_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [fix2-t4x2] $*"; }
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
  mkdir -p logs/revision data/revision/fix_02 results/revision/fix_02
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

stop_stale() {
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

preflight() {
  section "Preflight"
  git log -1 --oneline
  nvidia-smi || true
  python --version
  python -m py_compile \
    experiments/fix_02_scaled_headline_n500.py \
    experiments/revision_utils.py
  curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
  curl -fsS http://127.0.0.1:11435/api/tags >/dev/null
  log "preflight OK"
}

setup_stage() {
  repo_setup
  install_deps
  ensure_ollama_binary
  stop_stale
  mkdir -p "${OLLAMA_MODELS_DIR}"
  start_server 0 11434
  start_server 1 11435
  pull_model
  preflight
}

run_shard() {
  local gpu="$1"
  local port="$2"
  local tag="$3"
  shift 3
  local seeds=("$@")
  local resume_args=()
  if [ "${FIX2_RESUME_PARTIAL:-0}" = "1" ]; then
    resume_args=(--resume_partial)
  fi

  cd "${REPO_DIR}"
  section "Run Fix 2 ${tag} on GPU${gpu}: seeds ${seeds[*]}"
  # shellcheck disable=SC2086
  env CUDA_VISIBLE_DEVICES="${gpu}" \
    OLLAMA_HOST="http://127.0.0.1:${port}" \
    OLLAMA_BASE_URL="http://127.0.0.1:${port}" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    python -u experiments/fix_02_scaled_headline_n500.py \
      --datasets ${FIX2_DATASETS} \
      --n "${FIX2_N}" \
      --seeds "${seeds[@]}" \
      --backend ollama \
      --model "${MODEL}" \
      --max_contexts "${FIX2_MAX_CONTEXTS}" \
      --save_every "${FIX2_SAVE_EVERY}" \
      --output_tag "${tag}" \
      "${resume_args[@]}" \
      2>&1 | tee "logs/revision/fix_02_${tag}_kaggle_t4x2.log"
}

import_fix2_outputs() {
  cd "${REPO_DIR}"
  section "Import partial Fix 2 outputs"
  local zip_path=""
  local extracted_root=""
  local candidates=()
  if [ -n "${FIX2_REPAIR_ZIP:-}" ]; then
    candidates+=("${FIX2_REPAIR_ZIP}")
  fi
  candidates+=(
    "/kaggle/working/AAA_FIX2_T4X2_OUTPUTS.zip"
    "/kaggle/working/fix2_t4x2_outputs.zip"
    "/kaggle/working/OUTPUT_FIX2.zip"
    "${REPO_DIR}/OUTPUT_FIX2.zip"
    "${REPO_DIR}/fix2_t4x2_outputs.zip"
  )

  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      zip_path="${candidate}"
      break
    fi
  done

  if [ -z "${zip_path}" ] && [ -d /kaggle/input ]; then
    zip_path="$(find /kaggle/input /kaggle/working -maxdepth 10 -type f \( -iname '*fix2*.zip' -o -iname '*FIX2*.zip' \) | sort | head -n 1 || true)"
  fi

  if [ -n "${zip_path}" ] && [ -f "${zip_path}" ]; then
    log "importing ${zip_path}"
    unzip -o -q "${zip_path}" -d "${REPO_DIR}"
  else
    extracted_root="$(find /kaggle/input /kaggle/working -maxdepth 12 -type d -path '*/data/revision/fix_02' | sort | head -n 1 || true)"
    if [ -z "${extracted_root}" ] || [ ! -d "${extracted_root}" ]; then
      log "ERROR: no partial Fix 2 zip or extracted data/revision/fix_02 folder found."
      log "Upload or attach fix2_t4x2_outputs.zip, or set FIX2_REPAIR_ZIP=/path/to/zip."
      exit 1
    fi

    log "importing extracted dataset from ${extracted_root}"
    mkdir -p data/revision/fix_02 results/revision/fix_02 logs/revision "${LOG_DIR}"
    cp -av "${extracted_root}/." data/revision/fix_02/

    local extracted_base
    extracted_base="$(cd "${extracted_root}/../../.." && pwd)"
    if [ -d "${extracted_base}/results/revision/fix_02" ]; then
      cp -av "${extracted_base}/results/revision/fix_02/." results/revision/fix_02/ || true
    fi
    if [ -d "${extracted_base}/logs/revision" ]; then
      cp -av "${extracted_base}/logs/revision/." logs/revision/ || true
    fi
    if [ -d "${extracted_base}/kaggle/working/fix2_t4x2_logs" ]; then
      cp -av "${extracted_base}/kaggle/working/fix2_t4x2_logs/." "${LOG_DIR}/" || true
    fi
  fi

  find data/revision/fix_02 -maxdepth 1 -type f -name '*.csv' -print | sort
}

merge_stage() {
  cd "${REPO_DIR}"
  section "Merge Fix 2 shards"
  python - <<'PYMERGE'
from pathlib import Path

import pandas as pd

from experiments.fix_02_scaled_headline_n500 import aggregate, write_columns
from experiments.revision_utils import write_markdown_table

out_data = Path("data/revision/fix_02")
out_results = Path("results/revision/fix_02")
out_data.mkdir(parents=True, exist_ok=True)
out_results.mkdir(parents=True, exist_ok=True)

partial_paths = sorted(out_data.glob("*_partial_gpu*.csv"))
shard_paths = sorted(out_data.glob("per_query_gpu*.csv"))
paths = partial_paths if partial_paths else shard_paths
if not paths:
    raise SystemExit("No Fix 2 shard or partial CSVs found")

frames = []
for path in paths:
    df = pd.read_csv(path)
    frames.append(df)

merged = pd.concat(frames, ignore_index=True)
dedupe_cols = ["dataset", "seed", "question", "condition"]
before = len(merged)
if set(dedupe_cols).issubset(merged.columns):
    merged = merged.drop_duplicates(dedupe_cols, keep="last").reset_index(drop=True)
merged = merged.drop(columns=["source_shard"], errors="ignore")
merged.to_csv(out_data / "per_query.csv", index=False)
write_columns()

summary, contrasts = aggregate(merged)
summary.to_csv(out_results / "headline_table.csv", index=False)
contrasts.to_csv(out_results / "paired_contrasts.csv", index=False)
write_markdown_table(
    out_results / "summary.md",
    "Fix 2 - scaled headline n=500 x 5 seeds",
    {"Headline Table": summary, "Paired Contrasts": contrasts},
)
print(f"[Fix02 merge] inputs={len(paths)} rows={len(merged)} dropped_duplicates={before - len(merged)}")
for path in paths:
    print(f"[Fix02 merge] {path} rows={len(pd.read_csv(path))}")
PYMERGE
}

partial_rows() {
  shopt -s nullglob
  local files=(data/revision/fix_02/*_partial_gpu*.csv)
  if [ "${#files[@]}" -eq 0 ]; then
    echo 0
  else
    sum_rows "${files[@]}"
  fi
}

heartbeat_status() {
  local final_rows shard0_rows shard1_rows partial
  final_rows=$(row_count data/revision/fix_02/per_query.csv)
  shard0_rows=$(row_count data/revision/fix_02/per_query_gpu0.csv)
  shard1_rows=$(row_count data/revision/fix_02/per_query_gpu1.csv)
  partial=$(partial_rows)
  log "heartbeat fix2_final=${final_rows} shard_gpu0=${shard0_rows} shard_gpu1=${shard1_rows} partial_rows=${partial} expected_final=${FIX2_EXPECTED_ROWS}"
  tail -n 4 logs/revision/fix_02_gpu0_kaggle_t4x2.log 2>/dev/null | sed 's/^/[gpu0] /' || true
  tail -n 4 logs/revision/fix_02_gpu1_kaggle_t4x2.log 2>/dev/null | sed 's/^/[gpu1] /' || true
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

  # shellcheck disable=SC2206
  seeds0=(${FIX2_SEEDS_GPU0})
  # shellcheck disable=SC2206
  seeds1=(${FIX2_SEEDS_GPU1})

  run_shard 0 11434 gpu0 "${seeds0[@]}" &
  pid0=$!
  run_shard 1 11435 gpu1 "${seeds1[@]}" &
  pid1=$!

  while kill -0 "${pid0}" 2>/dev/null || kill -0 "${pid1}" 2>/dev/null; do
    sleep "${HEARTBEAT_SECONDS}"
    heartbeat_status
  done

  set +e
  wait "${pid0}"
  rc0=$?
  wait "${pid1}"
  rc1=$?
  set -e

  kill "${watch0}" "${watch1}" >/dev/null 2>&1 || true
  trap - EXIT

  if [ "${rc0}" -ne 0 ] || [ "${rc1}" -ne 0 ]; then
    log "ERROR: Fix 2 shard rc gpu0=${rc0}, gpu1=${rc1}"
    tail -n 100 logs/revision/fix_02_gpu0_kaggle_t4x2.log || true
    tail -n 100 logs/revision/fix_02_gpu1_kaggle_t4x2.log || true
    exit 1
  fi

  merge_stage
  local final_rows
  final_rows=$(row_count data/revision/fix_02/per_query.csv)
  if [ "${final_rows}" -lt "${FIX2_EXPECTED_ROWS}" ]; then
    log "ERROR: final row count ${final_rows} below expected ${FIX2_EXPECTED_ROWS}"
    exit 1
  fi
  log "Fix 2 completed with ${final_rows} rows"
}

repair_stage() {
  setup_stage
  import_fix2_outputs

  FIX2_RESUME_PARTIAL=1 run_shard 0 11434 gpu0 43
  merge_stage

  local final_rows
  final_rows=$(row_count data/revision/fix_02/per_query.csv)
  if [ "${final_rows}" -lt "${FIX2_EXPECTED_ROWS}" ]; then
    log "ERROR: repaired final row count ${final_rows} below expected ${FIX2_EXPECTED_ROWS}"
    exit 1
  fi
  log "Fix 2 repaired with ${final_rows} rows"
  package_stage
}

status_stage() {
  repo_setup
  section "Status"
  heartbeat_status
  find data/revision/fix_02 results/revision/fix_02 -maxdepth 2 -type f -print | sort
}

package_stage() {
  repo_setup
  section "Package"
  rm -f /kaggle/working/fix2_t4x2_outputs.zip /kaggle/working/AAA_FIX2_T4X2_OUTPUTS.zip
  zip -r /kaggle/working/fix2_t4x2_outputs.zip \
    data/revision/fix_02 \
    results/revision/fix_02 \
    logs/revision/fix_02_* \
    "${LOG_DIR}" \
    CODEX.md REVISION_RUNBOOK.md REVISION_SUMMARY.md
  cp /kaggle/working/fix2_t4x2_outputs.zip /kaggle/working/AAA_FIX2_T4X2_OUTPUTS.zip
  ls -lh /kaggle/working/fix2_t4x2_outputs.zip /kaggle/working/AAA_FIX2_T4X2_OUTPUTS.zip
}

case "${STAGE}" in
  setup)
    setup_stage
    ;;
  parallel)
    parallel_stage
    ;;
  merge)
    repo_setup
    merge_stage
    ;;
  repair)
    repair_stage
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
    echo "Usage: $0 {setup|parallel|merge|repair|status|package|full}" >&2
    exit 2
    ;;
esac
