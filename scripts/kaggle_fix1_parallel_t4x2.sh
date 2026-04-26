#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-mistral}"
REPO_DIR="${REPO_DIR:-/kaggle/working/rag-hallucination-detection}"
LOG_DIR="${LOG_DIR:-/kaggle/working/fix1_parallel_logs}"
mkdir -p "${LOG_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [fix1-t4x2] $*"; }
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

start_server() {
  local gpu="$1"
  local port="$2"
  local log_file="${LOG_DIR}/ollama_gpu${gpu}.log"
  local url="http://127.0.0.1:${port}"
  if curl -fsS "${url}/api/tags" >/dev/null 2>&1; then
    log "ollama gpu=${gpu} port=${port} already live"
    return
  fi
  log "starting ollama gpu=${gpu} port=${port}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" OLLAMA_HOST="127.0.0.1:${port}" \
    OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=1 \
    ollama serve >"${log_file}" 2>&1 &
  for _ in $(seq 1 60); do
    if curl -fsS "${url}/api/tags" >/dev/null 2>&1; then
      log "ollama gpu=${gpu} port=${port} live"
      return
    fi
    sleep 2
  done
  log "ERROR: ollama gpu=${gpu} port=${port} failed"
  tail -n 80 "${log_file}" || true
  exit 1
}

run_shard() {
  local shard="$1"
  local gpu="$2"
  local port="$3"
  local start="$4"
  local end="$5"
  local out="data/revision/fix_01/per_query_shard${shard}.csv"
  local log_file="logs/revision/fix_01_generate_shard${shard}.log"
  section "Shard ${shard} gpu=${gpu} pairs=[${start},${end})"
  env CUDA_VISIBLE_DEVICES="${gpu}" OLLAMA_HOST="http://127.0.0.1:${port}" \
    PYTHONUNBUFFERED=1 python -u experiments/fix_01_causal_matched_pairs.py \
      --stage generate \
      --backend ollama \
      --model "${MODEL}" \
      --resume \
      --pair_start "${start}" \
      --pair_end "${end}" \
      --per_query_out "${out}" \
      --save_every 2 \
      --progress_every 10 \
      2>&1 | tee "${log_file}"
}

cd /kaggle/working
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --branch main https://github.com/Saket-Maganti/rag-hallucination-detection.git "${REPO_DIR}"
else
  git -C "${REPO_DIR}" fetch origin main
  git -C "${REPO_DIR}" checkout main
  git -C "${REPO_DIR}" pull --ff-only origin main
fi
cd "${REPO_DIR}"
mkdir -p logs/revision data/revision/fix_01 results/revision/fix_01

section "Environment"
git log -1 --oneline
nvidia-smi || true
python --version

section "Dependencies"
apt-get update -y
apt-get install -y zstd curl git zip
python -m pip install -U pip setuptools wheel
python -m pip install -e pip-package
python -m pip install \
  pandas numpy scipy scikit-learn matplotlib seaborn tabulate tqdm \
  datasets sentence-transformers transformers torch accelerate \
  langchain langchain-community langchain-core langchain-text-splitters langchain-ollama \
  chromadb

section "Ollama setup"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
start_server 0 11434
start_server 1 11435

OLLAMA_HOST="127.0.0.1:11434" ollama pull "${MODEL}"
OLLAMA_HOST="127.0.0.1:11435" ollama pull "${MODEL}" || true

section "Preflight"
python -m py_compile experiments/fix_01_causal_matched_pairs.py experiments/revision_utils.py

if [ ! -s data/revision/fix_01/matched_pairs.csv ]; then
  section "Construct matched pairs"
  PYTHONUNBUFFERED=1 python -u experiments/fix_01_causal_matched_pairs.py \
    --stage construct \
    --dataset squad \
    --n_target 200 \
    --seed 42 \
    --max_contexts 400 \
    --candidate_limit 400 \
    --run_tag primary_n200 \
    2>&1 | tee logs/revision/fix_01_construct_parallel.log
fi

section "Run two shards in parallel"
run_shard 0 0 11434 0 100 &
pid0=$!
run_shard 1 1 11435 100 200 &
pid1=$!

while kill -0 "${pid0}" 2>/dev/null || kill -0 "${pid1}" 2>/dev/null; do
  sleep 60
  r0=$(row_count data/revision/fix_01/per_query_shard0.csv)
  r1=$(row_count data/revision/fix_01/per_query_shard1.csv)
  n0=0
  n1=0
  [[ "${r0}" =~ ^[0-9]+$ ]] && n0="${r0}"
  [[ "${r1}" =~ ^[0-9]+$ ]] && n1="${r1}"
  log "heartbeat shard0=${r0}/200 shard1=${r1}/200 total=$((n0 + n1))/400"
  tail -n 3 "${LOG_DIR}/ollama_gpu0.log" 2>/dev/null | sed 's/^/[ollama0] /' || true
  tail -n 3 "${LOG_DIR}/ollama_gpu1.log" 2>/dev/null | sed 's/^/[ollama1] /' || true
done

wait "${pid0}"
wait "${pid1}"

section "Merge shards"
python - <<'PY'
from pathlib import Path
import pandas as pd
paths = [Path("data/revision/fix_01/per_query_shard0.csv"), Path("data/revision/fix_01/per_query_shard1.csv")]
dfs = [pd.read_csv(p) for p in paths if p.exists() and p.stat().st_size > 0]
if not dfs:
    raise SystemExit("no shard outputs found")
df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates(subset=["pair_id", "set_type"], keep="last")
df = df.sort_values(["pair_id", "set_type"]).reset_index(drop=True)
df.to_csv("data/revision/fix_01/per_query.csv", index=False)
print(f"merged rows={len(df)} complete_pairs={df.groupby('pair_id')['set_type'].nunique().eq(2).sum()}")
PY

section "Analyze"
PYTHONUNBUFFERED=1 python -u experiments/fix_01_causal_matched_pairs.py \
  --stage analyze \
  2>&1 | tee logs/revision/fix_01_analyze_parallel.log

section "Package"
rm -f /kaggle/working/fix1_parallel_outputs.zip
zip -r /kaggle/working/fix1_parallel_outputs.zip \
  data/revision/fix_01 results/revision/fix_01 logs/revision/fix_01_* "${LOG_DIR}"
ls -lh /kaggle/working/fix1_parallel_outputs.zip
log "done"
