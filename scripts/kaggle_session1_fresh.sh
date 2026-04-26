#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Saket-Maganti/rag-hallucination-detection.git}"
REPO_DIR="${REPO_DIR:-/kaggle/working/rag-hallucination-detection}"
MODEL="${MODEL:-mistral}"

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

echo "[session1] commit: $(git rev-parse --short HEAD)"
nvidia-smi || true
python --version

apt-get update -y
apt-get install -y zstd curl git zip

python -m pip install -U pip setuptools wheel
python -m pip install -e pip-package
python -m pip install \
  pandas numpy scipy scikit-learn matplotlib seaborn tabulate tqdm \
  datasets sentence-transformers transformers torch accelerate \
  langchain langchain-community langchain-core langchain-text-splitters langchain-ollama \
  chromadb

bash scripts/kaggle_ollama_guard.sh "${MODEL}"

python -m py_compile \
  experiments/fix_01_causal_matched_pairs.py \
  experiments/fix_05_coherence_preserving_noise.py \
  experiments/fix_11_raptor_full_table.py \
  experiments/revision_utils.py \
  src/ragas_scorer.py src/vectara_hem_scorer.py

if [ ! -s data/revision/fix_01/matched_pairs.csv ]; then
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
PYTHONUNBUFFERED=1 python -u experiments/fix_01_causal_matched_pairs.py \
  --stage generate \
  --backend ollama \
  --model "${MODEL}" \
  --resume \
  --save_every 2 \
  --progress_every 10 \
  2>&1 | tee logs/revision/fix_01_generate_session1.log

PYTHONUNBUFFERED=1 python -u experiments/fix_01_causal_matched_pairs.py \
  --stage analyze \
  2>&1 | tee logs/revision/fix_01_analyze_session1.log

bash scripts/kaggle_ollama_guard.sh "${MODEL}"
PYTHONUNBUFFERED=1 python -u experiments/fix_05_coherence_preserving_noise.py \
  --n 200 \
  --seed 42 \
  --backend ollama \
  --model "${MODEL}" \
  --max_contexts 300 \
  --n_noise 1 2 3 \
  --save_every 25 \
  2>&1 | tee logs/revision/fix_05_noise_slope_session1.log

bash scripts/kaggle_ollama_guard.sh "${MODEL}"
PYTHONUNBUFFERED=1 python -u experiments/fix_11_raptor_full_table.py \
  --datasets squad pubmedqa hotpotqa \
  --n 100 \
  --backend ollama \
  --model "${MODEL}" \
  --max_contexts 150 \
  --raptor_clusters 6 \
  2>&1 | tee logs/revision/fix_11_raptor_full_table_session1.log

rm -f /kaggle/working/revision_session1_outputs.zip
zip -r /kaggle/working/revision_session1_outputs.zip \
  data/revision/fix_01 results/revision/fix_01 \
  data/revision/fix_05 results/revision/fix_05 \
  data/revision/fix_11 results/revision/fix_11 \
  logs/revision \
  CODEX.md REVISION_SUMMARY.md REVISION_RUNBOOK.md

ls -lh /kaggle/working/revision_session1_outputs.zip
echo "[session1] done"

