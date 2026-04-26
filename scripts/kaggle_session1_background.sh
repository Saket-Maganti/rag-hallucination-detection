#!/usr/bin/env bash
set -euo pipefail

LOG="${1:-/kaggle/working/session1_live.log}"
PID_FILE="${2:-/kaggle/working/session1.pid}"
REPO_DIR="${REPO_DIR:-/kaggle/working/rag-hallucination-detection}"

cd /kaggle/working
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --branch main https://github.com/Saket-Maganti/rag-hallucination-detection.git "${REPO_DIR}"
else
  git -C "${REPO_DIR}" fetch origin main
  git -C "${REPO_DIR}" checkout main
  git -C "${REPO_DIR}" pull --ff-only origin main
fi

cd "${REPO_DIR}"

if [ -s "${PID_FILE}" ]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" >/dev/null 2>&1; then
    echo "[background] session already running pid=${old_pid}"
    echo "[background] log=${LOG}"
    exit 0
  fi
fi

rm -f "${LOG}"
echo "[background] starting session1 at $(date)" | tee -a "${LOG}"
echo "[background] repo=$(pwd)" | tee -a "${LOG}"
echo "[background] commit=$(git rev-parse --short HEAD)" | tee -a "${LOG}"

nohup bash -lc "cd '${REPO_DIR}' && bash -x scripts/kaggle_session1_fresh.sh" >>"${LOG}" 2>&1 &
pid=$!
echo "${pid}" >"${PID_FILE}"
echo "[background] pid=${pid}"
echo "[background] log=${LOG}"

