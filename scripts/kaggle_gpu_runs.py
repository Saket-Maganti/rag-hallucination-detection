"""
scripts/kaggle_gpu_runs.py
==========================

The two GPU-only experimental items, packaged as one self-contained script
suitable for a Kaggle Notebook (T4 x2 / P100 / L4) or Colab GPU runtime.

What runs here, and why it can't run on the M4 Air:

    Item #1 — Mechanistic attention probe
        Loads Mistral-7B-Instruct-v0.2 with `output_attentions=True` and
        `attn_implementation="eager"`. Eager attention + 7B fp16 + a 4 K
        context for matched (coherent, fragmented) prompt pairs spikes to
        ~14-16 GB activation memory; MPS unified memory is technically
        big enough but the eager path is *very* slow there
        (~5 min/prompt) and would not finish in any reasonable session.
        On a T4 it is ~3-5 s per prompt; ~80 pairs → ~10 min wall-clock
        plus model load.

    Item #4 — Head-to-head vs Self-RAG + CRAG
        Self-RAG inference uses the published `selfrag/selfrag_llama2_7b`
        checkpoint via HF Transformers (not Ollama). 7 B fp16 ≈ 14 GB;
        again M4 unified memory fits but with no CUDA graph fusion or
        flash-attn it is impractically slow. The wrapper is identical in
        shape across devices, so we just point it at "cuda" here.
        CRAG and the HCPC variants run on CPU/MPS but we keep them in this
        script so the head-to-head table is filled from one machine in one
        sitting.

How to use this on Kaggle
-------------------------
1. Create a new Kaggle Notebook, set Accelerator = GPU T4 x2 (or P100).
2. Add the GitHub repo as a dataset OR clone it in the first cell:
       !git clone https://github.com/Saket-Maganti/rag-hallucination-detection.git
       %cd rag-hallucination-detection
       !pip install -q -r requirements.txt
3. Upload the adversarial JSONLs and any cached HCPC v2 logs as a Kaggle
   dataset (data/adversarial/*.jsonl, results/hcpc_v2/logs/*.json).
4. Run:
       !python3 scripts/kaggle_gpu_runs.py --task mechanistic
       !python3 scripts/kaggle_gpu_runs.py --task headtohead --datasets squad pubmedqa hotpotqa --n_questions 30
5. Download `results/mechanistic/` and `results/headtohead/` back to the
   M4 for analysis and figure generation.

If a single 12-hour session is not enough, both underlying runners
(`run_mechanistic_analysis.py` and `run_headtohead_comparison.py`) write
checkpoints per pair / per (dataset, condition) tuple, so a re-run picks
up where it stopped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _ensure_cuda() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[Kaggle] CUDA available: {name}")
            return "cuda"
        print("[Kaggle] WARNING: CUDA not available; falling back to CPU. "
              "The mechanistic + Self-RAG runs will be very slow.")
        return "cpu"
    except Exception:
        return "cpu"


def run_mechanistic(args) -> int:
    """Item #1 — attention-entropy probe on adversarial pairs."""
    device = _ensure_cuda()
    cmd = [
        sys.executable,
        os.path.join(ROOT, "experiments", "run_mechanistic_analysis.py"),
        "--model",  args.model,
        "--source", args.source,
        "--device", device,
        "--max_new_tokens", str(args.max_new_tokens),
    ]
    if args.max_pairs:
        cmd.extend(["--max_pairs", str(args.max_pairs)])
    print("[Kaggle] →", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def run_headtohead(args) -> int:
    """Item #4 — five-condition head-to-head with Self-RAG + CRAG."""
    device = _ensure_cuda()
    cmd = [
        sys.executable,
        os.path.join(ROOT, "experiments", "run_headtohead_comparison.py"),
        "--datasets",        *args.datasets,
        "--n_questions",     str(args.n_questions),
        "--selfrag_device",  device,
    ]
    print("[Kaggle] →", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["mechanistic", "headtohead", "both"])
    # Mechanistic options
    parser.add_argument("--model",  default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--source", default="adversarial",
                        choices=["adversarial", "hcpc_logs", "both"])
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_pairs",      type=int, default=0)
    # Head-to-head options
    parser.add_argument("--datasets",     nargs="+",
                        default=["squad", "pubmedqa", "hotpotqa"])
    parser.add_argument("--n_questions",  type=int, default=30)
    args = parser.parse_args()

    if args.task in ("mechanistic", "both"):
        rc = run_mechanistic(args)
        if rc != 0:
            print(f"[Kaggle] mechanistic exited rc={rc}")
            if args.task == "mechanistic":
                sys.exit(rc)
    if args.task in ("headtohead", "both"):
        rc = run_headtohead(args)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
