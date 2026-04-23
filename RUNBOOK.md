# Paper-revision execution runbook

This file is the operational counterpart to the
"Paper-revision infrastructure" table in the README. It tells you, on an
M4 Air + Kaggle setup, **what to run, in what order, where, and how long
it takes**. Every step is resumable: each runner writes per-tuple
checkpoints, so a 3 a.m. crash never costs more than the in-progress
tuple.

## Hardware assumptions

- **M4 Air, 16 GB unified memory** — runs Ollama (Mistral / Llama-3 /
  Qwen-2.5 7B in Q4_K_M ≈ 4–5 GB each), sentence-transformers on MPS,
  and DeBERTa-v3-base NLI on MPS comfortably.
- **Kaggle Notebook with GPU** — used only for items that need CUDA
  attention or the published Self-RAG checkpoint.
- Disk: ~30 GB free for Chroma collections (one per dataset × embedder).

## One-time prerequisites (M4)

```bash
# Python env
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Ollama models (install Ollama first: https://ollama.com/download)
ollama pull mistral
ollama pull llama3
bash scripts/setup_qwen.sh        # pulls qwen2.5:7b-instruct + alias

# Sentence-Transformers will lazy-download embedders on first use.
# To pre-warm them (~3 GB total):
python3 -c "from src.embedders import build_embedder; \
            [build_embedder(n)._lazy_load() for n in ('minilm','bge-large','e5-large','gte-large')]"
```

## Execution order

The dependency graph is `5 → {1, 8} → {2, 3} → 4 → {6, 7}`. Items 1 and 4
require GPU; everything else runs on the M4. Steps below are listed in
that dependency order with M4 vs Kaggle marked.

| Step | Where | Item | Command | Wall-clock | Notes |
|------|-------|------|---------|-----------|-------|
| 0 | M4 | setup | `bash scripts/setup_qwen.sh` | 10 min | one-time, ~4 GB download |
| 1 | M4 | #5 adversarial | `python3 experiments/run_adversarial_coherence.py` | 2.5 h | 40 cases × Mistral-7B; produces matched pairs that feed step 4 |
| 2 | M4 | **#8 multi-retriever** | `python3 experiments/run_multi_retriever_ablation.py --datasets squad pubmedqa --embedders minilm bge-large e5-large gte-large --n_questions 30` | 3 h | Headline table for the rebuttal; first-run also downloads ~3 GB of embedders |
| 3 | M4 | #2 multidataset (M4-feasible) | `python3 experiments/run_multidataset_validation.py --datasets squad pubmedqa hotpotqa triviaqa naturalqs --models mistral llama3 qwen2.5 --n_questions 30` | 6–8 h (overnight) | Resumable per (dataset, model) tuple |
| 4 | M4 | #2 cont. — FinanceBench | `python3 experiments/run_multidataset_validation.py --datasets financebench --models mistral llama3 qwen2.5 --n_questions 30` | 1.5 h | Optional; HF dataset is gated, skip if blocked |
| 5 | Kaggle (T4) | #4 head-to-head | `python3 scripts/kaggle_gpu_runs.py --task headtohead --datasets squad pubmedqa hotpotqa --n_questions 30` | 4–6 h | Self-RAG = ~3 s/query on T4; one Kaggle session covers it |
| 6 | Kaggle (P100/T4) | #1 mechanistic | `python3 scripts/kaggle_gpu_runs.py --task mechanistic --source adversarial` | 3–5 h | Needs step 1 output uploaded as a Kaggle dataset |
| 7 | M4 | stats | `python3 experiments/run_significance_tests.py` | 5 min | Re-run on combined data |
| 8 | M4 | #7 packager | `python3 scripts/package_benchmark.py --paradox_csv results/multidataset/per_query.csv --paradox_summary_csv results/multidataset/summary.csv --adversarial_dir data/adversarial --output_dir release/context_coherence_bench_v1` | 10 min | Packs HuggingFace-loadable bundle |
| — | defer | #6 Prolific | (skipped) | — | Needs IRB/ethics + ~$500 budget; defer to camera-ready |

Total clock time on M4 + one Kaggle day: **~3 days, mostly passive**.

## What you watch for

After step 2 (multi-retriever) the headline answer to the reviewer's #1
critique is in `results/multi_retriever/paradox_by_embedder.csv`:

- Column `paradox_drop` (faith_baseline − faith_v1) should be **positive
  for every row**. If it is, the coherence framing survives the
  strong-retriever objection and we can keep the current paper title.
- If `paradox_drop` collapses to ~0 for the 335 M embedders, the paper
  must be reframed as "failure mode of weak retrievers" — still a real
  contribution, but the title and abstract should change before the
  next submission.

After step 3, the expanded Table 2 in §Results goes from 2 datasets × 2
generators to **5–6 datasets × 3 generators**. The HotpotQA row is
particularly informative because multi-hop questions are exactly where
coherence should matter most.

After step 5 (head-to-head), the §3.5–3.6 "no external comparison"
caveat is gone — Self-RAG and CRAG appear as additional rows in the
same table.

After step 6 (mechanistic), §7.5 has actual attention-entropy and
retrieved-mass numbers per layer, not a placeholder. This is the piece
that converts "correlation" critiques into "we measured the mechanism".

## Kaggle session checklist

When you start the Kaggle notebook for steps 5/6:

1. Settings → Accelerator → **GPU T4 x2** (P100 also fine; L4 if Pro).
2. Settings → Internet → **On** (needed for HF model downloads).
3. First cell:
   ```python
   !git clone https://github.com/Saket-Maganti/rag-hallucination-detection.git
   %cd rag-hallucination-detection
   !pip install -q -r requirements.txt
   ```
4. Upload `data/adversarial/*.jsonl` and (for step 6 only) the
   `results/hcpc_v2/logs/` directory as Kaggle datasets, and symlink
   them into the cloned repo.
5. Run the GPU script for one task at a time; `--task headtohead`
   first, then `--task mechanistic` in a second session if memory is
   tight.
6. At the end, zip and download `results/headtohead/`,
   `results/mechanistic/`, then commit them on the M4.

## When things go wrong

- **HF rate-limit on dataset download** — wait 15 min, or pre-download
  on the M4 with `datasets-cli` and upload as a Kaggle dataset.
- **Ollama OOM on M4** — close other apps; Q4 7B + DeBERTa NLI fits in
  ~12 GB peak. Worst case, switch to `--models mistral` only and re-run
  llama3/qwen2.5 sequentially.
- **Self-RAG fails to load** — the wrapper falls back gracefully and the
  other four conditions still produce results. Re-run later with
  `--task headtohead` only.
- **A (dataset, embedder) tuple errored mid-run** — delete its row from
  `results/multi_retriever/completed_tuples.json` and re-launch; the
  driver will resume only the missing tuple.

## Final paper deliverables

Once steps 1–8 are done, the camera-ready submission has:

- Expanded Table 2: 5–6 datasets × 3 generators
- Multi-retriever row: paradox magnitude across 4 embedders (NEW)
- Self-RAG / CRAG comparison columns
- Mechanistic figure: layer-wise Δ entropy and Δ retrieved mass
- Adversarial detection AUCs across 4 categories
- Released benchmark bundle (HuggingFace-loadable)

That collectively answers every concrete critique from the Apr-2026
review, and is the form in which the paper is competitive at NeurIPS.
