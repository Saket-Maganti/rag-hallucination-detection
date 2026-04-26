# CODEX Handoff - NeurIPS Revision

This file is the working handoff for the NeurIPS revision of:

**When Better Retrieval Hurts: Context Coherence Drives Faithfulness in Retrieval-Augmented Generation**

It records what has been built, what has actually been run, the zero-dollar
execution plan, and the current checklist. Do not claim an experiment is
complete until its CSV/log/result files exist.

## Current Commit

Latest pushed revision notebook commit at the time this file was written:

- `9911200f` - `Add fresh Kaggle session 1 notebook`

Main GPU notebook:

- `notebooks/revision_session1_kaggle_fresh.ipynb`
- `notebooks/revision_fix5_11_kaggle_t4x2.ipynb` for the split Fix 5 +
  Fix 11 run on Kaggle T4 x2.

Most robust Session 1 entry point:

- `scripts/kaggle_session1_fresh.sh`
- `scripts/kaggle_ollama_guard.sh`
- `scripts/kaggle_fix1_only.sh` if you want the causal gate before running
  the rest of Session 1.
- `scripts/kaggle_fix1_parallel_t4x2.sh` if Kaggle gives T4 x2; this shards
  Fix 1 across two Ollama servers and is the fastest zero-dollar gate run.
- `scripts/kaggle_fix5_11_t4x2.sh` plus
  `scripts/kaggle_stream_fix5_11_t4x2.py` run Fix 5 and Fix 11 only, with
  two Ollama servers, visible wrapper heartbeats, and a final zip package.

The Session 1 script prints timestamped sections and 60-second heartbeats for
long jobs. Heartbeats include elapsed time, CSV row count, expected row count,
and the last few Ollama log lines.

Primary trackers:

- `REVISION_SUMMARY.md`
- `REVISION_RUNBOOK.md`
- `experiments/fix_01_log.md`

## Hard Constraints We Decided

- Zero dollars: no paid APIs, no paid Colab, no Together.ai paid calls, no
  OpenAI/Anthropic judge calls.
- Do not use Groq; it burns quota/minutes too fast.
- Use only M4 Air, free Kaggle/Colab GPU sessions, local Ollama, and local
  Hugging Face models.
- Fix 1 is the gating experiment. Do not finish narrative rewrites around
  causality until Fix 1 generation and analysis are complete.
- Report null results honestly.
- If Fix 1 fails, downgrade causal/mechanistic language to predictive or
  diagnostic language throughout.
- Do not p-hack. Use preregistered sample sizes, seed counts, Wilcoxon tests,
  10000 bootstrap resamples, Wilson CIs for binary rates, and effect sizes.
- Fix 7 independent 70B reproduction is budget-blocked under zero-dollar mode
  unless truly free 70B-capable compute becomes available.

## What Is Already Done

### Code And Paper Scaffolds

All Fixes 1-11 have code/log/LaTeX scaffolds:

- `experiments/fix_01_causal_matched_pairs.py`
- `experiments/fix_02_scaled_headline_n500.py`
- `experiments/fix_03_multimetric_faithfulness.py`
- `experiments/fix_04_tau_generalization.py`
- `experiments/fix_05_coherence_preserving_noise.py`
- `experiments/fix_06_baseline_h2h_pareto.py`
- `experiments/fix_07_together_70b_reproduction.py`
- `experiments/fix_09_partial_correlations.py`
- `experiments/fix_11_raptor_full_table.py`
- `experiments/revision_utils.py`
- `ragpaper/sections/revision/fix_*.tex`

Shared helpers/wrappers added:

- `src/ragas_scorer.py`
- `src/vectara_hem_scorer.py`
- `src/together_llm.py`
- `src/openai_llm.py`
- `src/anthropic_llm.py`

The OpenAI/Anthropic/Together wrappers are present for reproducibility, but
they are not part of the current zero-dollar plan.

### Fix 1 Result

Fix 1 matched-pair construction has already run successfully:

- valid matched pairs: `200`
- skipped queries: `0`
- mean absolute similarity gap: `0.006351`
- maximum absolute similarity gap: `0.018512`
- mean CCS gap: `0.532634`
- minimum CCS gap: `0.264139`
- maximum HIGH/LOW passage overlap: `1`

Files:

- `data/revision/fix_01/matched_pairs.csv`
- `data/revision/fix_01/per_query_construct_only.csv`
- `results/revision/fix_01/construction_summary.csv`
- `results/revision/fix_01/match_diagnostics.csv`

Fix 1 generation and analysis were completed locally after Kaggle/Colab output
instability:

- generated rows: `400`
- complete matched pairs: `200`
- HIGH-CCS mean faithfulness: `0.636195`
- LOW-CCS mean faithfulness: `0.638587`
- HIGH minus LOW: `-0.002392`
- Wilcoxon one-sided greater p-value: `0.628268`
- bootstrap 95% CI: `[-0.021651, 0.016819]`
- `h1_supported`: `False`

This means the paper must downgrade causal/mechanistic language around CCS.

### Validation

Last local validation passed:

```bash
python3 -m py_compile ...
python3 -m pytest tests/test_ccs.py pip-package/tests/test_core.py
# 21 passed
```

## GPU / No-GPU Split

### Does Not Need Hosted GPU

Run these on the M4 Air:

- validation and tests
- Fix 1 construction
- Fix 1 analysis after generation exists
- Fix 3 human-agreement analysis after manual labels exist
- Fix 8 theory reframe integration
- Fix 9 partial correlations
- Fix 10 scope/deployment rewrite
- CSV/table aggregation

Estimated M4 runtimes:

| Task | Runtime |
| --- | ---: |
| Validation/tests | 1-3 min |
| Fix 1 construction | already done; 10-40 min if rerun |
| Fix 1 analysis | seconds-2 min |
| Fix 3 human-agreement analysis | seconds-2 min |
| Fix 8 / Fix 10 manual integration | 30-90 min |
| Fix 9 partial correlations | seconds-5 min |

### Hosted GPU Strongly Recommended

Can run on M4 Air, but use free Kaggle/Colab GPU when available:

| Task | M4 Air | Free Kaggle/Colab GPU |
| --- | ---: | ---: |
| Fix 1 generation/NLI | 2-5h | 45-120 min |
| Fix 2 headline n=500 x 5 | 18-36h | 8-16h |
| Fix 3 second-NLI + local judge | 12-24h | 4-10h |
| Fix 4 tau matrix | 18-36h | 8-16h |
| Fix 5 noise slope | 4-8h | 1.5-4h |
| Fix 6 baselines without Self-RAG | 4-8h | 2-4h |
| Fix 11 RAPTOR table | 2-5h | 1-2.5h |

### Required Or Blocked

- Fix 6 with `--include_selfrag` requires CUDA-class GPU and enough VRAM.
- Fix 7 true independent 70B reproduction is budget-blocked at `$0`.

## Session Plan

### Session 1 - Free Kaggle GPU

Current split run for Kaggle T4 x2:

- `notebooks/revision_fix5_11_kaggle_t4x2.ipynb`
- runs only Fix 5 and Fix 11
- starts two Ollama servers: GPU0 on port `11434`, GPU1 on port `11435`
- writes `/kaggle/working/fix5_11_t4x2_outputs.zip`
- expected runtime: roughly `2.5-4.5h` after setup on T4 x2

Use this before trying another all-in-one session.

Notebook:

- `notebooks/revision_session1_kaggle_fresh.ipynb`

Runs:

- Fix 1 generation + analysis
- Fix 5 coherence-preserving noise slope
- Fix 11 RAPTOR full table

Estimated runtime:

- setup + Mistral pull: 15-35 min
- Fix 1: 45-120 min
- Fix 5: 1.5-4h
- Fix 11: 1-2.5h
- total: roughly 4-9h

Output:

- `/kaggle/working/revision_session1_outputs.zip`

Kaggle settings:

- Internet: ON
- Accelerator: GPU
- Use the fresh notebook, not the older patched notebook.

Best current run command in a fresh Kaggle cell:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d rag-hallucination-detection/.git ]; then
  git clone --branch main https://github.com/Saket-Maganti/rag-hallucination-detection.git
else
  git -C rag-hallucination-detection pull --ff-only origin main
fi
cd rag-hallucination-detection
bash scripts/kaggle_session1_fresh.sh
```

Faster gate-only run:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d rag-hallucination-detection/.git ]; then
  git clone --branch main https://github.com/Saket-Maganti/rag-hallucination-detection.git
else
  git -C rag-hallucination-detection pull --ff-only origin main
fi
cd rag-hallucination-detection
bash scripts/kaggle_fix1_only.sh
```

This writes `/kaggle/working/fix1_outputs.zip`.

Fastest T4 x2 gate-only run:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d rag-hallucination-detection/.git ]; then
  git clone --branch main https://github.com/Saket-Maganti/rag-hallucination-detection.git
else
  git -C rag-hallucination-detection pull --ff-only origin main
fi
cd rag-hallucination-detection
bash scripts/kaggle_fix1_parallel_t4x2.sh
```

This writes `/kaggle/working/fix1_parallel_outputs.zip`.

Ollama recovery command:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/rag-hallucination-detection
git pull --ff-only origin main
bash scripts/kaggle_ollama_guard.sh mistral
tail -n 80 /kaggle/working/ollama.log || true
ollama list
```

### Session 2 - Free GPU

Run Fix 2 headline cell:

```bash
python3 experiments/fix_02_scaled_headline_n500.py \
  --datasets squad \
  --n 500 \
  --seeds 41 42 43 44 45 \
  --backend ollama \
  --model mistral \
  --max_contexts 600 \
  --save_every 25
```

Expected free-GPU runtime: 8-16h.

If GPU sessions are unstable, split by seed:

```bash
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 41 --backend ollama --model mistral
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 42 --backend ollama --model mistral
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 43 --backend ollama --model mistral
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 44 --backend ollama --model mistral
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 45 --backend ollama --model mistral
```

### Session 3 - Free GPU

Run Fix 4 and Fix 6:

```bash
python3 experiments/fix_04_tau_generalization.py \
  --datasets squad pubmedqa hotpotqa naturalqs triviaqa \
  --taus 0.30 0.40 0.50 0.60 0.70 \
  --n 100 \
  --seed 42 \
  --backend ollama \
  --model mistral \
  --max_contexts 150 \
  --save_every 25

python3 experiments/fix_06_baseline_h2h_pareto.py \
  --datasets squad hotpotqa \
  --n 200 \
  --backend ollama \
  --model mistral \
  --max_contexts 250 \
  --raptor_clusters 6
```

Expected free-GPU runtime:

- Fix 4: 8-16h
- Fix 6 without Self-RAG: 2-4h

### Session 4 - After Fix 2

Run Fix 3 after Fix 2 outputs exist:

```bash
python3 experiments/fix_03_multimetric_faithfulness.py \
  --input data/revision/fix_02/per_query.csv \
  --second_nli_model vectara/hallucination_evaluation_model \
  --judge_backend ollama \
  --judge_model mistral \
  --build_human_eval \
  --human_n 100 \
  --save_every 50
```

Expected free-GPU runtime: 4-10h.

If Vectara HEM fails:

```bash
python3 experiments/fix_03_multimetric_faithfulness.py \
  --input data/revision/fix_02/per_query.csv \
  --second_nli_model roberta-large-mnli \
  --judge_backend ollama \
  --judge_model mistral \
  --build_human_eval \
  --human_n 100
```

## Reviewer Fix Checklist

Status meanings:

- Done: result has been generated and can be reported.
- Constructed: dataset/setup exists, but final expensive result pending.
- Code: script/log/LaTeX exists, execution pending.
- Budget-blocked: cannot honestly complete under zero-dollar constraint.

| Fix | Weakness | Current Status | Next Action |
| --- | --- | --- | --- |
| 1 | Causal-vs-correlational CCS claim | Done, H1 unsupported | Downgrade causal/mechanistic language. |
| 2 | Headline-cell sample size | Code | Run Session 2. |
| 3 | Single faithfulness metric | Code | Run after Fix 2; add human labels if possible. |
| 4 | Tau-tuning leakage | Code | Run Session 3. |
| 5 | SQuAD noise slope | Code | Run split Kaggle Fix 5 + Fix 11 notebook. |
| 6 | Baseline head-to-head | Code | Run Session 3 without Self-RAG first. |
| 7 | Independent 70B reproduction | Budget-blocked | Disclose zero-dollar limitation; do not fake it. |
| 8 | Info-theory overclaim | Code | Integrate after Fix 1 result. |
| 9 | Self-confidence partial correlation | Code | Run locally when input CSV exists. |
| 10 | Deployment scope | Code | Integrate after Fix 1 and long-form framing are known. |
| 11 | RAPTOR full table | Code | Run split Kaggle Fix 5 + Fix 11 notebook. |

## Immediate Next Step

Open this notebook in a fresh Kaggle T4 x2 session:

```text
notebooks/revision_fix5_11_kaggle_t4x2.ipynb
```

Run all cells top to bottom. When complete, download:

```text
/kaggle/working/fix5_11_t4x2_outputs.zip
```

Then copy/unzip the outputs into the local repo and update:

- `REVISION_SUMMARY.md`
- `experiments/fix_01_log.md`
- `experiments/fix_05_log.md`
- `experiments/fix_11_log.md`
- the corresponding `ragpaper/sections/revision/*.tex` files

## Caution

Do not report smoke-test outputs in the paper. Only report preregistered full
runs or explicitly label smaller runs as smoke/debug runs.
