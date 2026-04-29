# NeurIPS Revision Execution Runbook

This runbook gives the exact commands for the senior-review revision suite.
Fix 1 is the gating experiment: run and interpret it before making any causal
claim in the paper. The remaining scripts are ready, but their paper language
should be wired only after Fix 1 is known.

For the high-level checklist and handoff decisions, see
[`codex.md`](codex.md). For the comprehensive revision book see
[`README.md`](README.md).

Current result state:

- Fix 1 is complete and H1 is unsupported.
- Fix 2 is complete (`7500` rows).
- Fix 3 is complete (`7500` rows), imported from the verified Kaggle T4 x2
  package.
- Fix 4 is complete (`7500` rows), imported from the same verified package.
- Fix 5 is complete (`1591` rows).
- Fix 9 has a limited local run: the no-control confidence association
  survives, but the available CSV lacks the similarity/redundancy controls.
- Fix 11 is complete (`300` rows).
- Fix 6 is the next compute run. Use
  `notebooks/revision_fix6_kaggle_t4x2_fresh.ipynb` on Kaggle T4 x2.

All commands assume:

```bash
cd /path/to/anonymized/repo
mkdir -p logs/revision
```

## Zero-Dollar Operating Mode

This plan assumes no paid APIs, no Groq quota burn, no Together.ai paid calls,
no OpenAI/Anthropic judge calls, and no paid Colab. Use only:

- M4 Air local Ollama.
- Free Kaggle/Colab GPU sessions when available.
- Free, local Hugging Face models for NLI/judging.
- Manual human annotation for the optional human-eval cell.

Important consequence: Fix 7, the independent 70B reproduction, is not
genuinely executable under a zero-dollar budget unless you already have free
70B access elsewhere. A 70B model does not fit on a normal free T4/P100/L4
notebook or an M4 Air. Keep the script, but mark the empirical fix as blocked
by budget if no free 70B endpoint/accelerator is available.

## Platform Guidance

Your M4 Air is useful for construction, analysis, CSV aggregation, paper patches,
and overnight local Mistral runs. It is not the fastest place to run thousands of
7B generations plus NLI scoring because sustained generation will throttle.

Recommended split:

| Platform | Best use | Avoid |
| --- | --- | --- |
| M4 Air | Fix 1 construction, Fix 8/10 paper patches, Fix 9 analysis, small smoke runs | Full n=500 x 5 headline generation unless you can leave it overnight/longer |
| Kaggle GPU / free Colab T4/P100/L4 | Local 7B generation, second-NLI scoring, RAPTOR/CRAG/Self-RAG CUDA runs | Depending on it for guaranteed uninterrupted long jobs |
| Free local/API alternatives | None assumed in zero-dollar mode | 70B reproduction and paid LLM-as-judge claims |

Quick accelerator check on hosted notebooks:

```bash
nvidia-smi
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## GPU / No-GPU Split

Here "GPU" means a hosted NVIDIA GPU such as free Kaggle/Colab T4, P100, L4,
or any free A100 you can access. Your M4 Air can run many commands locally, but
full generation-heavy runs will take much longer.

Session 1 fresh Kaggle notebook:
`notebooks/revision_session1_kaggle_fresh.ipynb`.
It runs Fix 1 generation/analyze, Fix 5, and Fix 11, then writes
`/kaggle/working/revision_session1_outputs.zip`.

Most robust Session 1 path: run the all-in-one script from one Kaggle cell:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d rag-hallucination-detection/.git ]; then
  git clone --branch main https://example.com/anonymized/repository.git
else
  git -C rag-hallucination-detection pull --ff-only origin main
fi
cd rag-hallucination-detection
bash scripts/kaggle_session1_fresh.sh
```

This runner prints timestamped section headers and a heartbeat every 60 seconds
for long jobs, including elapsed time, current CSV row count, expected row
count, and the last few Ollama log lines. If there is no heartbeat for several
minutes, the Kaggle kernel itself is likely stalled.

If you want the highest-impact result first, run only Fix 1:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d rag-hallucination-detection/.git ]; then
  git clone --branch main https://example.com/anonymized/repository.git
else
  git -C rag-hallucination-detection pull --ff-only origin main
fi
cd rag-hallucination-detection
bash scripts/kaggle_fix1_only.sh
```

This writes `/kaggle/working/fix1_outputs.zip` and should take roughly
45-120 minutes on a free T4 session after setup. Run the full Session 1 script
later for Fix 5 and Fix 11.

If Kaggle gives you **T4 x2**, use the two-GPU Fix 1 sharded runner instead:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d rag-hallucination-detection/.git ]; then
  git clone --branch main https://example.com/anonymized/repository.git
else
  git -C rag-hallucination-detection pull --ff-only origin main
fi
cd rag-hallucination-detection
bash scripts/kaggle_fix1_parallel_t4x2.sh
```

This starts two Ollama servers on ports `11434` and `11435`, pins one shard to
each T4 with `CUDA_VISIBLE_DEVICES`, merges the shard CSVs, analyzes Fix 1, and
writes `/kaggle/working/fix1_parallel_outputs.zip`. This is the fastest
zero-dollar path for the causal gate.

If Kaggle shows no live output, run Session 1 in the background and monitor the
log from a separate cell:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d rag-hallucination-detection/.git ]; then
  git clone --branch main https://example.com/anonymized/repository.git
else
  git -C rag-hallucination-detection pull --ff-only origin main
fi
cd rag-hallucination-detection
bash scripts/kaggle_session1_background.sh
```

Monitor cell:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/rag-hallucination-detection
git pull --ff-only origin main || true
bash scripts/kaggle_tail_session1.sh
```

If Ollama fails with `Connection refused`, run:

```bash
%%bash
set -euo pipefail
cd /kaggle/working/rag-hallucination-detection
git pull --ff-only origin main
bash scripts/kaggle_ollama_guard.sh mistral
tail -n 80 /kaggle/working/ollama.log || true
ollama list
```

Then rerun only the failed experiment cell. The guard restarts an unresponsive
Ollama process and verifies `/api/tags` before generation starts.

### Split Kaggle T4 x2 Run: Fix 5 + Fix 11 Only

Because the all-in-one notebook has been flaky in the Kaggle UI, the preferred
next cloud run is now split into a dedicated Fix 5 + Fix 11 notebook:

```text
notebooks/revision_fix5_11_kaggle_t4x2_fresh.ipynb
```

Kaggle settings:

- Internet: ON
- Accelerator: GPU
- GPU type: T4 x2

The notebook runs four cells:

1. kill stale Ollama/revision processes, clone/pull repo, and show `nvidia-smi`
2. setup dependencies, install/verify Ollama, and start two Ollama servers
3. run Fix 5 on GPU0 and Fix 11 on GPU1 in parallel
4. status/package outputs
5. optional debug cell if setup fails

The robust command used by the notebook is:

```bash
python -u scripts/kaggle_stream_fix5_11_t4x2.py --stage setup --heartbeat 15
python -u scripts/kaggle_stream_fix5_11_t4x2.py --stage parallel --heartbeat 30
python -u scripts/kaggle_stream_fix5_11_t4x2.py --stage status --heartbeat 15
python -u scripts/kaggle_stream_fix5_11_t4x2.py --stage package --heartbeat 15
```

Outputs:

```text
/kaggle/working/fix5_11_t4x2_outputs.zip
```

Expected runtime after setup:

- Fix 5: roughly 1.5-3h
- Fix 11: roughly 45-90 min
- parallel wall time: roughly 2.5-4.5h including overhead

The wrapper prints a heartbeat even if Bash, downloads, Ollama, or generation
are quiet. Heartbeats include row counts for Fix 5 final/partial CSVs and Fix
11 final/per-dataset partial CSVs.

### Does Not Need Hosted GPU

Run these on the M4 Air:

```bash
# Validation
python3 -m py_compile \
  experiments/fix_01_causal_matched_pairs.py \
  experiments/fix_02_scaled_headline_n500.py \
  experiments/fix_03_multimetric_faithfulness.py \
  experiments/fix_04_tau_generalization.py \
  experiments/fix_05_coherence_preserving_noise.py \
  experiments/fix_06_baseline_h2h_pareto.py \
  experiments/fix_07_together_70b_reproduction.py \
  experiments/fix_09_partial_correlations.py \
  experiments/fix_11_raptor_full_table.py \
  experiments/revision_utils.py \
  src/ragas_scorer.py src/together_llm.py src/vectara_hem_scorer.py \
  src/openai_llm.py src/anthropic_llm.py

python3 -m pytest tests/test_ccs.py pip-package/tests/test_core.py

# Fix 1 construction, already completed for the primary run.
python3 experiments/fix_01_causal_matched_pairs.py \
  --stage construct \
  --dataset squad \
  --n_target 200 \
  --seed 42 \
  --max_contexts 400 \
  --candidate_limit 400 \
  --run_tag primary_n200

# Fix 1 analysis, after generation output exists.
python3 experiments/fix_01_causal_matched_pairs.py --stage analyze

# Fix 3 human-agreement analysis, after manual labels exist.
python3 experiments/fix_03_multimetric_faithfulness.py \
  --input data/revision/fix_02/per_query.csv \
  --skip_second_nli \
  --skip_ragas \
  --human_rated_path data/revision/fix_03/human_eval_rated.jsonl

# Fix 9 partial correlations.
python3 experiments/fix_09_partial_correlations.py \
  --input results/confidence_calibration/per_query.csv \
  --ccs_col ccs \
  --confidence_col self_confidence \
  --mean_sim_col mean_retrieval_similarity \
  --redundancy_col passage_redundancy

# Fix 8 and Fix 10 are paper-only patch review/integration.
sed -n '1,240p' ragpaper/sections/revision/fix_08_theory_reframe.tex
sed -n '1,240p' ragpaper/sections/revision/fix_10_scope_deployment.tex
```

No-hosted-GPU runtime:

| Task | M4 Air runtime |
| --- | ---: |
| Validation tests | 1-3 minutes |
| Fix 1 construction | already done; otherwise 10-40 minutes |
| Fix 1 analysis | seconds-minutes |
| Fix 3 human-agreement analysis | seconds-minutes |
| Fix 8 / Fix 10 paper patches | 30-90 minutes manual integration |
| Fix 9 partial correlations | seconds-minutes |

### Hosted GPU Strongly Recommended

These can run on the M4 Air, but use free Kaggle/Colab GPU if available:

```bash
# Fix 1 generation/NLI.
python3 experiments/fix_01_causal_matched_pairs.py \
  --stage generate \
  --backend ollama \
  --model mistral \
  --resume \
  --save_every 10 \
  --progress_every 10

# Fix 2 headline n=500 x 5 seeds.
python3 experiments/fix_02_scaled_headline_n500.py \
  --datasets squad \
  --n 500 \
  --seeds 41 42 43 44 45 \
  --backend ollama \
  --model mistral \
  --max_contexts 600 \
  --save_every 25

# Fix 3 second-NLI + local judge over Fix 2 outputs.
python3 experiments/fix_03_multimetric_faithfulness.py \
  --input data/revision/fix_02/per_query.csv \
  --second_nli_model vectara/hallucination_evaluation_model \
  --judge_backend ollama \
  --judge_model mistral \
  --build_human_eval \
  --human_n 100 \
  --save_every 50

# Fix 4 tau-generalization matrix.
python3 experiments/fix_04_tau_generalization.py \
  --datasets squad pubmedqa hotpotqa naturalqs triviaqa \
  --taus 0.30 0.40 0.50 0.60 0.70 \
  --n 100 \
  --seed 42 \
  --backend ollama \
  --model mistral \
  --max_contexts 150 \
  --save_every 25

# Fix 5 coherence-preserving noise.
python3 experiments/fix_05_coherence_preserving_noise.py \
  --n 200 \
  --seed 42 \
  --backend ollama \
  --model mistral \
  --max_contexts 300 \
  --n_noise 1 2 3 \
  --save_every 25

# Fix 6 baselines without Self-RAG.
python3 experiments/fix_06_baseline_h2h_pareto.py \
  --datasets squad hotpotqa \
  --n 200 \
  --backend ollama \
  --model mistral \
  --max_contexts 250 \
  --raptor_clusters 6

# Fix 11 RAPTOR full table.
python3 experiments/fix_11_raptor_full_table.py \
  --datasets squad pubmedqa hotpotqa \
  --n 100 \
  --backend ollama \
  --model mistral \
  --max_contexts 150 \
  --raptor_clusters 6
```

Recommended-hosted-GPU runtime:

| Task | M4 Air | Free Kaggle/Colab GPU |
| --- | ---: | ---: |
| Fix 1 generation/NLI | 2-5h | 45-120m |
| Fix 2 headline n=500 x 5 | 18-36h | 8-16h |
| Fix 3 second-NLI + local judge | 12-24h | 4-10h |
| Fix 4 tau matrix | 18-36h | 8-16h |
| Fix 5 noise slope | 4-8h | 1.5-4h |
| Fix 6 baselines without Self-RAG | 4-8h | 2-4h |
| Fix 11 RAPTOR table | 2-5h | 1-2.5h |

### Hosted GPU Required Or Budget-Blocked

```bash
# Fix 6 Self-RAG path: requires CUDA-class GPU and enough VRAM.
python3 experiments/fix_06_baseline_h2h_pareto.py \
  --datasets squad hotpotqa \
  --n 200 \
  --backend ollama \
  --model mistral \
  --max_contexts 250 \
  --raptor_clusters 6 \
  --include_selfrag \
  --selfrag_model selfrag/selfrag_llama2_7b \
  --selfrag_8bit
```

Fix 7 independent 70B reproduction is budget-blocked in zero-dollar mode unless
you get free 70B-capable compute. Do not run it on the M4 Air or ordinary free
T4/P100/L4 sessions.

## Common Setup

Local M4 Air:

```bash
brew install ollama || true
ollama pull mistral
ollama serve
```

Kaggle Ollama setup note: use `bash scripts/kaggle_ollama_guard.sh mistral`.
It installs `zstd` if needed, starts or restarts Ollama, verifies the local
API, and pulls Mistral. If the install cell fails with `requires zstd for
extraction`, the guard handles it.

Python environment:

```bash
cd /path/to/anonymized/repo
python3 -m venv .venv-revision
source .venv-revision/bin/activate
python3 -m pip install -U pip
if [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt; fi
python3 -m pip install -e pip-package
python3 -m pip install scipy pandas numpy scikit-learn matplotlib seaborn
python3 -m pip install sentence-transformers transformers torch datasets
```

On Kaggle, always `cd /kaggle/working/rag-hallucination-detection` before
running dependency or experiment cells. The Session 1 notebook does this
explicitly after the latest patch.

API keys:

```bash
# Zero-dollar mode does not require API keys.
# Leave GROQ_API_KEY, TOGETHER_API_KEY, OPENAI_API_KEY, and ANTHROPIC_API_KEY unset.
```

Validation:

```bash
python3 -m py_compile \
  experiments/fix_01_causal_matched_pairs.py \
  experiments/fix_02_scaled_headline_n500.py \
  experiments/fix_03_multimetric_faithfulness.py \
  experiments/fix_04_tau_generalization.py \
  experiments/fix_05_coherence_preserving_noise.py \
  experiments/fix_06_baseline_h2h_pareto.py \
  experiments/fix_07_together_70b_reproduction.py \
  experiments/fix_09_partial_correlations.py \
  experiments/fix_11_raptor_full_table.py \
  experiments/revision_utils.py \
  src/ragas_scorer.py src/together_llm.py src/vectara_hem_scorer.py \
  src/openai_llm.py src/anthropic_llm.py

python3 -m pytest tests/test_ccs.py pip-package/tests/test_core.py
```

## Fix 1: Causal Coherence Intervention

Construction has already completed for the preregistered primary run:

```bash
python3 experiments/fix_01_causal_matched_pairs.py \
  --stage construct \
  --dataset squad \
  --n_target 200 \
  --seed 42 \
  --max_contexts 400 \
  --candidate_limit 400 \
  --run_tag primary_n200 \
  2>&1 | tee logs/revision/fix_01_construct.log
```

Observed construction result: 200 matched pairs, zero skips, max mean-similarity
gap 0.018512, mean CCS gap 0.532634.

Run generation/NLI locally on M4 Air:

```bash
python3 experiments/fix_01_causal_matched_pairs.py \
  --stage generate \
  --backend ollama \
  --model mistral \
  --resume \
  --save_every 10 \
  --progress_every 10 \
  2>&1 | tee logs/revision/fix_01_generate_m4_mistral.log
```

Free GPU notebook option: run the same command on Kaggle/Colab after pulling
Mistral or using the repo's local model setup. This keeps the backend
zero-cost and avoids all API quota.

Analyze after generation:

```bash
python3 experiments/fix_01_causal_matched_pairs.py \
  --stage analyze \
  2>&1 | tee logs/revision/fix_01_analyze.log
```

Estimated runtime:

| Platform | Runtime |
| --- | ---: |
| M4 Air, Ollama Mistral | 2-5 hours for 400 generations plus NLI |
| Kaggle/Colab T4 or L4, local 7B | 45-120 minutes after setup |
| Free A100, if you happen to get one | 20-45 minutes |

Decision rule: keep causal/mechanistic framing only if Wilcoxon, bootstrap CI,
and effect-size criteria in [`status.md`](status.md) pass. If not, downgrade.

## Fix 2: Headline-Cell Rigor Upgrade

Primary full command:

```bash
python3 experiments/fix_02_scaled_headline_n500.py \
  --datasets squad \
  --n 500 \
  --seeds 41 42 43 44 45 \
  --backend ollama \
  --model mistral \
  --max_contexts 600 \
  --save_every 25 \
  2>&1 | tee logs/revision/fix_02_squad_n500x5_mistral.log
```

Parallel seed split, useful on multiple notebooks or terminals:

```bash
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 41 --backend ollama --model mistral 2>&1 | tee logs/revision/fix_02_seed41.log
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 42 --backend ollama --model mistral 2>&1 | tee logs/revision/fix_02_seed42.log
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 43 --backend ollama --model mistral 2>&1 | tee logs/revision/fix_02_seed43.log
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 44 --backend ollama --model mistral 2>&1 | tee logs/revision/fix_02_seed44.log
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 500 --seeds 45 --backend ollama --model mistral 2>&1 | tee logs/revision/fix_02_seed45.log
```

Estimated runtime:

| Platform | Runtime |
| --- | ---: |
| M4 Air | 18-36 hours |
| Kaggle/Colab T4 or L4 | 8-16 hours |
| Free A100, if available | 3-6 hours |

Outputs:

- `data/revision/fix_02/per_query.csv`
- `results/revision/fix_02/headline_table.csv`
- `results/revision/fix_02/paired_contrasts.csv`

## Fix 3: Multi-Metric Faithfulness

Run after Fix 2 completes. In zero-dollar mode, use a local Ollama judge. This
is slower and less reviewer-convincing than GPT-4o-mini/Claude, so lean on the
second NLI model plus the optional human eval to strengthen the section.

```bash
python3 experiments/fix_03_multimetric_faithfulness.py \
  --input data/revision/fix_02/per_query.csv \
  --second_nli_model vectara/hallucination_evaluation_model \
  --judge_backend ollama \
  --judge_model mistral \
  --build_human_eval \
  --human_n 100 \
  --save_every 50 \
  2>&1 | tee logs/revision/fix_03_multimetric_local.log
```

If Vectara HEM is unavailable, use MNLI:

```bash
python3 experiments/fix_03_multimetric_faithfulness.py \
  --input data/revision/fix_02/per_query.csv \
  --second_nli_model roberta-large-mnli \
  --judge_backend ollama \
  --judge_model mistral \
  --build_human_eval \
  --human_n 100 \
  2>&1 | tee logs/revision/fix_03_multimetric_mnli_local.log
```

Human ratings, after two annotators fill the template:

```bash
python3 experiments/fix_03_multimetric_faithfulness.py \
  --input data/revision/fix_02/per_query.csv \
  --skip_second_nli \
  --skip_ragas \
  --human_rated_path data/revision/fix_03/human_eval_rated.jsonl \
  2>&1 | tee logs/revision/fix_03_human_agreement.log
```

Estimated runtime:

| Platform | Runtime |
| --- | ---: |
| M4 Air, local judge | 12-24 hours |
| Free Colab/Kaggle GPU, local judge | 4-10 hours |
| Free A100, if available | 1-3 hours |

## Fix 4: Cross-Dataset Tau Generalization

Primary command:

```bash
python3 experiments/fix_04_tau_generalization.py \
  --datasets squad pubmedqa hotpotqa naturalqs triviaqa \
  --taus 0.30 0.40 0.50 0.60 0.70 \
  --n 100 \
  --seed 42 \
  --backend ollama \
  --model mistral \
  --max_contexts 150 \
  --save_every 25 \
  2>&1 | tee logs/revision/fix_04_tau_matrix.log
```

Parallel dataset split:

```bash
for ds in squad pubmedqa hotpotqa naturalqs triviaqa; do
  python3 experiments/fix_04_tau_generalization.py \
    --datasets "$ds" \
    --taus 0.30 0.40 0.50 0.60 0.70 \
    --n 100 \
    --seed 42 \
    --backend ollama \
    --model mistral \
    2>&1 | tee "logs/revision/fix_04_${ds}.log"
done
```

Estimated runtime:

| Platform | Runtime |
| --- | ---: |
| M4 Air | 18-36 hours |
| Kaggle/Colab T4 or L4 | 8-16 hours |
| Free A100, if available | 3-7 hours |

## Fix 5: Coherence-Preserving Noise Slope

```bash
python3 experiments/fix_05_coherence_preserving_noise.py \
  --n 200 \
  --seed 42 \
  --backend ollama \
  --model mistral \
  --max_contexts 300 \
  --n_noise 1 2 3 \
  --save_every 25 \
  2>&1 | tee logs/revision/fix_05_noise_slope.log
```

Estimated runtime:

| Platform | Runtime |
| --- | ---: |
| M4 Air | 4-8 hours |
| Kaggle/Colab T4 or L4 | 1.5-4 hours |
| Free A100, if available | 45-120 minutes |

## Fix 6: Proper Baseline Head-to-Head

Run HCPC-v2, CRAG, and RAPTOR-2L:

```bash
python3 experiments/fix_06_baseline_h2h_pareto.py \
  --datasets squad hotpotqa \
  --n 200 \
  --backend ollama \
  --model mistral \
  --max_contexts 250 \
  --raptor_clusters 6 \
  2>&1 | tee logs/revision/fix_06_baselines_no_selfrag.log
```

Run with Self-RAG on CUDA:

```bash
python3 experiments/fix_06_baseline_h2h_pareto.py \
  --datasets squad hotpotqa \
  --n 200 \
  --backend ollama \
  --model mistral \
  --max_contexts 250 \
  --raptor_clusters 6 \
  --include_selfrag \
  --selfrag_model selfrag/selfrag_llama2_7b \
  --selfrag_8bit \
  2>&1 | tee logs/revision/fix_06_baselines_with_selfrag.log
```

Estimated runtime:

| Platform | Runtime |
| --- | ---: |
| M4 Air, no Self-RAG | 4-8 hours |
| Kaggle/Colab GPU, no Self-RAG | 2-4 hours |
| Free A100, if available, with Self-RAG | 3-6 hours |

## Fix 7: Independent 70B Reproduction

Zero-dollar status: blocked unless you already have free 70B access. The
Together.ai script is preserved for reproducibility, but do not run it under
the no-spend constraint.

Budget-blocked command, for documentation only:

```bash
python3 experiments/fix_07_together_70b_reproduction.py \
  --n 100 \
  --seed 42 \
  --max_contexts 200 \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --reference_magnitude 0.100 \
  2>&1 | tee logs/revision/fix_07_together_70b.log
```

Zero-dollar paper action: state that independent 70B reproduction is not run
under the revision budget, and do not rely on the original 70B result as a
central claim.

## Fix 8: Theory Reframing

No compute command. Wire the LaTeX patch after Fix 1 is known:

```bash
sed -n '1,240p' ragpaper/sections/revision/fix_08_theory_reframe.tex
```

Runtime: minutes of manual integration.

## Fix 9: Self-Confidence Partial Correlations

Use the confidence calibration output if present:

```bash
python3 experiments/fix_09_partial_correlations.py \
  --input results/confidence_calibration/per_query.csv \
  --ccs_col ccs \
  --confidence_col self_confidence \
  --mean_sim_col mean_retrieval_similarity \
  --redundancy_col passage_redundancy \
  2>&1 | tee logs/revision/fix_09_partial_correlations.log
```

If the confidence run uses different column names, change the column flags
rather than editing the script.

Estimated runtime: seconds to a few minutes.

## Fix 10: Scope Deployment Claim

No compute command. Wire the LaTeX patch after Fix 1 and long-form results are
known:

```bash
sed -n '1,240p' ragpaper/sections/revision/fix_10_scope_deployment.tex
```

Runtime: minutes of manual integration.

## Fix 11: RAPTOR Full Table

```bash
python3 experiments/fix_11_raptor_full_table.py \
  --datasets squad pubmedqa hotpotqa \
  --n 100 \
  --backend ollama \
  --model mistral \
  --max_contexts 150 \
  --raptor_clusters 6 \
  2>&1 | tee logs/revision/fix_11_raptor_full_table.log
```

Estimated runtime:

| Platform | Runtime |
| --- | ---: |
| M4 Air | 2-5 hours |
| Kaggle/Colab T4 or L4 | 1-2.5 hours |
| Free A100, if available | 30-90 minutes |

## Suggested Execution Order

1. Run Fix 6 no-Self-RAG on Kaggle T4 x2 and download/package immediately.
2. Optionally run Fix 6 Self-RAG only after the smoke test passes.
3. Decide whether to regenerate the confidence-calibration CSV with
   `mean_retrieval_similarity` and `passage_redundancy`; otherwise report Fix 9
   as suggestive only.
4. Wire Fix 8 and Fix 10 into the paper because Fix 1's null result requires
   scoped predictive/diagnostic language.
5. Update paper tables and narrative for completed Fixes 1-5, 9, and 11, then
   add Fix 6 when its package is downloaded and verified.
6. Keep Fix 7 budget-blocked unless genuinely free 70B-capable compute appears.

## Fast Smoke Tests

Use these before committing to long runs:

```bash
python3 experiments/fix_01_causal_matched_pairs.py --stage construct --n_target 5 --max_contexts 20 --candidate_limit 20 --run_tag smoke
python3 experiments/fix_02_scaled_headline_n500.py --datasets squad --n 5 --seeds 41 --backend ollama --model mistral --max_contexts 20
python3 experiments/fix_04_tau_generalization.py --datasets squad pubmedqa --taus 0.40 0.50 --n 5 --backend ollama --model mistral --max_contexts 20
python3 experiments/fix_05_coherence_preserving_noise.py --n 5 --backend ollama --model mistral --max_contexts 20
python3 experiments/fix_06_baseline_h2h_pareto.py --datasets squad --n 5 --backend ollama --model mistral --max_contexts 20
python3 experiments/fix_11_raptor_full_table.py --datasets squad --n 5 --backend ollama --model mistral --max_contexts 20
```

Smoke outputs should not be reported in the paper.
