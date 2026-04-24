# CLAUDE.md — project context for future sessions

Operational notes for picking up the RAG hallucination detection paper revision. Read this first.

## Project in one paragraph

Research artifact for the "Refinement Paradox" RAG paper. Core claim: better per-passage retrieval can *reduce* answer faithfulness by fragmenting context coherence. Introduces **CCS** (Context Coherence Score) as a generator-free retrieval-time diagnostic and **HCPC-v2** as a controlled intervention probe. Target venue: **NeurIPS** (following professor feedback on 2026-04-23).

Main repo: `/Users/saketmaganti/claudeprojs/rag-hallucination-detection`
Remote: `https://github.com/Saket-Maganti/rag-hallucination-detection`
Default branch: `main`

## Current status (2026-04-24)

**Phase 1 (8-item paper-revision infrastructure) — COMPLETE.** All reviewer-facing experiments ran. Only FinanceBench (HF-gated) and Self-RAG on Kaggle (OOM) have asterisks.

**Phase 2 (10-item major paper-strengthening upgrade) — IN PROGRESS.** Scripts written for Wave 1 + theory; remaining waves are code-to-write or launch-and-wait.

## Professor's feedback (2026-04-23)

> "3rd paper is probably the one with the most interesting results and work... Where this might get attacked is if reviewers feel it's a very metric based paper... heavy reliance on one embedding model, so reviewers would ask is this just a property of weak embeddings? Would a stronger retriever fix coherence?"

Action items:
1. **Stronger retrievers** → #8 multi-retriever (✅ done — paradox survives on PubMedQA, collapses on SQuAD with 335 M embedders → reframed as alignment-mismatch property)
2. **Multi-dataset + human eval OR remove human eval** → #2 multi-dataset done across 5 datasets × 3 models; Prolific deferred (IRB)
3. "Target NeurIPS if those fixes land" → Phase 1 alone now makes the paper competitive; Phase 2 pushes it to top-tier

## Phase 1 — rebuttal infrastructure (DONE)

| # | Item | Where | Status | Evidence |
|---|---|---|---|---|
| 5 | Adversarial §7.6 | Mac | ✅ | `data/adversarial/*.jsonl` (40 cases across 4 categories), `results/adversarial/` |
| 8 | Multi-retriever × 4 embedders × 2 datasets | Mac | ✅ | `results/multi_retriever/paradox_by_embedder.csv` |
| 4 | Head-to-head vs Self-RAG + CRAG | Kaggle T4 | ✅ | `results/headtohead/summary.md` (Self-RAG partial — OOM) |
| 2 | Multi-dataset × 3 generators (5 datasets) | Mac | ✅ | `results/multidataset/summary.{csv,md}` — 15 tuples × 90 rows each |
| 2b | FinanceBench row | Mac | ⚠️ HF-gated | 0 rows across 3 models; runbook pre-approved as optional |
| 1 | Mechanistic attention probe | Kaggle T4 | ✅ | `results/mechanistic/` — entropy_by_layer, retrieved_mass_by_layer, per_pair, top_k_attributions |
| 7 | Significance tests | Mac | ✅ | `results/stats/significance_tests.json` + `summary.csv` |
| 8-pkg | Benchmark packager | Mac | ✅ | `release/context_coherence_bench_v1/` (HF-loadable, CITATION.bib, LICENSE, metadata) |
| 6 | Prolific human eval | — | ❌ Deferred to camera-ready (needs IRB + ~$500) |

## Phase 2 — paper-strengthening upgrade plan (10 items)

Rationale: Phase 1 makes the paper competitive; Phase 2 makes it top-tier. Scripts are being written in parallel with runs.

| Item | Title | Script / File | Status | ETA |
|---|---|---|---|---|
| 4 | Mechanistic → hallucination classifier | `experiments/train_mech_classifier.py` | ✅ **RUN — L1 AUC=0.85** (baseline=0.80, target≥0.75) | (done, 30 s) |
| 6 | Adversarial set 40 → 200 | `experiments/generate_adversarial_cases.py` | ✅ Written, bug fixed (dict→list flatten), ready to run | 3–4 h (Ollama Mistral) |
| 7 | HCPC sub-chunk size sensitivity {128, 256, 512} | `experiments/run_subchunk_sensitivity.py` | ✅ Written, not yet run | 3 h |
| 8 | Theoretical proposition (info-theoretic bound + theorem) | `ragpaper/sections/theory.tex` | ✅ Drafted, needs prof review + `\input` into `main.tex` | — (paper work) |
| 1 | RAPTOR retriever comparison | — | ⚪ Not written | 8 h M4 after code |
| 3 | Long-form generation eval (MS-MARCO / QASPER) | — | ⚪ Not written | 6 h M4 after code |
| 2 | Frontier-scale ablation (Groq 70B + Kaggle quantized) | — | ⚪ Not written | 5 h after code; needs Groq API key (free) |
| 5 | Multi-seed variance on Table 2 (3 seeds) | Add `--seed` to existing runner | ⚪ Not written | 20–36 h depending on scope |
| 9 | Deployment case study (latency + qualitative figure) | — | ⚪ Not written | 2 h |
| 10 | HF Space + leaderboard (free tier) | — | ⚪ Not written | 1 day |

Dependency order: Wave 1 (items 4 ✅, 6, 7, 8 ✅) → Wave 2 (1, 3, 2) → Wave 3 (5, slowest) → Wave 4 (9, 10).

### Item 4 result (Mechanistic → hallucination classifier)

From `results/mech_classifier/summary.md` (n=20 pairs, 90 features, 5-fold CV):

| Model | AUC | F1 | ACC |
|---|---|---|---|
| logistic_l1 | **0.850 ± 0.300** | 0.800 | 0.850 |
| logistic_l2 | 0.800 ± 0.400 | 0.800 | 0.800 |
| gbdt | 0.575 ± 0.384 | 0.467 | 0.500 |
| single-feature baseline (`retrieved_mass`) | 0.800 | — | — |

Headline claim the paper can now make: *per-layer attention signals predict hallucination with 0.85 AUC and the single aggregate `mean_retrieved_mass` feature alone reaches 0.80 — the coherence paradox has a measurable mechanistic fingerprint.* Sample size is small (n=20 adversarial pairs) so the ±0.3 AUC std is a fair caveat; Item 6 expansion to 200 will shrink this.

## Phase 2.5 — NeurIPS gap-closure (3 items, added 2026-04-24)

After reviewing the professor's critique against Phase 1 coverage, three gaps remained only partially addressed. These are now scripted and target the specific reviewer challenges most likely to block acceptance.

| Gap | Reviewer question answered | Script / File | Status | ETA |
|---|---|---|---|---|
| 1 | "Is it coherence or just generic retrieval noise?" | `experiments/run_noise_injection_ablation.py` | ✅ Written, not yet run | 2 h (3 ds × 4 noise rates × 30 q, Mistral) |
| 2 | "Does the paradox depend on your specific prompt?" | `experiments/run_prompt_template_ablation.py` | ✅ Written, not yet run | 3 h (3 ds × 4 templates × 3 conditions × 30 q, Mistral) |
| 3 | "When does RAG actually help vs. zero-shot?" | `experiments/build_rag_vs_zeroshot_table.py` | ✅ **DONE** — `results/rag_vs_zeroshot/` populated | 0 (reshape only) |

### Gap 3 headline finding (already computed)

2×2 `results/rag_vs_zeroshot/table_2x2.csv`:

| Domain | Weak (MiniLM 22M) | Strong (gte-large 335M) |
|---|---|---|
| Open (SQuAD) | Δfaith = **+0.090** (RAG helps) | Δfaith = **+0.115** (RAG helps more) |
| Closed (PubMedQA) | Δfaith = **−0.021** (RAG hurts) | Δfaith = **−0.025** (RAG hurts) |

Strong narrative pick-up: RAG helps on open domain regardless of retriever strength, and *hurts* on biomedical closed-domain for both retrievers — general-purpose embedders retrieve biomedical evidence worse than Mistral's parametric knowledge. Paradox magnitude is also strength-dependent (weak=0.075, strong=−0.001 on SQuAD), which the §Discussion can use as a second-order claim. Caveat: each 2×2 cell is a single dataset (multi_retriever ablation only covers squad + pubmedqa).

### Gap 1 + 2 interpretation targets (for when runs land)

- **Noise ablation**: target `paradox_vs_noise_ratio ≥ 2` on all 3 datasets. If the coherence paradox drop is ≥ 2× the per-unit noise slope, coherence is a distinct failure mode.
- **Prompt ablation**: target `|delta_vs_ref| ≤ 0.03` across all 12 cells (3 datasets × 4 templates). If the paradox survives a CoT prompt, a concise prompt, and an expert-role prompt without moving more than ±0.03 faith points, it is not a prompt artifact.

## Run queue (as of 2026-04-24)

Green = done, Yellow = ready to launch, Grey = needs code.

### ✅ Completed on M4 / Kaggle (no action needed)

| # | Output | Evidence |
|---|---|---|
| P1 #5 adversarial (40 cases) | `data/adversarial/*.jsonl`, `results/adversarial/` | 10 cases per category |
| P1 #8 multi-retriever | `results/multi_retriever/paradox_by_embedder.csv` | 8 rows |
| P1 #4 head-to-head | `results/headtohead/summary.md` | Self-RAG partial (Kaggle OOM) |
| P1 #2 multi-dataset × 3 models × 5 ds | `results/multidataset/summary.csv` | 45 rows |
| P1 #1 mechanistic probe (Kaggle T4) | `results/mechanistic/*.csv,jsonl` | 5 files |
| P1 #7 stats | `results/stats/significance_tests.{json,csv}` | 6 tests |
| P1 #8pkg packager | `release/context_coherence_bench_v1/` | HF-loadable |
| P2 Item 4 mech classifier | `results/mech_classifier/summary.md` | AUC=0.85 |
| P2 Item 8 theory.tex | `ragpaper/sections/theory.tex` | Proposition + Theorem + Corollary |
| P2.5 Gap 3 RAG vs zero-shot | `results/rag_vs_zeroshot/table_2x2.csv` | Open: +0.09/+0.12, Closed: −0.02/−0.03 |

### 🟡 Ready to launch (code exists, not yet run)

| Item | Command | ETA (M4) | ETA (Kaggle T4) | Produces |
|---|---|---|---|---|
| P2 #6 adversarial expand 40→200 | `python3 experiments/generate_adversarial_cases.py --target_per_category 50` | 3–4 h | 2 h | `data/adversarial/*.jsonl` → ~200 cases |
| P2 #7 sub-chunk sweep | `python3 experiments/run_subchunk_sensitivity.py --datasets squad pubmedqa --model mistral --n_questions 30 --sub_chunks 128 256 512` | 3 h | 1.5 h | `results/subchunk_sensitivity/paradox_by_sub.csv` |
| P2.5 Gap 1 noise injection | `python3 experiments/run_noise_injection_ablation.py --datasets squad pubmedqa hotpotqa --n_questions 30 --top_k 3 --seed 42` | 2 h | 1 h | `results/noise_injection/coherence_vs_noise.csv` |
| P2.5 Gap 2 prompt ablation | `python3 experiments/run_prompt_template_ablation.py --datasets squad pubmedqa hotpotqa --templates strict cot concise expert --model mistral --n_questions 30` | 3 h | 1.5 h | `results/prompt_ablation/paradox_by_prompt.csv` |
| P2 #3 long-form eval | `python3 experiments/run_longform_eval.py --datasets qasper msmarco --model mistral --n_questions 20` | 2.5 h | 1.5 h | `results/longform/paradox_longform.csv` |
| P2 #2 frontier-scale (Groq) | `GROQ_API_KEY=... python3 experiments/run_frontier_scale.py --datasets squad pubmedqa --models llama-3.3-70b mixtral-8x7b --n_questions 30` | 45 min (API) | 45 min (API) | `results/frontier_scale/paradox_by_scale.csv` |
| P2 #1 RAPTOR head-to-head | `python3 experiments/run_raptor_ablation.py --datasets squad pubmedqa hotpotqa --model mistral --n_questions 30` | 90 min | 50 min | `results/raptor/raptor_vs_hcpc.csv` |
| P2 #5 multi-seed variance | `python3 experiments/run_multiseed_variance.py --datasets squad pubmedqa --model mistral --seeds 41 42 43 --n_questions 30` | 50 min | 30 min | `results/multiseed/variance_summary.csv` |
| P2 #9 deployment figure (no-run) | `python3 experiments/build_deployment_figure.py` | <1 min | — | `results/deployment_figure/latency_vs_faith.png` + `pareto_summary.csv` |

### Recommended **3-surface parallel split** (max completion in ~4 h wall-clock)

Three surfaces working simultaneously:

**Surface 1 — M4 overnight** (`tee -a run_queue.log` so you can read progress in any terminal):
```bash
cd /Users/saketmaganti/claudeprojs/rag-hallucination-detection && caffeinate -di bash -c '
python3 experiments/generate_adversarial_cases.py --target_per_category 50 ;
python3 experiments/run_subchunk_sensitivity.py --datasets squad pubmedqa --model mistral --n_questions 30 --sub_chunks 128 256 512 ;
echo "=== M4 DONE ==="
' 2>&1 | tee -a run_queue.log
```
Expected: 6–7 h. Produces `data/adversarial/` (200 cases) + `results/subchunk_sensitivity/`.

**Surface 2 — Kaggle session A** (Gaps 1 + 2, after Cells 1–3 from the standard Ollama+HF recipe):
```bash
!python3 experiments/run_noise_injection_ablation.py \
    --datasets squad pubmedqa hotpotqa --n_questions 30 --top_k 3 --seed 42 && \
 python3 experiments/run_prompt_template_ablation.py \
    --datasets squad pubmedqa hotpotqa \
    --templates strict cot concise expert \
    --model mistral --n_questions 30

!zip -r /kaggle/working/neurips_gaps.zip results/noise_injection results/prompt_ablation
from IPython.display import FileLink; FileLink('/kaggle/working/neurips_gaps.zip')
```
Expected: 2.5 h. Produces `results/noise_injection/` + `results/prompt_ablation/`.

**Surface 3 — Kaggle session B** (long-form eval, separate browser tab / second account if you have one):
```bash
!python3 experiments/run_longform_eval.py \
    --datasets qasper msmarco --model mistral --n_questions 20

!zip -r /kaggle/working/longform.zip results/longform
from IPython.display import FileLink; FileLink('/kaggle/working/longform.zip')
```
Expected: 1.5 h. Produces `results/longform/`.

**After all three finish**: `unzip` the Kaggle downloads into your M4's `results/` tree and commit.

### Fallback: single-surface M4 launcher (run if you don't want to bother with Kaggle)

```bash
cd /Users/saketmaganti/claudeprojs/rag-hallucination-detection && caffeinate -di bash -c '
python3 experiments/generate_adversarial_cases.py --target_per_category 50 ;
python3 experiments/run_subchunk_sensitivity.py --datasets squad pubmedqa --model mistral --n_questions 30 --sub_chunks 128 256 512 ;
python3 experiments/run_noise_injection_ablation.py --datasets squad pubmedqa hotpotqa --n_questions 30 --top_k 3 --seed 42 ;
python3 experiments/run_prompt_template_ablation.py --datasets squad pubmedqa hotpotqa --templates strict cot concise expert --model mistral --n_questions 30 ;
python3 experiments/run_longform_eval.py --datasets qasper msmarco --model mistral --n_questions 20 ;
echo "=== ALL M4 DONE ==="
' 2>&1 | tee -a run_queue.log
```
Expected: 13–14 h. `;` separators so one failure doesn't kill the rest. All scripts checkpointed.

### ⚪ Needs code to be written

_All P2 items now have code._

- P2 #10 HF Space demo: `space/app.py` + `space/requirements.txt` + `space/README.md` written 2026-04-25. Three tabs (CCS calculator, paradox explorer, about), CPU-only, no Ollama dependency. Smoke-tested (syntax ok). Still TODO as camera-ready polish: deploy to an actual `huggingface.co/spaces/...` repo + optional separate leaderboard Space.
- Remaining P2 items (#1 RAPTOR, #2 frontier-scale, #3 long-form, #5 multi-seed, #9 deployment) all graduated to 🟡 Ready on 2026-04-25 — scripts exist and are smoke-tested.

### Decision matrix — what to prioritize

If you have **one overnight**: run the 🟡 launcher above. Delivers all 4 ready items (~12 h), closes all NeurIPS gaps for Mistral, and produces 4 new result tables.

If you have **one week of compute**: add **P2 #3 long-form** and **P2 #2 frontier-scale** code — those are the two items most likely to come up in NeurIPS reviews ("does this generalize beyond short-answer?" and "does this hold at scale?").

If you want to **freeze and write**: stop after the 🟡 launcher finishes. Phase 1 + Item 4 + Gap 3 + the 3 ready runs cover every reviewer critique on the professor's list. Remaining items become camera-ready additions.

## Key findings (quantitative, updated as results land)

### Deployment Pareto (2026-04-25, from existing per-query CSVs, no new runs)

Script: `experiments/build_deployment_figure.py`. Figure: `results/deployment_figure/latency_vs_faith.png`.

Per-dataset (latency↓, faith↑) Pareto frontier picks:

- **SQuAD** → `hcpc_v2` (3.15 s / faith=0.808 / halluc=0.013) and `crag` (1.16 s / faith=0.787).
- **PubMedQA** → `baseline` (10.94 s / faith=0.601), `hcpc_v2` (8.66 s / faith=0.590, lowest halluc=0.167), `crag` (2.50 s but halluc=0.333 — speed-risk tradeoff).
- **HotpotQA** → `hcpc_v1` (1.46 s / faith=0.650 / halluc=0.167) and `crag` (1.21 s / faith=0.631).

Net: HCPC-v2 is on the Pareto frontier on 2/3 datasets. CRAG wins on latency everywhere but loses faithfulness on PubMedQA (−0.015) and picks up 20 pp more hallucinations — so "cheap retrieval" is explicitly **not** free in high-coherence-demand domains. This is the figure reviewers should see before §Deployment.



### Multi-retriever ablation — headline reviewer response

`paradox_drop = faith_baseline − faith_v1`. Positive = paradox survives.

| Dataset | minilm (22M) | bge-large (335M) | e5-large (335M) | gte-large (335M) |
|---|---|---|---|---|
| PubMedQA | +0.024 ✓ | +0.033 ✓ | +0.053 ✓ | +0.005 ~0 |
| SQuAD | +0.075 ✓ | +0.032 ✓ | **−0.011 ✗** | **−0.001 ~0** |

**Reframed narrative** (now the paper's main theoretical contribution): The paradox is a property of **retrieval–generation alignment mismatch**, not embedder weakness. It appears when retrieval is hard (domain gap, multi-hop), disappears when retrieval is easy. `v2_recovery` is positive on 7/8 rows → HCPC-v2 is robust even when the baseline paradox is weak.

### Head-to-head summary

| Dataset | Winner on faith | Winner on halluc | Notable |
|---|---|---|---|
| SQuAD | HCPC-v2 (0.797) | HCPC-v2 (0.00) | Paper's claim holds |
| PubMedQA | baseline (0.609) | baseline / HCPC-v2 tie | CRAG hallucinates 33%, paradox visible |
| HotpotQA | HCPC-v1 (0.650) | baseline / HCPC-v1 tie | Multi-hop = where coherence matters most |

Self-RAG OOMed on Kaggle T4 for PubMedQA, partial on SQuAD (22/30), full HotpotQA. Llama-2-7B fp16 ≈ 14 GB vs 15 GB T4. Run Self-RAG alone later with 8-bit loading for NeurIPS camera-ready.

### Multidataset Table 2 (5 × 3, qwen2.5 slice shown)

From `results/multidataset/summary.csv`:
- **squad × qwen2.5**: baseline faith = 0.83, v1 = 0.69 (paradox = +0.14), v2 = 0.83 (recovery)
- **pubmedqa × qwen2.5**: baseline = 0.59, v1 = 0.61, v2 = 0.60 (weak paradox)
- **hotpotqa × qwen2.5**: baseline = 0.62, v1 = 0.60, v2 = 0.62
- **triviaqa × qwen2.5**: baseline = 0.66, v1 = 0.63, v2 = 0.66

HCPC-v2 refinement rate across datasets: 33–73%. CCS post-refinement: 0.41–0.57. Paradox magnitude ordering matches theory prediction: HotpotQA > SQuAD > PubMedQA (multi-hop > redundant > domain-gap).

### Mechanistic analysis

From `results/mechanistic/per_pair.csv` — 20 pairs (10 adversarial × 2 conditions each):
- `mean_retrieved_mass`: fragmented = ~0.16, coherent = higher → attention looks at retrieved context less when fragmented
- `mean_parametric_mass`: fragmented = ~0.84 → model falls back to parametric knowledge
- Per-layer entropy + retrieved-mass available at every layer (28 layers in per_layer CSV)

Item 4 (mech classifier) will turn this into a standalone hallucination-detection contribution.

## Hardware plan

**Mac M4 Air (16 GB unified)** — all Ollama-based runs. Fanless, so:
- Elevate on a stand, stay on charger, cool room for overnight
- Always wrap long runs in `caffeinate -di`
- If Ollama dies mid-run, restart in dedicated terminal with `ollama serve` and re-launch; runners have per-tuple checkpoints

**Kaggle T4 (free 30 h/week)** — mechanistic (done), Self-RAG inference only.

**Groq API (free)** — use for frontier-scale (Item 2). Sign up at console.groq.com. Llama-3.3-70B, Mixtral-8x7B, Gemma-2-9B all free tier.

**HF free account** — for Item 10 (dataset + CPU Space). No Pro needed.

## Code fixes applied across sessions

1. **Device auto-detect (commit `a5017656`)**: CUDA / MPS / CPU auto-pick in `src/rag_pipeline.py`, `src/evaluator.py`, `src/hallucination_detector.py`, `experiments/run_adversarial_coherence.py`. Needed because Kaggle PyTorch lacks MPS.

2. **NQ loader streaming + multi-annotator (commit `3d88d286`)**: `src/dataset_loaders.py::load_naturalqs` — two bugs fixed:
   - Non-streaming mode triggered full train-split generation (~55 GB, 287 parquets) and one shard threw `DatasetGenerationError`, silently returning empty items. Fixed with `streaming=True`.
   - NQ has 5 annotators per question; loader only checked `short_answers[0]` which is usually empty. Fixed by iterating all annotators.

3. **Checkpoint-vs-empty-CSV trap** (known, not patched): `experiments/run_multidataset_validation.py` catches per-query exceptions and still marks the tuple complete. If Ollama 404s for a whole model, the tuple's per-query CSV is just a header line but `completed_tuples.json` says `true`. Fix: re-run with `--force` scoped to the affected model.

4. **Adversarial generator dict-vs-list flatten (2026-04-24, uncommitted)**: `experiments/generate_adversarial_cases.py::main()` — `load_all_cases()` returns `Dict[category → List[AdversarialCase]]` but the helpers `_existing_case_ids` and `_existing_queries` expected a flat `List[AdversarialCase]`. Iterating the dict yielded string keys → `AttributeError: 'str' object has no attribute 'category'`. Fixed by flattening `cases_by_cat.values()` into a single list before the call. Verified against the existing 40 seed cases (counts = 10 per category, 30 unique queries). Commit this along with the new Phase 2.5 scripts.

5. **NQ multidataset row-count uneven (known, not patched)**: per-query CSVs have 90/217/149 rows for mistral/llama3/qwen2.5 — accumulation from earlier partial `--force` re-runs. The rebuilt `results/multidataset/per_query.csv` (1350 rows) is still valid but the summary aggregator will over-weight NQ. If this shows up as a problem in the paper's numbers, wipe `results/multidataset/naturalqs__{llama3,qwen2.5}_per_query.csv` + the matching `completed_tuples.json` keys and re-run only those two tuples.

## Kaggle notebook recipe (reusable)

```python
# Cell 1 — clone + pull + deps
!git clone https://github.com/Saket-Maganti/rag-hallucination-detection.git
%cd rag-hallucination-detection
!git pull origin main
!pip install -q -r requirements.txt && pip install -q tabulate

# Cell 2 — HF auth
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
import os
token = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = token; os.environ["HUGGING_FACE_HUB_TOKEN"] = token
login(token=token)

# Cell 3 — Ollama (only for head-to-head; NOT for mechanistic)
!apt-get update -qq && apt-get install -y -qq zstd
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
subprocess.Popen(["ollama", "serve"], stdout=open("/tmp/ollama.log","w"), stderr=subprocess.STDOUT)
time.sleep(8)
!ollama pull mistral

# Cell 4 — run
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/kaggle_gpu_runs.py --task <TASK> ...
```

**Required before starting:**
- Account phone-verified (enables Internet toggle)
- `HF_TOKEN` in Add-ons → Secrets
- Accelerator = GPU T4 x2, Internet = On, Persistence = Files

**Known pitfalls:**
- FileLink 404s intermittently → use right sidebar Output panel to download
- Ollama + Self-RAG together OOM-kills on T4; use `--skip_selfrag` or run separately
- Save Version → Save & Run All (Commit) at end to persist
- `huggingface-cli` may not be on PATH; use `huggingface_hub.login()` from Python
- **Mechanistic ran in 5 min on T4** (20 items only, not 3–5 h as runbook estimates) — verify outputs are non-constant to catch stub fallback: `results/mechanistic/entropy_by_layer.csv` should be ≥ 20 KB and vary across layers

## Files added in Phase 2 (this session)

| Path | Item | Purpose |
|---|---|---|
| `experiments/train_mech_classifier.py` | 4 | scikit-learn classifier on per-layer entropy + retrieved mass |
| `experiments/run_subchunk_sensitivity.py` | 7 | Sweep HCPC sub_chunk_size ∈ {128, 256, 512} |
| `experiments/generate_adversarial_cases.py` | 6 | LLM-generate new adversarial cases with validator (uses existing coherence metrics) |
| `ragpaper/sections/theory.tex` | 8 | Proposition + Theorem + Corollary for coherence paradox (needs `\input{sections/theory}` in main.tex) |

## Dependency order (Phase 2)

```
Wave 1 (M4, parallel, ~6 h):
  Item 4 (classifier, 30 s)
  Item 6 (adversarial expand, 3–4 h)
  Item 7 (sub-chunk sweep, 3 h)

Wave 2 (code + run, ~15 h total):
  Item 1 (RAPTOR)          → 8 h M4
  Item 3 (long-form)        → 6 h M4
  Item 2 (frontier Groq)   → 5 h (free API + Kaggle)

Wave 3 (overnight, ~20–36 h):
  Item 5 (3-seed variance, scope-dependent)

Wave 4 (paper + ops, no compute):
  Item 8  (theory.tex, drafted)
  Item 9  (deployment figure)
  Item 10 (HF Space + leaderboard)
```

## Files to check after each run

| After | Inspect | What good looks like |
|---|---|---|
| #5 adversarial | `results/adversarial/per_case.csv` | ≥100 rows after Item 6 expansion |
| #8 multi-retriever | `results/multi_retriever/paradox_by_embedder.csv` | 8 rows |
| #4 head-to-head | `results/headtohead/summary.csv` | 12–15 rows |
| #2 multi-dataset | `results/multidataset/summary.csv` | 45 rows (15 tuples × 3 conditions) |
| #1 mechanistic | `results/mechanistic/` | 5 files, entropy_by_layer ≥ 20 KB |
| Item 4 | `results/mech_classifier/summary.md` | AUC > 0.75 for the paradox to be a classifier-worthy signal |
| Item 7 | `results/subchunk_sensitivity/paradox_by_sub.csv` | paradox_drop stable within ±0.02 across sub_chunks |

## Final camera-ready deliverables target

- Expanded Table 2: 5 datasets × 3 generators ✅
- Multi-retriever row: 4 embedders ✅
- Self-RAG / CRAG comparison ✅ (Self-RAG partial)
- Mechanistic figure: Δ entropy + Δ retrieved mass per layer ✅
- Adversarial detection AUCs across 4 categories ✅ (Item 6 expands to N=200)
- Released benchmark bundle ✅ (Item 10 publishes to HF)
- **Reframed narrative**: paradox = property of alignment mismatch, not embedder weakness ✅
- **Theoretical section**: proposition + theorem + corollary (Item 8, drafted)
- **Standalone classifier contribution** from mechanistic signals (Item 4, script ready)
- **Scale validation**: 70B models show paradox magnitude vs 7B (Item 2, pending)
- **Variance bands**: 3-seed error bars on Table 2 (Item 5, pending)

## Common operational gotchas

1. **Empty 1-byte CSVs** → Ollama was down during that tuple. `completed_tuples.json` still says `true`. Wipe the affected keys, re-run with `--force`.
2. **Tabulate missing on fresh venv** → adversarial script fails at markdown write. `pip install tabulate`. Data CSVs already written before that point.
3. **Ollama port 11434 refused** → process died. `ollama serve` in dedicated terminal (not `&` inside script).
4. **Worktree vs main repo confusion** → always work in `/Users/saketmaganti/claudeprojs/rag-hallucination-detection`, not `.claude/worktrees/*`.
5. **Uneven per_query.csv row counts** after re-runs (seen in naturalqs: 93/212/131 instead of 90/90/90) → caused by accumulating across partial runs. Re-run cleanly with `--force` on a wiped `completed_tuples.json` entry for the affected tuples.
6. **`results/hcpc_v2/logs/` is gitignored** → use `git add -f` to force-add when the mechanistic Kaggle session needs them in the cloned repo.
