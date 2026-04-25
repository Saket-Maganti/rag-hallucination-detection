# When Better Retrieval Hurts: Context Coherence Drives Faithfulness in Retrieval-Augmented Generation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9--3.12-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-30%2F30%20passing-brightgreen?style=flat-square)
![Queries](https://img.shields.io/badge/Total_Queries-7000+-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**A behavioral study of how retrieval-augmented language models fail when retrieved evidence loses internal coherence.**

</div>

## 📦 Released artifacts

| Resource | Where | What |
|---|---|---|
| 🐍 **Pip package** | `pip install context-coherence` | Standalone CCS metric + decision gate (pure-numpy core) |
| 📚 **HF Dataset** | [`saketmgnt/context-coherence-bench`](https://huggingface.co/datasets/saketmgnt/context-coherence-bench) | `load_dataset(...)`-able benchmark with 5 configs |
| 🎯 **Zenodo DOI** | [`10.5281/zenodo.19757291`](https://doi.org/10.5281/zenodo.19757291) | Permanent benchmark archive (citable) |
| 🎮 **Live demo** | [`huggingface.co/spaces/saketmgnt/sakkk`](https://huggingface.co/spaces/saketmgnt/sakkk) | Gradio app: CCS calculator + paradox explorer |
| 📓 **Colab tutorial** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saket-Maganti/rag-hallucination-detection/blob/main/notebooks/colab_tutorial.ipynb) | 6-cell walkthrough, < 5 min |
| 📄 **Paper** | [`ragpaper/main.pdf`](ragpaper/main.pdf) | NeurIPS 2026 submission, 64 pages |
| 🔌 **LangChain integration** | [`integrations/langchain/`](integrations/langchain/) | Drop-in `CoherenceGatedRetriever` |

---

## The failure mode: the Refinement Paradox

RAG research is built on an axiom: better retrieval $\Rightarrow$ more faithful answers. We show this axiom is unreliable, and in some regimes inverted.

We identify a fundamental failure mode — the **refinement paradox** — in which *improving per-passage retrieval quality actively degrades answer faithfulness*. The cause is not a bug in any particular retriever. It is a property of how language models reason over structured context: when the retrieved set is coherent, generators complete it faithfully; when refinement fragments the set, generators silently patch the missing connective evidence with parametric memory.

| Regime | Faithfulness | Hallucination | Mean Query-Passage Sim. |
|--------|-------------|---------------|-------------------------|
| Baseline (contiguity preserved) | 0.7995 | 0.0% | 0.634 |
| Aggressive refinement | **0.6407** | **10.0%** | 0.624 |
| Coherence-preserving perturbation | 0.7874 | 0.0% | 0.634 |

*SQuAD, Mistral-7B, chunk=1024, k=3, n=30 per condition.*

Three regimes, near-identical passage-level retrieval similarity, but a 15.9-percentage-point faithfulness gap. The residual is **context coherence** — a set-level property of retrieved evidence that standard RAG evaluation does not measure.

---

## Scientific contribution

This repository is the artifact for a paper whose contribution is **a failure mode**, not a pipeline. In detail:

1. **A new failure mode in RAG.** We name, characterize, and empirically demonstrate the *refinement paradox*: retrieval improvements that maximize per-passage relevance can degrade answer faithfulness by fragmenting the evidence structure the generator relies on.
2. **Context coherence as the mediating variable.** We argue — and provide converging evidence — that faithfulness in RAG is mediated by the structural coherence of the retrieved set, not by per-passage relevance. This reframes a retrieval-quality problem as a *generator-behavior* problem.
3. **Two controlled instruments to test the hypothesis.**
   - *HCPC-v2* — a selective-refinement **probe** that perturbs coherence in a targeted way. It is used here as a causal intervention, not as a proposed system.
   - *CCS* (Context Coherence Score) — a **generator-free retrieval-time diagnostic** that separates safe from unsafe contexts without answer-level evaluation.
4. **A stronger prediction, confirmed.** In a domain-mismatched regime, RAG produces more hallucination than zero-shot prompting ($2.5\times$ on PubMedQA). This follows directly from the coherence framework and is not predicted by standard RAG evaluation.

The code, experiments, and results in this repository exist to *support* these claims. HCPC-v2 and CCS are instruments, not products.

---

## Behavioral signatures

### 1. Retrieval similarity decouples from faithfulness

Regimes matched on passage-level similarity (0.634 vs. 0.624) produce a 0 pp vs. 10 pp hallucination gap. Per-passage relevance is not a sufficient statistic for RAG faithfulness.

### 2. Retrieval can be harmful, not merely insufficient

| Model | Zero-shot Halluc. | Best RAG Halluc. | Retrieval helps? |
|-------|-------------------|------------------|------------------|
| Mistral / SQuAD | 5.0% | 3.0% | yes |
| Mistral / PubMedQA | 8.0% | 20.0% | **no** |
| Llama-3 / SQuAD | 13.0% | 0.0% | yes |
| Llama-3 / PubMedQA | 4.0% | 10.0% | **no** |

A direct prediction of the coherence framework: when retrieval returns topically adjacent but semantically inconsistent passages, parametric memory — which is at least internally consistent — is safer than a low-coherence context.

### 3. Chunk size — an upstream determinant of coherence — dominates faithfulness variance

Among three tested configuration parameters (chunk size, top-$k$, prompt strategy), chunk size accounts for **90.1% of between-factor variance** on SQuAD (ANOVA $F=19.72$, $p<0.001$, Cohen's $d=1.41$). Prompt strategy accounts for <0.1%. Chunk size matters because it directly controls how much connective evidence survives retrieval — precisely the quantity the coherence framework flags as causal.

### 4. CCS tracks hallucination across regimes

$\text{CCS} = \overline{S} - \sigma(S)$, where $S_{ij}$ is pairwise chunk similarity. On SQuAD: CCS=1.00 coincides with 0% hallucination. On PubMedQA: CCS=0.57 coincides with 20% hallucination. CCS is computable at retrieval time without running the generator — a practical artifact of the scientific claim that coherence mediates faithfulness.

---

## Study design

Six-phase controlled evaluation, 6,050+ total queries. The design isolates the retrieval regime as the independent variable while holding the generator, scorer, and data fixed.

| Phase | Role in the argument | Queries |
|-------|----------------------|---------|
| 1 | Chunk size ablation — tests upstream coherence effect | 720 |
| 2 | Multi-model validation — rules out generator-specific explanations | 1440 |
| 3 | Rerank-only control — separates scoring from fragmentation | 240 |
| 4 | Zero-shot comparison — tests the "RAG is harmful" prediction | 350 |
| 5 | Threshold sweep — rules out threshold-tuning explanations | 3000 |
| 6 | Refinement-regime head-to-head — central causal test | 180 |

All conditions share the same embedding, retrieval store, generator, and NLI faithfulness scorer. The only varied quantity is how the retrieved set is assembled.

---

## Controlled instruments

```
Query
  |
  v
[all-MiniLM-L6-v2] --> ChromaDB (cosine) --> top-k chunks
                                                  |
                                +----------+-------+-------+----------+
                                |          |               |          |
                           Baseline   Rerank-only     Aggressive    Selective
                         (contiguity) (scoring only)  refinement   perturbation
                                                     (HCPC-v1)     (HCPC-v2 probe)
                                                  |
                                                  v
                                    [Mistral-7B / Llama-3-8B via Ollama]
                                                  |
                                                  v
                                    [DeBERTa-v3 NLI faithfulness scoring]
```

**HCPC-v2 as a coherence probe (not a proposed system).** It differs from aggressive refinement along three dimensions, each of which targets a specific coherence-destroying failure mode:
1. *AND-gated triggering* — prevents over-intervention on adequate passages
2. *Rank protection* — protects highest-evidence candidates from fragmentation
3. *Contiguity-preserving merge-back* — re-joins adjacent sub-chunks to restore connective evidence

If coherence is the causal mediator, this perturbation should recover faithfulness at fixed retrieval similarity. It does.

---

## Project structure

```
rag-hallucination-detection/
|-- main.py                        # Entry point (demo / eval / ablation modes)
|-- requirements.txt               # Python dependencies
|-- src/                           # Shared experimental substrate
|   |-- rag_pipeline.py            # RAG pipeline (chunking + generation)
|   |-- hcpc_retriever.py          # Aggressive-refinement regime (HCPC-v1)
|   |-- hcpc_v2_retriever.py       # Coherence-preserving probe (HCPC-v2)
|   |-- reranker.py                # Rerank-only control
|   |-- hallucination_detector.py  # NLI-based faithfulness measurement
|   |-- coherence_metrics.py       # CCS diagnostic + retrieval statistics
|   |-- data_loader.py             # SQuAD loader
|   |-- pubmedqa_loader.py         # PubMedQA loader
|   |-- evaluator.py               # Evaluation harness
|   |-- retrieval_metrics.py       # Retrieval-level metrics
|   |-- adaptive_chunker.py        # Adaptive chunking (supplementary)
|   |-- failure_logger.py          # Per-query failure logging
|   +-- ablation.py                # Ablation runner
|-- experiments/                   # One script per experimental phase
|-- scripts/                       # Analysis utilities
|-- results/                       # All experimental outputs by phase
```

The code is experimental scaffolding. The scientific content lives in `ragpaper/`.

---

## Reproducing the experiments

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) with Mistral-7B and Llama-3-8B pulled
- ~8 GB RAM (16 GB recommended with MPS acceleration)

### Installation

```bash
git clone https://github.com/Saket-Maganti/rag-hallucination-detection.git
cd rag-hallucination-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull mistral
ollama pull llama3
```

### Running a specific phase

```bash
# Phase 1 — chunk size ablation (upstream coherence effect)
python experiments/run_pubmedqa_ablation.py

# Phase 2 — multi-model validation (rules out model-specific causes)
python experiments/run_multimodel_ablation.py

# Phase 3 — rerank-only control (isolates scoring from fragmentation)
python experiments/run_reranker_experiment.py

# Phase 4 — zero-shot comparison (tests the "RAG is harmful" prediction)
python experiments/run_zeroshot_baseline.py

# Phase 5 — threshold sweep (rules out tuning artifacts)
python experiments/run_threshold_sensitivity.py

# Phase 6 — central causal test
python experiments/run_hcpc_v2_ablation.py

# Statistical tests
python experiments/run_significance_tests.py
```

---

## Experimental substrate

| Component | Implementation | Role |
|-----------|---------------|------|
| Embedding | all-MiniLM-L6-v2 (82M, 384-dim) | held fixed across conditions |
| Vector store | ChromaDB v0.4, cosine | held fixed |
| Generation | Mistral-7B + Llama-3-8B via Ollama | two families rule out model-specific causes |
| Cross-encoder | ms-marco-MiniLM-L-6-v2 (66M) | used inside refinement regimes |
| NLI scorer | DeBERTa-v3-base (435M) | constant measurement instrument |
| Acceleration | Apple Silicon MPS | reproducibility |
| Framework | LangChain + sentence-transformers + PyTorch | substrate, not contribution |

None of the scientific claims depend on these specific choices.

---

## What this work argues against

- **"Better retrieval $\Rightarrow$ more faithful RAG."** This is not reliably true. We show a direct counter-example.
- **"Aggressive post-retrieval refinement is safe by default."** It is not; the rerank-plus-sub-chunk-replacement recipe shipped in many RAG frameworks is a concrete instance of the refinement paradox.
- **"Retrieval metrics are a sufficient proxy for RAG quality."** They are not; two regimes matched on passage-level similarity can differ by 16 pp faithfulness.
- **"Prompt engineering is the first-order lever for faithfulness."** Chunk size (an upstream coherence determinant) accounts for ~450× more variance than prompt style among the parameters we tested.

---

## Paper-revision infrastructure 

The following experimental harnesses are built and committed. Items
1-7 are the original revision plan; **item 8 was added in response to
the Apr-2026 review** (multi-retriever ablation, the single critique
the original 7 items did not address). Most items are still pending
execution — code is in place so the expensive runs can be launched
once compute is allocated. See [`RUNBOOK.md`](RUNBOOK.md) for the
operational order, runtimes, and Kaggle vs M4 split.

| # | Addition | Code | Status |
|---|----------|------|--------|
| 1 | **Mechanistic interpretability** — attention-entropy probe on Mistral-7B with `output_attentions=True`; matched-pair protocol (adversarial controls and HCPC v1-vs-v2 contexts) reporting per-layer Δ entropy and Δ retrieved-mass | [`src/mechanistic.py`](src/mechanistic.py), [`experiments/run_mechanistic_analysis.py`](experiments/run_mechanistic_analysis.py) | code only — Kaggle GPU |
| 2 | **6-dataset validation** — SQuAD, PubMedQA, NaturalQuestions, TriviaQA, HotpotQA (distractor), FinanceBench. Checkpointed per (dataset, generator, condition) tuple for safe resume | [`src/dataset_loaders.py`](src/dataset_loaders.py), [`experiments/run_multidataset_validation.py`](experiments/run_multidataset_validation.py) | code only — M4 |
| 3 | **Third generator (Qwen2.5-7B-Instruct)** alongside Mistral-7B and Llama-3-8B | [`src/generators.py`](src/generators.py), [`scripts/setup_qwen.sh`](scripts/setup_qwen.sh) | model not yet pulled — M4 |
| 4 | **Head-to-head vs Self-RAG + CRAG** — wrapper around the published `selfrag/selfrag_llama2_7b` checkpoint (with reflection-token parsing) + faithful CRAG reimplementation (cross-encoder substitutes the unreleased T5 evaluator; substitution documented transparently). Five conditions: baseline, hcpc_v1, hcpc_v2, crag, selfrag | [`src/selfrag_wrapper.py`](src/selfrag_wrapper.py), [`src/crag_retriever.py`](src/crag_retriever.py), [`experiments/run_headtohead_comparison.py`](experiments/run_headtohead_comparison.py) | code only — Kaggle GPU |
| 5 | **Adversarial coherence cases** — 40 hand-authored items across 4 subsets (disjoint / contradict / drift / control). Per-signal detection AUC across embedding-coherence and NLI-pairwise signals | [`data/adversarial/*.jsonl`](data/adversarial), [`src/adversarial_cases.py`](src/adversarial_cases.py), [`experiments/run_adversarial_coherence.py`](experiments/run_adversarial_coherence.py), `compute_nli_pairwise()` in [`src/coherence_metrics.py`](src/coherence_metrics.py) | data + code, no results — M4 |
| 6 | **Human evaluation** — 500-item stratified sample → 1000 Prolific annotation tasks. Fleiss' κ (binary + ordinal) and Spearman NLI ↔ human correlation with prespecified paradox-survives-human check | [`experiments/prepare_prolific_study.py`](experiments/prepare_prolific_study.py), [`experiments/analyze_prolific_results.py`](experiments/analyze_prolific_results.py) | deferred to camera-ready (IRB + budget) |
| 7 | **ContextCoherenceBench release** — packager that bundles adversarial cases, multi-dataset per-query results, and human ratings into a HuggingFace-loadable layout with sha256 manifest | [`scripts/package_benchmark.py`](scripts/package_benchmark.py), [`data/benchmark/README.md`](data/benchmark/README.md) | runs after 1-5 — M4 |
| **8** | **Multi-retriever ablation (added in revision)** — Phase-6 central contrast (baseline / hcpc-v1 / hcpc-v2) replicated across MiniLM-L6 (82M, weak baseline), BGE-large (335M), E5-large (335M), GTE-large (335M). Directly answers the reviewer critique "is the paradox just a property of weak embeddings?" by holding everything else constant and varying only the dense retriever | [`src/embedders.py`](src/embedders.py), [`experiments/run_multi_retriever_ablation.py`](experiments/run_multi_retriever_ablation.py) | code only — M4 |

Execution order (dependency-first): `5 → {1, 8} → {2, 3} → 4 → 7`.
Item 6 (Prolific) is deferred to camera-ready per the reviewer's
"either fully execute or remove it" guidance.

Operational steps, wall-clock estimates, and the M4-vs-Kaggle split
live in [`RUNBOOK.md`](RUNBOOK.md). Full pipeline:
[`bash scripts/run_all_m4.sh`](scripts/run_all_m4.sh) on the M4, then
[`python3 scripts/kaggle_gpu_runs.py`](scripts/kaggle_gpu_runs.py) on
a Kaggle GPU notebook for items 1 and 4.

---

## References

1. Lewis et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS.
2. Rajpurkar et al. (2016). [SQuAD](https://arxiv.org/abs/1606.05250). EMNLP.
3. Jin et al. (2019). [PubMedQA](https://arxiv.org/abs/1909.06146). EMNLP.
4. Jiang et al. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825).
5. Meta AI (2024). [Llama 3](https://ai.meta.com/blog/meta-llama-3/).
6. He et al. (2021). [DeBERTaV3](https://arxiv.org/abs/2111.09543).
7. Reimers & Gurevych (2019). [Sentence-BERT](https://arxiv.org/abs/1908.10084). EMNLP.
8. Asai et al. (2024). [Self-RAG](https://arxiv.org/abs/2310.11511). ICLR.
9. Yan et al. (2024). [Corrective RAG](https://arxiv.org/abs/2401.15884).
10. Xu et al. (2023). [RECOMP](https://arxiv.org/abs/2310.04408).
