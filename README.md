# When Retrieval Improvements Hurt: Context Coherence and Faithfulness in RAG

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange?style=flat-square)
![Queries](https://img.shields.io/badge/Total_Queries-6050+-red?style=flat-square)

**A controlled empirical study of how post-retrieval refinement, chunk size, and context coherence affect hallucination in Retrieval-Augmented Generation**

</div>

---

## The Refinement Paradox

RAG systems assume better retrieval yields more faithful answers. We show this assumption is unreliable:

| Condition | Faithfulness | Hallucination | Mean Similarity |
|-----------|-------------|---------------|-----------------|
| Baseline (no refinement) | 0.7995 | 0.0% | 0.634 |
| HCPC-v1 (aggressive refinement) | 0.6407 | 10.0% | 0.624 |
| **HCPC-v2 (selective refinement)** | **0.7874** | **0.0%** | **0.634** |

*SQuAD, Mistral-7B, chunk=1024, k=3, n=30 per condition*

HCPC-v1 refines 100% of queries and **degrades** faithfulness by 15.9pp despite maintaining retrieval similarity. HCPC-v2 recovers by intervening on only 16.7% of queries. The mechanism: aggressive refinement fragments context coherence, forcing the generator to fill gaps from parametric memory.

---

## Key Findings

### 1. Chunk size dominates faithfulness variance

Among the three configuration parameters tested (chunk size, top-k, prompt strategy), chunk size accounts for **90.1% of between-factor variance** on SQuAD (ANOVA: F=19.72, p<0.001, Cohen's d=1.41). Prompt strategy accounts for <0.1%.

### 2. RAG can hurt in specialized domains

| Model | Zero-shot Halluc. | Best RAG Halluc. | RAG Helps? |
|-------|-------------------|------------------|------------|
| Mistral / SQuAD | 5.0% | 3.0% | Yes |
| Mistral / PubMedQA | 8.0% | 20.0% | **No** |
| Llama-3 / SQuAD | 13.0% | 0.0% | Yes |
| Llama-3 / PubMedQA | 4.0% | 10.0% | **No** |

On PubMedQA, a general-domain encoder retrieves topically adjacent but semantically inconsistent passages, producing more hallucinations than zero-shot prompting.

### 3. Model architectures respond differently to retrieval configuration

| Model | PubMedQA: Chunk Size | PubMedQA: Top-k |
|-------|---------------------|-----------------|
| Mistral-7B | 1.5% | **97.9%** |
| Llama-3-8B | **95.2%** | 0.4% |

*Proportion of between-factor faithfulness variance*

A single set of "RAG best practices" does not transfer across model families.

### 4. Context Coherence Score (CCS) separates safe from dangerous contexts

CCS = mean pairwise chunk similarity - standard deviation. On SQuAD: CCS=1.0 coincides with 0% hallucination. On PubMedQA: CCS=0.57 coincides with 20% hallucination. CCS is computable at retrieval time without generation.

---

## Study Design

Six experimental phases, 6,050+ total queries:

| Phase | Focus | Queries | Primary Variable |
|-------|-------|---------|-----------------|
| 1 | Chunk size ablation | 720 | chunk size {256, 512, 1024} |
| 2 | Multi-model validation | 1440 | Mistral-7B vs Llama-3-8B |
| 3 | Re-ranking study | 240 | with/without cross-encoder |
| 4 | Zero-shot baseline | 350 | retrieval vs no retrieval |
| 5 | Threshold sweep | 3000 | similarity and CE thresholds |
| 6 | HCPC head-to-head | 180 | baseline vs HCPC-v1 vs HCPC-v2 |

All conditions share the same pipeline, embedding model, and inference stack. Queries are sampled randomly and held fixed across conditions within each phase.

---

## Architecture

```
Query
  |
  v
[all-MiniLM-L6-v2] --> ChromaDB (cosine) --> top-k chunks
                                                  |
                                    +--------------+--------------+
                                    |              |              |
                                Baseline      HCPC-v1        HCPC-v2
                              (no refine)   (OR-gate,      (AND-gate,
                                             100% refine)   rank prot,
                                                            merge-back)
                                    |              |              |
                                    +--------------+--------------+
                                                  |
                                                  v
                                    [Mistral-7B / Llama-3-8B via Ollama]
                                                  |
                                                  v
                                    [DeBERTa-v3 NLI faithfulness scoring]
```

**HCPC-v2 decision logic:**
1. **AND-gated detection** -- both cosine similarity AND cross-encoder must flag a chunk before refinement triggers
2. **Rank protection** -- top-2 chunks are never refined
3. **Contiguity-preserving merge-back** -- adjacent sub-chunks are re-merged to restore narrative flow

---

## Project Structure

```
rag-hallucination-detection/
|-- main.py                     # Entry point (demo / eval / ablation modes)
|-- requirements.txt            # Python dependencies
|-- src/                        # Core pipeline
|   |-- rag_pipeline.py         # RAG pipeline with chunking + generation
|   |-- hcpc_retriever.py       # HCPC-v1 (aggressive refinement)
|   |-- hcpc_v2_retriever.py    # HCPC-v2 (selective refinement + CCS)
|   |-- reranker.py             # Cross-encoder re-ranking
|   |-- hallucination_detector.py  # NLI-based faithfulness scoring
|   |-- coherence_metrics.py    # CCS and retrieval diagnostics
|   |-- data_loader.py          # SQuAD loader
|   |-- pubmedqa_loader.py      # PubMedQA loader
|   |-- evaluator.py            # Evaluation harness
|   |-- retrieval_metrics.py    # Retrieval-level metrics
|   |-- adaptive_chunker.py     # Adaptive chunking (experimental)
|   |-- failure_logger.py       # Per-query failure logging
|   +-- ablation.py             # Ablation study runner
|-- experiments/                # Experiment scripts (one per phase)
|   |-- run_hcpc_v2_ablation.py # Phase 6: HCPC head-to-head
|   |-- run_multimodel_ablation.py # Phase 2: multi-model validation
|   |-- run_reranker_experiment.py # Phase 3: re-ranking study
|   |-- run_zeroshot_baseline.py   # Phase 4: zero-shot comparison
|   |-- run_threshold_sensitivity.py # Phase 5: threshold sweep
|   |-- run_coherence_analysis.py  # Coherence metric analysis
|   +-- ...                     # Additional experiment scripts
|-- scripts/                    # Utility scripts
|   |-- generate_consolidated_results.py
|   |-- compute_nli_human_correlation.py
|   +-- generate_validation_sample.py
|-- results/                    # All experimental outputs
|   |-- hcpc_v2/                # Phase 6 results
|   |-- multimodel/             # Phase 2 results
|   |-- reranker/               # Phase 3 results
|   |-- zeroshot/               # Phase 4 results
|   |-- threshold_sensitivity/  # Phase 5 results
|   |-- stats/                  # ANOVA, significance tests
|   +-- ...
+-- ragpaper/                   # LaTeX paper source
    |-- arxiv/main.tex          # Main document
    |-- sections/               # Paper sections
    |-- figures/                # TikZ figures
    +-- references.bib          # Bibliography
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) with Mistral-7B and/or Llama-3-8B pulled
- ~8GB RAM minimum (16GB recommended for MPS acceleration)

### Installation

```bash
git clone https://github.com/Saket-Maganti/rag-hallucination-detection.git
cd rag-hallucination-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pull models

```bash
ollama pull mistral
ollama pull llama3
```

### Quick demo

```bash
python main.py --mode demo
```

### Run experiments

All experiment scripts are in `experiments/` and can be run from anywhere:

```bash
# Phase 1: Chunk size ablation (~45 min)
python experiments/run_pubmedqa_ablation.py

# Phase 2: Multi-model validation (~6 hours)
python experiments/run_multimodel_ablation.py

# Phase 3: Re-ranking (~6 hours)
python experiments/run_reranker_experiment.py

# Phase 4: Zero-shot baseline (~45 min)
python experiments/run_zeroshot_baseline.py

# Phase 5: Threshold sweep (~2 hours)
python experiments/run_threshold_sensitivity.py

# Phase 6: HCPC head-to-head (~1 hour)
python experiments/run_hcpc_v2_ablation.py

# Statistical tests (~2 min, after experiments complete)
python experiments/run_significance_tests.py
```

---

## Tech Stack

| Component | Implementation |
|-----------|---------------|
| Embedding | all-MiniLM-L6-v2 (82M params, 384-dim) |
| Vector store | ChromaDB v0.4, cosine similarity |
| Generation | Mistral-7B + Llama-3-8B via Ollama |
| Cross-encoder | ms-marco-MiniLM-L-6-v2 (66M params) |
| NLI scorer | DeBERTa-v3-base (435M params) |
| Acceleration | Apple Silicon MPS backend |
| Framework | LangChain + sentence-transformers + PyTorch |

---

## Practical Recommendations

Based on 6,050+ queries across six experimental phases:

1. **Validate coherence before deploying RAG in specialized domains.** If CCS is low, fix the embedding model or use domain-specific retrieval before adding refinement layers.
2. **Tune chunk size before prompt engineering.** The faithfulness return from chunk size selection is ~450x larger than from prompt style.
3. **Apply post-retrieval refinement selectively.** Aggressive refinement can degrade faithfulness even when retrieval similarity improves. Use AND-gated detection with rank protection.
4. **Run a zero-shot baseline.** RAG with a mismatched encoder can produce more hallucinations than no retrieval at all.
5. **Tune per-model.** Different architectures respond to retrieval parameters in fundamentally different ways.

---

## Paper

The full paper is available in `ragpaper/`. To compile:

```bash
cd ragpaper/arxiv
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## References

1. Lewis et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS.
2. Rajpurkar et al. (2016). [SQuAD: 100,000+ Questions for Machine Comprehension](https://arxiv.org/abs/1606.05250). EMNLP.
3. Jin et al. (2019). [PubMedQA: A Biomedical Research QA Dataset](https://arxiv.org/abs/1909.06146). EMNLP.
4. Jiang et al. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825).
5. Meta AI (2024). [Llama 3](https://ai.meta.com/blog/meta-llama-3/).
6. He et al. (2021). [DeBERTaV3](https://arxiv.org/abs/2111.09543).
7. Reimers & Gurevych (2019). [Sentence-BERT](https://arxiv.org/abs/1908.10084). EMNLP.
8. Asai et al. (2024). [Self-RAG](https://arxiv.org/abs/2310.11511). ICLR.
9. Yan et al. (2024). [Corrective RAG](https://arxiv.org/abs/2401.15884).
10. Xu et al. (2023). [RECOMP: Context Compression](https://arxiv.org/abs/2310.04408).
