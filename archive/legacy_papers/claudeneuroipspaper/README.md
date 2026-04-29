# claudeneuroipspaper — fresh NeurIPS submission draft

A clean rewrite of the RAG-faithfulness paper that abandons the v2.0
"refinement paradox" framing and rebuilds around what the
pre-registered revision experiments actually show.

## Title

**When Retrieval Quality Decouples From Faithfulness: A Pre-Registered Audit of RAG Evaluation**

## What's in this folder

```
claudeneuroipspaper/
├── README.md           ← this file
├── main.tex            ← entry point; \input each section
├── references.bib      ← citations
├── figures/
│   ├── fig_causal_paired.pdf   ← Fix 1 paired-diff plot
│   └── fig_pareto.pdf          ← Fix 6 faith-vs-latency Pareto
└── sections/
    ├── abstract.tex
    ├── introduction.tex
    ├── background.tex
    ├── methodology.tex
    ├── causal_intervention.tex     ← Fix 1 (null)
    ├── scaling.tex                  ← Fix 2 (paradox collapse)
    ├── multimetric.tex              ← Fix 3 (metric divergence)
    ├── tau_generalization.tex       ← Fix 4 (τ leakage)
    ├── coherence_noise.tex          ← Fix 5 (surviving signal)
    ├── headtohead.tex               ← Fix 6 + Fix 11 (baselines + RAPTOR)
    ├── benchmark.tex                ← released artifact
    ├── related.tex
    ├── discussion.tex
    ├── limitations.tex
    └── conclusion.tex
```

## Why a new paper instead of patching the old one

The pre-registered revision invalidated the original three pillars:

1. **Phenomenon (Fix 2).** SQuAD/Mistral paradox magnitude collapsed
   from 0.069 (n=30) to 0.006–0.020 per seed at n=500 × 5 seeds; only
   1 of 5 seeds reaches p<0.05; pooled bootstrap CI on the paradox
   contains zero.
2. **Mechanism (Fix 1).** Pre-registered matched-similarity
   intervention does not support CCS as a causal mediator at fixed
   mean retrieval similarity (paired Wilcoxon p=0.628, Cohen's
   d_z=−0.017, 95% CI [−0.022, +0.017]).
3. **Intervention (Fix 3 + Fix 6).** HCPC-v2's recovery is
   metric-dependent (paradox 0.011 / 0.032 / 0.140 under
   DeBERTa / mnli / RAGAS) and does not dominate strong baselines
   (CRAG wins on HotpotQA; HCPC-v2 ties RAPTOR-2L on SQuAD; Self-RAG
   underperforms in this matched short-answer harness).

The new paper is built from these honest results with the surviving
positive finding (Fix 5: coherence-preserving noise produces a
smaller faithfulness drop than random noise at matched rate) as the
central remaining claim.

## What survived and is reportable

- **Multi-metric divergence (Fix 3, n=7500)** as the methodology
  contribution: faithfulness scoring is fragile across DeBERTa,
  roberta-large-mnli, and a RAGAS-style LLM-judge.
- **Coherence-vs-similarity disentanglement (Fix 5, n=1591)** as the
  surviving structural finding: coherence carries signal independent
  of similarity at matched noise rate.
- **Tau cross-dataset leakage (Fix 4, n=7500)** as a methodology
  caveat applicable to any retrieval threshold tuning.
- **Pre-registered head-to-head (Fix 6, n=1600)** with HCPC-v2,
  CRAG, RAPTOR-2L, and Self-RAG.
- **RAPTOR full table (Fix 11, n=300)** with offline build cost,
  index size, and p50/p99 latency.
- **Released benchmark** (HF Dataset + Zenodo DOI + pip + LangChain).

## Recommended track

**NeurIPS Datasets & Benchmarks track.** This paper's primary
contribution is the pre-registered audit + benchmark + scripts. The
D&B track values released artifacts and methodology critiques over
algorithmic novelty, which fits the new framing better than the main
research track.

If targeting the main track, the lead must be the multi-metric
divergence finding (Fix 3) framed as a fragility critique of RAG
evaluation generally, not as a paper about HCPC-v2.

## How to compile

```bash
cd claudeneuroipspaper
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
```

The document uses standard `article` class with NeurIPS-style margins
so it can be swapped for the official `neurips_2026.sty` after the
official style file is published.

## Status

- All numerical claims are grounded in committed CSVs under
  `data/revision/fix_NN/` and `results/revision/fix_NN/`.
- All effect sizes, p-values, and CIs are pre-registered in
  `experiments/fix_NN_log.md`.
- Released benchmark + DOI unchanged from v2.0.
