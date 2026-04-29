# Research Framing

## One-Sentence Pitch

RAG faithfulness is under-identified by the evidence most papers report: local
retrieval scores, context coherence alone, and a single faithfulness metric are
all insufficient to support deployment-level claims.

## Why This Is Research, Not Just Benchmarking

The paper tests a falsifiable structure:

1. If local relevance identifies faithfulness, refinement that improves query
   similarity should not reduce faithfulness.
2. If the strong CCS interpretation is correct, HIGH-CCS contexts should beat LOW-CCS contexts at
   fixed query similarity.
3. If faithfulness is well measured, different reasonable judges should give
   compatible intervention magnitudes on the same generations.
4. If policy claims are robust, thresholds and baselines should transfer across
   datasets and cost surfaces.

The experiments reject or qualify each part. The resulting contribution is a
negative identifiability result for RAG evaluation, with CCS retained as a
diagnostic variable rather than an explanation of support.

## Latest Empirical State

- The strong CCS interpretation is rejected in the primary matched-similarity test.
- The scaled DeBERTa refinement effect is small and seed-sensitive.
- Metric choice is now the cleanest positive research result: DeBERTa,
  second-NLI, and RAGAS-style scoring give materially different magnitudes on
  identical generations.
- Human calibration is complete: 99 examples, two raters, Cohen's kappa 0.774,
  with only moderate alignment for the closest automatic scorer.
- Coherence still matters diagnostically: coherence-preserving noise is less
  damaging than random off-topic noise.
- The full baseline run is complete and verified. It rejects a single method ranking:
  RAPTOR-2L narrowly leads SQuAD faithfulness, CRAG leads HotpotQA, and
  Self-RAG underperforms in this matched short-answer harness.

## Strongest NeurIPS Angle

NeurIPS main-track framing:

> We give controlled evidence that common RAG faithfulness claims are not
> identifiable from local retrieval scores. This is a methodological research
> result with immediate consequences for how RAG methods should be evaluated.

Datasets & Benchmarks fallback:

> The anonymized benchmark operationalizes these controls and exposes which RAG
> claims survive matched similarity, multi-metric scoring, threshold transfer,
> human calibration, and cost accounting.
