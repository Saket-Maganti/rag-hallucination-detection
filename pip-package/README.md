# context-coherence

Context Coherence Score (CCS) — a generator-free retrieval-time
diagnostic for RAG hallucination, with a coherence-gated retriever
policy (CCSGate). Pure-numpy core; sentence-transformers is optional.

## Install

```bash
pip install context-coherence
# or, with the embedder extras:
pip install context-coherence[embedders]
```

## Quick start

```python
import numpy as np
from context_coherence import ccs, CCSGate

# 1. Score a retrieval set you've already embedded
embeddings = np.random.randn(5, 384)        # (n_chunks, embed_dim)
score = ccs(embeddings)                      # in [-1, 1]
print(f"CCS = {score:.3f}")

# 2. Score raw text (needs an embedder)
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
score = ccs(["passage 1", "passage 2", "passage 3"], embedder=st)

# 3. As a deployment decision
gate = CCSGate(threshold=0.5)
if gate.fires(retrieved_chunks, embedder=st):
    # Coherence too low → re-retrieve, expand k, or fall back
    pass
else:
    # Coherent set → pass to generator
    pass

# 4. Logging-friendly payload
print(gate.decision(retrieved_chunks, embedder=st))
# {'ccs': 0.42, 'threshold': 0.5, 'fires': True}
```

## Why CCS?

In retrieval-augmented generation, per-passage similarity is *not*
sufficient for faithfulness. A retrieval set with high mean similarity
but high variance (two tight sub-clusters) starves the generator of the
connective evidence it relies on, and the generator silently fills the
gap with parametric memory. CCS measures both:

```
CCS(C) = mean(off-diag cosine sim) − std(off-diag cosine sim)
```

A set with high CCS is consistently coherent; a set with low CCS is
internally fragmented even if individual passages are highly relevant.

In the companion paper (Maganti, 2026) we show:

- The **refinement paradox**: per-passage refinement raises similarity
  but cuts SQuAD faithfulness from 0.80 to 0.64, introducing 10%
  hallucination where there was none.
- HCPC-v2, a coherence-gated retriever using CCS, recovers
  faithfulness with 0% hallucination while intervening on only 17%
  of queries.
- The paradox **persists at frontier scale**: Llama-3.3-70B
  reproduces the magnitude exactly (0.100 vs 0.100 at 7B); GPT-OSS-120B
  still shows a +0.030 paradox.

## Bigger picture

This package ships the metric + the gate. For the full reproducible
benchmark (adversarial cases, paradox per-query records, frontier-scale
results, mechanistic probes), see:

- **Code**: <https://github.com/Saket-Maganti/rag-hallucination-detection>
- **Benchmark (DOI)**: <https://doi.org/10.5281/zenodo.19757291>
- **Interactive demo**: <https://huggingface.co/spaces/saketmgnt/sakkk>

## Citation

```bibtex
@inproceedings{maganti2026coherence,
  title  = {When Better Retrieval Hurts: Context Coherence Drives
            Faithfulness in Retrieval-Augmented Generation},
  author = {Maganti, Saket},
  booktitle = {Proceedings of NeurIPS 2026 (under review)},
  year   = {2026},
}
```

## License

MIT.
