# LangChain integration notes

## What this is

A drop-in `BaseRetriever` (`CoherenceGatedRetriever`) that wraps any
LangChain retriever with a Context Coherence Score (CCS) decision gate.

## Local usage (today, no PR needed)

```python
import sys
sys.path.insert(0, "integrations/langchain")
from coherence_gated_retriever import CoherenceGatedRetriever

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
base = store.as_retriever(search_kwargs={"k": 5})

gated = CoherenceGatedRetriever(
    base_retriever=base,
    embeddings=embeddings,
    ccs_threshold=0.5,
    on_low_coherence="expand_k",   # double k when coherence is low
    log_decisions=True,             # attach CCS to each Doc's metadata
)

docs = gated.get_relevant_documents("What is the cause of X?")
for d in docs:
    print(d.metadata.get("ccs_decision"))
```

## Roadmap to upstream PR (`langchain-ai/langchain`)

We aim for a contrib PR that adds `CoherenceGatedRetriever` to
`langchain.retrievers`. Sketch of the PR:

### Files to add

```
libs/langchain/langchain/retrievers/coherence_gated.py
libs/langchain/tests/unit_tests/retrievers/test_coherence_gated.py
```

### PR description (draft)

> **Title**: feat(retrievers): add CoherenceGatedRetriever
>
> **Summary**: Adds a coherence-gated retriever that wraps any
> `BaseRetriever` and applies a single decision rule based on the
> Context Coherence Score (CCS) of the retrieved set. Implements the
> deployment policy from Maganti (2026), *"When Better Retrieval Hurts:
> Context Coherence Drives Faithfulness in RAG"*. The gate fires when
> CCS falls below a threshold; remediation strategies include
> `expand_k`, `fall_back`, and `skip_refinement`. Adds no new
> dependencies (uses numpy which is already required). Full test
> coverage; behaviour is a no-op pass-through when the gate doesn't
> fire.
>
> **Why**: Per-passage relevance is not sufficient for faithfulness.
> CCS is a generator-free retrieval-time signal that is shown across
> 6,050+ queries and 4 generators (7B → 120B) to predict generation
> failures. The gate adds one `if` to the request path with negligible
> overhead (one pairwise-similarity computation, O(k²) in the embedding
> dimension).
>
> **References**:
>   - Paper: <https://github.com/Saket-Maganti/rag-hallucination-detection>
>   - Standalone metric: `pip install context-coherence`
>   - Benchmark DOI: <https://doi.org/10.5281/zenodo.19757291>

### Tests to include

- Gate fires on synthetically-incoherent embeddings
- Gate doesn't fire on synthetically-coherent embeddings
- All three remediation strategies behave as documented
- Empty input → empty output (no crash)
- Single-doc input → pass through (CCS is trivially 1.0)
- Async path mirrors sync path

### Estimated review time

LangChain contrib PRs typically merge in 1-3 weeks. The retriever
adds no new dependencies and the change surface is small (~150 lines),
which usually accelerates review.

## Why we ship this in our repo first

Even if the upstream PR takes weeks, the integration file is usable
immediately by anyone who clones our repo or copies the file. It gives
us a citable artifact ("CoherenceGatedRetriever, available at...")
without blocking on upstream.
