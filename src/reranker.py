"""
Re-ranker using cross-encoder/ms-marco-MiniLM-L-6-v2.
Takes retrieved chunks and re-scores them by relevance to the query,
returning the top-k most relevant chunks in ranked order.
"""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[Reranker] Loading: {model_name}")
        self.model = CrossEncoder(model_name)
        print(f"[Reranker] Ready")

    def rerank(self, query: str, docs: list[Document], top_k: int = None) -> list[Document]:
        """
        Re-score and re-order documents by relevance to query.
        Returns top_k most relevant docs (or all if top_k is None).
        """
        if not docs:
            return docs

        pairs = [(query, doc.page_content[:512]) for doc in docs]
        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        reranked = [doc for _, doc in scored_docs]
        return reranked[:top_k] if top_k else reranked