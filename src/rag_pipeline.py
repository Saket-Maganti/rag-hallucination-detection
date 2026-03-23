"""
RAG Pipeline with Hallucination Detection
Uses Ollama (Mistral-7B) + ChromaDB + LangChain
"""

import os
import time
from typing import Optional
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


# ── Prompt templates ──────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say "I cannot find this information in the provided context."

Context:
{context}

Question: {question}

Answer based strictly on the context above:"""
)


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 3,
        model_name: str = "mistral",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str = "./chroma_db"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.model_name = model_name
        self.persist_dir = persist_dir

        print(f"[RAG] Loading embedding model: {embed_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "mps"}   # Apple Silicon GPU
        )

        print(f"[RAG] Connecting to Ollama ({model_name})")
        self.llm = OllamaLLM(model=model_name, temperature=0.1)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.vectorstore: Optional[Chroma] = None

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_documents(self, documents: list[Document], collection_name: str = "qasper"):
        """Chunk and embed documents into ChromaDB."""
        print(f"[RAG] Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"[RAG] Created {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")

        print("[RAG] Building vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_dir
        )
        print(f"[RAG] Indexed {len(chunks)} chunks into ChromaDB")
        return len(chunks)

    def load_existing_index(self, collection_name: str = "qasper"):
        """Load an existing ChromaDB index from disk."""
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"[RAG] Loaded existing index from {self.persist_dir}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve top-k relevant chunks for a query."""
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call index_documents() first.")
        return self.vectorstore.similarity_search(query, k=self.top_k)

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, question: str, retrieved_docs: list[Document]) -> dict:
        """Generate an answer from retrieved context."""
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = RAG_PROMPT.format(context=context, question=question)

        t0 = time.time()
        answer = self.llm.invoke(prompt)
        latency = round(time.time() - t0, 2)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "retrieved_docs": retrieved_docs,
            "latency_s": latency
        }

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """End-to-end: retrieve → generate."""
        retrieved = self.retrieve(question)
        result = self.generate(question, retrieved)
        return result
