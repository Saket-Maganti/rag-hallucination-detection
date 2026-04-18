"""
src/dataset_loaders.py
Unified loader interface for the 6 datasets used in the multi-dataset
generalization study (§Results, expanded Table 2).

Loaders return (documents, qa_pairs) with a common schema:

  documents : List[Document]   page_content = passage / context, metadata = {paper_id, title}
  qa_pairs  : List[dict]       {paper_id, question, ground_truth}

Datasets:
  squad         — Wikipedia reading comprehension (existing, in data_loader)
  pubmedqa      — biomedical QA               (existing, in pubmedqa_loader)
  naturalqs     — Natural Questions (open-domain QA over Wikipedia)
  triviaqa      — TriviaQA (Wikipedia-validated trivia)
  hotpotqa      — multi-hop QA across multiple Wikipedia paragraphs
  financebench  — financial QA with SEC filings as context

For datasets we don't already have loaders for, this module wraps the
HuggingFace datasets library. FinanceBench is loaded from the official
PatronusAI release.

NOTE: Each loader caps at `max_papers` rows for tractable experimentation
and deterministic sampling (seed=42). The cap is applied after
deduplication of contexts to keep the document store small.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Dict, List, Tuple

from langchain_core.documents import Document
from tqdm import tqdm

# Re-export the existing loaders for uniform access
from src.data_loader import load_qasper as _load_squad
from src.pubmedqa_loader import load_pubmedqa as _load_pubmedqa


SEED = 42


# ── Natural Questions ────────────────────────────────────────────────────────

def load_naturalqs(
    split: str = "validation",
    max_papers: int = 50,
) -> Tuple[List[Document], List[dict]]:
    """
    Natural Questions (open-domain). We use the simplified version curated by
    Google Research, which provides short answers and the Wikipedia document
    text. Many examples have null short answers; we keep only those with a
    valid short answer span.
    """
    from datasets import load_dataset
    print(f"[Data] Loading Natural Questions ({split})...")
    try:
        ds = load_dataset("google-research-datasets/natural_questions",
                          "default", split=split, streaming=False)
    except Exception:
        # Fallback: the lighter "nq_open" subset (questions + answers, no docs)
        # paired with a separately loaded Wikipedia dump. We stub by skipping
        # context (callers will see empty passages).
        ds = load_dataset("nq_open", split="validation")

    docs: List[Document] = []
    qa: List[dict] = []
    rng = random.Random(SEED)
    items = list(ds)
    rng.shuffle(items)

    seen_contexts = {}
    for item in tqdm(items, desc="NQ"):
        question = item.get("question") or item.get("question_text", "")
        if isinstance(question, dict):
            question = question.get("text", "")
        if not question:
            continue

        # NQ default config: long_answer / short_answers stored under
        # annotations[0].
        ground_truth = ""
        if "annotations" in item:
            ann = item["annotations"]
            if isinstance(ann, dict):
                short_answers = ann.get("short_answers", [])
                if short_answers and short_answers[0].get("text"):
                    ground_truth = short_answers[0]["text"][0]
            elif isinstance(ann, list) and ann:
                sa = ann[0].get("short_answers", [])
                if sa and sa[0].get("text"):
                    ground_truth = sa[0]["text"][0]
        elif "answer" in item:
            answer = item["answer"]
            if isinstance(answer, list) and answer:
                ground_truth = answer[0]
            elif isinstance(answer, str):
                ground_truth = answer
        if not ground_truth:
            continue

        # Context: try document.tokens then document.html stripped, else skip
        context = ""
        doc_obj = item.get("document", {})
        if isinstance(doc_obj, dict):
            toks = doc_obj.get("tokens", {})
            if isinstance(toks, dict) and toks.get("token"):
                # Filter HTML tokens
                tok_list = toks.get("token", [])
                is_html = toks.get("is_html", [False] * len(tok_list))
                context = " ".join(t for t, h in zip(tok_list, is_html) if not h)
        if not context:
            continue
        # Cap context length to keep ChromaDB manageable
        context = context[:8000]
        ctx_key = context[:200]
        if ctx_key in seen_contexts:
            paper_id = seen_contexts[ctx_key]
        else:
            paper_id = str(len(seen_contexts))
            seen_contexts[ctx_key] = paper_id
            docs.append(Document(page_content=context, metadata={
                "paper_id": paper_id, "title": question[:60]
            }))
        qa.append({
            "paper_id": paper_id,
            "question": question,
            "ground_truth": ground_truth,
        })
        if len(docs) >= max_papers:
            break

    print(f"[Data] NQ: {len(docs)} contexts, {len(qa)} QA pairs")
    return docs, qa


# ── TriviaQA ─────────────────────────────────────────────────────────────────

def load_triviaqa(
    split: str = "validation",
    max_papers: int = 50,
) -> Tuple[List[Document], List[dict]]:
    """
    TriviaQA, RC variant (questions paired with Wikipedia / web evidence).
    We use the rc.nocontext config and append the verified Wikipedia
    evidence text as context.
    """
    from datasets import load_dataset
    print(f"[Data] Loading TriviaQA ({split})...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split=split)

    docs: List[Document] = []
    qa: List[dict] = []
    rng = random.Random(SEED)
    items = list(ds)
    rng.shuffle(items)

    seen_contexts = {}
    for item in tqdm(items, desc="TriviaQA"):
        question = item.get("question", "")
        answer_obj = item.get("answer", {})
        ground_truth = ""
        if isinstance(answer_obj, dict):
            ground_truth = (answer_obj.get("value") or
                            (answer_obj.get("aliases") or [""])[0])
        if not (question and ground_truth):
            continue

        # entity_pages.wiki_context holds long-form Wikipedia evidence
        ent_pages = item.get("entity_pages", {})
        wiki_ctxs = ent_pages.get("wiki_context", []) if isinstance(ent_pages, dict) else []
        context = " ".join(wiki_ctxs)[:8000].strip() if wiki_ctxs else ""
        if not context:
            continue

        ctx_key = context[:200]
        if ctx_key in seen_contexts:
            paper_id = seen_contexts[ctx_key]
        else:
            paper_id = str(len(seen_contexts))
            seen_contexts[ctx_key] = paper_id
            docs.append(Document(page_content=context, metadata={
                "paper_id": paper_id, "title": question[:60]
            }))
        qa.append({
            "paper_id": paper_id,
            "question": question,
            "ground_truth": ground_truth,
        })
        if len(docs) >= max_papers:
            break

    print(f"[Data] TriviaQA: {len(docs)} contexts, {len(qa)} QA pairs")
    return docs, qa


# ── HotpotQA (multi-hop) ─────────────────────────────────────────────────────

def load_hotpotqa(
    split: str = "validation",
    max_papers: int = 50,
) -> Tuple[List[Document], List[dict]]:
    """
    HotpotQA distractor split: each example has 10 paragraphs, of which 2-3
    are 'gold' supporting evidence. We index ALL paragraphs (gold + distractors)
    as separate documents — this is the multi-hop generalization test for the
    coherence paradox: a correct answer requires retrieving and integrating
    multiple distinct passages.
    """
    from datasets import load_dataset
    print(f"[Data] Loading HotpotQA ({split})...")
    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)

    docs: List[Document] = []
    qa: List[dict] = []
    rng = random.Random(SEED)
    items = list(ds)
    rng.shuffle(items)

    paper_counter = 0
    for item in tqdm(items, desc="HotpotQA"):
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not (question and answer):
            continue
        ctx_obj = item.get("context", {})
        titles = ctx_obj.get("title", []) if isinstance(ctx_obj, dict) else []
        sentences_per_title = ctx_obj.get("sentences", []) if isinstance(ctx_obj, dict) else []
        if not titles:
            continue

        # Each title's sentences become one Document
        for title, sents in zip(titles, sentences_per_title):
            content = " ".join(sents).strip()
            if not content:
                continue
            paper_id = f"{paper_counter}_{title[:40]}"
            docs.append(Document(page_content=content, metadata={
                "paper_id": paper_id, "title": title,
                "supports_question": question[:80],
            }))
            paper_counter += 1
        qa.append({
            "paper_id": str(paper_counter),
            "question": question,
            "ground_truth": answer,
            "supporting_titles": item.get("supporting_facts", {}).get("title", []),
            "is_multihop": True,
        })
        # Each HotpotQA example contributes ~10 paragraphs; cap on docs.
        if len(docs) >= max_papers * 10:
            break

    print(f"[Data] HotpotQA: {len(docs)} paragraphs, {len(qa)} multi-hop QA pairs")
    return docs, qa


# ── FinanceBench ─────────────────────────────────────────────────────────────

def load_financebench(
    split: str = "test",
    max_papers: int = 50,
) -> Tuple[List[Document], List[dict]]:
    """
    FinanceBench (Patronus AI release): financial QA over SEC filings.
    The dataset provides a `evidence` field with the relevant filing text and
    a `question` + `answer` pair. Each (filing, evidence) pair becomes one
    Document for indexing.
    """
    from datasets import load_dataset
    print(f"[Data] Loading FinanceBench ({split})...")
    try:
        ds = load_dataset("PatronusAI/financebench", split=split)
    except Exception as exc:
        print(f"[Data] FinanceBench load failed ({exc}); the dataset may be gated. "
              "Apply for access at https://huggingface.co/datasets/PatronusAI/financebench")
        return [], []

    docs: List[Document] = []
    qa: List[dict] = []
    rng = random.Random(SEED)
    items = list(ds)
    rng.shuffle(items)

    seen_contexts = {}
    for item in tqdm(items, desc="FinanceBench"):
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not (question and answer):
            continue
        # The dataset exposes either `evidence` (list of dicts) or
        # `evidence_text` depending on version; handle both.
        evidence = ""
        if "evidence" in item:
            ev = item["evidence"]
            if isinstance(ev, list):
                evidence = " ".join(e.get("evidence_text", "") if isinstance(e, dict) else str(e) for e in ev)
            else:
                evidence = str(ev)
        elif "evidence_text" in item:
            evidence = item["evidence_text"] or ""
        if not evidence:
            continue
        evidence = evidence[:8000]
        ctx_key = evidence[:200]
        if ctx_key in seen_contexts:
            paper_id = seen_contexts[ctx_key]
        else:
            paper_id = str(len(seen_contexts))
            seen_contexts[ctx_key] = paper_id
            docs.append(Document(page_content=evidence, metadata={
                "paper_id": paper_id,
                "title":    item.get("doc_name", question)[:60],
            }))
        qa.append({
            "paper_id":     paper_id,
            "question":     question,
            "ground_truth": answer,
            "company":      item.get("company", ""),
            "doc_period":   item.get("doc_period", ""),
        })
        if len(docs) >= max_papers:
            break

    print(f"[Data] FinanceBench: {len(docs)} contexts, {len(qa)} QA pairs")
    return docs, qa


# ── Unified registry ─────────────────────────────────────────────────────────

LoaderFn = Callable[[str, int], Tuple[List[Document], List[dict]]]

DATASET_REGISTRY: Dict[str, Dict] = {
    "squad": {
        "loader": _load_squad,
        "default_split": "validation",
        "domain": "general (Wikipedia)",
        "task_type": "single-hop extractive",
    },
    "pubmedqa": {
        "loader": _load_pubmedqa,
        "default_split": "train",
        "domain": "biomedical (PubMed abstracts)",
        "task_type": "single-hop yes/no/maybe",
    },
    "naturalqs": {
        "loader": load_naturalqs,
        "default_split": "validation",
        "domain": "general (Wikipedia, open-domain)",
        "task_type": "single-hop extractive",
    },
    "triviaqa": {
        "loader": load_triviaqa,
        "default_split": "validation",
        "domain": "general (Wikipedia, trivia)",
        "task_type": "single-hop extractive",
    },
    "hotpotqa": {
        "loader": load_hotpotqa,
        "default_split": "validation",
        "domain": "general (Wikipedia, multi-paragraph)",
        "task_type": "multi-hop reasoning",
    },
    "financebench": {
        "loader": load_financebench,
        "default_split": "test",
        "domain": "finance (SEC filings)",
        "task_type": "domain-specific factoid + numerical",
    },
}


def load_dataset_by_name(
    name: str,
    max_papers: int = 50,
    split: str = None,
) -> Tuple[List[Document], List[dict]]:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}; choices = {list(DATASET_REGISTRY)}")
    spec = DATASET_REGISTRY[name]
    actual_split = split or spec["default_split"]
    return spec["loader"](split=actual_split, max_papers=max_papers)
