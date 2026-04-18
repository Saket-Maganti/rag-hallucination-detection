"""
src/selfrag_wrapper.py
Wrapper around the published Self-RAG checkpoint
(`selfrag/selfrag_llama2_7b`) for our shared evaluation pipeline.

Self-RAG was trained to interleave special "reflection tokens" with normal
generation. The relevant tokens for our comparison:

    [Retrieve]   — emitted before retrieval-augmented generation
    [No Retrieval] — emitted when the model believes retrieval is unneeded
    [Relevant] / [Irrelevant]    — judges retrieved passage relevance
    [Fully supported] / [Partially supported] / [No support]
                  — judges whether the generated text is supported by retrieval
    [Utility:5..1] — final answer quality estimate

We expose a `SelfRAGGenerator` with a `generate()` method that mirrors the
shape of `RAGPipeline.generate()` so the head-to-head harness (Item A4) can
swap it in. The wrapper also returns the parsed reflection-token decisions
so we can report aggregate Self-RAG behaviour alongside its faithfulness.

This module does NOT load the model at import time. Call `.load()` first.
Inference requires a GPU (T4 or better; ~14 GB fp16).

Reference: Asai et al., 2024 — https://arxiv.org/abs/2310.11511
HF checkpoint: https://huggingface.co/selfrag/selfrag_llama2_7b
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


REFLECTION_TOKENS = [
    "[Retrieve]", "[No Retrieval]", "[Continue to Use Evidence]",
    "[Relevant]", "[Irrelevant]",
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
]


@dataclass
class SelfRAGOutput:
    answer:           str
    raw_text:         str
    reflection_tokens: Dict[str, int] = field(default_factory=dict)
    parsed_decisions: Dict[str, str] = field(default_factory=dict)
    retrieval_used:   bool = True
    relevance_judgement: Optional[str] = None
    support_judgement:   Optional[str] = None
    utility_score:       Optional[int] = None


class SelfRAGGenerator:
    """
    Lazy-loaded wrapper around the Self-RAG-7B checkpoint.

    Parameters
    ----------
    model_name : str
        HuggingFace identifier; default `selfrag/selfrag_llama2_7b`.
    device : str | None
        "cuda" / "mps" / "cpu". Auto-detected if None.
    dtype : str
        torch dtype name. Default "float16".
    """

    PROMPT_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n[Retrieve]<paragraph>{paragraph}</paragraph>"
    )

    def __init__(
        self,
        model_name: str = "selfrag/selfrag_llama2_7b",
        device: Optional[str] = None,
        dtype: str = "float16",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype  = dtype
        self._model = None
        self._tok   = None

    def load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = getattr(torch, self.dtype)
        print(f"[SelfRAG] Loading {self.model_name} on {self.device}...")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self._model.eval()
        print(f"[SelfRAG] Ready.")

    # ── Generation ───────────────────────────────────────────────────────────

    def _build_paragraph_block(self, retrieved_docs: List[Document]) -> str:
        """Self-RAG's training format places passages between <paragraph> tags."""
        return " ".join(d.page_content.strip() for d in retrieved_docs)

    def generate(
        self,
        question: str,
        retrieved_docs: List[Document],
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Match RAGPipeline.generate() shape so the head-to-head harness can
        treat us as a drop-in generator.
        """
        import torch
        self.load()

        paragraph = self._build_paragraph_block(retrieved_docs)
        prompt = self.PROMPT_TEMPLATE.format(
            instruction=question, paragraph=paragraph
        )
        enc = self._tok(prompt, return_tensors="pt").to(self.device)

        t0 = time.time()
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        latency = round(time.time() - t0, 2)

        gen_ids = out[0, enc["input_ids"].shape[-1]:]
        raw = self._tok.decode(gen_ids, skip_special_tokens=False)
        parsed = self._parse_reflections(raw)

        # Strip reflection tokens for the user-facing answer
        answer_text = raw
        for tok in REFLECTION_TOKENS:
            answer_text = answer_text.replace(tok, "")
        # Trim Self-RAG's internal markup
        answer_text = re.sub(r"<paragraph>.*?</paragraph>", "", answer_text, flags=re.DOTALL)
        answer_text = answer_text.strip()

        context = "\n\n---\n\n".join(d.page_content for d in retrieved_docs)
        return {
            "question":         question,
            "answer":           answer_text,
            "context":          context,
            "retrieved_docs":   retrieved_docs,
            "latency_s":        latency,
            "selfrag_raw":      raw,
            "selfrag_parsed":   parsed,
        }

    @staticmethod
    def _parse_reflections(raw: str) -> Dict[str, Any]:
        """Extract reflection-token decisions from the raw Self-RAG output."""
        result: Dict[str, Any] = {
            "reflection_token_counts": {},
            "retrieval_used":   "[Retrieve]" in raw or "<paragraph>" in raw,
            "relevance":        None,
            "support":          None,
            "utility":          None,
        }
        for tok in REFLECTION_TOKENS:
            cnt = raw.count(tok)
            if cnt:
                result["reflection_token_counts"][tok] = cnt
        if "[Relevant]" in raw:
            result["relevance"] = "relevant"
        elif "[Irrelevant]" in raw:
            result["relevance"] = "irrelevant"
        if "[Fully supported]" in raw:
            result["support"] = "full"
        elif "[Partially supported]" in raw:
            result["support"] = "partial"
        elif "[No support / Contradictory]" in raw:
            result["support"] = "none"
        m = re.search(r"\[Utility:([1-5])\]", raw)
        if m:
            result["utility"] = int(m.group(1))
        return result
