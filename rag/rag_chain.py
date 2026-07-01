"""
RAG Chain — Retrieval-augmented generation pipeline.
Integrates HybridRetriever + BGEReranker + Gemini (via native google-genai + LangSmith wrap_gemini).
LangSmith tracing is automatic via wrap_gemini.
"""
from typing import List, Dict, Any

from google import genai as _genai
from langsmith import wrappers as _ls_wrappers

from rag.retriever import HybridRetriever
from rag.reranker import BGEReranker
from config.settings import settings


TAX_SYSTEM_PROMPT = """You are FinAssist AI, an expert Indian tax advisor specializing in FY 2025-26 taxation.

CRITICAL RULES:
1. NEVER perform tax calculations yourself. Always state numbers from the context or say "use the Tax Calculator".
2. Always cite the source section/document when providing legal information.
3. Be specific about FY 2025-26 / AY 2026-27 rules.
4. Clearly distinguish between Old and New Tax Regime rules.
5. If unsure, say so - never hallucinate tax rules.
6. Keep answers clear, structured, and actionable.

RETRIEVED LEGAL CONTEXT:
{context}

SOURCE CITATIONS:
{citations}
"""


def format_context(docs: List[Dict[str, Any]]) -> tuple[str, str]:
    """Format retrieved docs into context string and citations list."""
    context_parts = []
    citations = []

    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        source = meta.get("source", "Unknown")
        section = meta.get("section", "")

        context_parts.append(f"[Chunk {i}] {text}")

        cite = f"[{i}] {source}"
        if section and section != "general":
            cite += f" - Section {section}"
        citations.append(cite)

    return "\n\n---\n\n".join(context_parts), "\n".join(citations)


class RAGChain:
    """
    Full RAG pipeline:
    Query -> Hybrid Retrieval -> Reranking -> Prompt -> Gemini (traced) -> Answer
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: BGEReranker,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        # Initialize native google-genai client with LangSmith wrap_gemini tracing
        try:
            raw_client = _genai.Client(api_key=settings.gemini_api_key)
            self._client = _ls_wrappers.wrap_gemini(
                raw_client,
                tracing_extra={
                    "tags": ["gemini", "finassist-ai", "rag-chain"],
                    "metadata": {"project": settings.langsmith_project},
                },
            )
        except Exception:
            self._client = _genai.Client(api_key=settings.gemini_api_key)

    def _call_gemini(self, prompt: str, retries: int = 3) -> str:
        """Call Gemini with automatic retry on 503/429 and model fallback."""
        import time
        models_to_try = [settings.llm_model, "gemini-2.0-flash", "gemini-2.0-flash-lite"]
        last_error = None

        for model in models_to_try:
            for attempt in range(retries):
                try:
                    resp = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    return resp.text or ""
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "EXHAUSTED" in err_str:
                        wait = 2 ** attempt  # 1s, 2s, 4s
                        time.sleep(wait)
                        continue
                    else:
                        break  # try next model

        return f"[Gemini unavailable: {last_error}. Please try again in a moment.]"

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run a question through the full RAG pipeline.
        Returns answer + source docs for citation display.
        """
        # Retrieve and rerank
        candidates = self.retriever.retrieve(question, top_k=self.top_k_retrieve)
        reranked = self.reranker.rerank(question, candidates, top_k=self.top_k_rerank)
        context, citations = format_context(reranked)

        # Build prompt
        prompt = TAX_SYSTEM_PROMPT.format(context=context, citations=citations)
        prompt += f"\n\nUser question: {question}"

        # Call Gemini with LangSmith tracing
        answer = self._call_gemini(prompt)

        return {
            "question": question,
            "answer": answer,
            "source_docs": reranked,
            "citations": citations,
        }

    def query_stream(self, question: str):
        """Stream the Gemini response token by token (for Streamlit st.write_stream)."""
        candidates = self.retriever.retrieve(question, top_k=self.top_k_retrieve)
        reranked = self.reranker.rerank(question, candidates, top_k=self.top_k_rerank)
        context, citations = format_context(reranked)

        prompt = TAX_SYSTEM_PROMPT.format(context=context, citations=citations)
        prompt += f"\n\nUser question: {question}"

        # Stream via native SDK
        for chunk in self._client.models.generate_content_stream(
            model=settings.llm_model,
            contents=prompt,
        ):
            if chunk.text:
                yield chunk.text
