"""
LangGraph Nodes — individual processing steps in the FinAssist AI workflow.

Node execution order (determined by graph edges):
  detect_intent → [tax_engine] → [form16_parser] → rag_retrieve → generate_answer
"""
import json
import re
from typing import Dict, Any

from google import genai as _genai
from langsmith import wrappers as _ls_wrappers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from recommendation_engine.state import (
    GraphState, TaxEngineResult, RetrievedContext,
    OptimizationResult, Form16Result, UserMessage,
)
from config.settings import settings


# ─── Shared Gemini client (native google-genai + LangSmith tracing) ────────────
_gemini_client = None

def get_gemini_client() -> _genai.Client:
    """Return a LangSmith-traced google-genai Client (lazy singleton)."""
    global _gemini_client
    if _gemini_client is None:
        try:
            raw_client = _genai.Client(api_key=settings.gemini_api_key)
            _gemini_client = _ls_wrappers.wrap_gemini(
                raw_client,
                tracing_extra={
                    "tags": ["gemini", "finassist-ai", "langgraph"],
                    "metadata": {"project": settings.langsmith_project},
                },
            )
        except Exception:
            # Fallback: untraced client
            _gemini_client = _genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


def _call_gemini(prompt: str, retries: int = 3) -> str:
    """Call Gemini with automatic retry on 503/429 and model fallback."""
    import time
    models_to_try = [settings.llm_model, "gemini-2.0-flash", "gemini-2.0-flash-lite"]
    last_error = None

    for model in models_to_try:
        for attempt in range(retries):
            try:
                client = get_gemini_client()
                resp = client.models.generate_content(
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
                    continue  # retry same model
                else:
                    break  # different error, try next model
        # If we got here without returning, try next model

    return f"[Gemini unavailable: {last_error}. Please try again in a moment.]"


# ════════════════════════════════════════════════════════════════════════════════
#  NODE 1: Intent Detection
# ════════════════════════════════════════════════════════════════════════════════

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for a tax assistant.
Classify the user query into exactly ONE of these intents:
- tax_question: General question about tax rules, deductions, sections
- calculation_request: User wants to calculate actual tax (mentions income/salary)
- optimization_request: User wants to find tax savings opportunities
- form16_analysis: User mentions Form 16 or wants to analyze a document
- regime_comparison: User wants to compare old vs new tax regime
- general_chat: Greeting or completely off-topic

Respond with ONLY the intent label, nothing else."""),
    ("human", "{query}"),
])


def detect_intent_node(state: GraphState) -> GraphState:
    """Classify user intent and set routing flags."""
    state.processing_steps.append("intent_detection")

    try:
        prompt = f"""You are an intent classifier for a tax assistant.
Classify the user query into exactly ONE of these intents:
- tax_question: General question about tax rules, deductions, sections
- calculation_request: User wants to calculate actual tax (mentions income/salary)
- optimization_request: User wants to find tax savings opportunities
- form16_analysis: User mentions Form 16 or wants to analyze a document
- regime_comparison: User wants to compare old vs new tax regime
- general_chat: Greeting or completely off-topic

Respond with ONLY the intent label, nothing else.

User query: {state.user_query}"""

        intent_raw = _call_gemini(prompt).strip().lower()

        # Map to valid intent
        valid_intents = [
            "tax_question", "calculation_request", "optimization_request",
            "form16_analysis", "regime_comparison", "general_chat"
        ]
        intent = intent_raw if intent_raw in valid_intents else "tax_question"
        state.intent = intent

        # Set routing flags
        state.needs_calculation = intent in ("calculation_request", "optimization_request", "regime_comparison")
        state.needs_retrieval = intent not in ("general_chat",)
        state.needs_optimization = intent == "optimization_request"
        state.needs_form16 = intent == "form16_analysis" and state.form16_path is not None

    except Exception as e:
        state.intent = "tax_question"
        state.needs_retrieval = True
        state.error = f"Intent detection error: {e}"

    return state


# ════════════════════════════════════════════════════════════════════════════════
#  NODE 2: Form16 Parsing
# ════════════════════════════════════════════════════════════════════════════════

def form16_parser_node(state: GraphState) -> GraphState:
    """Parse Form 16 PDF and populate tax_profile."""
    state.processing_steps.append("form16_parsing")

    if not state.form16_path:
        return state

    try:
        from form16_parser.parser import extract_form16

        form16_data = extract_form16(state.form16_path)
        tax_profile = form16_data.to_tax_profile()

        state.tax_profile = tax_profile
        state.form16_result = Form16Result(
            extracted=True,
            data=form16_data.model_dump(),
            confidence=form16_data.extraction_confidence or 0.0,
            notes=form16_data.parsing_notes or [],
        )
        state.needs_calculation = True

    except Exception as e:
        state.form16_result = Form16Result(
            extracted=False,
            notes=[f"Parsing error: {str(e)}"],
        )
        state.error = str(e)

    return state


# ════════════════════════════════════════════════════════════════════════════════
#  NODE 3: Tax Engine
# ════════════════════════════════════════════════════════════════════════════════

def tax_engine_node(state: GraphState) -> GraphState:
    """Run deterministic tax calculation from tax_profile."""
    state.processing_steps.append("tax_engine")

    if not state.tax_profile:
        return state

    try:
        from tax_engine.calculator import calculate_full_tax

        result = calculate_full_tax(state.tax_profile)

        state.tax_result = TaxEngineResult(
            old_regime_tax=result["old_regime"]["total_tax"],
            new_regime_tax=result["new_regime"]["total_tax"],
            recommended_regime=result["recommendation"]["regime"],
            tax_savings=result["recommendation"]["tax_savings"],
            full_result=result,
        )

    except Exception as e:
        state.error = f"Tax engine error: {e}"

    return state


# ════════════════════════════════════════════════════════════════════════════════
#  NODE 4: Optimization Engine
# ════════════════════════════════════════════════════════════════════════════════

def optimization_node(state: GraphState) -> GraphState:
    """Find unused deduction opportunities."""
    state.processing_steps.append("optimization")

    if not state.tax_profile or not state.tax_result:
        return state

    try:
        from optimization.optimizer import find_optimization_opportunities

        opt_result = find_optimization_opportunities(
            profile=state.tax_profile,
            tax_result=state.tax_result.full_result,
        )

        state.optimization_result = OptimizationResult(
            total_potential_savings=opt_result["total_potential_additional_savings"],
            opportunities=opt_result["opportunities"],
            summary=opt_result["summary"],
        )

    except Exception as e:
        state.error = f"Optimization error: {e}"

    return state


# ════════════════════════════════════════════════════════════════════════════════
#  NODE 5: RAG Retrieval
# ════════════════════════════════════════════════════════════════════════════════

_retriever = None
_reranker = None


def get_rag_pipeline():
    """Lazy-load the RAG pipeline (embedder + FAISS + reranker)."""
    global _retriever, _reranker

    if _retriever is None:
        from embeddings.embedder import BGEEmbedder
        from vectordb.faiss_store import FAISSStore
        from rag.retriever import HybridRetriever
        from rag.reranker import BGEReranker

        embedder = BGEEmbedder(settings.embedding_model)
        faiss_store = FAISSStore(
            embedding_dim=embedder.embedding_dim,
            index_path=settings.faiss_index_path,
        )
        faiss_store.load()

        _retriever = HybridRetriever(
            embedder=embedder,
            faiss_store=faiss_store,
            top_k=settings.top_k_retrieval,
        )
        _reranker = BGEReranker(settings.reranker_model)

    return _retriever, _reranker


def rag_retrieve_node(state: GraphState) -> GraphState:
    """Retrieve legal context using hybrid RAG."""
    state.processing_steps.append("rag_retrieval")

    try:
        retriever, reranker = get_rag_pipeline()
        query = state.user_query

        candidates = retriever.retrieve(query, top_k=settings.top_k_retrieval)
        reranked = reranker.rerank(query, candidates, top_k=settings.top_k_rerank)

        # Format context
        context_parts = []
        citations = []
        for i, doc in enumerate(reranked, 1):
            meta = doc.get("metadata", {})
            source = meta.get("source", "")
            section = meta.get("section", "")
            context_parts.append(f"[{i}] {doc['text']}")
            cite = f"[{i}] {source}"
            if section and section != "general":
                cite += f" — Section {section}"
            citations.append(cite)

        state.rag_context = RetrievedContext(
            context="\n\n---\n\n".join(context_parts),
            citations="\n".join(citations),
            source_docs=reranked,
        )

    except FileNotFoundError:
        # FAISS index not built yet — continue without retrieval
        state.rag_context = RetrievedContext(
            context="[Index not built yet. Run: python scripts/build_index.py]",
            citations="",
        )
    except Exception as e:
        state.error = f"RAG error: {e}"
        state.rag_context = RetrievedContext(context="", citations="")

    return state


# ════════════════════════════════════════════════════════════════════════════════
#  NODE 6: Answer Generation
# ════════════════════════════════════════════════════════════════════════════════

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are FinAssist AI — an expert Indian tax advisor for FY 2025-26.

RULES:
1. NEVER calculate tax yourself. Use numbers from <TAX_RESULT> only.
2. Always cite legal sections when making factual claims.
3. Be clear about Old vs New Regime differences.
4. Structure your answer with clear sections when the response is long.
5. End with actionable next steps when appropriate.

<TAX_RESULT>
{tax_context}
</TAX_RESULT>

<LEGAL_CONTEXT>
{rag_context}
</LEGAL_CONTEXT>

<SOURCES>
{citations}
</SOURCES>

<OPTIMIZATION>
{optimization_context}
</OPTIMIZATION>
"""),
    ("human", "{user_query}"),
])


def generate_answer_node(state: GraphState) -> GraphState:
    """Generate the final answer using Gemini, combining all gathered context."""
    state.processing_steps.append("answer_generation")

    # Build tax context string
    tax_context = "No tax calculation performed."
    if state.tax_result:
        tr = state.tax_result
        tax_context = (
            f"Old Regime Tax: Rs.{tr.old_regime_tax:,.0f}\n"
            f"New Regime Tax: Rs.{tr.new_regime_tax:,.0f}\n"
            f"Recommended: {tr.recommended_regime} Regime\n"
            f"Tax Savings with recommendation: Rs.{tr.tax_savings:,.0f}"
        )

    # Build optimization context
    opt_context = "No optimization analysis performed."
    if state.optimization_result:
        opt = state.optimization_result
        opt_context = opt.summary
        if opt.opportunities:
            lines = [f"- {o['title']}: saves Rs.{o['estimated_tax_savings']:,.0f}" for o in opt.opportunities[:3]]
            opt_context += "\n" + "\n".join(lines)

    rag_context = state.rag_context.context if state.rag_context else ""
    citations = state.rag_context.citations if state.rag_context else ""

    try:
        prompt = f"""You are FinAssist AI - an expert Indian tax advisor for FY 2025-26.

RULES:
1. NEVER calculate tax yourself. Use numbers from TAX_RESULT only.
2. Always cite legal sections when making factual claims.
3. Be clear about Old vs New Regime differences.
4. Structure your answer with clear sections when the response is long.
5. End with actionable next steps when appropriate.

TAX_RESULT:
{tax_context}

LEGAL_CONTEXT:
{rag_context or 'No specific legal context retrieved.'}

SOURCES:
{citations or 'None'}

OPTIMIZATION:
{opt_context}

User question: {state.user_query}"""

        answer = _call_gemini(prompt)
        state.final_answer = answer

    except Exception as e:
        state.final_answer = f"I encountered an error generating the response: {e}"
        state.error = str(e)

    return state
