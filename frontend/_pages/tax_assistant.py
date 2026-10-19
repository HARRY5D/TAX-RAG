"""
Tax Assistant — Chat interface backed by the Gemini Agentic Orchestrator.

Architecture (Phase 3):
  Primary path  → recommendation_engine.agent.run_agent()  [Gemini tool-calling]
  Fallback path → recommendation_engine.graph.run_query()  [LangGraph DAG]

The fallback fires automatically if Gemini is unavailable (bad API key, quota exceeded).
All LangGraph traces still appear in LangSmith from the fallback path.
"""
import os
import sys
import tempfile
import markdown
import streamlit as st
from frontend.components.cards import source_citation_card


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _badge(label: str) -> str:
    """Render a small inline tool-call badge in HTML."""
    colors = {
        "retrieve_tax_docs":              ("#4ECDC4", "#0d2b2a"),
        "calculate_tax":                  ("#FFD700", "#1a1400"),
        "analyze_form16":                 ("#FF6B6B", "#2a0000"),
        "get_optimization_opportunities": ("#a78bfa", "#1a0040"),
        "web_search_tax_updates":         ("#60a5fa", "#00162a"),
    }
    bg, fg = colors.get(label, ("#94a3b8", "#0a0f1e"))
    icons = {
        "retrieve_tax_docs":              "RAG",
        "calculate_tax":                  "Calc",
        "analyze_form16":                 "Form16",
        "get_optimization_opportunities": "Optimize",
        "web_search_tax_updates":         "Web",
    }
    icon = icons.get(label, label)
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'padding:2px 8px;border-radius:6px;font-size:0.72rem;'
        f'font-weight:700;margin:2px 3px;">{icon}</span>'
    )


def _extract_result_agent(result: dict) -> tuple:
    """
    Extract (answer, citations, steps, tool_badges_html) from agent result dict.
    """
    answer   = result.get("final_answer") or "I could not generate a response."
    steps    = result.get("processing_steps") or result.get("steps") or []
    citations = result.get("citations", "")

    # If citations came through rag_context dict
    if not citations:
        rag_ctx = result.get("rag_context") or {}
        if isinstance(rag_ctx, dict):
            citations = rag_ctx.get("citations", "")
        elif hasattr(rag_ctx, "citations"):
            citations = rag_ctx.citations or ""

    # Build tool badge HTML from tool_calls_made
    tool_calls = result.get("tool_calls_made") or []
    badges_html = ""
    if tool_calls:
        badges_html = "".join(_badge(tc["tool"]) for tc in tool_calls)

    return answer, citations, steps, badges_html


def _extract_result_graph(result: dict) -> tuple:
    """
    Extract (answer, citations, steps, badges_html) from LangGraph result dict.
    Compatible with the old run_query() return format.
    """
    answer = result.get("final_answer") or "I could not generate a response."
    rag_ctx = result.get("rag_context") or {}
    if isinstance(rag_ctx, dict):
        citations = rag_ctx.get("citations", "")
    elif hasattr(rag_ctx, "citations"):
        citations = rag_ctx.citations or ""
    else:
        citations = ""
    steps = result.get("processing_steps") or []
    return answer, citations, steps, ""


def _run_query_and_store(user_query: str, mode: str):
    """
    Run the agent query and append result to st.session_state.messages.
    Tries Gemini Agent first; falls back to LangGraph DAG if unavailable.
    """
    backend = st.session_state.get("backend_mode", "gemini")  # "gemini" | "langgraph"

    with st.spinner("Analyzing your question..."):
        form16_path = (
            st.session_state.get("form16_path")
            if mode == " Document Mode (Form16)"
            else None
        )

        # Build chat history for multi-turn context
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.get("messages", [])
        ]

        # Attach any existing tax profile from session state
        tax_profile = dict(st.session_state.get("last_tax_profile") or {})

        result = None
        used_backend = backend

        # ── Primary: Gemini Agent ──────────────────────────────────────────
        if backend == "gemini":
            try:
                from recommendation_engine.agent import run_agent
                result = run_agent(
                    user_query=user_query,
                    tax_profile=tax_profile,
                    form16_path=form16_path,
                    chat_history=chat_history,
                )
                if result.get("error") and not result.get("final_answer"):
                    raise RuntimeError(result["error"])
                answer, citations, steps, badges = _extract_result_agent(result)
            except Exception as e:
                # Fallback to LangGraph
                used_backend = "langgraph_fallback"
                result = None
                st.warning(f"Gemini Agent unavailable ({e}). Falling back to LangGraph mode.")

        # ── Fallback: LangGraph DAG ────────────────────────────────────────
        if result is None:
            try:
                from recommendation_engine.graph import run_query
                result = run_query(
                    user_query=user_query,
                    mode="document" if form16_path else "chat",
                    tax_profile=tax_profile or None,
                    form16_path=form16_path,
                    chat_history=chat_history,
                )
                answer, citations, steps, badges = _extract_result_graph(result)
            except Exception as e:
                answer   = f"**Error:** {e}\n\nIf it's a retrieval error, run:\n```\npython scripts/build_index.py\n```"
                citations = ""
                steps     = []
                badges    = ""

        st.session_state.messages.append({
            "role":        "assistant",
            "content":     answer,
            "citations":   citations,
            "steps":       steps,
            "badges_html": badges,
            "backend":     used_backend,
        })


# ─── Main page ─────────────────────────────────────────────────────────────────

def show_tax_assistant():
    st.markdown('<p class="section-header"> Tax Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Ask anything about Indian taxes — backed by legal documents & AI</p>',
        unsafe_allow_html=True,
    )

    # ── Session state init ──────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "form16_path" not in st.session_state:
        st.session_state.form16_path = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "backend_mode" not in st.session_state:
        st.session_state.backend_mode = "gemini"

    # ── Top controls row ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        mode = st.radio(
            "Mode",
            options=[" Chat Mode", " Document Mode (Form16)"],
            horizontal=True,
            help="Chat Mode: ask any question. Document Mode: upload Form16 for personalized analysis.",
            key="mode_selector",
        )

    with col2:
        backend_label = st.selectbox(
            "Engine",
            options=["Gemini Agent (Agentic)", "LangGraph (Classic)"],
            index=0 if st.session_state.backend_mode == "gemini" else 1,
            key="backend_selector",
            help="Gemini Agent: dynamic tool-calling. LangGraph: original fixed DAG pipeline.",
        )
        st.session_state.backend_mode = "gemini" if "Gemini" in backend_label else "langgraph"

    with col3:
        if mode == " Document Mode (Form16)":
            uploaded = st.file_uploader("Upload Form 16 PDF", type=["pdf"], key="form16_upload")
            if uploaded:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    st.session_state.form16_path = tmp.name
                st.success("Form 16 uploaded!")
        else:
            st.session_state.form16_path = None

    # Form16 auto-loaded banner
    if st.session_state.get("form16_loaded"):
        employer = st.session_state.get("form16_employer", "your employer")
        st.success(f"Form 16 loaded from {employer} — tax profile auto-filled in Tax Calculator & Optimizer.")

    st.divider()

    # ── Suggestion buttons (only when chat empty) ───────────────────────────
    if not st.session_state.messages and st.session_state.pending_query is None:
        st.markdown("Try asking:")
        suggestions = [
            "I earn Rs.12 lakh. Which tax regime should I choose?",
            "Can I claim both HRA and home loan deductions?",
            "What is the 80C deduction limit for FY 2025-26?",
            "I invested Rs.75,000 in ELSS. What else can I claim?",
            "How much can I save with NPS investment?",
        ]
        st.markdown(
            '<style>'
            'div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] button {'
            'min-height:82px !important;height:auto !important;white-space:normal !important;'
            'word-wrap:break-word !important;text-align:left !important;'
            'padding:10px 12px !important;font-size:0.82rem !important;'
            'line-height:1.45 !important;align-items:flex-start !important;}'
            '</style>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(suggestions))
        for i, (col, s) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(s, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": s})
                    st.session_state.pending_query = s
                    st.rerun()
        st.markdown("")

    # ── Process pending suggestion query ────────────────────────────────────
    if st.session_state.pending_query is not None:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        _run_query_and_store(query, mode)
        st.rerun()

    # ── Chat history display ────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            user_content = markdown.markdown(msg.get("content", ""), extensions=["tables", "fenced_code"])
            st.markdown(
                f'<div class="chat-user">'
                f'<p class="chat-label" style="color:#4ECDC4;">You</p>'
                f'<div class="chat-body">{user_content}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            badges = msg.get("badges_html", "")
            backend_tag = msg.get("backend", "")
            backend_note = ""
            if backend_tag == "langgraph_fallback":
                backend_note = ' <span style="font-size:0.7rem;color:#94a3b8;">[LangGraph fallback]</span>'
            elif backend_tag == "langgraph":
                backend_note = ' <span style="font-size:0.7rem;color:#94a3b8;">[LangGraph]</span>'

            badges_row = f'<div style="margin-bottom:6px;">{badges}</div>' if badges else ""
            assistant_content = markdown.markdown(msg.get("content", ""), extensions=["tables", "fenced_code"])

            st.markdown(
                f'<div class="chat-assistant">'
                f'<p class="chat-label" style="color:#FF6B6B;">FinAssist AI{backend_note}</p>'
                f'{badges_row}'
                f'<div class="chat-body">{assistant_content}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Citations expander
            citations = msg.get("citations", "")
            if citations:
                with st.expander("Sources", expanded=False):
                    for cite in citations.split("\n"):
                        if cite.strip():
                            st.markdown(f"- {cite.strip()}")

            # Processing steps debug
            steps = msg.get("steps", [])
            if steps:
                with st.expander("Processing steps", expanded=False):
                    st.markdown(" → ".join(steps))

    # ── Chat input ──────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a tax question...", key="chat_input")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        _run_query_and_store(user_input, mode)
        st.rerun()

    # ── Clear chat ──────────────────────────────────────────────────────────
    if st.session_state.messages:
        st.markdown("---")
        if st.button("Clear Conversation", key="clear_chat"):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()
