"""
Tax Assistant — Primary chat interface with LangGraph orchestration.
Supports both Chat Mode (no upload) and Document Mode (Form16 upload).
"""
import os
import sys
import tempfile
import streamlit as st
from frontend.components.cards import source_citation_card


def show_tax_assistant():
    st.markdown('<p class="section-header">💬 Tax Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Ask anything about Indian taxes — backed by legal documents & AI</p>', unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "form16_path" not in st.session_state:
        st.session_state.form16_path = None
    if "form16_data" not in st.session_state:
        st.session_state.form16_data = None

    # ─── Mode selector + optional Form16 upload ────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.radio(
            "Mode",
            options=["💬 Chat Mode", "📄 Document Mode (Form16)"],
            horizontal=True,
            help="Chat Mode: ask any question. Document Mode: upload Form16 for personalized analysis.",
        )

    with col2:
        if mode == "📄 Document Mode (Form16)":
            uploaded = st.file_uploader("Upload Form 16 PDF", type=["pdf"], key="form16_upload")
            if uploaded:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    st.session_state.form16_path = tmp.name
                st.success("✅ Form 16 uploaded!")
        else:
            st.session_state.form16_path = None

    st.divider()

    # ─── Suggested questions ───────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "I earn ₹12 lakh. Which tax regime should I choose?",
            "Can I claim both HRA and home loan deductions?",
            "What is the 80C deduction limit for FY 2025-26?",
            "I invested ₹75,000 in ELSS. What else can I claim?",
            "How much can I save with NPS investment?",
        ]
        cols = st.columns(len(suggestions))
        for i, (col, s) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(s, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": s})
                    st.rerun()

        st.markdown("")

    # ─── Chat History ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
                <p class="chat-label" style="color:#4ECDC4;">You</p>
                <p style="margin:0;">{msg["content"]}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-assistant">
                <p class="chat-label" style="color:#FF6B6B;">FinAssist AI</p>
                <div style="margin:0;">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Show citations if available
            if "citations" in msg and msg["citations"]:
                with st.expander("📚 Sources", expanded=False):
                    for cite in msg["citations"].split("\n"):
                        if cite.strip():
                            st.markdown(f"- {cite.strip()}")

    # ─── Chat Input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a tax question...", key="chat_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Analyzing..."):
            try:
                from recommendation_engine.graph import run_query

                form16_path = st.session_state.form16_path if mode == "📄 Document Mode (Form16)" else None

                result = run_query(
                    user_query=user_input,
                    mode="document" if form16_path else "chat",
                    form16_path=form16_path,
                )

                # Handle both dict and Pydantic state
                if isinstance(result, dict):
                    answer = result.get("final_answer", "I couldn't generate a response.")
                    citations = result.get("rag_context", {}).get("citations", "") if result.get("rag_context") else ""
                    steps = result.get("processing_steps", [])
                else:
                    answer = result.final_answer or "I couldn't generate a response."
                    citations = result.rag_context.citations if result.rag_context else ""
                    steps = result.processing_steps

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "steps": steps,
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}\n\nPlease ensure the RAG index is built: `python scripts/build_index.py`"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "citations": "",
                })

        st.rerun()

    # ─── Clear chat button ─────────────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("🗑️ Clear Conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
