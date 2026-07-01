"""
LangSmith Configuration for FinAssist AI.

Sets up:
1. LANGSMITH_* env vars  — picked up by LangChain/LangGraph automatically
2. LANGCHAIN_* env vars  — legacy aliases, still needed by some LangChain versions
3. wrap_gemini()         — wraps the raw google-genai client for direct tracing

Usage:
    from config.langsmith_config import configure_langsmith, get_gemini_client
"""
import os
from config.settings import settings


def configure_langsmith() -> None:
    """Set all required LangSmith + LangChain tracing environment variables."""
    api_key  = settings.langsmith_api_key
    endpoint = settings.langsmith_endpoint
    project  = settings.langsmith_project

    # ── LangSmith native vars (LangSmith SDK >= 0.1) ──────────────────────────
    os.environ["LANGSMITH_TRACING"]  = "true"
    os.environ["LANGSMITH_API_KEY"]  = api_key
    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    os.environ["LANGSMITH_PROJECT"]  = project

    # ── LangChain tracing vars (required by langchain-core / LangGraph) ───────
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = api_key
    os.environ["LANGCHAIN_ENDPOINT"]   = endpoint
    os.environ["LANGCHAIN_PROJECT"]    = project

    # ── Google API key ─────────────────────────────────────────────────────────
    os.environ["GOOGLE_API_KEY"]  = settings.gemini_api_key
    os.environ["GEMINI_API_KEY"]  = settings.gemini_api_key

    if settings.debug:
        print(f"[LangSmith] Tracing -> {endpoint}")
        print(f"[LangSmith] Project  -> {project}")


def get_gemini_client():
    """
    Return a LangSmith-traced google-genai client using wrappers.wrap_gemini().
    Use this for any direct google-genai calls outside LangChain.

    Example:
        client = get_gemini_client()
        resp = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Your prompt here",
        )
        print(resp.text)
    """
    try:
        from google import genai
        from langsmith import wrappers

        gemini_client = genai.Client()
        traced_client = wrappers.wrap_gemini(
            gemini_client,
            tracing_extra={
                "tags": ["gemini", "finassist-ai"],
                "metadata": {
                    "integration": "google-genai",
                    "project": settings.langsmith_project,
                },
            },
        )
        return traced_client

    except ImportError as e:
        # Fall back to untraced client if langsmith wrappers not available
        from google import genai
        return genai.Client()
