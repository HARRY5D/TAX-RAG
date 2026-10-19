"""
LangSmith Configuration for FinAssist AI.

Traces both:
  - Gemini Agent (via explicit create_run / @traceable in agent.py)
  - LangGraph DAG (auto-traced by langchain-core when LANGCHAIN_TRACING_V2=true)

Completely non-blocking — if LangSmith is unreachable or the key is invalid,
the app continues normally with tracing silently disabled.
"""
import logging
import os

logger = logging.getLogger(__name__)

_configured = False
_tracing_enabled = False


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing was successfully configured."""
    return _tracing_enabled


def _disable_tracing() -> None:
    """Turn off all tracing environment flags."""
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2",
                "LANGCHAIN_CALLBACKS_BACKGROUND"):
        os.environ[var] = "false"


def configure_langsmith() -> None:
    """
    Configure LangSmith + LangChain tracing.
    Idempotent. Silently disables tracing if key is invalid or server unreachable.
    """
    global _configured, _tracing_enabled
    if _configured:
        return
    _configured = True

    try:
        from config.settings import settings

        api_key  = settings.langsmith_api_key or ""
        endpoint = settings.langsmith_endpoint or "https://api.smith.langchain.com"
        project  = settings.langsmith_project  or "TAX_RAG"

        # ── Key sanity check (real keys are 50+ chars) ─────────────────────────
        if len(api_key) < 30:
            logger.warning(
                f"LangSmith DISABLED — LANGSMITH_API_KEY too short ({len(api_key)} chars). "
                "Get your full key at https://smith.langchain.com/settings#api-keys"
            )
            _disable_tracing()
            return

        # ── Probe: test write access to /runs (not just /info read) ──────────
        try:
            import requests, datetime, uuid
            # Send a minimal test run and immediately delete it
            test_id = str(uuid.uuid4())
            probe = requests.post(
                f"{endpoint}/runs",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "id": test_id,
                    "name": "_probe",
                    "run_type": "chain",
                    "start_time": datetime.datetime.utcnow().isoformat() + "Z",
                    "inputs": {},
                    "session_name": project,
                },
                timeout=6,
            )
            if probe.status_code in (401, 403):
                logger.warning(
                    f"LangSmith DISABLED — key lacks write permission to /runs ({probe.status_code}). "
                    "Create a new Personal API key at https://smith.langchain.com/settings#api-keys "
                    f"and make sure the '{project}' project exists in your workspace."
                )
                _disable_tracing()
                return
            # Clean up toxic/conflicting legacy variables if present
            os.environ.pop("LANGSMITH_WORKSPACE_ID", None)

            # Clean up the test run if created (LangSmith returns 200, 201, or 202)
            if probe.status_code in (200, 201, 202):
                try:
                    requests.patch(
                        f"{endpoint}/runs/{test_id}",
                        headers={"x-api-key": api_key, "Content-Type": "application/json"},
                        json={"end_time": datetime.datetime.utcnow().isoformat() + "Z", "outputs": {}},
                        timeout=3,
                    )
                except Exception:
                    pass
        except Exception as probe_err:
            logger.warning(f"LangSmith probe failed ({probe_err}). Tracing disabled.")
            _disable_tracing()
            return

        # ── Set environment variables ───────────────────────────────────────────
        os.environ["LANGSMITH_TRACING"]  = "true"
        os.environ["LANGSMITH_API_KEY"]  = api_key
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGSMITH_PROJECT"]  = project

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"]    = api_key
        os.environ["LANGCHAIN_ENDPOINT"]   = endpoint
        os.environ["LANGCHAIN_PROJECT"]    = project

        # Prevent "Event loop is closed" in Streamlit — force synchronous ingestion
        os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"]         = "false"
        os.environ["LANGSMITH_BACKGROUND_FLUSH_INTERVAL_MS"] = "0"

        _tracing_enabled = True
        logger.info(f"LangSmith ENABLED → project '{project}' at {endpoint}")

    except Exception as e:
        logger.warning(f"LangSmith configure error: {e}. Tracing disabled.")
        _disable_tracing()
