"""
FinAssist AI — Gemini Agentic Orchestrator (Phase 2)
======================================================
Replaces local Ollama generation with Google Gemini acting as a dynamic
tool-calling agent. The old LangGraph DAG (graph.py / nodes.py) is
PRESERVED and still runs for backward-compatible tracing via LangSmith.

Architecture:
    User Query
        │
        ▼
    GeminiAgent.run(query, profile, history)
        │
        ├── Gemini decides which tool(s) to call
        │
        ├── retrieve_tax_docs()    → Hybrid RAG
        ├── calculate_tax()        → Python Tax Engine
        ├── analyze_form16()       → Form 16 Parser
        ├── get_optimization_opportunities() → Gap Advisor
        └── web_search_tax_updates()         → Live Search
        │
        ▼
    Gemini synthesizes final answer
        │
        ▼
    Returns: { final_answer, tool_calls_made, citations, steps, ... }

LangSmith Tracing:
    - Every tool invocation is tagged with metadata
    - run_id and project match LANGSMITH_PROJECT=TAX_RAG
    - Tool inputs/outputs visible in LangSmith dashboard

Anti-Hallucination Guarantees:
    1. calculate_tax() results are labelled AUTHORITATIVE — DO NOT OVERRIDE
    2. Section citations are derived only from retrieve_tax_docs() output
    3. System prompt forbids Gemini from doing arithmetic itself
    4. Regime locking: if user requests old/new, only that regime is shown
    5. Freelancer rule: is_freelancer=True → Section 44ADA applied
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# ─── System prompt (anti-hallucination rules) ──────────────────────────────────

_SYSTEM_PROMPT = """You are FinAssist AI — an expert Indian tax advisor for FY 2025-26 (AY 2026-27).

CORE ANTI-HALLUCINATION RULES (MANDATORY):
1. NEVER calculate tax amounts yourself. Call calculate_tax() and use ONLY those numbers.
2. NEVER cite Income Tax Act sections not returned by retrieve_tax_docs().
3. NEVER invent deduction limits, slab rates, or cess percentages from memory.
4. If calculate_tax() returns a number, use it verbatim. Do not round, re-derive, or "explain" with different math.
5. All answers MUST be grounded in tool outputs.

SPECIAL RULES:
- FREELANCER / 44ADA: If the user mentions freelance, self-employed, consultant, or professional income:
  * Apply Section 44ADA (50% presumptive profit). Pass is_freelancer=True to calculate_tax().
  * TDS is under Section 194J (professional fees), NOT 194-I (rent). Never confuse these.
- CRYPTO / VDA: All Virtual Digital Assets (Bitcoin, ETH, NFTs, etc.) are taxed FLAT 30% + 4% cess = 31.2% under Section 115BBH.
  * Section 87A rebate does NOT apply to VDA gains.
  * VDA losses CANNOT be set off against other income.
- REGIME LOCK: If the user says "old regime" or "new regime", show ONLY that regime's numbers.

RESPONSE FORMAT:
- Use plain English with Rs. (not ₹ symbol)
- Present key figures clearly: Old Regime Tax, New Regime Tax, Recommended Regime, Savings
- Keep it concise. Don't pad with generic financial advice.
- End with relevant section references from the RAG results.
"""

# ─── Helpers ───────────────────────────────────────────────────────────────────

def _detect_regime(query: str) -> str | None:
    q = query.lower()
    if any(x in q for x in ["old regime", "old tax regime", "traditional regime"]):
        return "old"
    if any(x in q for x in ["new regime", "115bac", "new tax regime", "default regime"]):
        return "new"
    return None


def _extract_income(query: str) -> dict:
    """Parse gross income from freeform query text."""
    text = query.lower().replace(",", "")
    def to_amount(n, unit):
        n = float(n)
        if "crore" in unit or "cr" in unit: return n * 1e7
        if "lakh" in unit or "lpa" in unit or "lac" in unit or unit.strip() == "l": return n * 1e5
        if "k" in unit: return n * 1e3
        return n
    patterns = [
        (r"(?:earn|salary|ctc|package|income|gross)[^\d]*(\d+\.?\d*)\s*(lpa|lakh|lakhs|lac|crore|cr|k)", "gross"),
        (r"(?:if\s+i\s+earn|suppose|say my income is|assuming)[^\d]*(\d+\.?\d*)\s*(lpa|lakh|lakhs|lac|crore|cr|k)", "gross"),
        (r"(\d+\.?\d*)\s*(lpa|lakh|lakhs|lac)\b", "gross"),
    ]
    profile: dict = {}
    for pat, _ in patterns:
        m = re.search(pat, text)
        if m:
            amt = to_amount(m.group(1), m.group(2))
            if amt > 0:
                profile["gross_salary"] = amt
                break
    if any(k in text for k in ["freelancer", "freelance", "self-employed", "consultant", "contractor"]):
        profile["is_freelancer"] = True
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# _RawGeminiClient — Direct HTTP client bypassing google-genai SDK auth
# Uses X-goog-api-key header (works with AQ. format auth keys, same as curl)
# ─────────────────────────────────────────────────────────────────────────────

class _RawGeminiClient:
    """
    Thin wrapper around the Gemini REST API.
    Sends X-goog-api-key header so AQ. auth keys work correctly
    (identical to: curl -H 'X-goog-api-key: AQ.xxx' .../generateContent).
    """
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str):
        self._api_key = api_key

    @staticmethod
    def _fn_decl(fn) -> dict:
        """Build a FunctionDeclaration dict from a Python callable."""
        import inspect
        sig = inspect.signature(fn)
        props: dict = {}
        required: list[str] = []
        for name, p in sig.parameters.items():
            ann = p.annotation
            if ann is inspect.Parameter.empty:
                ptype = "string"
            elif ann in (int, float):
                ptype = "number"
            elif ann is bool:
                ptype = "boolean"
            elif ann is dict or (hasattr(ann, "__origin__") and ann.__origin__ is dict):
                ptype = "object"
            else:
                ptype = "string"
            props[name] = {"type": ptype, "description": name}
            if p.default is inspect.Parameter.empty:
                required.append(name)
        return {
            "name": fn.__name__,
            "description": (fn.__doc__ or fn.__name__).strip().split("\n")[0][:200],
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }

    def _serialize_message(self, msg) -> dict:
        """Convert message to REST API dict format."""
        if isinstance(msg, dict):
            return msg
        # Handle legacy gtypes.Content objects gracefully
        role = getattr(msg, "role", "user")
        parts_raw = getattr(msg, "parts", [])
        parts = []
        for p in parts_raw:
            if isinstance(p, dict):
                parts.append(p)
            elif hasattr(p, "text") and p.text:
                parts.append({"text": p.text})
        return {"role": role, "parts": parts}

    def generate_content(
        self,
        model: str,
        messages: list,
        tools: list | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
    ) -> dict:
        """
        Call Gemini generateContent REST endpoint.
        Retries on 503 overload with backoff, then falls back to gemini-2.0-flash.
        Returns a normalized dict: {text, parts, ...}
        """
        import time
        import requests as _req

        url_tmpl = self.BASE + "/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self._api_key,
        }
        body: dict = {
            "contents": [self._serialize_message(m) for m in messages],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if tools:
            body["tools"] = [{"functionDeclarations": [self._fn_decl(fn) for fn in tools]}]

        # Try primary model first (with retries), then stable alias fallback
        # gemini-flash-latest always resolves to the latest available flash model
        # Retryable: 503 overload, 429 rate-limit, ReadTimeout
        import requests.exceptions as _rex
        models_to_try = [model, "gemini-flash-latest"]
        if model == "gemini-flash-latest":
            models_to_try = [model]  # no point duplicating the alias
        last_error = ""
        for attempt_model in models_to_try:
            url = url_tmpl.format(model=attempt_model)
            for attempt in range(3):
                wait = 2 ** attempt  # 1s, 2s, 4s
                try:
                    resp = _req.post(url, headers=headers, json=body, timeout=120)  # 120s for tool-calling chains
                    if resp.status_code == 200:
                        data = resp.json()
                        candidate = data.get("candidates", [{}])[0]
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        text = "".join(p.get("text", "") for p in parts if "text" in p)
                        if attempt_model != model:
                            logger.warning(f"Used fallback model '{attempt_model}' (primary '{model}' unavailable)")
                        return {"parts": parts, "text": text, "raw": data, "model_used": attempt_model}

                    err_msg = resp.json().get("error", {}).get("message", resp.text[:200])
                    if resp.status_code in (503, 429):
                        logger.warning(f"{resp.status_code} on '{attempt_model}' attempt {attempt+1}/3 — retrying in {wait}s")
                        time.sleep(wait)
                        last_error = f"{resp.status_code} {err_msg}"
                        continue
                    # Non-retryable error
                    raise RuntimeError(f"{resp.status_code} {err_msg}")

                except (RuntimeError, KeyboardInterrupt):
                    raise
                except _rex.ReadTimeout:
                    logger.warning(f"ReadTimeout on '{attempt_model}' attempt {attempt+1}/3 — retrying in {wait}s")
                    time.sleep(wait)
                    last_error = f"ReadTimeout after 120s"
                    continue
                except Exception as e:
                    raise RuntimeError(str(e)) from e

        raise RuntimeError(f"All models unavailable after retries. Last error: {last_error}")


# ─────────────────────────────────────────────────────────────────────────────
# GeminiAgent
# ─────────────────────────────────────────────────────────────────────────────

class GeminiAgent:
    """
    Gemini-based agentic orchestrator for FinAssist AI.

    Usage:
        agent = GeminiAgent()
        result = agent.run(user_query="I earn 18 LPA, should I pick old or new regime?")
    """

    def __init__(self):
        self._client = None
        self._model_name: str = "gemini-3.8-flash"

    # ── Lazy Gemini client ────────────────────────────────────────────────────

    def _get_client(self) -> "_RawGeminiClient":
        if self._client is not None:
            return self._client
        try:
            from config.settings import settings
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set in .env")
            # Use raw HTTP client — works with AQ. auth keys via X-goog-api-key header
            # (same as: curl -H 'X-goog-api-key: AQ.xxx' ...)
            self._client = _RawGeminiClient(api_key=settings.gemini_api_key)
            # Prefer LLM_MODEL env var (llm_model), then GEMINI_MODEL (gemini_model), then hardcoded alias
            model_raw = (
                getattr(settings, "llm_model", "") or
                getattr(settings, "gemini_model", "") or
                "gemini-flash-latest"
            ).removeprefix("models/")
            self._model_name = model_raw
            logger.info(f"GeminiAgent: using model '{self._model_name}' (raw HTTP)")
            return self._client
        except Exception as e:
            logger.error(f"GeminiAgent init failed: {e}")
            raise

    # ── Tool callables list (passed directly to GenerateContentConfig) ─────────

    @staticmethod
    def _get_tool_callables() -> list:
        """
        Return the five tool functions as a plain list of callables.
        google-genai v2.x auto-converts callables → FunctionDeclaration via
        their docstrings + type annotations when passed to
        GenerateContentConfig(tools=[...]).
        """
        try:
            from recommendation_engine.tools import (
                retrieve_tax_docs,
                calculate_tax,
                analyze_form16,
                get_optimization_opportunities,
                web_search_tax_updates,
            )
            return [
                retrieve_tax_docs,
                calculate_tax,
                analyze_form16,
                get_optimization_opportunities,
                web_search_tax_updates,
            ]
        except Exception as e:
            logger.warning(f"Could not load tool callables: {e}")
            return []

    # ── Execute a single tool call returned by Gemini ─────────────────────────

    @staticmethod
    def _dispatch_tool(name: str, args: dict) -> Any:
        """Map Gemini function call name → actual Python function, run it."""
        from recommendation_engine.tools import (
            retrieve_tax_docs, calculate_tax, analyze_form16,
            get_optimization_opportunities, web_search_tax_updates,
        )
        dispatch = {
            "retrieve_tax_docs":            retrieve_tax_docs,
            "calculate_tax":                calculate_tax,
            "analyze_form16":               analyze_form16,
            "get_optimization_opportunities": get_optimization_opportunities,
            "web_search_tax_updates":       web_search_tax_updates,
        }
        fn = dispatch.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            # Wrap each tool call as a LangSmith child run
            try:
                from langsmith import traceable
                traced_fn = traceable(
                    fn,
                    name=f"tool:{name}",
                    run_type="tool",
                    tags=["finassist", "gemini-agent", name],
                )
                return traced_fn(**args)
            except ImportError:
                return fn(**args)
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return {"error": str(e)}

    # ── LangSmith metadata tagging ────────────────────────────────────────────

    @staticmethod
    def _configure_langsmith():
        try:
            from config.langsmith_config import configure_langsmith
            configure_langsmith()
        except Exception:
            pass

    # ── Main entry point (LangSmith-traced) ──────────────────────────────────

    def run(
        self,
        user_query: str,
        tax_profile: dict | None = None,
        form16_path: str | None = None,
        chat_history: list | None = None,
        preferred_regime: str | None = None,
    ) -> dict:
        """
        Run the Gemini Agent tool-calling loop for a given query.

        Returns a plain dict:
        {
            "final_answer":    str,
            "tool_calls_made": [{"tool": name, "args": {...}, "result": {...}}, ...],
            "citations":       str,
            "steps":           [str, ...],
            "processing_steps": [str, ...],   # alias of steps for graph compat
            "error":           str | None,
        }
        """
        self._configure_langsmith()
        steps: list[str] = ["gemini_agent_start"]
        tool_calls_made: list[dict] = []
        citations = ""
        final_answer = ""
        error = None

        # ── Open a top-level LangSmith run (whole agent + child tool calls visible) ──
        _ls_run = None
        try:
            from langsmith import Client as _LSClient
            from config.settings import settings as _s
            _ls_client = _LSClient()
            _ls_run = _ls_client.create_run(
                name="GeminiAgent.run",
                run_type="chain",
                project_name=_s.langsmith_project,
                inputs={"query": user_query, "tax_profile": tax_profile or {}},
                tags=["finassist", "gemini-agent", _s.gemini_model],
            )
        except Exception as _ls_err:
            logger.debug(f"LangSmith run create skipped: {_ls_err}")

        try:
            client = self._get_client()
        except Exception as e:
            return {
                "final_answer": f"[Gemini Agent unavailable: {e}. Check GEMINI_API_KEY in .env]",
                "tool_calls_made": [], "citations": "", "steps": steps,
                "processing_steps": steps, "error": str(e),
            }

        # ── Detect context from query ──────────────────────────────────────
        regime = preferred_regime or _detect_regime(user_query)
        parsed_profile = _extract_income(user_query)
        merged_profile = {**(tax_profile or {}), **parsed_profile}

        # ── Inject prior tax_profile as user context if available ──────────
        context_prefix = ""
        if merged_profile.get("gross_salary"):
            context_prefix = (
                f"\n[User context: gross_salary=Rs.{merged_profile['gross_salary']:,.0f}"
            )
            if merged_profile.get("is_freelancer"):
                context_prefix += ", taxpayer_type=Freelancer/Professional (44ADA)"
            if regime:
                context_prefix += f", preferred_regime={regime.upper()}"
            context_prefix += "]\n"

        regime_lock = ""
        if regime == "old":
            regime_lock = "\n[REGIME LOCK: Show ONLY Old Regime figures. Do NOT mention New Regime.]\n"
        elif regime == "new":
            regime_lock = "\n[REGIME LOCK: Show ONLY New Regime (Section 115BAC) figures. Do NOT mention Old Regime.]\n"

        full_query = _SYSTEM_PROMPT + regime_lock + context_prefix + "\nUser question: " + user_query

        # ── Agentic tool-calling loop ──────────────────────────────────────
        try:
            from google.genai import types as gtypes

            # Load tool callables — passed directly to GenerateContentConfig.
            # In google-genai v2.x, callables in tools= are auto-converted to
            # FunctionDeclarations from their signatures + docstrings.
            tool_callables = self._get_tool_callables()
            messages: list = []

            # Include chat history as plain dicts (REST API format)
            for h in (chat_history or []):
                role = "model" if h.get("role") in ("assistant", "model") else "user"
                content = h.get("content", "")
                messages.append({"role": role, "parts": [{"text": content}]})

            # Add current query
            messages.append({"role": "user", "parts": [{"text": full_query}]})

            MAX_ITERATIONS = 6
            for iteration in range(MAX_ITERATIONS):
                steps.append(f"agent_iteration_{iteration + 1}")

                response = client.generate_content(
                    model=self._model_name,
                    messages=messages,
                    tools=tool_callables,
                    temperature=0.1,
                    max_output_tokens=4096,
                )

                # Collect the model response
                model_parts: list = []
                function_calls = []

                for part in response.get("parts", []):
                    model_parts.append(part)
                    if "functionCall" in part:
                        function_calls.append(part["functionCall"])

                # Add model message to history
                messages.append({"role": "model", "parts": model_parts})

                if not function_calls:
                    # No tool calls → final text answer
                    final_answer = response.get("text", "") or ""
                    steps.append("agent_final_answer")
                    break

                # ── Dispatch each tool call ────────────────────────────────
                tool_response_parts: list = []
                for fc in function_calls:
                    tool_name = fc.get("name", "")
                    tool_args = fc.get("args", {})
                    steps.append(f"tool_call:{tool_name}")

                    tool_result = self._dispatch_tool(tool_name, tool_args)

                    # Capture citations from RAG
                    if tool_name == "retrieve_tax_docs":
                        citations = tool_result.get("citations", "")

                    # Inject Form 16 path if agent requested it
                    if tool_name == "analyze_form16" and not tool_args.get("file_path") and form16_path:
                        tool_result = self._dispatch_tool("analyze_form16", {"file_path": form16_path})

                    tool_calls_made.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                    })

                    tool_response_parts.append({
                        "functionResponse": {
                            "name": tool_name,
                            "response": {"result": json.dumps(tool_result, default=str)},
                        }
                    })

                # Add tool responses to history
                messages.append({"role": "user", "parts": tool_response_parts})

            else:
                # Exceeded MAX_ITERATIONS — use whatever was last generated
                steps.append("agent_max_iterations_reached")
                if not final_answer:
                    final_answer = response.text or "[Agent did not produce a final answer]"

        except Exception as e:
            logger.error(f"GeminiAgent.run error: {e}")
            error = str(e)
            final_answer = f"[Agent error: {e}]"

        # ── Close the LangSmith run with outputs ──────────────────────────────
        result = {
            "final_answer":    final_answer,
            "tool_calls_made": tool_calls_made,
            "citations":       citations,
            "steps":           steps,
            "processing_steps": steps,
            "error":           error,
            "rag_context": {
                "context":   "",
                "citations": citations,
                "source_docs": [],
            },
        }
        if _ls_run is not None:
            try:
                _ls_client.update_run(
                    _ls_run.id,
                    outputs={"final_answer": final_answer, "tool_calls": len(tool_calls_made), "error": error},
                    error=error,
                    end_time=__import__("datetime").datetime.utcnow(),
                )
            except Exception as _ls_err:
                logger.debug(f"LangSmith run update skipped: {_ls_err}")
        return result


# ─── Module-level singleton ────────────────────────────────────────────────────
_agent_instance: GeminiAgent | None = None


def get_agent() -> GeminiAgent:
    """Return the module-level GeminiAgent singleton."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = GeminiAgent()
    return _agent_instance


def run_agent(
    user_query: str,
    tax_profile: dict | None = None,
    form16_path: str | None = None,
    chat_history: list | None = None,
) -> dict:
    """
    Convenience wrapper — same interface as graph.run_query().
    Returns a plain dict compatible with the Streamlit frontend.
    """
    return get_agent().run(
        user_query=user_query,
        tax_profile=tax_profile,
        form16_path=form16_path,
        chat_history=chat_history,
    )
