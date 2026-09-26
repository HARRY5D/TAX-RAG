"""
Gemini Section Verifier — FinAssist AI
=======================================
Uses Google Gemini API (google-generativeai) to independently verify
that the legal section citations in the local LLM's answer are correct.

WHAT THIS DOES:
  - Extracts all Section numbers (e.g., "Section 80C", "44ADA") from
    the local LLM answer.
  - Asks Gemini: "Are these sections correctly cited for this tax question?"
  - Gemini returns a structured confirmation / correction.
  - Any corrections are appended to the final answer as a verification footer.

WHAT THIS DOES NOT DO:
  - It does NOT verify the tax calculation numbers (those come from our
    deterministic Python engine and are always correct).
  - It does NOT replace the local LLM answer.
  - It does NOT generate the primary response.

GRACEFUL DEGRADATION:
  - If Gemini API key is missing, quota exceeded, or returns 4xx/5xx,
    the verifier silently marks result as skipped=True and the main answer
    is returned unchanged.
"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def _extract_section_refs(text: str) -> List[str]:
    """
    Extract Indian Income Tax Act section references from text.
    Handles formats like:
      - Section 80C, Section 80D, Section 44ADA
      - Sec 24(b), Sec 10(13A)
      - 80CCD(1B), 44AD, 194J
    """
    # Pattern: optional "Section"/"Sec" prefix, then number + optional letter/parens
    pattern = r"\b(?:[Ss]ection\s+|[Ss]ec\.?\s*)?(\d+[A-Z]{0,4}(?:\(\w+\))?)\b"
    matches = re.findall(pattern, text)
    # Deduplicate while preserving order; filter out bare numbers like "2025"
    seen = set()
    sections = []
    for m in matches:
        # Only keep if it looks like a real section (has at least one letter or is ≤ 3 digits)
        if re.search(r"[A-Za-z]", m) or (m.isdigit() and int(m) <= 300):
            normalized = f"Section {m}"
            if normalized not in seen:
                seen.add(normalized)
                sections.append(normalized)
    return sections


class GeminiVerifier:
    """
    Verifies legal section citations using Google Gemini API.
    Initialized once and reused across calls (lazy singleton pattern in nodes.py).
    """

    def __init__(self):
        self._model = None
        self._available = None  # None = not tested yet; True/False after first call

    def _get_model(self):
        """Lazy-load the Gemini client. Returns None if unavailable."""
        if self._model is not None:
            return self._model

        try:
            from config.settings import settings
            if not settings.gemini_api_key:
                logger.warning("GeminiVerifier: GEMINI_API_KEY not set. Skipping verification.")
                self._available = False
                return None

            import google.genai as genai
            client = genai.Client(api_key=settings.gemini_api_key)
            self._model = client
            # Build candidate list: primary -> fallback -> safety nets
            primary = getattr(settings, 'gemini_model', 'gemini-1.5-flash')
            if primary.startswith('models/'):
                primary = primary[len('models/'):]
            fallback = getattr(settings, 'gemini_fallback_model', 'gemini-3.5-flash')
            # Safety nets use current 3.x generation (1.5-series retired mid-2026)
            raw_candidates = [primary, fallback, 'gemini-3.5-flash', 'gemini-3.5-flash-lite']
            seen: set = set()
            self._model_candidates = [
                m for m in raw_candidates if not (m in seen or seen.add(m))
            ]
            self._model_name = self._model_candidates[0]
            self._exhausted_models: set = set()
            self._available = True
            logger.info(f"GeminiVerifier: initialized. candidates={self._model_candidates}")
            return self._model

        except ImportError:
            logger.warning(
                "GeminiVerifier: 'google-genai' package not installed. "
                "Run: pip install google-genai"
            )
            self._available = False
            return None
        except Exception as e:
            logger.warning(f"GeminiVerifier: initialization failed — {e}")
            self._available = False
            return None

    def verify_sections(
        self,
        user_question: str,
        local_answer: str,
        rag_citations: str = "",
    ) -> dict:
        """
        Verify that section citations in `local_answer` are correct for `user_question`.

        Returns a dict with:
          - verified (bool): True if all cited sections are correct
          - verified_sections (list): confirmed correct sections
          - corrections (list): list of correction strings if any are wrong
          - gemini_raw_response (str): raw Gemini reply
          - skipped (bool): True if Gemini was unavailable
        """
        model = self._get_model()
        if model is None:
            return {
                "verified": False,
                "verified_sections": [],
                "corrections": [],
                "gemini_raw_response": "",
                "skipped": True,
            }

        cited_sections = _extract_section_refs(local_answer)
        if not cited_sections:
            return {
                "verified": True,
                "verified_sections": [],
                "corrections": [],
                "gemini_raw_response": "No section citations found to verify.",
                "skipped": False,
            }

        sections_str = ", ".join(cited_sections)
        verification_prompt = f"""You are an Indian Income Tax expert for FY 2025-26.

A tax assistant answered the following user question and cited these legal sections:

USER QUESTION: {user_question}

CITED SECTIONS: {sections_str}

RAG CONTEXT (what the assistant had access to):
{rag_citations or "No RAG citations available."}

YOUR TASK:
For each cited section, respond with one of:
  ✅ CORRECT — [Section X] is correctly cited for this context
  ❌ WRONG — [Section X] is incorrectly cited. The correct section is [Y] because [reason]
  ⚠️ PARTIAL — [Section X] is cited but incomplete or misleading: [explanation]

Be extremely concise. Only verify section numbers, not tax amounts.
If all sections are correct, say: "ALL SECTIONS VERIFIED ✅"
"""

        # Try each model in order; skip models that already hit quota this session
        candidates = [
            m for m in getattr(self, '_model_candidates', [self._model_name])
            if m not in getattr(self, '_exhausted_models', set())
        ]
        if not candidates:
            logger.warning("GeminiVerifier: all model quotas exhausted for this session.")
            return {
                "verified": False,
                "verified_sections": [],
                "corrections": [],
                "gemini_raw_response": "All Gemini model quotas exhausted.",
                "skipped": True,
            }

        client = model
        last_error = None
        for model_name in candidates:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=verification_prompt,
                )
                raw = response.text.strip() if response.text else ""
                self._model_name = model_name  # remember which model worked

                corrections: List[str] = []
                verified_sections: List[str] = []
                for line in raw.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if "❌ WRONG" in line or "❌" in line:
                        corrections.append(line)
                    elif "⚠️ PARTIAL" in line or "⚠️" in line:
                        corrections.append(line)
                    elif "✅ CORRECT" in line or "✅" in line:
                        m = re.search(r"Section\s+\S+", line)
                        verified_sections.append(m.group(0) if m else line)

                all_correct = len(corrections) == 0 or "ALL SECTIONS VERIFIED" in raw
                return {
                    "verified": all_correct,
                    "verified_sections": verified_sections,
                    "corrections": corrections,
                    "gemini_raw_response": raw,
                    "skipped": False,
                }

            except Exception as e:
                last_error = e
                err = str(e).lower()
                is_quota = "quota" in err or "429" in err or "rate" in err or "resource_exhausted" in err
                if is_quota:
                    logger.warning(f"GeminiVerifier: quota exceeded for '{model_name}'. Trying next model.")
                    if not hasattr(self, '_exhausted_models'):
                        self._exhausted_models = set()
                    self._exhausted_models.add(model_name)
                    continue  # try next candidate
                else:
                    logger.warning(f"GeminiVerifier: non-quota error on '{model_name}' — {e}")
                    break  # non-quota error, don't retry

        return {
            "verified": False,
            "verified_sections": [],
            "corrections": [f"Gemini verification unavailable: {last_error}"],
            "gemini_raw_response": str(last_error),
            "skipped": True,
        }

    def format_verification_footer(self, result: dict) -> str:
        """
        Format the verification result as a footer to append to the final answer.
        Returns empty string if verification was skipped or all sections correct.
        """
        if result.get("skipped"):
            return ""  # Don't show anything if Gemini was unavailable

        corrections = result.get("corrections", [])
        verified = result.get("verified_sections", [])

        if not corrections and not verified:
            return ""  # Nothing to report

        lines = ["\n\n---\n**📋 Section Citation Verification (by Gemini)**"]

        if verified:
            lines.append(f"✅ Confirmed correct: {', '.join(verified)}")

        if corrections:
            lines.append("⚠️ **Corrections found:**")
            for c in corrections:
                lines.append(f"  - {c}")
            lines.append(
                "\n*Note: The tax calculation numbers above are from our deterministic "
                "Python engine and are unaffected by section citation accuracy.*"
            )

        return "\n".join(lines)


# Module-level singleton
_verifier_instance = None


def get_verifier() -> GeminiVerifier:
    """Return the module-level GeminiVerifier singleton."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = GeminiVerifier()
    return _verifier_instance
