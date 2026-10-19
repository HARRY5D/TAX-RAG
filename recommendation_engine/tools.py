"""
FinAssist AI — Unified Tool Registry (Phase 1)
================================================
Declares all agent tools as typed, annotated callables that Gemini
can call via function-calling / MCP.

Tools:
  1. retrieve_tax_docs(query, top_k)       — Hybrid RAG: FAISS + BM25 + BGE Rerank
  2. calculate_tax(profile)               — Deterministic Python Tax Engine
  3. analyze_form16(file_path)            — Form 16 PDF Parser
  4. get_optimization_opportunities(profile) — Deduction Gap Advisor
  5. web_search_tax_updates(query)        — Lightweight DuckDuckGo search

Each tool:
  - Has a clear docstring (becomes the Gemini function description)
  - Accepts and returns plain dicts / primitives (JSON-serializable)
  - Is decorated with @langsmith_traceable for full LangSmith visibility
  - Never raises — returns {"error": "..."} on failure
"""
from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy singletons (loaded only on first call) ───────────────────────────────
_retriever = None
_reranker  = None


def _get_rag_pipeline():
    """Lazy-load FAISS + BGE retrieval pipeline (singleton)."""
    global _retriever, _reranker
    if _retriever is None:
        from config.settings import settings
        from embeddings.embedder import BGEEmbedder
        from vectordb.faiss_store import FAISSStore
        from rag.retriever import HybridRetriever
        from rag.reranker import BGEReranker
        embedder    = BGEEmbedder(settings.embedding_model)
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


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1: Hybrid RAG Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_tax_docs(query: str, top_k: int = 5) -> dict:
    """
    Retrieve relevant Indian Income Tax legal clauses and rules.

    Use this tool when the user asks any question about:
    - Tax sections (80C, 80D, HRA, NPS, 44ADA, 115BBH, etc.)
    - Deduction rules, limits, or eligibility
    - Old vs New regime differences
    - ITR filing requirements or document checklists
    - Crypto / VDA tax treatment

    Args:
        query:  The user's tax question in plain English.
        top_k:  Number of chunks to return (default 5).

    Returns:
        {
            "context":     "Combined text of retrieved legal chunks",
            "citations":   "Numbered source citations",
            "chunks_count": int,
            "error":       "..." (only if retrieval failed)
        }
    """
    try:
        from config.settings import settings
        retriever, reranker = _get_rag_pipeline()
        candidates = retriever.retrieve(query, top_k=settings.top_k_retrieval)
        reranked   = reranker.rerank(query, candidates, top_k=int(top_k))

        context_parts: list[str] = []
        citations:     list[str] = []
        for i, doc in enumerate(reranked, 1):
            meta    = doc.get("metadata", {})
            source  = meta.get("source", "")
            section = meta.get("section", "")
            context_parts.append(f"[{i}] {doc.get('text', '')}")
            cite = f"[{i}] {source}"
            if section and section != "general":
                cite += f" — Section {section}"
            citations.append(cite)

        return {
            "context":      "\n\n---\n\n".join(context_parts),
            "citations":    "\n".join(citations),
            "chunks_count": len(reranked),
        }
    except FileNotFoundError:
        return {
            "context": "[RAG index not built. Run: python scripts/build_index.py]",
            "citations": "",
            "chunks_count": 0,
        }
    except Exception as e:
        logger.error(f"retrieve_tax_docs error: {e}")
        return {"context": "", "citations": "", "chunks_count": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2: Deterministic Tax Calculator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_tax(
    gross_salary: float = 0,
    basic_salary: float = 0,
    hra_received: float = 0,
    rent_paid: float = 0,
    elss: float = 0,
    ppf: float = 0,
    epf: float = 0,
    life_insurance: float = 0,
    home_loan_principal: float = 0,
    nsc: float = 0,
    other_80c: float = 0,
    health_insurance_self: float = 0,
    health_insurance_parents: float = 0,
    additional_nps_80ccd1b: float = 0,
    employer_nps: float = 0,
    savings_interest: float = 0,
    home_loan_interest: float = 0,
    other_income: float = 0,
    is_metro: bool = False,
    is_govt_employer: bool = False,
    is_self_occupied: bool = True,
    is_freelancer: bool = False,
) -> dict:
    """
    Calculate income tax under both Old and New regimes for FY 2025-26.
    Results are AUTHORITATIVE — Gemini MUST use these numbers and MUST NOT recalculate.

    Use this tool when the user:
    - Provides their salary or income figures
    - Asks "how much tax will I pay?"
    - Wants a regime comparison
    - Mentions specific investments (ELSS, PPF, NPS, etc.)

    Args:
        gross_salary:            Annual CTC / gross income in Rs.
        basic_salary:            Basic salary (default 50% of gross)
        hra_received:            HRA received from employer
        rent_paid:               Annual rent paid
        elss / ppf / epf / ...: 80C investments in Rs.
        health_insurance_self:   Health insurance premium (self + family)
        health_insurance_parents:Health insurance for parents
        additional_nps_80ccd1b:  Additional NPS contribution (80CCD1B — max 50,000)
        employer_nps:            Employer's NPS contribution (80CCD2)
        savings_interest:        Savings account interest (80TTA)
        home_loan_interest:      Home loan interest paid (Sec 24b)
        other_income:            Other taxable income
        is_metro:                True if renting in metro city (HRA)
        is_govt_employer:        True for government employees (NPS)
        is_self_occupied:        True if home loan is for self-occupied property
        is_freelancer:           True → applies Section 44ADA 50% presumptive profit

    Returns:
        Full tax breakdown dict with old_regime, new_regime, recommendation keys.
        Always includes source="tax_engine_deterministic".
    """
    # ── Coerce all inputs to correct types (Gemini may pass numbers as strings) ──
    gross_salary          = float(gross_salary          or 0)
    basic_salary          = float(basic_salary          or 0) or gross_salary * 0.5
    hra_received          = float(hra_received          or 0)
    rent_paid             = float(rent_paid             or 0)
    elss                  = float(elss                  or 0)
    ppf                   = float(ppf                   or 0)
    epf                   = float(epf                   or 0)
    life_insurance        = float(life_insurance        or 0)
    home_loan_principal   = float(home_loan_principal   or 0)
    nsc                   = float(nsc                   or 0)
    other_80c             = float(other_80c             or 0)
    health_insurance_self    = float(health_insurance_self    or 0)
    health_insurance_parents = float(health_insurance_parents or 0)
    additional_nps_80ccd1b   = float(additional_nps_80ccd1b  or 0)
    employer_nps          = float(employer_nps          or 0)
    savings_interest      = float(savings_interest      or 0)
    home_loan_interest    = float(home_loan_interest    or 0)
    other_income          = float(other_income          or 0)
    is_metro              = bool(is_metro)
    is_govt_employer      = bool(is_govt_employer)
    is_self_occupied      = bool(is_self_occupied)
    is_freelancer         = bool(is_freelancer)

    profile = dict(
        gross_salary=gross_salary,
        basic_salary=basic_salary,
        hra_received=hra_received,
        rent_paid=rent_paid,
        elss=elss, ppf=ppf, epf=epf,
        life_insurance=life_insurance,
        home_loan_principal=home_loan_principal,
        nsc=nsc, other_80c=other_80c,
        health_insurance_self=health_insurance_self,
        health_insurance_parents=health_insurance_parents,
        additional_nps_80ccd1b=additional_nps_80ccd1b,
        employer_nps=employer_nps,
        savings_interest=savings_interest,
        home_loan_interest=home_loan_interest,
        other_income=other_income,
        is_metro=is_metro,
        is_govt_employer=is_govt_employer,
        is_self_occupied=is_self_occupied,
        is_freelancer=is_freelancer,
    )
    try:
        from tax_engine.tax_tools import run_all_tax_tools
        result = run_all_tax_tools(profile)
        result["source"] = "tax_engine_deterministic"
        result["trust_level"] = "AUTHORITATIVE — DO NOT OVERRIDE"
        return result
    except Exception as e:
        logger.error(f"calculate_tax error: {e}")
        return {"error": str(e), "source": "tax_engine_deterministic"}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3: Form 16 Parser
# ─────────────────────────────────────────────────────────────────────────────

def analyze_form16(file_path: str) -> dict:
    """
    Parse an uploaded Form 16 PDF and extract the taxpayer's full tax profile.

    Use this tool when the user uploads or mentions a Form 16 document.
    Extracts: employer name, PAN, gross salary, all allowances,
    Chapter VI-A deductions (80C/80D/80G), and TDS paid.

    Args:
        file_path: Absolute or relative path to the Form 16 PDF.

    Returns:
        {
            "extracted": True/False,
            "tax_profile": { ... same keys as calculate_tax() ... },
            "employer": "...",
            "assessment_year": "...",
            "tds_deducted": float,
            "confidence": float (0–1),
            "notes": ["..."],
            "error": "..." (only on failure)
        }
    """
    try:
        from form16_parser.parser import extract_form16
        form16_data = extract_form16(file_path)
        tax_profile = form16_data.to_tax_profile()
        return {
            "extracted":       True,
            "tax_profile":     tax_profile,
            "employer":        getattr(form16_data, "employer_name", ""),
            "assessment_year": getattr(form16_data, "assessment_year", ""),
            "tds_deducted":    float(getattr(form16_data, "tds_deducted", 0) or 0),
            "confidence":      float(form16_data.extraction_confidence or 0.0),
            "notes":           form16_data.parsing_notes or [],
        }
    except Exception as e:
        logger.error(f"analyze_form16 error: {e}")
        return {"extracted": False, "tax_profile": {}, "error": str(e), "notes": [str(e)]}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4: Tax Optimization Advisor
# ─────────────────────────────────────────────────────────────────────────────

def get_optimization_opportunities(
    gross_salary: float = 0,
    current_80c: float = 0,
    current_80d_self: float = 0,
    current_80d_parents: float = 0,
    current_nps_80ccd1b: float = 0,
    current_home_loan_interest: float = 0,
) -> dict:
    """
    Identify unused tax-saving capacity and suggest investment opportunities.

    Use this tool when the user:
    - Asks "how can I save more tax?"
    - Asks about unused 80C / 80D / NPS capacity
    - Wants to know what deductions they're missing

    Args:
        gross_salary:              Annual gross income
        current_80c:               Total current 80C investments
        current_80d_self:          Health insurance for self/family
        current_80d_parents:       Health insurance for parents
        current_nps_80ccd1b:       Additional NPS (80CCD1B)
        current_home_loan_interest:Home loan interest

    Returns:
        {
            "opportunities": [...],
            "total_potential_savings": float,
            "summary": "...",
        }
    """
    # Gemini function-calling may pass numeric args as strings — coerce all upfront
    gross_salary              = float(gross_salary or 0)
    current_80c               = float(current_80c or 0)
    current_80d_self          = float(current_80d_self or 0)
    current_80d_parents       = float(current_80d_parents or 0)
    current_nps_80ccd1b       = float(current_nps_80ccd1b or 0)
    current_home_loan_interest = float(current_home_loan_interest or 0)

    profile = dict(
        gross_salary=gross_salary,
        basic_salary=gross_salary * 0.5,
        elss=0, ppf=0, epf=current_80c,
        health_insurance_self=current_80d_self,
        health_insurance_parents=current_80d_parents,
        additional_nps_80ccd1b=current_nps_80ccd1b,
        home_loan_interest=current_home_loan_interest,
    )
    try:
        from tax_engine.calculator import calculate_full_tax
        from optimization.optimizer import find_optimization_opportunities
        tax_result = calculate_full_tax(profile)
        opt = find_optimization_opportunities(profile=profile, tax_result=tax_result)
        return opt
    except Exception as e:
        logger.error(f"get_optimization_opportunities error: {e}")
        return {"opportunities": [], "total_potential_additional_savings": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5: Web Search (live tax updates / budget circulars)
# ─────────────────────────────────────────────────────────────────────────────

def web_search_tax_updates(query: str, max_results: int = 4) -> dict:
    """
    Search official Indian tax & government websites for the latest rulings,
    budget circulars, and income tax notifications.

    Searches the following authoritative sources:
      - incometax.gov.in       (Income Tax Department)
      - cbic.gov.in            (Central Board of Indirect Taxes)
      - finmin.nic.in          (Ministry of Finance)
      - cleartax.in            (leading Indian tax guidance portal)
      - economictimes.com      (financial news)

    Use this tool when:
    - The user asks about recent budget changes or Finance Act amendments
    - The query involves FY 2026-27 or later year rules
    - The curated knowledge base may not have the latest circular / notification
    - User needs an authoritative source URL to verify a claim

    Args:
        query:       Tax topic to search (e.g. "new regime 2026 slab change")
        max_results: Number of results to return (default 4)

    Returns:
        {
            "results": [
                {
                    "title":   "Official page title",
                    "snippet": "Relevant excerpt",
                    "url":     "https://incometax.gov.in/...",
                    "source":  "incometax.gov.in"
                },
                ...
            ],
            "citations": "Numbered citation list (same format as RAG citations)",
            "error":     "..." (only on failure)
        }
    """
    # Trusted Indian tax source domains (in priority order)
    TRUSTED_SITES = [
        "incometax.gov.in",
        "cbic.gov.in",
        "finmin.nic.in",
        "cleartax.in",
        "economictimes.indiatimes.com",
        "taxguru.in",
    ]

    results: list[dict] = []

    # ── Strategy 1: Google Custom Search API (if keys available) ──────────
    try:
        from config.settings import settings
        # settings may have GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID if user set them
        cse_key = getattr(settings, "google_cse_api_key", "") or ""
        cse_id  = getattr(settings, "google_cse_id", "")     or ""

        if cse_key and cse_id:
            import urllib.request, urllib.parse, json as _json
            site_query = " OR ".join(f"site:{s}" for s in TRUSTED_SITES[:4])
            encoded_q  = urllib.parse.quote(f"{query} India income tax {site_query}")
            url = (
                f"https://www.googleapis.com/customsearch/v1"
                f"?key={cse_key}&cx={cse_id}&q={encoded_q}&num={max_results}"
            )
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = _json.loads(resp.read())
            for item in data.get("items", [])[:max_results]:
                domain = item.get("displayLink", "")
                results.append({
                    "title":   item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url":     item.get("link", ""),
                    "source":  domain,
                })
    except Exception as cse_err:
        logger.debug(f"Google CSE unavailable: {cse_err}")

    # ── Strategy 2: Direct HTTP fetch from official IT portal ─────────────
    if not results:
        try:
            import urllib.request
            import html
            # Search incometax.gov.in search endpoint
            search_url = f"https://www.incometax.gov.in/iec/foportal/search?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": "Mozilla/5.0 FinAssist-AI/1.0 (research)"},
            )
            # We just return the URL as a reference link since parsing requires JS
            results.append({
                "title":   f"Income Tax Portal Search: {query}",
                "snippet": "Visit the official Income Tax Department portal for authoritative information.",
                "url":     search_url,
                "source":  "incometax.gov.in",
            })
        except Exception as e:
            logger.debug(f"IT portal fetch failed: {e}")

    # ── Strategy 3: Curated static reference links ────────────────────────
    # Always append known authoritative deep links relevant to common queries
    query_lower = query.lower()
    static_refs: list[dict] = []
    if any(k in query_lower for k in ["80c", "deduction", "investment"]):
        static_refs.append({
            "title":   "Section 80C Deductions — Income Tax Department",
            "snippet": "Official guide to Section 80C eligible investments and Rs.1,50,000 limit.",
            "url":     "https://www.incometax.gov.in/iec/foportal/help/individual/return-applicable-1#SFundamental",
            "source":  "incometax.gov.in",
        })
    if any(k in query_lower for k in ["nps", "80ccd", "pension"]):
        static_refs.append({
            "title":   "NPS Tax Benefits — PFRDA",
            "snippet": "Section 80CCD(1B) additional deduction up to Rs.50,000 for NPS subscribers.",
            "url":     "https://www.pfrda.org.in/index1.cshtml?lsid=6",
            "source":  "pfrda.org.in",
        })
    if any(k in query_lower for k in ["new regime", "115bac", "budget 2024", "budget 2025"]):
        static_refs.append({
            "title":   "New Tax Regime — Budget 2024-25 Changes",
            "snippet": "Revised slabs, Rs.75,000 standard deduction, and updated Section 115BAC provisions.",
            "url":     "https://www.incometax.gov.in/iec/foportal/help/individual/return-applicable-1#new-tax-regime",
            "source":  "incometax.gov.in",
        })
    if any(k in query_lower for k in ["crypto", "vda", "bitcoin", "nft", "virtual"]):
        static_refs.append({
            "title":   "VDA / Crypto Tax — Section 115BBH",
            "snippet": "30% flat tax on Virtual Digital Asset gains. Section 87A rebate not applicable to VDA.",
            "url":     "https://www.incometax.gov.in/iec/foportal/help/individual/return-applicable-1#VDA",
            "source":  "incometax.gov.in",
        })
    if any(k in query_lower for k in ["44ada", "freelancer", "presumptive", "professional"]):
        static_refs.append({
            "title":   "Section 44ADA Presumptive Taxation — Income Tax India",
            "snippet": "50% of gross receipts treated as profit for eligible professionals.",
            "url":     "https://www.incometax.gov.in/iec/foportal/help/individual/return-applicable-1#Presumptive",
            "source":  "incometax.gov.in",
        })

    # Merge, deduplicate, and cap at max_results
    all_results = results + static_refs
    seen_urls: set = set()
    deduped: list[dict] = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped.append(r)

    final = deduped[:max_results]

    # Build citation string in same format as RAG citations
    citations = "\n".join(
        f"[W{i}] {r['title']} — {r['source']} ({r['url']})"
        for i, r in enumerate(final, 1)
    )

    return {
        "results":   final,
        "citations": citations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — list of all tools exposed to the Gemini Agent
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    retrieve_tax_docs,
    calculate_tax,
    analyze_form16,
    get_optimization_opportunities,
    web_search_tax_updates,
]

