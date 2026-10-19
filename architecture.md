# FinAssist AI — Architecture Reference

> **Living document.** Updated across sessions. Contains the complete architectural blueprint, recent changelog, and the transition plan to the Gemini Agentic Tool-Calling & MCP architecture.

---

## Project Purpose & Vision

FinAssist AI is an intelligent Indian Tax Assistant for FY 2025-26 (AY 2026-27). 

### Core Tenets
1. **Zero LLM Math (Deterministic Computation)**: All numerical calculations (tax slabs, deductions, 87A rebate, HRA, 80C/80D caps, 44ADA presumptive profit) are executed by pure, deterministic Python code.
2. **Agentic Tool-Calling**: The LLM (Google Gemini) acts as an intelligent reasoning and orchestration agent that invokes specialized tools (RAG, Tax Engine, Form 16 Parser, Web Search) dynamically based on user intent.
3. **Hybrid RAG Grounding**: Legal context is retrieved via dense BGE embeddings, FAISS vector search, BM25 keyword matching, and cross-encoder BGE reranking.
4. **Full Observability & Tracing**: All agent decisions, tool invocations, and historical graph nodes are fully instrumented with **LangSmith**.
5. **MCP & Modular Tool Interfaces**: Core modules are packaged as standardized tool interfaces ready for Model Context Protocol (MCP) servers and FastAPI/Streamlit integration.

---

## High-Level Architecture Comparison

### Phase 1 & 2 (Historical): Sequential Pipeline & LangGraph DAG
Earlier versions executed a predefined sequential DAG pipeline:
```text
User Query ──► detect_intent ──► maybe_parse_form16 ──► calculate_tax ──► optimize ──► retrieve_context ──► generate_answer (Ollama) ──► gemini_verify ──► Final Answer
```
*(Preserved and logged in `recommendation_engine/graph.py` for backward compatibility, ablation testing, and LangSmith historical evaluation).*

### Phase 3 (Current & Active): Agentic Tool-Calling Architecture (Gemini Agent)
In the modern architecture, Google Gemini acts as the central reasoning agent with direct function/tool-calling capabilities:

```text
                             User Query / Chat Message
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Gemini Agent   │
                               │ (Function/Tool) │
                               └────────┬────────┘
                                        │
                         Dynamically chooses tool(s)
                                        │
          ┌───────────────┬─────────────┴───────────────┬────────────────┐
          ▼               ▼                             ▼                ▼
┌──────────────────┐ ┌──────────────┐          ┌────────────────┐ ┌─────────────┐
│retrieve_tax_docs │ │calculate_tax │          │analyze_form16  │ │web_search   │
│  (Hybrid RAG)    │ │(Tax Engine)  │          │(Form16 Parser) │ │(Live Info)  │
└────────┬─────────┘ └──────┬───────┘          └───────┬────────┘ └──────┬──────┘
         │                  │                          │                 │
    FAISS + BM25      Deterministic             PDF Extraction       Budget / IT
    + BGE Rerank      Python Slabs              + OCR Fallback       Notifications
         │                  │                          │                 │
   Legal Chunks        Tax Output                  Tax Profile        Web Snippets
         └──────────────────┼──────────────────────────┴─────────────────┘
                            ▼
                   ┌─────────────────┐
                   │  Gemini Agent   │
                   │ (Synthesizer)   │
                   └────────┬────────┘
                            │
                            ▼
                   Final Response to User
```

---

## Where Model Context Protocol (MCP) Fits

All internal engines are packaged as clean, modular tool interfaces that conform to standard tool calling and MCP server definitions:

```text
Gemini Agent / MCP Client
    │
    ├── retrieve_tax_docs(query, top_k) ──► Hybrid RAG (FAISS + BM25 + BGE Reranker)
    ├── calculate_tax(profile_dict)     ──► Deterministic Python Tax Engine
    ├── analyze_form16(pdf_path)        ──► Form 16 Part A/B Parser
    ├── optimize_tax(profile_dict)      ──► Deduction Gap & Savings Engine
    └── web_search(query)               ──► Live Tax Circulars & Finance News
```

### Component Roles & Guarantees
- **`retrieve_tax_docs()`**: Retrieves authoritative legal clauses from curated tax markdown documents (`80C.md`, `80D.md`, `HRA.md`, `Presumptive_Taxation.md`, etc.).
- **`calculate_tax()`**: Computes exact tax under Old and New regimes, 87A rebate, standard deductions, and cess. Guarantees mathematical precision.
- **`analyze_form16()`**: Parses employer, PAN, gross salary, allowances, chapter VI-A deductions, and TDS directly from uploaded PDFs.
- **`optimize_tax()`**: Analyzes unutilized capacities in 80C, 80CCD(1B), and 80D to suggest optimal investment avenues.

---

## Directory Map

```
TAX_QA/
├── architecture.md             ← THIS FILE (Living architecture & roadmap)
├── .env                        ← Local secrets & API keys (Git ignored)
├── .env.example                ← Environment template
├── .gitignore                  ← Configured to exclude .env, faiss_index, pycache
├── requirements.txt            ← Project dependencies
│
├── config/
│   ├── settings.py             ← Pydantic settings & environment validation
│   └── langsmith_config.py     ← LangSmith tracing setup and client
│
├── data/
│   ├── raw/                    ← Source PDFs (Income Tax Act sections, circulars)
│   └── curated/                ← Curated markdown tax rules (FY 2025-26)
│       ├── 80C.md
│       ├── 80D.md
│       ├── HRA.md
│       ├── Home_Loan.md
│       ├── NPS.md
│       ├── Old_vs_New_Regime.md
│       ├── Presumptive_Taxation.md   ← Section 44ADA freelancer rules
│       ├── Tax_Slabs.md              ← Salaried vs Freelancer tax slabs
│       ├── ITR_Filing.md
│       ├── ITR_Documents.md
│       └── VDA_Crypto.md
│
├── preprocessing/              ← PDF text & table extraction pipeline
│   ├── digital_extractor.py    ← pdfplumber / PyPDF digital text extraction
│   ├── ocr_extractor.py        ← pytesseract OCR fallback for scanned PDFs
│   ├── pdf_detector.py         ← Scanned vs digital PDF detector
│   └── section_detector.py     ← Regex & layout section boundary detector
│
├── chunking/                   ← Semantic & structural chunking
│   ├── semantic_chunker.py
│   └── metadata_tagger.py
│
├── embeddings/
│   └── embedder.py             ← BGEEmbedder (BAAI/bge-small-en-v1.5) with real-time log flush
│
├── vectordb/
│   ├── faiss_store.py          ← FAISS Vector Store manager (search, save, load)
│   └── faiss_index/            ← Serialized index & metadata (git ignored)
│
├── rag/
│   ├── retriever.py            ← HybridRetriever (Dense FAISS + Sparse BM25)
│   ├── reranker.py             ← BGEReranker (BAAI/bge-reranker-base)
│   └── rag_chain.py            ← RAG prompt templates & citation extractors
│
├── tax_engine/                 ← DETERMINISTIC PYTHON ENGINE (Zero LLM Math)
│   ├── slabs.py                ← Old/New regime tax slabs & 87A rebate
│   ├── hra.py                  ← HRA exemption (Sec 10(13A))
│   ├── deductions.py           ← 80C, 80D, 80TTA/80TTB caps
│   ├── nps.py                  ← NPS (80CCD(1), 80CCD(1B), 80CCD(2))
│   ├── home_loan.py            ← Section 24(b) interest & 80EEA
│   ├── rental.py               ← Rental income & 30% standard deduction
│   ├── calculator.py           ← Master orchestrator: calculate_full_tax()
│   └── tax_tools.py            ← Structured tool wrappers & JSON schemas
│
├── optimization/
│   └── optimizer.py            ← find_optimization_opportunities()
│
├── form16_parser/
│   ├── parser.py               ← 19-field PDF Form 16 extraction
│   └── schema.py               ← Form 16 Pydantic models
│
├── recommendation_engine/      ← Orchestration & Agent Layer
│   ├── agent.py                ← [NEW] Gemini Agent with dynamic tool calling
│   ├── tools.py                ← [NEW] Unified LangChain / Gemini tool registry
│   ├── state.py                ← GraphState (Pydantic state container)
│   ├── nodes.py                ← LangGraph node functions (historical & fallback)
│   ├── graph.py                ← LangGraph workflow definition (historical)
│   └── gemini_verifier.py      ← Section citation verifier
│
├── sample_form16s/             ← Test Form 16 PDFs (Junior Dev, Senior Mgr, Fresher)
│
├── api/                        ← FastAPI backend
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│
├── frontend/                   ← Streamlit Multi-page App
│   ├── app.py                  ← Main UI entrypoint & navigation
│   ├── _pages/
│   │   ├── dashboard.py        ← Financial overview & KPI cards
│   │   ├── tax_calculator.py   ← Interactive tax simulator
│   │   ├── tax_optimizer.py    ← Optimization engine & capacity charts
│   │   ├── regime_comparator.py← Side-by-side Old vs New regime comparison
│   │   ├── form16_analyzer.py  ← Form 16 upload, parsing, and auto-sync
│   │   └── tax_assistant.py    ← Chat interface connected to Gemini Agent
│   └── components/
│       ├── cards.py            ← UI metric cards
│       └── charts.py           ← Plotly breakdown charts & donuts
│
└── scripts/
    ├── build_index.py          ← Build FAISS index from chunks
    ├── process_pdfs.py         ← Extract & chunk PDFs
    └── ragas_eval.py           ← RAG evaluation metrics
```

---

## Key Design Decisions & Evolution

| Component | Historical Approach | New Agentic Approach | Rationale |
|---|---|---|---|
| **LLM Generation** | Local Ollama (Qwen2.5-Coder / DeepSeek) | Google Gemini 2.5 Flash / Pro | High speed, reliable tool calling / function calling, native JSON parsing, eliminating local GPU/RAM bottlenecks. |
| **Tax Calculation** | Deterministic Python Engine | Deterministic Python Engine (as Tool) | Zero LLM math. LLM receives structured tax results via tool calls and explains them without recalculating. |
| **Orchestration** | Hardcoded LangGraph DAG | Gemini Agent with Function Calling | Gemini dynamically decides whether to query RAG, calculate tax, parse Form 16, or search the web based on conversation context. |
| **Observability** | LangSmith (LangGraph Tracing) | LangSmith (Agent & Tool Call Tracing) | Every step, tool invocation, token count, and latency is transparently tracked in LangSmith. |
| **RAG Retrieval** | FAISS + BM25 + BGE Rerank | FAISS + BM25 + BGE Rerank (as Tool) | Preserves the high retrieval accuracy (96.7% context hit) without altering the curated knowledge base. |
| **Form 16 Integration** | Manual button click | Automatic session-state propagation + Tool | Instantly fills tax calculator and optimizer upon upload; can also be parsed dynamically by the agent. |

---

## Anti-Hallucination & Math Protection Framework

1. **Deterministic Authority**: The prompt explicitly enforces:
   ```text
   The numerical data provided by `calculate_tax` is authoritative and computed by a deterministic Python engine.
   You are strictly FORBIDDEN from recalculating, adjusting, or generating new arithmetic figures.
   ```
2. **Strict Section Citations**: Citations must be derived exclusively from `retrieve_tax_docs` or verified against official Income Tax Act sections.
3. **Regime Locking**: When a user specifies a target regime (Old or New), the agent adheres strictly to the chosen regime's slab limits and deduction eligibility.
4. **Freelancer & Presumptive Taxation (Sec 44ADA)**: Automatically applies 50% presumptive profit calculation when freelance income is identified.

---

## Implementation Plan: Transition to Gemini Agentic Tool Architecture

### Phase 1: Tool Interface Normalization & Registry
- **Objective**: Standardize all existing engines (`rag/retriever.py`, `tax_engine/calculator.py`, `form16_parser/parser.py`, `optimization/optimizer.py`) into typed, annotated tool functions.
- **Action Items**:
  1. Create `recommendation_engine/tools.py` declaring tools with clear docstrings and Pydantic argument schemas:
     - `retrieve_tax_docs(query: str, top_k: int = 5)`
     - `calculate_tax_breakdown(gross_salary: float, basic_salary: float, ...)`
     - `parse_form16_document(file_path: str)`
     - `get_optimization_advice(gross_salary: float, current_80c: float, ...)`
     - `web_search_tax_updates(query: str)`
  2. Verify JSON schema generation for Gemini function declarations.

### Phase 2: Gemini Agent Implementation with LangSmith Tracing
- **Objective**: Build the Gemini Agent orchestrator that binds the tools and executes the tool-call loop.
- **Action Items**:
  1. Implement `recommendation_engine/agent.py` utilizing Google GenAI / LangChain Gemini integration.
  2. Configure LangSmith callback handlers and tags for complete tracing.
  3. Support multi-turn chat memory and session state context passing.
  4. Retain `recommendation_engine/graph.py` and `recommendation_engine/nodes.py` with clear documentation for backward-compatible comparison.

### Phase 3: Frontend & Chat Assistant Integration
- **Objective**: Connect Streamlit `tax_assistant.py` to the new Gemini Agent.
- **Action Items**:
  1. Update `frontend/_pages/tax_assistant.py` to stream responses from the Gemini Agent.
  2. Display interactive tool call badges (e.g., `[Called Tool: retrieve_tax_docs]`, `[Called Tool: calculate_tax]`) in the chat UI for enhanced transparency.
  3. Ensure seamless handoff between Form 16 upload, Tax Calculator, and Tax Assistant via `st.session_state`.

### Phase 4: Verification, Evaluation & Documentation
- **Objective**: Validate end-to-end functionality against sample scenarios.
- **Action Items**:
  1. Test complex multi-intent queries (e.g., *"I earn 18 LPA, have 1.5L in EPF, and pay 30k rent. Which regime is better and what sections apply?"*).
  2. Confirm tool invocation correctness in LangSmith traces.
  3. Run RAG evaluation script (`scripts/ragas_eval.py`) to confirm zero regressions in context hit and faithfulness.

---

## Session Progress Log

### Session 1 to 3
- Full project scaffold, tax engine modules, FAISS vector store, Streamlit UI, and initial LangSmith tracing.
- Presumptive taxation (Section 44ADA) added with curated legal docs.

### Session 4
- Implemented structured tax engine tool wrappers in `tax_tools.py`.
- Gemini verifier node added for post-hoc citation checks.
- Enhanced query parsing for hypothetical income scenarios.

### Session 5
- **Form 16 Auto-Propagation**: Form 16 upload now automatically populates session state tax profiles.
- **Tax Optimizer Numeric Safety**: Implemented `_ss_int()` helper to eliminate `StreamlitMixedNumericTypesError`.
- **Diagnostic Logging**: Added `flush=True` across FAISS store, BGE embedder, retriever, and reranker.
- **Sample Test Suite**: Generated 3 realistic Form 16 test PDFs (`sample_form16s/`).
- **Repository Hygiene**: Pushed cleaned codebase to `https://github.com/HARRY5D/TAX-RAG.git` with `.env` and pycache exclusions.
- **Agentic Architecture Blueprint**: Updated `architecture.md` outlining the Gemini Agent Tool-Calling transition plan.
