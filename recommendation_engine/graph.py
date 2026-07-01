"""
LangGraph Orchestrator — central workflow graph for FinAssist AI.

Full flow:
  User Query
      │
  detect_intent
      │
  ┌───┴──────────────────────────────┐
  │ (if form16 doc uploaded)         │
  form16_parser                      │
      │                              │
  tax_engine ◄────────────────────────┘ (if profile available)
      │
  optimization (if intent = optimization)
      │
  rag_retrieve (if needs retrieval)
      │
  generate_answer
      │
  Final Answer
"""
from langgraph.graph import StateGraph, END
from recommendation_engine.state import GraphState
from recommendation_engine.nodes import (
    detect_intent_node,
    form16_parser_node,
    tax_engine_node,
    optimization_node,
    rag_retrieve_node,
    generate_answer_node,
)


# ─── Conditional routing functions ────────────────────────────────────────────

def should_parse_form16(state: GraphState) -> str:
    if state.needs_form16 and state.form16_path:
        return "parse_form16"
    return "check_calculation"


def should_calculate_tax(state: GraphState) -> str:
    if state.needs_calculation and state.tax_profile:
        return "calculate_tax"
    return "check_optimization"


def should_optimize(state: GraphState) -> str:
    if state.needs_optimization and state.tax_result is not None:
        return "optimize"
    return "retrieve_context"


def should_retrieve(state: GraphState) -> str:
    if state.needs_retrieval:
        return "retrieve"
    return "generate"


def build_finassist_graph() -> StateGraph:
    """
    Construct and compile the FinAssist LangGraph workflow.
    """
    # Use dict-based state for LangGraph compatibility
    workflow = StateGraph(GraphState)

    # ─── Add nodes ─────────────────────────────────────────────────────────────
    workflow.add_node("detect_intent", detect_intent_node)
    workflow.add_node("parse_form16", form16_parser_node)
    workflow.add_node("calculate_tax", tax_engine_node)
    workflow.add_node("optimize", optimization_node)
    workflow.add_node("retrieve_context", rag_retrieve_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # ─── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("detect_intent")

    # ─── Edges with conditional routing ────────────────────────────────────────
    workflow.add_conditional_edges(
        "detect_intent",
        should_parse_form16,
        {
            "parse_form16": "parse_form16",
            "check_calculation": "calculate_tax",
        },
    )

    workflow.add_edge("parse_form16", "calculate_tax")

    workflow.add_conditional_edges(
        "calculate_tax",
        should_optimize,
        {
            "optimize": "optimize",
            "retrieve_context": "retrieve_context",
        },
    )

    workflow.add_edge("optimize", "retrieve_context")

    workflow.add_conditional_edges(
        "retrieve_context",
        should_retrieve,
        {
            "retrieve": "retrieve_context",
            "generate": "generate_answer",
        },
    )

    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ─── Singleton compiled graph ──────────────────────────────────────────────────
_compiled_graph = None


def get_graph():
    """Return the compiled LangGraph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_finassist_graph()
    return _compiled_graph


def run_query(
    user_query: str,
    mode: str = "chat",
    tax_profile: dict = None,
    form16_path: str = None,
    chat_history: list = None,
) -> GraphState:
    """
    Run a user query through the full LangGraph workflow.

    Args:
        user_query: The user's question/request
        mode: "chat" (default) or "document" (Form16 upload)
        tax_profile: Optional dict with income/deduction data
        form16_path: Optional path to Form 16 PDF
        chat_history: Previous conversation turns

    Returns:
        Final GraphState with answer and all intermediate results
    """
    from config.langsmith_config import configure_langsmith
    configure_langsmith()

    initial_state = GraphState(
        user_query=user_query,
        mode=mode,
        tax_profile=tax_profile or {},
        form16_path=form16_path,
        chat_history=chat_history or [],
    )

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    return final_state
