# ============================================================
# agents/edit_agent/graph.py
# Phase 5 — LangGraph StateGraph orchestration
#
#   classify_intent → plan_edit → execute_edit
# ============================================================

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.edit_agent.state           import EditAgentState
from agents.edit_agent.intent_classifier import classify_intent
from agents.edit_agent.planner         import plan_edit
from agents.edit_agent.executor        import execute_edit


def build_edit_graph() -> StateGraph:
    """Assemble and compile the Phase 5 edit agent graph."""

    graph = StateGraph(EditAgentState)

    # ── Nodes ──────────────────────────────────────────────
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("plan_edit",       plan_edit)
    graph.add_node("execute_edit",    execute_edit)

    # ── Edges ──────────────────────────────────────────────
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "plan_edit")
    graph.add_edge("plan_edit",       "execute_edit")
    graph.add_edge("execute_edit",    END)

    # ── Checkpointer (multi-turn editing memory) ───────────
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Module-level compiled graph — import this from other modules
edit_graph = build_edit_graph()
