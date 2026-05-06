# ============================================================
# agents/edit_agent/graph.py
# Phase 5 — LangGraph StateGraph orchestration
#
#   classify_intent → plan_edit → execute_edit
# ============================================================

from __future__ import annotations

from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.edit_agent.state           import EditAgentState
from agents.edit_agent.intent_node     import detect_edit_intent
from agents.edit_agent.planner         import plan_edit
from agents.edit_agent.router_node     import route_edit_execution
from agents.edit_agent.executor        import execute_edit


def build_edit_graph() -> StateGraph:
    """Assemble and compile the Phase 5 edit agent graph."""

    graph = StateGraph(EditAgentState)

    # ── Nodes ──────────────────────────────────────────────
    graph.add_node("classify_intent", detect_edit_intent)
    graph.add_node("plan_edit",       plan_edit)
    graph.add_node("route_edit",      route_edit_execution)
    graph.add_node("execute_edit",    execute_edit)

    # ── Edges ──────────────────────────────────────────────
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "plan_edit")
    graph.add_edge("plan_edit",       "route_edit")
    graph.add_edge("route_edit",      "execute_edit")
    graph.add_edge("execute_edit",    END)

    # ── Checkpointer (multi-turn editing memory) ───────────
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        root = Path(__file__).resolve().parent.parent.parent
        ckpt_dir = root / "data" / "state_versions"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "edit_agent_checkpoints.db"
        checkpointer = SqliteSaver.from_conn_string(str(ckpt_path))
    except Exception:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# Module-level compiled graph — import this from other modules
edit_graph = build_edit_graph()
