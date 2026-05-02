# ============================================================
# graph.py
# LangGraph StateGraph — Writer's Room Workflow
# ============================================================
#
# Flow:
#   START
#     └─► mode_selector_node
#               ├─ "manual" ──► validator_node
#               │                    ├─ invalid ──► END (failed)
#               │                    └─ valid   ──► hitl_node
#               └─ "auto"   ──► scriptwriter_node
#                                    └──────────────► hitl_node
#                                                         ├─ rejected ──► END (failed)
#                                                         └─ approved ──► character_node
#                                                                              └──► image_node
#                                                                                       └──► memory_commit_node
#                                                                                                   └──► END
# ============================================================

from langgraph.graph import StateGraph, START, END

from agents.story_agent.state import AgentState
from agents.story_agent.nodes import (
    mode_selector_node,
    scriptwriter_node,
    validator_node,
    hitl_node,
    character_node,
    image_node,
    memory_commit_node,
)


# ─── Routing Functions ────────────────────────────────────────────────────────

def route_after_mode_selector(state: AgentState) -> str:
    """Route to scriptwriter (auto) or validator (manual)."""
    if state.get("input_mode") == "manual":
        return "validator_node"
    return "scriptwriter_node"


def route_after_validator(state: AgentState) -> str:
    """Reject invalid scripts; valid ones go to HITL."""
    if state.get("script_valid"):
        return "hitl_node"
    return END


def route_after_hitl(state: AgentState) -> str:
    """If human approved → continue; else → END."""
    if state.get("hitl_approved"):
        return "character_node"
    return END


def route_after_status(state: AgentState) -> str:
    """Generic failure check — abort if status is 'failed'."""
    if state.get("status") == "failed":
        return END
    return "image_node"


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the full Writer's Room LangGraph workflow.
    Returns a compiled CompiledStateGraph ready for .invoke() / .ainvoke().
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────
    builder.add_node("mode_selector_node",  mode_selector_node)
    builder.add_node("scriptwriter_node",   scriptwriter_node)
    builder.add_node("validator_node",      validator_node)
    builder.add_node("hitl_node",           hitl_node)
    builder.add_node("character_node",      character_node)
    builder.add_node("image_node",          image_node)
    builder.add_node("memory_commit_node",  memory_commit_node)

    # ── Entry ─────────────────────────────────────────────
    builder.add_edge(START, "mode_selector_node")

    # ── Mode selector branches ────────────────────────────
    builder.add_conditional_edges(
        "mode_selector_node",
        route_after_mode_selector,
        {
            "validator_node":   "validator_node",
            "scriptwriter_node": "scriptwriter_node",
        },
    )

    # ── Both paths converge at HITL ───────────────────────
    builder.add_edge("scriptwriter_node", "hitl_node")

    builder.add_conditional_edges(
        "validator_node",
        route_after_validator,
        {"hitl_node": "hitl_node", END: END},
    )

    # ── HITL gate ─────────────────────────────────────────
    builder.add_conditional_edges(
        "hitl_node",
        route_after_hitl,
        {"character_node": "character_node", END: END},
    )

    # ── Character → Image ─────────────────────────────────
    builder.add_conditional_edges(
        "character_node",
        route_after_status,
        {"image_node": "image_node", END: END},
    )

    # ── Image → Memory commit → END ───────────────────────
    builder.add_edge("image_node",          "memory_commit_node")
    builder.add_edge("memory_commit_node",  END)

    return builder.compile()
