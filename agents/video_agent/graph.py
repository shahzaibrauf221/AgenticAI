# ============================================================
# graph.py — Phase 3 LangGraph (video pipeline)
# ============================================================
#
# Topology:
#
#   START
#     │
#     ▼
#   scene_loader_node
#     │
#     ▼
#   video_gen_node (loops until all scenes generated)
#     │
#     ▼
#   compositor_node
#     │
#     ▼
#   finalizer_node
#     │
#     ▼
#    END
#
# Note: Phase 3 expects audio_tracks to already be in state (loaded
# from MCP memory by scene_loader_node, or passed by the orchestrator).
# ============================================================

from langgraph.graph import StateGraph, START, END
from agents.video_agent.state import AgentState
from agents.video_agent.nodes import (
    scene_loader_node,
    video_gen_node,
    compositor_node,
    finalizer_node,
)


# ─── Fan-outs ────────────────────────────────────────────────────────────────

def route_after_loader(state: AgentState):
    """Start sequential video generation."""
    return "video_gen_node"


def route_after_video_gen(state: AgentState):
    """Loop video generation until all scenes complete, then compose."""
    scenes = state.get("scenes", []) or []
    clips  = state.get("video_clips") or {}
    if len(clips) < len(scenes):
        return "video_gen_node"
    return "compositor_node"


# ─── Build Graph ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("scene_loader_node", scene_loader_node)
    builder.add_node("video_gen_node",    video_gen_node)
    builder.add_node("compositor_node",   compositor_node)
    builder.add_node("finalizer_node",    finalizer_node)

    builder.add_edge(START, "scene_loader_node")
    builder.add_edge("scene_loader_node", "video_gen_node")

    # Sequential video_gen feeds back to itself until all scenes are done,
    # then hands off directly to compositor (Seedance-only mode).
    builder.add_conditional_edges(
        "video_gen_node",
        route_after_video_gen,
        ["video_gen_node", "compositor_node"]
    )
    builder.add_edge("compositor_node", "finalizer_node")
    builder.add_edge("finalizer_node",  END)

    return builder.compile()
