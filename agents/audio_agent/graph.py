# ============================================================
# graph.py — Phase 2 LangGraph (audio only)
# ============================================================
#
# Topology:
#
#   START
#     │
#     ▼
#   scene_parser_node
#     │
#     └─► Send(voice_synth_node) × N    (parallel per scene)
#                       │
#                       ▼
#               audio_finalizer_node
#                       │
#                       ▼
#                      END
#
# Phase 3 (video) consumes Phase 2's outputs (scene_NN_full.wav files +
# the same script.json from Phase 1) — see agents/video_agent/graph.py.
# ============================================================

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents.audio_agent.state import AgentState
from agents.audio_agent.nodes import (
    scene_parser_node,
    voice_synth_node,
    audio_finalizer_node,
)


# ─── Fan-out ─────────────────────────────────────────────────────────────────

def fanout_audio(state: AgentState):
    """Launch one voice_synth per scene in parallel."""
    scenes    = state.get("scenes", []) or []
    completed = set(state.get("completed_scene_ids") or [])
    sends     = []

    for scene in scenes:
        sid = scene.get("scene_id")
        if sid in completed:
            continue
        sends.append(Send("voice_synth_node", {**state, "_current_scene": scene}))

    if not sends:
        return [Send("audio_finalizer_node", state)]
    return sends


# ─── Build Graph ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("scene_parser_node",    scene_parser_node)
    builder.add_node("voice_synth_node",     voice_synth_node)
    builder.add_node("audio_finalizer_node", audio_finalizer_node)

    builder.add_edge(START, "scene_parser_node")

    builder.add_conditional_edges(
        "scene_parser_node",
        fanout_audio,
        ["voice_synth_node", "audio_finalizer_node"],
    )

    builder.add_edge("voice_synth_node",     "audio_finalizer_node")
    builder.add_edge("audio_finalizer_node", END)

    return builder.compile()
