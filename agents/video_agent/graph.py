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
#     ├─► Send(video_gen_node)  × N    ┐  parallel per scene
#     └─► Send(face_swap_node)  × N    ┘  (face_swap awaits video_gen via
#                                          file polling on shared paths)
#                       │
#                       ▼
#                  av_barrier_node
#                       │
#                       └─► Send(lip_sync_node) × N
#                                  │
#                                  ▼
#                            compositor_node
#                                  │
#                                  ▼
#                            finalizer_node
#                                  │
#                                  ▼
#                                 END
#
# Note: Phase 3 expects audio_tracks to already be in state (loaded
# from MCP memory by scene_loader_node, or passed by the orchestrator).
# ============================================================

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents.video_agent.state import AgentState
from agents.video_agent.nodes import (
    scene_loader_node,
    video_gen_node,
    face_swap_node,
    lip_sync_node,
    compositor_node,
    finalizer_node,
)


# ─── Fan-outs ────────────────────────────────────────────────────────────────

def fanout_video(state: AgentState):
    """Launch video_gen + face_swap per scene in parallel."""
    scenes    = state.get("scenes", []) or []
    completed = set(state.get("completed_scene_ids") or [])
    sends     = []

    for scene in scenes:
        sid = scene.get("scene_id")
        if sid in completed:
            continue
        sends.append(Send("video_gen_node", {**state, "_current_scene": scene}))
        sends.append(Send("face_swap_node", {**state, "_current_scene": scene}))

    if not sends:
        return [Send("compositor_node", state)]
    return sends


def fanout_lipsync(state: AgentState):
    """After barrier, fan out lip_sync per scene that has audio + face-swapped video."""
    scenes    = state.get("scenes", []) or []
    audios    = state.get("audio_tracks") or {}
    videos    = state.get("face_swapped_clips") or {}
    completed = set(state.get("completed_scene_ids") or [])

    sends = []
    for scene in scenes:
        sid  = scene.get("scene_id")
        skey = str(sid)
        if sid in completed:
            continue
        if skey in audios and skey in videos:
            sends.append(Send("lip_sync_node", {**state, "_current_scene": scene}))

    if not sends:
        return [Send("compositor_node", state)]
    return sends


# ─── Barrier ─────────────────────────────────────────────────────────────────

async def av_barrier_node(state: AgentState) -> dict:
    """Sync point — LangGraph waits for all fanned-out branches before this runs."""
    audios = state.get("audio_tracks") or {}
    videos = state.get("face_swapped_clips") or {}
    print(f"\n[Barrier] Audio: {len(audios)}  |  Face-swapped video: {len(videos)}")
    return {"current_node": "av_barrier_node"}


# ─── Build Graph ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("scene_loader_node", scene_loader_node)
    builder.add_node("video_gen_node",    video_gen_node)
    builder.add_node("face_swap_node",    face_swap_node)
    builder.add_node("av_barrier_node",   av_barrier_node)
    builder.add_node("lip_sync_node",     lip_sync_node)
    builder.add_node("compositor_node",   compositor_node)
    builder.add_node("finalizer_node",    finalizer_node)

    builder.add_edge(START, "scene_loader_node")

    builder.add_conditional_edges(
        "scene_loader_node",
        fanout_video,
        ["video_gen_node", "face_swap_node", "compositor_node"],
    )

    builder.add_edge("video_gen_node", "av_barrier_node")
    builder.add_edge("face_swap_node", "av_barrier_node")

    builder.add_conditional_edges(
        "av_barrier_node",
        fanout_lipsync,
        ["lip_sync_node", "compositor_node"],
    )

    builder.add_edge("lip_sync_node",   "compositor_node")
    builder.add_edge("compositor_node", "finalizer_node")
    builder.add_edge("finalizer_node",  END)

    return builder.compile()
