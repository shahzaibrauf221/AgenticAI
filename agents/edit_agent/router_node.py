from __future__ import annotations

from agents.edit_agent.state import EditAgentState


def route_edit_execution(state: EditAgentState) -> EditAgentState:
    target = state.get("target", "video")
    intent = state.get("intent", "recompose_video")

    # Conditional re-entry points:
    # - script edits re-enter Phase 1
    # - audio edits re-enter Phase 2
    # - video/video_frame edits re-enter Phase 3 (or apply local frame filters)
    rerun_from = "phase3"
    rerun_pipeline = False

    if target == "script":
        rerun_from = "phase1"
        rerun_pipeline = True
    elif target == "audio":
        rerun_from = "phase2"
        rerun_pipeline = True
    elif target in ("video", "video_frame"):
        rerun_from = "phase3"
        rerun_pipeline = intent in ("recompose_video", "add_subtitle", "remove_subtitle")

    return {
        **state,
        "rerun_from": rerun_from,
        "rerun_pipeline": rerun_pipeline,
        "logs": [
            f"[router] target={target} intent={intent} rerun_from={rerun_from} rerun_pipeline={rerun_pipeline}"
        ],
    }

