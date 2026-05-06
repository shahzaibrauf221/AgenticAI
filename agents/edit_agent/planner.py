# ============================================================
# agents/edit_agent/planner.py
# Phase 5 — Edit Planner node
#
# Receives classified intent → produces an ordered list of
# execution steps that the executor will carry out.
# ============================================================

from __future__ import annotations

from agents.edit_agent.state import EditAgentState


# ─── Plan templates keyed by target ──────────────────────────────────────────

_PLANS: dict[str, list[str]] = {
    "audio": [
        "load_current_state",
        "identify_audio_segments",
        "apply_audio_edit",
        "rebuild_timing_manifest",
        "snapshot_state",
    ],
    "video_frame": [
        "load_current_state",
        "identify_target_scenes",
        "apply_image_edit",
        "snapshot_state",
    ],
    "video": [
        "load_current_state",
        "apply_video_edit",
        "recompose_final_video",
        "snapshot_state",
    ],
    "script": [
        "load_current_state",
        "apply_script_edit",
        "rerun_pipeline_from_phase1_if_requested",
        "snapshot_state",
    ],
}


async def plan_edit(state: EditAgentState) -> EditAgentState:
    """
    LangGraph node: map the classified target → ordered execution plan.
    """
    target = state.get("target", "video")
    intent = state.get("intent", "recompose_video")
    scope  = state.get("scope",  "all")

    plan = _PLANS.get(target, _PLANS["video"]).copy()

    log = (
        f"[planner] intent={intent}, target={target}, scope={scope} "
        f"→ plan={plan}"
    )

    return {
        **state,
        "plan":   plan,
        "status": "executing",
        "logs":   [log],
    }
