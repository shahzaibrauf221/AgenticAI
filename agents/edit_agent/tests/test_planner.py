# ============================================================
# tests/test_planner.py
# Phase 5 — Planner node unit tests
# ============================================================

import asyncio
import pytest
from agents.edit_agent.planner import plan_edit


def _base_state(**kwargs):
    return {
        "raw_query":       "test query",
        "current_version": "v1",
        "intent":          "recompose_video",
        "target":          "video",
        "scope":           "all",
        "parameters":      {},
        "plan":            [],
        "result":          {},
        "new_version":     "",
        "status":          "planning",
        "errors":          [],
        "logs":            [],
        **kwargs,
    }


@pytest.mark.asyncio
async def test_audio_plan():
    state = _base_state(target="audio", intent="change_voice_tone")
    out   = await plan_edit(state)
    assert "apply_audio_edit"      in out["plan"]
    assert "snapshot_state"        in out["plan"]
    assert "load_current_state"    in out["plan"]
    assert out["status"]           == "executing"


@pytest.mark.asyncio
async def test_video_frame_plan():
    state = _base_state(target="video_frame", intent="make_scene_darker")
    out   = await plan_edit(state)
    assert "apply_image_edit"   in out["plan"]
    assert "snapshot_state"     in out["plan"]


@pytest.mark.asyncio
async def test_video_plan():
    state = _base_state(target="video", intent="remove_subtitle")
    out   = await plan_edit(state)
    assert "recompose_final_video" in out["plan"]
    assert "snapshot_state"        in out["plan"]


@pytest.mark.asyncio
async def test_script_plan():
    state = _base_state(target="script", intent="regenerate_script")
    out   = await plan_edit(state)
    assert "cascade_to_audio" in out["plan"]
    assert "cascade_to_video" in out["plan"]
    assert "snapshot_state"   in out["plan"]


@pytest.mark.asyncio
async def test_unknown_target_falls_back_to_video():
    state = _base_state(target="unknown_target", intent="something")
    out   = await plan_edit(state)
    assert "recompose_final_video" in out["plan"]


@pytest.mark.asyncio
async def test_plan_always_has_snapshot_step():
    for target in ("audio", "video_frame", "video", "script"):
        state = _base_state(target=target)
        out   = await plan_edit(state)
        assert "snapshot_state" in out["plan"], f"Missing snapshot_state for target={target}"
