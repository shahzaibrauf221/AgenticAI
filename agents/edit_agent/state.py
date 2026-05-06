# ============================================================
# agents/edit_agent/state.py
# LangGraph AgentState for the Phase 5 Edit Agent
# ============================================================

from __future__ import annotations
from typing import TypedDict, Annotated, Optional
import operator


class EditAgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    raw_query:   str           # free-text edit command from user
    current_version: str      # state version being edited (e.g. "v2")

    # ── Classified intent ──────────────────────────────────
    intent:      str           # e.g. "change_voice_tone"
    target:      str           # "audio" | "video_frame" | "video" | "script"
    scope:       str           # e.g. "character:Narrator" | "scene:2" | "all"
    parameters:  dict          # arbitrary key/value params for the edit

    # ── Execution ──────────────────────────────────────────
    plan:        list[str]     # steps the executor will take
    result:      dict          # output from executor
    new_version: str           # version tag after snapshot
    rerun_from: str            # phase1 | phase2 | phase3
    rerun_pipeline: bool       # whether to re-enter orchestrator pipeline

    # ── Pipeline control ───────────────────────────────────
    status:      str           # "classifying" | "planning" | "executing" | "done" | "failed"
    errors:      Annotated[list[str], operator.add]
    logs:        Annotated[list[str], operator.add]
