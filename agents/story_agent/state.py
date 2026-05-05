# ============================================================
# state.py
# Shared LangGraph AgentState for the Writer's Room system
# ============================================================

from typing import TypedDict, Annotated, Literal
import operator


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    input_mode:   Literal["manual", "auto"]   # "manual" = user uploads script, "auto" = prompt
    user_input:   str                          # raw prompt OR raw script text

    # ── Script ─────────────────────────────────────────────
    script:       dict                         # parsed / generated script dict
    script_valid: bool                         # set by validator
    validation_errors:   list[str]             # structural errors
    validation_warnings: list[str]             # non-fatal warnings

    # ── Human-in-the-loop ──────────────────────────────────
    hitl_approved: bool                        # True once human approves
    hitl_feedback: str                         # optional human feedback

    # ── Characters & Images ────────────────────────────────
    characters:   list[dict]                   # character profiles
    images:       Annotated[list, operator.add] # accumulated character image results
    scene_images: Annotated[list, operator.add] # accumulated scene background image results

    # ── Pipeline control ───────────────────────────────────
    status:       str                          # "processing" | "awaiting_review" | "complete" | "failed"
    errors:       Annotated[list, operator.add] # accumulated pipeline errors
    current_node: str                          # last node that ran (for logging)