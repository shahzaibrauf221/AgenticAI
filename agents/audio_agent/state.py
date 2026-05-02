# ============================================================
# state.py
# Phase 2 — Audio Agent LangGraph state.
# ============================================================
#
# Only audio-related fields are tracked here. Video state lives in
# Phase 3 (agents/video_agent/state.py). The two phases share Phase 1's
# script.json + character_db.json as input, but produce independent
# outputs that Phase 3 (or the orchestrator) can combine.
# ============================================================

from typing import TypedDict, Annotated
import operator


# ── Reducers for parallel branches ───────────────────────────

def _merge_dict(left: dict, right: dict) -> dict:
    """Merge two dicts by key (right wins on conflict)."""
    out = dict(left or {})
    out.update(right or {})
    return out


def _last_value(left, right):
    """Take the right (most recent) value when parallel branches write the same key."""
    return right if right is not None else left


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────
    scene_manifest_path:  str         # Phase 1 output (legacy)
    character_db_path:    str         # Phase 1 output (legacy)
    script_path:          str         # Phase 1 unified output (preferred)

    # ── Loaded Phase 1 outputs ────────────────────────────
    scene_manifest:       dict
    character_db:         dict

    # ── Task graph ────────────────────────────────────────
    task_graph:           dict
    scenes:               list

    # ── Per-scene audio outputs (keyed by scene_id) ───────
    # Parallel nodes write here — dict-merge reducer prevents clobber.
    audio_tracks:         Annotated[dict, _merge_dict]

    # ── Pipeline control ──────────────────────────────────
    status:               Annotated[str, _last_value]
    current_node:         Annotated[str, _last_value]
    errors:               Annotated[list, operator.add]

    # ── Fault tolerance ───────────────────────────────────
    resume_from_memory:   bool
    completed_scene_ids:  list

    # ── Internal routing (Send() per-scene payload) ───────
    _current_scene:       Annotated[dict, _last_value]
