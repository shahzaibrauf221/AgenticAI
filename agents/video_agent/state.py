# ============================================================
# state.py
# Phase 3 — Video Agent LangGraph state.
# ============================================================
#
# Phase 3 receives:
#   • Phase 1's script.json (or scene_manifest.json + character_db.json)
#   • Phase 2's audio outputs (file paths in audio_tracks)
#
# Phase 3 produces:
#   • video_clips         — base per-scene videos
#   • face_swapped_clips  — videos with character faces composited in
#   • final_scenes        — lip-synced per-scene final MP4s
#   • final_video         — the single concatenated final_output.mp4
# ============================================================

from typing import TypedDict, Annotated
import operator


# ── Reducers for parallel branches ───────────────────────────

def _merge_dict(left: dict, right: dict) -> dict:
    out = dict(left or {})
    out.update(right or {})
    return out


def _last_value(left, right):
    return right if right is not None else left


class AgentState(TypedDict, total=False):
    # ── Input paths (Phase 1 outputs) ─────────────────────
    scene_manifest_path:  str
    character_db_path:    str
    script_path:          str

    # ── Loaded Phase 1 outputs ────────────────────────────
    scene_manifest:       dict
    character_db:         dict

    # ── Phase 2 input (audio tracks per scene) ────────────
    # If Phase 2 ran in-process, this is populated from its state.
    # If running Phase 3 standalone, it's loaded from MCP memory or the
    # phase2_audio_summary.json file.
    audio_tracks:         Annotated[dict, _merge_dict]
    timing_manifest_path: str
    timing_manifest:      dict
    scene_audio_durations: Annotated[dict, _merge_dict]

    # ── Task graph ────────────────────────────────────────
    task_graph:           dict
    scenes:               list

    # ── Per-scene video outputs ───────────────────────────
    video_clips:          Annotated[dict, _merge_dict]
    face_swapped_clips:   Annotated[dict, _merge_dict]
    final_scenes:         Annotated[dict, _merge_dict]

    # ── Final compositor output ───────────────────────────
    final_video:          Annotated[str, _last_value]

    # ── Pipeline control ──────────────────────────────────
    status:               Annotated[str, _last_value]
    current_node:         Annotated[str, _last_value]
    errors:               Annotated[list, operator.add]
    
    # ── Visual Consistency & State ────────────────────────
    global_seed:          Annotated[int, _last_value]
    last_scene_frame_b64: Annotated[str, _last_value]

    # ── Fault tolerance ───────────────────────────────────
    resume_from_memory:   bool
    completed_scene_ids:  list

    # ── Internal routing ──────────────────────────────────
    _current_scene:       Annotated[dict, _last_value]
