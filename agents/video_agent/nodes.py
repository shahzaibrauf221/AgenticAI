# ============================================================
# nodes.py — Phase 3 LangGraph node implementations (video only)
#
# Contains:
#   • scene_loader_node     — Loads Phase 1 script + Phase 2 audio results
#   • video_gen_node        — Generates base video per scene
#   • face_swap_node        — Applies character face-swap
#   • lip_sync_node         — Syncs lips to Phase 2 audio (mixed dialogue+BGM)
#   • compositor_node       — Concatenates all final scenes into final_output.mp4
#   • finalizer_node        — Writes summary, marks pipeline complete
#
# All MCP tool calls go to studio_floor_server (port 8200) — same server
# as Phase 2 but using a different subset of tools.
# ============================================================

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.video_agent.state import AgentState

# ─── MCP client ──────────────────────────────────────────────────────────────

MCP_CONFIG = {
    "studio_floor": {
        "url":       "http://localhost:8200/mcp",
        "transport": "streamable_http",
        "httpx_client_factory": None,
    }
}

import httpx as _httpx
_DEFAULT_TIMEOUT = _httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=30.0)


def _custom_httpx_factory(headers=None, timeout=None, auth=None):
    return _httpx.AsyncClient(
        headers=headers, timeout=_DEFAULT_TIMEOUT, auth=auth, follow_redirects=True,
    )

MCP_CONFIG["studio_floor"]["httpx_client_factory"] = _custom_httpx_factory

_tool_map: dict = {}


async def _get_tools() -> dict:
    global _tool_map
    if not _tool_map:
        client = MultiServerMCPClient(MCP_CONFIG)
        tools  = await client.get_tools(server_name="studio_floor")
        _tool_map = {t.name: t for t in tools}
    return _tool_map


def _extract_text(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return _extract_text(result.content)
    if isinstance(result, list):
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
            if isinstance(block, str):
                return block
        return json.dumps(result)
    if isinstance(result, dict) and result.get("type") == "text" and "text" in result:
        return result["text"]
    return json.dumps(result)


async def _call_tool(tool_name: str, **kwargs) -> str:
    tmap = await _get_tools()
    if tool_name not in tmap:
        raise ValueError(f"Tool '{tool_name}' not found. Available: {list(tmap.keys())}")
    result = await tmap[tool_name].ainvoke(kwargs)
    return _extract_text(result)


def _safe_parse(text: str):
    import re
    clean = re.sub(r"```json|```", "", text).strip()
    return json.loads(clean)


def _safe_filename_part(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).lower()


def _expected_base_video_path(scene_id: int, location: str) -> Path:
    """Mirror studio_floor_server's output convention."""
    base_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "video"
    safe_loc = _safe_filename_part(location[:40]) if location else f"scene_{scene_id}"
    return base_dir / f"scene_{scene_id:02d}_{safe_loc}.mp4"


async def _wait_for_file(path: Path, timeout_s: float = 120.0, poll_s: float = 0.5) -> bool:
    elapsed = 0.0
    while elapsed < timeout_s:
        if path.exists() and path.stat().st_size > 0:
            return True
        await asyncio.sleep(poll_s)
        elapsed += poll_s
    return False


# ─── Node 1: Scene Loader ─────────────────────────────────────────────────────

async def scene_loader_node(state: AgentState) -> dict:
    """
    Loads Phase 1's script.json (preferred) or scene_manifest.json (fallback)
    AND pulls Phase 2's audio_tracks from MCP memory if not already in state.
    """
    print("\n[Scene Loader] Loading Phase 1 + Phase 2 outputs...")

    manifest_path = Path(state.get("scene_manifest_path", ""))
    char_db_path  = Path(state.get("character_db_path", ""))
    script_path   = (Path(state.get("script_path", ""))
                     if state.get("script_path")
                     else (manifest_path.parent / "script.json"))

    manifest = None
    char_db  = None

    # Prefer unified script.json
    if script_path.exists():
        try:
            unified = json.loads(script_path.read_text(encoding="utf-8"))
            manifest = {
                "title":  unified.get("story", {}).get("title", "Untitled"),
                "genre":  unified.get("story", {}).get("genre", ""),
                "scenes": unified.get("scenes", []),
            }
            char_db = {"characters": unified.get("characters", [])}
            print(f"[Scene Loader] Source: script.json")
        except Exception as e:
            print(f"[Scene Loader] script.json unreadable: {e}")

    if manifest is None:
        if not manifest_path.exists():
            return {
                "status":       "failed",
                "errors":       [f"No Phase 1 input found at {manifest_path}"],
                "current_node": "scene_loader_node",
            }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        char_db  = (json.loads(char_db_path.read_text(encoding="utf-8"))
                    if char_db_path.exists() else {"characters": []})
        print(f"[Scene Loader] Source: scene_manifest.json (fallback)")

    scenes = manifest.get("scenes", [])
    print(f"[Scene Loader] {len(scenes)} scenes, {len(char_db.get('characters', []))} characters")

    # Pull Phase 2 audio results from MCP memory if not in state
    audio_tracks = dict(state.get("audio_tracks") or {})
    if not audio_tracks:
        print(f"[Scene Loader] Loading Phase 2 audio from MCP memory...")
        try:
            mem_raw = await _call_tool("query_memory", category="audio_track", limit=100)
            mem = _safe_parse(mem_raw)
            for entry in mem.get("entries", []):
                try:
                    payload = json.loads(entry.get("value", "{}"))
                    sid = payload.get("scene_id")
                    if sid is not None:
                        audio_tracks[str(sid)] = payload
                except Exception:
                    continue
            print(f"[Scene Loader] Loaded audio for {len(audio_tracks)} scene(s) from memory")
        except Exception as e:
            print(f"[Scene Loader] Could not load audio from memory: {e}")

    # Build task graph
    raw = await _call_tool("get_task_graph", scene_manifest_json=json.dumps(manifest))
    parsed = _safe_parse(raw)
    task_graph = parsed.get("task_graph", {})

    # Resume support
    completed_ids: list = []
    if state.get("resume_from_memory"):
        try:
            mem_raw = await _call_tool("query_memory", category="final_scene", limit=100)
            mem = _safe_parse(mem_raw)
            for entry in mem.get("entries", []):
                try:
                    payload = json.loads(entry.get("value", "{}"))
                    sid = payload.get("scene_id")
                    if sid is not None and Path(payload.get("file", "")).exists():
                        completed_ids.append(sid)
                except Exception:
                    continue
            if completed_ids:
                print(f"[Scene Loader] Resuming — {len(completed_ids)} done: {completed_ids}")
        except Exception:
            pass

    return {
        "scene_manifest":      manifest,
        "character_db":        char_db,
        "audio_tracks":        audio_tracks,
        "task_graph":          task_graph,
        "scenes":              scenes,
        "completed_scene_ids": completed_ids,
        "status":              "processing",
        "current_node":        "scene_loader_node",
    }


# ─── Node 2: Video Gen ────────────────────────────────────────────────────────

async def video_gen_node(state: AgentState) -> dict:
    """PDF §5.3 — generate base video for one scene."""
    scene = state.get("_current_scene", {})
    sid   = scene.get("scene_id")
    chars = scene.get("characters", [])

    duration = max(3.0, sum(max(1, len(d.get("line", "").split())) / 2.5
                            for d in scene.get("dialogue", []) if isinstance(d, dict)))

    print(f"  [Video Gen    | scene {sid}] Rendering {duration:.1f}s base video...")

    raw = await _call_tool(
        "query_stock_footage",
        scene_id=sid, location=scene.get("location", ""),
        characters=chars, visual_cue=scene.get("scene_visual_cue", ""),
        duration_s=duration,
    )
    base = _safe_parse(raw)
    base_video = base.get("file", "")

    payload = {"scene_id": sid, "base_video": base_video,
               "duration_s": duration, "characters": chars}

    await _call_tool(
        "commit_memory",
        key=f"video_scene_{sid:02d}" if isinstance(sid, int) else f"video_scene_{sid}",
        value=json.dumps(payload),
        category="video_clip",
    )

    print(f"  [Video Gen    | scene {sid}] ✓ {base_video}")
    return {"video_clips": {str(sid): payload}}


# ─── Node 3: Face Swap ────────────────────────────────────────────────────────

async def face_swap_node(state: AgentState) -> dict:
    """PDF §5.4 — validate identity, then composite character face onto base video."""
    scene   = state.get("_current_scene", {})
    char_db = state.get("character_db", {})
    sid     = scene.get("scene_id")
    chars   = scene.get("characters", [])
    location = scene.get("location", "")
    primary = chars[0] if chars else "Unknown"
    skey    = str(sid)

    print(f"  [Face Swap    | scene {sid}] Validating identity for '{primary}'...")
    raw = await _call_tool(
        "identity_validator",
        character_name=primary, character_db_json=json.dumps(char_db),
    )
    validation = _safe_parse(raw)

    if not validation.get("valid"):
        print(f"  [Face Swap    | scene {sid}] ✗ Identity FAILED: {validation.get('reason')}")
        expected = _expected_base_video_path(sid, location)
        await _wait_for_file(expected, timeout_s=120.0)
        payload = {
            "scene_id": sid, "base_video": str(expected),
            "face_swapped_file": str(expected),
            "primary_character": primary, "identity_verified": False,
            "skipped_reason": validation.get("reason"),
        }
        await _call_tool("commit_memory",
                         key=f"faceswap_scene_{sid:02d}" if isinstance(sid, int) else f"faceswap_scene_{sid}",
                         value=json.dumps(payload), category="face_swap")
        return {"face_swapped_clips": {skey: payload}}

    expected = _expected_base_video_path(sid, location)
    print(f"  [Face Swap    | scene {sid}] ✓ Identity verified — waiting for base video...")
    if not await _wait_for_file(expected, timeout_s=120.0):
        print(f"  [Face Swap    | scene {sid}] ✗ Base video never appeared at {expected}")
        return {"face_swapped_clips": {skey: {
            "scene_id": sid, "status": "failed",
            "error": f"Base video not found at {expected}",
            "primary_character": primary, "identity_verified": True,
            "face_swapped_file": "",
        }}}

    base_video = str(expected)
    print(f"  [Face Swap    | scene {sid}] Compositing '{primary}' onto frames...")

    raw = await _call_tool(
        "face_swapper",
        scene_id=sid, base_video_path=base_video,
        character_name=primary, reference_path=validation.get("reference_path", ""),
    )
    swap = _safe_parse(raw)
    swapped_file = swap.get("file", "")

    payload = {
        "scene_id": sid, "base_video": base_video,
        "face_swapped_file": swapped_file, "primary_character": primary,
        "identity_verified": True,
        "frames_processed": swap.get("frames_processed", 0),
        "used_real_portrait": swap.get("used_real_portrait", False),
    }

    await _call_tool(
        "commit_memory",
        key=f"faceswap_scene_{sid:02d}" if isinstance(sid, int) else f"faceswap_scene_{sid}",
        value=json.dumps(payload), category="face_swap",
    )

    print(f"  [Face Swap    | scene {sid}] ✓ {swap.get('frames_processed', 0)} frame(s) swapped")
    return {"face_swapped_clips": {skey: payload}}


# ─── Node 4: Lip Sync ─────────────────────────────────────────────────────────

async def lip_sync_node(state: AgentState) -> dict:
    """PDF §5.5 — align Phase 2's mixed audio to face-swapped video."""
    scene = state.get("_current_scene", {})
    sid   = scene.get("scene_id")
    skey  = str(sid)

    audio_entry = (state.get("audio_tracks") or {}).get(skey, {})
    video_entry = (state.get("face_swapped_clips") or {}).get(skey, {})

    audio_files = audio_entry.get("files", [])
    video_file  = video_entry.get("face_swapped_file", "")

    if not video_file:
        print(f"  [Lip Sync     | scene {sid}] ⚠ No video — skipping")
        return {
            "final_scenes": {skey: {"scene_id": sid, "status": "skipped",
                                    "reason": "no video"}},
        }

    if not audio_files:
        print(f"  [Lip Sync     | scene {sid}] ⚠ No audio — skipping")
        return {
            "final_scenes": {skey: {"scene_id": sid, "status": "skipped",
                                    "reason": "no audio"}},
        }

    print(f"  [Lip Sync     | scene {sid}] Aligning audio + video...")

    raw = await _call_tool(
        "lip_sync_aligner",
        scene_id=sid, video_path=video_file, audio_files=audio_files,
    )
    try:
        result = _safe_parse(raw)
    except Exception as e:
        result = {"status": "error", "error": str(e), "scene_id": sid}

    await _call_tool(
        "commit_memory",
        key=f"final_scene_{sid:02d}" if isinstance(sid, int) else f"final_scene_{sid}",
        value=json.dumps(result), category="final_scene",
    )

    score = result.get("lip_sync_score", 0)
    dur   = result.get("duration_s", 0)
    print(f"  [Lip Sync     | scene {sid}] ✓ {result.get('file', 'N/A')}  "
          f"(score={score}, {dur}s)")

    return {"final_scenes": {skey: result}}


# ─── Node 5: Compositor ───────────────────────────────────────────────────────

def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


async def compositor_node(state: AgentState) -> dict:
    """
    PDF §4 Phase 3 final task — concatenate per-scene MP4s into final_output.mp4.

    Uses ffmpeg's concat demuxer when available; otherwise reports the list
    of per-scene files so the user can stitch manually.
    """
    print(f"\n[Compositor] Stitching final scenes into final_output.mp4...")

    finals = state.get("final_scenes", {}) or {}
    scenes = state.get("scenes", []) or []

    # Order scenes by scene_id
    ordered_scenes = sorted(
        [s for s in scenes if isinstance(s, dict)],
        key=lambda s: s.get("scene_id", 0),
    )

    # Collect per-scene final MP4 paths in scene order
    scene_files = []
    for s in ordered_scenes:
        sid = str(s.get("scene_id"))
        info = finals.get(sid, {})
        f = info.get("file", "")
        if f and Path(f).exists() and Path(f).stat().st_size > 0:
            scene_files.append(f)
        else:
            print(f"  [Compositor] ⚠ scene {sid}: no usable final file")

    if not scene_files:
        return {
            "status":       "failed",
            "errors":       ["Compositor: no scene files to stitch"],
            "current_node": "compositor_node",
        }

    out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "final_output.mp4"

    if not _has_ffmpeg():
        print(f"  [Compositor] ✗ ffmpeg not available — cannot concatenate")
        print(f"  [Compositor]   scene files would have been:")
        for f in scene_files:
            print(f"      • {f}")
        return {
            "status":       "complete",
            "current_node": "compositor_node",
            "errors":       ["ffmpeg not available — final_output.mp4 not generated"],
        }

    # Build concat list file
    concat_list = out_dir / "_concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{Path(f).resolve()}'" for f in scene_files) + "\n",
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(final_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            # Fallback: re-encode if stream copy fails (codec mismatch)
            print(f"  [Compositor] copy failed, re-encoding...")
            cmd_re = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                str(final_path),
            ]
            proc = subprocess.run(cmd_re, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                return {
                    "status":       "failed",
                    "current_node": "compositor_node",
                    "errors":       [f"ffmpeg concat failed: {proc.stderr[-500:]}"],
                }
    except subprocess.TimeoutExpired:
        return {
            "status":       "failed",
            "current_node": "compositor_node",
            "errors":       ["ffmpeg concat timed out"],
        }
    finally:
        try:
            concat_list.unlink()
        except Exception:
            pass

    print(f"  [Compositor] ✓ {final_path}  ({final_path.stat().st_size / (1024*1024):.1f} MB)")

    await _call_tool(
        "commit_memory", key="final_video",
        value=json.dumps({"file": str(final_path),
                          "scenes_concatenated": len(scene_files)}),
        category="summary",
    )

    return {
        "final_video":  str(final_path),
        "current_node": "compositor_node",
    }


# ─── Node 6: Finalizer ────────────────────────────────────────────────────────

async def finalizer_node(state: AgentState) -> dict:
    """Write run summary log."""
    finals = state.get("final_scenes", {})
    audios = state.get("audio_tracks", {})
    videos = state.get("face_swapped_clips", {})

    summary = {
        "title":             (state.get("scene_manifest") or {}).get("title", "Untitled"),
        "scenes_total":      len(state.get("scenes") or []),
        "audio_done":        len(audios),
        "video_done":        len(videos),
        "final_mp4s":        len(finals),
        "final_video":       state.get("final_video", ""),
        "scene_files":       {sid: v.get("file") for sid, v in finals.items()},
        "lip_sync_scores":   {sid: v.get("lip_sync_score") for sid, v in finals.items()},
        "identity_verified": {sid: v.get("identity_verified") for sid, v in videos.items()},
    }

    logs_dir = Path(__file__).resolve().parent.parent.parent / "data" / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / "phase3_video_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Finalizer] Run summary → {summary_path}")

    await _call_tool(
        "commit_memory", key="phase3_video_summary",
        value=json.dumps(summary), category="summary",
    )

    return {
        "status":       "complete",
        "current_node": "finalizer_node",
    }
