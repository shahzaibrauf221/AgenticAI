# ============================================================
# nodes.py — Phase 3 LangGraph node implementations (video only)
#
# Contains:
#   • scene_loader_node     — Loads Phase 1 script + Phase 2 audio results
#   • video_gen_node        — Generates base video per scene (detailed duration)
#   • face_swap_node        — Applies character face-swap
#   • lip_sync_node         — Syncs lips to Phase 2 audio; falls back to
#                             background image + ambient audio when missing
#   • _make_bg_image_scene  — ffmpeg helper: static image + ambient audio → MP4
#   • _ambient_audio_for_scene — selects/generates ambient sound for a scene
#   • compositor_node       — Concatenates all final scenes into final_output.mp4
#   • finalizer_node        — Writes summary, marks pipeline complete
#
# All MCP tool calls go to studio_floor_server (port 8200).
# ============================================================

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
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


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ─── Duration calculator ─────────────────────────────────────────────────────

# Speech / reading pace constants
_WORDS_PER_SECOND    = 2.3    # typical voiced narration pace
_ACTION_BEAT_SECONDS = 3.0    # pause budget per action/visual beat line
_INTER_LINE_GAP      = 0.6    # reaction pause between dialogue lines
_TRANSITION_PAD      = 1.5    # lead-in + lead-out breathing room per scene
_MIN_SCENE_DURATION  = 6.0    # never shorter than this (seconds)
_MAX_SCENE_DURATION  = 180.0  # hard cap to prevent runaway scenes


def _compute_scene_duration(scene: dict) -> float:
    """
    Estimate a realistic scene duration from ALL available scene fields:

      1. scene["duration_s"]       — explicit override, respected as-is
      2. dialogue lines            — word count / speaking rate + inter-line gap
      3. action / stage directions — fixed beat duration each
      4. scene["description"]      — background narration if no dialogue present
      5. scene["visual_cue"]       — extra pad for each named visual cut/transition
      6. Transition padding        — lead-in + lead-out breathing room

    Returns a float in seconds, clamped to [_MIN_SCENE_DURATION, _MAX_SCENE_DURATION].
    """
    # 1. Explicit override
    explicit = scene.get("duration_s") or scene.get("duration")
    if explicit:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass

    total = 0.0

    # 2. Dialogue lines
    for beat in scene.get("dialogue", []) or []:
        if not isinstance(beat, dict):
            continue
        line  = beat.get("line", "") or beat.get("text", "") or ""
        words = len(line.split())
        if words:
            total += words / _WORDS_PER_SECOND + _INTER_LINE_GAP

    # 3. Stage directions / action beats
    for action in (scene.get("actions", [])
                   or scene.get("action_beats", [])
                   or []):
        if isinstance(action, str) and action.strip():
            total += _ACTION_BEAT_SECONDS
        elif isinstance(action, dict):
            desc = action.get("description", "") or action.get("action", "") or ""
            if desc.strip():
                total += _ACTION_BEAT_SECONDS

    # 4. Scene description as narration (used only when no dialogue)
    if total == 0.0:
        description = (scene.get("description", "")
                       or scene.get("scene_description", "")
                       or scene.get("summary", "")
                       or "")
        words = len(description.split())
        if words:
            total += words / _WORDS_PER_SECOND

    # 5. Visual cue padding
    visual_cue = (scene.get("visual_cue", "")
                  or scene.get("scene_visual_cue", "")
                  or "")
    if visual_cue.strip():
        cuts = max(1, visual_cue.lower().count("cut to")
                   + visual_cue.lower().count("fade")
                   + visual_cue.lower().count("dissolve"))
        total += cuts * 1.0

    # 6. Transition breathing room
    total += _TRANSITION_PAD

    return max(_MIN_SCENE_DURATION, min(_MAX_SCENE_DURATION, total))


# ─── Background image + ambient audio helpers ────────────────────────────────

_GENRE_AMBIENT_MAP = {
    "action":      "action cinematic tense",
    "horror":      "horror dark ambient drone",
    "comedy":      "light upbeat comedy underscore",
    "romance":     "romantic soft piano",
    "drama":       "dramatic strings underscore",
    "thriller":    "thriller suspense low drone",
    "sci-fi":      "sci-fi electronic ambient",
    "fantasy":     "fantasy orchestral ambient",
    "documentary": "neutral documentary ambient",
    "adventure":   "adventure cinematic sweep",
}

_LOCATION_AMBIENT_MAP = {
    "forest":     "forest birdsong nature ambient",
    "city":       "city traffic urban ambient",
    "street":     "street crowd urban ambient",
    "office":     "office interior quiet ambient",
    "house":      "interior home quiet ambient",
    "home":       "interior home quiet ambient",
    "school":     "school hallway ambient",
    "beach":      "ocean waves beach ambient",
    "ocean":      "ocean waves sea ambient",
    "desert":     "desert wind ambient",
    "mountain":   "mountain wind nature ambient",
    "space":      "deep space sci-fi ambient",
    "hospital":   "hospital interior ambient",
    "bar":        "bar crowd indoor ambient",
    "restaurant": "restaurant crowd indoor ambient",
    "car":        "car interior engine ambient",
    "night":      "night crickets dark ambient",
}


def _pick_ambient_keywords(scene: dict, manifest: dict) -> str:
    """
    Derive the best ambient sound description from scene + manifest metadata,
    falling back gracefully through several heuristics.
    """
    # 1. Explicit ambient / mood tag on the scene itself
    for key in ("ambient_sound", "ambient", "mood_sound", "background_music"):
        val = scene.get(key, "")
        if val and isinstance(val, str):
            return val

    # 2. Location keyword matching
    location = (scene.get("location", "") or "").lower()
    for kw, track in _LOCATION_AMBIENT_MAP.items():
        if kw in location:
            return track

    # 3. Genre from manifest
    genre = (manifest.get("genre", "") or "").lower()
    for kw, track in _GENRE_AMBIENT_MAP.items():
        if kw in genre:
            return track

    # 4. Scene mood / tone field
    mood = (scene.get("mood", "") or scene.get("tone", "") or "").lower()
    for kw, track in _GENRE_AMBIENT_MAP.items():
        if kw in mood:
            return track

    # 5. Safe default
    return "neutral cinematic ambient underscore"


async def _fetch_or_generate_ambient(
    scene: dict,
    manifest: dict,
    sid,
    duration_s: float,
) -> str:
    """
    Ask the MCP server for a background ambient audio track.
    Returns the file path (str) or "" on failure.
    """
    keywords = _pick_ambient_keywords(scene, manifest)
    print(f"  [Ambient Audio | scene {sid}] Requesting: '{keywords}' ({duration_s:.1f}s)")
    try:
        raw = await _call_tool(
            "query_ambient_audio",
            scene_id=sid,
            keywords=keywords,
            duration_s=duration_s,
            location=scene.get("location", ""),
        )
        result = _safe_parse(raw)
        path = result.get("file", "")
        if path:
            print(f"  [Ambient Audio | scene {sid}] ✓ {path}")
        else:
            print(f"  [Ambient Audio | scene {sid}] ⚠ MCP returned no file")
        return path
    except Exception as e:
        print(f"  [Ambient Audio | scene {sid}] ✗ Error: {e}")
        return ""


async def _fetch_or_generate_bg_image(
    scene: dict,
    manifest: dict,
    sid,
) -> str:
    """
    Ask the MCP server for a background image matching the scene.
    Returns the file path (str) or "" on failure.
    """
    visual_cue = (scene.get("scene_visual_cue", "")
                  or scene.get("visual_cue", "")
                  or scene.get("description", "")
                  or scene.get("location", "")
                  or "cinematic scene")
    print(f"  [BG Image      | scene {sid}] Requesting: '{visual_cue[:60]}'")
    try:
        raw = await _call_tool(
            "query_stock_image",
            scene_id=sid,
            visual_cue=visual_cue,
            location=scene.get("location", ""),
            mood=scene.get("mood", "") or scene.get("tone", ""),
        )
        result = _safe_parse(raw)
        path = result.get("file", "")
        if path:
            print(f"  [BG Image      | scene {sid}] ✓ {path}")
        else:
            print(f"  [BG Image      | scene {sid}] ⚠ MCP returned no image")
        return path
    except Exception as e:
        print(f"  [BG Image      | scene {sid}] ✗ Error: {e}")
        return ""


def _make_bg_image_scene(
    image_path: str,
    audio_path: str,
    duration_s: float,
    output_path: Path,
) -> bool:
    """
    Compose a static background image + ambient audio into an MP4 using ffmpeg.

    • Video: looped still image → 1920×1080, 24 fps, letterboxed/pillarboxed
    • Audio: ambient track trimmed/padded to duration_s
    • Falls back to black frame if image missing; silent track if audio missing.

    Returns True on success.
    """
    if not _has_ffmpeg():
        print("  [BG Composer] ✗ ffmpeg not available")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_path and Path(image_path).exists():
        video_input = [
            "-loop", "1",
            "-framerate", "24",
            "-i", str(Path(image_path).resolve()),
        ]
    else:
        # Pure black frame
        video_input = [
            "-f", "lavfi",
            "-i", "color=c=black:size=1920x1080:rate=24",
        ]

    if audio_path and Path(audio_path).exists():
        audio_input = ["-i", str(Path(audio_path).resolve())]
        audio_map   = ["-map", "1:a"]
        audio_codec = ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        audio_input = ["-f", "lavfi",
                       "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        audio_map   = ["-map", "1:a"]
        audio_codec = ["-c:a", "aac", "-b:a", "64k"]

    cmd = [
        "ffmpeg", "-y",
        *video_input,
        *audio_input,
        "-map", "0:v",
        *audio_map,
        "-c:v", "libx264", "-preset", "fast", "-tune", "stillimage",
        "-pix_fmt", "yuv420p",
        "-vf", ("scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"),
        *audio_codec,
        "-t", str(duration_s),
        str(output_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            return True
        print(f"  [BG Composer] ✗ ffmpeg error:\n{proc.stderr[-400:]}")
        return False
    except subprocess.TimeoutExpired:
        print("  [BG Composer] ✗ ffmpeg timed out")
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
    print(f"[Scene Loader] {len(scenes)} scenes, "
          f"{len(char_db.get('characters', []))} characters")

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
    """
    Generate base video for one scene.

    Duration is computed from ALL available scene data:
      explicit duration_s  >  dialogue word count  >  action beats
      >  description narration  >  visual cue cuts  >  transition padding.

    This ensures the video is long enough to hold all content comfortably.
    """
    scene = state.get("_current_scene", {})
    sid   = scene.get("scene_id")
    chars = scene.get("characters", [])

    duration = _compute_scene_duration(scene)

    n_dialogue = len(scene.get("dialogue", []) or [])
    n_actions  = len(scene.get("actions", [])
                     or scene.get("action_beats", []) or [])
    print(f"  [Video Gen    | scene {sid}] Rendering {duration:.1f}s "
          f"({n_dialogue} dialogue lines, {n_actions} action beats)")

    raw = await _call_tool(
        "query_stock_footage",
        scene_id=sid,
        location=scene.get("location", ""),
        characters=chars,
        visual_cue=scene.get("scene_visual_cue", "") or scene.get("visual_cue", ""),
        duration_s=duration,
        mood=scene.get("mood", "") or scene.get("tone", ""),
        description=scene.get("description", "") or scene.get("scene_description", ""),
    )
    base = _safe_parse(raw)
    base_video = base.get("file", "")

    payload = {
        "scene_id":   sid,
        "base_video": base_video,
        "duration_s": duration,
        "characters": chars,
    }

    await _call_tool(
        "commit_memory",
        key=f"video_scene_{sid:02d}" if isinstance(sid, int) else f"video_scene_{sid}",
        value=json.dumps(payload),
        category="video_clip",
    )

    print(f"  [Video Gen    | scene {sid}] ✓ {base_video} ({duration:.1f}s)")
    return {"video_clips": {str(sid): payload}}


# ─── Node 3: Face Swap ────────────────────────────────────────────────────────

async def face_swap_node(state: AgentState) -> dict:
    """PDF §5.4 — validate identity, then composite character face onto base video."""
    scene    = state.get("_current_scene", {})
    char_db  = state.get("character_db", {})
    sid      = scene.get("scene_id")
    chars    = scene.get("characters", [])
    location = scene.get("location", "")
    primary  = chars[0] if chars else "Unknown"
    skey     = str(sid)

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
        await _call_tool(
            "commit_memory",
            key=f"faceswap_scene_{sid:02d}" if isinstance(sid, int) else f"faceswap_scene_{sid}",
            value=json.dumps(payload), category="face_swap",
        )
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


# ─── Node 4: Lip Sync — with full fallback logic ─────────────────────────────

async def lip_sync_node(state: AgentState) -> dict:
    """
    PDF §5.5 — align Phase 2's mixed audio to face-swapped video.

    Fallback matrix
    ──────────────────────────────────────────────────────────────
    video ✓  audio ✓  → normal lip_sync_aligner                (Case A)
    video ✓  audio ✗  → mix ambient audio onto existing video  (Case B)
    video ✗  audio ✓  → bg image + dialogue audio → ffmpeg MP4 (Case C)
    video ✗  audio ✗  → bg image + ambient audio  → ffmpeg MP4 (Case D)
    ──────────────────────────────────────────────────────────────
    No scene is ever silently dropped.
    """
    scene    = state.get("_current_scene", {})
    manifest = state.get("scene_manifest", {}) or {}
    sid      = scene.get("scene_id")
    skey     = str(sid)

    audio_entry = (state.get("audio_tracks") or {}).get(skey, {})
    video_entry = (state.get("face_swapped_clips") or {}).get(skey, {})

    audio_files = audio_entry.get("files", [])
    video_file  = video_entry.get("face_swapped_file", "")

    duration = _compute_scene_duration(scene)

    out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Case A: Both present → normal lip-sync ────────────────────────────────
    if video_file and audio_files:
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
        print(f"  [Lip Sync     | scene {sid}] ✓ {result.get('file', 'N/A')} "
              f"(score={score}, {dur}s)")
        return {"final_scenes": {skey: result}}

    # ── Case B: Video ✓, Audio ✗ → overlay ambient sound ─────────────────────
    if video_file and not audio_files:
        print(f"  [Lip Sync     | scene {sid}] ⚠ No dialogue audio — "
              f"adding ambient track to existing video")

        ambient_path = await _fetch_or_generate_ambient(scene, manifest, sid, duration)
        out_file = out_dir / f"scene_{sid}_ambient.mp4"

        if ambient_path and _has_ffmpeg():
            cmd = [
                "ffmpeg", "-y",
                "-i", str(Path(video_file).resolve()),
                "-i", str(Path(ambient_path).resolve()),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(out_file),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if proc.returncode == 0:
                    result = {
                        "scene_id":     sid,
                        "file":         str(out_file),
                        "status":       "ambient_audio_fallback",
                        "duration_s":   duration,
                        "ambient_track": ambient_path,
                    }
                    await _call_tool(
                        "commit_memory",
                        key=f"final_scene_{sid:02d}" if isinstance(sid, int) else f"final_scene_{sid}",
                        value=json.dumps(result), category="final_scene",
                    )
                    print(f"  [Lip Sync     | scene {sid}] ✓ Ambient mix → {out_file}")
                    return {"final_scenes": {skey: result}}
                else:
                    print(f"  [Lip Sync     | scene {sid}] ✗ ffmpeg ambient mix failed: "
                          f"{proc.stderr[-300:]}")
            except subprocess.TimeoutExpired:
                print(f"  [Lip Sync     | scene {sid}] ✗ ffmpeg ambient mix timed out")

        # Sub-fallback: return silent video as-is
        result = {
            "scene_id":   sid,
            "file":       video_file,
            "status":     "no_audio_fallback",
            "duration_s": duration,
        }
        await _call_tool(
            "commit_memory",
            key=f"final_scene_{sid:02d}" if isinstance(sid, int) else f"final_scene_{sid}",
            value=json.dumps(result), category="final_scene",
        )
        print(f"  [Lip Sync     | scene {sid}] ⚠ Using video without audio track")
        return {"final_scenes": {skey: result}}

    # ── Cases C & D: No face-swapped video → build from background image ──────
    print(f"  [Lip Sync     | scene {sid}] ⚠ No face-swapped video — "
          f"compositing background-image scene ({duration:.1f}s)")

    bg_image = await _fetch_or_generate_bg_image(scene, manifest, sid)

    # For audio: prefer dialogue track (Case C), else ambient (Case D)
    audio_src: str = audio_files[0] if audio_files else ""
    if not audio_src:
        audio_src = await _fetch_or_generate_ambient(scene, manifest, sid, duration)

    out_file = out_dir / f"scene_{sid}_bg_fallback.mp4"

    success = _make_bg_image_scene(
        image_path=bg_image or "",
        audio_path=audio_src or "",
        duration_s=duration,
        output_path=out_file,
    )

    if success:
        result = {
            "scene_id":   sid,
            "file":       str(out_file),
            "status":     "bg_image_fallback",
            "duration_s": duration,
            "bg_image":   bg_image,
            "audio_src":  audio_src,
        }
        print(f"  [Lip Sync     | scene {sid}] ✓ BG-image scene → {out_file}")
    else:
        result = {
            "scene_id": sid,
            "file":     "",
            "status":   "failed",
            "reason":   "no video, no ffmpeg, or ffmpeg composition error",
        }
        print(f"  [Lip Sync     | scene {sid}] ✗ Could not build fallback scene")

    await _call_tool(
        "commit_memory",
        key=f"final_scene_{sid:02d}" if isinstance(sid, int) else f"final_scene_{sid}",
        value=json.dumps(result), category="final_scene",
    )
    return {"final_scenes": {skey: result}}


# ─── Node 5: Compositor ───────────────────────────────────────────────────────

async def compositor_node(state: AgentState) -> dict:
    """
    PDF §4 Phase 3 final task — concatenate per-scene MP4s into final_output.mp4.

    Uses ffmpeg's concat demuxer when available; otherwise reports the list
    of per-scene files so the user can stitch manually.
    """
    print(f"\n[Compositor] Stitching final scenes into final_output.mp4...")

    finals = state.get("final_scenes", {}) or {}
    scenes = state.get("scenes", []) or []

    ordered_scenes = sorted(
        [s for s in scenes if isinstance(s, dict)],
        key=lambda s: s.get("scene_id", 0),
    )

    scene_files = []
    for s in ordered_scenes:
        sid  = str(s.get("scene_id"))
        info = finals.get(sid, {})
        f    = info.get("file", "")
        if f and Path(f).exists() and Path(f).stat().st_size > 0:
            scene_files.append(f)
        else:
            print(f"  [Compositor] ⚠ scene {sid}: no usable final file "
                  f"(status={info.get('status', 'missing')})")

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
        for f in scene_files:
            print(f"      • {f}")
        return {
            "status":       "complete",
            "current_node": "compositor_node",
            "errors":       ["ffmpeg not available — final_output.mp4 not generated"],
        }

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

    print(f"  [Compositor] ✓ {final_path}  "
          f"({final_path.stat().st_size / (1024*1024):.1f} MB)")

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
        "scene_statuses":    {sid: v.get("status", "unknown") for sid, v in finals.items()},
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
