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
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from moviepy import VideoFileClip, vfx

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
# query_stock_footage blocks the server for the whole ByteDance lifecycle: Seedance
# generation (poll loop, up to BYTEDANCE_POLL_TIMEOUT_S=900s) + the MP4 download
# (up to BYTEDANCE_DOWNLOAD_MAX_ATTEMPTS retries against a flaky TOS bucket). A single
# scene can legitimately take 15+ min, so the MCP client's read timeout MUST exceed
# that or the agent disconnects mid-call and the server's result is orphaned
# (ConnectionResetError WinError 10054). Override with PHASE3_MCP_READ_TIMEOUT_S.
_MCP_READ_TIMEOUT_S = float(os.getenv("PHASE3_MCP_READ_TIMEOUT_S", "1800"))
_DEFAULT_TIMEOUT = _httpx.Timeout(connect=10.0, read=_MCP_READ_TIMEOUT_S, write=60.0, pool=30.0)


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


def _get_duration(file_path: str) -> float:
    """Return duration in seconds of a media file via ffprobe, default 5.0."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip())
    except Exception:
        return 5.0


def _expected_base_video_path(scene_id: int, location: str = "") -> Path:
    """Locate the base video for a scene.

    Strategy (in order):
      1. Glob VIDEO_DIR for scene_{scene_id:02d}_*.mp4 — works regardless of
         what slug was used to name the file (prompt-based or location-based).
      2. Fall back to the old location-derived filename so the function
         remains backwards-compatible when called with a known path.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "video"
    # Prefer an exact match on disk (handles prompt-named files correctly)
    matches = sorted(
        base_dir.glob(f"scene_{scene_id:02d}_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]  # newest on disk — avoids stale filename after multiple runs
    # Legacy fallback: reconstruct from location (may not exist yet)
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
_MIN_SCENE_DURATION  = 0.5    # allow short scenes when audio is short
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


def _duration_from_state(state: AgentState, scene_id) -> float:
    """
    Resolve scene audio duration in priority order:
      1) Phase 2 timing_manifest-derived per-scene map
      2) Phase 2 audio_tracks[scene_id].total_duration
      3) 0.0 (caller fallback)
    """
    sid = str(scene_id)
    by_scene = state.get("scene_audio_durations") or {}
    if sid in by_scene:
        try:
            val = float(by_scene[sid])
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass

    audio_entry = (state.get("audio_tracks") or {}).get(sid, {})
    try:
        val = float(audio_entry.get("total_duration", 0.0) or 0.0)
        if val > 0:
            return val
    except (TypeError, ValueError):
        pass
    return 0.0


def _disk_audio_tracks_from_mixed_wavs() -> dict[str, dict]:
    """
    Read outputs/audio/scene_NN_full.wav and build audio_tracks entries with ffprobe duration.

    Prefer over Chroma audio_track rows when present: Chroma is append-only, so reruns leave
    multiple historical rows per scene_id unless queries dedupe.

    Matches the paths video_gen_node mux logic already checks for.
    """

    audio_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "audio"
    if not audio_dir.is_dir():
        return {}

    out: dict[str, dict] = {}
    for wav in sorted(audio_dir.glob("scene_*_full.wav")):
        m = re.match(r"scene_(\d+)_full\.wav$", wav.name, re.IGNORECASE)
        if not m:
            continue
        sid = str(int(m.group(1)))
        dur = float(_get_duration(str(wav)))
        if dur <= 0:
            continue
        out[sid] = {
            "scene_id":       int(sid),
            "files":          [str(wav.resolve())],
            "bgm_file":       "",
            "mood":           "neutral",
            "total_duration": round(dur, 3),
            "_source":        "disk:scene_*_full.wav",
        }
    return out


def _dedupe_chroma_audio_tracks(entries: list) -> dict[str, dict]:
    """Latest row per scene_id using metadata timestamp."""

    best: dict[str, tuple[str, dict]] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        try:
            payload = json.loads(entry.get("value") or "{}")
        except Exception:
            continue
        sid_raw = payload.get("scene_id")
        if sid_raw is None:
            continue
        sid = str(sid_raw)
        ts = str(entry.get("timestamp") or "")
        prev = best.get(sid)
        if prev is None or ts >= prev[0]:
            best[sid] = (ts, payload)
    return {sid: pl for sid, (_, pl) in best.items()}


async def _load_audio_tracks_merged(state: AgentState, scenes: list) -> tuple[dict, list[str]]:
    logs: list[str] = []

    merged: dict = dict(state.get("audio_tracks") or {})
    disk_map = _disk_audio_tracks_from_mixed_wavs()

    # On-disk mixed wav always wins — matches ffprobe duration and ffmpeg mux targets.
    for sid, row in disk_map.items():
        merged[sid] = row
    if disk_map:
        logs.append(f"[Scene Loader] Disk mixed audio overlay for scene(s): {sorted(disk_map.keys())}")

    wanted: set[str] = set()
    for s in scenes or []:
        if isinstance(s, dict) and s.get("scene_id") is not None:
            wanted.add(str(s["scene_id"]))

    missing = sorted(
        {sid for sid in wanted if float((merged.get(sid) or {}).get("total_duration") or 0.0) <= 0},
        key=lambda x: int(x) if str(x).isdigit() else 0,
    )
    if not missing:
        logs.append("[Scene Loader] Scene audio durations resolved (disk and/or orchestrator)")
        return merged, logs

    logs.append("[Scene Loader] MCP Chroma fallback for scenes missing disk/orchestrator audio...")
    try:
        mem_raw = await _call_tool("query_memory", category="audio_track", limit=200)
        mem = _safe_parse(mem_raw)
        by_sid = _dedupe_chroma_audio_tracks(mem.get("entries") or [])
        logs.append(f"[Scene Loader] Chroma audio_track rows (deduped): {len(by_sid)} scene(s)")
        for sid in missing:
            pl = by_sid.get(sid)
            if pl and float(pl.get("total_duration") or 0.0) > 0:
                merged[sid] = pl
                logs.append(f"[Scene Loader] Chroma audio used for scene {sid}")
            else:
                logs.append(f"[Scene Loader] No usable audio payload for scene {sid} — run Phase 2")
    except Exception as exc:
        logs.append(f"[Scene Loader] Chroma audio load failed: {exc}")

    return merged, logs


def _build_video_gen_prompt(scene: dict, manifest) -> str:
    """
    Build the full visual prompt for Seedance / fallback render from the screenplay.

    Previously only scene_visual_cue + location + mood were sent, which drops
    action_description and story context and makes clips feel "off-topic" vs
    the user's prompt and dialogue.
    """
    m = manifest or {}
    chunks: list[str] = []

    title = (m.get("title") or "").strip()
    genre = (m.get("genre") or "").strip()
    if title or genre:
        head = title if title else "Untitled"
        if genre:
            head = f"{head}, {genre}"
        chunks.append(head)

    logline = (m.get("logline") or "").strip()
    if logline:
        chunks.append(logline)

    visual_cue = (scene.get("scene_visual_cue") or scene.get("visual_cue") or "").strip()
    if visual_cue:
        chunks.append(visual_cue)

    action = (scene.get("action_description") or "").strip()
    if action:
        chunks.append(action)

    loc = (scene.get("location") or "").strip()
    tod = (scene.get("time_of_day") or "").strip()
    if loc or tod:
        chunks.append(", ".join(p for p in (loc, tod) if p))

    mood = (scene.get("mood") or scene.get("tone") or "").strip()
    if mood:
        chunks.append(f"Mood: {mood}")

    text = ". ".join(chunks) if chunks else ""
    if not text.strip():
        text = f"cinematic scene at {loc or 'unknown location'}"
    return text


def _build_scene_duration_map(timing_manifest: dict) -> dict:
    """Aggregate timing_manifest segments into per-scene durations (seconds)."""
    per_scene_ms: dict[str, int] = {}
    for seg in timing_manifest.get("segments", []) or []:
        if not isinstance(seg, dict):
            continue
        sid = seg.get("scene_id")
        if sid is None:
            continue
        start_ms = int(seg.get("start_ms", 0) or 0)
        end_ms = int(seg.get("end_ms", 0) or 0)
        seg_ms = max(0, end_ms - start_ms)
        if seg_ms <= 0:
            continue
        skey = str(sid)
        per_scene_ms[skey] = per_scene_ms.get(skey, 0) + seg_ms
    return {sid: round(ms / 1000.0, 3) for sid, ms in per_scene_ms.items() if ms > 0}


def _sync_clip_duration(raw_video: str, target_duration: float) -> str:
    """
    Loop or trim raw video so visual length exactly matches target_duration.
    Returns the synced file path (or raw_video if sync fails).
    """
    if not raw_video or target_duration <= 0:
        return raw_video

    raw_path = Path(raw_video)
    if not raw_path.exists():
        return raw_video

    synced_path = str(raw_path.with_name(f"{raw_path.stem}_synced.mp4"))
    clip = None
    synced = None
    try:
        clip = VideoFileClip(raw_video)
        if clip.duration <= 0:
            return raw_video

        fps = clip.fps or 8
        if target_duration < clip.duration:
            synced = clip.subclipped(0, target_duration)
        else:
            loops = max(1, math.ceil(target_duration / clip.duration))
            synced = clip.with_effects([vfx.Loop(n=loops)]).subclipped(0, target_duration)

        synced.write_videofile(synced_path, fps=fps, logger=None)
        return synced_path
    except Exception:
        return raw_video
    finally:
        try:
            if synced is not None:
                synced.close()
        except Exception:
            pass
        try:
            if clip is not None:
                clip.close()
        except Exception:
            pass


def _mux_scene_audio(video_path: str, audio_path: str, duration_s: float) -> str:
    """
    Attach scene audio to scene video (no lip-sync step).
    Returns muxed video path, or original path on failure.
    """
    if not video_path or not Path(video_path).exists():
        return video_path
    if not audio_path or not Path(audio_path).exists() or not _has_ffmpeg():
        return video_path

    in_v = Path(video_path)
    out_v = in_v.with_name(f"{in_v.stem}_aud.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-stream_loop", "-1", "-i", str(in_v),
                "-i", str(Path(audio_path)),
                "-t", str(max(0.5, float(duration_s))),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest",
                str(out_v),
            ],
            check=True,
            timeout=180,
        )
        if out_v.exists() and out_v.stat().st_size > 0:
            return str(out_v)
    except Exception:
        pass
    return video_path


def _concat_video_shots(shot_paths: list[str], out_path: Path) -> str:
    """Concatenate multiple generated shots into one scene clip."""
    valid = [p for p in shot_paths if p and Path(p).exists()]
    if not valid:
        return ""
    if len(valid) == 1:
        return valid[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _has_ffmpeg():
        return valid[0]

    concat_txt = out_path.with_suffix(".concat.txt")
    concat_txt.write_text(
        "\n".join(f"file '{Path(p).resolve()}'" for p in valid) + "\n",
        encoding="utf-8",
    )
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_txt),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out_path),
            ],
            capture_output=True, text=True, timeout=240,
        )
        if proc.returncode == 0 and out_path.exists():
            return str(out_path)
    except Exception:
        pass
    finally:
        try:
            concat_txt.unlink(missing_ok=True)
        except Exception:
            pass
    return valid[0]


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

    import random
    seed = state.get("global_seed") or random.randint(1, 2147483647)
    print(f"[Scene Loader] Global Seed: {seed}")

    manifest = None
    char_db  = None

    # Prefer unified script.json
    if script_path.exists():
        try:
            print(f"[Scene Loader] script.json path: {script_path.resolve()}")
            unified = json.loads(script_path.read_text(encoding="utf-8"))
            story = unified.get("story") or {}
            manifest = {
                "title":   story.get("title", "Untitled"),
                "genre":   story.get("genre", ""),
                "logline": story.get("logline", ""),
                "themes":  story.get("themes", []) or [],
                "scenes":  unified.get("scenes", []),
            }
            char_db = {"characters": unified.get("characters", [])}
            print(f"[Scene Loader] Source: script.json "
                  f"— «{manifest.get('title', 'Untitled')}» ({manifest.get('genre', '')})")
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
    print(
        "[Scene Loader] Hint: Exported MP4 basename uses the first segment of "
        "`_build_video_gen_prompt` — usually manifest title/genre plus scene prose. "
        "If titles match an older project, regenerate Phase 1 (new script.json)."
    )

    # Load Phase 2 timing_manifest.json for precise A/V duration sync.
    timing_manifest = {}
    scene_audio_durations = {}
    timing_candidates = []
    if script_path:
        timing_candidates.append(script_path.parent / "timing_manifest.json")
    if manifest_path:
        timing_candidates.append(manifest_path.parent / "timing_manifest.json")
    explicit_timing = state.get("timing_manifest_path")
    if explicit_timing:
        timing_candidates.insert(0, Path(explicit_timing))
    for candidate in timing_candidates:
        if candidate and candidate.exists():
            try:
                timing_manifest = json.loads(candidate.read_text(encoding="utf-8"))
                scene_audio_durations = _build_scene_duration_map(timing_manifest)
                print(f"[Scene Loader] timing_manifest loaded: {candidate} "
                      f"({len(scene_audio_durations)} scene durations)")
                break
            except Exception as e:
                print(f"[Scene Loader] timing_manifest unreadable at {candidate}: {e}")

    # Phase 2 audio: orchestrator passes dict; standalone Phase 3 must not trust Chroma blindly.
    audio_tracks, audio_logs = await _load_audio_tracks_merged(state, scenes)
    for line in audio_logs:
        print(line)

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

    # ── Clear stale per-scene MP4s from previous runs ────────────────────────
    # Old PIL-rendered clips (640x360@12fps) MUST be removed so the face_swapper
    # and compositor don't accidentally pick them up instead of fresh Kaggle
    # outputs (768x512@24fps).  The final_output.mp4 in outputs/final/ is NOT
    # deleted here so the user can always view the last successful run.
    _video_dir  = Path(__file__).resolve().parent.parent.parent / "outputs" / "video"
    _scenes_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "raw_scenes"
    _removed = 0
    for _d in (_video_dir, _scenes_dir):
        for _f in _d.glob("*.mp4"):
            try:
                _f.unlink()
                _removed += 1
            except OSError:
                pass
    if _removed:
        print(f"[Scene Loader] Cleared {_removed} stale MP4(s) from previous run")

    return {
        "scene_manifest":      manifest,
        "character_db":        char_db,
        "audio_tracks":        audio_tracks,
        "timing_manifest":     timing_manifest,
        "scene_audio_durations": scene_audio_durations,
        "task_graph":          task_graph,
        "scenes":              scenes,
        "completed_scene_ids": completed_ids,
        "global_seed":         seed,
        "status":              "scenes_loaded",
        "current_node":        "scene_loader_node",
    }


# ─── Node 2: Video Gen ────────────────────────────────────────────────────────

async def video_gen_node(state: AgentState) -> dict:
    """
    Generate base video for one scene.

    Duration priority:
      1. Phase 2 actual audio duration (from audio_tracks in state)
      2. Heuristic fallback via _compute_scene_duration()
      3. Minimum floor of _MIN_SCENE_DURATION (6.0s)

    Sends only (scene_id, prompt, duration_s) to the remote SD 1.5 + MoviePy
    GPU worker, or the clean local fallback renderer.
    """
    scene = state.get("_current_scene")
    if not scene:
        # Find first ungenerated scene
        scenes = sorted(state.get("scenes", []), key=lambda x: x.get("scene_id"))
        clips = state.get("video_clips") or {}
        for s in scenes:
            if str(s.get("scene_id")) not in clips:
                scene = s
                break
    
    if not scene:
        print("  [Video Gen] No more scenes to generate.")
        return {}

    sid   = scene.get("scene_id")
    chars = scene.get("characters", [])

    # ── Priority 1: Phase 2 timing manifest / audio durations ────────────────
    resolved_audio_duration = _duration_from_state(state, sid)
    if resolved_audio_duration > 0:
        duration = max(_MIN_SCENE_DURATION, float(resolved_audio_duration))
        print(f"  [Video Gen    | scene {sid}] Using audio duration: {duration:.2f}s")
    else:
        duration = _compute_scene_duration(scene)
        print(f"  [Video Gen    | scene {sid}] No audio track — heuristic duration: {duration:.1f}s")

    n_dialogue = len(scene.get("dialogue", []) or [])
    n_actions  = len(scene.get("actions", [])
                     or scene.get("action_beats", []) or [])
    print(f"  [Video Gen    | scene {sid}] Rendering {duration:.1f}s "
          f"({n_dialogue} dialogue lines, {n_actions} action beats)")

    # ── Single-call Seedance prompt per scene (full screenplay context) ───────
    manifest = state.get("scene_manifest") or {}
    prompt_text = _build_video_gen_prompt(scene, manifest)
    neg_prompt = "low quality, blurry, distorted anatomy, extra limbs, noisy, artifacts"

    print(f"  [Video Gen    | scene {sid}] prompt ({len(prompt_text)} chars): "
          f"{prompt_text[:240]}{'…' if len(prompt_text) > 240 else ''}")

    seed = state.get("global_seed", -1)
    # A slow/flaky ByteDance + TOS download can blow past even the generous MCP read
    # timeout (or the server can crash mid-response). Don't let one scene kill the
    # whole graph — degrade to "no base video" and let lip_sync_node build the
    # background-image fallback for this scene.
    raw_video = ""
    try:
        raw = await _call_tool(
            "query_stock_footage",
            scene_id=sid,
            prompt=prompt_text,
            num_frames=16,
            width=512,
            height=512,
            seed=seed if isinstance(seed, int) else -1,
            negative_prompt=neg_prompt,
            denoising_strength=0.0,
            init_image_b64="",
        )
        base = _safe_parse(raw)
        raw_video = base.get("file", "") or ""
    except Exception as e:
        print(f"  [Video Gen    | scene {sid}] ✗ query_stock_footage failed "
              f"({type(e).__name__}: {e}) — continuing without a base video for this scene")

    # Final trim/fit to exact target audio duration.
    base_video = _sync_clip_duration(raw_video, duration) if raw_video else raw_video

    # Attach mixed scene audio.
    skey = str(sid)
    audio_entry = (state.get("audio_tracks") or {}).get(skey, {})
    audio_mix = Path(__file__).resolve().parent.parent.parent / "outputs" / "audio" / f"scene_{int(sid):02d}_full.wav" if str(sid).isdigit() else None
    audio_src = ""
    if audio_mix and audio_mix.exists():
        audio_src = str(audio_mix)
    elif audio_entry.get("files"):
        first_audio = audio_entry.get("files", [])[0]
        if first_audio and Path(first_audio).exists():
            audio_src = str(first_audio)
    base_video = _mux_scene_audio(base_video, audio_src, duration) if base_video else base_video

    # Mirror into raw_scenes/ so outputs match the lip-sync path and are easy to find.
    if base_video and Path(base_video).exists() and Path(base_video).is_file():
        raw_scenes = Path(__file__).resolve().parent.parent.parent / "outputs" / "raw_scenes"
        raw_scenes.mkdir(parents=True, exist_ok=True)
        try:
            sid_int = int(sid) if str(sid).isdigit() else 0
            dest = raw_scenes / (f"scene_{sid_int:02d}.mp4" if sid_int else f"scene_{sid}.mp4")
            shutil.copy2(base_video, dest)
            print(f"  [Video Gen    | scene {sid}] mirror → {dest}")
        except OSError as e:
            print(f"  [Video Gen    | scene {sid}] ⚠ could not mirror to raw_scenes: {e}")

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
    final_entry = {
        "scene_id": sid,
        "file": base_video,
        "status": "video_only",
        "duration_s": duration,
    }
    return {
        "video_clips": {str(sid): payload},
        "final_scenes": {str(sid): final_entry},
    }


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

    # Prefer the path already committed by video_gen_node (authoritative source).
    # Fall back to disk-scan helper only when running face-swap in isolation.
    video_clip_entry = (state.get("video_clips") or {}).get(skey, {})
    stored_path = video_clip_entry.get("base_video", "")
    if stored_path and Path(stored_path).exists():
        expected = Path(stored_path)
    else:
        expected = _expected_base_video_path(sid, location)

    if not validation.get("valid"):
        print(f"  [Face Swap    | scene {sid}] ✗ Identity FAILED: {validation.get('reason')}")
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

    duration = _duration_from_state(state, sid) or _compute_scene_duration(scene)

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

    # ── Compose a filter_complex that normalises every clip to 1280x720@24fps
    # then concatenates them.  This is robust to mixed resolutions / frame-rates
    # (e.g. 640x360@12fps PIL clips vs 768x512@24fps Kaggle clips).
    TARGET_W, TARGET_H, TARGET_FPS = 1280, 720, 24

    n = len(scene_files)
    # Build -i args and filter_complex string
    input_args = []
    for f in scene_files:
        input_args += ["-i", f]

    # ── Probe each file for audio stream presence ─────────────────────────────
    # Clips coming from Kaggle /face_swap have no audio; clips from lip_sync_aligner do.
    # Using [i:a:0] on an audio-less file crashes the entire filter_complex.
    has_audio_list = []
    for f in scene_files:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", f],
            capture_output=True, text=True,
        )
        has_audio_list.append(bool(probe.stdout.strip()))

    # Scale+fps filter for each video input, then concat
    filter_parts = []
    for i in range(n):
        filter_parts.append(
            f"[{i}:v]scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={TARGET_FPS}[v{i}];"
        )
    video_labels = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{video_labels}concat=n={n}:v=1:a=0[vout];")

    # Audio: use the real audio stream if present, else synthesise silence
    audio_filter_parts = []
    for i in range(n):
        if has_audio_list[i]:
            audio_filter_parts.append(f"[{i}:a:0]aresample=44100[a{i}];")
        else:
            audio_filter_parts.append(
                f"anullsrc=r=44100:cl=stereo,atrim=0:{_get_duration(scene_files[i])}[a{i}];"
            )
    audio_labels = "".join(f"[a{i}]" for i in range(n))
    audio_filter_parts.append(f"{audio_labels}concat=n={n}:v=0:a=1[aout]")

    full_filter = "".join(filter_parts) + "".join(audio_filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", full_filter,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(final_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            # Fallback: try without audio (some clips may have no audio stream)
            print(f"  [Compositor] audio concat failed: {proc.stderr[-400:]}")
            print(f"  [Compositor] retrying video-only...")
            # Rebuild without audio concat
            v_only_filter = "".join(filter_parts)  # just [v0][v1]...concat[vout]
            cmd_vonly = [
                "ffmpeg", "-y",
                *input_args,
                "-filter_complex", v_only_filter,
                "-map", "[vout]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-an",
                "-movflags", "+faststart",
                str(final_path),
            ]
            proc = subprocess.run(cmd_vonly, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                return {
                    "status":       "failed",
                    "current_node": "compositor_node",
                    "errors":       [f"ffmpeg concat failed: {proc.stderr[-600:]}"],
                }
    except subprocess.TimeoutExpired:
        return {
            "status":       "failed",
            "current_node": "compositor_node",
            "errors":       ["ffmpeg concat timed out"],
        }

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
