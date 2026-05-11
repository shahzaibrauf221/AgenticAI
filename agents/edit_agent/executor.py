# ============================================================
# agents/edit_agent/executor.py
# Phase 5 — Edit Executor node
#
# Executes the plan by:
#  1. Loading current pipeline state from StateManager
#  2. Applying the requested edit (audio / video_frame / video / script)
#  3. Snapshotting the new state
#
# MCP servers are called when available (ports 8100/8200).
# Every handler gracefully degrades if a server is offline.
# ============================================================

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.edit_agent.state import EditAgentState
from agents.edit_agent.tools.opencv_filters import apply_filter_to_image
from state_manager.manager import StateManager

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUTS_DIR  = _PROJECT_ROOT / "outputs"
_AUDIO_DIR    = _OUTPUTS_DIR / "audio"
_IMAGE_DIR    = _OUTPUTS_DIR / "image_assets"
_FRAMES_DIR   = _OUTPUTS_DIR / "frames"
_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
_FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def _phase1_outputs_dir() -> Path:
    """Directory where Phase 1 (writers_room_server) writes scene_manifest.json.

    Canonically <project_root>/outputs/. Some older callers passed
    <project_root>/data/outputs/, which is why a pipeline rerun would die with
    "scene_manifest.json not found at ...data/outputs/...". Resolve to wherever
    the manifest actually is, preferring outputs/.
    """
    if (_OUTPUTS_DIR / "scene_manifest.json").exists():
        return _OUTPUTS_DIR
    legacy = _PROJECT_ROOT / "data" / "outputs"
    if (legacy / "scene_manifest.json").exists():
        return legacy
    return _OUTPUTS_DIR

# ─── MCP helpers ──────────────────────────────────────────────────────────────

_STUDIO_CONFIG = {
    "studio_floor": {
        "url":       "http://localhost:8200/mcp",
        "transport": "streamable_http",
    }
}
_WRITERS_CONFIG = {
    "writers_room": {
        "url":       "http://localhost:8100/mcp",
        "transport": "streamable_http",
    }
}

_studio_tools:  dict = {}
_writers_tools: dict = {}


async def _get_studio_tools() -> dict:
    global _studio_tools
    if not _studio_tools:
        try:
            client = MultiServerMCPClient(_STUDIO_CONFIG)
            tools  = await client.get_tools(server_name="studio_floor")
            _studio_tools = {t.name: t for t in tools}
        except Exception:
            pass
    return _studio_tools


async def _get_writers_tools() -> dict:
    global _writers_tools
    if not _writers_tools:
        try:
            client = MultiServerMCPClient(_WRITERS_CONFIG)
            tools  = await client.get_tools(server_name="writers_room")
            _writers_tools = {t.name: t for t in tools}
        except Exception:
            pass
    return _writers_tools


def _extract_text(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return _extract_text(result.content)
    if isinstance(result, list):
        for b in result:
            if isinstance(b, dict) and b.get("type") == "text":
                return b["text"]
            if isinstance(b, str):
                return b
        return json.dumps(result)
    if isinstance(result, dict) and result.get("type") == "text":
        return result.get("text", "")
    return json.dumps(result)


async def _call_studio(tool: str, logs: list[str], **kwargs) -> dict | None:
    tmap = await _get_studio_tools()
    if tool not in tmap:
        logs.append(f"[executor] Studio MCP offline or tool '{tool}' unavailable")
        return None
    try:
        raw = _extract_text(await tmap[tool].ainvoke(kwargs))
        return json.loads(raw)
    except Exception as e:
        logs.append(f"[executor] Studio MCP call failed: {e}")
        return None


async def _call_writers(tool: str, logs: list[str], **kwargs) -> dict | None:
    tmap = await _get_writers_tools()
    if tool not in tmap:
        logs.append(f"[executor] Writers MCP offline or tool '{tool}' unavailable")
        return None
    try:
        raw = _extract_text(await tmap[tool].ainvoke(kwargs))
        return json.loads(raw)
    except Exception as e:
        logs.append(f"[executor] Writers MCP call failed: {e}")
        return None


async def _mcp_available(server: str) -> bool:
    port = 8200 if server == "studio" else 8100
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"http://localhost:{port}/mcp")
            return r.status_code < 500
    except Exception:
        return False


# ─── Executor node ────────────────────────────────────────────────────────────

async def execute_edit(state: EditAgentState) -> EditAgentState:
    intent     = state.get("intent",     "recompose_video")
    target     = state.get("target",     "video")
    scope      = state.get("scope",      "all")
    parameters = state.get("parameters", {})
    version    = state.get("current_version", StateManager.latest_version() or "v1")
    rerun_from = state.get("rerun_from", "phase3")
    rerun_pipeline = bool(state.get("rerun_pipeline", False))

    logs: list[str] = [f"[executor] intent={intent} target={target} scope={scope}"]
    result: dict    = {}

    try:
        try:
            current_state = StateManager.get_state(version)
        except ValueError:
            current_state = {}
            logs.append(f"[executor] Version {version} not found, using empty state")

        if intent == "clarify_request":
            note = (
                f"Your command targets {scope} but didn't say what to change. "
                f"Try: 'make {scope} darker', 'add background music to {scope}', "
                f"'change the dialogue in {scope}', 'speed up {scope}', "
                f"'regenerate the image for {scope}', or 'change voice tone in {scope}'."
            )
            logs.append(f"[executor] clarify_request — no edit applied. {note}")
            return {
                **state,
                "result": {"type": "clarify", "applied": False, "note": note,
                           "scope": scope, "asset_paths": []},
                "new_version": version,
                "status": "needs_clarification",
                "logs": logs,
            }

        if target == "audio":
            result = await _handle_audio_edit(intent, scope, parameters, current_state, logs)
        elif target == "video_frame":
            result = await _handle_video_frame_edit(intent, scope, parameters, current_state, logs)
        elif target == "video":
            result = await _handle_video_edit(intent, scope, parameters, current_state, logs)
        elif target == "script":
            result = await _handle_script_edit(intent, scope, parameters, current_state, logs)
        else:
            raise ValueError(f"Unknown target: {target}")

        current_state["last_edit"] = f"{intent} on {scope}"
        current_state["edit_history"] = current_state.get("edit_history", []) + [
            {"intent": intent, "target": target, "scope": scope, "parameters": parameters}
        ]

        asset_paths = result.get("asset_paths", [])

        if rerun_pipeline:
            logs.append(f"[executor] rerun requested from {rerun_from}")
            from agents.orchestrator.pipeline import run_targeted_rerun

            pipeline_result = await run_targeted_rerun(
                entry_phase=rerun_from,
                phase1_dir=_phase1_outputs_dir(),
                prompt=parameters.get("prompt", current_state.get("user_prompt", "")),
            )
            result["pipeline_rerun"] = pipeline_result
            phase3 = pipeline_result.get("phase3", {})
            if isinstance(phase3, dict) and phase3.get("final_video"):
                asset_paths.append(str(phase3["final_video"]))

        new_version = StateManager.snapshot(
            state=current_state,
            asset_paths=asset_paths,
            label=f"{intent} ({scope})",
        )
        logs.append(f"[executor] Snapshot saved as {new_version}")

        return {**state, "result": result, "new_version": new_version, "status": "done", "logs": logs}

    except Exception as exc:
        logs.append(f"[executor] ERROR: {exc}")
        return {**state, "result": {"error": str(exc)}, "new_version": version, "status": "failed", "errors": [str(exc)], "logs": logs}


# ─── Audio edits ──────────────────────────────────────────────────────────────

async def _handle_audio_edit(intent: str, scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    logs.append(f"[executor:audio] intent={intent} scope={scope}")

    # ── change_voice_tone / change_voice_speed — re-synthesize via MCP ────────
    if intent in ("change_voice_tone", "change_voice_speed", "regenerate_audio"):
        return await _resynthesize_voice(intent, scope, parameters, state, logs)

    # ── add / remove background music ─────────────────────────────────────────
    if intent == "add_background_music":
        return await _add_bgm(scope, parameters, state, logs)

    if intent == "remove_background_music":
        return await _remove_bgm(scope, state, logs)

    logs.append(f"[executor:audio] Unhandled audio intent: {intent}")
    return {"type": "audio", "intent": intent, "asset_paths": [], "applied": False}


async def _resynthesize_voice(intent: str, scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    """
    Re-run TTS for the target character/scene via the Studio Floor MCP server.
    Falls back to direct gTTS if MCP is offline.
    """
    # Determine which character and emotion/speed to use
    character = scope.replace("character:", "").strip() if scope.startswith("character:") else None
    scene_id  = int(scope.replace("scene:", "")) if scope.startswith("scene:") else None
    emotion   = parameters.get("tone", parameters.get("emotion", "neutral"))
    tld_map   = {"slow": "co.uk", "fast": "com.au", "medium": "com",
                 "whispered": "co.uk", "excited": "com.au"}
    tld       = tld_map.get(parameters.get("pace", ""), "com")

    # Find relevant dialogue lines from state
    scenes = state.get("scenes", [])
    if not scenes:
        # Try loading from disk
        manifest = _PROJECT_ROOT / "outputs" / "scene_manifest.json"
        if manifest.exists():
            try:
                scenes = json.loads(manifest.read_text()).get("scenes", [])
            except Exception:
                pass

    target_lines: list[dict] = []
    for sc in scenes:
        sid = sc.get("scene_id")
        if scene_id is not None and str(sid) != str(scene_id):
            continue
        for dl in sc.get("dialogue", []):
            spk = dl.get("speaker", "")
            if character and character.lower() not in spk.lower():
                continue
            target_lines.append({"scene_id": sid, "speaker": spk, "line": dl.get("line", "")})

    if not target_lines:
        logs.append("[executor:audio] No matching dialogue lines found in state — checking disk")
        # If no state data, just report what we'd do
        return {
            "type": "audio", "intent": intent, "scope": scope,
            "asset_paths": [], "applied": False,
            "note": "No dialogue lines found in state. Run Phase 1 first to generate script."
        }

    logs.append(f"[executor:audio] Found {len(target_lines)} line(s) to re-synthesize")

    studio_up = await _mcp_available("studio")
    asset_paths: list[str] = []
    results: list[dict]    = []

    for entry in target_lines:
        sid  = entry["scene_id"]
        spk  = entry["speaker"]
        line = entry["line"]

        if studio_up:
            logs.append(f"[executor:audio] MCP → voice_cloning_synthesizer scene={sid} char={spk} emotion={emotion}")
            res = await _call_studio(
                "voice_cloning_synthesizer", logs,
                scene_id=sid,
                character_name=spk,
                dialogue_line=line,
                emotion=emotion,
                tld=tld,
            )
            if res and res.get("status") in ("success", "fallback"):
                fpath = res.get("file", "")
                if fpath and Path(fpath).exists():
                    asset_paths.append(fpath)
                logs.append(f"[executor:audio] ✓ {spk} scene {sid} → {Path(fpath).name if fpath else '?'} ({res.get('provider','?')})")
                results.append(res)
            else:
                logs.append(f"[executor:audio] MCP returned error: {res}")
                # Fallback to direct gTTS
                fb = _direct_gtts(sid, spk, line, emotion, tld, logs)
                if fb:
                    asset_paths.append(fb)
        else:
            logs.append("[executor:audio] Studio MCP offline — using direct gTTS fallback")
            fb = _direct_gtts(sid, spk, line, emotion, tld, logs)
            if fb:
                asset_paths.append(fb)

    return {
        "type":        "audio",
        "intent":      intent,
        "emotion":     emotion,
        "lines_processed": len(target_lines),
        "asset_paths": asset_paths,
        "applied":     bool(asset_paths),
        "results":     results,
    }


def _direct_gtts(scene_id: int, character: str, line: str, emotion: str, tld: str, logs: list[str]) -> str | None:
    """Direct gTTS call without MCP — used as offline fallback."""
    try:
        from gtts import gTTS
        safe_char = re.sub(r"[^a-z0-9]", "_", character.lower())
        line_hash = hashlib.md5(line.encode()).hexdigest()[:6]
        out_path  = _AUDIO_DIR / f"scene_{scene_id:02d}_{safe_char}_{line_hash}_edit.mp3"
        gTTS(text=line, lang="en", tld=tld).save(str(out_path))
        logs.append(f"[executor:audio] gTTS fallback ✓ → {out_path.name}")
        return str(out_path)
    except ImportError:
        logs.append("[executor:audio] gTTS not installed — cannot synthesize offline")
    except Exception as e:
        logs.append(f"[executor:audio] gTTS failed: {e}")
    return None


def _overlay_bgm_on_scene_clip(scene_id: int, bgm_path: str, logs: list[str]) -> str | None:
    """Mix `bgm_path` (ducked under any existing dialogue) into the scene's best
    clip → outputs/video/scene_NN_..._bgm.mp4, so the next recompose picks it up.

    Self-contained: no Phase 2/3 rerun, no ByteDance, doesn't disturb other per-scene
    edit variants. Returns the new clip path, or None if ffmpeg/clip/bgm is missing.
    """
    if not _has_ffmpeg() or not bgm_path or not Path(bgm_path).exists():
        return None
    src = _pick_scene_clip(_OUTPUTS_DIR / "video", scene_id)
    if src is None:
        logs.append(f"[executor:audio] no scene-{scene_id} clip to overlay BGM onto — run Phase 3 first")
        return None
    out = src.with_name(f"{src.stem}_bgm{src.suffix}")
    has_audio = bool(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True).stdout.strip())
    if has_audio:
        # dialogue stays at full volume; BGM ducked to ~25%
        fc = ("[1:a]volume=0.25[bg];"
              "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]")
        cmd = ["ffmpeg", "-y", "-i", str(src), "-i", str(Path(bgm_path)),
               "-filter_complex", fc, "-map", "0:v", "-map", "[aout]",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-i", str(src), "-i", str(Path(bgm_path)),
               "-map", "0:v", "-map", "1:a",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            logs.append(f"[executor:audio] ✓ BGM mixed into scene {scene_id} → {out.name}")
            return str(out)
        logs.append(f"[executor:audio] ✗ ffmpeg BGM overlay scene {scene_id}: {proc.stderr[:250]}")
    except subprocess.TimeoutExpired:
        logs.append(f"[executor:audio] ✗ ffmpeg BGM overlay timed out scene {scene_id}")
    return None


async def _add_bgm(scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    scene_id = int(scope.replace("scene:", "")) if scope.startswith("scene:") else 1
    mood     = parameters.get("mood", "cinematic")
    duration = float(parameters.get("duration", 10.0))

    # 1) Get a BGM track (MCP, else local synth fallback).
    bgm_path = ""
    res: dict | None = None
    if await _mcp_available("studio"):
        logs.append(f"[executor:audio] MCP → generate_background_music scene={scene_id} mood={mood}")
        res = await _call_studio("generate_background_music", logs,
                                 scene_id=scene_id, mood=mood, duration_s=duration)
        if res and res.get("status") == "success":
            bgm_path = res.get("file", "") or ""
        else:
            logs.append(f"[executor:audio] BGM MCP failed: {res}")
    if not bgm_path:
        logs.append("[executor:audio] Generating BGM directly (no MCP)")
        bgm_path = _generate_bgm_direct(scene_id, mood, duration, logs) or ""
    if not bgm_path:
        return {"type": "audio", "intent": "add_background_music", "scene_id": scene_id,
                "mood": mood, "asset_paths": [], "applied": False,
                "note": "could not produce a BGM track"}

    # 2) Mix it straight onto the scene's video clip so a recompose picks it up.
    new_clip = _overlay_bgm_on_scene_clip(scene_id, bgm_path, logs)
    assets = [p for p in (new_clip, bgm_path) if p]
    note = ("run 'recompose the video' to fold the new music into the final cut"
            if new_clip else
            "BGM track produced but no scene clip to overlay it onto — run Phase 3, then redo this edit")
    return {"type": "audio", "intent": "add_background_music", "scene_id": scene_id,
            "mood": mood, "bgm_track": bgm_path, "output": new_clip,
            "asset_paths": assets, "applied": bool(new_clip), "note": note,
            **({"result": res} if res else {})}


def _generate_bgm_direct(scene_id: int, mood: str, duration: float, logs: list[str]) -> str | None:
    """Generate a simple sine-wave BGM track using numpy/scipy if available."""
    try:
        import numpy as np
        import wave, struct

        _MOODS = {
            "tense":      (110, [82.4, 110.0, 146.8]),
            "mysterious": (70,  [146.8, 220.0, 277.2]),
            "upbeat":     (130, [261.6, 329.6, 392.0]),
            "romantic":   (80,  [196.0, 246.9, 293.7]),
            "cinematic":  (90,  [220.0, 277.2, 329.6]),
            "neutral":    (90,  [220.0, 277.2, 329.6]),
        }
        bpm, freqs = _MOODS.get(mood, _MOODS["neutral"])
        sample_rate = 22050
        n_samples   = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        wave_data = np.zeros(n_samples)
        for f in freqs:
            wave_data += np.sin(2 * np.pi * f * t)
        # Fade in/out
        fade = int(sample_rate * 0.5)
        wave_data[:fade]  *= np.linspace(0, 1, fade)
        wave_data[-fade:] *= np.linspace(1, 0, fade)
        wave_data = (wave_data / wave_data.max() * 0.4 * 32767).astype(np.int16)

        out_path = _AUDIO_DIR / f"bgm_scene_{scene_id:02d}_{mood}_edit.wav"
        with wave.open(str(out_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{len(wave_data)}h", *wave_data))

        logs.append(f"[executor:audio] BGM generated → {out_path.name}")
        return str(out_path)
    except Exception as e:
        logs.append(f"[executor:audio] BGM generation failed: {e}")
        return None


async def _remove_bgm(scope: str, state: dict, logs: list[str]) -> dict:
    """Remove BGM by deleting matching BGM files."""
    scene_id = scope.replace("scene:", "").strip() if scope.startswith("scene:") else "all"
    removed: list[str] = []
    pattern = f"bgm_scene_{int(scene_id):02d}_*" if scene_id != "all" else "bgm_*.wav"
    for f in _AUDIO_DIR.glob(pattern):
        f.unlink(missing_ok=True)
        removed.append(str(f))
        logs.append(f"[executor:audio] Removed BGM: {f.name}")
    return {"type": "audio", "intent": "remove_background_music",
            "removed": removed, "asset_paths": [], "applied": bool(removed)}


# ─── Video-frame (image) edits ────────────────────────────────────────────────

_FILTER_PRESETS: dict[str, dict] = {
    "darker":    {"brightness": 0.5,  "contrast": 1.0},
    "brighter":  {"brightness": 1.6,  "contrast": 1.1},
    "sepia":     {"sepia": True},
    "grayscale": {"grayscale": True},
    "warm":      {"hue_shift": 15},
    "cool":      {"hue_shift": -20},
    "vintage":   {"brightness": 0.85, "sepia": True},
    "vivid":     {"saturation": 1.5,  "contrast": 1.2},
}

# FFmpeg -vf equivalents — used when scenes only exist as MP4 clips
_FFMPEG_FILTER_VF: dict[str, str] = {
    "darker":    "eq=brightness=-0.2:contrast=1.0",
    "brighter":  "eq=brightness=0.15:contrast=1.1",
    "grayscale": "format=gray",
    "vivid":     "eq=saturation=1.5:contrast=1.2",
    "sepia":     ("colorchannelmixer=rr=0.393:rg=0.769:rb=0.189"
                  ":gr=0.349:gg=0.686:gb=0.168"
                  ":br=0.272:bg=0.534:bb=0.131"),
    "warm":      "colorbalance=rs=0.1:bs=-0.1",
    "cool":      "colorbalance=rs=-0.1:bs=0.1",
    "vintage":   ("eq=saturation=0.7:brightness=-0.05,"
                  "colorchannelmixer=rr=0.9:rg=0.5:rb=0.1"
                  ":gr=0.3:gg=0.7:gb=0.2:br=0.2:bg=0.5:bb=0.5"),
}


def _apply_filter(image_path: Path, filter_name: str, output_path: Path) -> bool:
    return apply_filter_to_image(image_path=image_path, filter_name=filter_name, output_path=output_path)


async def _handle_video_frame_edit(intent: str, scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    logs.append(f"[executor:video_frame] intent={intent} scope={scope}")

    # ── Visual filters (local — no MCP needed) ────────────────────────────────
    if intent in ("make_scene_darker", "make_scene_brighter", "apply_color_filter"):
        filter_map = {
            "make_scene_darker":   "darker",
            "make_scene_brighter": "brighter",
            "apply_color_filter":  parameters.get("filter", "sepia"),
        }
        return await _apply_scene_visual_filter(filter_map[intent], scope, logs)

    # ── Change character design — call Writers Room MCP ───────────────────────
    if intent == "change_character_design":
        return await _regenerate_character_image(scope, parameters, state, logs)

    # ── Regenerate scene image — call Studio Floor MCP ────────────────────────
    if intent in ("regenerate_scene_image", "change_scene_style"):
        return await _regenerate_scene_image(intent, scope, parameters, state, logs)

    logs.append(f"[executor:video_frame] Unhandled: {intent}")
    return {"type": "video_frame", "intent": intent, "asset_paths": [], "applied": False}


def _scene_mp4s_exist(scope: str) -> bool:
    """True if outputs/video/ has at least one *source* scene clip for `scope`.

    Excludes our own filter outputs (scene_NN_..._darker.mp4 etc.) so a re-run of
    'make scene N darker' still sees the original clip to filter.
    """
    video_dir = _OUTPUTS_DIR / "video"
    if not video_dir.is_dir():
        return False
    if scope.startswith("scene:"):
        sid = scope.split(":", 1)[1].strip().zfill(2)
        pattern = f"scene_{sid}_*.mp4"
    else:
        pattern = "scene_*.mp4"
    known_filters = set(_FFMPEG_FILTER_VF)
    for p in video_dir.glob(pattern):
        if p.stem.rsplit("_", 1)[-1] in known_filters:
            continue  # this is itself a filter output
        return True
    return False


async def _apply_scene_visual_filter(filter_name: str, scope: str, logs: list[str]) -> dict:
    """Darken / brighten / colour-filter a scene.

    This project renders scenes as MP4 clips and the final video is stitched from
    those clips — the still frames in outputs/frames/ are stale render artifacts the
    final video never uses. So filter the scene video(s) first (writes
    scene_NN_..._<filter>.mp4, which the recompose picker prefers as the newest
    variant), and fall back to filtering still frames only when there's no scene clip.
    """
    if _scene_mp4s_exist(scope):
        result = await _apply_video_color_filter(filter_name, scope, logs)
        if result.get("applied"):
            return result
        logs.append("[executor:video_frame] scene MP4 filter produced nothing — "
                    "falling back to still frames")
    return await _apply_image_filter(filter_name, scope, logs)


async def _apply_image_filter(filter_name: str, scope: str, logs: list[str]) -> dict:
    exts = {".png", ".jpg", ".jpeg", ".webp"}

    def _collect_from_dir(d: Path) -> list[Path]:
        """Recursively collect image files from a directory."""
        return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    if scope.startswith("scene:"):
        sid = scope.split(":")[1]
        sid_padded = sid.zfill(2)  # "1" → "01"

        # Look in outputs/frames/scene_01/ (and skip _swapped variants)
        targets: list[Path] = []
        for folder in _FRAMES_DIR.iterdir():
            if not folder.is_dir():
                continue
            name = folder.name
            if "_swapped" in name:
                continue  # skip swapped folders
            if name == f"scene_{sid_padded}" or name == f"scene_{sid}"                or name == f"scene{sid_padded}" or name == f"scene{sid}":
                targets.extend(_collect_from_dir(folder))

        # Fallback: flat image_assets dir
        if not targets:
            all_images = [p for p in _IMAGE_DIR.iterdir() if p.suffix.lower() in exts]
            targets = [
                p for p in all_images
                if f"scene_{sid_padded}" in p.name or f"scene_{sid}" in p.name
                or f"scene{sid_padded}" in p.name or f"scene{sid}" in p.name
            ]
    else:
        # "all" — collect from every scene_XX folder (skip _swapped)
        targets = []
        for folder in _FRAMES_DIR.iterdir():
            if folder.is_dir() and "_swapped" not in folder.name:
                targets.extend(_collect_from_dir(folder))
        if not targets:
            targets = [p for p in _IMAGE_DIR.iterdir() if p.suffix.lower() in exts]

    if not targets:
        # No still images for this scope — fall back to filtering the scene MP4(s).
        logs.append(f"[executor:video_frame] No images for {scope}; trying scene MP4 fallback")
        return await _apply_video_color_filter(filter_name, scope, logs)

    logs.append(f"[executor:video_frame] Filter '{filter_name}' on {len(targets)} image(s)")
    asset_paths: list[str] = []
    for img_path in targets:
        out = _IMAGE_DIR / f"{img_path.stem}_{filter_name}{img_path.suffix}"
        if _apply_filter(img_path, filter_name, out):
            asset_paths.append(str(out))
            logs.append(f"[executor:video_frame] ✓ {out.name}")
        else:
            logs.append(f"[executor:video_frame] ✗ Could not process {img_path.name}")

    return {"type": "video_frame", "intent": f"apply_{filter_name}_filter",
            "filter": filter_name, "images_edited": len(asset_paths),
            "asset_paths": asset_paths, "applied": bool(asset_paths)}


async def _apply_video_color_filter(filter_name: str, scope: str, logs: list[str]) -> dict:
    """
    Fallback for projects where scenes are MP4 clips, not still frames.
    Picks the best per-scene MP4 in outputs/video/ and applies an FFmpeg
    -vf filter, writing scene_NN_..._<filter>.mp4 alongside the source.
    """
    vf = _FFMPEG_FILTER_VF.get(filter_name)
    if not vf:
        msg = f"No FFmpeg mapping for filter '{filter_name}'"
        logs.append(f"[executor:video_frame] {msg}")
        return {"type": "video_frame", "intent": f"apply_{filter_name}_filter",
                "filter": filter_name, "asset_paths": [], "applied": False, "note": msg}

    if not _has_ffmpeg():
        msg = "FFmpeg not installed — cannot apply color filter to scene MP4"
        logs.append(f"[executor:video_frame] {msg}")
        return {"type": "video_frame", "intent": f"apply_{filter_name}_filter",
                "filter": filter_name, "asset_paths": [], "applied": False, "note": msg}

    video_dir = _OUTPUTS_DIR / "video"
    if not video_dir.exists():
        msg = f"{video_dir} does not exist — run Phase 3 first"
        logs.append(f"[executor:video_frame] {msg}")
        return {"type": "video_frame", "intent": f"apply_{filter_name}_filter",
                "filter": filter_name, "asset_paths": [], "applied": False, "note": msg}

    # Resolve which scene MP4(s) to filter.
    if scope.startswith("scene:"):
        sid = scope.split(":", 1)[1].strip()
        sid_padded = sid.zfill(2)
        prefix = f"scene_{sid_padded}_"
    else:
        prefix = "scene_"

    all_mp4s = [p for p in video_dir.glob(f"{prefix}*.mp4")
                if f"_{filter_name}" not in p.stem]  # don't re-filter our own outputs

    # Group by scene_id and prefer the most-processed variant per scene.
    by_scene: dict[str, Path] = {}
    def _rank(p: Path) -> int:
        n = p.name
        if "_synced_aud" in n: return 3
        if "_synced"     in n: return 2
        return 1
    for p in all_mp4s:
        # Extract scene_NN prefix (first two underscore-separated tokens)
        parts = p.stem.split("_")
        key = "_".join(parts[:2]) if len(parts) >= 2 else p.stem
        if key not in by_scene or _rank(p) > _rank(by_scene[key]):
            by_scene[key] = p

    sources = sorted(by_scene.values())
    if not sources:
        msg = f"No '{prefix}*.mp4' found in {video_dir}"
        logs.append(f"[executor:video_frame] {msg}")
        return {"type": "video_frame", "intent": f"apply_{filter_name}_filter",
                "filter": filter_name, "asset_paths": [], "applied": False, "note": msg}

    asset_paths: list[str] = []
    for src in sources:
        out = src.with_name(f"{src.stem}_{filter_name}{src.suffix}")
        cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-c:a", "copy", str(out)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                asset_paths.append(str(out))
                logs.append(f"[executor:video_frame] ✓ {out.name}")
            else:
                logs.append(f"[executor:video_frame] ✗ ffmpeg failed for {src.name}: "
                            f"{proc.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logs.append(f"[executor:video_frame] ✗ ffmpeg timeout on {src.name}")

    return {"type": "video_frame", "intent": f"apply_{filter_name}_filter",
            "filter": filter_name, "videos_edited": len(asset_paths),
            "asset_paths": asset_paths, "applied": bool(asset_paths)}


async def _regenerate_character_image(scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    """Call Writers Room MCP → generate_character_image for the target character."""
    character = scope.replace("character:", "").strip() if scope.startswith("character:") else scope
    style     = parameters.get("style", "photorealistic")

    # Find character appearance from state or disk
    appearance = parameters.get("appearance", "")
    if not appearance:
        chars = state.get("characters", [])
        if not chars:
            manifest = _PROJECT_ROOT / "outputs" / "character_db.json"
            if manifest.exists():
                try:
                    data  = json.loads(manifest.read_text())
                    chars = data.get("characters", data.get("character_db", []))
                    if isinstance(chars, dict):
                        chars = list(chars.values())
                except Exception:
                    pass
        for c in chars:
            name = c.get("name") or c.get("character_name", "")
            if character.lower() in name.lower():
                appearance = c.get("appearance", c.get("visual_description", ""))
                break

    if not appearance:
        appearance = f"{character}, protagonist, detailed character design"

    writers_up = await _mcp_available("writers")
    if writers_up:
        logs.append(f"[executor:video_frame] MCP → generate_character_image char={character}")
        res = await _call_writers("generate_character_image", logs,
                                  character_name=character,
                                  appearance=appearance,
                                  style=style)
        if res and res.get("status") == "success":
            fpath = res.get("file", "")
            logs.append(f"[executor:video_frame] ✓ Image generated → {Path(fpath).name if fpath else '?'}")
            return {"type": "video_frame", "intent": "change_character_design",
                    "character": character, "provider": res.get("provider", "?"),
                    "asset_paths": [fpath] if fpath else [], "applied": True, "result": res}
        logs.append(f"[executor:video_frame] MCP failed: {res}")

    # Fallback: Pollinations AI directly
    logs.append(f"[executor:video_frame] Writers MCP offline — calling Pollinations directly")
    return await _pollinations_image(character, appearance, style, logs)


async def _pollinations_image(character: str, appearance: str, style: str, logs: list[str]) -> dict:
    import urllib.parse
    style_suffix = {
        "photorealistic": "cinematic portrait, professional photography, 8k, sharp focus",
        "animated":       "2D animation style, vibrant colors, expressive, clean lines",
        "painterly":      "oil painting, dramatic lighting, artistic brushwork",
    }.get(style, "cinematic portrait, 8k")

    prompt  = f"Character portrait of {character}: {appearance}. {style_suffix}. White background."
    encoded = urllib.parse.quote(prompt)
    url     = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&enhance=true"

    safe_name = re.sub(r"[^a-z0-9]", "_", character.lower())
    out_path  = _IMAGE_DIR / f"{safe_name}_edit.png"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.get(url)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
            out_path.write_bytes(r.content)
            logs.append(f"[executor:video_frame] ✓ Pollinations image → {out_path.name}")
            return {"type": "video_frame", "intent": "change_character_design",
                    "character": character, "provider": "pollinations",
                    "asset_paths": [str(out_path)], "applied": True}
        logs.append(f"[executor:video_frame] Pollinations HTTP {r.status_code}")
    except Exception as e:
        logs.append(f"[executor:video_frame] Pollinations failed: {e}")

    return {"type": "video_frame", "intent": "change_character_design",
            "character": character, "asset_paths": [], "applied": False,
            "note": "Both MCP and Pollinations failed. Check network connectivity."}


async def _regenerate_scene_image(intent: str, scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    """Re-generate a scene image via Studio Floor MCP query_stock_image, or Pollinations fallback."""
    scene_id  = int(scope.replace("scene:", "")) if scope.startswith("scene:") else 1
    style     = parameters.get("style", "cinematic")

    # Get scene visual description from state
    description = parameters.get("description", "")
    if not description:
        scenes = state.get("scenes", [])
        for sc in scenes:
            if str(sc.get("scene_id")) == str(scene_id):
                description = sc.get("scene_visual_cue", sc.get("action_description", ""))
                break
    if not description:
        description = f"Scene {scene_id} — cinematic establishing shot"

    if style != "cinematic":
        description = f"{description}, {style} style"

    studio_up = await _mcp_available("studio")
    if studio_up:
        logs.append(f"[executor:video_frame] MCP → query_stock_image scene={scene_id}")
        res = await _call_studio("query_stock_image", logs, scene_description=description)
        if res and res.get("status") == "success":
            fpath = res.get("file", "")
            logs.append(f"[executor:video_frame] ✓ Scene image → {Path(fpath).name if fpath else '?'}")
            return {"type": "video_frame", "intent": intent, "scene_id": scene_id,
                    "asset_paths": [fpath] if fpath else [], "applied": True}

    # Fallback: Pollinations
    logs.append("[executor:video_frame] Falling back to Pollinations for scene image")
    import urllib.parse
    prompt  = f"Cinematic scene: {description}. Movie still, professional cinematography, high quality."
    encoded = urllib.parse.quote(prompt)
    url     = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&nologo=true"
    out_path = _IMAGE_DIR / f"scene_{scene_id:02d}_regen_edit.png"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.get(url)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
            out_path.write_bytes(r.content)
            logs.append(f"[executor:video_frame] ✓ Pollinations scene image → {out_path.name}")
            return {"type": "video_frame", "intent": intent, "scene_id": scene_id,
                    "asset_paths": [str(out_path)], "applied": True, "provider": "pollinations"}
    except Exception as e:
        logs.append(f"[executor:video_frame] Pollinations scene image failed: {e}")

    return {"type": "video_frame", "intent": intent, "scene_id": scene_id,
            "asset_paths": [], "applied": False}


# ─── Full-video edits ─────────────────────────────────────────────────────────

def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


async def _handle_video_edit(intent: str, scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    logs.append(f"[executor:video] intent={intent}")

    # ── recompose_video: re-stitch the final MP4 from per-scene clips, picking
    # the newest variant per scene so edits like '_darker' get picked up.
    if intent == "recompose_video":
        return await _recompose_final_video(state, logs)

    # ── Per-scene retime: when scoped to one scene, adjust THAT clip and write
    # scene_NN_..._<fast|slow>.mp4 so the next recompose stitches it in place
    # (mirrors the per-scene colour-filter path). scope='all' falls through to
    # the whole-video retime below.
    if intent in ("speed_up_scene", "slow_down_scene") and scope.startswith("scene:") and _scene_mp4s_exist(scope):
        return await _apply_scene_speed(intent, scope, parameters, logs)

    video_candidates = sorted((_OUTPUTS_DIR / "final").glob("*.mp4")) if (_OUTPUTS_DIR / "final").exists() else []
    if not video_candidates:
        video_candidates = list(_OUTPUTS_DIR.glob("*.mp4"))
    if not video_candidates:
        logs.append("[executor:video] No MP4 found in outputs")
        return {"type": "video", "intent": intent, "asset_paths": [], "applied": False,
                "note": "No MP4 found. Run Phase 3 first."}

    source   = video_candidates[0]
    out_dir  = _OUTPUTS_DIR / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"final_output_{intent}.mp4"

    if not _has_ffmpeg():
        logs.append("[executor:video] FFmpeg not installed")
        return {"type": "video", "intent": intent, "asset_paths": [], "applied": False,
                "note": "FFmpeg not installed. Install with: apt install ffmpeg"}

    cmd: list[str] = []

    if intent in ("speed_up_scene", "slow_down_scene"):
        speed  = float(parameters.get("factor", 1.5 if intent == "speed_up_scene" else 0.75))
        speed  = max(0.25, min(4.0, speed))
        vf     = f"setpts={1/speed:.4f}*PTS"
        af     = f"atempo={min(2.0, max(0.5, speed))}"
        cmd    = ["ffmpeg", "-y", "-i", str(source), "-filter:v", vf, "-filter:a", af, str(out_path)]

    elif intent == "remove_subtitle":
        cmd = ["ffmpeg", "-y", "-i", str(source), "-vf", "subtitles=/dev/null", "-c:a", "copy", str(out_path)]
        # Simpler: just copy (subtitles are usually not burned in unless explicitly added)
        cmd = ["ffmpeg", "-y", "-i", str(source), "-c", "copy", "-sn", str(out_path)]

    elif intent == "add_subtitle":
        srt = str(_OUTPUTS_DIR / "subtitles.srt")
        if not Path(srt).exists():
            cmd = ["ffmpeg", "-y", "-i", str(source), "-c", "copy", str(out_path)]
            logs.append("[executor:video] subtitles.srt not found — copying video unchanged")
        else:
            cmd = ["ffmpeg", "-y", "-i", str(source), "-vf", f"subtitles={srt}", "-c:a", "copy", str(out_path)]

    else:  # recompose_video / generic
        cmd = ["ffmpeg", "-y", "-i", str(source), "-c", "copy", str(out_path)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            logs.append(f"[executor:video] ✓ FFmpeg → {out_path.name}")
            return {"type": "video", "intent": intent, "source": str(source),
                    "output": str(out_path), "asset_paths": [str(out_path)], "applied": True}
        else:
            logs.append(f"[executor:video] FFmpeg error: {proc.stderr[:300]}")
            return {"type": "video", "intent": intent, "asset_paths": [], "applied": False,
                    "ffmpeg_error": proc.stderr[:300]}
    except subprocess.TimeoutExpired:
        logs.append("[executor:video] FFmpeg timed out")
        return {"type": "video", "intent": intent, "asset_paths": [], "applied": False, "note": "FFmpeg timed out"}


async def _apply_scene_speed(intent: str, scope: str, parameters: dict, logs: list[str]) -> dict:
    """Retime ONE scene's clip → outputs/video/scene_NN_..._<fast|slow>.mp4.

    `setpts` requires a video re-encode (can't -c copy when changing PTS); audio is
    retimed with a chained `atempo` (each link is limited to 0.5..2.0). The output
    keeps the source's tier tag (e.g. ..._synced_aud_fast.mp4) so `_pick_scene_clip`
    picks it as the newest variant on the next recompose.
    """
    if not _has_ffmpeg():
        logs.append("[executor:video] FFmpeg not installed")
        return {"type": "video", "intent": intent, "asset_paths": [], "applied": False,
                "note": "FFmpeg not installed"}

    video_dir = _OUTPUTS_DIR / "video"
    sid = scope.split(":", 1)[1].strip()
    try:
        scene_id = int(sid)
    except ValueError:
        return {"type": "video", "intent": intent, "asset_paths": [], "applied": False,
                "note": f"bad scene scope {scope!r}"}

    src = _pick_scene_clip(video_dir, scene_id)
    if src is None:
        return {"type": "video", "intent": intent, "asset_paths": [], "applied": False,
                "note": f"no clip for scene {scene_id} in {video_dir} — run Phase 3 first"}

    speed = float(parameters.get("factor", 1.5 if intent == "speed_up_scene" else 0.75))
    speed = max(0.25, min(4.0, speed))
    tag   = "fast" if speed >= 1.0 else "slow"
    out   = src.with_name(f"{src.stem}_{tag}{src.suffix}")

    # atempo on an input with no audio stream is a hard error — probe first.
    has_audio = bool(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True).stdout.strip())

    cmd = ["ffmpeg", "-y", "-i", str(src),
           "-filter:v", f"setpts={1.0 / speed:.4f}*PTS",
           "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if has_audio:
        af_links, rem = [], speed
        while rem > 2.0:
            af_links.append("atempo=2.0"); rem /= 2.0
        while rem < 0.5:
            af_links.append("atempo=0.5"); rem /= 0.5
        af_links.append(f"atempo={rem:.4f}")
        cmd += ["-filter:a", ",".join(af_links), "-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd.append(str(out))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            logs.append(f"[executor:video] ✓ scene {scene_id} retimed x{speed:g} → {out.name}")
            return {"type": "video", "intent": intent, "scene_id": scene_id, "speed": speed,
                    "source": str(src), "output": str(out), "asset_paths": [str(out)],
                    "applied": True,
                    "note": "run 'recompose the video' to stitch the retimed clip into the final cut"}
        logs.append(f"[executor:video] ✗ ffmpeg retiming scene {scene_id}: {proc.stderr[:300]}")
        return {"type": "video", "intent": intent, "asset_paths": [], "applied": False,
                "ffmpeg_error": proc.stderr[:300]}
    except subprocess.TimeoutExpired:
        logs.append(f"[executor:video] ✗ ffmpeg timed out retiming scene {scene_id}")
        return {"type": "video", "intent": intent, "asset_paths": [], "applied": False,
                "note": "FFmpeg timed out"}


# ─── Recompose helper ─────────────────────────────────────────────────────────

def _pick_scene_clip(video_dir: Path, scene_id: int) -> Path | None:
    """
    Pick the best per-scene MP4 for `recompose_video`.
    Tier:  _synced_aud > _synced > plain.
    Within tier: newest mtime wins, so a freshly-produced _darker / _sepia
    variant beats the original `_synced_aud.mp4` automatically.
    """
    sid_padded = str(scene_id).zfill(2)
    cands = list(video_dir.glob(f"scene_{sid_padded}_*.mp4"))
    if not cands:
        return None

    def _tier(p: Path) -> int:
        n = p.name
        if "_synced_aud" in n: return 3
        if "_synced"     in n: return 2
        return 1

    cands.sort(key=lambda p: (_tier(p), p.stat().st_mtime), reverse=True)
    return cands[0]


async def _recompose_final_video(state: dict, logs: list[str]) -> dict:
    """
    Stitch outputs/video/scene_NN_*.mp4 (newest variant per scene) into
    outputs/final/final_output.mp4 — without re-running phase 3.
    """
    if not _has_ffmpeg():
        msg = "FFmpeg not installed — cannot recompose final video"
        logs.append(f"[executor:video] {msg}")
        return {"type": "video", "intent": "recompose_video",
                "asset_paths": [], "applied": False, "note": msg}

    video_dir = _OUTPUTS_DIR / "video"
    if not video_dir.exists():
        msg = f"{video_dir} does not exist — run Phase 3 first"
        logs.append(f"[executor:video] {msg}")
        return {"type": "video", "intent": "recompose_video",
                "asset_paths": [], "applied": False, "note": msg}

    # Resolve scene order from manifest (fall back to filename ordering).
    scenes = state.get("scenes", [])
    if not scenes:
        manifest = _PROJECT_ROOT / "outputs" / "scene_manifest.json"
        if manifest.exists():
            try:
                scenes = json.loads(manifest.read_text(encoding="utf-8")).get("scenes", [])
            except Exception:
                pass

    scene_ids: list[int] = []
    if scenes:
        for s in sorted(scenes, key=lambda x: x.get("scene_id", 0)):
            try:
                scene_ids.append(int(s.get("scene_id")))
            except (TypeError, ValueError):
                continue
    else:
        # Fallback: derive from filenames
        seen: set[int] = set()
        for p in sorted(video_dir.glob("scene_*.mp4")):
            m = re.match(r"scene_(\d+)_", p.name)
            if m:
                seen.add(int(m.group(1)))
        scene_ids = sorted(seen)

    if not scene_ids:
        msg = "No scenes found in manifest or outputs/video/"
        logs.append(f"[executor:video] {msg}")
        return {"type": "video", "intent": "recompose_video",
                "asset_paths": [], "applied": False, "note": msg}

    scene_files: list[Path] = []
    for sid in scene_ids:
        clip = _pick_scene_clip(video_dir, sid)
        if clip is None:
            logs.append(f"[executor:video] ⚠ no MP4 for scene {sid:02d} — skipping")
            continue
        scene_files.append(clip)
        logs.append(f"[executor:video] scene {sid:02d} → {clip.name}")

    if not scene_files:
        msg = "No usable per-scene MP4s found"
        logs.append(f"[executor:video] {msg}")
        return {"type": "video", "intent": "recompose_video",
                "asset_paths": [], "applied": False, "note": msg}

    out_dir = _OUTPUTS_DIR / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_output.mp4"

    # Probe for audio streams (face-swap clips may have none).
    has_audio: list[bool] = []
    for f in scene_files:
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(f)],
                capture_output=True, text=True, timeout=20,
            )
            has_audio.append(bool(probe.stdout.strip()))
        except Exception:
            has_audio.append(False)

    # Normalise every clip to 1280x720@24fps then concat — robust to mixed inputs.
    TW, TH, TFPS = 1280, 720, 24
    n = len(scene_files)
    input_args: list[str] = []
    for f in scene_files:
        input_args += ["-i", str(f)]

    vparts = [
        f"[{i}:v]scale={TW}:{TH}:force_original_aspect_ratio=decrease,"
        f"pad={TW}:{TH}:(ow-iw)/2:(oh-ih)/2,fps={TFPS}[v{i}];"
        for i in range(n)
    ]
    vparts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout];")

    aparts: list[str] = []
    for i, f in enumerate(scene_files):
        if has_audio[i]:
            aparts.append(f"[{i}:a:0]aresample=44100[a{i}];")
        else:
            try:
                dur_proc = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(f)],
                    capture_output=True, text=True, timeout=20,
                )
                dur = float(dur_proc.stdout.strip() or "1.0")
            except Exception:
                dur = 1.0
            aparts.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{dur:.3f}[a{i}];")
    aparts.append("".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]")

    full_filter = "".join(vparts) + "".join(aparts)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *input_args,
        "-filter_complex", full_filter,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        logs.append("[executor:video] FFmpeg timed out during recompose")
        return {"type": "video", "intent": "recompose_video",
                "asset_paths": [], "applied": False, "note": "FFmpeg timed out"}

    if proc.returncode != 0:
        logs.append(f"[executor:video] FFmpeg error: {proc.stderr[:400]}")
        return {"type": "video", "intent": "recompose_video",
                "asset_paths": [], "applied": False,
                "ffmpeg_error": proc.stderr[:400]}

    logs.append(f"[executor:video] ✓ Recomposed → {out_path}")
    return {"type": "video", "intent": "recompose_video",
            "scenes_used": [str(p) for p in scene_files],
            "output": str(out_path),
            "asset_paths": [str(out_path)], "applied": True}


# ─── Script edits ─────────────────────────────────────────────────────────────

async def _handle_script_edit(intent: str, scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    logs.append(f"[executor:script] intent={intent} scope={scope}")

    if intent == "regenerate_script":
        return await _regenerate_script(parameters, state, logs)

    if intent == "change_scene_dialogue":
        return await _rewrite_dialogue(scope, parameters, state, logs)

    if intent == "change_scene_tone":
        return _change_tone(scope, parameters, state, logs)

    logs.append(f"[executor:script] Unhandled: {intent}")
    return {"type": "script", "intent": intent, "asset_paths": [], "applied": False}


async def _regenerate_script(parameters: dict, state: dict, logs: list[str]) -> dict:
    """
    Call Writers Room MCP → generate_script_segment to produce a new script,
    then save it via save_scene_manifest. Falls back to reporting if MCP offline.
    """
    prompt     = parameters.get("prompt", state.get("user_prompt", ""))
    num_scenes = int(parameters.get("num_scenes", 3))

    if not prompt:
        # Try reading original prompt from disk summary
        summary_path = _PROJECT_ROOT / "data" / "outputs" / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                prompt  = summary.get("prompt", "")
            except Exception:
                pass

    if not prompt:
        return {"type": "script", "intent": "regenerate_script", "asset_paths": [], "applied": False,
                "note": "No prompt found. Pass 'prompt' in parameters or run Phase 1 first."}

    writers_up = await _mcp_available("writers")
    if not writers_up:
        logs.append("[executor:script] Writers MCP offline — cannot regenerate script without it")
        return {"type": "script", "intent": "regenerate_script", "asset_paths": [], "applied": False,
                "note": "Writers Room MCP server (port 8100) is not running. Start it with: python mcp_servers/writers_room_server.py"}

    logs.append(f"[executor:script] MCP → generate_script_segment prompt='{prompt[:60]}…'")
    res = await _call_writers("generate_script_segment", logs, prompt=prompt, num_scenes=num_scenes)

    if not res:
        return {"type": "script", "intent": "regenerate_script", "asset_paths": [], "applied": False,
                "note": "generate_script_segment returned no result"}

    # Save via MCP
    logs.append("[executor:script] MCP → save_scene_manifest")
    save_res = await _call_writers("save_scene_manifest", logs, script_json=json.dumps(res))
    if save_res and save_res.get("status") == "saved":
        manifest_path = save_res.get("path", "")
        logs.append(f"[executor:script] ✓ New script saved → {manifest_path}")
        return {"type": "script", "intent": "regenerate_script",
                "scene_count": save_res.get("scene_count", 0),
                "title": save_res.get("title", ""),
                "asset_paths": [manifest_path] if manifest_path else [],
                "applied": True, "result": res}

    # Fallback: save to disk directly
    out = _PROJECT_ROOT / "outputs" / "scene_manifest_regen.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    logs.append(f"[executor:script] Saved regenerated script to {out.name}")
    return {"type": "script", "intent": "regenerate_script",
            "asset_paths": [str(out)], "applied": True, "result": res}


async def _rewrite_dialogue(scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    """Rewrite dialogue for a scene using the Writers Room LLM, or patch it directly."""
    scene_id     = int(scope.replace("scene:", "")) if scope.startswith("scene:") else None
    new_dialogue = parameters.get("dialogue", "")
    character    = parameters.get("character", "")

    scenes = state.get("scenes", [])
    if not scenes:
        manifest = _PROJECT_ROOT / "outputs" / "scene_manifest.json"
        if manifest.exists():
            try:
                scenes = json.loads(manifest.read_text()).get("scenes", [])
            except Exception:
                pass

    patched = False
    for sc in scenes:
        if scene_id is None or str(sc.get("scene_id")) == str(scene_id):
            if new_dialogue:
                if character:
                    for dl in sc.get("dialogue", []):
                        if character.lower() in dl.get("speaker", "").lower():
                            dl["line"] = new_dialogue
                            patched = True
                else:
                    if sc.get("dialogue"):
                        sc["dialogue"][0]["line"] = new_dialogue
                        patched = True
            sc.setdefault("edit_notes", []).append(
                f"dialogue edited: {parameters}"
            )

    if patched:
        state["scenes"] = scenes
        out = _PROJECT_ROOT / "outputs" / "scene_manifest_edited.json"
        data = json.loads(out.read_text()) if out.exists() else {}
        data["scenes"] = scenes
        out.write_text(json.dumps(data, indent=2, default=str))
        logs.append(f"[executor:script] ✓ Dialogue patched and saved to {out.name}")
        return {"type": "script", "intent": "change_scene_dialogue",
                "scene_id": scene_id, "asset_paths": [str(out)], "applied": True}

    return {"type": "script", "intent": "change_scene_dialogue",
            "asset_paths": [], "applied": False,
            "note": f"Scene {scene_id} not found or dialogue parameter missing."}


def _change_tone(scope: str, parameters: dict, state: dict, logs: list[str]) -> dict:
    """Patch the tone field of a scene in the manifest."""
    scene_id = int(scope.replace("scene:", "")) if scope.startswith("scene:") else None
    tone     = parameters.get("tone", "neutral")

    scenes = state.get("scenes", [])
    if not scenes:
        manifest = _PROJECT_ROOT / "outputs" / "scene_manifest.json"
        if manifest.exists():
            try:
                scenes = json.loads(manifest.read_text()).get("scenes", [])
            except Exception:
                pass

    patched = False
    for sc in scenes:
        if scene_id is None or str(sc.get("scene_id")) == str(scene_id):
            sc["tone"] = tone
            patched = True

    if patched:
        state["scenes"] = scenes
        out = _PROJECT_ROOT / "outputs" / "scene_manifest_edited.json"
        data = json.loads(out.read_text()) if out.exists() else {}
        data["scenes"] = scenes
        out.write_text(json.dumps(data, indent=2, default=str))
        logs.append(f"[executor:script] ✓ Tone → '{tone}' for scene {scene_id}, saved to {out.name}")
        return {"type": "script", "intent": "change_scene_tone",
                "tone": tone, "scene_id": scene_id,
                "asset_paths": [str(out)], "applied": True}

    return {"type": "script", "intent": "change_scene_tone",
            "asset_paths": [], "applied": False,
            "note": f"Scene {scene_id} not found in state."}