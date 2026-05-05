# ============================================================
# nodes.py — Phase 2 LangGraph node implementations (audio only)
#
# Contains:
#   • scene_parser_node      — Loads Phase 1 outputs (script.json preferred)
#   • voice_synth_node       — TTS + BGM + mix per scene
#   • audio_finalizer_node   — Aggregates audio results, writes summary
#
# Video nodes (video_gen, face_swap, lip_sync, finalizer) live in
# Phase 3 — see agents/video_agent/nodes.py.
#
# All MCP tool calls go to studio_floor_server (port 8200).
# ============================================================

import asyncio
import json
import os
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.audio_agent.state import AgentState

# ─── MCP client ──────────────────────────────────────────────────────────────

MCP_CONFIG = {
    "studio_floor": {
        "url":       "http://localhost:8200/mcp",
        "transport": "streamable_http",
        "httpx_client_factory": None,  # set below
    }
}

# Override default HTTP timeout (audio gen + mix can be slow)
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


# ─── Voice profile derivation ─────────────────────────────────────────────────

_EMOTION_BY_TRAIT = {
    "determined": "confident", "resourceful": "confident",
    "intelligent": "calm", "mysterious": "calm",
    "nervous": "anxious", "manipulative": "anxious",
    "sad": "sad", "enigmatic": "calm", "curious": "calm",
    "independent": "confident", "empathetic": "warm",
    "suspicious": "anxious",
}
_TLD_POOL = ["com", "co.uk", "com.au", "co.in", "ca", "ie"]


def _build_voice_profile(character_name: str, char_db: dict) -> dict:
    """Derive a TTS voice profile for `character_name` from the character DB."""
    chars = (char_db or {}).get("characters", []) if isinstance(char_db, dict) else []
    found = next(
        (c for c in chars
         if isinstance(c, dict)
         and (c.get("name") or "").strip().lower() == character_name.strip().lower()),
        {},
    )

    # Prefer voice_personality from the unified script.json (Phase 1 spec)
    vp = found.get("voice_personality", {}) if isinstance(found, dict) else {}
    if vp:
        emotion = vp.get("tone", "neutral")
        accent  = vp.get("accent", "american").lower()
        accent_to_tld = {"american": "com", "british": "co.uk",
                         "australian": "com.au", "indian": "co.in",
                         "canadian": "ca", "irish": "ie"}
        tld = accent_to_tld.get(accent, "com")
        return {
            "character_id": found.get("character_id", ""),
            "character":    character_name,
            "voice_id":     f"voice_{character_name.lower().replace(' ', '_')}",
            "tld":          tld,
            "emotion":      emotion,
            "traits":       found.get("personality_traits", [])[:3],
        }

    # Fallback — derive from traits + gender
    idx = sum(ord(c) for c in character_name) % len(_TLD_POOL)
    tld = _TLD_POOL[idx]

    traits = [t.lower() for t in found.get("personality_traits", [])]
    emotion = "neutral"
    for t in traits:
        if t in _EMOTION_BY_TRAIT:
            emotion = _EMOTION_BY_TRAIT[t]
            break

    gender = (found.get("gender") or "").lower()
    if emotion == "neutral" and gender:
        emotion = "warm" if gender == "female" else "calm"

    return {
        "character_id": found.get("character_id", ""),
        "character":    character_name,
        "voice_id":     f"voice_{character_name.lower().replace(' ', '_')}",
        "tld":          tld,
        "emotion":      emotion,
        "traits":       traits[:3],
    }


# ─── Node 1: Scene Parser ─────────────────────────────────────────────────────

async def scene_parser_node(state: AgentState) -> dict:
    """
    PDF §5.1 + spec §4 Phase 2.
    Loads Phase 1's UNIFIED script.json (preferred) or scene_manifest.json
    (fallback). Also loads characters separately if needed.
    """
    print("\n[Scene Parser] Loading Phase 1 outputs...")

    manifest_path = Path(state.get("scene_manifest_path", ""))
    char_db_path  = Path(state.get("character_db_path", ""))
    # Look for spec-compliant script.json next to the original manifest
    script_path   = Path(state.get("script_path", "")) if state.get("script_path") \
                    else (manifest_path.parent / "script.json")

    manifest = None
    char_db  = None
    used_unified = False

    if script_path.exists():
        try:
            unified = json.loads(script_path.read_text(encoding="utf-8"))
            manifest = {
                "title":  unified.get("story", {}).get("title", "Untitled"),
                "genre":  unified.get("story", {}).get("genre", ""),
                "scenes": unified.get("scenes", []),
            }
            char_db = {"characters": unified.get("characters", [])}
            used_unified = True
            print(f"[Scene Parser] Source: script.json (spec-compliant)")
        except Exception as e:
            print(f"[Scene Parser] script.json present but unreadable: {e}")

    if manifest is None:
        if not manifest_path.exists():
            return {
                "status":       "failed",
                "errors":       [f"Neither script.json nor scene_manifest.json found near {manifest_path}"],
                "current_node": "scene_parser_node",
            }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        char_db  = (json.loads(char_db_path.read_text(encoding="utf-8"))
                    if char_db_path.exists() else {"characters": []})
        print(f"[Scene Parser] Source: scene_manifest.json (legacy fallback)")

    print(f"[Scene Parser] Title: {manifest.get('title')}  |  Scenes: {len(manifest.get('scenes', []))}")
    print(f"[Scene Parser] Characters loaded: {len(char_db.get('characters', []))}")

    raw = await _call_tool("get_task_graph", scene_manifest_json=json.dumps(manifest))
    parsed = _safe_parse(raw)
    task_graph = parsed.get("task_graph", {})

    scenes = manifest.get("scenes", [])
    print(f"[Scene Parser] Task graph generated — {len(scenes)} parallel scene workloads")

    completed_ids: list = []
    if state.get("resume_from_memory"):
        mem_raw = await _call_tool("query_memory", category="audio_track", limit=100)
        mem = _safe_parse(mem_raw)
        for entry in mem.get("entries", []):
            try:
                payload = json.loads(entry.get("value", "{}"))
                sid = payload.get("scene_id")
                files = payload.get("files", [])
                if sid is not None and files and all(Path(f).exists() for f in files):
                    completed_ids.append(sid)
            except Exception:
                continue
        if completed_ids:
            print(f"[Scene Parser] Resuming — {len(completed_ids)} already complete: {completed_ids}")

    await _call_tool(
        "commit_memory",
        key=f"audio_task_graph_{manifest.get('title', 'untitled').replace(' ', '_')}",
        value=json.dumps({"scenes": len(scenes), "title": manifest.get("title"),
                          "source": "script.json" if used_unified else "scene_manifest.json"}),
        category="task_graph",
    )

    return {
        "scene_manifest":      manifest,
        "character_db":        char_db,
        "task_graph":          task_graph,
        "scenes":              scenes,
        "completed_scene_ids": completed_ids,
        "status":              "processing",
        "current_node":        "scene_parser_node",
    }


# ─── Node 2: Voice Synth + BGM + Mix ──────────────────────────────────────────

async def voice_synth_node(state: AgentState) -> dict:
    """
    PDF §5.2 Voice Synthesis Agent + PDF §4 Phase 2 BGM.

    For each scene this node:
      1. Builds a per-character voice profile from the character DB
      2. Synthesizes each dialogue line via TTS with that profile
      3. Generates mood-appropriate BGM (from scene.tone)
      4. Pre-mixes dialogue + BGM → scene_NN_full.wav

    Silent-dialogue handling:
      Even when the scene has no dialogue, we still generate BGM matching the
      scene's tone for the scene's intended duration, and we copy that BGM
      to scene_NN_full.wav so Phase 3's lip_sync_aligner has audio to mux.
    """
    scene = state.get("_current_scene", {})
    sid   = scene.get("scene_id")
    dialogue = scene.get("dialogue", [])
    char_db  = state.get("character_db", {})

    mood = scene.get("tone", "neutral") or "neutral"

    print(f"  [Voice Synth  | scene {sid}] {len(dialogue)} line(s), mood={mood}...")

    audio_files = []
    total_duration = 0.0
    profiles_used = {}

    for line_entry in dialogue:
        if not isinstance(line_entry, dict):
            continue
        speaker = line_entry.get("speaker", "Unknown")
        line    = line_entry.get("line", "")
        if not line.strip():
            continue

        profile = _build_voice_profile(speaker, char_db)
        profiles_used[speaker] = profile

        raw = await _call_tool(
            "voice_cloning_synthesizer",
            scene_id=sid, character_name=speaker, dialogue_line=line,
            voice_profile=profile["voice_id"], emotion=profile["emotion"],
            tld=profile["tld"],
        )
        try:
            result = _safe_parse(raw)
            audio_files.append(result.get("file"))
            total_duration += result.get("duration_s", 0)
        except Exception as e:
            print(f"    [Voice Synth | scene {sid}] line failed: {e}")

    # ── Silent scene? Use the scene's planned duration so we still get BGM ──
    is_silent_scene = (total_duration == 0.0)
    if is_silent_scene:
        # Prefer scene.duration_s from Phase 1 spec output; otherwise estimate
        # from action description length (≈ 8 seconds per "beat").
        planned = float(scene.get("duration_s", 0) or 0)
        if planned <= 0:
            action_words = len((scene.get("action_description") or "").split())
            planned = max(5.0, min(20.0, action_words / 3.0 + 4.0))
        total_duration = planned
        print(f"  [Voice Synth  | scene {sid}] ⓘ Silent scene — planning {planned:.1f}s of BGM only")

    # Generate BGM (always, even for silent scenes)
    bgm_file = ""
    if total_duration > 0:
        try:
            print(f"  [BGM          | scene {sid}] Generating {mood} music ({total_duration:.1f}s)...")
            raw = await _call_tool(
                "generate_background_music",
                scene_id=sid, mood=mood, duration_s=total_duration,
            )
            bgm_result = _safe_parse(raw)
            bgm_file   = bgm_result.get("file", "")
            print(f"  [BGM          | scene {sid}] ✓ {bgm_file}")
        except Exception as e:
            print(f"  [BGM          | scene {sid}] failed: {e}")

    # ── Build scene_NN_full.wav (the audio track Phase 3 muxes) ──────────────
    if audio_files and bgm_file:
        # Normal case: dialogue + BGM mix
        try:
            print(f"  [Mix          | scene {sid}] dialogue + BGM (BGM @ -18dB)...")
            await _call_tool(
                "mix_audio_with_bgm",
                scene_id=sid, dialogue_files=audio_files,
                bgm_file=bgm_file, bgm_volume_db=-18.0,
            )
        except Exception as e:
            print(f"  [Mix          | scene {sid}] failed: {e}")
    elif is_silent_scene and bgm_file:
        # Silent scene: use BGM as the full track so Phase 3 still produces a video
        try:
            import shutil as _shutil
            target = Path(bgm_file).parent / f"scene_{sid:02d}_full.wav"
            _shutil.copyfile(bgm_file, target)
            print(f"  [Mix          | scene {sid}] silent → copied BGM as full track")
        except Exception as e:
            print(f"  [Mix          | scene {sid}] could not stage silent BGM: {e}")

    payload = {
        "scene_id":       sid,
        "files":          audio_files,
        "bgm_file":       bgm_file,
        "mood":           mood,
        "total_duration": round(total_duration, 2),
        "is_silent":      is_silent_scene,
        "voice_profiles": {k: v for k, v in profiles_used.items()},
    }

    await _call_tool(
        "commit_memory",
        key=f"audio_scene_{sid:02d}" if isinstance(sid, int) else f"audio_scene_{sid}",
        value=json.dumps(payload),
        category="audio_track",
    )

    profiles_summary = (", ".join(
        f"{k}({v['emotion']},{v['tld']})" for k, v in profiles_used.items()
    ) or "BGM-only")
    print(f"  [Voice Synth  | scene {sid}] ✓ {len(audio_files)} dialogue clip(s), "
          f"{total_duration:.1f}s  [{profiles_summary}]  mood={mood}")

    return {"audio_tracks": {str(sid): payload}}


# ─── Node 3: Audio Finalizer ──────────────────────────────────────────────────

async def audio_finalizer_node(state: AgentState) -> dict:
    """Aggregates all audio results, writes a summary log."""
    audios = state.get("audio_tracks", {})
    scenes = state.get("scenes", []) or []

    summary = {
        "title":        (state.get("scene_manifest") or {}).get("title", "Untitled"),
        "scenes_total": len(scenes),
        "audio_done":   len(audios),
        "scene_audio":  {sid: v.get("files") for sid, v in audios.items()},
        "bgm_files":    {sid: v.get("bgm_file") for sid, v in audios.items()},
        "moods":        {sid: v.get("mood") for sid, v in audios.items()},
        "durations":    {sid: v.get("total_duration") for sid, v in audios.items()},
    }

    # Write summary alongside the data/outputs/ tree
    logs_dir = Path(__file__).resolve().parent.parent.parent / "data" / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / "phase2_audio_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Audio Finalizer] Summary → {summary_path}")

    await _call_tool(
        "commit_memory", key="phase2_audio_summary",
        value=json.dumps(summary), category="summary",
    )

    return {
        "status":       "complete",
        "current_node": "audio_finalizer_node",
    }