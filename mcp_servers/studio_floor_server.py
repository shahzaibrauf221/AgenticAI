# ============================================================
# studio_floor_server.py — Phase 2 MCP server
# WITH ADDED BGM tools per PDF §4 Phase 2:
#   "background music selection or generation per scene mood"
#
# New tools:
#   • generate_background_music(mood, duration_s)
#       — procedurally generates mood-appropriate BGM via tone synthesis
#         (no model downloads, no API key required)
#   • mix_audio_with_bgm(dialogue_files, bgm_file, out_path)
#       — mixes BGM under dialogue at -18dB so dialogue stays clear
#
# All existing tools are unchanged. Existing pipeline still works.
# ============================================================

import base64
import hashlib
import json
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
import random
import requests
import shutil
import struct
import subprocess
import time
import uuid
import wave
from datetime import datetime

import chromadb
from mcp.server.fastmcp import FastMCP
from shared.utils.bytedance_video_client import (
    ByteDanceVideoClient,
    ByteDanceVideoClientError,
)

# ─── Load .env ────────────────────────────────────────────────────────────────
def _load_env(force_override: bool = False):
    for candidate in [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent / ".env",
    ]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key or not value:
                    continue
                # Prefer project .env for API keys/model endpoints to avoid stale
                # inherited shell values causing hard-to-debug 404/401 responses.
                if force_override or key.endswith("_API_KEY") or key.startswith("BYTEDANCE_"):
                    os.environ[key] = value
                elif key not in os.environ:
                    os.environ[key] = value
            break

_load_env(force_override=True)

# ─── Init ─────────────────────────────────────────────────────────────────────
mcp = FastMCP("studio_floor", port=8200)

BASE_DIR    = Path(__file__).parent.parent
MEMORY_DIR  = BASE_DIR / "memory"
OUTPUT_DIR  = BASE_DIR / "outputs"
AUDIO_DIR   = OUTPUT_DIR / "audio"
BGM_DIR     = OUTPUT_DIR / "bgm"
VIDEO_DIR   = OUTPUT_DIR / "video"
FRAMES_DIR  = OUTPUT_DIR / "frames"
SCENES_DIR  = OUTPUT_DIR / "raw_scenes"
LOGS_DIR    = BASE_DIR / "logs"

for d in (MEMORY_DIR, OUTPUT_DIR, AUDIO_DIR, BGM_DIR, VIDEO_DIR,
          FRAMES_DIR, SCENES_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHROMA_DIR = MEMORY_DIR / "chroma"

# ─── Vector Memory ────────────────────────────────────────────────────────────
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_memory = _chroma_client.get_or_create_collection(
    name="studio_floor_memory",
    metadata={"description": "Phase 2 resumability & intermediate outputs"},
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffprobe_duration(path) -> float:
    """Return audio/video duration in seconds via ffprobe; 0.0 on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).lower()


def _deterministic_color(seed: str) -> tuple:
    h = hashlib.md5(seed.encode()).hexdigest()
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _find_phase1_image_dir() -> list:
    candidates = []
    current = BASE_DIR.resolve()
    for _ in range(6):
        for sub in [current / "outputs" / "image_assets", current / "image_assets"]:
            if sub not in candidates and sub.exists():
                candidates.append(sub)
        if current.parent != current:
            try:
                for sibling in current.parent.iterdir():
                    if sibling.is_dir() and any(k in sibling.name.lower()
                                                 for k in ("writers-room", "writersroom",
                                                          "phase1", "22i-")):
                        target = sibling / "outputs" / "image_assets"
                        if target.exists() and target not in candidates:
                            candidates.append(target)
            except (OSError, PermissionError):
                pass
        if current.parent == current:
            break
        current = current.parent
    return candidates


def _find_character_png(character_name: str) -> str:
    safe   = _safe_name(character_name)
    tokens = [t for t in safe.split("_") if t]
    for search_dir in _find_phase1_image_dir():
        exact = search_dir / f"{safe}.png"
        if exact.exists():
            return str(exact.resolve())
        for png in search_dir.glob("*.png"):
            stem = png.stem.lower()
            if all(tok in stem for tok in tokens if len(tok) > 2):
                return str(png.resolve())
        if tokens:
            for png in search_dir.glob("*.png"):
                if png.stem.lower().startswith(tokens[0]):
                    return str(png.resolve())
    return ""


# ─── 1. Scene Parser Tool ─────────────────────────────────────────────────────

@mcp.tool()
def get_task_graph(scene_manifest_json: str) -> str:
    """Decompose scene manifest into a parallel task graph (PDF §3.2)."""
    try:
        manifest = json.loads(scene_manifest_json)
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"Invalid JSON: {e}"})

    scenes = manifest.get("scenes", [])
    task_graph = {
        "generated_at": datetime.utcnow().isoformat(),
        "title":        manifest.get("title", "Untitled"),
        "scene_count":  len(scenes),
        "tasks":        [],
    }

    for scene in scenes:
        sid = scene.get("scene_id")
        task_graph["tasks"].append({
            "scene_id":   sid,
            "location":   scene.get("location", ""),
            "characters": scene.get("characters", []),
            "branches": {
                "audio": {"tool": "voice_cloning_synthesizer", "status": "pending"},
                "bgm":   {"tool": "generate_background_music", "status": "pending"},
                "video": {
                    "steps": [
                        {"tool": "query_stock_footage",  "status": "pending"},
                        {"tool": "identity_validator",   "status": "pending"},
                        {"tool": "face_swapper",         "status": "pending"},
                    ],
                },
                "fusion": {"tool": "lip_sync_aligner", "depends_on": ["audio", "video"]},
            },
        })

    log_path = LOGS_DIR / f"task_graph_{int(time.time())}.json"
    log_path.write_text(json.dumps(task_graph, indent=2))

    return json.dumps({
        "status":     "ok",
        "scenes":     len(scenes),
        "log_path":   str(log_path),
        "task_graph": task_graph,
    })


# ─── 2. Voice Synthesis Tool ──────────────────────────────────────────────────

def _generate_silent_wav(path: Path, duration_s: float, sample_rate: int = 22050):
    n_frames = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", int(800 * math.sin(2 * math.pi * 220 * i / sample_rate)))
            for i in range(n_frames)
        )
        w.writeframes(frames)


_EMOTION_SLOW = {"sad": True, "anxious": False, "calm": False, "warm": False,
                 "confident": False, "neutral": False}
_EMOTION_TEMPO = {"sad": 0.92, "anxious": 1.10, "calm": 0.97, "warm": 1.00,
                  "confident": 1.05, "neutral": 1.00}

# ── edge-tts: gender-aware Neural voices per accent ──────────────────────────
# Multiple distinct voices per (accent, gender) so same-gender characters don't
# sound identical. Round-robin per character_name hash. All voice IDs verified
# against `edge_tts.list_voices()` — deprecated names (Davis, Tony, en-AU-William)
# removed.
_EDGE_VOICE_MAP: dict[tuple[str, str], list[str]] = {
    ("american",  "male"):   ["en-US-GuyNeural", "en-US-AndrewNeural",
                              "en-US-BrianNeural", "en-US-ChristopherNeural",
                              "en-US-EricNeural", "en-US-RogerNeural"],
    ("american",  "female"): ["en-US-JennyNeural", "en-US-AriaNeural",
                              "en-US-AvaNeural", "en-US-EmmaNeural",
                              "en-US-MichelleNeural"],
    ("british",   "male"):   ["en-GB-RyanNeural", "en-GB-ThomasNeural"],
    ("british",   "female"): ["en-GB-SoniaNeural", "en-GB-LibbyNeural"],
    ("australian","male"):   ["en-US-GuyNeural"],     # MS removed en-AU male voices
    ("australian","female"): ["en-AU-NatashaNeural"],
    ("indian",    "male"):   ["en-IN-PrabhatNeural"],
    ("indian",    "female"): ["en-IN-NeerjaNeural"],
    ("canadian",  "male"):   ["en-CA-LiamNeural"],
    ("canadian",  "female"): ["en-CA-ClaraNeural"],
    ("irish",     "male"):   ["en-IE-ConnorNeural"],
    ("irish",     "female"): ["en-IE-EmilyNeural"],
}

# tld → accent (inverse of audio_agent's accent_to_tld map)
_TLD_TO_ACCENT = {
    "com": "american", "co.uk": "british", "com.au": "australian",
    "co.in": "indian", "ca": "canadian", "ie": "irish",
}

# emotion → (rate%, pitch Hz) for edge-tts prosody.
# edge-tts requires signed values — "+0%" / "+0Hz", never bare "0%".
_EMOTION_PROSODY = {
    "sad":       ("-10%", "-30Hz"),
    "anxious":   ("+12%", "+20Hz"),
    "calm":      ("-3%",  "-5Hz"),
    "warm":      ("+0%",  "+5Hz"),
    "confident": ("+5%",  "+0Hz"),
    "neutral":   ("+0%",  "+0Hz"),
}


def _resolve_accent(accent: str, tld: str) -> str:
    a = (accent or "").strip().lower()
    if a in {"american", "british", "australian", "indian", "canadian", "irish"}:
        return a
    return _TLD_TO_ACCENT.get((tld or "").lower(), "american")


def _normalize_gender(gender: str) -> str:
    g = (gender or "").strip().lower()
    if g in {"male", "m", "man", "boy"}:
        return "male"
    if g in {"female", "f", "woman", "girl"}:
        return "female"
    return "female"  # default — better than silently shipping a male voice for unknowns


def _pick_edge_voice(accent: str, gender: str, character_name: str) -> str:
    accent = _resolve_accent(accent, "com") if accent else "american"
    gender = _normalize_gender(gender)
    voices = _EDGE_VOICE_MAP.get((accent, gender))
    if not voices:
        # accent unsupported — fall back to American of the requested gender
        voices = _EDGE_VOICE_MAP.get(("american", gender), ["en-US-JennyNeural"])
    idx = sum(ord(c) for c in (character_name or "")) % len(voices)
    return voices[idx]


def _tts_edge(
    text: str,
    path: Path,
    accent: str,
    gender: str,
    emotion: str,
    character_name: str,
) -> tuple[bool, str]:
    """Synthesize via Microsoft Edge TTS. Returns (ok, voice_id_used)."""
    try:
        import asyncio
        import edge_tts
    except ImportError as e:
        print(f"  [TTS/edge] import failed: {e}")
        return False, ""

    voice = _pick_edge_voice(accent, gender, character_name)
    rate, pitch = _EMOTION_PROSODY.get(emotion, ("0%", "+0Hz"))
    mp3_raw = path.with_suffix(".raw.mp3")

    async def _run():
        comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        await comm.save(str(mp3_raw))

    try:
        try:
            asyncio.run(asyncio.wait_for(_run(), timeout=30.0))
        except RuntimeError:
            # We're inside a running loop (rare in MCP sync tool, but handle it).
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(asyncio.wait_for(_run(), timeout=30.0))
            finally:
                loop.close()
    except Exception as e:
        print(f"  [TTS/edge] synth failed (voice={voice}, emotion={emotion}): {e}")
        return False, voice

    if not mp3_raw.exists() or mp3_raw.stat().st_size == 0:
        print(f"  [TTS/edge] empty output for voice={voice}")
        return False, voice

    if not _has_ffmpeg():
        target = path.with_suffix(".mp3")
        try:
            mp3_raw.rename(target)
        except OSError:
            pass
        return target.exists(), voice

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_raw),
             "-ar", "22050", "-ac", "1", str(path)],
            check=True, timeout=30,
        )
    except Exception as e:
        print(f"  [TTS/edge] ffmpeg re-encode failed: {e}")
        try:
            mp3_raw.rename(path.with_suffix(".mp3"))
        except OSError:
            return False, voice
        return path.with_suffix(".mp3").exists(), voice
    finally:
        mp3_raw.unlink(missing_ok=True)

    return path.exists(), voice


def _tts_gtts(text: str, path: Path, tld: str = "com", emotion: str = "neutral") -> bool:
    try:
        from gtts import gTTS
        slow = _EMOTION_SLOW.get(emotion, False)
        tts = gTTS(text=text, lang="en", tld=tld, slow=slow)
        mp3_raw = path.with_suffix(".raw.mp3")
        tts.save(str(mp3_raw))
        if not _has_ffmpeg():
            mp3_raw.rename(path.with_suffix(".mp3"))
            return path.with_suffix(".mp3").exists()
        tempo = _EMOTION_TEMPO.get(emotion, 1.0)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_raw),
             "-filter:a", f"atempo={tempo}", "-ar", "22050", "-ac", "1",
             str(path)],
            check=True, timeout=30,
        )
        mp3_raw.unlink(missing_ok=True)
        return path.exists()
    except Exception as e:
        print(f"  [TTS/gTTS] failed (tld={tld}, emotion={emotion}): {e}")
        return False


@mcp.tool()
def voice_cloning_synthesizer(
    scene_id: int, character_name: str, dialogue_line: str,
    voice_profile: str = "default", emotion: str = "neutral", tld: str = "com",
    gender: str = "", accent: str = "",
) -> str:
    """PDF §5.2 Voice Synthesis Agent — gender-aware via edge-tts, gTTS fallback."""
    safe_char = _safe_name(character_name)
    line_hash = hashlib.md5(dialogue_line.encode()).hexdigest()[:6]
    out_path  = AUDIO_DIR / f"scene_{scene_id:02d}_{safe_char}_{line_hash}.wav"

    resolved_accent = _resolve_accent(accent, tld)
    resolved_gender = _normalize_gender(gender)

    # Primary: edge-tts (true male/female Neural voices).
    ok, voice_used = _tts_edge(
        dialogue_line, out_path,
        accent=resolved_accent, gender=resolved_gender,
        emotion=emotion, character_name=character_name,
    )
    if ok:
        actual = out_path if out_path.exists() else out_path.with_suffix(".mp3")
        actual_dur = _ffprobe_duration(actual) if _has_ffmpeg() else 0.0
        if actual_dur <= 0:
            words      = max(1, len(dialogue_line.split()))
            actual_dur = words / 2.5
        return json.dumps({
            "status": "success", "provider": f"edge-tts[{voice_used}]",
            "scene_id": scene_id, "character": character_name,
            "emotion": emotion, "gender": resolved_gender, "accent": resolved_accent,
            "voice_profile": voice_profile, "voice_id": voice_used,
            "file": str(actual), "duration_s": round(actual_dur, 2),
            "text": dialogue_line,
        })

    # Fallback 1: gTTS (no gender, but at least audible speech).
    if _tts_gtts(dialogue_line, out_path, tld=tld, emotion=emotion):
        actual = out_path if out_path.exists() else out_path.with_suffix(".mp3")
        actual_dur = _ffprobe_duration(actual) if _has_ffmpeg() else 0.0
        if actual_dur <= 0:
            words      = max(1, len(dialogue_line.split()))
            tempo      = _EMOTION_TEMPO.get(emotion, 1.0)
            actual_dur = (words / 2.5) / tempo
        return json.dumps({
            "status": "success", "provider": f"gTTS[{tld}]",
            "scene_id": scene_id, "character": character_name,
            "emotion": emotion, "gender": resolved_gender, "accent": resolved_accent,
            "voice_profile": voice_profile, "tld": tld,
            "file": str(actual), "duration_s": round(actual_dur, 2),
            "text": dialogue_line,
            "note": "edge-tts unavailable — gTTS is gender-neutral",
        })

    # Fallback 2: silent placeholder so the pipeline never stalls.
    words = max(1, len(dialogue_line.split()))
    duration_s = words / 2.5
    _generate_silent_wav(out_path, duration_s)
    return json.dumps({
        "status": "fallback", "provider": "silent_wav",
        "scene_id": scene_id, "character": character_name,
        "file": str(out_path), "duration_s": round(duration_s, 2),
        "text": dialogue_line,
    })


# ─── *** NEW: BGM tools per PDF §4 Phase 2 *** ────────────────────────────────

# Mood → musical parameters (key, tempo BPM, voice palette)
_MOOD_PARAMS = {
    "tense":      {"freqs": [82.41, 110.00, 146.83, 196.00],     # E A D G — minor low
                    "bpm": 110, "wave": "saw"},
    "somber":     {"freqs": [220.00, 261.63, 329.63, 392.00],    # A C E G — minor sad (octave up)
                    "bpm": 60,  "wave": "sine"},
    "upbeat":     {"freqs": [261.63, 329.63, 392.00, 523.25],    # C E G C major
                    "bpm": 130, "wave": "square"},
    "mysterious": {"freqs": [146.83, 220.00, 277.18, 415.30],    # D A Db Ab — bumped up octave for audibility
                    "bpm": 70,  "wave": "saw"},  # saw is more harmonically rich than sine
    "romantic":   {"freqs": [196.00, 246.94, 293.66, 392.00],    # G B D G — warm
                    "bpm": 80,  "wave": "sine"},
    "neutral":    {"freqs": [220.00, 277.18, 329.63, 440.00],    # A Db E A
                    "bpm": 90,  "wave": "sine"},
}


def _osc(wave_type: str, t: float, freq: float) -> float:
    """Oscillator helper — returns sample in [-1, 1]."""
    phase = 2 * math.pi * freq * t
    if wave_type == "sine":
        return math.sin(phase)
    if wave_type == "square":
        return 1.0 if math.sin(phase) >= 0 else -1.0
    if wave_type == "saw":
        # phase mod 2π → ramp from -1 to 1
        return 2 * (phase / (2 * math.pi) - math.floor(phase / (2 * math.pi) + 0.5))
    return math.sin(phase)


def _generate_bgm_wav(out_path: Path, mood: str, duration_s: float,
                       sample_rate: int = 22050):
    """
    Procedurally generate a mood-appropriate BGM track.
    Uses chord arpeggio + bassline at the mood's BPM.
    Saves as a 16-bit mono WAV.
    """
    params = _MOOD_PARAMS.get(mood, _MOOD_PARAMS["neutral"])
    freqs  = params["freqs"]
    bpm    = params["bpm"]
    wave_t = params["wave"]

    n_frames = int(duration_s * sample_rate)
    samples_per_beat = int(sample_rate * 60 / bpm)

    samples = bytearray()
    rng = random.Random(hash(mood))

    for i in range(n_frames):
        t = i / sample_rate

        # Slowly cycle through chord notes (one per beat)
        beat_index = (i // samples_per_beat) % len(freqs)
        chord_freq = freqs[beat_index]

        # Bass: lowest freq, sustained
        bass = _osc("sine", t, freqs[0]) * 0.18

        # Lead: current chord note with slow decay each beat
        beat_phase    = (i % samples_per_beat) / samples_per_beat
        lead_envelope = math.exp(-3 * beat_phase) * 0.5
        lead          = _osc(wave_t, t, chord_freq) * lead_envelope * 0.25

        # Pad: sustained higher harmonic
        pad = _osc("sine", t, chord_freq * 1.5) * 0.08

        # Combine + soft clip
        sample = bass + lead + pad
        sample = max(-1.0, min(1.0, sample))

        # Apply gentle fade-in / fade-out (200ms)
        fade_frames = int(0.2 * sample_rate)
        if i < fade_frames:
            sample *= i / fade_frames
        elif i > n_frames - fade_frames:
            sample *= (n_frames - i) / fade_frames

        samples.extend(struct.pack("<h", int(sample * 30000)))

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(samples))


@mcp.tool()
def generate_background_music(scene_id: int, mood: str, duration_s: float) -> str:
    """
    PDF §4 Phase 2: "background music selection or generation per scene mood".

    Procedurally generates a mood-appropriate music track — no model downloads,
    no API keys. Each mood produces a distinct musical palette
    (instrumentation, tempo, chord progression).

    Args:
        scene_id:    Scene this BGM is for.
        mood:        tense | somber | upbeat | mysterious | romantic | neutral
        duration_s:  Duration in seconds.

    Returns: {status, file, mood, duration_s}
    """
    if mood not in _MOOD_PARAMS:
        mood = "neutral"

    out_path = BGM_DIR / f"scene_{scene_id:02d}_bgm_{mood}.wav"
    _generate_bgm_wav(out_path, mood, max(2.0, duration_s))

    return json.dumps({
        "status":     "success",
        "scene_id":   scene_id,
        "mood":       mood,
        "file":       str(out_path),
        "duration_s": round(duration_s, 2),
        "provider":   "procedural_synth",
    })


@mcp.tool()
def mix_audio_with_bgm(
    scene_id: int, dialogue_files: list, bgm_file: str = "",
    bgm_volume_db: float = -10.0,
) -> str:
    """
    PDF §4 Phase 2: mix BGM track under dialogue.
    Auto-discovers BGM file from BGM_DIR if not provided or path invalid.
    """
    if isinstance(dialogue_files, str):
        try:
            dialogue_files = json.loads(dialogue_files)
        except json.JSONDecodeError:
            dialogue_files = [dialogue_files]

    print(f"  [mix | scene {scene_id}] dialogue_files: {len(dialogue_files)}")
    print(f"  [mix | scene {scene_id}] bgm_file param: '{bgm_file}'")

    if not dialogue_files:
        return json.dumps({"status": "error",
                           "error": "dialogue_files must be non-empty"})
    if not _has_ffmpeg():
        return json.dumps({"status": "error", "error": "ffmpeg required"})

    # ── AUTO-DISCOVER BGM file if path doesn't work ──────────
    bgm_path = Path(bgm_file) if bgm_file else None
    if not bgm_path or not bgm_path.exists():
        candidates = list(BGM_DIR.glob(f"scene_{scene_id:02d}_bgm_*.wav"))
        if candidates:
            bgm_path = candidates[0]
            print(f"  [mix | scene {scene_id}] auto-discovered BGM: {bgm_path.name}")
        else:
            print(f"  [mix | scene {scene_id}] no BGM found in {BGM_DIR}")
            bgm_path = None

    # ── Concat dialogue files ────────────────────────────────
    dialogue_concat = AUDIO_DIR / f"scene_{scene_id:02d}_dialogue.wav"
    list_file = dialogue_concat.with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join(f"file '{Path(f).resolve()}'" for f in dialogue_files
                   if Path(f).exists())
    )
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-ar", "22050", "-ac", "1", str(dialogue_concat)],
            check=True,
        )
        print(f"  [mix | scene {scene_id}] dialogue concat OK")
    except subprocess.CalledProcessError as e:
        return json.dumps({"status": "error",
                           "error": f"dialogue concat failed: {e}"})
    finally:
        list_file.unlink(missing_ok=True)

    out_path = AUDIO_DIR / f"scene_{scene_id:02d}_full.wav"

    # ── If no BGM, just copy dialogue ────────────────────────
    if not bgm_path:
        shutil.copy2(dialogue_concat, out_path)
        return json.dumps({
            "status": "success_no_bgm",
            "scene_id": scene_id,
            "file": str(out_path),
            "note": "no BGM file found; using dialogue only",
        })

    # ── Mix dialogue + BGM ───────────────────────────────────
    # IMPORTANT: amix has automatic gain compensation that
    # makes the quieter signal nearly inaudible. We use
    # weights=1 1 to keep both at full per-input gain, then
    # apply BGM attenuation manually via volume filter BEFORE
    # the mix. We also normalize=0 to prevent amix's auto-AGC.
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(dialogue_concat), "-i", str(bgm_path),
             "-filter_complex",
             f"[0:a]volume=1.0[dlg];"
             f"[1:a]volume={bgm_volume_db}dB[bg];"
             f"[dlg][bg]amix=inputs=2:duration=first:normalize=0[a]",
             "-map", "[a]", "-ar", "22050", "-ac", "1",
             str(out_path)],
            capture_output=True, text=True, check=True,
        )
        print(f"  [mix | scene {scene_id}] ✓ MIXED (BGM @ {bgm_volume_db}dB, normalize=0)")
    except subprocess.CalledProcessError as e:
        print(f"  [mix | scene {scene_id}] ✗ ffmpeg failed: {e.stderr}")
        shutil.copy2(dialogue_concat, out_path)
        return json.dumps({
            "status": "mix_failed_used_dialogue_only",
            "scene_id": scene_id,
            "file": str(out_path),
            "error": str(e.stderr) if e.stderr else str(e),
        })

    return json.dumps({
        "status":     "success",
        "scene_id":   scene_id,
        "file":       str(out_path),
        "bgm_used":   str(bgm_path),
        "bgm_volume": bgm_volume_db,
    })


# ─── 3. Video Gen Tool ─────────────────────────────────────────────────────────

def _render_scene_video_pil_clean(
    out_path: Path,
    scene_id: int,
    prompt: str,
    duration_s: float,
    fps: int = 12,
) -> bool:
    """Local fallback renderer — clean background + text only, no character cards."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    W, H = 640, 360
    n_frames = max(1, int(duration_s * fps))
    scene_frames_dir = FRAMES_DIR / f"scene_{scene_id:02d}"
    scene_frames_dir.mkdir(parents=True, exist_ok=True)
    bg_color = _deterministic_color(prompt[:20] if prompt else f"scene_{scene_id}")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(n_frames):
        img = Image.new("RGBA", (W, H), bg_color + (255,))
        draw = ImageDraw.Draw(img)
        pulse = int(30 * math.sin(2 * math.pi * i / max(1, n_frames)))
        overlay_color = (
            max(0, min(255, bg_color[0] + pulse)),
            max(0, min(255, bg_color[1] + pulse)),
            max(0, min(255, bg_color[2] + pulse)),
        )
        draw.rectangle([(0, H - 80), (W, H)], fill=overlay_color + (255,))
        draw.text((20, 20), f"SCENE {scene_id}", fill=(255, 255, 255), font=font)
        draw.text((20, 50), f"PROMPT: {prompt[:50]}", fill=(230, 230, 230), font=font)
        # No character portraits, no ellipses, no character name labels
        draw.text((20, H - 30), f"frame {i+1}/{n_frames}", fill=(200, 200, 200), font=font)
        img.convert("RGB").save(scene_frames_dir / f"frame_{i:04d}.png")

    if _has_ffmpeg():
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-framerate", str(fps),
                 "-i", str(scene_frames_dir / "frame_%04d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(out_path)],
                check=True,
            )
            return out_path.exists()
        except subprocess.CalledProcessError as e:
            print(f"  [video/ffmpeg] failed: {e}")
            return False
    return True


@mcp.tool()
def query_stock_footage(
    scene_id: int,
    prompt: str,
    num_frames: int = 16,
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    negative_prompt: str = "",
    denoising_strength: float = 0.0,
    init_image_b64: str = "",
) -> str:
    """PDF §5.3 Video Gen tool — generates base video via ByteDance API.

    Args:
        scene_id:    Scene number.
        prompt:      Detailed visual prompt for image/video generation.
        num_frames:  Number of frames to generate (e.g. 16).
        width:       Video width.
        height:      Video height.

    Falls back to a clean local PIL placeholder if ByteDance API is not set
    or remote generation fails.
    """
    safe_loc = _safe_name(prompt[:40]) if prompt else f"scene_{scene_id}"
    out_path = VIDEO_DIR / f"scene_{scene_id:02d}_{safe_loc}.mp4"

    # ── Attempt ByteDance Seedance async API ───────────────────────────────────
    if os.environ.get("BYTEDANCE_API_BASE_URL") and os.environ.get("BYTEDANCE_API_KEY"):
        try:
            client = ByteDanceVideoClient.from_env()
            print(f"  [video | scene {scene_id}] Submitting async ByteDance task...")
            # Do not pass num_frames-derived duration (often 2s); seedance-1-5 rejects it.
            # Uses BYTEDANCE_VIDEO_DURATION + BYTEDANCE_ALLOWED_DURATIONS in the client.
            shot_kw: dict = {}
            if raw := os.environ.get("BYTEDANCE_SHOT_SECONDS"):
                try:
                    shot_kw["duration_s"] = max(1.0, float(raw.strip()))
                except ValueError:
                    pass
            # I2V forces the same reference image for every scene unless you vary URLs.
            # Opt-in only: set BYTEDANCE_I2V_ENABLED=1 and BYTEDANCE_I2V_IMAGE_URL=...
            i2v_url = (os.environ.get("BYTEDANCE_I2V_IMAGE_URL", "") or "").strip() or None
            i2v_on = (os.environ.get("BYTEDANCE_I2V_ENABLED", "") or "").strip().lower() in (
                "1", "true", "yes", "on",
            )
            image_url = i2v_url if (i2v_on and i2v_url) else None
            local_file = client.generate_video_and_wait(
                prompt=prompt,
                output_path=out_path,
                scene_id=scene_id,
                width=width,
                height=height,
                seed=seed,
                image_url=image_url,
                **shot_kw,
            )
            local_path = Path(local_file)
            if local_path.exists():
                print(f"  [video | scene {scene_id}] ✓ ByteDance video saved to {local_path} "
                      f"({local_path.stat().st_size // 1024} KB)")
                return json.dumps({
                    "status":     "success",
                    "scene_id":   scene_id,
                    "file":       str(local_path),
                    "num_frames": num_frames,
                    "provider":   "bytedance_seedance",
                })
        except ByteDanceVideoClientError as e:
            print(f"  [video | scene {scene_id}] ✗ ByteDance generation FAILED: {e}")
            print(f"  [video | scene {scene_id}]   → using local PIL fallback")
        except Exception as e:
            print(f"  [video | scene {scene_id}] ✗ Unexpected ByteDance error: {type(e).__name__}: {e}")
            print(f"  [video | scene {scene_id}]   → using local PIL fallback")

    # ── Local fallback (no character cards) ─────────────────────────────────
    fps = 8 # Match AnimateDiff target FPS for consistency
    derived_duration = num_frames / fps
    
    print(f"  [video | scene {scene_id}] Rendering local fallback ({derived_duration:.1f}s)")
    success = _render_scene_video_pil_clean(
        out_path=out_path,
        scene_id=scene_id,
        prompt=prompt,
        duration_s=derived_duration,
    )

    result_file = str(out_path) if out_path.exists() else str(FRAMES_DIR / f"scene_{scene_id:02d}")
    return json.dumps({
        "status":     "success" if success else "frames_only",
        "scene_id":   scene_id,
        "file":       result_file,
        "fps":        fps,
        "duration_s": derived_duration,
        "frame_count": num_frames,
        "provider":   "local_pil_clean",
        "ffmpeg_available": _has_ffmpeg(),
    })



# ─── 3b. Stock Image Tool (background fallback for lip_sync_node) ─────────────

@mcp.tool()
def query_stock_image(
    scene_id: int,
    visual_cue: str = "",
    location: str = "",
    mood: str = "",
) -> str:
    """Generate a background still image for a scene.

    Used by lip_sync_node as a fallback when no face-swapped video exists.
    Creates a clean PIL placeholder PNG (background colour + text, no character
    cards) and returns its path.

    Args:
        scene_id:   Scene number.
        visual_cue: Short visual description / prompt.
        location:   Scene location (used for colour seed).
        mood:       Scene mood (displayed as text).

    Returns: {status, scene_id, file}
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return json.dumps({"status": "error", "error": "PIL not installed"})

    out_path = FRAMES_DIR / f"scene_{scene_id:02d}_bg_image.png"
    W, H = 1920, 1080

    seed = location or visual_cue or f"scene_{scene_id}"
    bg_color = _deterministic_color(seed[:20])

    img  = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img)

    # Subtle gradient-feel: darker bottom bar
    darker = tuple(max(0, c - 60) for c in bg_color)
    draw.rectangle([(0, H - 200), (W, H)], fill=darker)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((60, 60),      f"SCENE {scene_id}",          fill=(255, 255, 255), font=font)
    draw.text((60, 100),     (visual_cue or "")[:120],      fill=(220, 220, 220), font=font)
    draw.text((60, H - 160), f"Location: {location[:80]}", fill=(200, 200, 200), font=font)
    if mood:
        draw.text((60, H - 120), f"Mood: {mood[:60]}", fill=(180, 180, 180), font=font)

    img.save(str(out_path))
    print(f"  [BG Image     | scene {scene_id}] ✓ {out_path}")
    return json.dumps({"status": "success", "scene_id": scene_id, "file": str(out_path)})


# ─── 4. Face Swap Tools ───────────────────────────────────────────────────────

@mcp.tool()
def identity_validator(character_name: str, character_db_json: str) -> str:
    """PDF §5.4 critical constraint: validate identity BEFORE swap."""
    try:
        db = json.loads(character_db_json)
    except json.JSONDecodeError as e:
        return json.dumps({"valid": False, "reason": f"Invalid character_db JSON: {e}"})

    characters = db.get("characters", []) if isinstance(db, dict) else db
    needle = character_name.strip().lower()

    for char in characters:
        if not isinstance(char, dict):
            continue
        name = (char.get("name") or "").strip().lower()
        if name == needle:
            png_path = _find_character_png(char.get("name", ""))
            return json.dumps({
                "valid": True, "character": character_name,
                "reference_path": png_path,
                "character_id": char.get("character_id", ""),
                "reason": ("Matched by name, Phase 1 reference image found."
                           if png_path
                           else "Matched by name (no reference image on disk)."),
            })

    return json.dumps({
        "valid": False, "character": character_name,
        "reason": f"No character matching '{character_name}' in character_db.",
    })


@mcp.tool()
def face_swapper(scene_id: int, base_video_path: str,
                  character_name: str, reference_path: str = "") -> str:
    """PDF §5.4 face_swapper.

    If GPU_WORKER_URL is set: streams the base video + reference portrait PNG to
    the Kaggle /face_swap endpoint and saves the returned MP4.
    Local fallback: copies the base video unmodified.
    """
    out_path = VIDEO_DIR / f"scene_{scene_id:02d}_faceswapped.mp4"

    if not Path(base_video_path).exists():
        return json.dumps({"status": "error", "scene_id": scene_id,
                           "error": f"base_video_path not found: {base_video_path}"})

    _NGROK_HEADERS = {
        "ngrok-skip-browser-warning": "1",
        "User-Agent": "MontageLocalServer/1.0",
    }

    gpu_url = os.environ.get("GPU_WORKER_URL", "").rstrip("/")

    # Resolve reference portrait
    if not reference_path or not Path(reference_path).exists():
        reference_path = _find_character_png(character_name)
    used_real_portrait = bool(reference_path and Path(reference_path).exists())

    # ── Attempt remote GPU face-swap ──────────────────────────────────────────
    if gpu_url and used_real_portrait:
        try:
            print(f"  [face_swap | scene {scene_id}] Sending to remote GPU for real face-swap")
            with open(base_video_path, "rb") as vf, open(reference_path, "rb") as pf:
                resp = requests.post(
                    f"{gpu_url}/face_swap",
                    files={
                        "source_image": (Path(reference_path).name, pf, "image/png"),
                        "target_video":  (Path(base_video_path).name, vf, "video/mp4"),
                    },
                    data={"scene_id": str(scene_id), "character_id": character_name[:20]},
                    headers=_NGROK_HEADERS,
                    timeout=300,
                )
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "json" not in ct:
                raise ValueError(f"Expected JSON, got '{ct}'. Prefix: {resp.text[:200]!r}")
            result = resp.json()
            video_b64 = result.get("video_b64", "")
            if video_b64:
                out_path.write_bytes(base64.b64decode(video_b64))
                print(f"  [face_swap | scene {scene_id}] ✓ Remote face-swap saved ({out_path.stat().st_size // 1024} KB)")
                return json.dumps({
                    "status": "success", "scene_id": scene_id,
                    "character": character_name, "file": str(out_path),
                    "frames_processed": -1,  # not tracked by remote worker
                    "reference_used": reference_path,
                    "used_real_portrait": True,
                    "provider": "remote_insightface",
                })
            else:
                print(f"  [face_swap | scene {scene_id}] ✗ Remote returned no video_b64: {list(result.keys())}")
        except Exception as e:
            print(f"  [face_swap | scene {scene_id}] ✗ Remote face-swap failed: {type(e).__name__}: {e}")
            print(f"  [face_swap | scene {scene_id}]   → using local pass-through fallback")
    elif gpu_url and not used_real_portrait:
        print(f"  [face_swap | scene {scene_id}] ⚠ No portrait for '{character_name}' — skipping remote face-swap")

    # ── Local fallback: pass base video through unmodified ────────────────────
    shutil.copy2(base_video_path, out_path)
    print(f"  [face_swap | scene {scene_id}] ℹ Local pass-through (no GPU swap performed)")

    return json.dumps({
        "status": "success", "scene_id": scene_id,
        "character": character_name, "file": str(out_path),
        "frames_processed": 0,
        "reference_used": reference_path or "none",
        "used_real_portrait": used_real_portrait,
        "provider": "local_passthrough",
    })


# ─── 5. Lip Sync Tool ─────────────────────────────────────────────────────────

def _concat_audio_files(audio_files: list, out_path: Path) -> float:
    if not audio_files:
        _generate_silent_wav(out_path, 1.0)
        return 1.0
    if _has_ffmpeg():
        list_file = out_path.with_suffix(".txt")
        list_file.write_text("\n".join(f"file '{Path(f).resolve()}'"
                                       for f in audio_files))
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "concat", "-safe", "0", "-i", str(list_file),
                 "-ar", "22050", "-ac", "1", str(out_path)],
                check=True,
            )
            list_file.unlink(missing_ok=True)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
                capture_output=True, text=True,
            )
            try:
                return float(probe.stdout.strip())
            except ValueError:
                return 0.0
        except subprocess.CalledProcessError:
            pass
    _generate_silent_wav(out_path, 2.0)
    return 2.0


@mcp.tool()
def lip_sync_aligner(scene_id: int, video_path: str, audio_files: list) -> str:
    """PDF §5.5 lip_sync_aligner.

    If GPU_WORKER_URL is set: streams video + concatenated audio to the Kaggle
    /lip_sync (Wav2Lip) endpoint and saves the returned MP4.
    Local fallback: hard-muxes the audio onto the video via ffmpeg (no mouth sync).
    """
    _NGROK_HEADERS = {
        "ngrok-skip-browser-warning": "1",
        "User-Agent": "MontageLocalServer/1.0",
    }

    if isinstance(audio_files, str):
        try:
            audio_files = json.loads(audio_files)
        except json.JSONDecodeError:
            audio_files = [audio_files]

    # ── Build / locate the concat audio ──────────────────────────────────────
    concat_audio = AUDIO_DIR / f"scene_{scene_id:02d}_full.wav"
    if not concat_audio.exists():
        audio_duration = _concat_audio_files(audio_files or [], concat_audio)
    else:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(concat_audio)],
            capture_output=True, text=True,
        )
        try:
            audio_duration = float(probe.stdout.strip())
        except ValueError:
            audio_duration = 5.0

    out_path = SCENES_DIR / f"scene_{scene_id:02d}.mp4"

    if not Path(video_path).exists():
        return json.dumps({"status": "error", "scene_id": scene_id,
                           "error": f"video not found: {video_path}"})

    # ── Attempt remote GPU Wav2Lip ────────────────────────────────────────────
    gpu_url = os.environ.get("GPU_WORKER_URL", "").rstrip("/")
    if gpu_url and concat_audio.exists():
        try:
            print(f"  [lip_sync | scene {scene_id}] Sending to remote Wav2Lip")
            with open(video_path, "rb") as vf, open(str(concat_audio), "rb") as af:
                resp = requests.post(
                    f"{gpu_url}/lip_sync",
                    files={
                        "video": (Path(video_path).name, vf, "video/mp4"),
                        "audio": (concat_audio.name,     af, "audio/wav"),
                    },
                    data={"scene_id": str(scene_id)},
                    headers=_NGROK_HEADERS,
                    timeout=300,
                )
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "json" not in ct:
                raise ValueError(f"Expected JSON, got '{ct}'. Prefix: {resp.text[:200]!r}")
            result = resp.json()
            video_b64 = result.get("video_b64", "")
            if video_b64:
                out_path.write_bytes(base64.b64decode(video_b64))
                # Measure real duration
                probe2 = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
                    capture_output=True, text=True,
                )
                try:
                    final_duration = float(probe2.stdout.strip())
                except ValueError:
                    final_duration = audio_duration
                drift = abs(final_duration - audio_duration)
                score = max(0.0, 1.0 - (drift / max(1.0, audio_duration)))
                print(f"  [lip_sync | scene {scene_id}] ✓ Remote Wav2Lip done ({out_path.stat().st_size // 1024} KB)")
                return json.dumps({
                    "status": "success", "scene_id": scene_id,
                    "file": str(out_path), "duration_s": round(final_duration, 2),
                    "audio_duration": round(audio_duration, 2),
                    "lip_sync_score": round(score, 3),
                    "drift_s": round(drift, 3),
                    "provider": "remote_wav2lip",
                })
            else:
                print(f"  [lip_sync | scene {scene_id}] ✗ Remote returned no video_b64: {list(result.keys())}")
        except Exception as e:
            print(f"  [lip_sync | scene {scene_id}] ✗ Remote Wav2Lip failed: {type(e).__name__}: {e}")
            print(f"  [lip_sync | scene {scene_id}]   → falling back to local ffmpeg mux")

    # ── Local fallback: ffmpeg hard-mux (no mouth sync) ───────────────────────
    if not _has_ffmpeg():
        return json.dumps({"status": "error", "scene_id": scene_id,
                           "error": "ffmpeg required for local fallback"})
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-stream_loop", "-1", "-i", str(video_path),
             "-i", str(concat_audio),
             "-t", str(audio_duration),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart", str(out_path)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return json.dumps({"status": "error", "error": f"mux: {e}"})

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
        capture_output=True, text=True,
    )
    try:
        final_duration = float(probe.stdout.strip())
    except ValueError:
        final_duration = audio_duration

    drift = abs(final_duration - audio_duration)
    score = max(0.0, 1.0 - (drift / max(1.0, audio_duration)))
    return json.dumps({
        "status": "success", "scene_id": scene_id,
        "file": str(out_path), "duration_s": round(final_duration, 2),
        "audio_duration": round(audio_duration, 2),
        "lip_sync_score": round(score, 3),
        "drift_s": round(drift, 3),
        "provider": "local_ffmpeg_mux",
    })


# ─── Memory Tools ─────────────────────────────────────────────────────────────

@mcp.tool()
def commit_memory(key: str, value: str, category: str = "general") -> str:
    entry_id  = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat()
    try:
        _memory.add(
            ids=[entry_id], documents=[value],
            metadatas=[{"key": key, "category": category, "timestamp": timestamp}],
        )
        return json.dumps({"status": "committed", "id": entry_id,
                           "key": key, "category": category})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "key": key})


@mcp.tool()
def query_memory(category: str = "", keyword: str = "", limit: int = 50) -> str:
    try:
        where_filter = {"category": category} if category else None
        if keyword:
            results = _memory.query(query_texts=[keyword], n_results=limit,
                                     where=where_filter)
            ids   = (results.get("ids")       or [[]])[0]
            docs  = (results.get("documents") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
        else:
            results = _memory.get(where=where_filter, limit=limit)
            ids   = results.get("ids")       or []
            docs  = results.get("documents") or []
            metas = results.get("metadatas") or []
        entries = [{
            "id":        ids[i],
            "key":       metas[i].get("key", "")       if i < len(metas) else "",
            "category":  metas[i].get("category", "")  if i < len(metas) else "",
            "timestamp": metas[i].get("timestamp", "") if i < len(metas) else "",
            "value":     docs[i]                        if i < len(docs)  else "",
        } for i in range(len(ids))]
        return json.dumps({"count": len(entries), "entries": entries})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e),
                           "count": 0, "entries": []})


if __name__ == "__main__":
    # Refresh critical env at runtime start (protect against stale shell env).
    _load_env(force_override=True)
    print(f"[Studio Floor MCP] starting on port 8200")
    print(f"[Studio Floor MCP] ffmpeg available: {_has_ffmpeg()}")
    print(f"[Studio Floor MCP] BYTEDANCE_API_BASE_URL: {os.environ.get('BYTEDANCE_API_BASE_URL', '')}")
    print(f"[Studio Floor MCP] BYTEDANCE_MODEL: {os.environ.get('BYTEDANCE_MODEL', '')}")
    print(f"[Studio Floor MCP] Phase 1 image search paths:")
    for p in _find_phase1_image_dir():
        print(f"    {p}  ({len(list(p.glob('*.png')))} PNGs)")
    print(f"[Studio Floor MCP] memory entries: {_memory.count()}")
    print(f"[Studio Floor MCP] BGM dir: {BGM_DIR}")
    mcp.run(transport="streamable-http")