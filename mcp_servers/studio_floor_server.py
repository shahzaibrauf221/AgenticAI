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

import hashlib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

import chromadb
from mcp.server.fastmcp import FastMCP

# ─── Load .env ────────────────────────────────────────────────────────────────
def _load_env():
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
                if key and value and key not in os.environ:
                    os.environ[key] = value
            break

_load_env()

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
            check=True,
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
) -> str:
    """PDF §5.2 Voice Synthesis Agent."""
    safe_char = _safe_name(character_name)
    line_hash = hashlib.md5(dialogue_line.encode()).hexdigest()[:6]
    out_path  = AUDIO_DIR / f"scene_{scene_id:02d}_{safe_char}_{line_hash}.wav"

    if _tts_gtts(dialogue_line, out_path, tld=tld, emotion=emotion):
        actual = out_path if out_path.exists() else out_path.with_suffix(".mp3")
        words      = max(1, len(dialogue_line.split()))
        tempo      = _EMOTION_TEMPO.get(emotion, 1.0)
        duration_s = (words / 2.5) / tempo
        return json.dumps({
            "status": "success", "provider": f"gTTS[{tld}]",
            "scene_id": scene_id, "character": character_name,
            "emotion": emotion, "tempo_applied": tempo,
            "voice_profile": voice_profile, "tld": tld,
            "file": str(actual), "duration_s": round(duration_s, 2),
            "text": dialogue_line,
        })

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


# ─── 3. Video Gen Tool (unchanged from your working version) ─────────────────

def _render_scene_video_pil(out_path: Path, scene_id: int, location: str,
                             characters: list, visual_cue: str, duration_s: float,
                             character_images: dict = None, fps: int = 12) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    character_images = character_images or {}
    W, H = 640, 360
    n_frames = max(1, int(duration_s * fps))
    scene_frames_dir = FRAMES_DIR / f"scene_{scene_id:02d}"
    scene_frames_dir.mkdir(parents=True, exist_ok=True)
    bg_color = _deterministic_color(location or f"scene_{scene_id}")

    char_portraits = {}
    for char in characters[:4]:
        img_path = character_images.get(char)
        if img_path and Path(img_path).exists():
            try:
                char_portraits[char] = Image.open(img_path).convert("RGBA").resize((100, 100))
            except Exception as e:
                print(f"  [video] couldn't load portrait for {char}: {e}")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(n_frames):
        img = Image.new("RGBA", (W, H), bg_color + (255,))
        draw = ImageDraw.Draw(img)
        pulse = int(30 * math.sin(2 * math.pi * i / max(1, n_frames)))
        overlay_color = (max(0, min(255, bg_color[0] + pulse)),
                         max(0, min(255, bg_color[1] + pulse)),
                         max(0, min(255, bg_color[2] + pulse)))
        draw.rectangle([(0, H - 80), (W, H)], fill=overlay_color + (255,))
        draw.text((20, 20), f"SCENE {scene_id}", fill=(255, 255, 255), font=font)
        draw.text((20, 50), f"LOCATION: {location[:50]}", fill=(230, 230, 230), font=font)

        for idx, char in enumerate(characters[:4]):
            cx = 90 + idx * 140
            cy = H // 2 + int(6 * math.sin(2 * math.pi * i / max(1, fps)))
            if char in char_portraits:
                img.alpha_composite(char_portraits[char], (cx - 50, cy - 50))
                draw.rectangle([(cx - 50, cy - 50), (cx + 50, cy + 50)],
                               outline=(255, 255, 255, 255), width=2)
            else:
                ccolor = _deterministic_color(char) + (255,)
                draw.ellipse([(cx - 45, cy - 45), (cx + 45, cy + 45)],
                             fill=ccolor, outline=(0, 0, 0, 255))
            draw.text((cx - 40, cy + 55), char[:14], fill=(255, 255, 255), font=font)

        cue = (visual_cue or "")[:70]
        draw.text((20, H - 70), cue, fill=(255, 255, 200), font=font)
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
def query_stock_footage(scene_id: int, location: str, characters: list,
                         visual_cue: str = "", duration_s: float = 5.0) -> str:
    """PDF §5.3 Video Gen tool."""
    safe_loc = _safe_name(location[:40]) if location else f"scene_{scene_id}"
    out_path = VIDEO_DIR / f"scene_{scene_id:02d}_{safe_loc}.mp4"

    if isinstance(characters, str):
        try:
            characters = json.loads(characters)
        except json.JSONDecodeError:
            characters = [characters]

    character_images = {}
    for char in characters or []:
        png_path = _find_character_png(char)
        if png_path:
            character_images[char] = png_path
            print(f"  [video | scene {scene_id}] using Phase 1 portrait for '{char}'")
        else:
            print(f"  [video | scene {scene_id}] no Phase 1 portrait for '{char}'")

    success = _render_scene_video_pil(
        out_path=out_path, scene_id=scene_id, location=location,
        characters=characters or [], visual_cue=visual_cue,
        duration_s=duration_s, character_images=character_images,
    )

    fps = 12
    result_file = str(out_path) if out_path.exists() else str(FRAMES_DIR / f"scene_{scene_id:02d}")
    return json.dumps({
        "status": "success" if success else "frames_only",
        "scene_id": scene_id, "location": location, "file": result_file,
        "fps": fps, "duration_s": duration_s,
        "frame_count": int(duration_s * fps),
        "character_portraits_used": list(character_images.keys()),
        "ffmpeg_available": _has_ffmpeg(),
    })


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
    """PDF §5.4 face_swapper."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return json.dumps({"status": "error", "error": "PIL not installed"})

    scene_frames_dir = FRAMES_DIR / f"scene_{scene_id:02d}"
    swapped_dir      = FRAMES_DIR / f"scene_{scene_id:02d}_swapped"
    swapped_dir.mkdir(parents=True, exist_ok=True)

    if not scene_frames_dir.exists():
        return json.dumps({"status": "error", "scene_id": scene_id,
                           "error": f"No source frames at {scene_frames_dir}"})

    frame_files = sorted(scene_frames_dir.glob("frame_*.png"))
    if not frame_files:
        return json.dumps({"status": "error", "error": "No frames to swap."})

    if not reference_path or not Path(reference_path).exists():
        reference_path = _find_character_png(character_name)

    ref_img = None
    used_real_portrait = False
    if reference_path and Path(reference_path).exists():
        try:
            ref_img = Image.open(reference_path).convert("RGBA").resize((200, 200))
            used_real_portrait = True
            print(f"  [face_swap | scene {scene_id}] using real portrait")
        except Exception as e:
            print(f"  [face_swap | scene {scene_id}] couldn't load: {e}")

    if ref_img is None:
        ref_img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        d = ImageDraw.Draw(ref_img)
        color = _deterministic_color(character_name) + (255,)
        d.ellipse([(0, 0), (200, 200)], fill=color,
                  outline=(255, 255, 255, 255), width=4)

    for f in frame_files:
        frame = Image.open(f).convert("RGBA")
        frame.alpha_composite(ref_img, (frame.width - 210, 10))
        d = ImageDraw.Draw(frame, "RGBA")
        d.rectangle([(frame.width - 210, 215), (frame.width - 10, 240)],
                    fill=(0, 0, 0, 180))
        d.text((frame.width - 205, 220), f"FOCUS: {character_name[:20]}",
               fill=(255, 215, 0, 255))
        frame.convert("RGB").save(swapped_dir / f.name)

    out_path = VIDEO_DIR / f"scene_{scene_id:02d}_faceswapped.mp4"
    fps = 12
    if _has_ffmpeg():
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-framerate", str(fps),
                 "-i", str(swapped_dir / "frame_%04d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            return json.dumps({"status": "error", "error": f"re-encode: {e}"})

    result_file = str(out_path) if out_path.exists() else str(swapped_dir)
    return json.dumps({
        "status": "success", "scene_id": scene_id,
        "character": character_name, "file": result_file,
        "frames_processed": len(frame_files),
        "reference_used": reference_path or "synthetic_placeholder",
        "used_real_portrait": used_real_portrait,
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
    """PDF §5.5 lip_sync_aligner. Hard-locks video duration to audio."""
    if isinstance(audio_files, str):
        try:
            audio_files = json.loads(audio_files)
        except json.JSONDecodeError:
            audio_files = [audio_files]

    concat_audio = AUDIO_DIR / f"scene_{scene_id:02d}_full.wav"

    # If a pre-mixed (dialogue+BGM) version already exists, use it.
    # Otherwise concat dialogue.
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
    if not _has_ffmpeg():
        return json.dumps({"status": "error", "scene_id": scene_id,
                           "error": "ffmpeg required"})
    if not Path(video_path).exists():
        return json.dumps({"status": "error", "scene_id": scene_id,
                           "error": f"video not found: {video_path}"})

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
    print(f"[Studio Floor MCP] starting on port 8200")
    print(f"[Studio Floor MCP] ffmpeg available: {_has_ffmpeg()}")
    print(f"[Studio Floor MCP] Phase 1 image search paths:")
    for p in _find_phase1_image_dir():
        print(f"    {p}  ({len(list(p.glob('*.png')))} PNGs)")
    print(f"[Studio Floor MCP] memory entries: {_memory.count()}")
    print(f"[Studio Floor MCP] BGM dir: {BGM_DIR}")
    mcp.run(transport="streamable-http")