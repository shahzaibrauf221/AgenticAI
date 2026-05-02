# ============================================================
# serializer.py
# Converts your existing Phase 1 outputs (scene_manifest.json +
# character_db.json) into the SPEC-COMPLIANT artifacts:
#
#   • story.json
#   • characters.json
#   • script.json
#   • phase2_audio_handoff.json
#   • phase3_video_handoff.json
#   • summary.json
#
# Usage:
#   python serializer.py                # uses outputs/ by default
#   python serializer.py --out <dir>    # write elsewhere
# ============================================================

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

# When run as a script (`python serializer.py`) the project root isn't on the
# import path. Make sure imports like `shared.schemas.*` resolve.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.schemas.schemas import (
    VoicePersonality, Character, DialogueLine, Scene,
    Story, ScriptOutput, Phase2AudioHandoff, Phase3VideoHandoff, RunSummary,
)


# ─── Trait → voice mapping (shared w/ Phase 2) ───────────────────────────────

_TRAIT_VOICE = {
    "determined":   dict(tone="confident", pace="medium", pitch="mid"),
    "resourceful":  dict(tone="confident", pace="medium", pitch="mid"),
    "intelligent":  dict(tone="calm",      pace="medium", pitch="mid"),
    "mysterious":   dict(tone="calm",      pace="slow",   pitch="low"),
    "nervous":      dict(tone="anxious",   pace="fast",   pitch="high"),
    "manipulative": dict(tone="anxious",   pace="medium", pitch="mid"),
    "enigmatic":    dict(tone="calm",      pace="slow",   pitch="low"),
    "curious":      dict(tone="calm",      pace="medium", pitch="mid"),
    "independent":  dict(tone="confident", pace="medium", pitch="mid"),
    "empathetic":   dict(tone="warm",      pace="medium", pitch="mid"),
    "suspicious":   dict(tone="anxious",   pace="medium", pitch="mid"),
}


def _derive_voice_personality(traits: list, gender: str) -> VoicePersonality:
    tone = pace = pitch = None
    for t in traits or []:
        mapped = _TRAIT_VOICE.get(t.lower())
        if mapped:
            tone = mapped["tone"]; pace = mapped["pace"]; pitch = mapped["pitch"]
            break

    gender_l = (gender or "").lower()
    accent_by_gender = {"male": "british", "female": "indian"}
    accent = accent_by_gender.get(gender_l, "american")

    return VoicePersonality(
        tone=tone or ("warm" if gender_l == "female" else "calm"),
        pace=pace or "medium",
        accent=accent,
        gender=gender_l or "neutral",
        pitch=pitch or "mid",
    )


def _derive_scene_tone(scene: dict) -> str:
    """Simple keyword-based mood detection for BGM."""
    text = (scene.get("action_description", "") + " "
            + scene.get("scene_visual_cue", "")).lower()
    if any(k in text for k in ("fight", "chase", "explo", "danger", "combat")):
        return "tense"
    if any(k in text for k in ("sad", "crying", "tear", "grief", "funeral")):
        return "somber"
    if any(k in text for k in ("laugh", "joke", "funny", "celebrat", "party")):
        return "upbeat"
    if any(k in text for k in ("mystery", "dark", "shadow", "whisper", "secret")):
        return "mysterious"
    if any(k in text for k in ("love", "romantic", "tender", "embrace")):
        return "romantic"
    return "neutral"


def _estimate_scene_duration(scene: dict) -> float:
    """Estimate duration based on dialogue word count + action description."""
    total_words = sum(
        len(d.get("line", "").split())
        for d in scene.get("dialogue", []) if isinstance(d, dict)
    )
    # ~2.5 words/sec for speech + 2s breathing room for action
    return max(3.0, total_words / 2.5 + 2.0)


# ─── Main serialization ──────────────────────────────────────────────────────

def serialize(phase1_outputs: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((phase1_outputs / "scene_manifest.json").read_text(encoding="utf-8"))
    char_db  = json.loads((phase1_outputs / "character_db.json").read_text(encoding="utf-8"))

    # ─── 1. Characters (with voice_personality) ──────────────────
    characters: list[Character] = []
    for c in char_db.get("characters", []):
        if not isinstance(c, dict):
            continue
        vp = _derive_voice_personality(c.get("personality_traits", []), c.get("gender", ""))
        ch = Character(
            character_id       = c.get("character_id", f"char_{len(characters)+1:03d}"),
            name               = c.get("name", "Unknown"),
            role               = c.get("role", "supporting"),
            appearance         = c.get("appearance", ""),
            costume            = c.get("costume", ""),
            personality_traits = c.get("personality_traits", []),
            voice_personality  = vp,
            reference_style    = c.get("reference_style", "photorealistic"),
            scenes_appeared    = c.get("scenes_appeared", []),
        )
        characters.append(ch)

    # ─── 2. Scenes (with tone + duration) ────────────────────────
    scenes: list[Scene] = []
    total_duration = 0.0
    for s in manifest.get("scenes", []):
        dur   = _estimate_scene_duration(s)
        total_duration += dur
        dialogue = [
            DialogueLine(
                speaker    = d.get("speaker", ""),
                line       = d.get("line", ""),
                visual_cue = d.get("visual_cue", ""),
                emotion    = d.get("emotion", "neutral"),
            )
            for d in s.get("dialogue", []) if isinstance(d, dict)
        ]
        scenes.append(Scene(
            scene_id           = s.get("scene_id", len(scenes) + 1),
            location           = s.get("location", ""),
            time_of_day        = s.get("time_of_day", "day"),
            tone               = _derive_scene_tone(s),
            characters         = s.get("characters", []),
            action_description = s.get("action_description", ""),
            dialogue           = dialogue,
            scene_visual_cue   = s.get("scene_visual_cue", ""),
            duration_s         = dur,
        ))

    # ─── 3. Story ────────────────────────────────────────────────
    story = Story(
        title            = manifest.get("title", "Untitled"),
        genre            = manifest.get("genre", "drama"),
        logline          = manifest.get("logline", ""),
        themes           = manifest.get("themes", []),
        arc              = manifest.get("arc", ""),
        total_duration_s = round(total_duration, 2),
    )

    # ─── 4. Write individual spec artifacts ──────────────────────
    (out_dir / "story.json").write_text(story.model_dump_json(indent=2))

    (out_dir / "characters.json").write_text(
        json.dumps({"characters": [c.model_dump() for c in characters]}, indent=2)
    )

    script_obj = ScriptOutput(story=story, scenes=scenes, characters=characters)
    (out_dir / "script.json").write_text(script_obj.model_dump_json(indent=2), encoding="utf-8")
    # ─── 5. Phase 2 handoff ──────────────────────────────────────
    voice_configs = {c.name: c.voice_personality.model_dump() for c in characters}
    music_moods   = {str(s.scene_id): s.tone for s in scenes}

    ph2 = Phase2AudioHandoff(
        voice_configs = voice_configs,
        segments      = [],   # Phase 2 fills this
        music_moods   = music_moods,
    )
    (out_dir / "phase2_audio_handoff.json").write_text(ph2.model_dump_json(indent=2))

    # ─── 6. Phase 3 handoff ──────────────────────────────────────
    scene_visuals = []
    transitions   = []
    for s in scenes:
        scene_visuals.append({
            "scene_id":       s.scene_id,
            "visual_prompt":  f"{s.scene_visual_cue}, {s.action_description}, {s.tone} mood",
            "location":       s.location,
            "time_of_day":    s.time_of_day,
            "camera":         "medium_shot",
            "duration_s":     s.duration_s,
        })
        transitions.append({
            "from_scene": s.scene_id,
            "to_scene":   s.scene_id + 1 if s.scene_id < len(scenes) else None,
            "type":       "fade",
            "duration_s": 0.5,
        })
    ph3 = Phase3VideoHandoff(scenes=scene_visuals, transitions=transitions)
    (out_dir / "phase3_video_handoff.json").write_text(ph3.model_dump_json(indent=2))

    # ─── 7. Summary ──────────────────────────────────────────────
    summary = RunSummary(
        run_id      = str(uuid.uuid4())[:8],
        status      = "processing",
        phase1_done = True,
        artifacts   = {
            "story":                   str(out_dir / "story.json"),
            "characters":              str(out_dir / "characters.json"),
            "script":                  str(out_dir / "script.json"),
            "phase2_audio_handoff":    str(out_dir / "phase2_audio_handoff.json"),
            "phase3_video_handoff":    str(out_dir / "phase3_video_handoff.json"),
        },
    )
    (out_dir / "summary.json").write_text(summary.model_dump_json(indent=2))

    print("=" * 60)
    print(f"  SPEC-COMPLIANT ARTIFACTS WRITTEN → {out_dir}")
    print("=" * 60)
    for f in ["story.json", "characters.json", "script.json",
              "phase2_audio_handoff.json", "phase3_video_handoff.json",
              "summary.json"]:
        size = (out_dir / f).stat().st_size
        print(f"  ✓ {f:<35} {size:>6} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", default="outputs",
                        help="Path to Phase 1 outputs dir (has scene_manifest.json + character_db.json)")
    parser.add_argument("--out", default="outputs",
                        help="Where to write the spec-compliant artifacts")
    args = parser.parse_args()

    serialize(Path(args.phase1), Path(args.out))


if __name__ == "__main__":
    main()
