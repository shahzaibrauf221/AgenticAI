# ============================================================
# schemas.py
# Pydantic schemas for the unified project state.
# Place this in your Phase 1 project root (alongside graph.py).
#
# Import from Phase 2 + 3 + 5 to guarantee schema consistency.
# ============================================================

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ─── Character ────────────────────────────────────────────────────────────────

class VoicePersonality(BaseModel):
    """Per-character voice parameters for Phase 2 TTS."""
    tone:       str = Field(default="neutral",   description="warm/cold/confident/anxious/sad/calm")
    pace:       str = Field(default="medium",    description="slow/medium/fast")
    accent:     str = Field(default="american",  description="american/british/australian/indian/etc")
    gender:     str = Field(default="neutral",   description="male/female/neutral")
    pitch:      str = Field(default="mid",       description="low/mid/high")


class Character(BaseModel):
    character_id:     str
    name:             str
    role:             str                                = Field(default="supporting")
    appearance:       str
    costume:          Optional[str]                      = ""
    personality_traits: List[str]                        = Field(default_factory=list)
    voice_personality: VoicePersonality                  = Field(default_factory=VoicePersonality)
    reference_style:  str                                = "photorealistic"
    scenes_appeared:  List[int]                          = Field(default_factory=list)


# ─── Script / Scenes ──────────────────────────────────────────────────────────

class DialogueLine(BaseModel):
    speaker:    str
    line:       str
    visual_cue: str = ""
    emotion:    str = "neutral"


class Scene(BaseModel):
    scene_id:           int
    location:           str
    time_of_day:        str                  = "day"
    tone:               str                  = "neutral"       # NEW — per-scene mood for BGM
    characters:         List[str]            = Field(default_factory=list)
    action_description: str                  = ""
    dialogue:           List[DialogueLine]   = Field(default_factory=list)
    scene_visual_cue:   str                  = ""
    shots:              List[str]            = Field(default_factory=list)
    duration_s:         float                = 5.0              # NEW — per-scene duration

    @field_validator("duration_s")
    @classmethod
    def duration_positive(cls, v: float) -> float:
        return max(1.0, float(v))

    @field_validator("shots")
    @classmethod
    def shots_exactly_four(cls, v: List[str]) -> List[str]:
        shots = [str(s).strip() for s in (v or []) if str(s).strip()]
        if len(shots) != 4:
            raise ValueError("shots must contain exactly 4 non-empty prompts")
        return shots


# ─── Top-level State Objects ─────────────────────────────────────────────────

class Story(BaseModel):
    """Spec: story — title, logline, themes, arc summary."""
    title:         str
    genre:         str             = "drama"
    logline:       str             = ""
    themes:        List[str]       = Field(default_factory=list)
    arc:           str             = ""
    total_duration_s: float        = 0.0


class ScriptOutput(BaseModel):
    """Unified Phase 1 output — maps 1:1 to spec §4 'validated JSON'."""
    story:      Story
    scenes:     List[Scene]
    characters: List[Character]


# ─── Phase 2 → Phase 3 handoff objects (per spec §4 / diagram) ───────────────

class AudioSegment(BaseModel):
    """Timing manifest row — spec §4 Phase 2 output."""
    scene_id:    int
    line_index:  int
    speaker:     str
    audio_file:  str
    start_ms:    int
    end_ms:      int
    text:        str


class TimingManifest(BaseModel):
    total_duration_ms:  int
    segments:           List[AudioSegment]


class Phase2AudioHandoff(BaseModel):
    """phase2_audio_handoff.json — consumed by Phase 3."""
    voice_configs:  dict   # {character_name: VoicePersonality dict + backend info}
    segments:       List[AudioSegment]
    music_moods:    dict   # {scene_id: mood_string}


class Phase3VideoHandoff(BaseModel):
    """phase3_video_handoff.json — scene-level visual prompts + camera."""
    scenes:         List[dict]   # free-form per-scene visual spec
    transitions:    List[dict]   # between-scene fade/cut settings


# ─── Run summary (used by Phase 4 dashboard) ─────────────────────────────────

class RunSummary(BaseModel):
    run_id:        str
    status:        Literal["processing", "complete", "failed"]
    phase1_done:   bool  = False
    phase2_done:   bool  = False
    phase3_done:   bool  = False
    artifacts:     dict  = Field(default_factory=dict)
    errors:        List[str] = Field(default_factory=list)
    tools_log:     List[dict] = Field(default_factory=list)
