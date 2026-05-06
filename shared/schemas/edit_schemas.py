from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

EditTarget = Literal["audio", "video_frame", "video", "script"]
EditScope = str

INTENT_TO_TARGET: dict[str, EditTarget] = {
    "change_voice_tone": "audio",
    "change_voice_speed": "audio",
    "add_background_music": "audio",
    "remove_background_music": "audio",
    "regenerate_audio": "audio",
    "make_scene_darker": "video_frame",
    "make_scene_brighter": "video_frame",
    "change_scene_style": "video_frame",
    "change_character_design": "video_frame",
    "apply_color_filter": "video_frame",
    "regenerate_scene_image": "video_frame",
    "remove_subtitle": "video",
    "add_subtitle": "video",
    "speed_up_scene": "video",
    "slow_down_scene": "video",
    "recompose_video": "video",
    "regenerate_script": "script",
    "change_scene_dialogue": "script",
    "change_scene_tone": "script",
}


class EditIntent(BaseModel):
    intent: str = Field(..., description="Canonical intent key")
    target: EditTarget
    scope: EditScope = Field(default="all")
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        v = (value or "").strip()
        return v if v else "all"

    @model_validator(mode="after")
    def validate_target_mapping(self) -> "EditIntent":
        expected = INTENT_TO_TARGET.get(self.intent)
        if expected is None:
            raise ValueError(f"Unknown intent '{self.intent}'.")
        if self.target != expected:
            raise ValueError(
                f"Intent '{self.intent}' must target '{expected}', got '{self.target}'."
            )
        return self

