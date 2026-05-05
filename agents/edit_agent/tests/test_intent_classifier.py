# ============================================================
# tests/test_intent_classifier.py
# Phase 5 — Intent Classification Tests (keyword fallback path)
# Tests 10+ edit query types without requiring API keys.
# ============================================================

import pytest
from agents.edit_agent.intent_classifier import _keyword_classify, _extract_scope

# ─── Keyword classifier tests (no API needed) ─────────────────────────────────

@pytest.mark.parametrize("query, expected_intent, expected_target", [
    ("Make scene 1 darker",                  "make_scene_darker",      "video_frame"),
    ("The scene is too bright, fix it",      "make_scene_brighter",    "video_frame"),
    ("Remove subtitle from video",           "remove_subtitle",        "video"),
    ("Add background music to scene 2",      "add_background_music",   "audio"),
    ("Change voice tone for Narrator",       "change_voice_tone",      "audio"),
    ("Speed up scene 3",                     "speed_up_scene",         "video"),
    ("Slow down the last scene",             "slow_down_scene",        "video"),
    ("Apply sepia color filter to scene 1",  "apply_color_filter",     "video_frame"),
    ("Regenerate the script",                "regenerate_script",      "script"),
    ("Change the dialogue in scene 2",       "change_scene_dialogue",  "script"),
    ("Update character design for hero",     "change_character_design","video_frame"),
    ("Remove BGM from all scenes",           "add_background_music",   "audio"),  # "bgm" keyword
])
def test_keyword_classify(query, expected_intent, expected_target):
    intent, target, scope, params = _keyword_classify(query)
    assert intent == expected_intent, f"Query '{query}' → got intent={intent}, expected {expected_intent}"
    assert target == expected_target, f"Query '{query}' → got target={target}, expected {expected_target}"


# ─── Scope extraction tests ───────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected_scope", [
    ("Make scene 1 darker",         "scene:1"),
    ("Apply filter to scene 3",     "scene:3"),
    ("Change character:Narrator",   "character:Narrator"),
    ("No specific scope here",      "all"),
    ("Change voice tone scene 5",   "scene:5"),
])
def test_extract_scope(query, expected_scope):
    scope = _extract_scope(query)
    assert scope == expected_scope, f"Query '{query}' → scope={scope}, expected {expected_scope}"
