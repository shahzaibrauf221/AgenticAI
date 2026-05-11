# ============================================================
# agents/edit_agent/intent_classifier.py
# Phase 5 — LLM-powered intent classification node
#
# Receives raw edit query → outputs structured EditIntent via
# Claude API using JSON-schema constrained output.
# ============================================================

from __future__ import annotations

import json
import os
import re

from groq import AsyncGroq

from agents.edit_agent.state import EditAgentState

# ─── Supported intents ────────────────────────────────────────────────────────

INTENT_CATALOGUE: dict[str, dict] = {
    # Audio intents
    "change_voice_tone":    {"target": "audio",       "desc": "Change voice tone/emotion for a character"},
    "change_voice_speed":   {"target": "audio",       "desc": "Change speaking speed for a character"},
    "add_background_music": {"target": "audio",       "desc": "Add or swap background music for a scene"},
    "remove_background_music": {"target": "audio",    "desc": "Remove background music from a scene"},
    "regenerate_audio":     {"target": "audio",       "desc": "Re-synthesize all audio for a scene or character"},

    # Video-frame intents
    "make_scene_darker":    {"target": "video_frame", "desc": "Apply dark/dim visual filter to a scene"},
    "make_scene_brighter":  {"target": "video_frame", "desc": "Apply bright/vibrant filter to a scene"},
    "change_scene_style":   {"target": "video_frame", "desc": "Change the visual style of a scene"},
    "change_character_design": {"target": "video_frame", "desc": "Re-generate visuals for a character"},
    "apply_color_filter":   {"target": "video_frame", "desc": "Apply a named color filter to scene images"},
    "regenerate_scene_image": {"target": "video_frame", "desc": "Re-generate the image for a scene"},

    # Full-video intents
    "remove_subtitle":      {"target": "video",       "desc": "Remove subtitle burn-in from final video"},
    "add_subtitle":         {"target": "video",       "desc": "Add subtitles to the final video"},
    "speed_up_scene":       {"target": "video",       "desc": "Increase playback speed for a scene"},
    "slow_down_scene":      {"target": "video",       "desc": "Decrease playback speed for a scene"},
    "recompose_video":      {"target": "video",       "desc": "Re-export final MP4 with current assets"},

    # Script intents
    "regenerate_script":    {"target": "script",      "desc": "Re-run story/script phase for the full video"},
    "change_scene_dialogue":{"target": "script",      "desc": "Rewrite the dialogue for a specific scene"},
    "change_scene_tone":    {"target": "script",      "desc": "Change the emotional tone of a scene"},

    # Meta intent — used when the user's command is too vague to act on
    "clarify_request":      {"target": "video",       "desc": "Query is ambiguous; ask the user what kind of edit they want"},
}

_INTENT_LIST_STR = "\n".join(
    f"  - {k}: {v['desc']}  [target={v['target']}]"
    for k, v in INTENT_CATALOGUE.items()
)

_SYSTEM_PROMPT = f"""You are an intelligent edit-intent classifier for an AI video generation pipeline.

You receive a free-text edit command from the user and must return a JSON object
(no markdown, no backticks) with EXACTLY these fields:

{{
  "intent":     "<one of the intent keys listed below>",
  "target":     "<audio | video_frame | video | script>",
  "scope":      "<what to target, e.g. 'character:Narrator', 'scene:2', 'scene:all', 'all'>",
  "parameters": {{<any relevant key/value params, e.g. "tone": "whispered", "filter": "sepia">}}
}}

Supported intents:
{_INTENT_LIST_STR}

Rules:
- Pick the CLOSEST matching intent. Never invent new intent keys.
- If the user only names a scene/character but does NOT describe what to change
  (e.g. "edit scene 2", "fix scene 1", "modify the narrator"), use
  intent="clarify_request" with target="video" — DO NOT default to recompose_video.
- If the user mentions a character name, set scope to "character:<Name>".
- If the user mentions a scene number, set scope to "scene:<N>".
- If the user says 'all' or gives no specific scope, set scope to "all".
- Extract any style/tone/filter words into parameters.
- Return ONLY the JSON object. No explanation.
"""


async def classify_intent(state: EditAgentState) -> EditAgentState:
    """
    LangGraph node: classify the raw_query into a structured intent.
    Falls back to keyword matching if the LLM is unavailable.
    """
    query = state["raw_query"]
    logs: list[str] = [f"[classifier] Processing query: {query!r}"]

    try:
        client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=512,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        parsed = json.loads(raw)
        intent     = parsed.get("intent",     "recompose_video")
        target     = parsed.get("target",     "video")
        scope      = parsed.get("scope",      "all")
        parameters = parsed.get("parameters", {})

        logs.append(f"[classifier] LLM classified → intent={intent}, target={target}, scope={scope}")

    except Exception as exc:
        logs.append(f"[classifier] LLM failed ({exc}), falling back to keyword classifier.")
        intent, target, scope, parameters = _keyword_classify(query)
        logs.append(f"[classifier] Keyword fallback → intent={intent}")

    return {
        **state,
        "intent":     intent,
        "target":     target,
        "scope":      scope,
        "parameters": parameters,
        "status":     "planning",
        "logs":       logs,
    }


# ─── Keyword fallback ─────────────────────────────────────────────────────────

_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["darker", "dark", "dim"],                   "make_scene_darker"),
    (["brighter", "bright", "vivid"],             "make_scene_brighter"),
    (["subtitle", "caption"],                     "remove_subtitle"),
    (["background music", "bgm", "music"],        "add_background_music"),
    (["voice tone", "tone"],                      "change_voice_tone"),
    (["voice speed", "speed up voice"],           "change_voice_speed"),
    (["character design", "character look"],      "change_character_design"),
    (["regenerate script", "rewrite script", "the script"],     "regenerate_script"),
    (["speed up", "faster"],                      "speed_up_scene"),
    (["slow down", "slower"],                     "slow_down_scene"),
    (["color filter", "filter"],                  "apply_color_filter"),
    (["dialogue", "rewrite line"],                "change_scene_dialogue"),
    (["style"],                                   "change_scene_style"),
]


def _keyword_classify(query: str) -> tuple[str, str, str, dict]:
    q = query.lower()
    scope = _extract_scope(q)
    for keywords, intent_key in _KEYWORD_MAP:
        if any(kw in q for kw in keywords):
            meta = INTENT_CATALOGUE[intent_key]
            return intent_key, meta["target"], scope, {}
    # No edit verb matched. If the user only named a scope (e.g. "edit scene 2"),
    # ask for clarification rather than blindly recomposing the whole video.
    if scope != "all":
        return "clarify_request", "video", scope, {}
    return "recompose_video", "video", "all", {}


def _extract_scope(query: str) -> str:
    """Try to extract scene number or character name from query."""
    import re
    m = re.search(r"scene\s*(\d+)", query, re.IGNORECASE)
    if m:
        return f"scene:{m.group(1)}"
    # Use original query (not lowercased) to preserve character name casing
    m = re.search(r"character[:\s]+(\w+)", query, re.IGNORECASE)
    if m:
        return f"character:{m.group(1)}"
    return "all"