# ============================================================
# nodes.py
# All LangGraph node implementations for the Writer's Room
# Each node uses MCP tools dynamically — no hardcoded API calls.
# ============================================================

import asyncio
import json
import os
import re

from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.story_agent.state import AgentState

# ─── MCP Client config ──────────────────────────────────────────────

MCP_CONFIG = {
    "writers_room": {
        "url":       "http://localhost:8100/mcp",
        "transport": "streamable_http",
    }
}

# Max retries and base delay (seconds) for transient TCP errors (WinError 10054)
_MCP_RETRIES = 3
_MCP_RETRY_BASE_DELAY = 2.0


async def _call_tool(tool_name: str, **kwargs) -> str:
    """
    Call a named MCP tool, returning a plain string.

    Creates a fresh MultiServerMCPClient per call so TCP connections are not
    held open between tool invocations (prevents WinError 10054 on Windows).
    Note: langchain-mcp-adapters >=0.1.0 does NOT support async-with on the
    client directly; use client.get_tools() without a context manager.
    Wraps the network request in retry logic with exponential backoff for
    transient connection resets.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MCP_RETRIES + 1):
        try:
            # Fresh client per attempt — intentionally NOT cached.
            client   = MultiServerMCPClient(MCP_CONFIG)
            tools    = await client.get_tools(server_name="writers_room")
            tool_map = {t.name: t for t in tools}

            if tool_name not in tool_map:
                raise ValueError(
                    f"Tool '{tool_name}' not found in MCP registry. "
                    f"Available: {list(tool_map.keys())}"
                )

            result = await tool_map[tool_name].ainvoke(kwargs)
            return _extract_text(result)

        except (ConnectionResetError, ConnectionAbortedError) as exc:
            last_exc = exc
            delay = _MCP_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"[MCP] ConnectionReset on '{tool_name}' "
                f"(attempt {attempt}/{_MCP_RETRIES}): {exc}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

        except Exception as exc:
            # Catch httpx RemoteProtocolError which wraps WinError 10054
            exc_str = str(exc)
            if "10054" in exc_str or "RemoteProtocol" in type(exc).__name__:
                last_exc = exc
                delay = _MCP_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[MCP] RemoteProtocolError on '{tool_name}' "
                    f"(attempt {attempt}/{_MCP_RETRIES}): {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                raise  # Non-transient errors propagate immediately.

    raise RuntimeError(
        f"MCP tool '{tool_name}' failed after {_MCP_RETRIES} attempts. "
        f"Last error: {last_exc}"
    ) from last_exc


def _extract_text(result) -> str:
    """
    Pull the plain-text string out of whatever LangChain returns.

    LangChain MCP adapters can return any of:
      • str                                      → use directly
      • list of content blocks [{"type":"text","text":"..."}]
      • a single content block dict {"type":"text","text":"..."}
      • a ToolMessage / AIMessage object with a .content attribute
      • anything else                            → json.dumps fallback
    """
    # Already a string — done.
    if isinstance(result, str):
        return result

    # Object with .content (ToolMessage, AIMessage, etc.)
    if hasattr(result, "content"):
        return _extract_text(result.content)

    # List of content blocks — grab first "text" block.
    if isinstance(result, list):
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
            if isinstance(block, str):
                return block
        return json.dumps(result)

    # Single content block dict.
    if isinstance(result, dict) and result.get("type") == "text" and "text" in result:
        return result["text"]

    # Fallback — serialise whatever we got.
    return json.dumps(result)

# ─── Internal helpers ─────────────────────────────────────────────────────────


def _safe_parse(text: str):
    """
    Strip markdown fences and parse JSON from LLM / tool output.

    Robust against three common LLM failure modes:
      1. ```json ... ```           — code fences
      2. "Here are the chars: {…}" — prose preamble before the JSON
      3. "{…} Hope this helps!"    — prose postamble after the JSON
    """
    clean = re.sub(r"```json|```", "", text).strip()

    # Fast path — clean JSON
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Slow path — find the outermost JSON object or array and try just that.
    # Look for the FIRST '{' or '[' and the LAST matching '}' or ']'.
    obj_start = clean.find("{")
    arr_start = clean.find("[")
    candidates = [c for c in (obj_start, arr_start) if c != -1]
    if not candidates:
        # Nothing JSON-like in the response at all.
        raise json.JSONDecodeError("No JSON object/array found in output", clean, 0)

    start = min(candidates)
    # Match the bracket type
    open_ch  = clean[start]
    close_ch = "}" if open_ch == "{" else "]"
    end      = clean.rfind(close_ch)
    if end <= start:
        raise json.JSONDecodeError("Unmatched JSON brackets in output", clean, start)

    snippet = clean[start:end + 1]
    return json.loads(snippet)


def _to_dict(data, fallback: dict) -> dict:
    """
    Ensure data is a dict. If it's a list, take the first element.
    If it's neither, return the fallback.
    """
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data:
        first = data[0]
        return first if isinstance(first, dict) else fallback
    return fallback


def _normalize_script(data) -> dict:
    """
    Ensure the script is always a dict with 'title', 'genre', and 'scenes'.
    Handles all Groq/LLaMA response variations.
    """
    if isinstance(data, list):
        scenes = data
        for i, s in enumerate(scenes, 1):
            if not isinstance(s, dict):
                scenes[i-1] = {"scene_id": i, "location": "Unknown",
                                "action_description": str(s), "dialogue": []}
            elif "scene_id" not in s:
                s["scene_id"] = i
        return {"title": "Untitled", "genre": "Drama", "scenes": scenes}

    if isinstance(data, dict):
        for key in ("scenes", "scene_list", "script", "screenplay", "acts"):
            if key in data and isinstance(data[key], list):
                if key != "scenes":
                    data["scenes"] = data.pop(key)
                break
        else:
            if "location" in data or "dialogue" in data:
                return {"title": "Untitled", "genre": "Drama", "scenes": [data]}
            data["scenes"] = []

        for i, s in enumerate(data.get("scenes", []), 1):
            if isinstance(s, dict) and "scene_id" not in s:
                s["scene_id"] = s.get("id", i)

        return data

    return {"title": "Untitled", "genre": "Drama", "scenes": []}


def _normalize_characters(data) -> list:
    """
    Ensure characters is always a list of dicts.
    Handles Groq returning a dict with nested list, or a single dict.
    """
    if isinstance(data, list):
        return [c if isinstance(c, dict) else {"name": str(c)} for c in data]
    if isinstance(data, dict):
        for key in ("characters", "character_list", "cast"):
            if key in data and isinstance(data[key], list):
                return data[key]
        if "name" in data:
            return [data]
    return []


# ─── Node 1: Mode Selector ────────────────────────────────────────────────────

def mode_selector_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "").strip()

    try:
        parsed = json.loads(user_input)
        if "scenes" in parsed:
            return {
                "input_mode":   "manual",
                "script":       parsed,
                "status":       "processing",
                "current_node": "mode_selector_node",
            }
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "input_mode":   "auto",
        "status":       "processing",
        "current_node": "mode_selector_node",
    }


# ─── Node 2a: Scriptwriter Agent ──────────────────────────────────────────────

async def scriptwriter_node(state: AgentState) -> dict:
    print("[Scriptwriter] Generating script from prompt…")

    prompt     = state.get("user_input", "")
    num_scenes = 3

    raw = await _call_tool("generate_script_segment", prompt=prompt, num_scenes=num_scenes)
    print(f"[Scriptwriter] Raw tool output (first 300 chars): {raw[:300]}")

    try:
        parsed = _safe_parse(raw)
    except json.JSONDecodeError:
        return {
            "status":       "failed",
            "errors":       [f"Scriptwriter could not parse LLM output: {raw[:200]}"],
            "current_node": "scriptwriter_node",
        }

    script = _normalize_script(parsed)

    print(f"[Scriptwriter] Generated '{script.get('title', 'Untitled')}' with "
          f"{len(script.get('scenes', []))} scenes.")

    await _call_tool(
        "commit_memory",
        key=f"script_{script.get('title', 'untitled').replace(' ', '_')}",
        value=json.dumps(script),
        category="script",
    )

    return {
        "script":       script,
        "status":       "processing",
        "current_node": "scriptwriter_node",
    }


# ─── Node 2b: Script Validator Agent ─────────────────────────────────────────

async def validator_node(state: AgentState) -> dict:
    print("[Validator] Validating uploaded script…")

    script_json = json.dumps(state.get("script", {}))
    raw         = await _call_tool("validate_script", script_json=script_json)

    try:
        result = _safe_parse(raw)
        result = _to_dict(result, {"valid": False, "errors": ["Unexpected response format."], "warnings": []})
    except json.JSONDecodeError:
        result = {"valid": False, "errors": ["Validator returned invalid JSON."], "warnings": []}

    is_valid = result.get("valid", False)
    errors   = result.get("errors", [])
    warnings = result.get("warnings", [])

    print(f"[Validator] valid={is_valid}  errors={errors}  warnings={warnings}")
    print(f"[Validator] {'✓ Script is structurally valid.' if not errors else f'✗ {len(errors)} error(s) found.'}")

    await _call_tool(
        "commit_memory",
        key="last_validation_result",
        value=json.dumps(result),
        category="general",
    )

    return {
        "script_valid":        is_valid,
        "validation_errors":   errors,
        "validation_warnings": warnings,
        "status":              "processing" if is_valid else "failed",
        "current_node":        "validator_node",
        "errors":              errors if not is_valid else [],
    }


# ─── Node 3: Human-in-the-Loop ────────────────────────────────────────────────

def hitl_node(state: AgentState) -> dict:
    script = state.get("script", {})
    title  = script.get("title", "Untitled")
    scenes = script.get("scenes", [])

    print("\n" + "=" * 60)
    print("HUMAN-IN-THE-LOOP REVIEW")
    print("=" * 60)
    print(f"Title  : {title}")
    print(f"Genre  : {script.get('genre', 'N/A')}")
    print(f"Scenes : {len(scenes)}")
    print()

    for i, s in enumerate(scenes, 1):
        if not isinstance(s, dict):
            print(f"  Scene {i}: [Unknown] — (unparseable scene)")
            continue
        sid      = s.get("scene_id") or s.get("id") or i
        location = s.get("location") or s.get("setting") or "Unknown"
        desc     = s.get("action_description") or s.get("description") or s.get("summary") or ""
        print(f"  Scene {sid}: [{location}] — {str(desc)[:80]}...")
    print()

    auto_approve = os.environ.get("HITL_AUTO_APPROVE", "").lower() in ("1", "true", "yes")

    if auto_approve:
        approved = True
        feedback = "Auto-approved."
    else:
        answer   = input("Approve script? [y/N] (or enter feedback): ").strip()
        approved = answer.lower() in ("y", "yes", "")
        feedback = answer if not approved else "Approved."

    print(f"[HITL] Decision: {'Approved' if approved else 'Rejected'}")
    print("=" * 60 + "\n")

    return {
        "hitl_approved": approved,
        "hitl_feedback": feedback,
        "status":        "processing" if approved else "failed",
        "current_node":  "hitl_node",
        "errors":        [] if approved else [f"Script rejected by human reviewer: {feedback}"],
    }


# ─── Node 4: Character Designer Agent ────────────────────────────────────────

async def character_node(state: AgentState) -> dict:
    print("[Character Designer] Extracting character profiles...")

    script_json = json.dumps(state.get("script", {}))

    raw = await _call_tool("extract_characters", script_json=script_json)
    print(f"[Character Designer] Raw tool output (first 400 chars): {raw[:400]}")

    try:
        parsed     = _safe_parse(raw)
        characters = _normalize_characters(parsed)
    except json.JSONDecodeError as e:
        print(f"[Character Designer] ⚠ Could not parse extract_characters output.")
        print(f"[Character Designer]   Error: {e}")
        print(f"[Character Designer]   Full raw output:")
        print(raw)
        # Don't kill the whole pipeline — synthesize characters from the script's speakers.
        speakers = set()
        for s in state.get("script", {}).get("scenes", []) or []:
            if not isinstance(s, dict):
                continue
            for d in s.get("dialogue", []) or []:
                if isinstance(d, dict) and d.get("speaker"):
                    speakers.add(d["speaker"])
        characters = [
            {"character_id": f"char_{i+1:03d}", "name": name,
             "appearance": "", "personality_traits": [],
             "reference_style": "photorealistic"}
            for i, name in enumerate(sorted(speakers))
        ]
        print(f"[Character Designer] ⓘ Falling back to {len(characters)} character(s) extracted from dialogue speakers.")

    print(f"[Character Designer] Found {len(characters)} character(s).")

    for char in characters:
        if not isinstance(char, dict):
            continue
        try:
            ref_raw = await _call_tool(
                "query_stock_footage",
                character_description=f"{char.get('name', 'Unknown')} — {char.get('appearance', '')}",
            )
            ref = _safe_parse(ref_raw)
            ref = _to_dict(ref, {})
            char["style_references"] = ref.get("references", [])
            char["prompt_suffix"]    = ref.get("suggested_prompt_suffix", "")
        except Exception as e:
            print(f"[Character Designer]   ⚠ stock footage lookup failed for {char.get('name')}: {e}")

        try:
            await _call_tool(
                "commit_memory",
                key=f"character_{char.get('character_id', char.get('name', 'unknown'))}",
                value=json.dumps(char),
                category="character",
            )
        except Exception as e:
            print(f"[Character Designer]   ⚠ memory commit failed for {char.get('name')}: {e}")

    return {
        "characters":   characters,
        "status":       "processing",
        "current_node": "character_node",
    }


# ─── Node 5: Image Synthesizer Agent ─────────────────────────────────────────

async def image_node(state: AgentState) -> dict:
    print("[Image Synthesizer] Generating character images...")
    characters = state.get("characters", [])
    new_images = []

    for char in characters:
        if not isinstance(char, dict):
            continue
        name       = char.get("name") or "Unknown"
        appearance = char.get("appearance", "")
        suffix     = char.get("prompt_suffix", "")
        style      = char.get("reference_style", "photorealistic")
        full_desc  = f"{appearance}. {suffix}".strip(". ")

        print(f"  Generating image for: {name}")

        raw = await _call_tool(
            "generate_character_image",
            character_name=name,
            appearance=full_desc,
            style=style,
        )
        try:
            img_result = _safe_parse(raw)
            img_result = _to_dict(img_result, {"status": "error", "character": name})
        except Exception:
            img_result = {"status": "error", "character": name, "raw": raw}

        new_images.append(img_result)

        await _call_tool(
            "commit_memory",
            key=f"image_{name.replace(' ', '_').lower()}",
            value=json.dumps(img_result),
            category="image",
        )

        print(f"  [{img_result.get('status', 'unknown')}] {name} -> {img_result.get('file', 'N/A')}")

    return {
        "images":       new_images,
        "status":       "processing",
        "current_node": "image_node",
    }


# ─── Node 6: Memory Commit Node ──────────────────────────────────────────────

async def memory_commit_node(state: AgentState) -> dict:
    print("[Memory Commit] Saving final outputs...")
    errors = []

    manifest_raw = await _call_tool(
        "save_scene_manifest",
        script_json=json.dumps(state.get("script", {})),
    )
    try:
        manifest_result = _safe_parse(manifest_raw)
        manifest_result = _to_dict(manifest_result, {})
        print(f"  scene_manifest.json -> {manifest_result.get('path', 'N/A')}")
    except Exception as e:
        errors.append(f"Failed to save scene_manifest: {e}")

    char_raw = await _call_tool(
        "save_character_db",
        characters_json=json.dumps(state.get("characters", [])),
    )
    try:
        char_result = _safe_parse(char_raw)
        char_result = _to_dict(char_result, {})
        print(f"  character_db.json -> {char_result.get('path', 'N/A')} "
              f"({char_result.get('count', 0)} characters)")
    except Exception as e:
        errors.append(f"Failed to save character_db: {e}")

    summary = {
        "title":      state.get("script", {}).get("title", "Untitled"),
        "scenes":     len(state.get("script", {}).get("scenes", [])),
        "characters": len(state.get("characters", [])),
        "images":     len(state.get("images", [])),
        "input_mode": state.get("input_mode"),
    }
    await _call_tool(
        "commit_memory",
        key=f"run_summary_{summary['title'].replace(' ', '_')}",
        value=json.dumps(summary),
        category="general",
    )

    return {
        "status":       "complete",
        "current_node": "memory_commit_node",
        "errors":       errors,
    }