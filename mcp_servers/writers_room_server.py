# ============================================================
# writers_room_server.py
# MCP Server — All tools for the Writer's Room multi-agent system
# Uses Groq (LLaMA) for LLM calls and ChromaDB for vector memory.
# Run: python mcp_servers/writers_room_server.py
# ============================================================

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
import requests
from groq import Groq
from mcp.server.fastmcp import FastMCP

# ─── Load .env automatically ──────────────────────────────────────────────────
def _load_env():
    for candidate in [
        Path(__file__).parent.parent / ".env",   # project root (writers-room/.env)
        Path(__file__).parent / ".env",           # mcp_servers/.env
    ]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key   = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
            break

_load_env()

# ─── Init ─────────────────────────────────────────────────────────────────────
mcp = FastMCP("writers_room", port=8100)

BASE_DIR   = Path(__file__).parent.parent
MEMORY_DIR = BASE_DIR / "memory"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_DIR  = OUTPUT_DIR / "image_assets"

MEMORY_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

CHROMA_DIR    = MEMORY_DIR / "chroma"
MANIFEST_FILE = OUTPUT_DIR / "scene_manifest.json"
CHAR_DB_FILE  = OUTPUT_DIR / "character_db.json"

# ─── ChromaDB Setup (Vector Memory — satisfies spec §3.3) ─────────────────────
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_memory_collection = _chroma_client.get_or_create_collection(
    name="writers_room_memory",
    metadata={"description": "Persistent memory for scripts, characters, images"},
)

# ─── Groq Setup ───────────────────────────────────────────────────────────────
_groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# `llama-3.1-70b-versatile` was decommissioned by Groq — replaced with the
# current Llama 4 Scout. Order is preference: 70B first, then Scout, then 8B
# instant as a fast fallback.
_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]

# ─── Rate limiter — max 10 calls/min ──────────────────────────────────────────
_last_call_time = 0.0

def _rate_limit():
    global _last_call_time
    gap = 6.0 - (time.time() - _last_call_time)
    if gap > 0:
        time.sleep(gap)
    _last_call_time = time.time()


def _llm(system: str, user: str) -> str:
    """
    LLM call via Groq with automatic retry and model fallback.
    Combines system + user into a single clear instruction for best results.
    """
    _rate_limit()
    last_error = None

    combined_user = f"{system}\n\nUser request: {user}"

    for model in _MODELS:
        for attempt in range(3):
            try:
                response = _groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Always return only valid JSON with no markdown fences, no explanation, no extra text."},
                        {"role": "user",   "content": combined_user},
                    ],
                    temperature=0.7,
                    max_tokens=4096,
                )
                if attempt > 0 or model != _MODELS[0]:
                    print(f"[Groq] Success with model={model} attempt={attempt + 1}")
                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                err_str    = str(e)

                if "429" in err_str or "rate_limit" in err_str.lower():
                    if attempt < 2:
                        wait = 30
                        print(f"[Groq] Rate limited on {model}. Waiting {wait}s (attempt {attempt + 1}/3)...")
                        time.sleep(wait)
                    else:
                        print(f"[Groq] Quota exhausted on {model}, trying next model...")
                        break
                else:
                    raise

    raise RuntimeError(f"All Groq models exhausted. Last error: {last_error}")


def _save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


def _unwrap_llm_output(raw: str) -> str:
    """
    Defensive unwrapper for LLM content blocks.
    Handles plain strings, content-block dicts, and lists of content blocks.
    Strips markdown fences.
    """
    def _strip_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1]
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0]
        return s.strip()

    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _strip_fences(raw)

    if isinstance(parsed, list):
        for block in parsed:
            if isinstance(block, dict) and block.get("type") == "text":
                return _strip_fences(block["text"])
        return raw

    if isinstance(parsed, dict) and parsed.get("type") == "text" and "text" in parsed:
        return _strip_fences(parsed["text"])

    return raw


# ─── Scriptwriter Tools ───────────────────────────────────────────────────────

@mcp.tool()
def generate_script_segment(prompt: str, num_scenes: int = 3) -> str:
    """
    Generate a structured multi-scene screenplay from a user prompt.
    Returns a JSON string with scenes, dialogues, and visual cues.
    """
    system = f"""You are a professional screenplay writer.
Write exactly {num_scenes} scenes for the story idea given by the user.
Keep the film compact: tight pacing, no filler monologues. Aim for satisfying scene beats with enough dialogue to anchor each clip (avoid ultra-sparse exchanges).
Return ONLY a valid JSON object with NO markdown, NO code fences, NO explanation.
Use exactly this structure:
{{
  "title": "actual story title here",
  "genre": "actual genre here",
  "scenes": [
    {{
      "scene_id": 1,
      "location": "specific place name",
      "time_of_day": "day or night or dawn etc",
      "characters": ["Character Name 1", "Character Name 2"],
      "action_description": "1-2 SHORT sentences: only the key physical beat and outcome for this scene",
      "dialogue": [
        {{
          "speaker": "Character Name",
          "line": "what they say",
          "visual_cue": "how they look or move while saying it"
        }}
      ],
      "scene_visual_cue": "overall visual description of the scene"
    }}
  ]
}}
Fill in ALL fields with real content. Do not use placeholder words like 'string' or 'name'.
HONOR THE USER'S CAST AND STYLE: If the idea calls for animals, anthropomorphic
characters, robots, monsters, or a cartoon/animated style (e.g. "a cat named Tom
chasing a mouse named Jerry"), KEEP THAT. Do NOT silently replace non-human
characters with humans, and do NOT convert an animated idea into live action. Set
`genre` to reflect the style (e.g. "animated comedy", "2D cartoon adventure") and
keep every `scene_visual_cue` consistent with it (e.g. include "2D cartoon style"
or "3D animated" plus the actual species: "orange tabby cat", "small grey mouse").
IMPORTANT RULES for scene_visual_cue:
1. KEYWORD FORMAT (MANDATORY — VIDEO GENERATOR CLIP LIMIT): Write `scene_visual_cue` as COMPACT, COMMA-SEPARATED KEYWORDS ONLY. NO full sentences. NO flowery prose. NO abstract metaphors. The field must read like a Stable Diffusion prompt, e.g.: "neon enigma, sci-fi mystery, rainy alleyway, woman detective 30s, wide shot, night, moody lighting" — or for animation: "tom and jerry chase, 2D cartoon style, orange cat, small grey mouse, suburban kitchen, wide shot, day, bright colors". STRICT HARD LIMIT: 40 words maximum. Violating this limit WILL cause the video generation pipeline to fail.
2. REQUIRED KEYWORD CATEGORIES (include all of these): [title slug], [genre], [animation/render style if not live-action: "2D cartoon"/"3D animated"/etc.], [environment/setting], [character: see rule 3], [camera angle], [time of day], [lighting mood].
3. CHARACTER SPECIFICITY (MANDATORY): Describe every character as keywords. For HUMAN characters include gender, approximate age, and build (e.g., "female detective 30s slim", "male villain 50s stocky"). For NON-HUMAN characters give the species/type and key visual traits instead (e.g., "orange tabby cat big eyes", "small grey mouse red bowtie", "rusty boxy robot"). Never describe an animal as if it were a person.
4. CAMERA ANGLE (MANDATORY): Always include one camera angle keyword (wide shot, medium shot, close-up, over-shoulder, tracking shot, low angle, bird's eye).
5. DIALOGUE LENGTH (MANDATORY): Every scene MUST include **2–3 dialogue entries**. Each "line" is ONE natural conversational sentence capped at ~22 words. (For silent-cartoon ideas, dialogue may be short reaction lines / onomatopoeia, but still provide 2–3 entries.)
6. STORY PROGRESSION (MANDATORY): Adjacent scenes must show clear progression — new beat, action change, or reveal.
7. QUALITY CHECK BEFORE RETURN: Verify `scene_visual_cue` is under 40 words and contains only comma-separated keywords. If any scene violates rules 1-6, rewrite it before returning JSON."""

    raw = _llm(system, prompt)

    # ── Server-side CLIP guard ─────────────────────────────────────────────────
    # Even if the LLM ignores the system prompt and generates flowery prose,
    # we truncate scene_visual_cue to ≤40 words before it can reach the video
    # generator and exceed the 77-token CLIP limit.
    try:
        import re as _re
        parsed = json.loads(_unwrap_llm_output(raw))
        mutated = False
        for scene in parsed.get("scenes", []):
            cue = scene.get("scene_visual_cue", "")
            if not cue:
                continue
            words = cue.split()
            if len(words) > 40:
                # Convert prose to compact keyword form: take first 40 words,
                # strip trailing punctuation, and separate by commas.
                keywords = [w.rstrip(".,;:!?") for w in words[:40]]
                scene["scene_visual_cue"] = ", ".join(kw for kw in keywords if kw)
                mutated = True
                print(
                    f"[ScriptWriter Guard] scene_id={scene.get('scene_id')} "
                    f"visual_cue truncated from {len(words)} to ≤40 words."
                )
        if mutated:
            raw = json.dumps(parsed)
    except Exception:
        pass  # If the output isn't JSON yet, let the orchestrator handle it.

    return raw


@mcp.tool()
def validate_script(script_json: str) -> str:
    """
    Validate a manually provided script JSON for structural correctness.
    Returns a JSON object:
      {"valid": bool, "errors": [...], "warnings": [...], "suggestions": [...]}

    Each suggestion is an actionable fix for a corresponding error.
    """
    errors      = []
    warnings    = []
    suggestions = []

    try:
        script = json.loads(script_json)
    except json.JSONDecodeError as e:
        return json.dumps({
            "valid":       False,
            "errors":      [f"Invalid JSON: {e}"],
            "warnings":    [],
            "suggestions": [
                "Verify the file is valid JSON. Check for missing commas, "
                "unclosed brackets, or unescaped quotes."
            ],
        })

    if not isinstance(script, dict):
        return json.dumps({
            "valid":       False,
            "errors":      ["Top-level payload is not a JSON object."],
            "warnings":    [],
            "suggestions": ['Wrap your script in an object: {"scenes": [...]}'],
        })

    scenes = script.get("scenes", [])
    if not scenes:
        errors.append("No 'scenes' array found.")
        suggestions.append(
            "Add a top-level 'scenes' array containing at least one scene object."
        )

    for i, scene in enumerate(scenes):
        sid = scene.get("scene_id", i + 1)

        # Scene header check (PDF §4 Mode 1)
        if not scene.get("location"):
            errors.append(f"Scene {sid}: missing 'location'.")
            suggestions.append(
                f"Scene {sid}: add a 'location' field "
                f"(e.g., 'City Street — Night', 'Abandoned Warehouse')."
            )

        # Action description check (PDF §4 Mode 1)
        if not scene.get("action_description"):
            warnings.append(f"Scene {sid}: missing 'action_description'.")
            suggestions.append(
                f"Scene {sid}: add an 'action_description' describing what "
                f"physically happens in the scene."
            )

        # Dialogue label checks (PDF §4 Mode 1)
        dialogues = scene.get("dialogue", [])
        if not dialogues:
            warnings.append(f"Scene {sid}: no dialogue entries.")
            suggestions.append(
                f"Scene {sid}: add at least one dialogue entry with "
                f"'speaker', 'line', and 'visual_cue' fields."
            )
        elif isinstance(dialogues, list):
            for j, d in enumerate(dialogues):
                if not isinstance(d, dict):
                    errors.append(
                        f"Scene {sid}, dialogue {j}: must be an object, not "
                        f"{type(d).__name__}."
                    )
                    suggestions.append(
                        f"Scene {sid}, dialogue {j}: replace with "
                        f'{{"speaker": "...", "line": "...", "visual_cue": "..."}}'
                    )
                    continue
                if not d.get("speaker"):
                    errors.append(f"Scene {sid}, dialogue {j}: missing 'speaker'.")
                    suggestions.append(
                        f"Scene {sid}, dialogue {j}: add a 'speaker' field "
                        f"with the character's name."
                    )
                if not d.get("line"):
                    errors.append(f"Scene {sid}, dialogue {j}: missing 'line'.")
                    suggestions.append(
                        f"Scene {sid}, dialogue {j}: add a 'line' field with "
                        f"the actual dialogue text."
                    )
                if not d.get("visual_cue"):
                    warnings.append(
                        f"Scene {sid}, dialogue {j}: missing 'visual_cue'."
                    )
                    suggestions.append(
                        f"Scene {sid}, dialogue {j}: add a 'visual_cue' "
                        f"describing blocking, expression, or camera note."
                    )

    return json.dumps({
        "valid":       len(errors) == 0,
        "errors":      errors,
        "warnings":    warnings,
        "suggestions": suggestions,
    })


# ─── Character Designer Tools ─────────────────────────────────────────────────

@mcp.tool()
def extract_characters(script_json: str) -> str:
    """
    Extract and formalize character identities from a script JSON.
    Returns a JSON array of character profiles.
    """
    system = """You are a character designer for films and animation.
Given a screenplay JSON, extract every named character and build a profile for each.
Return ONLY a valid JSON array with NO markdown, NO code fences, NO explanation.
Use exactly this structure:
[
  {
    "character_id": "char_001",
    "name": "actual character name",
    "age_range": "e.g. 30s — or 'n/a' for non-human characters",
    "gender": "male or female or other",
    "species": "human, or the actual species/type — e.g. 'cat', 'mouse', 'robot', 'dragon'",
    "personality_traits": ["trait1", "trait2", "trait3"],
    "appearance": "detailed physical description suitable for image generation",
    "costume": "what they wear (or 'none' / a collar / a bowtie etc. for animals)",
    "reference_style": "photorealistic OR animated — match the screenplay's style",
    "scenes_appeared": [1, 2, 3]
  }
]
Rules:
- If the screenplay is a cartoon / animated piece, set "reference_style" to "animated"; otherwise "photorealistic".
- For NON-HUMAN characters (animals, robots, creatures), set "species" accordingly, describe them as that species in "appearance" (NEVER as a human), use "n/a" for age_range if it doesn't apply, and "other" for gender if unknown.
- Fill ALL fields with real content based on the script. Do not use placeholder words."""

    clean = _unwrap_llm_output(script_json)
    return _llm(system, f"Extract characters from this screenplay:\n{clean}")


@mcp.tool()
def query_stock_footage(character_description: str) -> str:
    """
    Simulate querying a stock footage / reference library for character style references.
    """
    keywords = character_description.lower().split()[:5]
    return json.dumps({
        "query":      character_description,
        "style_tags": keywords,
        "references": [
            {"source": "internal_style_library", "tag": "cinematic_portrait"},
            {"source": "internal_style_library", "tag": "dramatic_lighting"},
        ],
        "suggested_prompt_suffix": "cinematic lighting, professional photography, 8k, highly detailed",
    })


# ─── Image Synthesis Tools ────────────────────────────────────────────────────

@mcp.tool()
def generate_character_image(character_name: str, appearance: str, style: str = "photorealistic") -> str:
    """
    Generate a character reference image.

    Priority order:
      1. Pollinations AI (free, no API key required)
      2. Hugging Face Inference API — serves Stable Diffusion 3.5 / FLUX
         (same models ComfyUI would run locally; accessed via HF for portability)
      3. OpenAI DALL-E 3               — if OPENAI_API_KEY is set
      4. Stub .txt file                — if all providers fail
    """
    import urllib.parse

    safe_name = character_name.replace(" ", "_").lower()
    out_path  = IMAGE_DIR / f"{safe_name}.png"

    style_suffix = {
        "photorealistic": "cinematic portrait, professional photography, 8k resolution, sharp focus",
        "animated":       "2D animation style, vibrant colors, expressive, clean lines",
        "painterly":      "oil painting style, dramatic lighting, artistic brushwork",
    }.get(style, "cinematic portrait, 8k resolution")

    full_prompt = (
        f"Character portrait of {character_name}: {appearance}. "
        f"{style_suffix}. Full character reference sheet, white background."
    )

    # ── 1. Pollinations AI (free, no API key needed) ──────
    print(f"  [Pollinations] Generating image for {character_name}...")
    try:
        encoded_prompt = urllib.parse.quote(full_prompt)
        poll_url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width=1024&height=1024&nologo=true&enhance=true"
        )
        resp = requests.get(poll_url, timeout=120)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
            out_path.write_bytes(resp.content)
            print(f"  [Pollinations] SUCCESS — {out_path}")
            return json.dumps({
                "status":    "success",
                "provider":  "pollinations",
                "character": character_name,
                "file":      str(out_path),
                "prompt":    full_prompt,
            })
        else:
            print(f"  [Pollinations] Failed — HTTP {resp.status_code}: {resp.text[:200]}")
    except requests.exceptions.Timeout:
        print("  [Pollinations] Timed out after 120s — trying next provider")
    except Exception as e:
        print(f"  [Pollinations] Error: {e} — trying next provider")

    # ── 2. Hugging Face (free) — Stable Diffusion 3.5 / FLUX ──
    HF_MODELS = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-3.5-medium",
        "runwayml/stable-diffusion-v1-5",
    ]
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_success = False
        for hf_model in HF_MODELS:
            if hf_success:
                break
            hf_url = (
                f"https://router.huggingface.co/hf-inference/models/"
                f"{hf_model}/v1/text-to-image"
            )
            print(f"  [HF] Trying {hf_model}...")
            for attempt in range(3):
                try:
                    resp = requests.post(
                        hf_url,
                        headers={
                            "Authorization": f"Bearer {hf_token}",
                            "Content-Type":  "application/json",
                        },
                        json={"inputs": full_prompt},
                        timeout=120,
                    )

                    if resp.headers.get("content-type", "").startswith("image/"):
                        out_path.write_bytes(resp.content)
                        print(f"  [HF] SUCCESS — {out_path}")
                        hf_success = True
                        return json.dumps({
                            "status":    "success",
                            "provider":  "huggingface",
                            "model":     hf_model,
                            "character": character_name,
                            "file":      str(out_path),
                        })

                    try:
                        err_json = resp.json()
                    except Exception:
                        err_json = {"error": resp.text[:200]}

                    error_msg = err_json.get("error", str(err_json)) if isinstance(err_json, dict) else str(err_json)
                    estimated = err_json.get("estimated_time", 20)   if isinstance(err_json, dict) else 20

                    if resp.status_code in (401, 403, 404, 410):
                        print(f"  [HF] {hf_model} HTTP {resp.status_code}: {error_msg} — skipping model")
                        break

                    wait = min(float(estimated) + 5, 60)
                    print(f"  [HF] {hf_model} loading (attempt {attempt+1}/3), waiting {wait:.0f}s... {error_msg}")
                    time.sleep(wait)

                except requests.exceptions.Timeout:
                    print(f"  [HF] {hf_model} timed out (attempt {attempt+1}/3)")
                except Exception as e:
                    print(f"  [HF] {hf_model} error: {e} — skipping model")
                    break

    # ── 3. OpenAI DALL-E 3 ───────────────────────────────
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type":  "application/json",
                },
                json={"model": "dall-e-3", "prompt": full_prompt, "n": 1, "size": "1024x1024"},
                timeout=60,
            )
            resp.raise_for_status()
            image_url = resp.json()["data"][0]["url"]
            img_data  = requests.get(image_url, timeout=30).content
            out_path.write_bytes(img_data)
            return json.dumps({
                "status":    "success",
                "provider":  "openai",
                "character": character_name,
                "file":      str(out_path),
                "url":       image_url,
            })
        except Exception as e:
            print(f"  [OpenAI] Failed for {character_name}: {e} — writing stub…")

    # ── 4. Stub fallback ─────────────────────────────────
    stub_path = IMAGE_DIR / f"{safe_name}_stub.txt"
    stub_path.write_text(
        f"[IMAGE STUB]\n"
        f"Character : {character_name}\n"
        f"Style     : {style}\n"
        f"Prompt    : {full_prompt}\n\n"
        f"All image providers failed. Check your internet connection.\n"
    )
    return json.dumps({
        "status":    "stub",
        "provider":  "none",
        "character": character_name,
        "file":      str(stub_path),
        "note":      "All providers failed. Check internet connection.",
    })


# ─── Memory Tools (ChromaDB Vector Store) ─────────────────────────────────────

@mcp.tool()
def commit_memory(key: str, value: str, category: str = "general") -> str:
    """
    Persist data to the shared vector memory store (ChromaDB).

    Embeds the value, then upserts it keyed by a unique id. The original
    `key` and `category` are stored as metadata, so retrieval can filter
    by category or match by keyword / semantic similarity.

    Args:
        key:      Logical identifier for this memory entry.
        value:    JSON string or text to store and embed.
        category: Memory category — script | character | image | general
    """
    entry_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat()

    try:
        _memory_collection.add(
            ids=[entry_id],
            documents=[value],
            metadatas=[{
                "key":       key,
                "category":  category,
                "timestamp": timestamp,
            }],
        )
        return json.dumps({
            "status":   "committed",
            "id":       entry_id,
            "key":      key,
            "category": category,
            "backend":  "chromadb",
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error":  f"ChromaDB commit failed: {e}",
            "key":    key,
        })


@mcp.tool()
def query_memory(category: str = "", keyword: str = "", limit: int = 20) -> str:
    """
    Query the shared vector memory store (ChromaDB).

    If `keyword` is given, performs a semantic similarity search.
    If only `category` is given, returns the most recent entries in that category.
    If neither is given, returns the most recent entries overall.

    Args:
        category: Filter by category (script | character | image | general).
        keyword:  Semantic search query — finds conceptually related entries.
        limit:    Max number of results (default 20).
    """
    try:
        where_filter = {"category": category} if category else None

        if keyword:
            results = _memory_collection.query(
                query_texts=[keyword],
                n_results=limit,
                where=where_filter,
            )
            entries = []
            ids      = (results.get("ids")       or [[]])[0]
            docs     = (results.get("documents") or [[]])[0]
            metas    = (results.get("metadatas") or [[]])[0]
            dists    = (results.get("distances") or [[]])[0]
            for i, doc_id in enumerate(ids):
                entries.append({
                    "id":        doc_id,
                    "key":       metas[i].get("key")       if i < len(metas) else "",
                    "category":  metas[i].get("category")  if i < len(metas) else "",
                    "timestamp": metas[i].get("timestamp") if i < len(metas) else "",
                    "value":     docs[i]                   if i < len(docs)  else "",
                    "distance":  dists[i]                  if i < len(dists) else None,
                })
        else:
            # No keyword — return recent entries, filtered by category if given.
            results = _memory_collection.get(
                where=where_filter,
                limit=limit,
            )
            entries = []
            ids   = results.get("ids")       or []
            docs  = results.get("documents") or []
            metas = results.get("metadatas") or []
            for i, doc_id in enumerate(ids):
                entries.append({
                    "id":        doc_id,
                    "key":       metas[i].get("key")       if i < len(metas) else "",
                    "category":  metas[i].get("category")  if i < len(metas) else "",
                    "timestamp": metas[i].get("timestamp") if i < len(metas) else "",
                    "value":     docs[i]                   if i < len(docs)  else "",
                })
            entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

        return json.dumps({
            "count":   len(entries),
            "entries": entries,
            "backend": "chromadb",
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error":  f"ChromaDB query failed: {e}",
            "count":  0,
            "entries": [],
        })


# ─── Output Commit Tools ──────────────────────────────────────────────────────

@mcp.tool()
def save_scene_manifest(script_json: str) -> str:
    """Save the finalized scene manifest JSON to outputs/scene_manifest.json."""
    try:
        clean = _unwrap_llm_output(script_json)
        data  = json.loads(clean)

        if not isinstance(data, dict) or "scenes" not in data:
            return json.dumps({
                "status": "error",
                "error":  "Parsed payload is not a valid script object (missing 'scenes' key).",
                "hint":   "Make sure you pass the output of generate_script_segment directly.",
            })

        data["generated_at"] = datetime.utcnow().isoformat()
        _save_json(MANIFEST_FILE, data)
        return json.dumps({
            "status":      "saved",
            "path":        str(MANIFEST_FILE),
            "title":       data.get("title", "unknown"),
            "scene_count": len(data["scenes"]),
        })
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"JSON decode failed: {e}"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool()
def save_character_db(characters_json: str) -> str:
    """Save the finalized character database to outputs/character_db.json."""
    try:
        clean = _unwrap_llm_output(characters_json)
        data  = json.loads(clean)

        if not isinstance(data, list):
            return json.dumps({
                "status": "error",
                "error":  "Parsed payload is not a JSON array of characters.",
                "hint":   "Make sure you pass the output of extract_characters directly.",
            })

        db = {"generated_at": datetime.utcnow().isoformat(), "characters": data}
        _save_json(CHAR_DB_FILE, db)
        return json.dumps({
            "status": "saved",
            "path":   str(CHAR_DB_FILE),
            "count":  len(data),
        })
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"JSON decode failed: {e}"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[Writer's Room MCP] ChromaDB collection: {_memory_collection.name}")
    print(f"[Writer's Room MCP] Persist directory:   {CHROMA_DIR}")
    print(f"[Writer's Room MCP] Existing entries:    {_memory_collection.count()}")
    mcp.run(transport="streamable-http")