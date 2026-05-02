# THE WRITER'S ROOM
## PROJECT MONTAGE — Phase 1
### Autonomous Story and Image Generation Layer

---

## Overview

The Writer's Room is a **multi-agent creative system** built on LangGraph that transforms raw human intent into structured, machine-interpretable narrative representations. It implements a Supervisor-Worker hierarchical model with 5 specialized agents communicating through shared LangGraph state and MCP-based dynamic tool discovery.

---

## Architecture

```
START
  │
  ▼
mode_selector_node          ← detects manual vs auto input
       │
  ┌────┴─────┐
  ▼          ▼
validator   scriptwriter    ← Agents 1 & 2 (dual-mode ingestion)
  │          │
  └────┬─────┘
       ▼
  hitl_node                 ← Human-in-the-Loop checkpoint
       │
       ▼
  character_node            ← Character Designer Agent
       │
       ▼
  image_node                ← Image Synthesizer Agent
       │
       ▼
  memory_commit_node        ← Persists all outputs
       │
      END
```

### MCP Tool Discovery
All agents query the MCP registry at runtime — **no hardcoded API calls**.

| Tool | Agent | Description |
|---|---|---|
| `generate_script_segment` | Scriptwriter | Generates scenes/dialogue/cues via LLM |
| `validate_script` | Validator | Checks scene headings, dialogue labels, actions |
| `extract_characters` | Character Designer | Builds character profiles from script |
| `query_stock_footage` | Character Designer | Style reference lookup |
| `generate_character_image` | Image Synthesizer | DALL-E 3 image generation |
| `commit_memory` | All | Persist to shared memory store |
| `query_memory` | All | Retrieve from memory store |
| `save_scene_manifest` | Memory Commit | Write `scene_manifest.json` |
| `save_character_db` | Memory Commit | Write `character_db.json` |

---

## File Structure

```
writers-room/
├── main.py                          # CLI entry point
├── graph.py                         # LangGraph StateGraph definition
├── requirements.txt
├── sample_script.json               # Test script for manual mode
├── agents/
│   ├── __init__.py
│   ├── state.py                     # AgentState TypedDict
│   └── nodes.py                     # All agent node implementations
├── mcp_servers/
│   └── writers_room_server.py       # MCP server (all tools)
├── memory/
│   └── memory_store.json            # Runtime — auto-created
└── outputs/                         # Runtime — auto-created
    ├── scene_manifest.json
    ├── character_db.json
    └── image_assets/
        └── *.png  (or *_stub.txt)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Required
export OPENAI_API_KEY=sk-...               # Optional (for real images)
export HITL_AUTO_APPROVE=1                 # Set to skip interactive review
```

### 3. Start the MCP server (in a separate terminal)
```bash
cd writers-room
python mcp_servers/writers_room_server.py
```

### 4. Run the pipeline
```bash
# Demo (built-in cyberpunk prompt)
python main.py --demo

# Custom prompt
python main.py --prompt "A medieval heist story where three thieves..."

# Manual script (Mode 1)
python main.py --script sample_script.json

# Interactive CLI
python main.py
```

---

## Agent Definitions

### 1. Scriptwriter Agent (`scriptwriter_node`)
**Role:** Transforms abstract prompts into structured, production-ready scripts.

**Reasoning Loop:**
1. Receives user goal/prompt
2. Invokes `generate_script_segment` via MCP (calls Claude Haiku internally)
3. Parses JSON response into structured `AgentState.script`
4. Commits script to memory via `commit_memory`

**MCP Tools:** `generate_script_segment`, `commit_memory`

---

### 2. Script Validator Agent (`validator_node`)
**Role:** Ensures correctness of manually provided scripts.

**Validation Checks:**
- Scene headers (`location` field exists)
- Dialogue labels (`speaker` field on every line)
- Action descriptions present

**Failure Handling:** Rejects script → routes to END with error list

**MCP Tools:** `validate_script`, `commit_memory`

---

### 3. Human-in-the-Loop Agent (`hitl_node`)
**Role:** Checkpoint before irreversible generation (images, final outputs).

**Why required:**
- Prevents hallucinated scripts from wasting image generation credits
- Ensures user intent alignment before downstream processing
- Supports feedback loop for script revision

**Integration:** Set `HITL_AUTO_APPROVE=1` for automated pipelines; unset for interactive review.

---

### 4. Character Designer Agent (`character_node`)
**Role:** Extracts and formalizes character identities from the approved script.

**Outputs per character:**
- Name, age range, gender
- Personality traits (list)
- Detailed appearance description (for image generation)
- Costume description
- Reference style (photorealistic / animated / painterly)
- Scenes appeared in
- Style references from stock library

**Key Feature:** Identity consistency enforced via `character_id` + memory commitment.

**MCP Tools:** `extract_characters`, `query_stock_footage`, `commit_memory`

---

### 5. Image Synthesizer Agent (`image_node`)
**Role:** Generates visual character reference images.

**Implementation:** DALL-E 3 via OpenAI API (accessed through MCP tool `generate_character_image`).  
**Fallback:** When `OPENAI_API_KEY` is not set, saves a stub `.txt` file with the generation prompt.

**Output:** `outputs/image_assets/<character_name>.png`

**MCP Tools:** `generate_character_image`, `commit_memory`

---

## State Schema

```python
class AgentState(TypedDict):
    input_mode:           "manual" | "auto"
    user_input:           str          # raw prompt or script text
    script:               dict         # parsed script
    script_valid:         bool
    validation_errors:    list[str]
    validation_warnings:  list[str]
    hitl_approved:        bool
    hitl_feedback:        str
    characters:           list[dict]
    images:               list[dict]   # uses operator.add reducer
    status:               str          # processing | complete | failed
    errors:               list[str]    # uses operator.add reducer
    current_node:         str
```

---

## Output Files

### `scene_manifest.json`
```json
{
  "title": "...",
  "genre": "...",
  "generated_at": "2026-04-05T...",
  "scenes": [
    {
      "scene_id": 1,
      "location": "City Street",
      "time_of_day": "Night",
      "characters": ["Alice", "Bob"],
      "action_description": "...",
      "dialogue": [
        { "speaker": "Alice", "line": "...", "visual_cue": "..." }
      ],
      "scene_visual_cue": "..."
    }
  ]
}
```

### `character_db.json`
```json
{
  "generated_at": "...",
  "characters": [
    {
      "character_id": "char_001",
      "name": "Alice",
      "age_range": "late 20s",
      "personality_traits": ["determined", "analytical"],
      "appearance": "...",
      "reference_style": "photorealistic",
      "scenes_appeared": [1, 2]
    }
  ]
}
```

---

## Evaluation Criteria

| Criteria | Implementation |
|---|---|
| **Agent Definition** (20 pts) | 5 agents with clear roles, reasoning loops in `nodes.py` |
| **Script Generation Quality** (15 pts) | Multi-scene, structured JSON with dialogue + visual cues |
| **MCP Integration** (15 pts) | All tools in `writers_room_server.py`, discovered dynamically via `_call_tool()` |
| **LangGraph Workflow** (10 pts) | Full `StateGraph` in `graph.py` with typed state + conditional routing |
| **Human-in-the-Loop** (10 pts) | `hitl_node` with approval gate, env-var bypass for testing |
| **Output Completeness** (5 pts) | `scene_manifest.json`, `character_db.json`, `image_assets/` all generated |

---

## Notes

- Memory persistence uses a flat JSON file (`memory/memory_store.json`) simulating a vector DB. Swap `commit_memory` / `query_memory` implementations for ChromaDB/FAISS without changing any agent code.
- The system is LLM-agnostic at the agent level — swap `claude-haiku-4-5-20251001` in the MCP server to any other model.
- All external calls go through MCP tools — agents never directly import `anthropic`, `openai`, or `requests`.
