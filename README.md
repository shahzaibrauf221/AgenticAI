# AgenticAI Project — AI-Powered Animated Video Generation

End-to-end multi-agent system that turns a single natural-language prompt into a polished short animated video.

---

## Phase Mapping

| Phase | Module | Status |
|-------|--------|--------|
| **Phase 1** — Story, Script & Character Design | `agents/story_agent/` (Writer's Room) | ✅ Implemented |
| **Phase 2** — Audio Generation & Integration | `agents/audio_agent/` | ✅ Implemented |
| **Phase 3** — Video Generation & Composition | `agents/video_agent/` | ✅ Implemented |
| **Phase 4** — Web Interface | `backend/`, `frontend/` | ⏳ TODO |
| **Phase 5** — Edit Agent + Undo | `agents/edit_agent/`, `state_manager/` | ⏳ TODO |

---

## Folder Structure

```
AgenticAI_Project/
├── README.md
├── requirements.txt
│
├── shared/
│   └── schemas/
│       └── schemas.py                 # Pydantic schemas — single source of truth
│
├── agents/
│   ├── orchestrator/                  # Cross-phase orchestration
│   │   ├── __init__.py
│   │   └── pipeline.py                # End-to-end Phase 1→2→3 driver
│   │
│   ├── story_agent/                   # Phase 1
│   │   ├── __init__.py
│   │   ├── state.py                   # AgentState (writer's room)
│   │   ├── nodes.py                   # mode_selector, scriptwriter, validator,
│   │   │                              # hitl, character, image, memory_commit
│   │   ├── graph.py                   # LangGraph StateGraph
│   │   ├── main.py                    # CLI entry point
│   │   ├── serializer.py              # → spec-compliant artifacts
│   │   └── tests/
│   │
│   ├── audio_agent/                   # Phase 2 — audio only
│   │   ├── __init__.py
│   │   ├── state.py                   # AgentState (audio)
│   │   ├── nodes.py                   # scene_parser, voice_synth, finalizer
│   │   ├── graph.py
│   │   ├── main.py
│   │   ├── manifest_writer.py         # timing_manifest.json builder
│   │   └── tests/
│   │
│   ├── video_agent/                   # Phase 3 — video only
│   │   ├── __init__.py
│   │   ├── state.py                   # AgentState (video)
│   │   ├── nodes.py                   # video_gen, face_swap, lip_sync, compositor
│   │   ├── graph.py
│   │   ├── main.py
│   │   └── tests/
│   │
│   └── edit_agent/                    # Phase 5 — placeholder
│
├── mcp_servers/
│   ├── writers_room_server.py         # Phase 1 MCP server (port 8100)
│   └── studio_floor_server.py         # Phases 2+3 MCP server (port 8200)
│
├── backend/                           # Phase 4 placeholder
├── frontend/                          # Phase 4 placeholder
├── state_manager/                     # Phase 5 placeholder
│
├── data/
│   ├── outputs/                       # All pipeline outputs
│   ├── temp/
│   └── state_versions/                # Phase 5 snapshots
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── scripts/
```

---

## Inter-Phase Data Flow

```
   prompt
     │
     ▼
┌─────────────────────────┐
│  Phase 1 — story_agent  │   writers_room_server.py (port 8100)
│  Writer's Room          │
└─────────────┬───────────┘
              │  outputs:
              │    • script.json            ← unified spec artifact
              │    • characters.json
              │    • story.json
              │    • scene_manifest.json    ← legacy
              │    • character_db.json      ← legacy
              │    • phase2_audio_handoff.json
              │    • phase3_video_handoff.json
              │
              ▼
┌─────────────────────────┐
│  Phase 2 — audio_agent  │   studio_floor_server.py (port 8200)
│  Voice + BGM            │   (audio-only subset)
└─────────────┬───────────┘
              │  outputs:
              │    • outputs/audio/scene_NN_*.wav
              │    • outputs/audio/scene_NN_full.wav  (dialogue + BGM mix)
              │    • timing_manifest.json
              │
              ▼
┌─────────────────────────┐
│  Phase 3 — video_agent  │   studio_floor_server.py (port 8200)
│  Video + Face + Lip     │   (video-only subset)
└─────────────┬───────────┘
              │  outputs:
              │    • outputs/video/scene_NN_*.mp4    (base)
              │    • outputs/face_swap/...            (face swapped)
              │    • outputs/final/scene_NN_final.mp4 (lip-synced)
              │    • outputs/final/final_output.mp4   (compositor — concat)
              ▼
            END
```

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Environment
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk-...        # optional fallback
export OPENAI_API_KEY=sk-...       # optional, for DALL-E images
export HITL_AUTO_APPROVE=1
```

### 3. Start MCP servers (each in its own terminal)
```bash
# Terminal 1
python mcp_servers/writers_room_server.py     # port 8100

# Terminal 2
python mcp_servers/studio_floor_server.py     # port 8200
```

### 4. Run individual phases
```bash
# Phase 1 only
python -m agents.story_agent.main --demo

# Phase 1 → spec artifacts
python -m agents.story_agent.serializer --phase1 data/outputs --out data/outputs

# Phase 2 only (needs Phase 1 outputs)
python -m agents.audio_agent.main --phase1-dir data/outputs

# Phase 3 only (needs Phase 1 + 2 outputs)
python -m agents.video_agent.main --phase1-dir data/outputs

# Full end-to-end
python -m agents.orchestrator.pipeline --prompt "A cyberpunk thriller..."
```

---

## Notes for the Team

- The shared JSON schema lives in `shared/schemas/schemas.py`. **Do not** redefine these types per phase — import them.
- Each phase's `state.py` defines its own LangGraph `AgentState` (this is allowed; LangGraph state is per-graph).
- The MCP servers `writers_room_server.py` and `studio_floor_server.py` are **owned by Members 1 and 2/3 respectively** — keep them in `mcp_servers/`.
- The `mcp/` folder (per the original scaffold zip) is reserved for **future MCP tool abstractions**. We're using FastMCP servers for now, which is simpler and works.
