#!/usr/bin/env python3
# ============================================================
# pipeline.py — Cross-phase orchestrator
#
# Drives Phase 1 → Phase 2 → Phase 3 end-to-end from a single prompt.
#
# Each phase compiles its own LangGraph and runs to completion. Outputs
# from each phase are persisted to disk + MCP memory, so phases stay
# loosely coupled — you can also re-run any phase in isolation via its
# own main.py.
#
# Usage:
#   python -m agents.orchestrator.pipeline --prompt "A cyberpunk thriller..."
#   python -m agents.orchestrator.pipeline --demo
#   python -m agents.orchestrator.pipeline --skip-phase1 --phase1-dir data/outputs
# ============================================================

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ─── Canonical state file location ───────────────────────────────────────────
# This MUST match OUTPUT_DIR in mcp_servers/writers_room_server.py.
# writers_room_server saves to <project_root>/outputs/scene_manifest.json.
# Passing any other directory to serialize() or to Phase 2/3 causes
# the "No such file or directory: '...data/outputs/scene_manifest.json'" bug.
OUTPUTS_DIR = _ROOT / "outputs"

DEMO_PROMPT = (
    "A cyberpunk noir thriller set in 2087 Neo-Tokyo. "
    "Detective Yuki investigates the disappearance of an AI rights activist "
    "named Kael who may have uploaded his consciousness into the city's network. "
    "Her only ally is ARIA, a rogue AI with questionable motives."
)

# Subdirectories of OUTPUTS_DIR that hold generated artifacts.
# These are wiped at the start of every fresh run_all() to prevent ghost files
# from previous projects leaking into new compositions.
_ARTIFACT_SUBDIRS = [
    "audio",
    "bgm",
    "final",
    "frames",
    "image_assets",
    "raw_scenes",
    "video",
]

# Root-level JSON state files written by writers_room_server.
_STATE_FILES = [
    "scene_manifest.json",
    "character_db.json",
    # Spec-compliant artifacts produced by the serializer:
    "story.json",
    "characters.json",
    "script.json",
    "phase2_audio_handoff.json",
    "phase3_video_handoff.json",
    "summary.json",
]


def _wipe_outputs(outputs_dir: Path) -> None:
    """
    Wipe all generated artifacts from a previous run so ghost files cannot
    contaminate the new composition.

    Strategy:
      • Each artifact subdirectory is deleted with shutil.rmtree and then
        immediately recreated empty — so MCP servers that call mkdir(exist_ok=True)
        never race against a missing parent.
      • Only the known state JSON files at the root are removed; the root
        directory itself and any unrecognised files are left untouched.
    """
    print("\n[Orchestrator] ▶ Wiping outputs from previous run...")
    removed_dirs  = 0
    removed_files = 0

    for subdir_name in _ARTIFACT_SUBDIRS:
        subdir = outputs_dir / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir, ignore_errors=True)
            removed_dirs += 1
        subdir.mkdir(parents=True, exist_ok=True)   # recreate empty

    for fname in _STATE_FILES:
        fpath = outputs_dir / fname
        if fpath.exists():
            fpath.unlink()
            removed_files += 1

    print(
        f"[Orchestrator] ✓ Cleared {removed_dirs} artifact dir(s) "
        f"and {removed_files} state file(s). Fresh slate ready.\n"
    )


# ─── Phase 1 ─────────────────────────────────────────────────────────────────

async def run_phase1(prompt: str, output_dir: Path) -> dict:
    print("\n" + "█" * 60)
    print("  PHASE 1 — STORY, SCRIPT & CHARACTER DESIGN")
    print("█" * 60)

    from agents.story_agent.graph import build_graph as build_phase1
    graph = build_phase1()

    initial_state = {
        "input_mode":          "auto",
        "user_input":          prompt,
        "script":              {},
        "script_valid":        False,
        "validation_errors":   [],
        "validation_warnings": [],
        "hitl_approved":       False,
        "hitl_feedback":       "",
        "characters":          [],
        "images":              [],
        "status":              "processing",
        "errors":              [],
        "current_node":        "",
    }

    final = await graph.ainvoke(initial_state)
    print(f"[Orchestrator] Phase 1 status: {final.get('status')}")

    # Run serializer to produce spec-compliant artifacts (script.json, etc.).
    # Both phase1_outputs and out_dir MUST point to OUTPUTS_DIR so the
    # serializer reads the scene_manifest.json the MCP server just wrote.
    print("\n[Orchestrator] Serializing Phase 1 outputs to spec format...")
    try:
        from agents.story_agent.serializer import serialize
        serialize(output_dir, output_dir)   # output_dir is always OUTPUTS_DIR
    except Exception as e:
        # Treat serializer failure as fatal — do NOT continue with stale data.
        print(f"[Orchestrator] ✗ Serializer failed: {e}")
        print("[Orchestrator] Aborting — scene_manifest.json may be missing or corrupt.")
        raise RuntimeError(f"Serializer failed: {e}") from e

    return final


# ─── Phase 2 ─────────────────────────────────────────────────────────────────

async def run_phase2(phase1_dir: Path) -> dict:
    print("\n" + "█" * 60)
    print("  PHASE 2 — AUDIO GENERATION")
    print("█" * 60)

    from agents.audio_agent.graph import build_graph as build_phase2
    graph = build_phase2()

    # ── Canonical state file: scene_manifest.json written by Phase 1 MCP tool.
    # script_path is intentionally left empty — Phase 2 must not silently load
    # a stale script.json when the real manifest is missing.
    manifest_path = phase1_dir / "scene_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"[Phase 2] scene_manifest.json not found at {manifest_path}. "
            "Phase 1 must complete successfully before running Phase 2."
        )

    initial_state: dict = {
        "scene_manifest_path": str(manifest_path),
        "character_db_path":   str(phase1_dir / "character_db.json"),
        "script_path":         "",   # Never fall back to stale script.json
        "audio_tracks":        {},
        "errors":              [],
    }
    final = await graph.ainvoke(initial_state)
    print(f"[Orchestrator] Phase 2 status: {final.get('status')}")
    return final


# ─── Phase 3 ─────────────────────────────────────────────────────────────────

async def run_phase3(phase1_dir: Path, audio_tracks: dict) -> dict:
    print("\n" + "█" * 60)
    print("  PHASE 3 — VIDEO GENERATION & COMPOSITION")
    print("█" * 60)

    from agents.video_agent.graph import build_graph as build_phase3
    graph = build_phase3()

    # ── Canonical state file: same scene_manifest.json used by all phases.
    # script_path is intentionally empty — Phase 3 must not silently fall back
    # to a stale script.json carrying an old project title.
    manifest_path = phase1_dir / "scene_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"[Phase 3] scene_manifest.json not found at {manifest_path}. "
            "Phase 1 must complete successfully before running Phase 3."
        )

    initial_state: dict = {
        "scene_manifest_path": str(manifest_path),
        "character_db_path":   str(phase1_dir / "character_db.json"),
        "script_path":         "",   # Never fall back to stale script.json
        "audio_tracks":        audio_tracks or {},
        "video_clips":         {},
        "face_swapped_clips":  {},
        "final_scenes":        {},
        "errors":              [],
    }
    final = await graph.ainvoke(initial_state)
    print(f"[Orchestrator] Phase 3 status: {final.get('status')}")
    return final


# ─── Driver ──────────────────────────────────────────────────────────────────

async def run_all(prompt: str, output_dir: Path,
                  skip_phase1: bool = False,
                  skip_phase2: bool = False,
                  phase1_dir: Path = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always use the canonical OUTPUTS_DIR as the single source of truth.
    # Ignore any user-supplied output_dir discrepancy — Phase 1 MCP server
    # always writes to OUTPUTS_DIR, so that is where all phases must read from.
    p1_dir = OUTPUTS_DIR

    # Wipe artifacts from any previous run BEFORE Phase 1 writes new state.
    # Skipped when --skip-phase1 is active (user intentionally reuses Phase 1
    # outputs) or when only re-running Phase 2/3 in isolation.
    if not skip_phase1:
        _wipe_outputs(p1_dir)

    # Default HITL to auto-approve when running the full pipeline
    os.environ.setdefault("HITL_AUTO_APPROVE", "1")

    p1_result = {}
    p2_result = {}
    p3_result = {}

    if not skip_phase1:
        p1_result = await run_phase1(prompt, p1_dir)
        if p1_result.get("status") == "failed":
            print("\n[Orchestrator] ✗ Phase 1 failed — aborting")
            return {"phase1": p1_result}

    if not skip_phase2:
        p2_result = await run_phase2(p1_dir)
        if p2_result.get("status") == "failed":
            print("\n[Orchestrator] ✗ Phase 2 failed — aborting")
            return {"phase1": p1_result, "phase2": p2_result}

    audio_tracks = p2_result.get("audio_tracks", {}) or {}
    p3_result = await run_phase3(p1_dir, audio_tracks)

    print("\n" + "█" * 60)
    print("  END-TO-END COMPLETE")
    print("█" * 60)
    print(f"  Phase 1: {p1_result.get('status', 'skipped'):<10} | "
          f"Phase 2: {p2_result.get('status', 'skipped'):<10} | "
          f"Phase 3: {p3_result.get('status', 'unknown')}")
    print(f"  Final video: {p3_result.get('final_video', 'N/A')}")
    print("█" * 60 + "\n")

    return {
        "phase1": p1_result,
        "phase2": p2_result,
        "phase3": p3_result,
    }


async def run_targeted_rerun(
    entry_phase: str,
    phase1_dir: Path,
    prompt: str = "",
) -> dict:
    """
    Re-enter pipeline at a specific phase for edit operations.
    entry_phase:
      - phase1: run 1 -> 2 -> 3
      - phase2: run 2 -> 3
      - phase3: run 3 only
    """
    phase1_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HITL_AUTO_APPROVE", "1")

    p1_result: dict = {}
    p2_result: dict = {}
    p3_result: dict = {}

    if entry_phase == "phase1":
        p1_result = await run_phase1(prompt, phase1_dir)
        if p1_result.get("status") == "failed":
            return {"phase1": p1_result}
        p2_result = await run_phase2(phase1_dir)
        if p2_result.get("status") == "failed":
            return {"phase1": p1_result, "phase2": p2_result}
        p3_result = await run_phase3(phase1_dir, p2_result.get("audio_tracks", {}) or {})
        return {"phase1": p1_result, "phase2": p2_result, "phase3": p3_result}

    if entry_phase == "phase2":
        p2_result = await run_phase2(phase1_dir)
        if p2_result.get("status") == "failed":
            return {"phase2": p2_result}
        p3_result = await run_phase3(phase1_dir, p2_result.get("audio_tracks", {}) or {})
        return {"phase2": p2_result, "phase3": p3_result}

    if entry_phase == "phase3":
        p3_result = await run_phase3(phase1_dir, audio_tracks={})
        return {"phase3": p3_result}

    raise ValueError(f"Unsupported entry_phase '{entry_phase}'")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end orchestrator (Phase 1 → 2 → 3)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", "-p", help="Story prompt for Phase 1")
    group.add_argument("--demo",   "-d", action="store_true", help="Run built-in demo prompt")

    # NOTE: --output-dir is kept for backwards CLI compatibility but is no longer
    # used as the canonical path.  All phases now resolve to OUTPUTS_DIR
    # (project_root/outputs/) which matches where writers_room_server.py saves.
    parser.add_argument("--output-dir", "-o", default=str(OUTPUTS_DIR),
                        help="Deprecated — kept for CLI compatibility. Actual "
                             "artifacts always go to outputs/ in the project root.")
    parser.add_argument("--phase1-dir", help="(Deprecated) Ignored — use OUTPUTS_DIR.")
    parser.add_argument("--skip-phase1", action="store_true", help="Use existing Phase 1 outputs")
    parser.add_argument("--skip-phase2", action="store_true", help="Use existing Phase 2 outputs from MCP memory")
    args = parser.parse_args()

    if args.demo:
        prompt = DEMO_PROMPT
    elif args.prompt:
        prompt = args.prompt
    elif args.skip_phase1:
        prompt = ""  # not needed when skipping
    else:
        parser.error("Provide --prompt, --demo, or --skip-phase1")

    asyncio.run(run_all(
        prompt      = prompt,
        output_dir  = OUTPUTS_DIR,
        skip_phase1 = args.skip_phase1,
        skip_phase2 = args.skip_phase2,
    ))


if __name__ == "__main__":
    main()
