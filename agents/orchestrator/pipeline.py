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
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DEMO_PROMPT = (
    "A cyberpunk noir thriller set in 2087 Neo-Tokyo. "
    "Detective Yuki investigates the disappearance of an AI rights activist "
    "named Kael who may have uploaded his consciousness into the city's network. "
    "Her only ally is ARIA, a rogue AI with questionable motives."
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

    # Run serializer to produce spec-compliant artifacts (script.json, etc.)
    print("\n[Orchestrator] Serializing Phase 1 outputs to spec format...")
    try:
        from agents.story_agent.serializer import serialize
        serialize(output_dir, output_dir)
    except Exception as e:
        print(f"[Orchestrator] Serializer failed: {e}")

    return final


# ─── Phase 2 ─────────────────────────────────────────────────────────────────

async def run_phase2(phase1_dir: Path) -> dict:
    print("\n" + "█" * 60)
    print("  PHASE 2 — AUDIO GENERATION")
    print("█" * 60)

    from agents.audio_agent.graph import build_graph as build_phase2
    graph = build_phase2()

    script_path = phase1_dir / "script.json"
    initial_state: dict = {
        "scene_manifest_path": str(phase1_dir / "scene_manifest.json"),
        "character_db_path":   str(phase1_dir / "character_db.json"),
        "script_path":         str(script_path) if script_path.exists() else "",
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

    script_path = phase1_dir / "script.json"
    initial_state: dict = {
        "scene_manifest_path": str(phase1_dir / "scene_manifest.json"),
        "character_db_path":   str(phase1_dir / "character_db.json"),
        "script_path":         str(script_path) if script_path.exists() else "",
        "audio_tracks":        audio_tracks or {},   # pass directly from Phase 2
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

    # Default: Phase 1 writes to output_dir
    p1_dir = phase1_dir or output_dir

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


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end orchestrator (Phase 1 → 2 → 3)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", "-p", help="Story prompt for Phase 1")
    group.add_argument("--demo",   "-d", action="store_true", help="Run built-in demo prompt")

    parser.add_argument("--output-dir", "-o", default="data/outputs",
                        help="Where Phase 1 writes its artifacts (default: data/outputs)")
    parser.add_argument("--phase1-dir", help="If --skip-phase1, where to find Phase 1 outputs")
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
        parser.error("Provide --prompt, --demo, or --skip-phase1 with --phase1-dir")

    output_dir = Path(args.output_dir)
    p1_dir     = Path(args.phase1_dir) if args.phase1_dir else None

    asyncio.run(run_all(
        prompt      = prompt,
        output_dir  = output_dir,
        skip_phase1 = args.skip_phase1,
        skip_phase2 = args.skip_phase2,
        phase1_dir  = p1_dir,
    ))


if __name__ == "__main__":
    main()
