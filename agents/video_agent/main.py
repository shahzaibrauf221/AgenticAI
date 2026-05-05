#!/usr/bin/env python3
# ============================================================
# main.py — Phase 3 (video_agent) entry point
#
# Usage:
#   python -m agents.video_agent.main --phase1-dir data/outputs
#   python -m agents.video_agent.main --script data/outputs/script.json
#   python -m agents.video_agent.main --resume      # resume from MCP memory
# ============================================================

import argparse
import asyncio
import sys
from pathlib import Path

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.video_agent.graph import build_graph


def print_summary(state: dict):
    print("\n" + "═" * 60)
    print("  PHASE 3 — VIDEO AGENT — PIPELINE COMPLETE")
    print("═" * 60)

    scenes = state.get("scenes", []) or []
    audios = state.get("audio_tracks", {}) or {}
    videos = state.get("face_swapped_clips", {}) or {}
    finals = state.get("final_scenes", {}) or {}

    print(f"  Title           : {(state.get('scene_manifest') or {}).get('title', 'N/A')}")
    print(f"  Scenes          : {len(scenes)}")
    print(f"  Audio loaded    : {len(audios)}")
    print(f"  Face-swap done  : {len(videos)}")
    print(f"  Final per-scene : {len(finals)}")
    print(f"  Final video     : {state.get('final_video', 'N/A')}")
    print(f"  Status          : {state.get('status', 'unknown')}")

    errors = state.get("errors", []) or []
    if errors:
        print(f"\n  ⚠  Errors:")
        for e in errors:
            print(f"     • {e}")
    print("═" * 60 + "\n")


async def run_pipeline(script_path: str = "", manifest_path: str = "",
                       char_db_path: str = "", resume: bool = False,
                       audio_tracks: dict = None):
    graph = build_graph()
    initial: dict = {
        "scene_manifest_path": manifest_path,
        "character_db_path":   char_db_path,
        "script_path":         script_path,
        "audio_tracks":        audio_tracks or {},
        "video_clips":         {},
        "face_swapped_clips":  {},
        "final_scenes":        {},
        "errors":              [],
        "resume_from_memory":  resume,
    }
    print(f"\n[Main] Starting Phase 3 video pipeline")
    final_state = await graph.ainvoke(initial)
    print_summary(final_state)
    return final_state


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Video Generation & Composition")
    parser.add_argument("--phase1-dir", "-d",
                        help="Directory containing Phase 1's script.json + character_db.json")
    parser.add_argument("--script", "-s",
                        help="Direct path to a unified script.json")
    parser.add_argument("--manifest", "-m",
                        help="Legacy path to scene_manifest.json")
    parser.add_argument("--characters", "-c",
                        help="Legacy path to character_db.json")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Skip scenes already present in MCP memory")
    args = parser.parse_args()

    script_path   = ""
    manifest_path = ""
    char_db_path  = ""

    if args.phase1_dir:
        d = Path(args.phase1_dir)
        if (d / "script.json").exists():
            script_path = str(d / "script.json")
        manifest_path = str(d / "scene_manifest.json")
        char_db_path  = str(d / "character_db.json")

    if args.script:
        script_path = args.script
    if args.manifest:
        manifest_path = args.manifest
    if args.characters:
        char_db_path = args.characters

    if not (script_path or manifest_path):
        parser.error("Provide --phase1-dir or --script or --manifest")

    asyncio.run(run_pipeline(
        script_path=script_path,
        manifest_path=manifest_path,
        char_db_path=char_db_path,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
