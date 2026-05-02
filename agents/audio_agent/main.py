#!/usr/bin/env python3
# ============================================================
# main.py — Phase 2 (audio_agent) entry point
#
# Usage:
#   python -m agents.audio_agent.main --phase1-dir data/outputs
#   python -m agents.audio_agent.main --script data/outputs/script.json
#   python -m agents.audio_agent.main --resume      # resume from MCP memory
# ============================================================

import argparse
import asyncio
import sys
from pathlib import Path

# Make project root importable when run as a script
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.audio_agent.graph import build_graph


def print_summary(state: dict):
    print("\n" + "═" * 60)
    print("  PHASE 2 — AUDIO AGENT — PIPELINE COMPLETE")
    print("═" * 60)

    audios = state.get("audio_tracks", {}) or {}
    scenes = state.get("scenes", []) or []
    print(f"  Title       : {(state.get('scene_manifest') or {}).get('title', 'N/A')}")
    print(f"  Scenes      : {len(scenes)}")
    print(f"  Audio done  : {len(audios)}")
    print(f"  Status      : {state.get('status', 'unknown')}")

    for sid, info in audios.items():
        files = info.get("files", []) or []
        print(f"    Scene {sid}: {len(files)} clip(s), {info.get('total_duration', 0)}s, "
              f"mood={info.get('mood', 'neutral')}")

    errors = state.get("errors", []) or []
    if errors:
        print(f"\n  ⚠  Errors:")
        for e in errors:
            print(f"     • {e}")
    print("═" * 60 + "\n")


async def run_pipeline(script_path: str = "", manifest_path: str = "",
                       char_db_path: str = "", resume: bool = False):
    graph = build_graph()
    initial: dict = {
        "scene_manifest_path": manifest_path,
        "character_db_path":   char_db_path,
        "script_path":         script_path,
        "audio_tracks":        {},
        "errors":              [],
        "resume_from_memory":  resume,
    }
    print(f"\n[Main] Starting Phase 2 audio pipeline")
    final_state = await graph.ainvoke(initial)
    print_summary(final_state)
    return final_state


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — Audio Generation")
    parser.add_argument("--phase1-dir", "-d",
                        help="Directory containing Phase 1's script.json + character_db.json")
    parser.add_argument("--script", "-s",
                        help="Direct path to a unified script.json (preferred over --phase1-dir)")
    parser.add_argument("--manifest", "-m",
                        help="Legacy path to scene_manifest.json (used if no script.json)")
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
