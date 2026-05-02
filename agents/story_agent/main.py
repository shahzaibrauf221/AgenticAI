#!/usr/bin/env python3
# ============================================================
# main.py
# Entry point for the Writer's Room multi-agent system
#
# Usage:
#   python main.py                          # interactive prompt
#   python main.py --prompt "A sci-fi thriller about..."
#   python main.py --script path/to/script.json
#   python main.py --demo                   # run built-in demo
# ============================================================

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# ── Make sure project root is on PYTHONPATH ───────────────
# This file lives at: <root>/agents/story_agent/main.py — go up 2 levels.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.story_agent.graph import build_graph


# ─── Pretty Print Final State ─────────────────────────────────────────────────

def print_summary(state: dict):
    print("\n" + "═" * 60)
    print("  WRITER'S ROOM — PIPELINE COMPLETE")
    print("═" * 60)

    script = state.get("script", {})
    print(f"\n📽  Title   : {script.get('title', 'N/A')}")
    print(f"🎭  Genre   : {script.get('genre', 'N/A')}")
    print(f"📋  Scenes  : {len(script.get('scenes', []))}")
    print(f"👤  Characters: {len(state.get('characters', []))}")
    print(f"🖼  Images  : {len(state.get('images', []))}")
    print(f"✅  Status  : {state.get('status', 'unknown')}")

    errors = state.get("errors", [])
    if errors:
        print(f"\n⚠  Errors ({len(errors)}):")
        for e in errors:
            print(f"   • {e}")

    print("\n── Scenes ───────────────────────────────────────────")
    for s in script.get("scenes", []):
        print(f"  [{s['scene_id']}] {s.get('location')} — {s.get('time_of_day', '')}")
        print(f"       {s.get('action_description', '')[:100]}…")

    print("\n── Characters ───────────────────────────────────────")
    for c in state.get("characters", []):
        print(f"  {c.get('name')} ({c.get('age_range', '')} / {c.get('gender', '')})")
        print(f"  Traits: {', '.join(c.get('personality_traits', []))}")

    print("\n── Output Files ─────────────────────────────────────")
    print(f"  outputs/scene_manifest.json")
    print(f"  outputs/character_db.json")
    for img in state.get("images", []):
        if img.get("file"):
            print(f"  {img['file']}")
    print("═" * 60 + "\n")


# ─── Run Pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(input_mode: str, user_input: str):
    """Build and invoke the LangGraph workflow."""
    os.environ.setdefault("HITL_AUTO_APPROVE", "1")   # remove for real interactive review

    graph = build_graph()

    initial_state = {
        "input_mode":            input_mode,
        "user_input":            user_input,
        "script":                {},
        "script_valid":          False,
        "validation_errors":     [],
        "validation_warnings":   [],
        "hitl_approved":         False,
        "hitl_feedback":         "",
        "characters":            [],
        "images":                [],
        "status":                "processing",
        "errors":                [],
        "current_node":          "",
    }

    print(f"\n[Main] Starting pipeline — mode={input_mode}")
    final_state = await graph.ainvoke(initial_state)
    print_summary(final_state)
    return final_state


# ─── CLI ──────────────────────────────────────────────────────────────────────

DEMO_PROMPT = (
    "A cyberpunk noir thriller set in 2087 Neo-Tokyo. "
    "Detective Yuki investigates the disappearance of an AI rights activist "
    "named Kael who may have uploaded his consciousness into the city's network. "
    "Her only ally is ARIA, a rogue AI with questionable motives."
)


def main():
    parser = argparse.ArgumentParser(description="Writer's Room — Autonomous Story Pipeline")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", "-p", type=str, help="Story prompt for autonomous generation.")
    group.add_argument("--script", "-s", type=str, help="Path to a manual script JSON file.")
    group.add_argument("--demo",   "-d", action="store_true", help="Run built-in demo prompt.")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(run_pipeline("auto", DEMO_PROMPT))

    elif args.script:
        path = Path(args.script)
        if not path.exists():
            print(f"Error: file not found — {args.script}")
            sys.exit(1)
        user_input = path.read_text()
        asyncio.run(run_pipeline("manual", user_input))

    elif args.prompt:
        asyncio.run(run_pipeline("auto", args.prompt))

    else:
        # Interactive mode
        print("Writer's Room — Autonomous Story & Image Generation")
        print("─" * 50)
        print("1) Generate from prompt (auto)")
        print("2) Validate existing script (manual)")
        choice = input("Select [1/2]: ").strip()

        if choice == "2":
            path = input("Path to script JSON: ").strip()
            user_input = Path(path).read_text()
            asyncio.run(run_pipeline("manual", user_input))
        else:
            prompt = input("Enter story prompt: ").strip() or DEMO_PROMPT
            asyncio.run(run_pipeline("auto", prompt))


if __name__ == "__main__":
    main()
