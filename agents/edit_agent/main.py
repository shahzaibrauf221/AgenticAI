# ============================================================
# agents/edit_agent/main.py
# Phase 5 — CLI entry point
#
# Usage:
#   python -m agents.edit_agent.main --query "Make scene 1 darker"
#   python -m agents.edit_agent.main --interactive
#   python -m agents.edit_agent.main --history
#   python -m agents.edit_agent.main --revert v2
# ============================================================

from __future__ import annotations

import argparse
import asyncio
import json
import sys

# CLI prints emoji/box-drawing chars; make stdout/stderr UTF-8 so a cp1252
# Windows console doesn't crash with UnicodeEncodeError (matches the other
# phase main.py entrypoints).
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream and _stream.encoding and _stream.encoding.lower() != "utf-8":
            _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agents.edit_agent.agent import EditAgent
from state_manager.manager import StateManager


async def _run_single(query: str, version: str | None) -> None:
    agent = EditAgent()
    print(f"\n🎬 Edit Agent — Processing: {query!r}\n{'─'*60}")
    final = await agent.run_edit(query, current_version=version)

    print(f"✅ Status:      {final.get('status')}")
    print(f"🎯 Intent:      {final.get('intent')}  [{final.get('target')}]")
    print(f"📍 Scope:       {final.get('scope')}")
    print(f"📦 New version: {final.get('new_version')}")
    print(f"\nResult:\n{json.dumps(final.get('result', {}), indent=2)}")
    print(f"\nLogs:")
    for line in final.get("logs", []):
        print(f"  {line}")
    if final.get("errors"):
        print(f"\n⚠️  Errors:")
        for e in final["errors"]:
            print(f"  {e}")


async def _interactive(version: str | None) -> None:
    agent = EditAgent()
    current = version or StateManager.latest_version() or "v1"
    print(f"\n🎬 Edit Agent — Interactive Mode  (current version: {current})")
    print("Type your edit command, 'undo', 'history', or 'quit'.\n")

    while True:
        try:
            query = input("edit> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            break

        if query.lower() == "history":
            _print_history()
            continue

        if query.lower().startswith("undo") or query.lower().startswith("revert"):
            parts = query.split()
            if len(parts) >= 2:
                target_version = parts[1]
            else:
                # Undo to second-latest
                hist = StateManager.history()
                target_version = hist[1]["version"] if len(hist) >= 2 else current
            print(f"↩️  Reverting to {target_version}…")
            try:
                bundle = StateManager.revert(target_version)
                current = target_version
                print(f"✅ Reverted to {target_version} — {bundle['label']}")
            except ValueError as e:
                print(f"❌ {e}")
            continue

        final = await agent.run_edit(query, current_version=current)
        current = final.get("new_version", current)

        print(f"  ✅ Done — saved as {current}")
        print(f"  🎯 {final.get('intent')}  [{final.get('target')}]  scope={final.get('scope')}")
        result = final.get("result", {})
        if result.get("asset_paths"):
            print(f"  📁 Assets: {', '.join(result['asset_paths'])}")
        if final.get("errors"):
            for e in final["errors"]:
                print(f"  ⚠️  {e}")


def _print_history() -> None:
    hist = StateManager.history()
    if not hist:
        print("  (no snapshots yet)")
        return
    print(f"\n{'Ver':<6} {'Label':<30} {'Changes':<40} {'Time'}")
    print("─" * 90)
    for h in hist:
        print(
            f"{h['version']:<6} "
            f"{h['label'][:28]:<30} "
            f"{h['diff_summary'][:38]:<40} "
            f"{h['created_at'][:19]}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m agents.edit_agent.main",
        description="Phase 5 — Intelligent Edit & Undo Agent",
    )
    parser.add_argument("--query",        type=str,  help="Edit command (non-interactive)")
    parser.add_argument("--version",      type=str,  help="Start from this state version")
    parser.add_argument("--interactive",  action="store_true", help="Interactive REPL mode")
    parser.add_argument("--history",      action="store_true", help="Print version history and exit")
    parser.add_argument("--revert",       type=str,  metavar="VERSION", help="Revert to a version")
    args = parser.parse_args()

    if args.history:
        _print_history()
        sys.exit(0)

    if args.revert:
        try:
            bundle = StateManager.revert(args.revert)
            print(f"✅ Reverted to {args.revert}: {bundle['label']}")
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
        sys.exit(0)

    if args.interactive:
        asyncio.run(_interactive(args.version))
        sys.exit(0)

    if args.query:
        asyncio.run(_run_single(args.query, args.version))
        sys.exit(0)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
