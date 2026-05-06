# ============================================================
# state_manager/manager.py
# Phase 5 — SQLite-backed snapshot / undo / history system
# ============================================================

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VERSIONS_DIR = _PROJECT_ROOT / "data" / "state_versions"
_DB_PATH      = _VERSIONS_DIR / "state_history.db"

_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─── DB bootstrap ─────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS versions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            version      TEXT    NOT NULL UNIQUE,
            label        TEXT    NOT NULL,
            state_json   TEXT    NOT NULL,
            asset_paths  TEXT    NOT NULL,
            diff_summary TEXT    NOT NULL,
            created_at   TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def _next_version(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT COUNT(*) AS c FROM versions").fetchone()
    return f"v{row['c'] + 1}"


def _copy_assets(asset_paths: list[str], version: str) -> list[str]:
    dest_dir = _VERSIONS_DIR / version
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in asset_paths:
        p = Path(src)
        if p.exists():
            dst = dest_dir / p.name
            shutil.copy2(p, dst)
            copied.append(str(dst))
        else:
            copied.append(src)
    return copied


def _diff_summary(old_json: Optional[str], new_state: dict) -> str:
    if old_json is None:
        return "Initial snapshot — pipeline generated from prompt."
    try:
        old = json.loads(old_json)
    except Exception:
        return "State changed (could not diff prior version)."

    changes: list[str] = []
    for key in ("story", "scenes", "characters"):
        if key not in old and key in new_state:
            changes.append(f"Added '{key}'")
        elif key in old and key not in new_state:
            changes.append(f"Removed '{key}'")

    old_scenes = len(old.get("scenes", []))
    new_scenes  = len(new_state.get("scenes", []))
    if old_scenes != new_scenes:
        changes.append(f"Scenes: {old_scenes} → {new_scenes}")

    old_chars = len(old.get("characters", []))
    new_chars  = len(new_state.get("characters", []))
    if old_chars != new_chars:
        changes.append(f"Characters: {old_chars} → {new_chars}")

    if "last_edit" in new_state:
        changes.append(f"Edit: {new_state['last_edit']}")

    return "; ".join(changes) if changes else "Minor state update."


class StateManager:
    """Append-only state history with full undo/revert support."""

    @staticmethod
    def snapshot(
        state: dict | None = None,
        asset_paths: list[str] | None = None,
        label: str = "pipeline run",
        version: str | None = None,
        state_json: dict | None = None,
    ) -> str:
        """Persist state + copy assets. Returns new version tag."""
        if state is None:
            state = state_json or {}
        asset_paths = asset_paths or []
        conn = _get_conn()
        version = version or _next_version(conn)

        prev_row = conn.execute(
            "SELECT state_json FROM versions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_json = prev_row["state_json"] if prev_row else None

        diff          = _diff_summary(prev_json, state)
        copied_assets = _copy_assets(asset_paths, version)

        conn.execute(
            """INSERT INTO versions
               (version, label, state_json, asset_paths, diff_summary, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                version,
                label,
                json.dumps(state, default=str),
                json.dumps(copied_assets),
                diff,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        return version

    @staticmethod
    def revert(version: str) -> dict:
        """Restore assets & state from a snapshot. Returns restored bundle."""
        conn = _get_conn()
        row  = conn.execute(
            "SELECT * FROM versions WHERE version = ?", (version,)
        ).fetchone()
        conn.close()

        if row is None:
            raise ValueError(f"Version '{version}' not found.")

        state       = json.loads(row["state_json"])
        asset_paths = json.loads(row["asset_paths"])
        _restore_assets(asset_paths)

        return {
            "version":     version,
            "state":       state,
            "asset_paths": asset_paths,
            "label":       row["label"],
            "created_at":  row["created_at"],
        }

    @staticmethod
    def history() -> list[dict]:
        """All versions newest-first."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT version, label, diff_summary, asset_paths, created_at "
            "FROM versions ORDER BY id DESC"
        ).fetchall()
        conn.close()
        return [
            {
                "version":     r["version"],
                "label":       r["label"],
                "diff_summary": r["diff_summary"],
                "asset_count": len(json.loads(r["asset_paths"])),
                "created_at":  r["created_at"],
            }
            for r in rows
        ]

    @staticmethod
    def get_state(version: str) -> dict:
        conn = _get_conn()
        row  = conn.execute(
            "SELECT state_json FROM versions WHERE version = ?", (version,)
        ).fetchone()
        conn.close()
        if row is None:
            raise ValueError(f"Version '{version}' not found.")
        return json.loads(row["state_json"])

    @staticmethod
    def latest_version() -> Optional[str]:
        conn = _get_conn()
        row  = conn.execute(
            "SELECT version FROM versions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return row["version"] if row else None


def _restore_assets(stored_paths: list[str]) -> None:
    outputs_dir = _PROJECT_ROOT / "data" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for src in stored_paths:
        p = Path(src)
        if p.exists():
            shutil.copy2(p, outputs_dir / p.name)
