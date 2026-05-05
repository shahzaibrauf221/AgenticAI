# ============================================================
# tests/test_state_manager.py
# Phase 5 — StateManager unit tests
# ============================================================

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Patch DB path to a temp dir so tests don't pollute production data ────────
_TMP_VERSIONS = None

@pytest.fixture(autouse=True)
def isolated_state_manager(tmp_path):
    """
    Redirect state_manager storage to a temp directory for each test.
    """
    import state_manager.manager as mgr
    orig_dir = mgr._VERSIONS_DIR
    orig_db  = mgr._DB_PATH

    mgr._VERSIONS_DIR = tmp_path / "state_versions"
    mgr._VERSIONS_DIR.mkdir()
    mgr._DB_PATH = mgr._VERSIONS_DIR / "state_history.db"

    yield

    mgr._VERSIONS_DIR = orig_dir
    mgr._DB_PATH      = orig_db


from state_manager.manager import StateManager


def test_snapshot_returns_version():
    v = StateManager.snapshot({"story": {"title": "Test"}}, label="initial")
    assert v == "v1"


def test_snapshot_increments():
    v1 = StateManager.snapshot({"story": {"title": "Test A"}})
    v2 = StateManager.snapshot({"story": {"title": "Test B"}})
    assert v1 == "v1"
    assert v2 == "v2"


def test_history_returns_newest_first():
    StateManager.snapshot({"x": 1}, label="first")
    StateManager.snapshot({"x": 2}, label="second")
    StateManager.snapshot({"x": 3}, label="third")

    hist = StateManager.history()
    assert len(hist) == 3
    assert hist[0]["version"] == "v3"   # newest first
    assert hist[2]["version"] == "v1"


def test_get_state_returns_correct_data():
    state_a = {"story": {"title": "Alpha"}, "scenes": []}
    state_b = {"story": {"title": "Beta"},  "scenes": [{"scene_id": 1}]}
    StateManager.snapshot(state_a)
    StateManager.snapshot(state_b)

    assert StateManager.get_state("v1")["story"]["title"] == "Alpha"
    assert StateManager.get_state("v2")["story"]["title"] == "Beta"


def test_get_state_raises_on_missing():
    with pytest.raises(ValueError, match="not found"):
        StateManager.get_state("v99")


def test_revert_restores_state():
    original = {"story": {"title": "Before"}, "scenes": []}
    StateManager.snapshot(original, label="before edit")
    StateManager.snapshot({"story": {"title": "After"}}, label="after edit")

    bundle = StateManager.revert("v1")
    assert bundle["state"]["story"]["title"] == "Before"
    assert bundle["label"] == "before edit"
    assert bundle["version"] == "v1"


def test_revert_raises_on_unknown_version():
    with pytest.raises(ValueError):
        StateManager.revert("v999")


def test_latest_version_none_when_empty():
    assert StateManager.latest_version() is None


def test_latest_version_returns_last():
    StateManager.snapshot({"x": 1})
    StateManager.snapshot({"x": 2})
    assert StateManager.latest_version() == "v2"


def test_diff_summary_initial():
    v = StateManager.snapshot({"story": {"title": "T"}})
    hist = StateManager.history()
    assert "Initial snapshot" in hist[0]["diff_summary"]


def test_diff_summary_detects_scene_change():
    StateManager.snapshot({"scenes": [{"scene_id": 1}, {"scene_id": 2}]})
    StateManager.snapshot({"scenes": [{"scene_id": 1}]})
    hist = StateManager.history()
    assert "Scenes: 2 → 1" in hist[0]["diff_summary"]


def test_snapshot_with_asset_paths(tmp_path):
    # Create a fake asset file
    asset = tmp_path / "dummy_video.mp4"
    asset.write_bytes(b"fake mp4 data")

    v = StateManager.snapshot({"x": 1}, asset_paths=[str(asset)], label="with asset")
    hist = StateManager.history()
    assert hist[0]["asset_count"] == 1
