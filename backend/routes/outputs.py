# ============================================================
# routes/outputs.py — Inspect & download generated artifacts
# ============================================================

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/api", tags=["outputs"])

ROOT     = Path(__file__).resolve().parent.parent.parent
OUTPUTS  = ROOT / "outputs"               # legacy Phase 1 dir
SPEC_DIR = ROOT / "data" / "outputs"      # spec-compliant artifacts


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@router.get("/outputs/summary")
def summary():
    """Top-level snapshot of what's available on disk."""
    script  = _load_json(SPEC_DIR / "script.json")
    chars   = _load_json(SPEC_DIR / "characters.json")
    summary = _load_json(SPEC_DIR / "summary.json")

    image_dir = OUTPUTS / "image_assets"
    images = []
    if image_dir.exists():
        for png in sorted(image_dir.glob("*.png")):
            images.append({"name": png.name, "url": f"/static/image_assets/{png.name}"})

    audio_dir = OUTPUTS / "audio"
    audios = []
    if audio_dir.exists():
        for wav in sorted(audio_dir.glob("*.wav")):
            audios.append({"name": wav.name, "url": f"/static/audio/{wav.name}"})

    final_dir = OUTPUTS / "final"
    finals = []
    if final_dir.exists():
        for mp4 in sorted(final_dir.glob("*.mp4")):
            finals.append({
                "name":     mp4.name,
                "url":      f"/static/final/{mp4.name}",
                "size_mb":  round(mp4.stat().st_size / (1024 * 1024), 2),
            })

    final_video_path = final_dir / "final_output.mp4"

    return {
        "story":       script.get("story", {}),
        "scenes":      script.get("scenes", []),
        "characters":  (chars or script).get("characters", []),
        "images":      images,
        "audio":       audios,
        "final_clips": finals,
        "final_video": (
            f"/static/final/final_output.mp4"
            if final_video_path.exists() else ""
        ),
        "summary":     summary,
    }


@router.get("/outputs/script")
def get_script():
    return _load_json(SPEC_DIR / "script.json") or {"error": "script.json not found"}


@router.get("/outputs/characters")
def get_characters():
    return _load_json(SPEC_DIR / "characters.json") or {"error": "characters.json not found"}
