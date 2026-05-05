# ============================================================
# manifest_writer.py
# Phase 2 addon — writes timing_manifest.json and updates
# phase2_audio_handoff.json after Phase 2 pipeline completes.
#
# Place this in your Phase 2 project root. Run AFTER `main.py`:
#
#   python main.py --phase1-dir ../writers-room
#   python manifest_writer.py --phase1-dir ../writers-room
# ============================================================

import argparse
import json
import subprocess
import shutil
from pathlib import Path


def _probe_duration_ms(path: Path) -> int:
    """Get duration of an audio file in milliseconds via ffprobe."""
    if not path.exists() or not shutil.which("ffprobe"):
        return 0
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return int(float(result.stdout.strip()) * 1000)
    except Exception:
        return 0


def build_timing_manifest(phase2_dir: Path, phase1_dir: Path, out_dir: Path):
    """
    Build timing_manifest.json by walking Phase 2's audio output folder
    and matching files back to scenes/dialogue lines in script.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull the unified script (produced by Phase 1 serializer)
    script_path = phase1_dir / "script.json"
    if not script_path.exists():
        raise FileNotFoundError(
            f"{script_path} not found. Run serializer.py in Phase 1 first."
        )
    script = json.loads(script_path.read_text(encoding="utf-8"))

    audio_dir = phase2_dir / "outputs" / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"No Phase 2 audio folder at {audio_dir}")

    segments = []
    cursor_ms = 0
    scenes = script.get("scenes", []) or []

    for scene in scenes:
        sid = scene.get("scene_id")
        for line_idx, d in enumerate(scene.get("dialogue", [])):
            speaker = d.get("speaker", "")
            # Find the matching audio file (filename format from Phase 2):
            #   scene_NN_<safe_char>_<hash>.wav/.mp3
            safe_char = "".join(c if c.isalnum() or c in "-_" else "_" for c in speaker).lower()
            pattern = f"scene_{sid:02d}_{safe_char}_*"
            matches = list(audio_dir.glob(pattern + ".wav")) + list(audio_dir.glob(pattern + ".mp3"))
            if not matches:
                print(f"  ⚠ no audio found for scene {sid} line {line_idx} ({speaker})")
                continue

            audio_file = str(matches[0].resolve())
            duration_ms = _probe_duration_ms(matches[0])
            if duration_ms == 0:
                # Fallback: estimate from word count
                words = max(1, len(d.get("line", "").split()))
                duration_ms = int((words / 2.5) * 1000)

            segments.append({
                "scene_id":   sid,
                "line_index": line_idx,
                "speaker":    speaker,
                "audio_file": audio_file,
                "start_ms":   cursor_ms,
                "end_ms":     cursor_ms + duration_ms,
                "text":       d.get("line", ""),
            })
            cursor_ms += duration_ms

    timing_manifest = {
        "total_duration_ms": cursor_ms,
        "segments":          segments,
    }
    (out_dir / "timing_manifest.json").write_text(
        json.dumps(timing_manifest, indent=2)
    )
    print(f"  ✓ timing_manifest.json — {len(segments)} segments, {cursor_ms/1000:.1f}s total")

    # Update phase2_audio_handoff.json with real segments
    handoff_path = out_dir / "phase2_audio_handoff.json"
    if handoff_path.exists():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    else:
        handoff = {"voice_configs": {}, "music_moods": {}}

    handoff["segments"] = segments
    handoff_path.write_text(json.dumps(handoff, indent=2))
    print(f"  ✓ phase2_audio_handoff.json updated with {len(segments)} segments")

    # Update summary.json
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["phase2_done"] = True
        summary["artifacts"]["timing_manifest"] = str(out_dir / "timing_manifest.json")
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  ✓ summary.json marked phase2_done=True")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", required=True,
                        help="Phase 1 folder containing outputs/script.json")
    parser.add_argument("--phase2-dir", default=".",
                        help="Phase 2 folder containing outputs/audio/")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write (defaults to --phase1-dir/outputs)")
    args = parser.parse_args()

    phase1 = Path(args.phase1_dir) / "outputs"
    phase2 = Path(args.phase2_dir)
    out    = Path(args.out_dir) if args.out_dir else phase1

    build_timing_manifest(phase2, phase1, out)


if __name__ == "__main__":
    main()
