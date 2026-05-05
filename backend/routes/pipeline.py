# ============================================================
# routes/pipeline.py — Endpoints to launch pipelines
# ============================================================
#
# POST /api/run/full       — Phase 1 → 2 → 3 from a prompt
# POST /api/run/phase1     — Phase 1 only
# POST /api/run/phase2     — Phase 2 only (uses existing Phase 1 outputs)
# POST /api/run/phase3     — Phase 3 only (uses existing Phase 1 + 2 outputs)
# GET  /api/jobs           — List all jobs in this session
# GET  /api/jobs/{id}      — Job snapshot (status, events, result)
# GET  /api/jobs/{id}/events — SSE stream of live events
# ============================================================

import asyncio
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.services import manager

router = APIRouter(prefix="/api", tags=["pipeline"])

# Project root — assumes backend/ lives at <root>/backend/
ROOT       = Path(__file__).resolve().parent.parent.parent
OUTPUTS    = ROOT / "data" / "outputs"
LEGACY_OUT = ROOT / "outputs"


# ─── Request models ─────────────────────────────────────────────────────────

class FullRunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


class PhaseRequest(BaseModel):
    phase1_dir: str = Field(default="data/outputs",
                            description="Directory with Phase 1 outputs")
    prompt:     str = Field(default="",
                            description="(Phase 1 only) story prompt")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_phase1_dir(p1: str) -> Path:
    """Accept absolute or root-relative phase1 dir; pick the one that has files."""
    candidate = Path(p1)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if (candidate / "scene_manifest.json").exists() or (candidate / "script.json").exists():
        return candidate
    # Fallback to data/outputs or outputs
    for alt in (OUTPUTS, LEGACY_OUT):
        if (alt / "scene_manifest.json").exists() or (alt / "script.json").exists():
            return alt
    return candidate   # let the pipeline raise its own error


# ─── Routes ─────────────────────────────────────────────────────────────────

@router.post("/run/full")
async def run_full(req: FullRunRequest):
    job = manager.create("full", prompt=req.prompt)

    async def _coro(j):
        from agents.orchestrator.pipeline import run_all
        os.environ.setdefault("HITL_AUTO_APPROVE", "1")
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        return await run_all(prompt=req.prompt, output_dir=OUTPUTS)

    await manager.run_in_background(job, _coro)
    return {"job_id": job.id, "status": job.status}


@router.post("/run/phase1")
async def run_phase1(req: PhaseRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Phase 1 requires a prompt")

    job = manager.create("phase1", prompt=req.prompt)

    async def _coro(j):
        from agents.orchestrator.pipeline import run_phase1 as r1
        os.environ.setdefault("HITL_AUTO_APPROVE", "1")
        out_dir = _resolve_phase1_dir(req.phase1_dir) or OUTPUTS
        out_dir.mkdir(parents=True, exist_ok=True)
        return await r1(req.prompt, out_dir)

    await manager.run_in_background(job, _coro)
    return {"job_id": job.id, "status": job.status}


@router.post("/run/phase2")
async def run_phase2(req: PhaseRequest):
    job = manager.create("phase2")

    async def _coro(j):
        from agents.orchestrator.pipeline import run_phase2 as r2
        p1 = _resolve_phase1_dir(req.phase1_dir)
        return await r2(p1)

    await manager.run_in_background(job, _coro)
    return {"job_id": job.id, "status": job.status}


@router.post("/run/phase3")
async def run_phase3(req: PhaseRequest):
    job = manager.create("phase3")

    async def _coro(j):
        from agents.orchestrator.pipeline import run_phase3 as r3
        p1 = _resolve_phase1_dir(req.phase1_dir)
        return await r3(p1, audio_tracks={})

    await manager.run_in_background(job, _coro)
    return {"job_id": job.id, "status": job.status}


# ─── Job inspection ─────────────────────────────────────────────────────────

@router.get("/jobs")
def list_jobs():
    return {
        "jobs": [
            {
                "id":          j.id,
                "kind":        j.kind,
                "status":      j.status,
                "started_at":  j.started_at,
                "finished_at": j.finished_at,
                "event_count": len(j.events),
                "prompt":      j.prompt[:100],
                "error":       j.error,
            }
            for j in manager.jobs.values()
        ]
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return {
        "id":          job.id,
        "kind":        job.kind,
        "status":      job.status,
        "prompt":      job.prompt,
        "started_at":  job.started_at,
        "finished_at": job.finished_at,
        "error":       job.error,
        "result":      job.result if isinstance(job.result, dict) else {},
        "events":      [
            {"ts": e.ts, "type": e.type, "message": e.message,
             "phase": e.phase, "extra": e.extra}
            for e in job.events
        ],
    }


@router.get("/jobs/{job_id}/events")
async def stream_events(job_id: str):
    """Server-Sent Events stream of live job updates."""
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    async def gen():
        last_idx = 0
        # First — replay events that already happened
        while True:
            # Drain new events
            while last_idx < len(job.events):
                e = job.events[last_idx]
                payload = json.dumps({
                    "ts": e.ts, "type": e.type, "message": e.message,
                    "phase": e.phase, "extra": e.extra,
                })
                yield f"event: {e.type}\ndata: {payload}\n\n"
                last_idx += 1

            if job.status in ("complete", "failed"):
                # Final marker so the client can close
                yield f"event: end\ndata: {json.dumps({'status': job.status})}\n\n"
                return

            # Wait for new events (with timeout heartbeat to keep connection alive)
            await manager.wait_for_event(job, timeout=15.0)
            # Keepalive comment line
            yield ": keepalive\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",   # nginx hint
            "Connection":      "keep-alive",
        },
    )
