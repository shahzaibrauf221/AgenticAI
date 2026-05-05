# ============================================================
# backend/routes/edit.py
# Phase 5 — FastAPI routes for the Edit Agent & Undo System
# ============================================================

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.edit_agent.agent import EditAgent
from state_manager.manager import StateManager

router = APIRouter(prefix="/api/edit", tags=["Phase 5 — Edit Agent"])

# ─── Request/Response schemas ─────────────────────────────────────────────────

class EditRequest(BaseModel):
    query:           str
    current_version: Optional[str] = None
    session_id:      Optional[str] = None


class RevertRequest(BaseModel):
    version: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/run")
async def run_edit(body: EditRequest) -> dict:
    """
    Execute a free-text edit command.
    Returns: { intent, target, scope, result, new_version, status, logs }
    """
    try:
        agent  = EditAgent(session_id=body.session_id)
        result = await agent.run_edit(
            body.query,
            current_version=body.current_version,
        )
        return {
            "intent":      result.get("intent"),
            "target":      result.get("target"),
            "scope":       result.get("scope"),
            "parameters":  result.get("parameters"),
            "result":      result.get("result"),
            "new_version": result.get("new_version"),
            "status":      result.get("status"),
            "logs":        result.get("logs", []),
            "errors":      result.get("errors", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/run/stream")
async def stream_edit(body: EditRequest) -> StreamingResponse:
    """
    Stream edit agent progress as Server-Sent Events.
    Each event: data: { node, state }
    """
    agent = EditAgent(session_id=body.session_id)

    async def _sse() -> AsyncIterator[str]:
        async for event in agent.stream_edit(
            body.query,
            current_version=body.current_version,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@router.get("/history")
def get_history() -> list[dict]:
    """Return all version snapshots newest-first."""
    return StateManager.history()


@router.post("/revert")
def revert_to_version(body: RevertRequest) -> dict:
    """Revert pipeline state to a prior snapshot."""
    try:
        bundle = StateManager.revert(body.version)
        return {
            "version":    bundle["version"],
            "label":      bundle["label"],
            "created_at": bundle["created_at"],
            "message":    f"Reverted to {bundle['version']} — {bundle['label']}",
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/state/{version}")
def get_version_state(version: str) -> dict:
    """Return the raw pipeline JSON state stored at a given version."""
    try:
        return StateManager.get_state(version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/latest")
def get_latest_version() -> dict:
    """Return the latest version tag."""
    v = StateManager.latest_version()
    if v is None:
        return {"version": None, "message": "No snapshots yet."}
    return {"version": v}
