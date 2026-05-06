# ============================================================
# app.py — Phase 4 FastAPI entry point
# ============================================================
#
# Runs on http://localhost:8000 by default.
# Serves:
#   • REST/SSE API at /api/*
#   • Frontend SPA at /
#   • Generated assets at /static/* (audio, video, images)
# ============================================================

import sys
from pathlib import Path

# Make project root importable when run as "uvicorn backend.app:app"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.pipeline import router as pipeline_router
from backend.routes.outputs  import router as outputs_router
from backend.routes.edit     import router as edit_router

app = FastAPI(
    title="AgenticAI Project — Phase 4 Web Interface",
    description="Orchestrates the Writer's Room → Audio → Video pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ──────────────────────────────────────────────────────────────────
app.include_router(pipeline_router)
app.include_router(outputs_router)
app.include_router(edit_router)

# ─── Static asset mounts ─────────────────────────────────────────────────────
# All generated artifacts live under <root>/outputs/. We expose them at /static/.
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(OUTPUTS_DIR)), name="static")


# ─── Frontend ────────────────────────────────────────────────────────────────
FRONTEND_DIR = ROOT / "frontend"
INDEX_HTML   = FRONTEND_DIR / "index.html"


@app.get("/", include_in_schema=False)
def serve_index():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML))
    return JSONResponse({
        "message": "Frontend not found. Visit /docs for the API.",
        "api":     "/docs",
    })


# Also serve any other files in frontend/ (CSS, JS, etc.)
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
