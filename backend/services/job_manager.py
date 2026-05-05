# ============================================================
# job_manager.py — Tracks pipeline runs and broadcasts progress
# ============================================================
#
# Each /api/run-* endpoint creates a Job, kicks off the pipeline in a
# background task, and captures stdout/stderr line-by-line. The frontend
# subscribes via SSE (/api/jobs/{job_id}/events) to get live updates.
# ============================================================

import asyncio
import io
import sys
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional


@dataclass
class JobEvent:
    ts:       float                    # unix timestamp
    type:     str                      # "log" | "phase" | "done" | "error"
    message:  str
    phase:    Optional[str] = None     # "phase1" | "phase2" | "phase3"
    extra:    dict           = field(default_factory=dict)


@dataclass
class Job:
    id:           str
    kind:         str                  # "full" | "phase1" | "phase2" | "phase3"
    prompt:       str                  = ""
    status:       str                  = "queued"  # queued | running | complete | failed
    started_at:   float                = 0.0
    finished_at:  float                = 0.0
    events:       list                 = field(default_factory=list)
    result:       dict                 = field(default_factory=dict)
    error:        str                  = ""
    # Internal — async event for SSE waiters
    _event_signal: asyncio.Event       = field(default_factory=asyncio.Event)


class JobManager:
    """Singleton-ish job tracker for the current process."""

    def __init__(self):
        self.jobs: dict = {}

    def create(self, kind: str, prompt: str = "") -> Job:
        job = Job(id=str(uuid.uuid4())[:8], kind=kind, prompt=prompt)
        self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def emit(self, job: Job, event_type: str, message: str,
             phase: Optional[str] = None, extra: Optional[dict] = None):
        evt = JobEvent(
            ts=time.time(), type=event_type, message=message,
            phase=phase, extra=extra or {},
        )
        job.events.append(evt)
        # Wake up any SSE waiters
        job._event_signal.set()
        job._event_signal.clear()

    async def run_in_background(self, job: Job,
                                 coro_factory: Callable[[Job], Awaitable]):
        """Schedule a coroutine, capturing stdout into job events."""
        async def runner():
            job.status     = "running"
            job.started_at = time.time()
            self.emit(job, "phase", f"Job started: {job.kind}")

            # Capture stdout/stderr → job events (so the pipeline's print()s show up)
            class _StreamCapture(io.TextIOBase):
                def __init__(self, jm, jb):
                    self.jm = jm; self.jb = jb; self.buf = ""
                def write(self, s):
                    if not s:
                        return 0
                    self.buf += s
                    while "\n" in self.buf:
                        line, self.buf = self.buf.split("\n", 1)
                        if line.strip():
                            self.jm.emit(self.jb, "log", line.rstrip())
                    return len(s)
                def flush(self):
                    if self.buf.strip():
                        self.jm.emit(self.jb, "log", self.buf.rstrip())
                        self.buf = ""

            cap = _StreamCapture(self, job)
            try:
                # Tee — also write to real stdout so the server terminal still shows progress
                class _Tee:
                    def __init__(self, *streams): self.streams = streams
                    def write(self, s):
                        for st in self.streams:
                            try: st.write(s)
                            except Exception: pass
                        return len(s)
                    def flush(self):
                        for st in self.streams:
                            try: st.flush()
                            except Exception: pass

                tee = _Tee(sys.__stdout__, cap)
                with redirect_stdout(tee), redirect_stderr(tee):
                    result = await coro_factory(job)

                job.result      = result or {}
                job.status      = "complete"
                job.finished_at = time.time()
                self.emit(job, "done", "Pipeline complete",
                          extra={"duration_s": round(job.finished_at - job.started_at, 1)})

            except Exception as e:
                import traceback
                job.error       = f"{type(e).__name__}: {e}"
                job.status      = "failed"
                job.finished_at = time.time()
                tb = traceback.format_exc()
                self.emit(job, "error", job.error, extra={"traceback": tb})

        # Fire and forget
        asyncio.create_task(runner())

    async def wait_for_event(self, job: Job, timeout: float = 30.0) -> bool:
        """Block until a new event arrives or timeout. Returns True on event, False on timeout."""
        try:
            await asyncio.wait_for(job._event_signal.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


# Module-level singleton
manager = JobManager()
