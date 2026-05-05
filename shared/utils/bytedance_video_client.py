import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


class ByteDanceVideoClientError(RuntimeError):
    """Raised when the ByteDance/Volcengine video workflow fails."""


@dataclass
class ByteDanceVideoClientConfig:
    base_url: str
    api_key: str
    model: str
    submit_path: str = "/contents/generations/tasks"
    task_path_template: str = "/contents/generations/tasks/{task_id}"
    timeout_s: int = 60
    poll_timeout_s: int = 900
    poll_backoff_start_s: float = 1.0
    poll_backoff_max_s: float = 15.0
    # Seedance text line suffixes (env overridable).
    # Empty video_resolution omits `--resolution …` (Seedance 1.5 examples use duration + camerafixed only).
    video_resolution: str = ""
    video_duration_seconds: int = 5
    camera_fixed: str = "false"
    # Cap `--duration` sent to the API (full scene length still used locally for sync).
    max_api_duration_seconds: int = 12
    # seedance-1-5-pro only accepts discrete output lengths for `text.duration`; 2 s from num_frames/8 fails.
    allowed_api_durations: tuple[int, ...] = (5, 10)


class ByteDanceVideoClient:
    """
    Async-task client for ByteDance/Volcengine Ark video generation (Seedance)
    using bearer authentication and the official content[] task schema.
    """

    def __init__(self, config: ByteDanceVideoClientConfig):
        if not config.base_url:
            raise ByteDanceVideoClientError("Missing BYTEDANCE_API_BASE_URL")
        if not config.api_key:
            raise ByteDanceVideoClientError("Missing BYTEDANCE_API_KEY")
        self.config = config

    @classmethod
    def from_env(cls) -> "ByteDanceVideoClient":
        duration_raw = os.getenv("BYTEDANCE_VIDEO_DURATION", "5")
        try:
            duration_int = max(1, int(float(duration_raw)))
        except ValueError:
            duration_int = 5

        cap_raw = os.getenv("BYTEDANCE_MAX_API_DURATION", "12")
        try:
            duration_cap = max(1, int(float(cap_raw)))
        except ValueError:
            duration_cap = 12

        allowed_raw = os.getenv("BYTEDANCE_ALLOWED_DURATIONS", "5,10").strip()
        allowed: list[int] = []
        for part in allowed_raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                allowed.append(max(1, int(float(part))))
            except ValueError:
                continue
        if not allowed:
            allowed = [5, 10]
        allowed_t = tuple(sorted(set(allowed)))

        cfg = ByteDanceVideoClientConfig(
            base_url=os.getenv("BYTEDANCE_API_BASE_URL", "").rstrip("/"),
            api_key=os.getenv("BYTEDANCE_API_KEY", ""),
            model=os.getenv("BYTEDANCE_MODEL", "seedance-1-5-pro-251215"),
            submit_path=os.getenv("BYTEDANCE_SUBMIT_PATH", "/contents/generations/tasks"),
            task_path_template=os.getenv(
                "BYTEDANCE_TASK_PATH_TEMPLATE", "/contents/generations/tasks/{task_id}"
            ),
            timeout_s=int(os.getenv("BYTEDANCE_HTTP_TIMEOUT_S", "60")),
            poll_timeout_s=int(os.getenv("BYTEDANCE_POLL_TIMEOUT_S", "900")),
            poll_backoff_start_s=float(os.getenv("BYTEDANCE_POLL_BACKOFF_START_S", "1.0")),
            poll_backoff_max_s=float(os.getenv("BYTEDANCE_POLL_BACKOFF_MAX_S", "15.0")),
            video_resolution=os.getenv("BYTEDANCE_VIDEO_RESOLUTION", "").strip(),
            video_duration_seconds=duration_int,
            camera_fixed=os.getenv("BYTEDANCE_CAMERA_FIXED", "false").strip().lower(),
            max_api_duration_seconds=duration_cap,
            allowed_api_durations=allowed_t,
        )
        return cls(cfg)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _extract_first(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> Any:
        for path in paths:
            cur: Any = payload
            ok = True
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    ok = False
                    break
                cur = cur[key]
            if ok and cur is not None:
                return cur
        return None

    @staticmethod
    def _build_seedance_text_line(
        visual_prompt: str,
        resolution: str,
        duration_seconds: int,
        camera_fixed: str,
    ) -> str:
        """
        Seedance task `content[].text`:
          With resolution: "...  --resolution 1080p  --duration N --camerafixed false"
          Without (1.5 style): "...  --duration N --camerafixed false"
        """
        base = (visual_prompt or "").strip()
        cam = "true" if camera_fixed in ("1", "true", "yes") else "false"
        dur = int(duration_seconds)
        res = (resolution or "").strip()
        if res:
            suffix = f"--resolution {res}  --duration {dur} --camerafixed {cam}"
        else:
            suffix = f"--duration {dur} --camerafixed {cam}"
        return f"{base}  {suffix}" if base else f"  {suffix}".strip()

    def _snap_to_allowed_duration(self, requested: int) -> int:
        """Pick a duration the model accepts — See InvalidParameter contents[0].text.duration."""
        capped = max(1, min(int(requested), self.config.max_api_duration_seconds))
        allowed = sorted(self.config.allowed_api_durations)
        if not allowed:
            return capped
        if capped in allowed:
            return capped
        for a in allowed:
            if a >= capped:
                return a
        return allowed[-1]

    @staticmethod
    def _is_invalid_content_text_error(msg: str) -> bool:
        m = (msg or "").lower()
        return "invalid content.text" in m or "invalid content text" in m

    @staticmethod
    def _sanitize_prompt_for_retry(prompt: str) -> str:
        """
        Seedance may reject certain long/structured prompts. Retry once with a simplified,
        single-line prompt (remove shot markers and clamp length).
        """
        p = (prompt or "").replace("\r", " ").replace("\n", " ").strip()
        p = re.sub(r"\s+", " ", p)
        # Remove common shot markers like "shot 3/12" and everything after ", shot …"
        p = re.sub(r",\s*shot\s+\d+\s*/\s*\d+.*$", "", p, flags=re.IGNORECASE)
        p = re.sub(r"\bshot\s+\d+\s*/\s*\d+\b", "", p, flags=re.IGNORECASE).strip(" ,")
        # Clamp to keep it well within typical content limits.
        if len(p) > 500:
            p = p[:500].rsplit(" ", 1)[0].rstrip(" ,")
        return p or (prompt or "").strip()

    def submit_video_task(
        self,
        prompt: str,
        scene_id: int | str,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        duration_s: float | None = None,
        image_url: str | None = None,
    ) -> str:
        """
        Submit async video generation task.

        JSON body matches Ark Seedance schema:
          { "model": "...", "content": [ { "type": "text", "text": "..." }, ... ] }

        Optional I2V: append { "type": "image_url", "image_url": { "url": "..." } }.
        width/height/seed/scene_id are not sent in body (API uses text flags + model).
        """
        url = f"{self.config.base_url}{self.config.submit_path}"

        duration_int = self.config.video_duration_seconds
        if duration_s is not None and duration_s > 0:
            duration_int = max(1, int(round(float(duration_s))))
        duration_int = min(duration_int, self.config.max_api_duration_seconds)
        duration_int = self._snap_to_allowed_duration(duration_int)

        text_line = self._build_seedance_text_line(
            prompt,
            self.config.video_resolution,
            duration_int,
            self.config.camera_fixed,
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": text_line},
        ]
        if image_url and str(image_url).strip():
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(image_url).strip()},
                }
            )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "content": content,
        }

        resp = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout_s,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            detail = resp.text[:1200] if resp.text else "(empty body)"
            raise ByteDanceVideoClientError(
                f"Task submit failed ({resp.status_code}): {detail}"
            ) from e

        body = resp.json()
        task_id = self._extract_first(
            body,
            [
                ("task_id",),
                ("id",),
                ("data", "task_id"),
                ("data", "id"),
                ("data", "task", "id"),
                ("output", "task_id"),
            ],
        )
        if not task_id:
            raise ByteDanceVideoClientError(
                f"Task submit response missing task id: keys={list(body.keys())} body={body!r}"
            )
        return str(task_id)

    def poll_task_until_success(self, task_id: str) -> str:
        status_url = f"{self.config.base_url}{self.config.task_path_template.format(task_id=task_id)}"
        deadline = time.time() + self.config.poll_timeout_s
        sleep_s = max(0.1, self.config.poll_backoff_start_s)

        while time.time() < deadline:
            resp = requests.get(status_url, headers=self._headers(), timeout=self.config.timeout_s)
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                raise ByteDanceVideoClientError(
                    f"Task polling failed for {task_id} ({resp.status_code}): {resp.text[:600]}"
                ) from e

            body = resp.json()
            raw_status = self._extract_first(
                body,
                [
                    ("status",),
                    ("state",),
                    ("data", "status"),
                    ("data", "state"),
                    ("output", "status"),
                ],
            )
            status = str(raw_status or "").upper()

            if status in {"SUCCESS", "SUCCEEDED", "DONE", "COMPLETED"}:
                video_url = self._extract_first(
                    body,
                    [
                        ("video_url",),
                        ("url",),
                        ("content", "video_url"),
                        ("data", "content", "video_url"),
                        ("data", "video_url"),
                        ("data", "url"),
                        ("output", "video_url"),
                        ("output", "url"),
                        ("result", "video_url"),
                        ("result", "url"),
                    ],
                )
                if not video_url:
                    raise ByteDanceVideoClientError(
                        f"Task {task_id} reached success but no video url in response."
                    )
                return str(video_url)

            if status in {"FAILED", "FAILURE", "ERROR", "CANCELED", "CANCELLED"}:
                reason = self._extract_first(
                    body,
                    [("error",), ("message",), ("data", "error"), ("data", "message")],
                )
                raise ByteDanceVideoClientError(
                    f"Task {task_id} ended with status={status}. Reason={reason}"
                )

            time.sleep(sleep_s)
            sleep_s = min(self.config.poll_backoff_max_s, sleep_s * 1.8)

        raise ByteDanceVideoClientError(
            f"Timed out waiting for task {task_id} after {self.config.poll_timeout_s}s"
        )

    def download_video(self, video_url: str, output_path: str | Path) -> str:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(video_url, stream=True, timeout=self.config.timeout_s) as resp:
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                raise ByteDanceVideoClientError(
                    f"Video download failed ({resp.status_code}): {video_url}"
                ) from e
            with out.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        if not out.exists() or out.stat().st_size == 0:
            raise ByteDanceVideoClientError(f"Downloaded file is empty at {out}")
        return str(out.resolve())

    def generate_video_and_wait(
        self,
        prompt: str,
        output_path: str | Path,
        scene_id: int | str,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        duration_s: float | None = None,
        image_url: str | None = None,
    ) -> str:
        def _attempt(p: str) -> str:
            task_id = self.submit_video_task(
                prompt=p,
                scene_id=scene_id,
                width=width,
                height=height,
                seed=seed,
                duration_s=duration_s,
                image_url=image_url,
            )
            video_url = self.poll_task_until_success(task_id)
            return self.download_video(video_url, output_path)

        try:
            return _attempt(prompt)
        except ByteDanceVideoClientError as e:
            if not self._is_invalid_content_text_error(str(e)):
                raise
            simplified = self._sanitize_prompt_for_retry(prompt)
            if simplified.strip() == (prompt or "").strip():
                raise
            return _attempt(simplified)
