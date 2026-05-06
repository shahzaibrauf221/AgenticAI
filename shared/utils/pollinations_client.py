import asyncio
import random
import re
from pathlib import Path
from urllib.parse import quote

import httpx


def _sanitize_prompt(prompt: str, max_chars: int = 260) -> str:
    """
    Pollinations is sensitive to very long prose prompts. Normalize into compact tags.
    """
    text = (prompt or "").replace("\n", ",")
    parts = [p.strip() for p in text.split(",")]
    keep: list[str] = []
    for p in parts:
        if not p:
            continue
        p = re.sub(r"\s+", " ", p)
        if p not in keep:
            keep.append(p)
    compact = ", ".join(keep)
    if len(compact) <= max_chars:
        return compact
    trimmed = compact[:max_chars].rsplit(" ", 1)[0].rstrip(" ,")
    return trimmed or compact[:max_chars]


async def fetch_pollinations_image(prompt: str, output_path: str, max_retries: int = 5) -> str:
    """
    Fetch a generated image from Pollinations and save it locally.

    URL format:
      https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true
    """
    cleaned_prompt = _sanitize_prompt(prompt)
    encoded_prompt = quote(cleaned_prompt.strip(), safe="")
    if not encoded_prompt:
        raise ValueError("prompt must be a non-empty string")

    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=1024&height=576&nologo=true"
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(connect=20.0, read=120.0, write=20.0, pool=30.0)
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.get(url)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"Transient status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                if not resp.content:
                    raise RuntimeError("Pollinations returned empty response body")
                out.write_bytes(resp.content)
                return str(out.resolve())
            except Exception as e:
                last_error = e
                if attempt >= max_retries:
                    break
                # Exponential backoff with jitter; treat 429 as normal transient throttle.
                base = min(12.0, 0.8 * (2**attempt))
                await asyncio.sleep(base + random.uniform(0.05, 0.5))

    raise RuntimeError(f"Pollinations fetch failed after retries: {last_error}")


def fetch_pollinations_image_sync(prompt: str, output_path: str) -> str:
    """Sync helper for non-async call sites."""
    return asyncio.run(fetch_pollinations_image(prompt, output_path))
