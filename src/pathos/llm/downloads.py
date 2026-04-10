"""Download manager for Ollama model pulls.

Tracks active downloads, streams progress, supports cancellation.
Max 2 concurrent downloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 2


@dataclass
class DownloadStatus:
    name: str
    status: str = "pending"  # pending | downloading | success | error | cancelled
    completed: int = 0
    total: int = 0
    percent: float = 0.0
    error: str = ""
    started_at: float = field(default_factory=time.time)


class DownloadManager:
    """Manages Ollama model downloads with progress tracking."""

    def __init__(self, ollama_base_url: str = "http://127.0.0.1:11434") -> None:
        self.ollama_url = ollama_base_url
        self._active: set[str] = set()
        self._cancel_requested: set[str] = set()
        self._progress: dict[str, DownloadStatus] = {}
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    def get_downloads(self) -> list[dict]:
        """Return all active/recent download statuses."""
        return [
            {
                "name": s.name,
                "status": s.status,
                "completed": s.completed,
                "total": s.total,
                "percent": s.percent,
                "error": s.error,
            }
            for s in self._progress.values()
            if s.status not in ("success",)  # keep errors visible
            or time.time() - s.started_at < 30  # show success for 30s
        ]

    def is_downloading(self, name: str) -> bool:
        return name in self._active

    async def pull(self, name: str):
        """Start pulling a model. Yields progress dicts as SSE-ready lines."""
        if self.is_downloading(name):
            yield {"status": "already_downloading", "name": name}
            return

        self._active.add(name)
        self._cancel_requested.discard(name)
        self._progress[name] = DownloadStatus(name=name, status="downloading")

        try:
            async with self._semaphore:
                async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0, read=3600.0)) as client:
                    async with client.stream(
                        "POST",
                        f"{self.ollama_url}/api/pull",
                        json={"name": name, "stream": True},
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            status = data.get("status", "")
                            total = data.get("total", 0)
                            completed = data.get("completed", 0)
                            pct = (completed / total * 100) if total > 0 else 0

                            self._progress[name].status = "downloading"
                            self._progress[name].total = total
                            self._progress[name].completed = completed
                            self._progress[name].percent = round(pct, 1)

                            yield {
                                "status": status,
                                "total": total,
                                "completed": completed,
                                "percent": round(pct, 1),
                            }

                            if name in self._cancel_requested:
                                self._progress[name].status = "cancelled"
                                yield {"status": "cancelled", "name": name}
                                return

                            if "error" in data:
                                self._progress[name].status = "error"
                                self._progress[name].error = data["error"]
                                yield {"status": "error", "error": data["error"]}
                                return

                self._progress[name].status = "success"
                self._progress[name].percent = 100.0
                yield {"status": "success", "name": name}

        except asyncio.CancelledError:
            self._progress[name].status = "cancelled"
            yield {"status": "cancelled", "name": name}
        except Exception as e:
            self._progress[name].status = "error"
            self._progress[name].error = str(e)
            yield {"status": "error", "error": str(e)}
        finally:
            self._active.discard(name)
            self._cancel_requested.discard(name)

    def cancel(self, name: str) -> bool:
        """Cancel an active download. Returns True if cancellation was requested."""
        if name in self._active:
            self._cancel_requested.add(name)
            self._progress[name].status = "cancelled"
            return True
        return False

    def clear_completed(self) -> None:
        """Remove completed/errored entries older than 60s."""
        now = time.time()
        to_remove = [
            name for name, s in self._progress.items()
            if s.status in ("success", "error", "cancelled")
            and now - s.started_at > 60
        ]
        for name in to_remove:
            del self._progress[name]
