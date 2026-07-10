"""Parent-process JSONL progress logging for long experiment runs."""

from __future__ import annotations

import json
import math
import os
import pathlib
import platform
import time
from datetime import datetime, timezone
from typing import Any


def _json_safe(value: Any) -> Any:
    """Map non-finite floats to None so strict JSON encoding never fails.

    NaN and inf are legitimate values for optional metrics (an unavailable PEB,
    an uninitialized condition number), but ``allow_nan=False`` rejects them.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def memory_snapshot_mb() -> float:
    """Return current-process RSS in MiB when available."""
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0**2))
    except (ImportError, OSError):
        return float("nan")


def format_eta(seconds: float | None) -> str:
    """Format a duration as HH:MM:SS, or --:--:-- when unknown."""
    if seconds is None:
        return "--:--:--"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "--:--:--"
    if value < 0.0 or value != value or value == float("inf"):
        return "--:--:--"
    total = int(round(value))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressLogger:
    """Append flushed progress events from the parent process."""

    def __init__(
        self,
        path: str | pathlib.Path,
        total_tasks: int,
        script_name: str,
    ) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        self.total_tasks = max(0, int(total_tasks))
        self.script_name = str(script_name)
        self.done_tasks = 0
        self.started = time.monotonic()
        self.hostname = platform.node()
        self.pid = os.getpid()

    def log(self, event_type: str, status: str, **kwargs: Any) -> dict[str, Any]:
        if event_type in {"task_done", "task_failed"}:
            self.done_tasks = min(self.total_tasks, self.done_tasks + 1)
        elif event_type == "finished":
            self.done_tasks = self.total_tasks
        elapsed = max(0.0, time.monotonic() - self.started)
        percent = (
            100.0 * self.done_tasks / self.total_tasks
            if self.total_tasks
            else 100.0
        )
        eta = (
            elapsed * (self.total_tasks - self.done_tasks) / self.done_tasks
            if self.done_tasks > 0 and self.done_tasks < self.total_tasks
            else 0.0
            if self.done_tasks >= self.total_tasks
            else None
        )
        rss = memory_snapshot_mb()
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": self.hostname,
            "pid": self.pid,
            "script": self.script_name,
            "event_type": str(event_type),
            "status": str(status),
            "done_tasks": self.done_tasks,
            "total_tasks": self.total_tasks,
            "percent": percent,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "rss_mb": rss if math.isfinite(rss) else None,
            "figure": kwargs.pop("figure", ""),
            "baseline_or_variant": kwargs.pop("baseline_or_variant", ""),
            "snr_db": kwargs.pop("snr_db", ""),
            "trial_id": kwargs.pop("trial_id", ""),
            "seed": kwargs.pop("seed", ""),
            "K": kwargs.pop("K", ""),
            "message": kwargs.pop("message", ""),
            "error": kwargs.pop("error", ""),
        }
        event.update(kwargs)
        self.handle.write(
            json.dumps(_json_safe(event), default=str, allow_nan=False) + "\n"
        )
        self.handle.flush()
        return event

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            self.handle.close()
