"""Shared CPU-thread and worker-memory controls for experiment runners."""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import sys
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_ACTIVE_THREAD_LIMITER: Any | None = None


def resolve_hybrid_resources(
    jobs: int,
    process_workers: int | None,
    blas_threads: int | str,
    n_tasks: int,
    memory_budget_gb: float | None = None,
    memory_per_worker_gb: float | None = None,
) -> dict[str, int]:
    """Resolve a process/thread plan within the requested CPU and memory budgets."""
    jobs = max(1, int(jobs))
    n_tasks = max(1, int(n_tasks))
    memory_cap: int | None = None
    if memory_budget_gb is not None and memory_per_worker_gb is not None:
        if float(memory_budget_gb) <= 0.0 or float(memory_per_worker_gb) <= 0.0:
            raise ValueError("memory budgets must be positive")
        memory_cap = max(
            1,
            int(math.floor(float(memory_budget_gb) / float(memory_per_worker_gb))),
        )

    if process_workers is None:
        workers = min(jobs, n_tasks)
    else:
        workers = max(1, int(process_workers))
        workers = min(workers, jobs)
    if memory_cap is not None:
        workers = min(workers, memory_cap)
    workers = max(1, workers)

    if isinstance(blas_threads, str):
        if blas_threads.lower() != "auto":
            raise ValueError("blas_threads must be a positive integer or 'auto'")
        threads = max(1, jobs // workers)
    else:
        threads = max(1, int(blas_threads))
        threads = min(threads, max(1, jobs // workers))

    return {
        "jobs": jobs,
        "process_workers": workers,
        "blas_threads": threads,
        "n_tasks": n_tasks,
        "estimated_cpu_slots": workers * threads,
    }


def apply_thread_limits(
    blas_threads: int,
    respect_existing: bool = False,
) -> None:
    """Apply environment and loaded-runtime BLAS thread limits."""
    global _ACTIVE_THREAD_LIMITER
    threads = max(1, int(blas_threads))
    value = str(threads)
    for name in THREAD_ENV_VARS:
        if respect_existing:
            os.environ.setdefault(name, value)
        else:
            os.environ[name] = value
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    if _ACTIVE_THREAD_LIMITER is not None:
        try:
            _ACTIVE_THREAD_LIMITER.restore_original_limits()
        except Exception:
            pass
    _ACTIVE_THREAD_LIMITER = threadpool_limits(limits=threads)


@contextlib.contextmanager
def thread_limit_context(blas_threads: int) -> Iterator[None]:
    """Temporarily constrain loaded native thread pools."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        with contextlib.nullcontext():
            yield
        return
    with threadpool_limits(limits=max(1, int(blas_threads))):
        yield


def memory_snapshot_mb() -> float:
    """Return current-process RSS in MiB, with a Linux /proc fallback."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0**2))
        except Exception:
            pass
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, IndexError, TypeError, ValueError):
        pass
    return float("nan")


class PeakRssSampler:
    """Sample current-process RSS while a bounded experiment block runs."""

    def __init__(self, enabled: bool, interval_s: float = 0.01):
        self.enabled = bool(enabled)
        self.interval_s = float(interval_s)
        self.peak_mb = float("nan")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.peak_mb = memory_snapshot_mb()

        def sample() -> None:
            while not self._stop.wait(self.interval_s):
                value = memory_snapshot_mb()
                if np.isfinite(value):
                    self.peak_mb = (
                        float(value)
                        if not np.isfinite(self.peak_mb)
                        else max(float(self.peak_mb), float(value))
                    )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, 2.0 * self.interval_s))
        value = memory_snapshot_mb()
        if np.isfinite(value):
            self.peak_mb = (
                float(value)
                if not np.isfinite(self.peak_mb)
                else max(float(self.peak_mb), float(value))
            )


def release_gpu_memory_pools() -> None:
    """Return cached free blocks from CuPy memory pools to the device.

    Guarded on ``"cupy" in sys.modules`` so CPU-only runs (where cupy is never
    imported by the backend selector) incur zero cost and never trigger CUDA
    context initialization. On GPU runs this stops each worker process from
    holding its peak device footprint for the whole job. Only blocks that are
    no longer referenced are released, so results are unchanged; the
    re-allocation cost on the next trial is negligible relative to trial time.
    """
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return
    try:
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def trim_memory() -> None:
    """Collect Python garbage and best-effort release free glibc arenas.

    When the CuPy GPU backend has been used in this process, also return cached
    free blocks from the CuPy memory pools to the device. This is a no-op on
    CPU-only runs, so CPU performance is unaffected.
    """
    gc.collect()
    release_gpu_memory_pools()
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def assert_row_is_light(row: dict[str, Any]) -> dict[str, Any]:
    """Return a flat CSV-safe row and reject large worker payloads."""
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                safe[key] = value.item()
            elif value.size <= 64:
                safe[key] = json.dumps(value.tolist(), separators=(",", ":"))
            else:
                raise ValueError(
                    f"worker row field {key!r} contains ndarray with {value.size} elements"
                )
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        elif isinstance(value, (list, tuple, dict)):
            payload = json.dumps(value, default=_json_default, separators=(",", ":"))
            if len(payload) > 16_384:
                raise ValueError(f"worker row field {key!r} contains a huge object")
            safe[key] = payload
        else:
            safe[key] = value
    if any(isinstance(value, np.ndarray) for value in safe.values()):
        raise AssertionError("worker row still contains an ndarray")
    return safe
