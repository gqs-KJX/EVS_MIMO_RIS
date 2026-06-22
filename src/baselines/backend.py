"""Array backends for optional benchmark-baseline acceleration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class BackendConfig:
    backend: Literal["cpu", "cupy", "auto"] = "cpu"
    gpu_device: int | None = 0
    gpu_batch_size: int | None = None
    cpu_batch_size: int | None = None
    cache_enabled: bool = True
    cache_memory_budget_gb: float | None = None
    gpu_memory_fraction: float | None = None
    dtype: Literal["complex128"] = "complex128"

    @classmethod
    def from_value(cls, value: "BackendConfig | dict[str, Any] | None") -> "BackendConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in dict(value).items() if key in fields})

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "cupy", "auto"}:
            raise ValueError("backend must be 'cpu', 'cupy', or 'auto'")
        if self.dtype != "complex128":
            raise ValueError("only complex128 is supported")
        for name in ("gpu_batch_size", "cpu_batch_size"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cache_memory_budget_gb is not None and self.cache_memory_budget_gb <= 0:
            raise ValueError("cache_memory_budget_gb must be positive")
        if self.gpu_memory_fraction is not None and not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must be in (0, 1]")


class CPUBackend:
    name = "cpu"
    xp = np
    available = True
    warning = ""
    device = None

    def to_device(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def to_host(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def asarray(self, x: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(x, dtype=dtype)

    def zeros(self, shape: tuple[int, ...] | int, dtype: Any) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def empty(self, shape: tuple[int, ...] | int, dtype: Any) -> np.ndarray:
        return np.empty(shape, dtype=dtype)

    def norm(self, x: Any) -> Any:
        return np.linalg.norm(x)

    def abs(self, x: Any) -> Any:
        return np.abs(x)

    def argmax(self, x: Any) -> Any:
        return np.argmax(x)

    def vdot(self, a: Any, b: Any) -> Any:
        return np.vdot(a, b)

    def matmul(self, a: Any, b: Any) -> Any:
        return np.matmul(a, b)

    def solve(self, A: Any, b: Any) -> Any:
        return np.linalg.solve(A, b)

    def lstsq(self, A: Any, b: Any, rcond: Any = None) -> Any:
        return np.linalg.lstsq(A, b, rcond=rcond)

    def synchronize(self) -> None:
        return None

    def memory_info(self) -> dict[str, Any]:
        return {"backend": "cpu", "device": None, "free_bytes": None, "total_bytes": None}


class CuPyBackend:
    name = "cupy"
    available = True

    def __init__(self, cp: Any, config: BackendConfig):
        self.xp = cp
        self.warning = ""
        self.device = int(config.gpu_device) if config.gpu_device is not None else None
        if self.device is not None:
            cp.cuda.Device(self.device).use()

    def to_device(self, x: Any) -> Any:
        return self.xp.asarray(x)

    def to_host(self, x: Any) -> np.ndarray:
        return self.xp.asnumpy(x)

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        return self.xp.asarray(x, dtype=dtype)

    def zeros(self, shape: tuple[int, ...] | int, dtype: Any) -> Any:
        return self.xp.zeros(shape, dtype=dtype)

    def empty(self, shape: tuple[int, ...] | int, dtype: Any) -> Any:
        return self.xp.empty(shape, dtype=dtype)

    def norm(self, x: Any) -> Any:
        return self.xp.linalg.norm(x)

    def abs(self, x: Any) -> Any:
        return self.xp.abs(x)

    def argmax(self, x: Any) -> Any:
        return self.xp.argmax(x)

    def vdot(self, a: Any, b: Any) -> Any:
        return self.xp.vdot(a, b)

    def matmul(self, a: Any, b: Any) -> Any:
        return self.xp.matmul(a, b)

    def solve(self, A: Any, b: Any) -> Any:
        return self.xp.linalg.solve(A, b)

    def lstsq(self, A: Any, b: Any, rcond: Any = None) -> Any:
        return self.xp.linalg.lstsq(A, b, rcond=rcond)

    def synchronize(self) -> None:
        self.xp.cuda.get_current_stream().synchronize()

    def memory_info(self) -> dict[str, Any]:
        free_bytes, total_bytes = self.xp.cuda.runtime.memGetInfo()
        pool = self.xp.get_default_memory_pool()
        return {
            "backend": "cupy",
            "device": self.device,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
        }


def get_backend(config_or_dict: BackendConfig | dict[str, Any] | None = None) -> CPUBackend | CuPyBackend:
    config = BackendConfig.from_value(config_or_dict)
    if config.backend == "cpu":
        return CPUBackend()
    try:
        import cupy as cp  # type: ignore[import-not-found]

        backend = CuPyBackend(cp, config)
        backend.synchronize()
        return backend
    except Exception as exc:
        message = f"CuPy backend unavailable ({type(exc).__name__}: {exc})"
        if config.backend == "cupy":
            raise RuntimeError(message) from exc
        warnings.warn(f"{message}; falling back to CPU.", RuntimeWarning, stacklevel=2)
        backend = CPUBackend()
        backend.warning = f"{message}; falling back to CPU."
        return backend


def estimate_array_bytes(shape: tuple[int, ...] | list[int] | int, dtype: Any) -> int:
    if isinstance(shape, int):
        shape = (shape,)
    return int(np.prod(tuple(int(value) for value in shape), dtype=np.int64) * np.dtype(dtype).itemsize)


def choose_batch_size(
    num_candidates: int,
    atom_length: int,
    memory_budget_bytes: int | float | None,
    dtype: Any,
) -> int:
    num_candidates = max(1, int(num_candidates))
    if memory_budget_bytes is None:
        return num_candidates
    bytes_per_candidate = max(estimate_array_bytes((int(atom_length),), dtype), 1)
    return max(1, min(num_candidates, int(float(memory_budget_bytes) // bytes_per_candidate)))
