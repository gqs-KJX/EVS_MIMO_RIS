"""Process-local bounded cache for baseline grids and static responses."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .common import hash_array


_FORBIDDEN_NAMES = {"Y_noisy", "y_noisy", "p_u_true", "true_position", "polarization_true"}


def _contains_forbidden_name(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key) in _FORBIDDEN_NAMES or _contains_forbidden_name(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_name(child) for child in value)
    return False


def estimate_value_bytes(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            estimate_value_bytes(key, seen) + estimate_value_bytes(child, seen)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(estimate_value_bytes(child, seen) for child in value)
    return sys.getsizeof(value)


@dataclass
class CacheDiagnostics:
    cache_enabled: bool
    cache_hits: int
    cache_misses: int
    cache_estimated_bytes: int


class BaselineCache:
    def __init__(self) -> None:
        self._values: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self.enabled = True
        self.memory_budget_bytes: int | None = None
        self.hits = 0
        self.misses = 0
        self.estimated_bytes = 0

    def configure(self, *, enabled: bool, memory_budget_gb: float | None) -> None:
        self.enabled = bool(enabled)
        self.memory_budget_bytes = (
            None if memory_budget_gb is None else max(1, int(float(memory_budget_gb) * 1024**3))
        )
        self._evict_to_budget()

    def clear(self) -> None:
        self._values.clear()
        self.estimated_bytes = 0

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            self.misses += 1
            return None
        entry = self._values.pop(key, None)
        if entry is None:
            self.misses += 1
            return None
        self._values[key] = entry
        self.hits += 1
        return entry[0]

    def put(self, key: str, value: Any) -> bool:
        if not self.enabled:
            return False
        if _contains_forbidden_name(value):
            raise ValueError("baseline cache cannot store noisy observations or truth fields")
        size = estimate_value_bytes(value)
        if self.memory_budget_bytes is not None and size > self.memory_budget_bytes:
            return False
        previous = self._values.pop(key, None)
        if previous is not None:
            self.estimated_bytes -= previous[1]
        self._values[key] = (value, size)
        self.estimated_bytes += size
        self._evict_to_budget()
        return key in self._values

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> Any:
        value = self.get(key)
        if value is not None:
            return value
        value = factory()
        self.put(key, value)
        return value

    def snapshot(self) -> CacheDiagnostics:
        return CacheDiagnostics(
            cache_enabled=self.enabled,
            cache_hits=self.hits,
            cache_misses=self.misses,
            cache_estimated_bytes=self.estimated_bytes,
        )

    def _evict_to_budget(self) -> None:
        if self.memory_budget_bytes is None:
            return
        while self._values and self.estimated_bytes > self.memory_budget_bytes:
            _, (_, size) = self._values.popitem(last=False)
            self.estimated_bytes -= size


def baseline_cache_key(
    baseline_name: str,
    scene: dict[str, Any],
    config: dict[str, Any],
    *,
    grid_profile: str | None = None,
    grid_sizes: Any = None,
) -> str:
    ris_shape = config.get("ris_shape")
    if ris_shape is None:
        ris_shape = np.asarray(scene.get("ris_grid", [])).shape
    geometry = {
        "baseline": str(baseline_name),
        "K": int(scene.get("K", config.get("K", 0))),
        "receiver_mode": str(scene.get("receiver_mode", config.get("receiver_mode", ""))),
        "grid_profile": str(grid_profile if grid_profile is not None else config.get("grid_profile", "")),
        "grid_sizes": grid_sizes,
        "ris_shape": tuple(int(value) for value in np.asarray(ris_shape).reshape(-1)),
        "M_R": int(np.prod(ris_shape)) if np.asarray(ris_shape).size else int(scene.get("M_R", 0)),
        "N": int(scene.get("N", config.get("N", 0))),
        "T": int(scene.get("T", config.get("T", 0))),
        "fc": float(scene.get("fc", config.get("fc", 0.0))),
        "wavelength": float(scene.get("wavelength", config.get("wavelength", 0.0))),
        "delta_f": float(scene.get("delta_f", config.get("delta_f", 0.0))),
        "c0": float(scene.get("c0", config.get("c0", 0.0))),
        "ue_bounds": hash_array(config.get("ue_bounds", [])),
        "delta_t_bounds": hash_array(config.get("delta_t_bounds", [])),
        "p_B": hash_array(scene.get("p_B", config.get("p_B", []))),
        "ris_centers": hash_array(scene.get("ris_centers", config.get("ris_centers", []))),
        "d_RB": hash_array(scene.get("d_RB", [])),
        "rotations": hash_array(scene.get("rotations", [])),
        "ris_grid": hash_array(scene.get("ris_grid", [])),
        "Omega": hash_array(scene.get("Omega", [])),
        "a_RB": hash_array(scene.get("a_RB", [])),
        "Theta": hash_array(scene.get("Theta", [])),
        "v_B": hash_array(scene.get("v_B", [])),
        "scene_hash": str(scene.get("scene_hash", "")),
    }
    payload = json.dumps(geometry, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


BASELINE_CACHE = BaselineCache()


def cache_diagnostics_delta(before: CacheDiagnostics, after: CacheDiagnostics) -> dict[str, Any]:
    return {
        "cache_enabled": after.cache_enabled,
        "cache_hits": max(0, after.cache_hits - before.cache_hits),
        "cache_misses": max(0, after.cache_misses - before.cache_misses),
        "cache_estimated_bytes": after.cache_estimated_bytes,
    }
