"""RIS-aided multidimensional OMP baseline."""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from .backend import BackendConfig, choose_batch_size, get_backend
from .cache import BASELINE_CACHE, baseline_cache_key, cache_diagnostics_delta
from .common import (
    BaselineResult,
    delay_grid_from_scene,
    geometric_support_to_position_ls,
    raw_atom_from_support,
    reconstruct_from_supports,
    simple_atom_normalize,
    vectorize_raw_observation,
)
from ..experiments.resource_control import trim_memory


def _ux_uy_direction_grid(size: int) -> list[tuple[int, int, float, float, np.ndarray]]:
    axis_count = max(3, int(np.sqrt(max(size, 3))))
    axis = np.linspace(-0.8, 0.8, axis_count)
    directions = []
    for ux_idx, ux in enumerate(axis):
        for uy_idx, uy in enumerate(axis):
            rem = 1.0 - ux * ux - uy * uy
            if rem <= 0.0:
                continue
            directions.append((int(ux_idx), int(uy_idx), float(ux), float(uy), np.array([ux, uy, np.sqrt(rem)], dtype=float)))
            directions.append((int(ux_idx), int(uy_idx), float(ux), float(uy), np.array([ux, uy, -np.sqrt(rem)], dtype=float)))
    return directions


def _support_key(support: dict[str, Any]) -> tuple[Any, ...]:
    direction = tuple(np.round(np.asarray(support.get("direction", [0, 0, 0]), dtype=float), 8))
    return (
        int(support.get("panel", 0)),
        direction,
        round(float(support.get("tau", 0.0)), 15),
        int(support.get("pol_index", 0)),
    )


def _ris_momp_supports(scene: dict, config: dict) -> Iterable[dict[str, Any]]:
    cfg = dict(config.get("baselines", {}).get("ris_momp", {}))
    directions = _ux_uy_direction_grid(int(cfg.get("direction_grid_size", 31)))
    taus = delay_grid_from_scene(scene, config, int(cfg.get("delay_grid_size", 41)))
    pol_indices = range(2 if bool(cfg.get("use_jones_basis", True)) else 1)
    for panel in range(int(scene["K"])):
        for direction_idx, (ux_idx, uy_idx, ux, uy, direction) in enumerate(directions):
            for tau_idx, tau in enumerate(taus):
                for pol_index in pol_indices:
                    yield {
                        "panel": int(panel),
                        "panel_index": int(panel),
                        "direction": direction,
                        "u_x": float(ux),
                        "u_y": float(uy),
                        "u_x_index": int(ux_idx),
                        "u_y_index": int(uy_idx),
                        "tau": float(tau),
                        "pol_index": int(pol_index),
                        "direction_index": int(direction_idx),
                        "tau_index": int(tau_idx),
                        "near_field": False,
                    }


def run_ris_momp_baseline(data: dict, config: dict) -> BaselineResult:
    start = time.perf_counter()
    scene = data["scene"]
    y_vec = vectorize_raw_observation(data["Y_noisy"])
    cfg = dict(config.get("baselines", {}).get("ris_momp", {}))
    backend_cfg = BackendConfig.from_value(
        config.get("baselines", {}).get("backend_config")
    )
    backend = get_backend(backend_cfg)
    BASELINE_CACHE.configure(
        enabled=backend_cfg.cache_enabled,
        memory_budget_gb=backend_cfg.cache_memory_budget_gb,
    )
    cache_before = BASELINE_CACHE.snapshot()
    cache_key = baseline_cache_key(
        "ris_momp",
        scene,
        config,
        grid_sizes=(
            int(cfg.get("direction_grid_size", 31)),
            int(cfg.get("delay_grid_size", 41)),
            bool(cfg.get("use_jones_basis", True)),
        ),
    )
    supports = BASELINE_CACHE.get_or_create(
        cache_key,
        lambda: list(_ris_momp_supports(scene, config)),
    )
    max_batch_memory_mb = float(cfg.get("max_batch_memory_mb", 256.0))
    requested_batch_size = int(cfg.get("batch_size", 256))
    memory_budget_bytes = max_batch_memory_mb * 1024.0**2
    if backend.name == "cupy" and backend_cfg.gpu_memory_fraction is not None:
        memory_budget_bytes = min(
            memory_budget_bytes,
            float(backend.memory_info()["free_bytes"]) * backend_cfg.gpu_memory_fraction,
        )
    memory_batch_size = choose_batch_size(
        len(supports), y_vec.size, memory_budget_bytes, np.complex128
    )
    requested_batch_size = (
        backend_cfg.gpu_batch_size
        if backend.name == "cupy" and backend_cfg.gpu_batch_size is not None
        else backend_cfg.cpu_batch_size
        if backend.name == "cpu" and backend_cfg.cpu_batch_size is not None
        else requested_batch_size
    )
    batch_size = max(1, min(int(requested_batch_size), memory_batch_size))
    num_batches = 0
    residual = y_vec.copy()
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()
    y_hat = np.zeros_like(y_vec)
    scoring_start = time.perf_counter()
    for _ in range(int(cfg.get("max_atoms", scene["K"]))):
        best_score = -np.inf
        best_support = None
        residual_device = (
            backend.asarray(residual, dtype=backend.xp.complex128)
            if backend.name == "cupy"
            else None
        )
        for batch_start in range(0, len(supports), batch_size):
            batch_supports = supports[batch_start : batch_start + batch_size]
            if backend.name == "cupy":
                atoms_cpu = np.column_stack(
                    [raw_atom_from_support(scene, config, support) for support in batch_supports]
                )
                atoms_device = backend.asarray(atoms_cpu, dtype=backend.xp.complex128)
                norms = backend.xp.linalg.norm(atoms_device, axis=0)
                atoms_device = atoms_device / backend.xp.where(norms > 0.0, norms, 1.0)
                scores_device = backend.abs(atoms_device.conj().T @ residual_device)
                for local_idx, support in enumerate(batch_supports):
                    if _support_key(support) in selected_keys:
                        scores_device[local_idx] = -backend.xp.inf
                local_best = int(backend.to_host(backend.argmax(scores_device)))
                local_score = float(backend.to_host(scores_device[local_best]))
                scores = None
            else:
                atoms_cpu = np.column_stack(
                    [
                        simple_atom_normalize(
                            raw_atom_from_support(scene, config, support)
                        )
                        for support in batch_supports
                    ]
                )
                scores = np.abs(atoms_cpu.conj().T @ residual)
            num_batches += 1
            if backend.name == "cupy":
                if local_score > best_score:
                    best_score = local_score
                    best_support = batch_supports[local_best]
            else:
                for local_idx, support in enumerate(batch_supports):
                    if _support_key(support) in selected_keys:
                        continue
                    score = float(scores[local_idx])
                    if score > best_score:
                        best_score = score
                        best_support = support
            if backend.name == "cupy":
                del atoms_device, scores_device
            del atoms_cpu, scores, batch_supports
            if bool(config.get("baselines", {}).get("trim_memory", True)):
                trim_memory()
        if best_support is None:
            break
        selected.append(dict(best_support))
        selected_keys.add(_support_key(best_support))
        _, y_hat_tensor, residual, _ = reconstruct_from_supports(scene, config, selected, y_vec)
        y_hat = y_hat_tensor.reshape(-1)
    backend.synchronize()
    p_hat, delta_t, geom_diag = geometric_support_to_position_ls(scene, selected, config)
    diagnostics = {
        "dictionary_mode": "batched_momp_equivalent",
        "grid_size": len(supports),
        "direction_grid_size": int(cfg.get("direction_grid_size", 31)),
        "delay_grid_size": int(cfg.get("delay_grid_size", 41)),
        "support_size": len(selected),
        "grid_sizes": {
            "direction": int(cfg.get("direction_grid_size", 31)),
            "delay": int(cfg.get("delay_grid_size", 41)),
            "panel": int(scene["K"]),
        },
        "selected_support": selected,
        "residual_norm": float(np.linalg.norm(residual)),
        "batch_size": batch_size,
        "max_batch_memory_mb": max_batch_memory_mb,
        "num_batches": num_batches,
        "backend": backend.name,
        "gpu_used": backend.name == "cupy",
        "gpu_num_batches": num_batches if backend.name == "cupy" else 0,
        "gpu_batch_size": batch_size if backend.name == "cupy" else "",
        "gpu_device": backend.device if backend.name == "cupy" else "",
        "scoring_time_s": time.perf_counter() - scoring_start,
        "backend_warning": backend.warning,
        **geom_diag,
    }
    diagnostics.update(cache_diagnostics_delta(cache_before, BASELINE_CACHE.snapshot()))
    raw_objective = float(np.linalg.norm(residual) ** 2 / y_vec.size)
    return BaselineResult(
        name="ris_momp",
        p_u=p_hat,
        delta_t=delta_t,
        Y_hat=y_hat.reshape(scene["I"], scene["N"], scene["T"]),
        raw_objective_final=raw_objective,
        components={
            "taus": np.asarray([support.get("tau", np.nan) for support in selected], dtype=float),
            "ranges": np.full(len(selected), np.nan),
        },
        selected_support=selected,
        runtime_s=time.perf_counter() - start,
        diagnostics=diagnostics,
    )
