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
    expand_jones_group,
    fit_position_clock_data_domain,
    geometric_support_to_position_ls,
    group_omp_select,
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
    )


def _range_grid(scene: dict, config: dict, size: int) -> np.ndarray:
    bounds = np.asarray(
        config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]),
        dtype=float,
    )
    corners = np.array(
        [[x, y, z] for x in bounds[0] for y in bounds[1] for z in bounds[2]],
        dtype=float,
    )
    ranges = []
    for panel in range(int(scene["K"])):
        ranges.extend(
            np.linalg.norm(
                corners - np.asarray(scene["ris_centers"][panel], dtype=float),
                axis=1,
            ).tolist()
        )
    return np.linspace(float(np.min(ranges)), float(np.max(ranges)), max(2, int(size)))


def _ris_momp_supports(scene: dict, config: dict) -> Iterable[dict[str, Any]]:
    cfg = dict(config.get("baselines", {}).get("ris_momp", {}))
    directions = _ux_uy_direction_grid(int(cfg.get("direction_grid_size", 31)))
    taus = delay_grid_from_scene(scene, config, int(cfg.get("delay_grid_size", 41)))
    ranges = _range_grid(scene, config, int(cfg.get("range_grid_size", cfg.get("direction_grid_size", 31))))
    for panel in range(int(scene["K"])):
        for direction_idx, (ux_idx, uy_idx, ux, uy, direction) in enumerate(directions):
            for range_idx, range_m in enumerate(ranges):
                q_global = np.asarray(scene["rotations"][panel], dtype=float).T @ (
                    float(range_m) * direction
                )
                position = np.asarray(scene["ris_centers"][panel], dtype=float) + q_global
                for tau_idx, tau in enumerate(taus):
                    yield {
                        "panel": int(panel),
                        "panel_index": int(panel),
                        "position": position.astype(float),
                        "range": float(range_m),
                        "direction": direction,
                        "u_x": float(ux),
                        "u_y": float(uy),
                        "u_x_index": int(ux_idx),
                        "u_y_index": int(uy_idx),
                        "range_index": int(range_idx),
                        "tau": float(tau),
                        "direction_index": int(direction_idx),
                        "tau_index": int(tau_idx),
                        "near_field": True,
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
            int(cfg.get("range_grid_size", cfg.get("direction_grid_size", 31))),
            int(cfg.get("delay_grid_size", 41)),
            "near_field_jones_group_omp",
        ),
    )
    groups = BASELINE_CACHE.get_or_create(
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
    memory_batch_size = choose_batch_size(len(groups), y_vec.size, memory_budget_bytes, np.complex128)
    requested_batch_size = (
        backend_cfg.gpu_batch_size
        if backend.name == "cupy" and backend_cfg.gpu_batch_size is not None
        else backend_cfg.cpu_batch_size
        if backend.name == "cpu" and backend_cfg.cpu_batch_size is not None
        else requested_batch_size
    )
    batch_size = max(1, min(int(requested_batch_size), memory_batch_size))
    max_groups = int(cfg.get("max_groups", cfg.get("max_atoms", scene["K"])))
    selected, expanded_supports, coeffs, y_hat, diagnostics = group_omp_select(
        scene,
        config,
        groups,
        y_vec,
        max_groups=max_groups,
        batch_size=batch_size,
        trim_memory_enabled=bool(config.get("baselines", {}).get("trim_memory", True)),
    )
    residual = y_vec - y_hat
    backend.synchronize()
    p_hat, delta_t, geom_diag = geometric_support_to_position_ls(scene, selected, config)
    panels = [int(support.get("panel", 0)) for support in selected]
    if bool(cfg.get("offgrid_refinement", True)) and selected:
        p_hat, delta_t, coeffs, y_hat, refined_groups, refine_diag = fit_position_clock_data_domain(
            scene,
            config,
            y_vec,
            p_hat,
            delta_t,
            panels=panels,
            model_variant="near_field",
            enabled=True,
            max_nfev=int(cfg.get("refinement_max_nfev", 60)),
        )
        selected = [
            {**selected[idx], **refined_groups[idx]}
            for idx in range(min(len(selected), len(refined_groups)))
        ]
        expanded_supports = [
            support for group in selected for support in expand_jones_group(group, 2)
        ]
        residual = y_vec - y_hat
        diagnostics.update(refine_diag)
    else:
        diagnostics["offgrid_refinement"] = False
        diagnostics["refinement_objective"] = ""
    diagnostics = {
        **diagnostics,
        "dictionary_mode": "near_field_range_aware_momp",
        "model_variant": "near_field_momp",
        "group_omp": True,
        "grid_size": len(groups),
        "direction_grid_size": int(cfg.get("direction_grid_size", 31)),
        "range_grid_size": int(cfg.get("range_grid_size", cfg.get("direction_grid_size", 31))),
        "delay_grid_size": int(cfg.get("delay_grid_size", 41)),
        "support_size": len(selected),
        "grid_sizes": {
            "direction": int(cfg.get("direction_grid_size", 31)),
            "range": int(cfg.get("range_grid_size", cfg.get("direction_grid_size", 31))),
            "delay": int(cfg.get("delay_grid_size", 41)),
            "panel": int(scene["K"]),
        },
        "selected_support": selected,
        "expanded_supports": expanded_supports,
        "coeff_norm": float(np.linalg.norm(coeffs)),
        "residual_norm": float(np.linalg.norm(residual)),
        "batch_size": batch_size,
        "max_batch_memory_mb": max_batch_memory_mb,
        "backend": backend.name,
        "gpu_used": backend.name == "cupy",
        "gpu_num_batches": 0,
        "gpu_batch_size": batch_size if backend.name == "cupy" else "",
        "gpu_device": backend.device if backend.name == "cupy" else "",
        "backend_warning": backend.warning,
        "max_groups": max_groups,
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
