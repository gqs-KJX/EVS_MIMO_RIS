"""RIS-aided multidimensional OMP baseline."""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from .backend import BackendConfig, choose_batch_size, get_backend
from .cache import BASELINE_CACHE, baseline_cache_key, cache_diagnostics_delta
from .factorized_scoring import factorized_fit_supports
from .common import (
    BaselineResult,
    delay_grid_from_scene,
    expand_jones_group,
    geometric_support_to_position_ls,
    group_omp_select,
    vectorize_raw_observation,
)


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
        round(float(support.get("range", np.nan)), 8)
        if "range" in support
        else None,
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
    ranges = _range_grid(
        scene,
        config,
        int(cfg.get("range_grid_size", cfg.get("direction_grid_size", 31))),
    )
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
                        "u_z_sign": float(np.sign(direction[2]) or 1.0),
                        "u_x_index": int(ux_idx),
                        "u_y_index": int(uy_idx),
                        "range_index": int(range_idx),
                        "tau": float(tau),
                        "direction_index": int(direction_idx),
                        "tau_index": int(tau_idx),
                        "near_field": True,
                        "group_size": 2,
                        "pol_indices": [0, 1],
                    }


def _direction_from_ux_uy(ux: float, uy: float, sign: float) -> np.ndarray:
    ux = float(np.clip(ux, -0.98, 0.98))
    uy = float(np.clip(uy, -0.98, 0.98))
    radius2 = ux * ux + uy * uy
    if radius2 >= 0.98:
        scale = np.sqrt(0.98 / (radius2 + 1.0e-15))
        ux *= scale
        uy *= scale
        radius2 = ux * ux + uy * uy
    uz = float(np.sign(sign) or 1.0) * np.sqrt(max(1.0 - radius2, 1.0e-12))
    return np.array([ux, uy, uz], dtype=float)


def _fit_groups(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    groups: list[dict[str, Any]],
    *,
    ridge: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    expanded = [support for group in groups for support in expand_jones_group(group, 2)]
    coeffs, y_hat, residual = factorized_fit_supports(
        scene, config, expanded, y_vec, ridge=ridge
    )
    return coeffs, y_hat, residual, expanded


def _grid_steps(groups: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    ux_values = sorted({round(float(group.get("u_x", 0.0)), 12) for group in groups})
    uy_values = sorted({round(float(group.get("u_y", 0.0)), 12) for group in groups})
    range_values = sorted({round(float(group.get("range", 0.0)), 12) for group in groups})
    tau_values = sorted({round(float(group.get("tau", 0.0)), 15) for group in groups})

    def spacing(values: list[float], fallback: float) -> float:
        if len(values) < 2:
            return fallback
        diffs = np.diff(np.asarray(values, dtype=float))
        diffs = diffs[np.abs(diffs) > 0.0]
        return float(np.min(np.abs(diffs))) if diffs.size else fallback

    return (
        spacing(ux_values, 0.2),
        spacing(uy_values, 0.2),
        spacing(range_values, 0.5),
        spacing(tau_values, 1.0e-9),
    )


def _refine_ris_momp_groups(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    selected: list[dict[str, Any]],
    all_groups: list[dict[str, Any]],
    *,
    levels: int,
    shrink: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, int]:
    if not selected:
        coeffs, y_hat, residual, _ = _fit_groups(scene, config, y_vec, [])
        return selected, coeffs, y_hat, residual, 0
    ux_step, uy_step, range_step, tau_step = _grid_steps(all_groups)
    refined = [dict(group) for group in selected]
    evals = 0
    range_min = min(float(group.get("range", 0.0)) for group in all_groups)
    range_max = max(float(group.get("range", 0.0)) for group in all_groups)
    tau_min = min(float(group["tau"]) for group in all_groups)
    tau_max = max(float(group["tau"]) for group in all_groups)
    for level in range(max(0, int(levels))):
        scale = float(shrink) ** level
        dux = ux_step * scale
        duy = uy_step * scale
        drange = range_step * scale
        dtau = tau_step * scale
        for group_index, group in enumerate(list(refined)):
            best_group = dict(group)
            _, _, best_residual, _ = _fit_groups(scene, config, y_vec, refined)
            best_objective = float(np.linalg.norm(best_residual) ** 2)
            for ox in (-dux, 0.0, dux):
                for oy in (-duy, 0.0, duy):
                    for orng in (-drange, 0.0, drange):
                        for ot in (-dtau, 0.0, dtau):
                            candidate = dict(group)
                            panel = int(group.get("panel", 0))
                            ux = float(group.get("u_x", 0.0)) + ox
                            uy = float(group.get("u_y", 0.0)) + oy
                            range_m = float(
                                np.clip(
                                    float(group.get("range", 0.0)) + orng,
                                    range_min,
                                    range_max,
                                )
                            )
                            tau = float(np.clip(float(group.get("tau", 0.0)) + ot, tau_min, tau_max))
                            direction = _direction_from_ux_uy(
                                ux, uy, float(group.get("u_z_sign", 1.0))
                            )
                            q_global = (
                                np.asarray(scene["rotations"][panel], dtype=float).T
                                @ (range_m * direction)
                            )
                            candidate["u_x"] = ux
                            candidate["u_y"] = uy
                            candidate["range"] = range_m
                            candidate["tau"] = tau
                            candidate["direction"] = direction
                            candidate["position"] = (
                                np.asarray(scene["ris_centers"][panel], dtype=float)
                                + q_global
                            )
                            candidate["near_field"] = True
                            trial = list(refined)
                            trial[group_index] = candidate
                            _, _, residual, _ = _fit_groups(scene, config, y_vec, trial)
                            objective = float(np.linalg.norm(residual) ** 2)
                            evals += 1
                            if objective < best_objective:
                                best_objective = objective
                                best_group = candidate
            refined[group_index] = best_group
    coeffs, y_hat, residual, _ = _fit_groups(scene, config, y_vec, refined)
    return refined, coeffs, y_hat, residual, evals


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
            "near_field_range_direction_delay_jones_group_omp",
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
        backend_config=backend_cfg,
        static_cache_key=cache_key,
    )
    residual = y_vec - y_hat
    coarse_residual_norm = float(np.linalg.norm(residual))
    backend.synchronize()
    p_hat, delta_t, geom_diag = geometric_support_to_position_ls(scene, selected, config)
    local_refinement = bool(cfg.get("local_refinement", cfg.get("offgrid_refinement", True)))
    refinement_levels = int(cfg.get("refinement_levels", 2))
    refinement_evals = 0
    if local_refinement and selected:
        selected, coeffs, y_hat, residual, refinement_evals = _refine_ris_momp_groups(
            scene,
            config,
            y_vec,
            selected,
            groups,
            levels=refinement_levels,
            shrink=float(cfg.get("refinement_shrink", 0.5)),
        )
        expanded_supports = [
            support for group in selected for support in expand_jones_group(group, 2)
        ]
        p_hat, delta_t, geom_diag = geometric_support_to_position_ls(scene, selected, config)
        diagnostics["offgrid_refinement"] = True
        diagnostics["refinement_objective"] = "direction_delay_group_ls"
    else:
        diagnostics["offgrid_refinement"] = False
        diagnostics["refinement_objective"] = ""
    diagnostics = {
        **diagnostics,
        "dictionary_mode": "near_field_range_aware_group_momp",
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
        "gpu_num_batches": diagnostics.get("gpu_num_batches", 0),
        "gpu_batch_size": batch_size if backend.name == "cupy" else "",
        "gpu_device": backend.device if backend.name == "cupy" else "",
        "backend_warning": backend.warning,
        "max_groups": max_groups,
        "momp_group_omp_enabled": True,
        "momp_score_mode": "factorized_group_projection",
        "momp_group_size": 2,
        "momp_max_groups": max_groups,
        "momp_selected_groups": selected,
        "momp_local_refinement_used": bool(local_refinement and selected),
        "momp_refinement_levels": refinement_levels if local_refinement else 0,
        "momp_refinement_num_evals": refinement_evals,
        "momp_coarse_residual_norm": coarse_residual_norm,
        "momp_refined_residual_norm": float(np.linalg.norm(residual)),
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
            "ranges": np.asarray([support.get("range", np.nan) for support in selected], dtype=float),
        },
        selected_support=selected,
        runtime_s=time.perf_counter() - start,
        diagnostics=diagnostics,
    )
