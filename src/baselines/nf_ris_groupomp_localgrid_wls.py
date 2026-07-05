"""Adapted near-field RIS group-OMP/local-grid/WLS localization baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .common import (
    BaselineResult,
    clock_grid_from_config,
    delay_grid_from_scene,
    expand_jones_group,
    group_omp_select,
    linear_ls_fit,
    raw_atom_from_support,
    simple_atom_normalize,
    supports_from_position_clock,
    supports_to_design,
    vectorize_raw_observation,
)
from ..geometry import local_geometry_from_position
from ..utils import scipy_is_available


def _direction_grid(size: int) -> list[tuple[int, int, float, float, np.ndarray]]:
    axis_count = max(3, int(np.ceil(np.sqrt(max(size, 3)))))
    axis = np.linspace(-0.8, 0.8, axis_count)
    directions = []
    for ux_idx, ux in enumerate(axis):
        for uy_idx, uy in enumerate(axis):
            rem = 1.0 - ux * ux - uy * uy
            if rem <= 0.0:
                continue
            direction = np.array([ux, uy, np.sqrt(rem)], dtype=float)
            directions.append((int(ux_idx), int(uy_idx), float(ux), float(uy), direction))
            if len(directions) >= size:
                return directions
    return directions


def _coarse_groups(scene: dict, config: dict) -> list[dict[str, Any]]:
    baseline_cfg = config.get("baselines", {})
    cfg = dict(baseline_cfg.get("nf_ris_groupomp_localgrid_wls", {}))
    directions = _direction_grid(int(cfg.get("direction_grid_size", 9)))
    taus = delay_grid_from_scene(scene, config, int(cfg.get("delay_grid_size", 9)))
    groups = []
    for panel in range(int(scene["K"])):
        for direction_idx, (ux_idx, uy_idx, ux, uy, direction) in enumerate(directions):
            for tau_idx, tau in enumerate(taus):
                groups.append(
                    {
                        "panel": int(panel),
                        "panel_index": int(panel),
                        "direction": direction,
                        "u_x": float(ux),
                        "u_y": float(uy),
                        "u_x_index": int(ux_idx),
                        "u_y_index": int(uy_idx),
                        "direction_index": int(direction_idx),
                        "tau": float(tau),
                        "tau_index": int(tau_idx),
                        "near_field": False,
                        "group_size": 2,
                        "pol_indices": [0, 1],
                    }
                )
    return groups


def _fit_near_field(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    p_u: np.ndarray,
    delta_t: float,
    *,
    ridge: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    groups = supports_from_position_clock(
        scene,
        np.asarray(p_u, dtype=float),
        float(delta_t),
        model_variant="near_field",
    )
    expanded = [support for group in groups for support in expand_jones_group(group, 2)]
    Phi = supports_to_design(scene, config, expanded)
    coeffs, y_hat, residual = linear_ls_fit(Phi, y_vec, ridge=ridge)
    return coeffs, y_hat, residual, groups


def _model_components(scene: dict, p_u: np.ndarray, delta_t: float) -> dict[str, np.ndarray]:
    ranges = []
    taus = []
    for panel in range(int(scene["K"])):
        range_m, _, _, _ = local_geometry_from_position(
            np.asarray(p_u, dtype=float),
            np.asarray(scene["ris_centers"][panel], dtype=float),
            np.asarray(scene["rotations"][panel], dtype=float),
        )
        ranges.append(float(range_m))
        taus.append(float((range_m + scene["d_RB"][panel]) / scene["c0"] + float(delta_t)))
    return {"ranges": np.asarray(ranges, dtype=float), "taus": np.asarray(taus, dtype=float)}


def _ray_position_clock(
    scene: dict,
    config: dict,
    selected: list[dict[str, Any]],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    bounds_p = np.asarray(
        config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]),
        dtype=float,
    )
    if not selected:
        p = np.mean(bounds_p, axis=1)
        dt = float(np.mean(clock_grid_from_config(config, 3)))
        return p, dt, {"coarse_ray_solver": "fallback_center", "coarse_clipped_to_bounds": False}
    lhs = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    for group in selected:
        panel = int(group.get("panel", 0))
        u_local = np.asarray(group.get("direction", [0.0, 0.0, 1.0]), dtype=float)
        u_local /= np.linalg.norm(u_local) + 1.0e-15
        u = np.asarray(scene["rotations"][panel], dtype=float).T @ u_local
        u /= np.linalg.norm(u) + 1.0e-15
        center = np.asarray(scene["ris_centers"][panel], dtype=float)
        projector = np.eye(3) - np.outer(u, u)
        lhs += projector
        rhs += projector @ center
    cond = float(np.linalg.cond(lhs)) if np.linalg.norm(lhs) > 0.0 else float("inf")
    p = np.linalg.pinv(lhs, rcond=1.0e-10) @ rhs
    clipped = bool(np.any(p < bounds_p[:, 0]) or np.any(p > bounds_p[:, 1]))
    p = np.clip(p, bounds_p[:, 0], bounds_p[:, 1])
    dt_values = []
    for group in selected:
        panel = int(group.get("panel", 0))
        tau = float(group.get("tau", 0.0))
        range_m = float(np.linalg.norm(p - np.asarray(scene["ris_centers"][panel], dtype=float)))
        dt_values.append(tau - (range_m + scene["d_RB"][panel]) / scene["c0"])
    bounds_dt = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    dt = float(np.clip(np.mean(dt_values) if dt_values else np.mean(bounds_dt), bounds_dt[0], bounds_dt[1]))
    return p.astype(float), dt, {
        "coarse_ray_solver": "pinv_ray_intersection",
        "coarse_ray_condition_number": cond,
        "coarse_clipped_to_bounds": clipped,
    }


def _objective(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    p_u: np.ndarray,
    delta_t: float,
    selected: list[dict[str, Any]],
    l1_tau_lambda: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    coeffs, y_hat, residual, groups = _fit_near_field(scene, config, y_vec, p_u, delta_t)
    value = float(np.linalg.norm(residual) ** 2)
    if float(l1_tau_lambda) > 0.0 and selected:
        penalty = 0.0
        for support in selected:
            panel = int(support.get("panel", 0))
            tau_hat = float(support.get("tau", 0.0))
            range_m = float(np.linalg.norm(np.asarray(p_u, dtype=float) - scene["ris_centers"][panel]))
            tau_model = (range_m + scene["d_RB"][panel]) / scene["c0"] + float(delta_t)
            penalty += abs(tau_hat - tau_model)
        value += float(l1_tau_lambda) * penalty
    return value, coeffs, y_hat, residual, groups


def _local_search(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    p0: np.ndarray,
    dt0: float,
    selected: list[dict[str, Any]],
    *,
    levels: int,
    position_radius_m: float,
    clock_radius_s: float,
    shrink: float,
    l1_tau_lambda: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], int, float]:
    bounds_p = np.asarray(
        config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]),
        dtype=float,
    )
    bounds_dt = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    p_best = np.clip(np.asarray(p0, dtype=float), bounds_p[:, 0], bounds_p[:, 1])
    dt_best = float(np.clip(float(dt0), bounds_dt[0], bounds_dt[1]))
    best_value, coeffs_best, y_hat_best, residual_best, groups_best = _objective(
        scene, config, y_vec, p_best, dt_best, selected, l1_tau_lambda
    )
    evals = 1
    for level in range(max(0, int(levels))):
        scale = float(shrink) ** level
        offsets = [-position_radius_m * scale, 0.0, position_radius_m * scale]
        dt_offsets = [-clock_radius_s * scale, 0.0, clock_radius_s * scale]
        center_p = p_best.copy()
        center_dt = dt_best
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    p = np.clip(center_p + np.array([dx, dy, dz]), bounds_p[:, 0], bounds_p[:, 1])
                    for dt_offset in dt_offsets:
                        dt = float(np.clip(center_dt + dt_offset, bounds_dt[0], bounds_dt[1]))
                        value, coeffs, y_hat, residual, groups = _objective(
                            scene, config, y_vec, p, dt, selected, l1_tau_lambda
                        )
                        evals += 1
                        if value < best_value:
                            best_value = value
                            p_best = p
                            dt_best = dt
                            coeffs_best = coeffs
                            y_hat_best = y_hat
                            residual_best = residual
                            groups_best = groups
    return p_best, dt_best, coeffs_best, y_hat_best, residual_best, groups_best, evals, best_value


def _wls_position_clock(
    scene: dict,
    config: dict,
    initial_p: np.ndarray,
    initial_dt: float,
    selected: list[dict[str, Any]],
    *,
    lambda_ray: float,
    lambda_tau: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    bounds_p = np.asarray(
        config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]),
        dtype=float,
    )
    bounds_dt = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    lower = np.r_[bounds_p[:, 0], bounds_dt[0]]
    upper = np.r_[bounds_p[:, 1], bounds_dt[1]]
    x0 = np.clip(np.r_[np.asarray(initial_p, dtype=float).reshape(3), float(initial_dt)], lower, upper)

    def residual_fn(x: np.ndarray) -> np.ndarray:
        p = x[:3]
        dt = float(x[3])
        residuals: list[float] = []
        for support in selected:
            panel = int(support.get("panel", 0))
            center = np.asarray(scene["ris_centers"][panel], dtype=float)
            direction = np.asarray(support.get("direction", [0.0, 0.0, 1.0]), dtype=float)
            direction = np.asarray(scene["rotations"][panel], dtype=float).T @ direction
            direction /= np.linalg.norm(direction) + 1.0e-15
            projector = np.eye(3) - np.outer(direction, direction)
            ray_residual = projector @ (p - center)
            residuals.extend((np.sqrt(float(lambda_ray)) * ray_residual).tolist())
            tau = float(support.get("tau", 0.0))
            tau_model = (np.linalg.norm(p - center) + scene["d_RB"][panel]) / scene["c0"] + dt
            residuals.append(np.sqrt(float(lambda_tau)) * (tau - tau_model) / 1.0e-9)
        if not residuals:
            residuals.extend((p - x0[:3]).tolist())
        return np.asarray(residuals, dtype=float)

    if scipy_is_available():
        try:
            from scipy.optimize import least_squares  # type: ignore[import-not-found]

            result = least_squares(
                residual_fn,
                x0,
                bounds=(lower, upper),
                max_nfev=int(config.get("baselines", {}).get("geometry_max_nfev", 100)),
                x_scale=np.maximum(upper - lower, 1.0e-12),
            )
            return result.x[:3].astype(float), float(result.x[3]), {
                "wls_solver": "scipy.optimize.least_squares",
                "wls_final_cost": float(result.cost),
                "wls_num_evals": int(result.nfev),
                "wls_success": bool(result.success),
            }
        except Exception as exc:  # noqa: BLE001 - fall back to local grid below.
            warning = f"wls_failed: {type(exc).__name__}: {exc}"
    else:
        warning = "wls_no_scipy_local_grid"

    best = x0.copy()
    best_cost = float(np.linalg.norm(residual_fn(best)) ** 2)
    evals = 1
    for radius in (0.25, 0.1, 0.04):
        for dx in (-radius, 0.0, radius):
            for dy in (-radius, 0.0, radius):
                for dz in (-radius, 0.0, radius):
                    for dt_ns in (-1.0, 0.0, 1.0):
                        x = np.clip(best + np.array([dx, dy, dz, dt_ns * 1.0e-9]), lower, upper)
                        cost = float(np.linalg.norm(residual_fn(x)) ** 2)
                        evals += 1
                        if cost < best_cost:
                            best = x
                            best_cost = cost
    return best[:3].astype(float), float(best[3]), {
        "wls_solver": "local_grid",
        "wls_final_cost": best_cost,
        "wls_num_evals": evals,
        "wls_success": True,
        "warning": warning,
    }


def run_nf_ris_groupomp_localgrid_wls_baseline(data: dict, config: dict) -> BaselineResult:
    start = time.perf_counter()
    scene = data["scene"]
    y_vec = vectorize_raw_observation(data["Y_noisy"])
    baseline_cfg = config.get("baselines", {})
    cfg = dict(baseline_cfg.get("nf_ris_groupomp_localgrid_wls", {}))
    groups = _coarse_groups(scene, config)
    max_groups = int(cfg.get("max_groups", scene["K"]))
    selected, expanded_supports, coeffs_coarse, y_hat_coarse, omp_diag = group_omp_select(
        scene,
        config,
        groups,
        y_vec,
        max_groups=max_groups,
        batch_size=int(cfg.get("batch_size", 64)),
        trim_memory_enabled=bool(config.get("baselines", {}).get("trim_memory", True)),
    )
    coarse_p, coarse_dt, ray_diag = _ray_position_clock(scene, config, selected)
    bounds_p = np.asarray(config.get("ue_bounds"), dtype=float)
    span = bounds_p[:, 1] - bounds_p[:, 0]
    auto_radius = float(max(np.min(span) / 4.0, 0.05))
    clock_radius_s = float(cfg.get("local_grid_clock_radius_ns", 2.0)) * 1.0e-9
    l1_tau_lambda = float(cfg.get("l1_tau_lambda", 0.0))
    nf_levels = int(cfg.get("coarse_to_nf_refinement_levels", 2))
    nf_p, nf_dt, nf_coeffs, nf_y_hat, nf_residual, nf_groups, nf_evals, nf_objective = _local_search(
        scene,
        config,
        y_vec,
        coarse_p,
        coarse_dt,
        selected,
        levels=nf_levels if bool(cfg.get("local_refinement", True)) else 0,
        position_radius_m=auto_radius,
        clock_radius_s=clock_radius_s,
        shrink=float(cfg.get("refinement_shrink", 0.5)),
        l1_tau_lambda=l1_tau_lambda,
    )

    local_grid_enabled = bool(cfg.get("local_grid_enabled", True))
    local_grid_iterations = int(cfg.get("local_grid_iterations", 5)) if local_grid_enabled else 0
    local_grid_levels = int(cfg.get("local_grid_refinement_levels", 2))
    local_grid_radius_value = cfg.get("local_grid_position_radius_m", "auto")
    local_grid_radius = auto_radius if str(local_grid_radius_value) == "auto" else float(local_grid_radius_value)
    local_grid_p = nf_p
    local_grid_dt = nf_dt
    local_grid_coeffs = nf_coeffs
    local_grid_y_hat = nf_y_hat
    local_grid_residual = nf_residual
    local_grid_groups = nf_groups
    local_grid_initial_objective = float(np.linalg.norm(nf_residual) ** 2)
    local_grid_final_objective = local_grid_initial_objective
    local_grid_evals = 0
    for iteration in range(local_grid_iterations):
        (
            local_grid_p,
            local_grid_dt,
            local_grid_coeffs,
            local_grid_y_hat,
            local_grid_residual,
            local_grid_groups,
            evals,
            local_grid_final_objective,
        ) = _local_search(
            scene,
            config,
            y_vec,
            local_grid_p,
            local_grid_dt,
            selected,
            levels=local_grid_levels,
            position_radius_m=local_grid_radius * (float(cfg.get("local_grid_shrink", 0.5)) ** iteration),
            clock_radius_s=clock_radius_s * (float(cfg.get("local_grid_shrink", 0.5)) ** iteration),
            shrink=float(cfg.get("local_grid_shrink", 0.5)),
            l1_tau_lambda=0.0,
        )
        local_grid_evals += evals

    wls_enabled = bool(cfg.get("wls_enabled", True))
    if wls_enabled:
        final_p, final_dt, wls_diag = _wls_position_clock(
            scene,
            config,
            local_grid_p,
            local_grid_dt,
            selected,
            lambda_ray=float(cfg.get("wls_lambda_ray", 1.0)),
            lambda_tau=float(cfg.get("wls_lambda_tau", 1.0)),
        )
    else:
        final_p, final_dt, wls_diag = local_grid_p, local_grid_dt, {
            "wls_solver": "disabled",
            "wls_final_cost": float("nan"),
            "wls_num_evals": 0,
            "wls_success": False,
        }
    clipped = bool(np.any(final_p < bounds_p[:, 0]) or np.any(final_p > bounds_p[:, 1]))
    final_p = np.clip(final_p, bounds_p[:, 0], bounds_p[:, 1])
    final_coeffs, final_y_hat, final_residual, final_groups = _fit_near_field(
        scene, config, y_vec, final_p, final_dt
    )
    warning_parts = [wls_diag.get("warning", "")]
    if bool(cfg.get("use_subris", True)):
        warning_parts.append("subris_fallback_to_physical_panel_centers")
    warning = "; ".join(part for part in warning_parts if part)
    diagnostics = {
        **omp_diag,
        **ray_diag,
        **wls_diag,
        "dictionary_mode": "nf_ris_groupomp_localgrid_wls_adapted",
        "model_variant": "near_field_ris_groupomp_localgrid_wls",
        "reference_algorithm": "GroupOMP_LocalGrid_WLS",
        "adaptation_note": "adapted_baseline_no_cpd_or_sage_claim",
        "cpd_omp_adapted_used": False,
        "coarse_direction_delay_grid_size": len(groups),
        "coarse_selected_groups": selected,
        "coarse_position": coarse_p,
        "coarse_delta_t": coarse_dt,
        "near_field_l1_refinement_used": bool(cfg.get("local_refinement", True)),
        "l1_tau_lambda": l1_tau_lambda,
        "near_field_l1_num_evals": nf_evals,
        "near_field_l1_objective": nf_objective,
        "sage_enabled": False,
        "sage_iterations": 0,
        "sage_num_evals": 0,
        "sage_initial_objective": float("nan"),
        "sage_final_objective": float("nan"),
        "local_grid_enabled": local_grid_enabled,
        "local_grid_iterations": local_grid_iterations,
        "local_grid_num_evals": local_grid_evals,
        "local_grid_initial_objective": local_grid_initial_objective,
        "local_grid_final_objective": local_grid_final_objective,
        "wls_enabled": wls_enabled,
        "subris_mode": "physical_panel_centers",
        "subris_shape": tuple(cfg.get("subris_shape", (2, 2))),
        "subris_fallback_used": bool(cfg.get("use_subris", True)),
        "final_clipped_to_bounds": clipped,
        "grid_size": len(groups),
        "support_size": len(selected),
        "selected_support": selected,
        "expanded_supports": expanded_supports,
        "coeff_norm": float(np.linalg.norm(final_coeffs)),
        "raw_coarse_objective": float(np.linalg.norm(y_vec - y_hat_coarse) ** 2),
        "raw_objective_final": float(np.linalg.norm(final_residual) ** 2 / y_vec.size),
        "backend": "cpu",
        "gpu_used": False,
        "backend_warning": "cpu_local_refinement_used",
        "warning": warning,
    }
    return BaselineResult(
        name="nf_ris_groupomp_localgrid_wls",
        p_u=final_p,
        delta_t=final_dt,
        Y_hat=final_y_hat.reshape(scene["I"], scene["N"], scene["T"]),
        raw_objective_final=float(np.linalg.norm(final_residual) ** 2 / y_vec.size),
        components=_model_components(scene, final_p, final_dt),
        selected_support=[*selected, *final_groups],
        runtime_s=time.perf_counter() - start,
        diagnostics=diagnostics,
    )
