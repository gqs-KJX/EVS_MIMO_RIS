"""RIS-MOMP adaptation with independent angular/delay coordinate searches.

The cited RIS-MOMP method does not form one Cartesian angle-delay dictionary.
It initializes a tuple of dictionary coordinates, refines one coordinate at a
time while holding the others fixed, competes the available source models,
and performs an orthogonal LS residual update.  In this repository every RIS
panel is one source model and its two Jones columns replace the paper's
equivalent complex-gain dimension.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .backend import BackendConfig, get_backend
from .cache import BASELINE_CACHE, baseline_cache_key, cache_diagnostics_delta
from .common import (
    BaselineResult,
    delay_grid_from_scene,
    expand_jones_group,
    geometric_support_to_position_ls,
    vectorize_raw_observation,
)
from .factorized_scoring import (
    FactorizedGroupScorer,
    build_group_factor_context,
    factorized_fit_supports,
)


def _direction_axis(size: int) -> np.ndarray:
    """Return one independent UPA direction-cosine dictionary."""
    return np.linspace(-0.95, 0.95, max(3, int(size)), dtype=float)


def _hemisphere_sign(scene: dict, config: dict, panel: int) -> float:
    """Resolve the RIS-normal ambiguity from the configured search region.

    A planar far-field dictionary only observes the two in-plane direction
    cosines.  The cited paper fixes the remaining cosine to one hemisphere.
    Here that hemisphere is chosen from the center of the declared UE bounds,
    without using the true UE position.
    """
    bounds = np.asarray(
        config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]),
        dtype=float,
    )
    center = np.mean(bounds, axis=1)
    q_local = np.asarray(scene["rotations"][panel], dtype=float) @ (
        center - np.asarray(scene["ris_centers"][panel], dtype=float)
    )
    return float(np.sign(q_local[2]) or 1.0)


def _direction_from_coordinates(ux: float, uy: float, sign: float) -> np.ndarray:
    """Map the two MOMP angular coordinates to a unit direction."""
    ux = float(np.clip(ux, -0.999, 0.999))
    uy = float(np.clip(uy, -0.999, 0.999))
    radius_sq = ux * ux + uy * uy
    if radius_sq >= 0.999**2:
        scale = 0.999 / np.sqrt(radius_sq + 1.0e-15)
        ux *= scale
        uy *= scale
        radius_sq = ux * ux + uy * uy
    uz = float(np.sign(sign) or 1.0) * np.sqrt(max(1.0 - radius_sq, 1.0e-12))
    return np.array([ux, uy, uz], dtype=float)


def _support(
    scene: dict,
    config: dict,
    panel: int,
    ux: float,
    uy: float,
    tau: float,
    *,
    ux_index: int = -1,
    uy_index: int = -1,
    tau_index: int = -1,
) -> dict[str, Any]:
    direction = _direction_from_coordinates(
        ux, uy, _hemisphere_sign(scene, config, int(panel))
    )
    return {
        "panel": int(panel),
        "panel_index": int(panel),
        "direction": direction,
        "u_x": float(direction[0]),
        "u_y": float(direction[1]),
        "u_z_sign": float(np.sign(direction[2]) or 1.0),
        "u_x_index": int(ux_index),
        "u_y_index": int(uy_index),
        "direction_index": (int(ux_index), int(uy_index)),
        "tau": float(tau),
        "tau_index": int(tau_index),
        "near_field": False,
        "group_size": 2,
        "pol_indices": [0, 1],
    }


def _score_groups(
    scene: dict,
    config: dict,
    groups: list[dict[str, Any]],
    residual: np.ndarray,
    backend_config: BackendConfig,
) -> tuple[int, float, str, bool]:
    if not groups:
        return -1, float("-inf"), "cpu", False
    scorer = FactorizedGroupScorer(
        build_group_factor_context(scene, config, groups), backend_config
    )
    best_index, best_score = scorer.best(residual)
    scorer.backend.synchronize()
    return best_index, best_score, scorer.backend.name, scorer.backend.name == "cupy"


def _coordinate_candidates(
    scene: dict,
    config: dict,
    panel: int,
    current: dict[str, Any],
    coordinate: str,
    values: np.ndarray,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for index, value in enumerate(np.asarray(values, dtype=float)):
        ux = float(value) if coordinate == "u_x" else float(current["u_x"])
        uy = float(value) if coordinate == "u_y" else float(current["u_y"])
        tau = float(value) if coordinate == "tau" else float(current["tau"])
        groups.append(
            _support(
                scene,
                config,
                panel,
                ux,
                uy,
                tau,
                ux_index=index if coordinate == "u_x" else int(current.get("u_x_index", -1)),
                uy_index=index if coordinate == "u_y" else int(current.get("u_y_index", -1)),
                tau_index=index if coordinate == "tau" else int(current.get("tau_index", -1)),
            )
        )
    return groups


def _coordinate_search_source(
    scene: dict,
    config: dict,
    panel: int,
    residual: np.ndarray,
    ux_grid: np.ndarray,
    uy_grid: np.ndarray,
    tau_grid: np.ndarray,
    backend_config: BackendConfig,
    *,
    sweeps: int,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Run MOMP coordinate initialization/refinement for one source panel."""
    bounds = np.asarray(config.get("ue_bounds"), dtype=float)
    prior_position = np.mean(bounds, axis=1)
    q_local = np.asarray(scene["rotations"][panel], dtype=float) @ (
        prior_position - np.asarray(scene["ris_centers"][panel], dtype=float)
    )
    prior_range = float(np.linalg.norm(q_local)) + 1.0e-15
    prior_direction = q_local / prior_range
    prior_tau = (
        prior_range + float(scene["d_RB"][panel])
    ) / float(scene["c0"]) + float(np.mean(config.get("delta_t_bounds", [0.0, 10.0e-9])))
    prior_indexes = (
        int(np.argmin(np.abs(ux_grid - prior_direction[0]))),
        int(np.argmin(np.abs(uy_grid - prior_direction[1]))),
        int(np.argmin(np.abs(tau_grid - prior_tau))),
    )
    broadside_indexes = (
        int(np.argmin(np.abs(ux_grid))),
        int(np.argmin(np.abs(uy_grid))),
        int(np.argmin(np.abs(tau_grid - prior_tau))),
    )
    initialization_indexes = list(dict.fromkeys((prior_indexes, broadside_indexes)))
    evaluations = 0
    backend_names: set[str] = set()
    gpu_used = False
    best_score = float("-inf")
    best_support: dict[str, Any] | None = None

    # The cited relaxed MOMP initialization is not reproduced in the SPAWC
    # paper.  Use two deterministic, truth-free starts (geometry-prior and
    # broadside), then apply the published one-coordinate-at-a-time updates.
    for ux_index, uy_index, tau_index in initialization_indexes:
        current = _support(
            scene,
            config,
            panel,
            float(ux_grid[ux_index]),
            float(uy_grid[uy_index]),
            float(tau_grid[tau_index]),
            ux_index=ux_index,
            uy_index=uy_index,
            tau_index=tau_index,
        )
        current_score = float("-inf")
        for _ in range(max(1, int(sweeps))):
            changed = False
            for coordinate, values in (
                ("u_x", ux_grid),
                ("u_y", uy_grid),
                ("tau", tau_grid),
            ):
                candidates = _coordinate_candidates(
                    scene, config, panel, current, coordinate, values
                )
                index, score, backend_name, coordinate_gpu = _score_groups(
                    scene, config, candidates, residual, backend_config
                )
                evaluations += len(candidates)
                backend_names.add(backend_name)
                gpu_used |= coordinate_gpu
                if index >= 0:
                    candidate = candidates[index]
                    changed |= any(
                        candidate.get(key) != current.get(key)
                        for key in ("u_x_index", "u_y_index", "tau_index")
                    )
                    current = candidate
                    current_score = float(score)
            if not changed:
                break
        if current_score > best_score:
            best_score = current_score
            best_support = current

    if best_support is None:
        best_support = _support(
            scene, config, panel, 0.0, 0.0, float(tau_grid[len(tau_grid) // 2])
        )
    return best_support, best_score, {
        "coordinate_evaluations": evaluations,
        "coordinate_initializations": len(initialization_indexes),
        "coordinate_backends": sorted(backend_names),
        "coordinate_gpu_used": gpu_used,
    }


def _refine_selected_coordinates(
    scene: dict,
    config: dict,
    residual: np.ndarray,
    support: dict[str, Any],
    backend_config: BackendConfig,
    ux_step: float,
    uy_step: float,
    tau_step: float,
    levels: int,
    shrink: float,
) -> tuple[dict[str, Any], int, bool]:
    """Optional multiresolution MOMP sweeps; no range coordinate is added."""
    current = dict(support)
    evaluations = 0
    gpu_used = False
    for level in range(max(0, int(levels))):
        scale = float(shrink) ** (level + 1)
        grids = (
            ("u_x", float(current["u_x"]) + scale * ux_step * np.arange(-2, 3)),
            ("u_y", float(current["u_y"]) + scale * uy_step * np.arange(-2, 3)),
            ("tau", float(current["tau"]) + scale * tau_step * np.arange(-2, 3)),
        )
        for coordinate, values in grids:
            candidates = _coordinate_candidates(
                scene,
                config,
                int(current["panel"]),
                current,
                coordinate,
                np.asarray(values, dtype=float),
            )
            index, _, _, coordinate_gpu = _score_groups(
                scene, config, candidates, residual, backend_config
            )
            evaluations += len(candidates)
            gpu_used |= coordinate_gpu
            if index >= 0:
                current = candidates[index]
    current["offgrid_refined"] = bool(levels > 0)
    return current, evaluations, gpu_used


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

    direction_size = int(cfg.get("direction_grid_size", 31))
    ux_size = int(cfg.get("ux_grid_size", direction_size))
    uy_size = int(cfg.get("uy_grid_size", direction_size))
    tau_size = int(cfg.get("delay_grid_size", 41))
    cache_key = baseline_cache_key(
        "ris_momp_coordinate_grids",
        scene,
        config,
        grid_sizes=(ux_size, uy_size, tau_size, "far_field_ux_uy_delay"),
    )
    ux_grid, uy_grid, tau_grid = BASELINE_CACHE.get_or_create(
        cache_key,
        lambda: (
            _direction_axis(ux_size),
            _direction_axis(uy_size),
            delay_grid_from_scene(scene, config, tau_size),
        ),
    )
    ux_step = float(np.min(np.diff(ux_grid))) if ux_grid.size > 1 else 0.1
    uy_step = float(np.min(np.diff(uy_grid))) if uy_grid.size > 1 else 0.1
    tau_step = float(np.min(np.diff(tau_grid))) if tau_grid.size > 1 else 1.0e-9

    selected: list[dict[str, Any]] = []
    expanded_supports: list[dict[str, Any]] = []
    coeffs = np.zeros(0, dtype=complex)
    y_hat = np.zeros_like(y_vec)
    residual = y_vec.copy()
    max_groups = int(cfg.get("max_groups", cfg.get("max_atoms", scene["K"])))
    coordinate_sweeps = int(cfg.get("coordinate_sweeps", 2))
    total_coordinate_evals = 0
    source_competitions = 0
    gpu_used = False
    source_trace: list[dict[str, Any]] = []

    for iteration in range(max(0, max_groups)):
        candidates: list[tuple[dict[str, Any], float, dict[str, Any]]] = []
        selected_panels = {int(item["panel"]) for item in selected}
        for panel in range(int(scene["K"])):
            if panel in selected_panels:
                continue
            support, score, diagnostics = _coordinate_search_source(
                scene,
                config,
                panel,
                residual,
                ux_grid,
                uy_grid,
                tau_grid,
                backend_cfg,
                sweeps=coordinate_sweeps,
            )
            candidates.append((support, score, diagnostics))
            total_coordinate_evals += int(diagnostics["coordinate_evaluations"])
            gpu_used |= bool(diagnostics["coordinate_gpu_used"])
        source_competitions += len(candidates)
        if not candidates:
            break
        best_support, best_score, best_diag = max(candidates, key=lambda item: item[1])
        if not np.isfinite(best_score):
            break
        source_trace.append(
            {
                "iteration": int(iteration),
                "selected_panel": int(best_support["panel"]),
                "score": float(best_score),
                "candidate_source_count": len(candidates),
                **best_diag,
            }
        )

        local_refinement = bool(
            cfg.get("local_refinement", cfg.get("offgrid_refinement", False))
        )
        refinement_levels = int(cfg.get("refinement_levels", 0))
        if local_refinement and refinement_levels > 0:
            best_support, evals, refinement_gpu = _refine_selected_coordinates(
                scene,
                config,
                residual,
                best_support,
                backend_cfg,
                ux_step,
                uy_step,
                tau_step,
                refinement_levels,
                float(cfg.get("refinement_shrink", 0.5)),
            )
            total_coordinate_evals += evals
            gpu_used |= refinement_gpu
        selected.append(best_support)
        expanded_supports = [
            item for group in selected for item in expand_jones_group(group, 2)
        ]
        coeffs, y_hat, residual = factorized_fit_supports(
            scene, config, expanded_supports, y_vec, ridge=1.0e-10
        )

    backend.synchronize()
    p_hat, delta_t, geometry_diag = geometric_support_to_position_ls(
        scene, selected, config
    )
    local_refinement_used = any(
        bool(item.get("offgrid_refined", False)) for item in selected
    )
    diagnostics = {
        "dictionary_mode": "ris_momp_independent_ux_uy_delay",
        "model_variant": "far_field_ris_momp_adaptation",
        "reference_algorithm": "RIS-MOMP adaptation",
        "adaptation_note": (
            "multi_ris_panel_sources_and_two_jones_gains_replace_"
            "paper_bs_direct_ris_sources_and_equivalent_aoa_gain"
        ),
        "group_omp": False,
        "offgrid_refinement": local_refinement_used,
        "refinement_objective": (
            "multiresolution_coordinate_projection" if local_refinement_used else ""
        ),
        "grid_size": int(ux_grid.size + uy_grid.size + tau_grid.size),
        "cartesian_grid_size": int(ux_grid.size * uy_grid.size * tau_grid.size),
        "cartesian_dictionary_materialized": False,
        "range_dictionary_used": False,
        "direction_grid_size": direction_size,
        "ux_grid_size": int(ux_grid.size),
        "uy_grid_size": int(uy_grid.size),
        "delay_grid_size": int(tau_grid.size),
        "support_size": len(selected),
        "selected_support": selected,
        "expanded_supports": expanded_supports,
        "expanded_support_count": len(expanded_supports),
        "selected_group_count": len(selected),
        "selected_panel_count": len({int(item["panel"]) for item in selected}),
        "selected_panels": [int(item["panel"]) for item in selected],
        "unique_panel_constraint": True,
        "coeff_norm": float(np.linalg.norm(coeffs)),
        "residual_norm": float(np.linalg.norm(residual)),
        "backend": backend.name,
        "gpu_used": bool(gpu_used or backend.name == "cupy"),
        "gpu_device": backend.device if backend.name == "cupy" else "",
        "backend_warning": backend.warning,
        "momp_group_omp_enabled": False,
        "momp_score_mode": "source_competing_coordinate_descent_projection",
        "momp_group_size": 2,
        "momp_max_groups": max_groups,
        "momp_selected_groups": selected,
        "momp_coordinate_sweeps": coordinate_sweeps,
        "momp_coordinate_evaluations": total_coordinate_evals,
        "momp_source_competitions": source_competitions,
        "momp_source_trace": source_trace,
        "momp_local_refinement_used": local_refinement_used,
        "momp_refinement_levels": int(cfg.get("refinement_levels", 0))
        if local_refinement_used
        else 0,
        "momp_refinement_num_evals": total_coordinate_evals,
        "momp_coarse_residual_norm": float(np.linalg.norm(y_vec)),
        "momp_refined_residual_norm": float(np.linalg.norm(residual)),
        **geometry_diag,
    }
    diagnostics.update(cache_diagnostics_delta(cache_before, BASELINE_CACHE.snapshot()))
    raw_objective = float(np.linalg.norm(residual) ** 2 / max(y_vec.size, 1))
    return BaselineResult(
        name="ris_momp",
        p_u=p_hat,
        delta_t=delta_t,
        Y_hat=y_hat.reshape(scene["I"], scene["N"], scene["T"]),
        raw_objective_final=raw_objective,
        components={
            "taus": np.asarray([item.get("tau", np.nan) for item in selected], dtype=float),
        },
        selected_support=selected,
        runtime_s=time.perf_counter() - start,
        diagnostics=diagnostics,
    )
