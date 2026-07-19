"""NF-RIS CPD-OMP-SAGE-WLS adaptation for the repository raw tensor.

The original paper uses a SISO tensor created by a Kronecker RIS profile.  The
repository instead observes an EVS x subcarrier x training tensor and uses
arbitrary per-panel phase profiles.  This adaptation preserves the published
algorithmic stages while using the exact repository operator ``c_k=Omega_k g_k``:

1. sequential rank-one CPD and orthogonal residual updates;
2. one-dimensional delay matching, far-field direction initialization, and a
   one-dimensional exact near-field range search;
3. cyclic SAGE complete-data updates with per-path raw-domain refinement;
4. FIM-weighted fusion of path parameters into common position and clock.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .als_cpd import complex_cp_als
from .common import (
    BaselineResult,
    build_jones_basis_evs_atoms,
    delay_grid_from_scene,
    delay_response,
    expand_jones_group,
    group_design,
    supports_from_position_clock,
    training_response_from_position,
    vectorize_raw_observation,
)
from .factorized_scoring import factorized_fit_supports
from ..geometry import local_geometry_from_position, near_field_spherical_response
from ..utils import scipy_is_available


def _normalized(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if np.isfinite(norm) and norm > 0.0 else array


def _direction(ux: float, uy: float, sign: float) -> np.ndarray:
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


def _range_grid(scene: dict, config: dict, size: int) -> np.ndarray:
    bounds = np.asarray(config.get("ue_bounds"), dtype=float)
    corners = np.array(
        [[x, y, z] for x in bounds[0] for y in bounds[1] for z in bounds[2]],
        dtype=float,
    )
    values: list[float] = []
    for panel in range(int(scene["K"])):
        values.extend(
            np.linalg.norm(
                corners - np.asarray(scene["ris_centers"][panel], dtype=float),
                axis=1,
            ).tolist()
        )
    return np.linspace(min(values), max(values), max(3, int(size)), dtype=float)


def _position_on_ray(
    scene: dict, panel: int, direction_local: np.ndarray, range_m: float
) -> np.ndarray:
    return np.asarray(scene["ris_centers"][panel], dtype=float) + (
        np.asarray(scene["rotations"][panel], dtype=float).T
        @ (float(range_m) * np.asarray(direction_local, dtype=float))
    )


def _nf_position_grid(config: dict, cfg: dict) -> np.ndarray:
    """Coarse near-field UE position grid inside the declared UE bounds.

    The original paper seeds the range/angle search from a far-field CPD
    estimate.  In the repository's deep near-field regime (range << Fraunhofer
    distance) the far-field steering dictionary is nearly orthogonal to the true
    spherical-wave response, so the far-field seed is uninformative.  Following
    the paper's own near-field distance dictionary (its (38)-(41) step), the
    coarse spatial estimate is instead obtained directly on a near-field grid.
    """
    bounds = np.asarray(config.get("ue_bounds"), dtype=float)
    nx = int(cfg.get("nf_grid_x", 11))
    ny = int(cfg.get("nf_grid_y", 11))
    nz = int(cfg.get("nf_grid_z", 11))
    gx = np.linspace(bounds[0, 0], bounds[0, 1], max(2, nx))
    gy = np.linspace(bounds[1, 0], bounds[1, 1], max(2, ny))
    gz = np.linspace(bounds[2, 0], bounds[2, 1], max(2, nz))
    return np.array(
        [[x, y, z] for x in gx for y in gy for z in gz], dtype=float
    )


def _nf_training_matrix(
    scene: dict, panel: int, positions: np.ndarray
) -> np.ndarray:
    """Column-normalized near-field training responses for a position grid.

    The known RIS-BS response and the RIS phase profile are applied once as a
    batched matrix product to keep the coarse near-field grid affordable.
    """
    center = np.asarray(scene["ris_centers"][panel], dtype=float)
    rotation = np.asarray(scene["rotations"][panel], dtype=float)
    ris_grid = np.asarray(scene["ris_grid"], dtype=float)
    wavelength = float(scene["wavelength"])
    a_rb = np.asarray(scene["a_RB"][panel], dtype=complex)
    omega = np.asarray(scene["Omega"][panel], dtype=complex)
    g_columns = np.empty((ris_grid.shape[0], len(positions)), dtype=complex)
    for index, pos in enumerate(positions):
        range_m, elevation, azimuth, _ = local_geometry_from_position(
            np.asarray(pos, dtype=float), center, rotation
        )
        a_ur = near_field_spherical_response(
            range_m, elevation, azimuth, ris_grid, wavelength
        )
        g_columns[:, index] = a_rb * a_ur
    responses = omega @ g_columns
    norms = np.linalg.norm(responses, axis=0)
    norms[norms == 0.0] = 1.0
    return responses / norms[None, :]


def _support_from_position(
    scene: dict, panel: int, position: np.ndarray, tau: float
) -> dict[str, Any]:
    """Build a near-field support from an absolute UE position and delay."""
    center = np.asarray(scene["ris_centers"][panel], dtype=float)
    rotation = np.asarray(scene["rotations"][panel], dtype=float)
    q_local = rotation @ (np.asarray(position, dtype=float) - center)
    range_m = float(np.linalg.norm(q_local)) + 1.0e-15
    direction_local = q_local / range_m
    return _near_field_support(
        scene,
        int(panel),
        float(direction_local[0]),
        float(direction_local[1]),
        range_m,
        float(tau),
        sign=float(np.sign(direction_local[2]) or 1.0),
    )


def _near_field_support(
    scene: dict,
    panel: int,
    ux: float,
    uy: float,
    range_m: float,
    tau: float,
    *,
    sign: float,
) -> dict[str, Any]:
    direction_local = _direction(ux, uy, sign)
    position = _position_on_ray(scene, panel, direction_local, range_m)
    elevation = float(np.arcsin(np.clip(direction_local[2], -1.0, 1.0)))
    azimuth = float(np.arctan2(direction_local[1], direction_local[0]))
    return {
        "panel": int(panel),
        "panel_index": int(panel),
        "position": position,
        "range": float(range_m),
        "direction": direction_local,
        "u_x": float(direction_local[0]),
        "u_y": float(direction_local[1]),
        "u_z_sign": float(np.sign(direction_local[2]) or 1.0),
        "elevation": elevation,
        "azimuth": azimuth,
        "tau": float(tau),
        "near_field": True,
        "group_size": 2,
        "pol_indices": [0, 1],
    }


def _match_delay_factor(
    scene: dict, delay_factor: np.ndarray, tau_grid: np.ndarray
) -> tuple[float, int, float]:
    target = _normalized(delay_factor)
    scores = np.asarray(
        [
            abs(np.vdot(_normalized(delay_response(scene, float(tau))), target)) ** 2
            for tau in tau_grid
        ],
        dtype=float,
    )
    index = int(np.argmax(scores))
    return float(tau_grid[index]), index, float(scores[index])


def _panel_evs_score(
    scene: dict, config: dict, panel: int, evs_factor: np.ndarray
) -> float:
    basis, _ = build_jones_basis_evs_atoms(scene, config, panel_index=panel)
    matrix = np.column_stack([_normalized(column) for column in basis[:2]])
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return float(np.linalg.norm(q.conj().T @ _normalized(evs_factor)) ** 2)


def _match_cpd_component(
    scene: dict,
    config: dict,
    evs_factor: np.ndarray,
    delay_factor: np.ndarray,
    training_factor: np.ndarray,
    available_panels: list[int],
    positions: np.ndarray,
    nf_matrices: dict[int, np.ndarray],
    taus: np.ndarray,
) -> dict[str, Any] | None:
    """Assign a CPD component to a panel and a near-field UE position.

    The EVS-mode factor selects the physical panel through its Maxwell-Jones
    subspace, the delay-mode factor selects the ToA on the delay grid, and the
    training-mode factor is matched against near-field spherical responses over
    a coarse UE position grid (the paper's near-field distance step, evaluated
    jointly over angle and range because the far-field seed is invalid here).
    """
    if not available_panels:
        return None
    tau, tau_index, delay_score = _match_delay_factor(scene, delay_factor, taus)
    target_training = _normalized(training_factor)
    best: tuple[float, int, int, float] | None = None
    for panel in available_panels:
        evs_score = _panel_evs_score(scene, config, panel, evs_factor)
        correlations = np.abs(nf_matrices[panel].conj().T @ target_training) ** 2
        position_index = int(np.argmax(correlations))
        training_score = float(correlations[position_index])
        score = evs_score * training_score
        if best is None or score > best[0]:
            best = (score, int(panel), position_index, training_score)
    if best is None:
        return None
    _, panel, position_index, training_score = best
    position = np.asarray(positions[position_index], dtype=float)
    support = _support_from_position(scene, panel, position, tau)
    support.update(
        {
            "position_index": int(position_index),
            "tau_index": int(tau_index),
            "cpd_evs_direction_score": float(best[0]),
            "cpd_delay_score": delay_score,
            "cpd_range_score": float(training_score),
        }
    )
    return support


def _cpd_omp(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    cfg: dict,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    positions = _nf_position_grid(config, cfg)
    nf_matrices = {
        panel: _nf_training_matrix(scene, panel, positions)
        for panel in range(int(scene["K"]))
    }
    taus = delay_grid_from_scene(
        scene, config, int(cfg.get("delay_grid_size", 41))
    )
    selected: list[dict[str, Any]] = []
    cpd_diagnostics: list[dict[str, Any]] = []
    coeffs = np.zeros(0, dtype=complex)
    y_hat = np.zeros_like(y_vec)
    residual = y_vec.copy()
    initial_norm = float(np.linalg.norm(y_vec)) + 1.0e-15
    max_groups = int(cfg.get("max_groups", scene["K"]))
    for iteration in range(max(0, max_groups)):
        if float(np.linalg.norm(residual)) / initial_norm <= float(
            cfg.get("cpd_residual_stop_rel", 1.0e-4)
        ):
            break
        factors, _, diagnostics = complex_cp_als(
            residual.reshape(scene["I"], scene["N"], scene["T"]),
            1,
            max_iter=int(cfg.get("cpd_max_iter", 80)),
            tol=float(cfg.get("cpd_tol", 1.0e-7)),
            reg=float(cfg.get("cpd_reg", 1.0e-8)),
        )
        available = [
            panel
            for panel in range(int(scene["K"]))
            if panel not in {int(item["panel"]) for item in selected}
        ]
        support = _match_cpd_component(
            scene,
            config,
            factors[0][:, 0],
            factors[1][:, 0],
            factors[2][:, 0],
            available,
            positions,
            nf_matrices,
            taus,
        )
        if support is None:
            break
        support["cpd_iteration"] = int(iteration)
        support["coarse_stage"] = "rank1_cpd_delay_direction_then_nf_range"
        selected.append(support)
        expanded = [
            item for group in selected for item in expand_jones_group(group, 2)
        ]
        coeffs, y_hat, residual = factorized_fit_supports(
            scene, config, expanded, y_vec, ridge=1.0e-10
        )
        cpd_diagnostics.append(
            {
                "iteration": int(iteration),
                "panel": int(support["panel"]),
                "tau": float(support["tau"]),
                "range": float(support["range"]),
                "orthogonal_residual_norm": float(np.linalg.norm(residual)),
                "als_iterations": int(diagnostics["als_iterations"]),
                "als_converged": bool(diagnostics["als_converged"]),
                "als_residual": float(diagnostics["als_residual"]),
            }
        )
    return selected, coeffs, y_hat, residual, cpd_diagnostics


def _support_state(support: dict[str, Any], c0: float) -> np.ndarray:
    return np.array(
        [
            float(support["u_x"]),
            float(support["u_y"]),
            float(support["range"]),
            float(support["tau"]) * float(c0),
        ],
        dtype=float,
    )


def _support_from_state(
    scene: dict,
    support: dict[str, Any],
    state: np.ndarray,
) -> dict[str, Any]:
    result = _near_field_support(
        scene,
        int(support["panel"]),
        float(state[0]),
        float(state[1]),
        float(state[2]),
        float(state[3]) / float(scene["c0"]),
        sign=float(support.get("u_z_sign", 1.0)),
    )
    for key in (
        "cpd_iteration",
        "direction_index",
        "u_x_index",
        "u_y_index",
        "range_index",
        "tau_index",
        "coarse_stage",
        "cpd_evs_direction_score",
        "cpd_delay_score",
        "cpd_range_score",
    ):
        if key in support:
            result[key] = support[key]
    return result


def _state_bounds(scene: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    ranges = _range_grid(scene, config, 3)
    taus = delay_grid_from_scene(scene, config, 3)
    lower = np.array([-0.999, -0.999, float(ranges[0]), float(taus[0]) * scene["c0"]])
    upper = np.array([0.999, 0.999, float(ranges[-1]), float(taus[-1]) * scene["c0"]])
    return lower, upper


def _refine_sage_component(
    scene: dict,
    config: dict,
    complete_data: np.ndarray,
    support: dict[str, Any],
    cfg: dict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lower, upper = _state_bounds(scene, config)
    x0 = np.clip(_support_state(support, scene["c0"]), lower, upper)
    ue_bounds = np.asarray(config.get("ue_bounds"), dtype=float)
    reference_energy = float(np.linalg.norm(complete_data) ** 2) + 1.0e-15
    evaluations = 0

    def raw_objective(state: np.ndarray) -> float:
        clipped = np.clip(np.asarray(state, dtype=float), lower, upper)
        candidate = _support_from_state(scene, support, clipped)
        expanded = expand_jones_group(candidate, 2)
        _, _, residual = factorized_fit_supports(
            scene, config, expanded, complete_data, ridge=1.0e-10
        )
        return float(np.linalg.norm(residual) ** 2)

    def objective(state: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        clipped = np.clip(np.asarray(state, dtype=float), lower, upper)
        candidate = _support_from_state(scene, support, clipped)
        raw_value = raw_objective(clipped)
        position = np.asarray(candidate["position"], dtype=float)
        below = np.maximum(ue_bounds[:, 0] - position, 0.0)
        above = np.maximum(position - ue_bounds[:, 1], 0.0)
        disk_violation = max(clipped[0] ** 2 + clipped[1] ** 2 - 0.999**2, 0.0)
        penalty = reference_energy * (
            100.0 * float(np.dot(below + above, below + above))
            + 100.0 * disk_violation**2
        )
        return float(raw_value + penalty)

    initial_search_objective = objective(x0)
    initial_objective = raw_objective(_support_state(support, scene["c0"]))
    best_state = x0.copy()
    solver = "coordinate_fallback"
    success = True
    if scipy_is_available():
        try:
            from scipy.optimize import minimize  # type: ignore[import-not-found]

            result = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                bounds=list(zip(lower, upper)),
                options={
                    "maxiter": int(cfg.get("sage_maxiter", 30)),
                    "xatol": float(cfg.get("sage_xatol", 1.0e-5)),
                    "fatol": float(cfg.get("sage_fatol", 1.0e-8)) * reference_energy,
                },
            )
            best_state = np.clip(np.asarray(result.x, dtype=float), lower, upper)
            solver = "scipy.optimize.Nelder-Mead"
            success = bool(result.success)
        except Exception:  # noqa: BLE001 - deterministic coordinate fallback below.
            pass

    if solver == "coordinate_fallback":
        steps = np.array(
            [0.08, 0.08, max((upper[2] - lower[2]) / 12.0, 0.02), max((upper[3] - lower[3]) / 12.0, 1.0e-3)],
            dtype=float,
        )
        best_value = initial_search_objective
        for level in range(max(1, int(cfg.get("sage_fallback_levels", 2)))):
            for coordinate in range(4):
                for offset in (-steps[coordinate], steps[coordinate]):
                    candidate = best_state.copy()
                    candidate[coordinate] += offset
                    value = objective(candidate)
                    if value < best_value:
                        best_value = value
                        best_state = np.clip(candidate, lower, upper)
            steps *= 0.5

    refined = _support_from_state(scene, support, best_state)
    refined["sage_refined"] = True
    final_search_objective = objective(best_state)
    final_objective = raw_objective(best_state)
    rolled_back = bool(
        (not np.isfinite(final_objective))
        or final_objective > initial_objective * (1.0 + 1.0e-12)
    )
    if rolled_back:
        refined = dict(support)
        refined["sage_refined"] = False
        final_objective = initial_objective
        final_search_objective = initial_search_objective
    return refined, {
        "solver": solver,
        "success": bool(success and not rolled_back),
        "rolled_back": rolled_back,
        "evaluations": evaluations,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "initial_search_objective": initial_search_objective,
        "final_search_objective": final_search_objective,
    }


def _component_from_coefficients(
    scene: dict,
    config: dict,
    support: dict[str, Any],
    coefficients: np.ndarray,
) -> np.ndarray:
    design = group_design(scene, config, support, normalize=True)
    return design @ np.asarray(coefficients, dtype=complex).reshape(2)


def _sage_refine(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    selected: list[dict[str, Any]],
    cfg: dict,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    refined = [dict(item) for item in selected]
    expanded = [item for group in refined for item in expand_jones_group(group, 2)]
    coeffs, y_hat, residual = factorized_fit_supports(
        scene, config, expanded, y_vec, ridge=1.0e-10
    )
    initial_objective = float(np.linalg.norm(residual) ** 2)
    previous_objective = initial_objective
    path_trace: list[dict[str, Any]] = []
    total_evaluations = 0
    cycles_run = 0
    max_cycles = int(cfg.get("sage_iterations", 2))
    if not bool(cfg.get("sage_enabled", True)) or not refined:
        max_cycles = 0
    for cycle in range(max(0, max_cycles)):
        cycles_run += 1
        for path_index in range(len(refined)):
            component = _component_from_coefficients(
                scene,
                config,
                refined[path_index],
                coeffs[2 * path_index : 2 * path_index + 2],
            )
            complete_data = y_vec - y_hat + component
            refined[path_index], diagnostics = _refine_sage_component(
                scene,
                config,
                complete_data,
                refined[path_index],
                cfg,
            )
            total_evaluations += int(diagnostics["evaluations"])
            path_trace.append(
                {
                    "cycle": int(cycle),
                    "path_index": int(path_index),
                    "panel": int(refined[path_index]["panel"]),
                    **diagnostics,
                }
            )
            expanded = [
                item for group in refined for item in expand_jones_group(group, 2)
            ]
            coeffs, y_hat, residual = factorized_fit_supports(
                scene, config, expanded, y_vec, ridge=1.0e-10
            )
        objective = float(np.linalg.norm(residual) ** 2)
        relative_change = abs(previous_objective - objective) / (
            previous_objective + 1.0e-15
        )
        previous_objective = objective
        if relative_change <= float(cfg.get("sage_convergence_tol", 1.0e-4)):
            break
    return refined, coeffs, y_hat, residual, {
        "sage_iterations": cycles_run,
        "sage_num_evals": total_evaluations,
        "sage_initial_objective": initial_objective,
        "sage_final_objective": float(np.linalg.norm(residual) ** 2),
        "sage_path_trace": path_trace,
    }


def _support_efim(
    scene: dict,
    config: dict,
    support: dict[str, Any],
    coefficients: np.ndarray,
    noise_variance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Numerically form a local channel-parameter EFIM after Jones removal."""
    state = _support_state(support, scene["c0"])
    lower, upper = _state_bounds(scene, config)
    design0 = group_design(scene, config, support, normalize=True)
    steps = np.array([1.0e-4, 1.0e-4, 1.0e-3, 1.0e-3], dtype=float)
    derivatives: list[np.ndarray] = []
    for coordinate, step in enumerate(steps):
        plus = state.copy()
        minus = state.copy()
        plus[coordinate] = min(plus[coordinate] + step, upper[coordinate])
        minus[coordinate] = max(minus[coordinate] - step, lower[coordinate])
        denominator = plus[coordinate] - minus[coordinate]
        if denominator <= 0.0:
            derivatives.append(np.zeros(design0.shape[0], dtype=complex))
            continue
        plus_design = group_design(
            scene,
            config,
            _support_from_state(scene, support, plus),
            normalize=True,
        )
        minus_design = group_design(
            scene,
            config,
            _support_from_state(scene, support, minus),
            normalize=True,
        )
        derivatives.append(
            ((plus_design - minus_design) @ coefficients) / denominator
        )
    derivative_matrix = np.column_stack(derivatives)
    nuisance_projection = design0 @ (
        np.linalg.pinv(design0.conj().T @ design0, rcond=1.0e-10)
        @ (design0.conj().T @ derivative_matrix)
    )
    efficient_derivative = derivative_matrix - nuisance_projection
    variance = float(noise_variance)
    if not np.isfinite(variance) or variance <= 0.0:
        variance = 1.0
    efim = (2.0 / variance) * np.real(
        efficient_derivative.conj().T @ efficient_derivative
    )
    efim = 0.5 * (efim + efim.T)
    eigenvalues, eigenvectors = np.linalg.eigh(efim)
    clipped = np.maximum(eigenvalues, 0.0)
    efim = (eigenvectors * clipped[None, :]) @ eigenvectors.T
    return efim, {
        "efim_eigenvalues": clipped,
        "efim_rank": int(np.sum(clipped > max(float(np.max(clipped)), 1.0) * 1.0e-10)),
    }


def _wls_position_clock(
    scene: dict,
    config: dict,
    selected: list[dict[str, Any]],
    coefficients: np.ndarray,
    noise_variance: float,
    cfg: dict,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    bounds_p = np.asarray(config.get("ue_bounds"), dtype=float)
    bounds_dt = np.asarray(config.get("delta_t_bounds"), dtype=float)
    positions = np.asarray([item["position"] for item in selected], dtype=float)
    p0 = (
        np.median(positions, axis=0)
        if positions.size
        else np.mean(bounds_p, axis=1)
    )
    p0 = np.clip(p0, bounds_p[:, 0], bounds_p[:, 1])
    dt_candidates = []
    for support in selected:
        panel = int(support["panel"])
        distance = float(
            np.linalg.norm(p0 - np.asarray(scene["ris_centers"][panel], dtype=float))
        )
        dt_candidates.append(
            float(support["tau"])
            - (distance + float(scene["d_RB"][panel])) / float(scene["c0"])
        )
    dt0 = float(np.median(dt_candidates)) if dt_candidates else float(np.mean(bounds_dt))
    dt0 = float(np.clip(dt0, bounds_dt[0], bounds_dt[1]))

    sqrt_weights: list[np.ndarray] = []
    weight_trace: list[dict[str, Any]] = []
    for path_index, support in enumerate(selected):
        try:
            efim, trace = _support_efim(
                scene,
                config,
                support,
                coefficients[2 * path_index : 2 * path_index + 2],
                noise_variance,
            )
            eigenvalues, eigenvectors = np.linalg.eigh(efim)
            sqrt_weight = (
                np.sqrt(np.maximum(eigenvalues, 0.0))[:, None]
                * eigenvectors.T
            )
            weight_trace.append({"panel": int(support["panel"]), **trace})
        except Exception as exc:  # noqa: BLE001 - energy proxy remains data-only.
            energy = float(
                np.linalg.norm(coefficients[2 * path_index : 2 * path_index + 2]) ** 2
            )
            sqrt_weight = np.sqrt(max(energy, 1.0e-12)) * np.eye(4)
            weight_trace.append(
                {
                    "panel": int(support["panel"]),
                    "efim_rank": 0,
                    "warning": f"efim_fallback: {type(exc).__name__}: {exc}",
                }
            )
        sqrt_weights.append(sqrt_weight)
    maximum_weight = max(
        [float(np.linalg.norm(weight, 2)) for weight in sqrt_weights] + [1.0]
    )
    sqrt_weights = [weight / maximum_weight for weight in sqrt_weights]

    lower = np.r_[bounds_p[:, 0], bounds_dt[0] * scene["c0"]]
    upper = np.r_[bounds_p[:, 1], bounds_dt[1] * scene["c0"]]
    x0 = np.clip(np.r_[p0, dt0 * scene["c0"]], lower, upper)

    def residual_fn(state: np.ndarray) -> np.ndarray:
        p = np.asarray(state[:3], dtype=float)
        clock_m = float(state[3])
        blocks: list[np.ndarray] = []
        for support, sqrt_weight in zip(selected, sqrt_weights):
            panel = int(support["panel"])
            q_local = np.asarray(scene["rotations"][panel], dtype=float) @ (
                p - np.asarray(scene["ris_centers"][panel], dtype=float)
            )
            range_m = float(np.linalg.norm(q_local)) + 1.0e-15
            direction_local = q_local / range_m
            tau_m = range_m + float(scene["d_RB"][panel]) + clock_m
            estimate = _support_state(support, scene["c0"])
            model = np.array(
                [direction_local[0], direction_local[1], range_m, tau_m],
                dtype=float,
            )
            blocks.append(sqrt_weight @ (estimate - model))
        if not blocks:
            return np.asarray(state - x0, dtype=float)
        return np.concatenate(blocks)

    solver = "bounded_initialization"
    success = False
    num_evals = 1
    solution = x0.copy()
    if bool(cfg.get("wls_enabled", True)) and scipy_is_available() and selected:
        try:
            from scipy.optimize import least_squares  # type: ignore[import-not-found]

            result = least_squares(
                residual_fn,
                x0,
                bounds=(lower, upper),
                max_nfev=int(cfg.get("wls_max_nfev", 100)),
                x_scale=np.maximum(upper - lower, 1.0e-9),
            )
            solution = result.x
            solver = "scipy.optimize.least_squares"
            success = bool(result.success)
            num_evals = int(result.nfev)
        except Exception:  # noqa: BLE001 - return the bounded initialization.
            pass
    return solution[:3].astype(float), float(solution[3] / scene["c0"]), {
        "wls_solver": solver,
        "wls_success": success,
        "wls_num_evals": num_evals,
        "wls_final_cost": float(np.linalg.norm(residual_fn(solution)) ** 2),
        "wls_weight_model": "local_channel_efim_after_jones_projection",
        "wls_weight_trace": weight_trace,
    }


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
        taus.append(
            (float(range_m) + float(scene["d_RB"][panel])) / float(scene["c0"])
            + float(delta_t)
        )
    return {"ranges": np.asarray(ranges), "taus": np.asarray(taus)}


def run_nf_ris_groupomp_localgrid_wls_baseline(
    data: dict, config: dict
) -> BaselineResult:
    start = time.perf_counter()
    scene = data["scene"]
    y_vec = vectorize_raw_observation(data["Y_noisy"])
    cfg = dict(
        config.get("baselines", {}).get("nf_ris_groupomp_localgrid_wls", {})
    )

    selected, coarse_coeffs, coarse_y_hat, coarse_residual, cpd_trace = _cpd_omp(
        scene, config, y_vec, cfg
    )
    coarse_objective = float(np.linalg.norm(coarse_residual) ** 2)
    (
        sage_supports,
        sage_coeffs,
        sage_y_hat,
        sage_residual,
        sage_diagnostics,
    ) = _sage_refine(scene, config, y_vec, selected, cfg)
    final_p, final_dt, wls_diagnostics = _wls_position_clock(
        scene,
        config,
        sage_supports,
        sage_coeffs,
        float(data.get("noise_variance", 1.0)),
        cfg,
    )
    bounds_p = np.asarray(config.get("ue_bounds"), dtype=float)
    bounds_dt = np.asarray(config.get("delta_t_bounds"), dtype=float)
    final_p = np.clip(final_p, bounds_p[:, 0], bounds_p[:, 1])
    final_dt = float(np.clip(final_dt, bounds_dt[0], bounds_dt[1]))
    final_groups = supports_from_position_clock(
        scene, final_p, final_dt, model_variant="near_field"
    )
    final_expanded = [
        item for group in final_groups for item in expand_jones_group(group, 2)
    ]
    final_coeffs, final_y_hat, final_residual = factorized_fit_supports(
        scene, config, final_expanded, y_vec, ridge=1.0e-10
    )
    sage_enabled = bool(cfg.get("sage_enabled", True)) and int(
        cfg.get("sage_iterations", 2)
    ) > 0 and bool(sage_supports)
    diagnostics = {
        "dictionary_mode": "nf_ris_cpd_omp_sage_wls_adaptation",
        "model_variant": "near_field_cpd_omp_sage_wls_adaptation",
        "reference_algorithm": "NF-RIS CPD-OMP-SAGE-WLS adaptation",
        "adaptation_note": (
            "evs_delay_training_cpd_with_panel_aware_spatial_search;_"
            "random_Omega_prevents_the_paper_Kronecker_RIS_axis_factorization"
        ),
        "cpd_omp_adapted_used": len(cpd_trace) > 0,
        "cpd_tensor_shape": (int(scene["I"]), int(scene["N"]), int(scene["T"])),
        "cpd_rank1_sequential": True,
        "cpd_trace": cpd_trace,
        "coarse_direction_delay_grid_size": int(
            len(_nf_position_grid(config, cfg))
            + int(cfg.get("delay_grid_size", 41))
        ),
        "coarse_selected_groups": selected,
        "coarse_position": (
            np.median(np.asarray([item["position"] for item in selected]), axis=0)
            if selected
            else np.mean(bounds_p, axis=1)
        ),
        "coarse_delta_t": float("nan"),
        "near_field_l1_refinement_used": False,
        "near_field_one_sparse_refinement_used": bool(cpd_trace),
        "l1_tau_lambda": 0.0,
        "near_field_distance_search": "near_field_position_grid_matched_filter",
        "sage_enabled": sage_enabled,
        **sage_diagnostics,
        "local_grid_enabled": False,
        "local_grid_iterations": 0,
        "local_grid_num_evals": 0,
        "wls_enabled": bool(cfg.get("wls_enabled", True)),
        **wls_diagnostics,
        "subris_mode": "not_used_normal_ris_adaptation",
        "subris_shape": (),
        "subris_fallback_used": False,
        "final_clipped_to_bounds": bool(
            np.any(final_p <= bounds_p[:, 0]) or np.any(final_p >= bounds_p[:, 1])
        ),
        "grid_size": int(
            len(_nf_position_grid(config, cfg))
            + int(cfg.get("delay_grid_size", 41))
        ),
        "support_size": len(sage_supports),
        "selected_support": sage_supports,
        "expanded_supports": final_expanded,
        "expanded_support_count": len(final_expanded),
        "selected_group_count": len(sage_supports),
        "selected_panel_count": len({int(item["panel"]) for item in sage_supports}),
        "selected_panels": [int(item["panel"]) for item in sage_supports],
        "unique_panel_constraint": True,
        "coeff_norm": float(np.linalg.norm(final_coeffs)),
        "raw_coarse_objective": coarse_objective,
        "raw_sage_objective": float(np.linalg.norm(sage_residual) ** 2),
        "raw_common_geometry_objective": float(np.linalg.norm(final_residual) ** 2),
        "raw_objective_final": float(
            np.linalg.norm(final_residual) ** 2 / max(y_vec.size, 1)
        ),
        "backend": "cpu",
        "gpu_used": False,
        "gpu_device": "",
        "coarse_backend": "cpu",
        "coarse_gpu_used": False,
        "local_refinement_backend": "cpu",
        "local_refinement_gpu_used": False,
        "wls_backend": "cpu",
        "mixed_backend": False,
        "backend_warning": "",
        "warning": "",
    }
    return BaselineResult(
        name="nf_ris_groupomp_localgrid_wls",
        p_u=final_p,
        delta_t=final_dt,
        Y_hat=final_y_hat.reshape(scene["I"], scene["N"], scene["T"]),
        raw_objective_final=float(
            np.linalg.norm(final_residual) ** 2 / max(y_vec.size, 1)
        ),
        components=_model_components(scene, final_p, final_dt),
        selected_support=sage_supports,
        runtime_s=time.perf_counter() - start,
        diagnostics=diagnostics,
    )
