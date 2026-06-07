"""Reliability-gated robust joint near-field position-domain projection."""

from __future__ import annotations

import copy
import time

import numpy as np

from .geometry import local_geometry_from_position, position_from_local_geometry
from .projections_delay import tau_from_pole
from .utils import bounded_coordinate_search, check_finite, scipy_is_available


def _panel_ordered_rank1_ratios(stage1_estimate: dict, k_paths: int) -> np.ndarray | None:
    ratios = stage1_estimate.get("stage1_rank1_ratios")
    if ratios is None:
        return None
    ratios = np.asarray(ratios, dtype=float).reshape(-1)
    if ratios.size != k_paths:
        return None
    assignment = stage1_estimate.get("assignment")
    if assignment is None:
        return ratios.copy()
    assignment = np.asarray(assignment, dtype=int).reshape(-1)
    if assignment.size != k_paths:
        return ratios.copy()
    ordered = np.ones(k_paths, dtype=float)
    for col, panel in enumerate(assignment):
        if 0 <= int(panel) < k_paths:
            ordered[int(panel)] = ratios[col]
    return ordered


def _confidence_weights(stage1_estimate: dict, config: dict, k_paths: int) -> tuple[np.ndarray, np.ndarray | None]:
    ratios = _panel_ordered_rank1_ratios(stage1_estimate, k_paths)
    if not bool(config.get("jnpp_use_confidence_weights", True)) or ratios is None:
        return np.ones(k_paths, dtype=float), ratios
    rho = float(config.get("jnpp_rank_weight_rho", 2.0))
    min_weight = float(config.get("jnpp_min_weight", 0.05))
    weights = np.exp(-rho * np.maximum(ratios, 0.0))
    weights = np.maximum(weights, min_weight)
    return weights.astype(float), ratios


def _stage1_position(stage1_estimate: dict, scene: dict, config: dict) -> np.ndarray:
    if "p_u" in stage1_estimate:
        p = np.asarray(stage1_estimate["p_u"], dtype=float).reshape(3)
        if np.all(np.isfinite(p)):
            return p
    positions = []
    ris_eta = np.asarray(stage1_estimate["ris_eta"], dtype=float)
    for k in range(scene["K"]):
        try:
            positions.append(
                position_from_local_geometry(
                    scene["ris_centers"][k],
                    scene["rotations"][k],
                    ris_eta[k, 0],
                    ris_eta[k, 1],
                    ris_eta[k, 2],
                )
            )
        except (ValueError, IndexError):
            pass
    if positions:
        p = np.median(np.asarray(positions, dtype=float), axis=0)
    else:
        bounds = np.asarray(config.get("ue_bounds", np.array([[0.0, 1.0]] * 3)), dtype=float)
        p = np.mean(bounds, axis=1)
    if "ue_bounds" in config:
        bounds = np.asarray(config["ue_bounds"], dtype=float)
        p = np.clip(p, bounds[:, 0], bounds[:, 1])
    return p


def _position_bounds(p0: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    if "ue_bounds" in config:
        bounds = np.asarray(config["ue_bounds"], dtype=float)
        if bounds.shape == (3, 2) and np.all(np.isfinite(bounds)):
            return bounds[:, 0].copy(), bounds[:, 1].copy()
    radius = float(config.get("jnpp_position_box_m", 1.5))
    return p0 - radius, p0 + radius


def _tau_stage1(stage1_estimate: dict, scene: dict) -> np.ndarray:
    return np.array(
        [tau_from_pole(pole, scene["delta_f"]) for pole in stage1_estimate["poles"]],
        dtype=float,
    )


def _response_and_jacobian(
    p_u: np.ndarray,
    panel: int,
    scene: dict,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(scene["rotations"][panel], dtype=float)
    q_vec = rotation @ (np.asarray(p_u, dtype=float).reshape(3) - scene["ris_centers"][panel])
    range_m = float(np.linalg.norm(q_vec))
    safe_range = max(range_m, eps)
    rho = np.asarray(scene["ris_grid"], dtype=float)
    diff = q_vec[None, :] - rho
    distances = np.linalg.norm(diff, axis=1)
    safe_distances = np.maximum(distances, eps)
    delta = distances - safe_range
    kappa = 2.0 * np.pi / float(scene["wavelength"])
    u_vec = np.exp(-1j * kappa * delta)
    geom_grad = diff / safe_distances[:, None] - q_vec[None, :] / safe_range
    ddelta_dp = geom_grad @ rotation
    du_dp = -1j * kappa * u_vec[:, None] * ddelta_dp
    a_eff = np.asarray(scene["Omega"][panel], dtype=complex) * np.asarray(
        scene["a_RB"][panel], dtype=complex
    )[None, :]
    h_vec = a_eff @ u_vec
    jac = a_eff @ du_dp
    return h_vec, jac


def _objective_and_grad_factory(
    c_tilde: np.ndarray,
    scene: dict,
    weights: np.ndarray,
    subset: tuple[int, ...],
    eps: float,
) -> tuple:
    counters = {"objective": 0, "gradient": 0}

    def value_and_grad(p_u: np.ndarray, need_grad: bool = True) -> tuple[float, np.ndarray]:
        counters["objective"] += 1
        if need_grad:
            counters["gradient"] += 1
        p_u = np.asarray(p_u, dtype=float).reshape(3)
        total = 0.0
        grad = np.zeros(3, dtype=float)
        for k in subset:
            c_k = c_tilde[:, k]
            h_k, jac_k = _response_and_jacobian(p_u, k, scene, eps)
            g_k = np.vdot(h_k, c_k)
            n_k = float(np.real(np.vdot(h_k, h_k)) + eps)
            c_norm = float(np.real(np.vdot(c_k, c_k)))
            abs_g_sq = float(np.abs(g_k) ** 2)
            e_k = c_norm - abs_g_sq / n_k
            total += float(weights[k]) * e_k
            if need_grad:
                for dim in range(3):
                    dh = jac_k[:, dim]
                    dg_term = 2.0 * float(np.real(np.conj(g_k) * np.vdot(dh, c_k)))
                    dn_term = 2.0 * float(np.real(np.vdot(h_k, dh)))
                    d_e = -((dg_term * n_k) - (abs_g_sq * dn_term)) / (n_k**2)
                    grad[dim] += float(weights[k]) * d_e
        return float(total), grad

    return value_and_grad, counters


def _numeric_grad(fun, p_u: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    grad = np.zeros(3, dtype=float)
    for dim in range(3):
        step = 1.0e-5 * max(1.0, abs(float(p_u[dim])))
        p_plus = p_u.copy()
        p_minus = p_u.copy()
        p_plus[dim] = min(upper[dim], p_plus[dim] + step)
        p_minus[dim] = max(lower[dim], p_minus[dim] - step)
        denom = p_plus[dim] - p_minus[dim]
        if denom <= 0.0:
            continue
        grad[dim] = (fun(p_plus) - fun(p_minus)) / denom
    return grad


def _gradient_check(fun, p0: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    value, analytic = fun(p0, True)
    numeric = _numeric_grad(lambda x: fun(x, False)[0], p0, lower, upper)
    err = float(np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1.0e-12))
    return {
        "jnpp_gradient_check_value": float(value),
        "jnpp_gradient_check_rel_error": err,
        "jnpp_gradient_check_analytic": analytic.copy(),
        "jnpp_gradient_check_numeric": numeric.copy(),
    }


def _starts(stage1_estimate: dict, scene: dict, config: dict, lower: np.ndarray, upper: np.ndarray) -> list[np.ndarray]:
    p0 = _stage1_position(stage1_estimate, scene, config)
    starts = [p0]
    perturb = float(config.get("jnpp_start_perturb_m", 0.25))
    for dim in range(3):
        for sign in (-1.0, 1.0):
            p = p0.copy()
            p[dim] += sign * perturb
            starts.append(p)
    if bool(config.get("jnpp_use_coarse_grid", False)):
        grid_n = int(config.get("jnpp_coarse_grid_points_per_dim", 3))
        axes = [np.linspace(lower[d], upper[d], max(2, grid_n)) for d in range(3)]
        starts.extend(np.array([x, y, z], dtype=float) for x in axes[0] for y in axes[1] for z in axes[2])
    max_starts = max(1, int(config.get("jnpp_num_starts", 4)))
    unique = []
    seen = set()
    for p in starts:
        p = np.clip(np.asarray(p, dtype=float), lower, upper)
        key = tuple(np.round(p, 9))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
        if len(unique) >= max_starts:
            break
    return unique


def _subsets(k_paths: int, config: dict) -> list[tuple[int, ...]]:
    subsets = [tuple(range(k_paths))]
    if bool(config.get("jnpp_use_leave_one_out", True)) and k_paths >= 3:
        for leave_out in range(k_paths):
            subsets.append(tuple(k for k in range(k_paths) if k != leave_out))
    max_candidates = max(1, int(config.get("jnpp_max_candidates", 1 + k_paths)))
    return subsets[:max_candidates]


def _minimize_subset(
    c_tilde: np.ndarray,
    scene: dict,
    config: dict,
    weights: np.ndarray,
    subset: tuple[int, ...],
    starts: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[list[dict], dict]:
    eps = float(config.get("eps", 1.0e-10))
    fun, counters = _objective_and_grad_factory(c_tilde, scene, weights, subset, eps)
    gradient_mode = "analytic"
    check = {}
    if bool(config.get("jnpp_check_gradient", False)) and starts:
        check = _gradient_check(fun, starts[0], lower, upper)

    candidates = []
    for start in starts:
        if scipy_is_available():
            from scipy.optimize import minimize

            def objective(x):
                value, grad = fun(x, True)
                if not np.isfinite(value) or not np.all(np.isfinite(grad)):
                    raise FloatingPointError("nonfinite JNPP objective or gradient")
                return value, grad

            try:
                result = minimize(
                    objective,
                    start,
                    method=str(config.get("jnpp_minimize_method", "L-BFGS-B")),
                    jac=True,
                    bounds=list(zip(lower, upper)),
                    options={
                        "maxiter": int(config.get("jnpp_max_iter", 80)),
                        "ftol": float(config.get("jnpp_ftol", 1.0e-10)),
                        "gtol": float(config.get("jnpp_gtol", 1.0e-6)),
                    },
                )
                p_hat = np.clip(np.asarray(result.x, dtype=float), lower, upper)
                value = float(result.fun)
                message = str(result.message)
                success = bool(result.success)
            except (FloatingPointError, ValueError):
                gradient_mode = "numerical_fallback"

                def objective_only(x):
                    return fun(x, False)[0]

                result = minimize(
                    objective_only,
                    start,
                    method="L-BFGS-B",
                    jac=None,
                    bounds=list(zip(lower, upper)),
                    options={"maxiter": int(config.get("jnpp_max_iter", 80))},
                )
                p_hat = np.clip(np.asarray(result.x, dtype=float), lower, upper)
                value = float(result.fun)
                message = str(result.message)
                success = bool(result.success)
        else:
            gradient_mode = "numerical_fallback"

            def objective_unit(x_unit):
                p = lower + np.clip(x_unit, 0.0, 1.0) * (upper - lower)
                return fun(p, False)[0]

            x0 = (start - lower) / np.maximum(upper - lower, eps)
            x_best, value, info = bounded_coordinate_search(
                objective_unit,
                x0,
                np.zeros(3),
                np.ones(3),
                max_iter=int(config.get("jnpp_max_iter", 80)),
            )
            p_hat = lower + np.clip(x_best, 0.0, 1.0) * (upper - lower)
            message = str(info.get("message", "bounded coordinate search"))
            success = bool(info.get("success", False))
        candidates.append(
            {
                "p_u": p_hat,
                "subset": tuple(int(k) for k in subset),
                "subset_objective": float(value),
                "optimizer_success": success,
                "optimizer_message": message,
            }
        )
    candidates.sort(key=lambda item: item["subset_objective"])
    return candidates[:1], {"jnpp_gradient_mode": gradient_mode, **counters, **check}


def _clock_postcheck(p_u: np.ndarray, tau_stage1: np.ndarray, scene: dict, config: dict) -> dict:
    delta_t = []
    for k in range(scene["K"]):
        q_vec = scene["rotations"][k] @ (p_u - scene["ris_centers"][k])
        delta_t.append(tau_stage1[k] - (np.linalg.norm(q_vec) + scene["d_RB"][k]) / scene["c0"])
    delta_t = np.asarray(delta_t, dtype=float)
    sigma_ns = float(np.std(delta_t) * 1.0e9)
    median_ns = float(np.median(delta_t) * 1.0e9)
    return {
        "delta_t_values_s": delta_t,
        "sigma_delta_t_ns": sigma_ns,
        "median_delta_t_ns": median_ns,
        "clock_consistent": bool(sigma_ns <= float(config.get("jnpp_clock_postcheck_ns", 0.5))),
    }


def _evaluate_all_candidate(
    candidate: dict,
    c_tilde: np.ndarray,
    scene: dict,
    config: dict,
    weights: np.ndarray,
    tau_stage1: np.ndarray,
) -> dict:
    all_subset = tuple(range(scene["K"]))
    fun, counters = _objective_and_grad_factory(
        c_tilde, scene, weights, all_subset, float(config.get("eps", 1.0e-10))
    )
    objective, _ = fun(candidate["p_u"], False)
    clock = _clock_postcheck(candidate["p_u"], tau_stage1, scene, config)
    out = dict(candidate)
    out.update(
        {
            "all_objective": float(objective),
            "jnpp_clock_std_ns": float(clock["sigma_delta_t_ns"]),
            "jnpp_delta_t_ns": float(clock["median_delta_t_ns"]),
            "jnpp_clock_consistent": bool(clock["clock_consistent"]),
            "delta_t_values_s": clock["delta_t_values_s"],
            "eval_objective_count": counters["objective"],
        }
    )
    return out


def _select_candidate(candidates: list[dict], config: dict) -> dict:
    consistent = [cand for cand in candidates if cand["jnpp_clock_consistent"]]
    pool = consistent if consistent else candidates
    best_objective = min(cand["all_objective"] for cand in pool)
    rel_tol = float(config.get("jnpp_clock_tie_rel_tol", 1.0e-3))
    close = [
        cand
        for cand in pool
        if cand["all_objective"] <= best_objective * (1.0 + rel_tol) + 1.0e-12
    ]
    return min(close, key=lambda cand: (cand["jnpp_clock_std_ns"], cand["all_objective"]))


def _build_refined_estimate(
    stage1_estimate: dict,
    c_tilde: np.ndarray,
    p_u: np.ndarray,
    tau_stage1: np.ndarray,
    scene: dict,
    config: dict,
) -> tuple[dict, list[dict]]:
    estimate = copy.deepcopy(stage1_estimate)
    k_paths = int(scene["K"])
    c_hat = np.empty_like(c_tilde, dtype=complex)
    ris_eta = np.empty((k_paths, 3), dtype=float)
    alpha = np.empty(k_paths, dtype=complex)
    details = []
    eps = float(config.get("eps", 1.0e-10))
    for k in range(k_paths):
        h_k, _ = _response_and_jacobian(p_u, k, scene, eps)
        alpha[k] = np.vdot(h_k, c_tilde[:, k]) / (np.vdot(h_k, h_k) + eps)
        c_hat[:, k] = alpha[k] * h_k
        ris_eta[k] = np.asarray(
            local_geometry_from_position(
                p_u, scene["ris_centers"][k], scene["rotations"][k]
            )[:3],
            dtype=float,
        )
        residual = np.linalg.norm(c_tilde[:, k] - c_hat[:, k]) / (
            np.linalg.norm(c_tilde[:, k]) + eps
        )
        details.append(
            {
                "path": int(k),
                "skipped": False,
                "accepted": True,
                "reason": "accepted_robust_jnpp",
                "selected_eta": ris_eta[k].copy(),
                "alpha": alpha[k],
                "relative_residual": float(residual),
            }
        )
    delta_t_values = []
    for k in range(k_paths):
        delta_t_values.append(tau_stage1[k] - (ris_eta[k, 0] + scene["d_RB"][k]) / scene["c0"])
    delta_t_init = float(np.median(np.asarray(delta_t_values, dtype=float)))
    if "delta_t_bounds" in config:
        bounds = np.asarray(config["delta_t_bounds"], dtype=float)
        delta_t_init = float(np.clip(delta_t_init, bounds[0], bounds[1]))
    estimate["C"] = c_hat
    estimate["ris_eta"] = ris_eta
    estimate["p_u"] = np.asarray(p_u, dtype=float).copy()
    estimate["delta_t"] = delta_t_init
    estimate["Delta_t_init"] = delta_t_init
    estimate["p_u_init"] = np.asarray(p_u, dtype=float).copy()
    estimate["jnpp_alpha"] = alpha
    estimate["columns_are_panel_ordered"] = True
    check_finite("robust JNPP C", c_hat)
    return estimate, details


def robust_jnpp_basin_recovery(stage1_estimate: dict, scene: dict, config: dict) -> tuple[dict, dict]:
    """Run factor-domain robust JNPP and return a VP initialization estimate."""
    total_start = time.perf_counter()
    k_paths = int(scene["K"])
    c_tilde = np.asarray(stage1_estimate["C"], dtype=complex)
    if c_tilde.shape != (scene["T"], k_paths):
        raise ValueError("stage1_estimate['C'] must have shape T x K in panel order")
    weights, ratios = _confidence_weights(stage1_estimate, config, k_paths)
    p0 = _stage1_position(stage1_estimate, scene, config)
    lower, upper = _position_bounds(p0, config)
    starts = _starts(stage1_estimate, scene, config, lower, upper)
    tau_stage1 = _tau_stage1(stage1_estimate, scene)
    subsets = _subsets(k_paths, config)

    raw_candidates = []
    objective_count = 0
    gradient_count = 0
    gradient_mode = "analytic"
    gradient_check = {}
    for subset in subsets:
        subset_candidates, diag = _minimize_subset(
            c_tilde, scene, config, weights, subset, starts, lower, upper
        )
        raw_candidates.extend(subset_candidates)
        objective_count += int(diag.get("objective", 0))
        gradient_count += int(diag.get("gradient", 0))
        if diag.get("jnpp_gradient_mode") == "numerical_fallback":
            gradient_mode = "numerical_fallback"
        gradient_check.update(
            {key: value for key, value in diag.items() if key.startswith("jnpp_gradient_check")}
        )

    evaluated = [
        _evaluate_all_candidate(cand, c_tilde, scene, config, weights, tau_stage1)
        for cand in raw_candidates
    ]
    objective_count += sum(int(cand.get("eval_objective_count", 0)) for cand in evaluated)
    best = _select_candidate(evaluated, config)
    estimate, ris_details = _build_refined_estimate(
        stage1_estimate, c_tilde, best["p_u"], tau_stage1, scene, config
    )

    runtime = float(time.perf_counter() - total_start)
    diagnostics = {
        "stage2_rescue_mode": "robust_jnpp",
        "jnpp_num_starts": int(len(starts)),
        "jnpp_num_candidates": int(len(evaluated)),
        "jnpp_use_confidence_weights": bool(config.get("jnpp_use_confidence_weights", True)),
        "jnpp_weights": weights.copy(),
        "jnpp_rank1_ratios_used": None if ratios is None else ratios.copy(),
        "jnpp_use_leave_one_out": bool(config.get("jnpp_use_leave_one_out", True)),
        "jnpp_best_objective": float(best["all_objective"]),
        "jnpp_best_position": np.asarray(best["p_u"], dtype=float).copy(),
        "jnpp_best_clock_std_ns": float(best["jnpp_clock_std_ns"]),
        "jnpp_best_delta_t_ns": float(best["jnpp_delta_t_ns"]),
        "jnpp_clock_consistent": bool(best["jnpp_clock_consistent"]),
        "jnpp_gradient_mode": gradient_mode,
        "jnpp_runtime_total": runtime,
        "jnpp_objective_eval_count": int(objective_count),
        "jnpp_gradient_eval_count": int(gradient_count),
        "jnpp_accepted_by_raw_vp": False,
        "jnpp_assignment_aware": bool(config.get("jnpp_assignment_aware", False)),
        "jnpp_assignment_aware_ran": False,
        "jnpp_candidates": evaluated,
        "updates": [
            {
                "ris_projection_details": ris_details,
                "mode4_assignment_order": list(range(k_paths)),
                "iteration_accepted": True,
                "iteration_sse_after": float(best["all_objective"]),
            }
        ],
        "z_hat_history": [],
        "residuals_noisy_rmse": [],
        "ris_projection_total_s": 0.0,
        "structured_refinement_total": runtime,
        **gradient_check,
    }
    return estimate, diagnostics
