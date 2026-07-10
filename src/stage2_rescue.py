"""Common Stage-II geometry seeds and normalized rescue polish."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import position_from_local_geometry
from .projections_delay import tau_from_pole
from .projections_ris import local_ris_search_config
from .robust_jnpp import robust_jnpp_geometry_consistency_score


@dataclass
class Stage2CommonState:
    stage1_estimate: dict
    refined_estimate: dict
    rescue_config: dict
    tau_hat_s: np.ndarray
    sigma_tau_s: np.ndarray
    sigma_tau_sq_s2: np.ndarray
    sigma_tau_source: str
    sigma_tau_used_floor: bool
    local_fix_records: list[dict]
    common_refinement_success: bool
    common_refinement_runtime_s: float
    common_refinement_diagnostics: dict = field(default_factory=dict)


def _finite_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if shape is not None and array.shape != shape:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def build_local_fix_records(
    estimate: dict,
    scene: dict,
    config: dict,
    *,
    source_stage: str = "refined",
) -> list[dict]:
    """Screen physical-panel local fixes using existing Stage-I validity fields."""
    k_paths = int(scene["K"])
    ris_eta = _finite_array(estimate.get("ris_eta"))
    residuals = _finite_array(estimate.get("stage1_ris_residuals"))
    assignment = estimate.get("column_to_panel_assignment", estimate.get("assignment"))
    try:
        assignment = np.asarray(assignment, dtype=int).reshape(-1)
    except (TypeError, ValueError):
        assignment = np.empty(0, dtype=int)
    valid_assignment = assignment.size == k_paths and sorted(assignment.tolist()) == list(range(k_paths))
    validity = estimate.get("stage1_local_geometry_valid")
    if validity is not None:
        try:
            validity = np.asarray(validity, dtype=bool).reshape(-1)
        except (TypeError, ValueError):
            validity = None
    records: list[dict] = []
    for panel in range(k_paths):
        reason = ""
        assigned_column = None
        if not valid_assignment:
            reason = "invalid_panel_assignment"
        else:
            columns = np.flatnonzero(assignment == panel)
            if columns.size != 1:
                reason = "invalid_panel_assignment"
            else:
                assigned_column = int(columns[0])
        if not reason and (ris_eta is None or ris_eta.shape != (k_paths, 3)):
            reason = "missing_local_geometry"
        if not reason and validity is not None and validity.size == k_paths and not bool(validity[panel]):
            reason = "geometry_validity_false"
        if not reason and not np.all(np.isfinite(ris_eta[panel])):
            reason = "nonfinite_local_fix"
        if not reason:
            range_m, theta, phi = (float(value) for value in ris_eta[panel])
            search = local_ris_search_config(scene, config, panel)
            if not (
                search["range_bounds"][0] <= range_m <= search["range_bounds"][1]
                and search["elev_bounds"][0] <= theta <= search["elev_bounds"][1]
                and search["az_bounds"][0] <= phi <= search["az_bounds"][1]
            ):
                reason = "local_geometry_out_of_domain"
        local_position = np.full(3, np.nan, dtype=float)
        if not reason:
            try:
                local_position = position_from_local_geometry(
                    scene["ris_centers"][panel],
                    scene["rotations"][panel],
                    *ris_eta[panel],
                )
            except (ValueError, TypeError, IndexError, np.linalg.LinAlgError):
                reason = "geometry_conversion_failure"
        records.append(
            {
                "panel_index": panel,
                "assigned_column_index": assigned_column,
                "valid": not bool(reason),
                "reject_reason": reason,
                "position": local_position,
                "eta": ris_eta[panel].copy() if ris_eta is not None and ris_eta.shape == (k_paths, 3) else np.full(3, np.nan),
                "residual_after": float(residuals[panel]) if residuals is not None and residuals.shape == (k_paths,) else float("nan"),
                "weight_source": "uniform_fallback",
                "weight_scalar": 1.0,
                "source_stage": source_stage,
            }
        )
    return records


def _null_basis_of_ones(k_paths: int) -> np.ndarray:
    if k_paths <= 1:
        return np.empty((k_paths, 0), dtype=float)
    _, _, vh = np.linalg.svd(np.ones((1, k_paths), dtype=float), full_matrices=True)
    return np.asarray(vh[1:, :].T, dtype=float)


def _whitening(covariance: np.ndarray) -> np.ndarray:
    covariance = (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.size == 0 or not np.all(np.isfinite(eigenvalues)):
        raise np.linalg.LinAlgError("invalid covariance spectrum")
    largest = float(np.max(eigenvalues))
    if not np.isfinite(largest) or largest <= 0.0:
        raise np.linalg.LinAlgError("nonpositive covariance scale")
    floor = max(largest * 1.0e-12, np.finfo(float).eps * largest)
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def decoupled_clock_estimate(
    state: Stage2CommonState,
    scene: dict,
    config: dict,
) -> dict:
    """Estimate the common clock offset without knowing the UE position.

    Each panel's compressed near-field response yields an absolute range
    ``r_k = ris_eta[k, 0]``, while the OFDM delay yields the clock-biased
    pseudorange ``m_k = c0 * tau_k - d_RB[k] = rho_k + s``.  Their difference

        s_k = m_k - r_k = s + eps_k

    does not depend on ``p``, so the clock is identifiable in closed form and
    each panel supplies an independent replica of it.  A median/MAD screen can
    therefore reject a corrupted delay even at K = 3, which is impossible once
    the clock has been annihilated (that consumes one of the K equations).

    Returns ``clock_s = NaN`` and ``available = False`` when the inputs are
    unusable; the caller then keeps its own clock estimate.
    """
    k_paths = int(scene["K"])
    result: dict[str, Any] = {
        "available": False,
        "clock_s": float("nan"),
        "clock_offset_m": float("nan"),
        "reason": "",
        "per_panel_offset_m": np.full(k_paths, np.nan),
        "inlier_mask": np.zeros(k_paths, dtype=bool),
        "robust_scale_m": float("nan"),
        "num_inliers": 0,
    }
    ranges = _finite_array(state.refined_estimate.get("ris_eta"))
    tau_hat = _finite_array(state.tau_hat_s, (k_paths,))
    centers = _finite_array(scene.get("ris_centers"), (k_paths, 3))
    d_rb = _finite_array(scene.get("d_RB"), (k_paths,))
    sigma_tau_sq = _finite_array(state.sigma_tau_sq_s2, (k_paths,))
    if ranges is None or ranges.shape != (k_paths, 3) or tau_hat is None or centers is None:
        result["reason"] = "missing_delay_or_range_data"
        return result
    if d_rb is None or sigma_tau_sq is None or not np.all(sigma_tau_sq > 0.0):
        result["reason"] = "missing_delay_or_range_data"
        return result

    valid = np.zeros(k_paths, dtype=bool)
    for record in state.local_fix_records:
        panel = int(record.get("panel_index", -1))
        if 0 <= panel < k_paths and bool(record.get("valid", False)):
            valid[panel] = True
    if not valid.any():
        result["reason"] = "no_valid_local_fixes"
        return result

    c0 = float(scene["c0"])
    pseudorange_m = c0 * tau_hat - d_rb
    range_m = ranges[:, 0]
    offset_m = pseudorange_m - range_m
    result["per_panel_offset_m"] = offset_m.copy()
    if not np.all(np.isfinite(offset_m[valid])):
        result["reason"] = "nonfinite_offsets"
        return result

    sigma_m_sq = c0 * c0 * sigma_tau_sq
    sigma_range_m = float(config.get("stage2_clock_sigma_range_m", 0.12))
    sigma_range_sq = np.full(k_paths, max(sigma_range_m, 1.0e-6) ** 2)
    combined_var = sigma_range_sq + sigma_m_sq

    candidates = offset_m[valid]
    center = float(np.median(candidates))
    mad_scale = float(np.median(np.abs(candidates - center))) * 1.4826
    model_scale = float(np.sqrt(np.mean(combined_var[valid])))
    scale = max(mad_scale, model_scale, 1.0e-6)
    kappa = float(config.get("stage2_clock_outlier_kappa", 3.0))
    inlier = valid & (np.abs(offset_m - center) <= kappa * scale)
    if not inlier.any():
        inlier = valid.copy()

    weights = 1.0 / combined_var
    clock_offset_m = float(np.sum(weights[inlier] * offset_m[inlier]) / np.sum(weights[inlier]))
    clock_s = clock_offset_m / c0
    if not np.isfinite(clock_s):
        result["reason"] = "nonfinite_clock"
        return result

    bounds = config.get("delta_t_bounds")
    if bounds is not None:
        lower, upper = float(bounds[0]), float(bounds[1])
        if not (lower <= clock_s <= upper):
            result["reason"] = "clock_out_of_bounds"
            return result

    result.update(
        {
            "available": True,
            "clock_s": clock_s,
            "clock_offset_m": clock_offset_m,
            "inlier_mask": inlier,
            "robust_scale_m": scale,
            "num_inliers": int(inlier.sum()),
        }
    )
    return result


def _terms(
    position: np.ndarray,
    state: Stage2CommonState,
    scene: dict,
    config: dict,
) -> dict:
    position = np.asarray(position, dtype=float).reshape(3)
    centers = np.asarray(scene["ris_centers"], dtype=float)
    d_rb = np.asarray(scene["d_RB"], dtype=float)
    c0 = float(scene["c0"])
    q = state.tau_hat_s - (np.linalg.norm(position[None, :] - centers, axis=1) + d_rb) / c0
    sigma_sq = np.asarray(state.sigma_tau_sq_s2, dtype=float)
    w_tau = 1.0 / (sigma_sq + 1.0e-30)
    weight_sum = float(np.sum(w_tau))
    if not np.all(np.isfinite(q)) or not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return {key: float("nan") for key in (
            "clock_term_raw_s2", "clock_term_normalized", "ris_term_raw",
            "ris_term_mean", "ris_term_normalized", "phi_stage2_normalized", "clock_s"
        )}
    q_bar = float(np.sum(w_tau * q) / weight_sum)
    clock_raw = float(np.sum(w_tau * (q - q_bar) ** 2))
    clock_norm = clock_raw / max(int(scene["K"]) - 1, 1)
    ris_diag = robust_jnpp_geometry_consistency_score(position, state.refined_estimate, scene, config)
    ris_raw = float(ris_diag.get("score", np.nan)) if ris_diag.get("available", False) else float("nan")
    ris_weights = np.asarray(ris_diag.get("weights", []), dtype=float).reshape(-1)
    eta_sum = (
        float(np.sum(ris_weights))
        if ris_weights.size == int(scene["K"]) and np.all(np.isfinite(ris_weights)) and np.all(ris_weights > 0.0)
        else float(scene["K"])
    )
    ris_mean = ris_raw / (eta_sum + 1.0e-30) if np.isfinite(ris_raw) else float("nan")
    scale = float(config.get("stage2_ris_normalization_scale", 1.0e-4))
    ris_norm = ris_mean / (scale + 1.0e-30) if np.isfinite(ris_mean) and scale > 0.0 else float("nan")
    lam = float(config.get("stage2_lambda_ris_normalized", 1.0))
    phi = clock_norm + lam * ris_norm if np.isfinite(clock_norm) and np.isfinite(ris_norm) else float("nan")
    return {
        "clock_term_raw_s2": clock_raw,
        "clock_term_normalized": clock_norm,
        "ris_term_raw": ris_raw,
        "ris_term_mean": ris_mean,
        "ris_term_normalized": ris_norm,
        "phi_stage2_normalized": phi,
        "clock_s": q_bar,
    }


def polish_stage2_seed(
    seed_position: np.ndarray,
    seed_clock_s: float,
    state: Stage2CommonState,
    scene: dict,
    config: dict,
    *,
    geometry_seed_impl: str,
) -> dict:
    """Apply the one shared bounded normalized-objective polish."""
    polish_start = time.perf_counter()
    bounds = _finite_array(config.get("ue_bounds"), (3, 2))
    seed_position = np.asarray(seed_position, dtype=float).reshape(3)
    if bounds is not None:
        projected = np.clip(seed_position, bounds[:, 0], bounds[:, 1])
    else:
        projected = seed_position.copy()
    projection_distance = float(np.linalg.norm(seed_position - projected))
    diagnostics = {
        "geometry_seed_impl": geometry_seed_impl,
        "seed_position": seed_position.copy(),
        "seed_clock_s": float(seed_clock_s),
        "projected_position": projected.copy(),
        "pllg_projection_distance_m": projection_distance,
        "polish_accepted": False,
    }
    if not np.all(np.isfinite(projected)) or not np.isfinite(seed_clock_s):
        diagnostics["polish_failure_reason"] = "invalid_projected_seed"
        diagnostics.update({f"before_{key}": float("nan") for key in (
            "clock_term_raw_s2", "clock_term_normalized", "ris_term_raw",
            "ris_term_mean", "ris_term_normalized", "phi_stage2_normalized"
        )})
        diagnostics.update({f"after_{key}": float("nan") for key in (
            "clock_term_raw_s2", "clock_term_normalized", "ris_term_raw",
            "ris_term_mean", "ris_term_normalized", "phi_stage2_normalized"
        )})
        diagnostics["polish_runtime_s"] = float(time.perf_counter() - polish_start)
        return {
            "position": projected,
            "clock_s": float("nan"),
            "rescue_available": False,
            "diagnostics": diagnostics,
            "runtime_s": diagnostics["polish_runtime_s"],
        }
    before = _terms(projected, state, scene, config)
    for key in ("clock_term_raw_s2", "clock_term_normalized", "ris_term_raw", "ris_term_mean", "ris_term_normalized", "phi_stage2_normalized"):
        diagnostics[f"before_{key}"] = before[key]
    position = projected.copy()
    after = before
    try:
        from scipy.optimize import minimize

        polish_bounds = None if bounds is None else [(float(row[0]), float(row[1])) for row in bounds]
        result = minimize(
            lambda value: _terms(value, state, scene, config)["phi_stage2_normalized"],
            projected,
            method="L-BFGS-B",
            bounds=polish_bounds,
            options={"ftol": 1.0e-12, "gtol": 1.0e-8},
        )
        candidate = np.asarray(result.x, dtype=float)
        candidate_terms = _terms(candidate, state, scene, config)
        tolerance = 1.0e-10 * max(1.0, abs(float(before["phi_stage2_normalized"])))
        diagnostics["polish_optimizer_success"] = bool(result.success)
        accepted = bool(
            np.all(np.isfinite(candidate))
            and np.isfinite(candidate_terms["phi_stage2_normalized"])
            and candidate_terms["phi_stage2_normalized"] <= before["phi_stage2_normalized"] + tolerance
        )
        if accepted:
            position = candidate
            after = candidate_terms
            diagnostics["polish_accepted"] = True
        else:
            diagnostics["polish_failure_reason"] = "polish_failure"
    except (ImportError, TypeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        diagnostics["polish_failure_reason"] = "polish_failure"
    for key in ("clock_term_raw_s2", "clock_term_normalized", "ris_term_raw", "ris_term_mean", "ris_term_normalized", "phi_stage2_normalized"):
        diagnostics[f"after_{key}"] = after[key]

    # Phi_S2 keeps its own closed-form clock (the weighted mean of q) so the
    # objective is unchanged.  Only the clock exported as the raw-domain VP
    # initializer may switch to the position-free decoupled estimator, which is
    # robust to a single corrupted delay.
    clock_s = float(after["clock_s"])
    estimator = str(config.get("stage2_clock_estimator", "decoupled_robust")).lower()
    if estimator not in {"decoupled_robust", "weighted_mean"}:
        raise ValueError(
            "stage2_clock_estimator must be 'decoupled_robust' or 'weighted_mean'"
        )
    diagnostics["clock_estimator"] = estimator
    diagnostics["clock_weighted_mean_s"] = clock_s
    diagnostics["clock_decoupled_s"] = float("nan")
    diagnostics["clock_decoupled_available"] = False
    diagnostics["clock_decoupled_reason"] = ""
    diagnostics["clock_decoupled_num_inliers"] = 0
    diagnostics["clock_decoupled_scale_m"] = float("nan")
    if estimator == "decoupled_robust":
        decoupled = decoupled_clock_estimate(state, scene, config)
        diagnostics["clock_decoupled_s"] = float(decoupled["clock_s"])
        diagnostics["clock_decoupled_available"] = bool(decoupled["available"])
        diagnostics["clock_decoupled_reason"] = str(decoupled["reason"])
        diagnostics["clock_decoupled_num_inliers"] = int(decoupled["num_inliers"])
        diagnostics["clock_decoupled_scale_m"] = float(decoupled["robust_scale_m"])
        if decoupled["available"]:
            clock_s = float(decoupled["clock_s"])

    diagnostics["final_clock_s"] = clock_s
    diagnostics["rescue_available"] = bool(np.all(np.isfinite(position)) and np.isfinite(clock_s))
    diagnostics["pllg_projected_x_m"] = float(projected[0])
    diagnostics["pllg_projected_y_m"] = float(projected[1])
    diagnostics["pllg_projected_z_m"] = float(projected[2])
    diagnostics["pllg_phi_before_polish"] = before["phi_stage2_normalized"]
    diagnostics["pllg_phi_after_polish"] = after["phi_stage2_normalized"]
    diagnostics["pllg_polish_success"] = bool(diagnostics["polish_accepted"])
    diagnostics["polish_runtime_s"] = float(time.perf_counter() - polish_start)
    return {
        "position": position,
        "clock_s": clock_s,
        "rescue_available": bool(diagnostics["rescue_available"]),
        "diagnostics": diagnostics,
        "runtime_s": diagnostics["polish_runtime_s"],
    }


def solve_stage2_pllg(state: Stage2CommonState, scene: dict, config: dict) -> dict:
    """Compute the PLLG geometry seed, then use the common Stage-II polish."""
    total_start = time.perf_counter()
    k_paths = int(scene["K"])
    diagnostics: dict[str, Any] = {
        "pllg_success": False,
        "pllg_failure_reason": "",
        "pllg_reweight_steps": int(config.get("stage2_pllg_reweight_steps", 1)),
        "pllg_pseudorange_block_weight": float(config.get("stage2_pllg_pseudorange_block_weight", 1.0)),
        "pllg_rank": 0,
        "pllg_condition_number": float("inf"),
        "pllg_matrix_shape": (0, 4),
        "local_fix_records": state.local_fix_records,
    }
    alpha = float(config.get("stage2_pllg_pseudorange_block_weight", 1.0))
    steps = int(config.get("stage2_pllg_reweight_steps", 1))
    if steps not in (0, 1):
        diagnostics["pllg_failure_reason"] = "linear_solve_failure"
        return {"position": np.full(3, np.nan), "clock_s": np.nan, "rescue_available": False, "failure_reason": "linear_solve_failure", "diagnostics": diagnostics, "runtime_s": time.perf_counter() - total_start}
    valid_records = [record for record in state.local_fix_records if record.get("valid", False)]
    if alpha < 0.0:
        diagnostics["pllg_failure_reason"] = "linear_solve_failure"
        return {"position": np.full(3, np.nan), "clock_s": np.nan, "rescue_available": False, "diagnostics": diagnostics, "runtime_s": time.perf_counter() - total_start}
    if not valid_records:
        diagnostics["pllg_failure_reason"] = "no_valid_local_fixes"
        return {"position": np.full(3, np.nan), "clock_s": np.nan, "rescue_available": False, "diagnostics": diagnostics, "runtime_s": time.perf_counter() - total_start}
    local_positions = np.asarray([record["position"] for record in valid_records], dtype=float)
    local_position = np.mean(local_positions, axis=0)
    seed_position = local_position.copy()
    seed_clock = float("nan")
    if alpha > 0.0:
        if k_paths < 2:
            diagnostics["pllg_failure_reason"] = "insufficient_paths"
        else:
            try:
                centers = np.asarray(scene["ris_centers"], dtype=float)
                d_rb = np.asarray(scene["d_RB"], dtype=float)
                c0 = float(scene["c0"])
                m = c0 * state.tau_hat_s - d_rb
                sigma_m_sq = c0 * c0 * state.sigma_tau_sq_s2
                b = m * m - np.sum(centers * centers, axis=1) - sigma_m_sq
                q_null = _null_basis_of_ones(k_paths)
                stage1_records = build_local_fix_records(
                    state.stage1_estimate, scene, config, source_stage="stage1"
                )
                stage1_positions = np.asarray(
                    [record["position"] for record in stage1_records if record.get("valid", False)],
                    dtype=float,
                )
                p_s1 = (
                    np.mean(stage1_positions, axis=0)
                    if stage1_positions.size
                    else np.mean(local_positions, axis=0)
                )
                rho0 = np.linalg.norm(p_s1[None, :] - centers, axis=1)
                sigma_sq = 4.0 * rho0 * rho0 * sigma_m_sq + 2.0 * sigma_m_sq * sigma_m_sq
                ann_cov = q_null.T @ np.diag(sigma_sq) @ q_null
                w_ann = _whitening(ann_cov)
                a_pr = np.column_stack((-2.0 * centers, 2.0 * m))
                h_pr = np.sqrt(alpha) * (w_ann @ (q_null.T @ a_pr))
                y_pr = np.sqrt(alpha) * (w_ann @ (q_null.T @ b))
                h_f = np.zeros((3 * len(valid_records), 4), dtype=float)
                y_f = local_positions.reshape(-1)
                h_f[:, :3] = np.tile(np.eye(3), (len(valid_records), 1))
                h = np.vstack([h_pr, h_f])
                y = np.concatenate([y_pr, y_f])
                solution, _, _, _ = np.linalg.lstsq(h, y, rcond=None)
                if steps not in (0, 1):
                    raise ValueError("stage2_pllg_reweight_steps must be 0 or 1")
                if steps == 1:
                    rho1 = np.linalg.norm(solution[:3][None, :] - centers, axis=1)
                    sigma_sq_1 = 4.0 * rho1 * rho1 * sigma_m_sq + 2.0 * sigma_m_sq * sigma_m_sq
                    w_ann_1 = _whitening(q_null.T @ np.diag(sigma_sq_1) @ q_null)
                    h_pr = np.sqrt(alpha) * (w_ann_1 @ (q_null.T @ a_pr))
                    y_pr = np.sqrt(alpha) * (w_ann_1 @ (q_null.T @ b))
                    h = np.vstack([h_pr, h_f])
                    y = np.concatenate([y_pr, y_f])
                    solution, _, _, _ = np.linalg.lstsq(h, y, rcond=None)
                diagnostics["pllg_matrix_shape"] = tuple(int(value) for value in h.shape)
                diagnostics["pllg_rank"] = int(np.linalg.matrix_rank(h))
                diagnostics["pllg_condition_number"] = float(np.linalg.cond(h))
                if diagnostics["pllg_rank"] != 4:
                    diagnostics["pllg_failure_reason"] = "rank_deficient"
                elif not np.isfinite(diagnostics["pllg_condition_number"]) or diagnostics["pllg_condition_number"] > float(config.get("stage2_pllg_cond_max", 1.0e12)):
                    diagnostics["pllg_failure_reason"] = "ill_conditioned"
                elif not np.all(np.isfinite(solution)):
                    diagnostics["pllg_failure_reason"] = "nonfinite_linear_solution"
                else:
                    seed_position = np.asarray(solution[:3], dtype=float)
                    diagnostics["pllg_linear_s_m"] = float(solution[3])
                    diagnostics["pllg_linear_clock_s"] = float(solution[3] / c0)
            except np.linalg.LinAlgError:
                diagnostics["pllg_failure_reason"] = "whitening_failure"
            except (KeyError, TypeError, ValueError, FloatingPointError):
                diagnostics["pllg_failure_reason"] = "linear_solve_failure"
    if alpha == 0.0 or not diagnostics["pllg_failure_reason"]:
        centers = np.asarray(scene["ris_centers"], dtype=float)
        d_rb = np.asarray(scene["d_RB"], dtype=float)
        ranges = np.linalg.norm(seed_position[None, :] - centers, axis=1)
        q = state.tau_hat_s - (ranges + d_rb) / float(scene["c0"])
        w_tau = 1.0 / (state.sigma_tau_sq_s2 + 1.0e-30)
        seed_clock = float(np.sum(w_tau * q) / np.sum(w_tau))
    diagnostics["pllg_linear_x_m"] = np.asarray(seed_position, dtype=float).copy()
    diagnostics["pllg_linear_runtime_s"] = float(time.perf_counter() - total_start)
    bounds = _finite_array(config.get("ue_bounds"), (3, 2))
    if bounds is not None and np.all(np.isfinite(seed_position)):
        projected = np.clip(seed_position, bounds[:, 0], bounds[:, 1])
        projection_distance = float(np.linalg.norm(seed_position - projected))
        diagnostics["pllg_projection_distance_m"] = projection_distance
        if projection_distance > float(config.get("stage2_pllg_max_projection_distance_m", 0.05)):
            diagnostics["pllg_failure_reason"] = "invalid_projected_seed"
            diagnostics["pllg_total_runtime_s"] = float(time.perf_counter() - total_start)
            return {
                "position": projected,
                "clock_s": float("nan"),
                "rescue_available": False,
                "failure_reason": "invalid_projected_seed",
                "diagnostics": diagnostics,
                "runtime_s": diagnostics["pllg_total_runtime_s"],
            }
    polished = polish_stage2_seed(seed_position, seed_clock, state, scene, config, geometry_seed_impl="pllg")
    diagnostics.update(polished["diagnostics"])
    diagnostics["pllg_success"] = not bool(diagnostics.get("pllg_failure_reason")) and bool(polished.get("rescue_available", False))
    diagnostics["pllg_total_runtime_s"] = float(time.perf_counter() - total_start)
    return {
        "position": polished["position"],
        "clock_s": polished["clock_s"],
        "rescue_available": bool(polished.get("rescue_available", False)) and not bool(diagnostics.get("pllg_failure_reason")),
        "failure_reason": str(diagnostics.get("pllg_failure_reason", "")),
        "diagnostics": diagnostics,
        "runtime_s": diagnostics["pllg_total_runtime_s"],
    }
