"""Common utilities for standalone benchmark baselines.

The helpers in this module deliberately avoid the proposed JNPP gate and
continuous raw-domain VP refinement.  Baselines may use discrete dictionaries,
linear least-squares over selected atoms, and neutral geometry least-squares
post-processing.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..channel_model import channel_components, synthesize_raw_tensor
from ..geometry import (
    elev_az_from_unit_vector,
    far_field_ris_response,
    local_geometry_from_position,
    maxwell_matrix,
    near_field_spherical_response,
    unit_vector_from_elev_az,
)
from ..metrics import position_error, relative_nmse
from ..utils import scipy_is_available


@dataclass
class BaselineResult:
    name: str
    p_u: np.ndarray | None
    delta_t: float | None
    Y_hat: np.ndarray | None
    raw_objective_final: float
    components: dict[str, Any] = field(default_factory=dict)
    selected_support: list[dict[str, Any]] = field(default_factory=list)
    runtime_s: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def vectorize_raw_observation(Y: np.ndarray) -> np.ndarray:
    """Vectorize raw-domain observations in the repository's VP ordering."""
    return np.asarray(Y, dtype=complex).reshape(-1)


def hash_array(value: Any) -> str:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(float), decimals=12)
    payload = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha256(payload).hexdigest()[:16]


def y_noisy_hash(data: dict) -> str:
    return hash_array(data["Y_noisy"])


def data_hash(data: dict) -> str:
    scene = data.get("scene", {})
    payload = {
        "Y_noisy": y_noisy_hash(data),
        "K": int(scene.get("K", 0)),
        "receiver_mode": str(scene.get("receiver_mode", "")),
        "noise_variance": _finite_float(data.get("noise_variance")),
        "p_B": hash_array(scene.get("p_B", [])),
        "ris_centers": hash_array(scene.get("ris_centers", [])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return default
    return value_float if np.isfinite(value_float) else default


def _rmse_array(estimate: Any, truth: Any) -> float:
    if estimate is None or truth is None:
        return float("nan")
    estimate_arr = np.asarray(estimate, dtype=float).reshape(-1)
    truth_arr = np.asarray(truth, dtype=float).reshape(-1)
    if estimate_arr.size == 0 or estimate_arr.size != truth_arr.size:
        return float("nan")
    return float(np.linalg.norm(estimate_arr - truth_arr) / np.sqrt(estimate_arr.size))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _nested_get(container: Any, path: tuple[str, ...], default: Any = "") -> Any:
    current = container
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def proposed_trace_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """Return lightweight branch/NGC diagnostics for proposed-method CSV rows."""
    final = result.get("final", {}) if isinstance(result, dict) else {}
    reliability = (
        result.get("reliability")
        if isinstance(result.get("reliability"), dict)
        else final.get("reliability") if isinstance(final.get("reliability"), dict) else {}
    )
    ngc_rescue_requested = result.get(
        "ngc_rescue_requested", final.get("ngc_rescue_requested", "")
    )
    return {
        "selected_branch": result.get(
            "selected_branch", final.get("selected_branch", "")
        ),
        "proposed_stage2_policy": reliability.get(
            "proposed_stage2_policy",
            _nested_get(result, ("stage1_config", "proposed_stage2_policy"), ""),
        ),
        "ngc_policy_active": result.get(
            "ngc_policy_active", final.get("ngc_policy_active", "")
        ),
        "ngc_rescue_requested": ngc_rescue_requested,
        "rescue_requested": ngc_rescue_requested,
        "ngc_selected_by": result.get(
            "ngc_selected_by", final.get("ngc_selected_by", "")
        ),
        "ngc_final_unreliable": result.get(
            "ngc_final_unreliable", final.get("ngc_final_unreliable", "")
        ),
        "ngc_direct_cert_status": result.get("ngc_direct_cert_status", ""),
        "ngc_direct_cert_reason": result.get("ngc_direct_cert_reason", ""),
        "ngc_direct_position_boundary_hit": result.get(
            "ngc_direct_position_boundary_hit", ""
        ),
        "ngc_direct_position_boundary_axis": result.get(
            "ngc_direct_position_boundary_axis", ""
        ),
        "ngc_direct_position_boundary_distance_m": result.get(
            "ngc_direct_position_boundary_distance_m", ""
        ),
        "ngc_direct_stage1_displacement_m": result.get(
            "ngc_direct_stage1_displacement_m", ""
        ),
        "ngc_direct_normalized_position_step": result.get(
            "ngc_direct_normalized_position_step", ""
        ),
        "global_vp_backend": final.get("global_vp_backend", ""),
        "global_vp_gpu_used": final.get("global_vp_gpu_used", ""),
        "global_vp_gpu_device": final.get("global_vp_gpu_device", ""),
        "global_vp_objective_backend": final.get(
            "global_vp_objective_backend", ""
        ),
        "global_vp_linear_solve_backend": final.get(
            "global_vp_linear_solve_backend",
            final.get("global_vp_lstsq_backend", ""),
        ),
        "vp_dictionary_mode": final.get("vp_dictionary_mode", ""),
        "vp_dictionary_mode_requested": final.get("vp_dictionary_mode_requested", ""),
        "vp_jacobian_mode": final.get("vp_jacobian_mode", ""),
        "vp_matrix_free_enabled": final.get("vp_matrix_free_enabled", ""),
        "vp_matrix_free_fallback_reason": final.get(
            "vp_matrix_free_fallback_reason", ""
        ),
        "vp_precontract_static_modes": final.get("vp_precontract_static_modes", ""),
        "vp_factor_cache_hits": final.get("vp_factor_cache_hits", ""),
        "vp_factor_cache_misses": final.get("vp_factor_cache_misses", ""),
        "vp_matrix_free_num_objective_calls": final.get(
            "vp_matrix_free_num_objective_calls", ""
        ),
        "vp_matrix_free_debug_num_compares": final.get(
            "vp_matrix_free_debug_num_compares", ""
        ),
        "vp_matrix_free_debug_rel_G_diff": final.get(
            "vp_matrix_free_debug_rel_G_diff", ""
        ),
        "vp_matrix_free_debug_rel_b_diff": final.get(
            "vp_matrix_free_debug_rel_b_diff", ""
        ),
        "vp_matrix_free_debug_rel_x_hat_diff": final.get(
            "vp_matrix_free_debug_rel_x_hat_diff", ""
        ),
        "vp_matrix_free_debug_rel_regularized_objective_diff": final.get(
            "vp_matrix_free_debug_rel_regularized_objective_diff", ""
        ),
        "vp_matrix_free_debug_rel_gradient_diff": final.get(
            "vp_matrix_free_debug_rel_gradient_diff", ""
        ),
    }


def make_baseline_row(
    result: BaselineResult,
    data: dict,
    config: dict,
    *,
    baseline: str | None = None,
    trial_id: int = 0,
    seed: int | None = None,
    snr_db: float | None = None,
    failed: bool = False,
    error: str = "",
    warning: str = "",
) -> dict[str, Any]:
    """Convert a baseline result to one benchmark CSV row."""
    scene = data.get("scene", {})
    true_components = data.get("true_components", {})
    y_true = data.get("Y_true")
    y_hat = result.Y_hat
    p_true = scene.get("p_u_true")
    p_hat = result.p_u
    if y_hat is not None and y_true is not None and y_hat.shape == y_true.shape:
        y_nmse = relative_nmse(y_hat, y_true)
        raw_objective = float(
            np.linalg.norm(vectorize_raw_observation(y_hat - data["Y_noisy"])) ** 2
            / max(np.size(data["Y_noisy"]), 1)
        )
    else:
        y_nmse = float("nan")
        raw_objective = _finite_float(result.raw_objective_final)
    if np.isfinite(_finite_float(result.raw_objective_final)):
        raw_objective = _finite_float(result.raw_objective_final)
    selected_support = result.selected_support or []
    diagnostics = result.diagnostics or {}
    row_warning = warning or str(diagnostics.get("warning", ""))
    return {
        "baseline": baseline or result.name,
        "trial_id": int(trial_id),
        "seed": int(seed if seed is not None else config.get("seed", 0)),
        "snr_db": float(snr_db if snr_db is not None else config.get("SNR_dB", float("nan"))),
        "K": int(scene.get("K", config.get("K", 0))),
        "data_hash": data_hash(data),
        "y_noisy_hash": y_noisy_hash(data),
        "failed": bool(failed),
        "error": str(error),
        "runtime_s": float(result.runtime_s),
        "position_error_m": (
            position_error(
                np.asarray(p_hat, dtype=float), np.asarray(p_true, dtype=float)
            )
            if p_hat is not None and p_true is not None
            else float("nan")
        ),
        # Retained for compatibility with pre-2026 benchmark trial CSVs.
        "position_rmse_m": (
            position_error(
                np.asarray(p_hat, dtype=float), np.asarray(p_true, dtype=float)
            )
            if p_hat is not None and p_true is not None
            else float("nan")
        ),
        "y_nmse": y_nmse,
        "range_rmse_m": _rmse_array(result.components.get("ranges"), true_components.get("ranges")),
        "tau_rmse_s": _rmse_array(result.components.get("taus"), true_components.get("taus")),
        "raw_objective_final": raw_objective,
        "support_size": len(selected_support),
        "grid_size": diagnostics.get("grid_size", ""),
        "dictionary_mode": diagnostics.get("dictionary_mode", ""),
        "group_omp": diagnostics.get("group_omp", False),
        "offgrid_refinement": diagnostics.get("offgrid_refinement", False),
        "refinement_objective": diagnostics.get("refinement_objective", ""),
        "model_variant": diagnostics.get("model_variant", ""),
        "selected_support": json.dumps(_jsonable(selected_support), separators=(",", ":")),
        "selected_group_count": diagnostics.get("selected_group_count", ""),
        "selected_panel_count": diagnostics.get("selected_panel_count", ""),
        "selected_panels": json.dumps(
            _jsonable(diagnostics.get("selected_panels", "")), separators=(",", ":")
        ),
        "unique_panel_constraint": diagnostics.get(
            "unique_panel_constraint", ""
        ),
        "expanded_support_count": diagnostics.get("expanded_support_count", ""),
        "active_coefficient_count": diagnostics.get(
            "active_coefficient_count", ""
        ),
        "active_panel_count": diagnostics.get("active_panel_count", ""),
        "als_geometry_mapping": diagnostics.get("als_geometry_mapping", ""),
        "als_geometry_assignment": json.dumps(
            _jsonable(diagnostics.get("als_geometry_assignment", "")),
            separators=(",", ":"),
        ),
        "als_geometry_unique_panel_count": diagnostics.get(
            "als_geometry_unique_panel_count", ""
        ),
        "als_geometry_coarse_score": diagnostics.get(
            "als_geometry_coarse_score", ""
        ),
        "als_geometry_refined_score": diagnostics.get(
            "als_geometry_refined_score", ""
        ),
        "als_geometry_refined_factor_score": diagnostics.get(
            "als_geometry_refined_factor_score", ""
        ),
        "als_geometry_refined_clock_std_ns": diagnostics.get(
            "als_geometry_refined_clock_std_ns", ""
        ),
        "als_geometry_refinement_used": diagnostics.get(
            "als_geometry_refinement_used", ""
        ),
        "als_geometry_refinement_success": diagnostics.get(
            "als_geometry_refinement_success", ""
        ),
        "als_geometry_refinement_evals": diagnostics.get(
            "als_geometry_refinement_evals", ""
        ),
        "peb_position_m": _finite_float(diagnostics.get("peb_position_m")),
        "peb_is_data_only": diagnostics.get("peb_is_data_only", ""),
        "peb_uses_regularization": diagnostics.get("peb_uses_regularization", ""),
        "nuisance_model": diagnostics.get("nuisance_model", ""),
        "clock_eliminated": diagnostics.get("clock_eliminated", ""),
        "efim_condition_number": diagnostics.get("efim_condition_number", ""),
        "batch_size": diagnostics.get("batch_size", ""),
        "max_batch_memory_mb": diagnostics.get("max_batch_memory_mb", ""),
        "num_batches": diagnostics.get("num_batches", ""),
        "baseline_backend": diagnostics.get("backend", "cpu"),
        "gpu_used": diagnostics.get("gpu_used", False),
        "gpu_device": diagnostics.get("gpu_device", ""),
        "gpu_num_batches": diagnostics.get("gpu_num_batches", 0),
        "gpu_batch_size": diagnostics.get("gpu_batch_size", ""),
        "cache_enabled": diagnostics.get("cache_enabled", False),
        "cache_hits": diagnostics.get("cache_hits", 0),
        "cache_misses": diagnostics.get("cache_misses", 0),
        "cache_estimated_bytes": diagnostics.get("cache_estimated_bytes", 0),
        "scoring_time_s": diagnostics.get("scoring_time_s", ""),
        "backend_warning": diagnostics.get("backend_warning", ""),
        "factorized_scoring": diagnostics.get("factorized_scoring", False),
        "score_mode": diagnostics.get("score_mode", ""),
        "coarse_backend": diagnostics.get("coarse_backend", ""),
        "coarse_gpu_used": diagnostics.get("coarse_gpu_used", ""),
        "local_refinement_backend": diagnostics.get(
            "local_refinement_backend", ""
        ),
        "local_refinement_gpu_used": diagnostics.get(
            "local_refinement_gpu_used", ""
        ),
        "wls_backend": diagnostics.get("wls_backend", ""),
        "mixed_backend": diagnostics.get("mixed_backend", False),
        "selected_grid_index": json.dumps(_jsonable(diagnostics.get("selected_grid_index", "")), separators=(",", ":")),
        "momp_group_omp_enabled": diagnostics.get("momp_group_omp_enabled", ""),
        "momp_score_mode": diagnostics.get("momp_score_mode", ""),
        "momp_group_size": diagnostics.get("momp_group_size", ""),
        "momp_max_groups": diagnostics.get("momp_max_groups", ""),
        "momp_selected_groups": json.dumps(_jsonable(diagnostics.get("momp_selected_groups", "")), separators=(",", ":")),
        "momp_local_refinement_used": diagnostics.get("momp_local_refinement_used", ""),
        "momp_refinement_levels": diagnostics.get("momp_refinement_levels", ""),
        "momp_refinement_num_evals": diagnostics.get("momp_refinement_num_evals", ""),
        "momp_coordinate_sweeps": diagnostics.get("momp_coordinate_sweeps", ""),
        "momp_coordinate_evaluations": diagnostics.get(
            "momp_coordinate_evaluations", ""
        ),
        "momp_source_competitions": diagnostics.get(
            "momp_source_competitions", ""
        ),
        "cartesian_dictionary_materialized": diagnostics.get(
            "cartesian_dictionary_materialized", ""
        ),
        "range_dictionary_used": diagnostics.get("range_dictionary_used", ""),
        "nf_mmpsr_cc_metric": diagnostics.get("nf_mmpsr_cc_metric", ""),
        "nf_mmpsr_top_candidates": diagnostics.get("nf_mmpsr_top_candidates", ""),
        "nf_mmpsr_local_refinement_used": diagnostics.get("nf_mmpsr_local_refinement_used", ""),
        "nf_mmpsr_refinement_levels": diagnostics.get("nf_mmpsr_refinement_levels", ""),
        "nf_mmpsr_refinement_num_evals": diagnostics.get("nf_mmpsr_refinement_num_evals", ""),
        "nf_mmpsr_coarse_best_score": diagnostics.get("nf_mmpsr_coarse_best_score", ""),
        "nf_mmpsr_refined_best_score": diagnostics.get("nf_mmpsr_refined_best_score", ""),
        "reference_algorithm": diagnostics.get("reference_algorithm", ""),
        "cpd_omp_adapted_used": diagnostics.get("cpd_omp_adapted_used", ""),
        "cpd_rank1_sequential": diagnostics.get("cpd_rank1_sequential", ""),
        "near_field_l1_refinement_used": diagnostics.get("near_field_l1_refinement_used", ""),
        "sage_enabled": diagnostics.get("sage_enabled", ""),
        "sage_iterations": diagnostics.get("sage_iterations", ""),
        "sage_num_evals": diagnostics.get("sage_num_evals", ""),
        "local_grid_enabled": diagnostics.get("local_grid_enabled", ""),
        "local_grid_iterations": diagnostics.get("local_grid_iterations", ""),
        "local_grid_num_evals": diagnostics.get("local_grid_num_evals", ""),
        "wls_enabled": diagnostics.get("wls_enabled", ""),
        "wls_final_cost": diagnostics.get("wls_final_cost", ""),
        "wls_weight_model": diagnostics.get("wls_weight_model", ""),
        "subris_mode": diagnostics.get("subris_mode", ""),
        "subris_shape": json.dumps(_jsonable(diagnostics.get("subris_shape", "")), separators=(",", ":")),
        "subris_fallback_used": diagnostics.get("subris_fallback_used", ""),
        "adaptation_note": diagnostics.get("adaptation_note", ""),
        "selected_branch": diagnostics.get("selected_branch", ""),
        "proposed_stage2_policy": diagnostics.get("proposed_stage2_policy", ""),
        "ngc_policy_active": diagnostics.get("ngc_policy_active", ""),
        "ngc_rescue_requested": diagnostics.get("ngc_rescue_requested", ""),
        "rescue_requested": diagnostics.get("rescue_requested", ""),
        "ngc_selected_by": diagnostics.get("ngc_selected_by", ""),
        "ngc_final_unreliable": diagnostics.get("ngc_final_unreliable", ""),
        "ngc_direct_cert_status": diagnostics.get("ngc_direct_cert_status", ""),
        "ngc_direct_cert_reason": diagnostics.get("ngc_direct_cert_reason", ""),
        "ngc_direct_position_boundary_hit": diagnostics.get(
            "ngc_direct_position_boundary_hit", ""
        ),
        "ngc_direct_position_boundary_axis": diagnostics.get(
            "ngc_direct_position_boundary_axis", ""
        ),
        "ngc_direct_position_boundary_distance_m": diagnostics.get(
            "ngc_direct_position_boundary_distance_m", ""
        ),
        "ngc_direct_stage1_displacement_m": diagnostics.get(
            "ngc_direct_stage1_displacement_m", ""
        ),
        "ngc_direct_normalized_position_step": diagnostics.get(
            "ngc_direct_normalized_position_step", ""
        ),
        "vp_dictionary_mode": diagnostics.get("vp_dictionary_mode", ""),
        "vp_dictionary_mode_requested": diagnostics.get(
            "vp_dictionary_mode_requested", ""
        ),
        "vp_jacobian_mode": diagnostics.get("vp_jacobian_mode", ""),
        "vp_matrix_free_enabled": diagnostics.get("vp_matrix_free_enabled", ""),
        "vp_matrix_free_fallback_reason": diagnostics.get(
            "vp_matrix_free_fallback_reason", ""
        ),
        "vp_precontract_static_modes": diagnostics.get(
            "vp_precontract_static_modes", ""
        ),
        "vp_factor_cache_hits": diagnostics.get("vp_factor_cache_hits", ""),
        "vp_factor_cache_misses": diagnostics.get("vp_factor_cache_misses", ""),
        "vp_matrix_free_num_objective_calls": diagnostics.get(
            "vp_matrix_free_num_objective_calls", ""
        ),
        "vp_matrix_free_debug_num_compares": diagnostics.get(
            "vp_matrix_free_debug_num_compares", ""
        ),
        "vp_matrix_free_debug_rel_G_diff": diagnostics.get(
            "vp_matrix_free_debug_rel_G_diff", ""
        ),
        "vp_matrix_free_debug_rel_b_diff": diagnostics.get(
            "vp_matrix_free_debug_rel_b_diff", ""
        ),
        "vp_matrix_free_debug_rel_x_hat_diff": diagnostics.get(
            "vp_matrix_free_debug_rel_x_hat_diff", ""
        ),
        "vp_matrix_free_debug_rel_regularized_objective_diff": diagnostics.get(
            "vp_matrix_free_debug_rel_regularized_objective_diff", ""
        ),
        "vp_matrix_free_debug_rel_gradient_diff": diagnostics.get(
            "vp_matrix_free_debug_rel_gradient_diff", ""
        ),
        "warning": row_warning,
    }


def linear_ls_fit(Phi: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust complex linear least-squares fit."""
    Phi = np.asarray(Phi, dtype=complex)
    y = np.asarray(y, dtype=complex).reshape(-1)
    if Phi.ndim == 1:
        Phi = Phi[:, None]
    if Phi.shape[0] != y.size:
        raise ValueError("Phi row count must match y length")
    if Phi.shape[1] == 0:
        y_hat = np.zeros_like(y)
        return np.zeros(0, dtype=complex), y_hat, y - y_hat
    ridge_value = float(ridge)
    if ridge_value > 0.0:
        gram = Phi.conj().T @ Phi
        rhs = Phi.conj().T @ y
        coeffs = np.linalg.solve(
            gram + ridge_value * np.eye(gram.shape[0], dtype=complex),
            rhs,
        )
    else:
        coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    y_hat = Phi @ coeffs
    return coeffs, y_hat, y - y_hat


def simple_atom_normalize(atom: np.ndarray) -> np.ndarray:
    atom = np.asarray(atom, dtype=complex).reshape(-1)
    norm = np.linalg.norm(atom)
    if not np.isfinite(norm) or norm <= 0.0:
        return atom
    return atom / norm


def build_jones_basis_evs_atoms(scene: dict, config: dict, path_index: int | None = None, panel_index: int | None = None) -> tuple[list[np.ndarray], list[str]]:
    """Return EVS basis responses for Jones basis vectors e1/e2."""
    _ = config
    k = int(panel_index if panel_index is not None else path_index if path_index is not None else 0)
    warnings: list[str] = []
    try:
        theta = np.asarray(scene["Theta"][k], dtype=complex)
        v_b = np.asarray(scene["v_B"][k], dtype=complex)
        mask = np.asarray(scene.get("evs_observation_mask", np.ones(6 * v_b.size)), dtype=float)
        atoms = [np.kron(v_b, theta[:, j]) * mask for j in range(2)]
    except Exception as exc:  # noqa: BLE001 - fallback for synthetic mocks.
        i_dim = int(scene.get("I", 1))
        atoms = [np.ones(i_dim, dtype=complex)]
        warnings.append(f"evs_jones_basis_fallback: {type(exc).__name__}: {exc}")
    return atoms, warnings


def delay_response(scene: dict, tau: float) -> np.ndarray:
    n_idx = np.arange(int(scene["N"]), dtype=float)
    pole = np.exp(-1j * 2.0 * np.pi * float(scene["delta_f"]) * float(tau))
    return pole ** n_idx


def training_response_from_position(scene: dict, panel: int, p_u: np.ndarray, *, near_field: bool = True) -> np.ndarray:
    panel = int(panel)
    if near_field:
        range_m, elevation, azimuth, _ = local_geometry_from_position(
            np.asarray(p_u, dtype=float),
            np.asarray(scene["ris_centers"][panel], dtype=float),
            np.asarray(scene["rotations"][panel], dtype=float),
        )
        a_ur = near_field_spherical_response(
            range_m,
            elevation,
            azimuth,
            np.asarray(scene["ris_grid"], dtype=float),
            float(scene["wavelength"]),
        )
    else:
        a_ur = far_field_ris_response(
            np.asarray(scene["ris_centers"][panel], dtype=float),
            np.asarray(p_u, dtype=float),
            np.asarray(scene["rotations"][panel], dtype=float),
            np.asarray(scene["ris_grid"], dtype=float),
            float(scene["wavelength"]),
        )
    g_elem = np.asarray(scene["a_RB"][panel], dtype=complex) * a_ur
    return np.asarray(scene["Omega"][panel], dtype=complex) @ g_elem


def training_response_from_direction(scene: dict, panel: int, direction_local: np.ndarray) -> np.ndarray:
    panel = int(panel)
    direction = np.asarray(direction_local, dtype=float).reshape(3)
    direction /= np.linalg.norm(direction) + 1.0e-15
    wavenumber = 2.0 * np.pi / float(scene["wavelength"])
    a_ur = np.exp(-1j * wavenumber * (np.asarray(scene["ris_grid"], dtype=float) @ direction))
    g_elem = np.asarray(scene["a_RB"][panel], dtype=complex) * a_ur
    return np.asarray(scene["Omega"][panel], dtype=complex) @ g_elem


def raw_atom_from_factors(evs: np.ndarray, delay: np.ndarray, training: np.ndarray) -> np.ndarray:
    return (
        np.asarray(evs, dtype=complex)[:, None, None]
        * np.asarray(delay, dtype=complex)[None, :, None]
        * np.asarray(training, dtype=complex)[None, None, :]
    ).reshape(-1)


def raw_atom_from_support(scene: dict, config: dict, support: dict[str, Any]) -> np.ndarray:
    panel = int(support.get("panel", 0))
    pol_index = int(support.get("pol_index", 0))
    evs_atoms, _ = build_jones_basis_evs_atoms(scene, config, panel_index=panel)
    evs = evs_atoms[min(pol_index, len(evs_atoms) - 1)]
    tau = float(support.get("tau", 0.0))
    delay = delay_response(scene, tau)
    if "position" in support:
        training = training_response_from_position(
            scene,
            panel,
            np.asarray(support["position"], dtype=float),
            near_field=bool(support.get("near_field", True)),
        )
    elif "direction" in support:
        training = training_response_from_direction(
            scene,
            panel,
            np.asarray(support["direction"], dtype=float),
        )
    else:
        training = np.ones(int(scene["T"]), dtype=complex)
    return raw_atom_from_factors(evs, delay, training)


def geometry_group_key(support: dict[str, Any]) -> tuple[Any, ...]:
    """Hash a geometry support without its Jones polarization column index."""
    position = support.get("position")
    direction = support.get("direction")
    return (
        int(support.get("panel", 0)),
        tuple(np.round(np.asarray(position, dtype=float).reshape(-1), 8))
        if position is not None
        else None,
        tuple(np.round(np.asarray(direction, dtype=float).reshape(-1), 8))
        if direction is not None
        else None,
        round(float(support.get("range", np.nan)), 8)
        if "range" in support
        else None,
        round(float(support.get("tau", 0.0)), 15),
        bool(support.get("near_field", True)),
    )


def expand_jones_group(support: dict[str, Any], num_polarizations: int = 2) -> list[dict[str, Any]]:
    """Expand one geometry support to Jones-basis raw-domain columns."""
    count = max(1, int(num_polarizations))
    base = {key: copy_value for key, copy_value in support.items() if key != "pol_index"}
    return [{**base, "pol_index": int(pol_index)} for pol_index in range(count)]


def supports_to_design(scene: dict, config: dict, supports: list[dict[str, Any]]) -> np.ndarray:
    atoms = [simple_atom_normalize(raw_atom_from_support(scene, config, support)) for support in supports]
    if not atoms:
        return np.empty((int(scene["I"]) * int(scene["N"]) * int(scene["T"]), 0), dtype=complex)
    return np.column_stack(atoms)


def group_design(
    scene: dict,
    config: dict,
    group: dict[str, Any],
    *,
    normalize: bool = True,
) -> np.ndarray:
    atoms = [
        raw_atom_from_support(scene, config, support)
        for support in expand_jones_group(group, 2)
    ]
    if normalize:
        atoms = [simple_atom_normalize(atom) for atom in atoms]
    return np.column_stack(atoms)


def group_projection_score(
    group_matrix: np.ndarray,
    residual: np.ndarray,
    *,
    rank_tol: float = 1.0e-10,
) -> float:
    """Return ||Q^H r||^2 for the numerically independent group subspace."""
    matrix = np.asarray(group_matrix, dtype=complex)
    residual = np.asarray(residual, dtype=complex).reshape(-1)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[0] != residual.size or matrix.shape[1] == 0:
        return float("-inf")
    columns = []
    for col in range(matrix.shape[1]):
        atom = simple_atom_normalize(matrix[:, col])
        if np.linalg.norm(atom) > 0.0:
            columns.append(atom)
    if not columns:
        return float("-inf")
    matrix = np.column_stack(columns)
    try:
        q, r = np.linalg.qr(matrix, mode="reduced")
        diag = np.abs(np.diag(r)) if r.size else np.array([], dtype=float)
        if diag.size:
            threshold = float(rank_tol) * max(float(np.max(diag)), 1.0)
            rank = int(np.sum(diag > threshold))
            if rank > 0:
                q = q[:, :rank]
                return float(np.linalg.norm(q.conj().T @ residual) ** 2)
    except np.linalg.LinAlgError:
        pass
    try:
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        pinv_fit = matrix @ (np.linalg.pinv(matrix, rcond=float(rank_tol)) @ residual)
        return float(np.linalg.norm(pinv_fit) ** 2)
    if s.size == 0:
        return float("-inf")
    threshold = float(rank_tol) * max(float(np.max(s)), 1.0)
    rank = int(np.sum(s > threshold))
    if rank <= 0:
        pinv_fit = matrix @ (np.linalg.pinv(matrix, rcond=float(rank_tol)) @ residual)
        return float(np.linalg.norm(pinv_fit) ** 2)
    q = u[:, :rank]
    return float(np.linalg.norm(q.conj().T @ residual) ** 2)


def group_omp_select(
    scene: dict,
    config: dict,
    groups: list[dict[str, Any]],
    y_vec: np.ndarray,
    *,
    max_groups: int,
    batch_size: int = 256,
    ridge: float = 1.0e-10,
    trim_memory_enabled: bool = True,
    backend_config: Any | None = None,
    static_cache_key: str | None = None,
    unique_panels: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    """Select groups by exact factorized projection energy and joint LS refits."""
    from .backend import BackendConfig
    from .cache import BASELINE_CACHE
    from .factorized_scoring import (
        FactorizedGroupScorer,
        build_group_factor_context,
        factorized_fit_supports,
    )

    _ = trim_memory_enabled  # Kept for CLI/API compatibility; trimming is method-scoped.
    y_vec = np.asarray(y_vec, dtype=complex).reshape(-1)
    residual = y_vec.copy()
    selected_groups: list[dict[str, Any]] = []
    expanded_supports: list[dict[str, Any]] = []
    coeffs = np.zeros(0, dtype=complex)
    y_hat = np.zeros_like(y_vec)
    last_best_score = float("nan")
    num_batches = 0
    scoring_start = time.perf_counter()
    context_key = (
        f"{static_cache_key}:factorized_group_v1"
        if static_cache_key is not None
        else None
    )
    if context_key is None:
        factor_context = build_group_factor_context(scene, config, groups)
    else:
        factor_context = BASELINE_CACHE.get_or_create(
            context_key,
            lambda: build_group_factor_context(scene, config, groups),
        )
    scorer = FactorizedGroupScorer(factor_context, backend_config)
    backend = scorer.backend
    backend_cfg = BackendConfig.from_value(backend_config)
    requested_batch_size = (
        backend_cfg.gpu_batch_size
        if backend.name == "cupy" and backend_cfg.gpu_batch_size is not None
        else backend_cfg.cpu_batch_size
        if backend.name == "cpu" and backend_cfg.cpu_batch_size is not None
        else batch_size
    )
    effective_batch_size = max(1, int(requested_batch_size))

    for _ in range(max(0, int(max_groups))):
        best_index, best_score = scorer.best(residual)
        num_batches += int(np.ceil(len(groups) / effective_batch_size))
        if best_index < 0 or not np.isfinite(best_score):
            break
        selected_group = dict(groups[best_index])
        selected_groups.append(selected_group)
        selected_key = geometry_group_key(selected_group)
        selected_panel = int(selected_group.get("panel", -1))
        scorer.exclude(
            index
            for index, group in enumerate(groups)
            if geometry_group_key(group) == selected_key
            or (
                unique_panels
                and int(group.get("panel", -2)) == selected_panel
            )
        )
        expanded_supports = [
            support
            for group in selected_groups
            for support in expand_jones_group(group, 2)
        ]
        coeffs, y_hat, residual = factorized_fit_supports(
            scene, config, expanded_supports, y_vec, ridge=ridge
        )
        last_best_score = float(best_score)
    backend.synchronize()

    diagnostics = {
        "group_omp": True,
        "selected_groups": selected_groups,
        "expanded_supports": expanded_supports,
        "selected_group_count": len(selected_groups),
        "selected_panel_count": len(
            {int(group.get("panel", -1)) for group in selected_groups}
        ),
        "selected_panels": [
            int(group.get("panel", -1)) for group in selected_groups
        ],
        "unique_panel_constraint": bool(unique_panels),
        "expanded_support_count": len(expanded_supports),
        "last_best_group_score": last_best_score,
        "score_mode": "factorized_group_projection",
        "factorized_scoring": True,
        "residual_norm": float(np.linalg.norm(residual)),
        "num_batches": num_batches,
        "batch_size": effective_batch_size,
        "scoring_time_s": time.perf_counter() - scoring_start,
        "backend": backend.name,
        "gpu_used": backend.name == "cupy",
        "gpu_num_batches": num_batches if backend.name == "cupy" else 0,
        "gpu_batch_size": effective_batch_size if backend.name == "cupy" else "",
        "gpu_device": backend.device if backend.name == "cupy" else "",
        "backend_warning": backend.warning,
    }
    return selected_groups, expanded_supports, coeffs, y_hat, diagnostics


def reconstruct_from_supports(
    scene: dict,
    config: dict,
    supports: list[dict[str, Any]],
    y_vec: np.ndarray,
    *,
    ridge: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Phi = supports_to_design(scene, config, supports)
    coeffs, y_hat_vec, residual = linear_ls_fit(Phi, y_vec, ridge=ridge)
    return coeffs, y_hat_vec.reshape(scene["I"], scene["N"], scene["T"]), residual, Phi


def position_grid_from_config(config: dict, shape: tuple[int, int, int]) -> list[np.ndarray]:
    bounds = np.asarray(config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]), dtype=float)
    axes = [np.linspace(bounds[idx, 0], bounds[idx, 1], int(shape[idx])) for idx in range(3)]
    return [np.array([x, y, z], dtype=float) for x in axes[0] for y in axes[1] for z in axes[2]]


def clock_grid_from_config(config: dict, size: int) -> np.ndarray:
    bounds = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    return np.linspace(float(bounds[0]), float(bounds[1]), int(size))


def direction_grid(angle_grid_size: int) -> list[np.ndarray]:
    size = max(int(angle_grid_size), 3)
    axis_count = max(3, int(np.ceil(np.sqrt(size))))
    ux_axis = np.linspace(-0.85, 0.85, axis_count)
    uy_axis = np.linspace(-0.85, 0.85, axis_count)
    directions = []
    for ux in ux_axis:
        for uy in uy_axis:
            rem = 1.0 - ux * ux - uy * uy
            if rem <= 0.0:
                continue
            directions.append(np.array([ux, uy, np.sqrt(rem)], dtype=float))
            if len(directions) >= size:
                return directions
    return directions


def delay_grid_from_scene(scene: dict, config: dict, size: int) -> np.ndarray:
    bounds = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    ue_bounds = np.asarray(config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]), dtype=float)
    corners = np.array(
        [[x, y, z] for x in ue_bounds[0] for y in ue_bounds[1] for z in ue_bounds[2]],
        dtype=float,
    )
    taus = []
    for panel in range(int(scene["K"])):
        ranges = np.linalg.norm(corners - np.asarray(scene["ris_centers"][panel]), axis=1)
        taus.extend(((ranges + scene["d_RB"][panel]) / scene["c0"] + bounds[0]).tolist())
        taus.extend(((ranges + scene["d_RB"][panel]) / scene["c0"] + bounds[1]).tolist())
    return np.linspace(float(np.min(taus)), float(np.max(taus)), int(size))


def geometric_support_to_position_ls(scene: dict, supports: list[dict[str, Any]], config: dict) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Estimate UE position/clock from neutral support geometry."""
    direct_positions = [
        np.asarray(support["position"], dtype=float).reshape(3)
        for support in supports
        if "position" in support
    ]
    taus = [float(support["tau"]) for support in supports if "tau" in support]
    if direct_positions:
        p_hat = np.median(np.asarray(direct_positions, dtype=float), axis=0)
        if taus:
            dt_values = []
            for support in supports:
                if "tau" not in support:
                    continue
                panel = int(support.get("panel", 0))
                dist = np.linalg.norm(p_hat - scene["ris_centers"][panel])
                dt_values.append(float(support["tau"]) - (dist + scene["d_RB"][panel]) / scene["c0"])
            delta_t = (
                float(np.median(dt_values))
                if dt_values
                else float(np.mean(clock_grid_from_config(config, 3)))
            )
        else:
            delta_t = float(np.mean(clock_grid_from_config(config, 3)))
        return p_hat, delta_t, {"geometry_solver": "direct_position_candidate"}

    bounds_p = np.asarray(config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]), dtype=float)
    bounds_dt = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    p0 = np.mean(bounds_p, axis=1)
    dt0 = float(np.mean(bounds_dt))
    if supports:
        panel_points = []
        for support in supports:
            if "direction" in support:
                panel = int(support.get("panel", 0))
                direction = np.asarray(support["direction"], dtype=float).reshape(3)
                direction /= np.linalg.norm(direction) + 1.0e-15
                panel_points.append(scene["ris_centers"][panel] + 4.0 * scene["rotations"][panel].T @ direction)
        if panel_points:
            p0 = np.mean(np.asarray(panel_points), axis=0)
            p0 = np.clip(p0, bounds_p[:, 0], bounds_p[:, 1])

    def residual_fn(chi: np.ndarray) -> np.ndarray:
        p = chi[:3]
        dt = float(chi[3])
        residuals: list[float] = []
        for support in supports:
            panel = int(support.get("panel", 0))
            q_local = scene["rotations"][panel] @ (p - scene["ris_centers"][panel])
            rng = np.linalg.norm(q_local) + 1.0e-15
            direction = q_local / rng
            if "direction" in support:
                direction_hat = np.asarray(support["direction"], dtype=float).reshape(3)
                direction_hat /= np.linalg.norm(direction_hat) + 1.0e-15
                residuals.extend((direction - direction_hat).tolist())
            if "tau" in support:
                tau_model = (rng + scene["d_RB"][panel]) / scene["c0"] + dt
                residuals.append((tau_model - float(support["tau"])) / 1.0e-9)
        if not residuals:
            residuals.extend((p - p0).tolist())
        return np.asarray(residuals, dtype=float)

    lower = np.r_[bounds_p[:, 0], bounds_dt[0]]
    upper = np.r_[bounds_p[:, 1], bounds_dt[1]]
    x0 = np.clip(np.r_[p0, dt0], lower, upper)
    if scipy_is_available():
        try:
            from scipy.optimize import least_squares  # type: ignore[import-not-found]

            result = least_squares(
                residual_fn,
                x0,
                bounds=(lower, upper),
                max_nfev=int(config.get("baselines", {}).get("geometry_max_nfev", 100)),
            )
            chi = result.x
            diagnostics = {
                "geometry_solver": "scipy.optimize.least_squares",
                "geometry_cost": float(result.cost),
                "geometry_success": bool(result.success),
            }
            return chi[:3].astype(float), float(chi[3]), diagnostics
        except Exception as exc:  # noqa: BLE001 - fall back to bounded center.
            return p0.astype(float), dt0, {"geometry_solver": "fallback_center", "warning": str(exc)}
    return p0.astype(float), dt0, {"geometry_solver": "fallback_center"}


def supports_from_position_clock(
    scene: dict,
    p_u: np.ndarray,
    delta_t: float,
    panels: list[int] | None = None,
    *,
    model_variant: str = "near_field",
) -> list[dict[str, Any]]:
    """Build one Jones geometry group per panel for a baseline model."""
    p_u = np.asarray(p_u, dtype=float).reshape(3)
    panel_indices = list(range(int(scene["K"]))) if panels is None else [int(p) for p in panels]
    near_field = str(model_variant) != "far_field"
    supports: list[dict[str, Any]] = []
    for panel in panel_indices:
        range_m, elev, az, q_local = local_geometry_from_position(
            p_u,
            np.asarray(scene["ris_centers"][panel], dtype=float),
            np.asarray(scene["rotations"][panel], dtype=float),
        )
        tau = (range_m + scene["d_RB"][panel]) / scene["c0"] + float(delta_t)
        support = {
            "panel": int(panel),
            "tau": float(tau),
            "range": float(range_m),
            "elevation": float(elev),
            "azimuth": float(az),
            "near_field": near_field,
        }
        if near_field:
            support["position"] = p_u.copy()
        else:
            support["direction"] = q_local / (np.linalg.norm(q_local) + 1.0e-15)
        supports.append(support)
    return supports


def fit_position_clock_data_domain(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    initial_position: np.ndarray,
    initial_clock: float,
    *,
    panels: list[int] | None = None,
    model_variant: str,
    enabled: bool = True,
    max_nfev: int = 60,
    ridge: float = 1.0e-10,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Refine p_u and clock by minimizing baseline data-domain LS residual."""
    y_vec = np.asarray(y_vec, dtype=complex).reshape(-1)
    bounds_p = np.asarray(
        config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]),
        dtype=float,
    )
    bounds_dt = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    lower = np.r_[bounds_p[:, 0], bounds_dt[0]]
    upper = np.r_[bounds_p[:, 1], bounds_dt[1]]
    x0 = np.clip(np.r_[np.asarray(initial_position, dtype=float).reshape(3), float(initial_clock)], lower, upper)
    warning = ""

    def fit_for_x(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        groups = supports_from_position_clock(
            scene,
            x[:3],
            float(x[3]),
            panels,
            model_variant=model_variant,
        )
        expanded = [support for group in groups for support in expand_jones_group(group, 2)]
        Phi = supports_to_design(scene, config, expanded)
        coeffs, y_hat, residual = linear_ls_fit(Phi, y_vec, ridge=ridge)
        return coeffs, y_hat, residual, groups

    coeffs0, y_hat0, residual0, groups0 = fit_for_x(x0)
    initial_norm = float(np.linalg.norm(residual0))
    x_best = x0.copy()
    coeffs_best = coeffs0
    y_hat_best = y_hat0
    residual_best = residual0
    groups_best = groups0
    success = False
    solver = "disabled"
    cost = float(initial_norm**2)

    if enabled and scipy_is_available():
        try:
            from scipy.optimize import least_squares  # type: ignore[import-not-found]

            def residual_real(x: np.ndarray) -> np.ndarray:
                _, _, residual, _ = fit_for_x(x)
                return np.r_[residual.real, residual.imag]

            result = least_squares(
                residual_real,
                x0,
                bounds=(lower, upper),
                max_nfev=int(max_nfev),
                x_scale=np.maximum(upper - lower, 1.0e-12),
            )
            x_candidate = np.asarray(result.x, dtype=float)
            coeffs_cand, y_hat_cand, residual_cand, groups_cand = fit_for_x(x_candidate)
            if np.linalg.norm(residual_cand) <= np.linalg.norm(residual_best) + 1.0e-12:
                x_best = x_candidate
                coeffs_best = coeffs_cand
                y_hat_best = y_hat_cand
                residual_best = residual_cand
                groups_best = groups_cand
            success = bool(result.success)
            solver = "scipy.optimize.least_squares"
            cost = float(result.cost)
        except Exception as exc:  # noqa: BLE001 - retain coarse fit on optimizer failure.
            warning = f"offgrid_refinement_failed: {type(exc).__name__}: {exc}"
            solver = "fallback_initial"
    elif enabled:
        solver = "fallback_initial_no_scipy"

    diagnostics = {
        "offgrid_refinement": bool(enabled),
        "refinement_objective": "data_domain_ls" if enabled else "",
        "refinement_solver": solver,
        "refinement_success": bool(success),
        "refinement_initial_residual_norm": initial_norm,
        "refinement_residual_norm": float(np.linalg.norm(residual_best)),
        "refinement_cost": cost,
        "refinement_initial_position": x0[:3].copy(),
        "refinement_initial_clock": float(x0[3]),
        "refined_position": x_best[:3].copy(),
        "refined_clock": float(x_best[3]),
        "warning": warning,
    }
    return (
        x_best[:3].astype(float),
        float(x_best[3]),
        coeffs_best,
        y_hat_best,
        groups_best,
        diagnostics,
    )


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.runtime_s = time.perf_counter() - self.start
        return False


def synthesize_from_position_jones(
    scene: dict,
    p_u: np.ndarray,
    delta_t: float,
    coeffs: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Synthesize a raw tensor from per-panel Jones coefficients."""
    k_paths = int(scene["K"])
    gamma = np.full(k_paths, np.pi / 4.0)
    eta = np.zeros(k_paths)
    comps = channel_components(scene, np.asarray(p_u, dtype=float), float(delta_t), gamma, eta)
    beta = np.zeros(k_paths, dtype=complex)
    coeffs = np.asarray(coeffs, dtype=complex).reshape(-1)
    for k in range(k_paths):
        start = 2 * k
        if start < coeffs.size:
            beta[k] = coeffs[start]
    return synthesize_raw_tensor(comps, beta), comps
