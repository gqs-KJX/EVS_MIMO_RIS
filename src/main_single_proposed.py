"""
Diagnostic runner for the proposed reliability-gated estimator.

This script is not the formal simulation sweep. It validates one realization
of the proposed algorithm:
    Stage-I initialization
    -> reliability check
    -> direct VP-WNLS or RIS-only basin recovery
    -> final raw-domain exact-spherical VP-WNLS.

Full legacy EVS/delay/RIS structured refinement is retained only as an
optional ablation comparison.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import io
import itertools
import json
import multiprocessing as mp
import os
import pathlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np


REPEAT_TRIAL_FIELDS = [
    "trial_id",
    "seed",
    "snr_db",
    "K",
    "failed",
    "error",
    "runtime_s",
    "p_true_x",
    "p_true_y",
    "p_true_z",
    "p_hat_x",
    "p_hat_y",
    "p_hat_z",
    "err_x_m",
    "err_y_m",
    "err_z_m",
    "position_error_m",
    "delta_t_true_s",
    "delta_t_hat_s",
    "delta_t_error_s",
    "clock_error_ns",
    "clock_range_error_m",
    "y_nmse",
    "raw_objective_final",
    "residual_norm",
    "noise_variance",
    "stage1_position_error_m",
    "stage2_position_error_m",
    "final_position_error_m",
    "selected_branch",
    "global_vp_mode",
    "global_vp_backend",
    "global_vp_gpu_used",
    "global_vp_gpu_num_objective_calls",
    "global_vp_cpu_gpu_objective_rel_diff",
    "global_vp_cpu_gpu_gradient_rel_diff",
    "global_vp_cpu_gpu_xhat_rel_diff",
    "jones_mode",
    "adaptive_enabled",
    "boundary_hit",
    "boundary_hit_axis",
    "distance_to_position_box_boundary_m",
    "z_rescue_triggered",
    "z_rescue_num_starts",
    "z_rescue_strategy",
    "z_rescue_num_probes",
    "z_rescue_num_full_refines",
    "z_rescue_probe_runtime_s",
    "z_rescue_full_refine_runtime_s",
    "z_rescue_refine_vp_mode",
    "z_rescue_best_z",
    "z_rescue_selected_reason",
    "stage2_warm_start_mode",
    "stage2_ris_warm_start_runtime_s",
    "stage2_ris_fresnel_lift_runtime_s",
    "stage2_ris_refine_runtime_s",
    "adaptive_jones_triggered",
    "adaptive_jones_trigger_reason",
    "global_vp_fixed_anchor_runtime_s",
    "global_vp_jones_runtime_s",
    "global_vp_optimizer_nfev",
    "global_vp_actual_residual_calls",
    "direct_boundary_hit",
    "rescue_boundary_hit",
    "branch_score_margin",
    "boundary_selection_rule_used",
    "warning",
]

REPEAT_NUMERIC_METRICS = [
    field
    for field in REPEAT_TRIAL_FIELDS
    if field
    not in {
        "failed",
        "error",
        "selected_branch",
        "global_vp_mode",
        "global_vp_backend",
        "jones_mode",
        "adaptive_enabled",
        "boundary_hit_axis",
        "z_rescue_selected_reason",
        "z_rescue_strategy",
        "z_rescue_refine_vp_mode",
        "stage2_warm_start_mode",
        "adaptive_jones_trigger_reason",
        "warning",
    }
]

STAGE2_DIAGNOSTIC_FIELDS = [
    "trial_id", "seed", "snr_db", "true_k", "stage2_rescue_impl",
    "ngc_direct_status", "ngc_direct_score", "rescue_triggered",
    "stage2_force_run_for_diagnostics", "rescue_available", "pllg_success",
    "pllg_failure_reason", "legacy_fallback_used", "legacy_fallback_reason",
    "num_valid_local_fixes", "local_weight_source", "delay_variance_source",
    "pllg_rank", "pllg_condition_number", "pllg_reweight_steps",
    "pllg_linear_x_m", "pllg_linear_y_m", "pllg_linear_z_m", "pllg_linear_s_m",
    "pllg_linear_clock_s", "pllg_projected_x_m", "pllg_projected_y_m",
    "pllg_projected_z_m", "pllg_projection_distance_m", "pllg_phi_before_polish",
    "pllg_phi_after_polish", "pllg_polish_success", "pllg_linear_runtime_s",
    "pllg_polish_runtime_s", "legacy_fallback_runtime_s", "stage2_total_runtime_s",
    "seed_position_error_m", "seed_z_error_m", "seed_clock_error_s",
    "final_position_error_m", "final_clock_error_s", "z_boundary_hit",
    "common_ris_refinement_success", "common_ris_refinement_impl",
    "common_ris_refinement_runtime_s", "common_ris_refinement_num_valid_local_fixes",
    "geometry_seed_impl", "pllg_pseudorange_block_weight", "delay_sigma_source",
    "delay_sigma_used_floor", "delay_sigma_min_s", "delay_sigma_max_s",
    "delay_sigma_values_json", "stage2_clock_term_raw_s2_before",
    "stage2_clock_term_normalized_before", "stage2_ris_term_raw_before",
    "stage2_ris_term_mean_before", "stage2_ris_term_normalized_before",
    "stage2_phi_normalized_before", "stage2_clock_term_raw_s2_after",
    "stage2_clock_term_normalized_after", "stage2_ris_term_raw_after",
    "stage2_ris_term_mean_after", "stage2_ris_term_normalized_after",
    "stage2_phi_normalized_after", "stage2_ris_normalization_scale",
    "stage2_lambda_ris_normalized", "polish_accepted",
    "stage2_clock_estimator", "stage2_clock_weighted_mean_s",
    "stage2_clock_decoupled_s", "stage2_clock_decoupled_available",
    "stage2_clock_decoupled_reason", "stage2_clock_decoupled_num_inliers",
    "stage2_clock_decoupled_scale_m",
    "rescue_candidate_admissible", "selector_guard_reject_reason",
    "selector_raw_degradation", "selector_raw_relative_improvement",
    "selector_boundary_guard_used", "selector_boundary_override_used",
]

STAGE2_LOCAL_FIX_DIAGNOSTIC_FIELDS = [
    "trial_id", "seed", "snr_db", "panel_index", "assigned_panel_index",
    "panel_match_correct", "local_fix_valid", "local_fix_reject_reason",
    "local_fix_x_m", "local_fix_y_m", "local_fix_z_m", "true_x_m", "true_y_m",
    "true_z_m", "local_error_x_m", "local_error_y_m", "local_error_z_m",
    "local_error_norm_m", "range_hat_m", "theta_hat_rad", "phi_hat_rad",
    "projection_residual_before", "projection_residual_after", "assignment_margin",
    "local_weight_source", "local_weight_scalar", "local_fix_source_stage",
    "common_refinement_impl", "stage1_local_fix_x_m", "stage1_local_fix_y_m",
    "stage1_local_fix_z_m", "refined_local_fix_x_m", "refined_local_fix_y_m",
    "refined_local_fix_z_m",
]

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from src.channel_model import (
        add_awgn,
        channel_components,
        generate_scene,
        synthesize_raw_tensor,
    )
    from src.config import apply_stage1_init_preset, default_config
    from src.diagnostics import (
        format_float_list,
        hankel_metric_summary,
        noise_metric_summary,
        estimate_position_from_ris_eta,
        parameter_errors_for_structured,
        parameter_errors_for_vp,
        run_delay_projection_self_test,
        run_ris_projection_self_test,
        run_tensor_factorization_shape_self_test,
        y_metric_summary,
        z_metric_summary,
    )
    from src.estimators import (
        global_exact_spherical_vp_refinement,
        initialize_from_hankel,
        ris_only_basin_recovery_fast,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from src.metrics import position_rmse, relative_nmse, rmse_abs
    from src.geometry import polarization_vector
    from src.projections_delay import tau_from_pole
    from src.projections_ris import (
        compressed_exact_response,
        local_ris_search_config,
        scaled_residual,
    )
    from src.robust_jnpp import (
        robust_jnpp_basin_recovery,
        robust_jnpp_geometry_consistency_score,
    )
    from src.stage2_rescue import (
        Stage2CommonState,
        build_local_fix_records,
        polish_stage2_seed,
        solve_stage2_pllg as _solve_stage2_pllg_backend,
    )
    from src.global_vp import data_only_efim_diagnostic, distance_to_box_boundary
    from src.tensor_utils import hankelize_frequency
    from src.utils import scipy_is_available
else:
    from .channel_model import (
        add_awgn,
        channel_components,
        generate_scene,
        synthesize_raw_tensor,
    )
    from .config import apply_stage1_init_preset, default_config
    from .diagnostics import (
        format_float_list,
        hankel_metric_summary,
        noise_metric_summary,
        estimate_position_from_ris_eta,
        parameter_errors_for_structured,
        parameter_errors_for_vp,
        run_delay_projection_self_test,
        run_ris_projection_self_test,
        run_tensor_factorization_shape_self_test,
        y_metric_summary,
        z_metric_summary,
    )
    from .estimators import (
        global_exact_spherical_vp_refinement,
        initialize_from_hankel,
        ris_only_basin_recovery_fast,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from .metrics import position_rmse, relative_nmse, rmse_abs
    from .geometry import polarization_vector
    from .projections_delay import tau_from_pole
    from .projections_ris import (
        compressed_exact_response,
        local_ris_search_config,
        scaled_residual,
    )
    from .robust_jnpp import (
        robust_jnpp_basin_recovery,
        robust_jnpp_geometry_consistency_score,
    )
    from .stage2_rescue import (
        Stage2CommonState,
        build_local_fix_records,
        polish_stage2_seed,
        solve_stage2_pllg as _solve_stage2_pllg_backend,
    )
    from .global_vp import data_only_efim_diagnostic, distance_to_box_boundary
    from .tensor_utils import hankelize_frequency
    from .utils import scipy_is_available


FINAL_PROPOSED_STAGE2_POLICY = "ngc_certified_ris_only"
FINAL_PROPOSED_STAGE2_RESCUE_TYPE = "ris_only"
FINAL_PROPOSED_RIS_RESCUE_IMPL = "local_ris_projection"


def _make_data(config: dict) -> dict:
    """Generate one reproducible synthetic channel and noisy observation."""
    data_start = time.perf_counter()
    rng = np.random.default_rng(config["seed"])
    scene = generate_scene(config, rng)
    true_components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(true_components, scene["beta_true"])
    y_noisy, noise_variance = add_awgn(
        y_true,
        config["SNR_dB"],
        rng,
        active_mask=scene.get("evs_observation_mask"),
    )
    data_generation_s = time.perf_counter() - data_start

    hankel_start = time.perf_counter()
    z_true = hankelize_frequency(y_true, scene["P"])
    z_noisy = hankelize_frequency(y_noisy, scene["P"])
    hankelization_s = time.perf_counter() - hankel_start

    assert y_true.shape == (scene["I"], scene["N"], scene["T"])
    assert y_noisy.shape == y_true.shape
    assert z_true.shape == (scene["I"], scene["P"], scene["L"], scene["T"])
    assert z_noisy.shape == z_true.shape

    return {
        "scene": scene,
        "true_components": true_components,
        "Y_true": y_true,
        "Y_noisy": y_noisy,
        "Z_true": z_true,
        "Z_noisy": z_noisy,
        "noise_variance": noise_variance,
        "timing": {
            "data_generation": data_generation_s,
            "hankelization": hankelization_s,
        },
    }


def _git_commit() -> str:
    """Return the current git commit hash when the script is run inside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _fmt(value, precision: int = 6) -> str:
    """Format scalars for compact diagnostic tables."""
    if value is None or value == "":
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return "NA"
    return f"{value_float:.{precision}e}"


def _fmt_fixed(value, precision: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return "NA"
    return f"{value_float:.{precision}f}"


def _fmt_vector(values: np.ndarray, scale: float = 1.0, precision: int = 4) -> str:
    arr = np.asarray(values).reshape(-1) * scale
    return "[" + ", ".join(_fmt_fixed(value, precision) for value in arr) + "]"


def _wrap_angle_rad(angle: float) -> float:
    return float(np.angle(np.exp(1j * angle)))


def _relative_complex_residual(target: np.ndarray, model: np.ndarray, eps: float) -> float:
    scale = np.vdot(model, target) / (np.vdot(model, model) + eps)
    return float(np.linalg.norm(target - scale * model) / (np.linalg.norm(target) + eps))


def _evs_model(scene: dict, path: int, gamma: float, eta_pol: float) -> np.ndarray:
    pol = scene["Theta"][path] @ polarization_vector(gamma, eta_pol)
    return np.kron(scene["v_B"][path], pol)


def _ris_local_residual(scene: dict, path: int, c_vec: np.ndarray, eta_local: np.ndarray) -> float:
    h_model = compressed_exact_response(
        eta_local,
        scene["Omega"][path],
        scene["a_RB"][path],
        scene["ris_grid"],
        scene["wavelength"],
    )
    value, _ = scaled_residual(c_vec, h_model, 1.0e-10)
    return float(np.sqrt(max(value, 0.0) / (np.linalg.norm(c_vec) ** 2 + 1.0e-10)))


def _evs_local_residual(
    scene: dict, path: int, a_vec: np.ndarray, gamma: float, eta_pol: float
) -> float:
    return _relative_complex_residual(
        a_vec, _evs_model(scene, path, gamma, eta_pol), 1.0e-10
    )


def _format_matrix(matrix: np.ndarray, precision: int = 4) -> str:
    arr = np.asarray(matrix, dtype=float)
    rows = []
    for row in arr:
        rows.append("[" + ", ".join(_fmt(value, precision) for value in row) + "]")
    return "[" + ", ".join(rows) + "]"


def _permutation_margin(
    costs: np.ndarray, orientation: str
) -> tuple[float, float, float] | None:
    arr = np.asarray(costs, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or not np.all(np.isfinite(arr)):
        return None
    scores = []
    k_paths = arr.shape[0]
    for perm in itertools.permutations(range(k_paths)):
        if orientation == "column_to_panel":
            score = sum(arr[col, perm[col]] for col in range(k_paths))
        elif orientation == "panel_to_column":
            score = sum(arr[perm[panel], panel] for panel in range(k_paths))
        else:
            raise ValueError(f"unknown assignment orientation {orientation!r}")
        scores.append(float(score))
    scores.sort()
    best = scores[0]
    second = scores[1] if len(scores) > 1 else float("inf")
    margin = (second - best) / max(abs(best), 1.0e-12)
    return best, second, float(margin)


def _inverse_assignment(column_to_panel: list[int]) -> list[int]:
    panel_to_column = [-1] * len(column_to_panel)
    for column, panel in enumerate(column_to_panel):
        panel_to_column[int(panel)] = int(column)
    return panel_to_column


def _print_ris_dimension_diagnostics(scene: dict, true_components: dict) -> None:
    """Print and assert RIS element-domain and compressed-domain dimensions."""
    print(f"M_Rx = {scene['M_Rx']}")
    print(f"M_Ry = {scene['M_Ry']}")
    print(f"M_R = {scene['M_R']}")
    for k in range(scene["K"]):
        g_k = true_components["g"][k]
        omega_k = scene["Omega"][k]
        c_k = true_components["c"][k]
        assert len(g_k) == scene["M_R"], "len(g_k) != M_R"
        assert omega_k.shape == (scene["T"], scene["M_R"]), "Omega_k shape mismatch"
        assert c_k.shape == (scene["T"],), "c_k shape mismatch"
        print(
            f"path {k}: len(g_k)={len(g_k)}, "
            f"Omega_k shape={omega_k.shape}, c_k shape={c_k.shape}"
        )


def _print_self_tests(scene: dict, config: dict, true_components: dict) -> None:
    """Run and print deterministic self-tests requested for diagnostics."""
    print("\n=== Self-tests ===")
    tensor_test = run_tensor_factorization_shape_self_test()
    print(f"tensor_unfolding_max_error = {tensor_test['max_mode_error']:.3e}")

    delay_test = run_delay_projection_self_test(scene["delta_f"])
    print(
        "delay_projection: "
        f"true_pole={delay_test['true_pole']:.6g}, "
        f"estimated_pole={delay_test['estimated_pole']:.6g}, "
        f"delay_error_s={delay_test['delay_error_s']:.3e}"
    )

    ris_test = run_ris_projection_self_test(scene, config, true_components)
    print(
        "ris_projection: "
        f"Phi_before={ris_test['phi_before']:.6e}, "
        f"Phi_after={ris_test['phi_after']:.6e}, "
        f"range_error={ris_test['range_error']:.3e}, "
        f"angle_error={ris_test['angle_error']:.3e}, "
        f"pinv_used={ris_test['used_pinv']}"
    )
    if ris_test["warning"]:
        print(f"WARNING OBJECTIVE_MISMATCH: {ris_test['warning']}")


def _weak_reasonable_stage1_config(config: dict) -> dict:
    """Return the normal Stage-I config; retained for legacy diagnostic callers."""
    return copy.deepcopy(config)


def _apply_main_single_defaults(config: dict) -> dict:
    """Apply main_single diagnostic defaults to a local config copy."""
    config.setdefault("diagnostic_mode", "performance")
    config.setdefault("run_full_legacy_comparison", False)
    config.setdefault("verbose_stage2", False)
    config.setdefault("print_progress", True)
    mode = str(config.get("diagnostic_mode", "performance")).lower()
    if mode not in ("smoke", "performance"):
        raise ValueError(f"unknown diagnostic_mode {mode!r}")
    config["diagnostic_mode"] = mode
    if mode == "smoke":
        config.setdefault("diagnostic_fast_problem_size", True)
        config.setdefault("diagnostic_fast_stage1_search", True)
    else:
        config["diagnostic_fast_problem_size"] = False
        config["diagnostic_fast_stage1_search"] = False
        config["M_A"] = max(int(config.get("M_A", 16)), 16)
        ris_shape = tuple(config.get("ris_shape", (64, 64)))
        config["ris_shape"] = (max(int(ris_shape[0]), 64), max(int(ris_shape[1]), 64))
        config["N"] = max(int(config.get("N", 63)), 63)
        config["P"] = max(int(config.get("P", 32)), 32)
        config["T"] = max(int(config.get("T", 256)), 256)
        ris_search = dict(config["ris_search"])
        floors = {
            "num_range": 9,
            "num_elev": 5,
            "num_az": 13,
            "num_exact_refine_starts": 0
            if config.get("stage1_ris_geometry_mode", "coarse_correlation")
            == "coarse_correlation"
            else 3,
            "num_lift_candidates": 1,
            "num_lift_steps": 1,
        }
        for key, floor in floors.items():
            ris_search[key] = max(int(ris_search.get(key, floor)), floor)
        config["ris_search"] = ris_search
    config.setdefault("stage2_adaptive", True)
    config.setdefault("stage2_rescue_type", FINAL_PROPOSED_STAGE2_RESCUE_TYPE)
    config.setdefault("stage2_rescue_impl", "legacy_multistart")
    config.setdefault("stage2_pllg_reweight_steps", 1)
    config.setdefault("stage2_pllg_cond_max", 1.0e12)
    config.setdefault("stage2_pllg_local_weight_mode", "auto")
    config.setdefault("stage2_pllg_pseudorange_block_weight", 0.0)
    config.setdefault("stage2_clock_estimator", "decoupled_robust")
    config.setdefault("stage2_clock_sigma_range_m", 0.12)
    config.setdefault("stage2_clock_outlier_kappa", 3.0)
    config.setdefault("stage2_delay_sigma_floor_ns", 0.5)
    config.setdefault("stage2_ris_normalization_scale", 1.0e-4)
    config.setdefault("stage2_lambda_ris_normalized", 1.0)
    config.setdefault("stage2_pllg_max_projection_distance_m", 0.05)
    config.setdefault("stage2_pllg_legacy_fallback", True)
    config.setdefault("stage2_force_run_for_diagnostics", False)
    config.setdefault("stage2_selector_guard", True)
    config.setdefault("stage2_selector_raw_degradation_abs_tol", 1.0e-8)
    config.setdefault("stage2_selector_raw_degradation_rel_tol", 1.0e-4)
    config.setdefault("stage2_selector_boundary_override_min_rel_improvement", 1.0e-3)
    config.setdefault("proposed_stage2_policy", FINAL_PROPOSED_STAGE2_POLICY)
    config.setdefault("stage2_ris_rescue_impl", FINAL_PROPOSED_RIS_RESCUE_IMPL)
    config.setdefault("rescue_accept_min_rel_improvement", 0.0)
    config.setdefault("rescue_accept_min_abs_improvement", 1.0e-8)
    config.setdefault("ngc_lambda_ris", 1.0)
    config.setdefault("ngc_clock_green_quantile", 0.99)
    config.setdefault("ngc_clock_red_quantile", 0.999)
    config.setdefault("ngc_clock_sigma_floor_ns", 0.5)
    config.setdefault("ngc_ris_green_threshold", 0.3)
    config.setdefault("ngc_ris_red_threshold", 0.7)
    config["stage2_mode"] = "none"
    config.setdefault("reliability_assignment_good", 1.0)
    config.setdefault("reliability_assignment_low", 0.3)
    config.setdefault("reliability_clock_good_ns", 0.1)
    config.setdefault("reliability_clock_bad_ns", 0.5)
    config.setdefault("reliability_ris_good", 0.3)
    config.setdefault("reliability_ris_bad", 0.7)
    global_vp = dict(config.get("global_vp", {}))
    global_vp.setdefault("solver", "least_squares")
    global_vp.setdefault("use_delay_prior", False)
    global_vp.setdefault("use_trust_region", False)
    config["global_vp"] = global_vp
    if bool(config.get("diagnostic_fast_problem_size", True)):
        config["M_A"] = min(int(config.get("M_A", 4)), 4)
        ris_shape = tuple(config.get("ris_shape", (8, 8)))
        config["ris_shape"] = (
            min(int(ris_shape[0]), 8),
            min(int(ris_shape[1]), 8),
        )
        config["N"] = min(int(config.get("N", 15)), 15)
        config["P"] = min(int(config.get("P", 8)), 8, int(config["N"]))
        config["T"] = min(int(config.get("T", 32)), 32)
    if bool(config.get("diagnostic_fast_stage1_search", True)):
        ris_search = dict(config["ris_search"])
        caps = {
            "num_range": 5,
            "num_elev": 3,
            "num_az": 7,
            "num_exact_refine_starts": 1,
            "num_lift_candidates": 1,
            "num_lift_steps": 1,
        }
        for key, cap in caps.items():
            ris_search[key] = min(int(ris_search.get(key, cap)), cap)
        config["ris_search"] = ris_search
    return config


def _stage1_clock_panel_order(stage1_estimate: dict, scene: dict) -> tuple[np.ndarray, np.ndarray, bool, list[int] | None]:
    """Return Stage-I tau/range arrays in physical RIS panel order."""
    tau_raw = np.array(
        [tau_from_pole(pole, scene["delta_f"]) for pole in stage1_estimate["poles"]],
        dtype=float,
    )
    ris_eta_raw = np.asarray(stage1_estimate["ris_eta"], dtype=float)
    range_raw = ris_eta_raw[:, 0]
    columns_are_panel_ordered = bool(stage1_estimate.get("columns_are_panel_ordered", False))
    panel_to_column = stage1_estimate.get("panel_to_column_assignment")
    if panel_to_column is not None:
        panel_to_column = [int(col) for col in panel_to_column]
    if columns_are_panel_ordered or panel_to_column is None:
        return tau_raw, range_raw, bool(columns_are_panel_ordered), panel_to_column

    k_paths = int(scene["K"])
    if len(panel_to_column) != k_paths:
        return tau_raw, range_raw, False, None
    tau_phys = np.empty(k_paths, dtype=float)
    range_phys = np.empty(k_paths, dtype=float)
    for panel in range(k_paths):
        col = panel_to_column[panel]
        tau_phys[panel] = tau_raw[col]
        range_phys[panel] = range_raw[col]
    return tau_phys, range_phys, True, panel_to_column


def _assignment_margin_from_costs(costs: np.ndarray, eps: float) -> float:
    """Return the Stage-I assignment margin over column-to-panel permutations."""
    arr = np.asarray(costs, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or not np.all(np.isfinite(arr)):
        return float("nan")
    scores = []
    for perm in itertools.permutations(range(arr.shape[0])):
        scores.append(float(sum(arr[col, perm[col]] for col in range(arr.shape[0]))))
    if not scores:
        return float("nan")
    scores.sort()
    best = scores[0]
    second = scores[1] if len(scores) > 1 else float("inf")
    return float((second - best) / (best + eps))


def _stage1_ris_residuals(stage1_estimate: dict, scene: dict) -> np.ndarray:
    """Return Stage-I per-path compressed RIS residuals when they can be evaluated."""
    _ = scene
    for key in ("stage1_ris_residuals", "ris_residuals", "ris_projection_residuals"):
        if key in stage1_estimate:
            values = np.asarray(stage1_estimate[key], dtype=float).reshape(-1)
            if values.size:
                return values
    return np.array([], dtype=float)


def _finite_less(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value < threshold)


def _finite_greater(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value > threshold)


def compute_stage1_reliability(stage1_estimate: dict, scene: dict, config: dict) -> dict:
    """Compute the diagnostic Stage-I reliability gate used by main_single only."""
    eps = float(config.get("eps", 1.0e-10))
    assignment_margin = stage1_estimate.get("stage1_assignment_margin")
    if assignment_margin is None and "assignment_costs" in stage1_estimate:
        assignment_margin = _assignment_margin_from_costs(
            stage1_estimate["assignment_costs"], eps
        )
    try:
        assignment_margin = float(assignment_margin)
    except (TypeError, ValueError):
        assignment_margin = float("nan")

    tau_stage1, range_stage1, used_panel_order, panel_to_column = (
        _stage1_clock_panel_order(stage1_estimate, scene)
    )
    delta_t_k = tau_stage1 - (range_stage1 + scene["d_RB"]) / scene["c0"]
    delta_t_k_ns = delta_t_k * 1.0e9
    sigma_delta_t = float(np.std(delta_t_k))
    sigma_delta_t_ns = sigma_delta_t * 1.0e9

    ris_residuals = _stage1_ris_residuals(stage1_estimate, scene)
    finite_ris = ris_residuals[np.isfinite(ris_residuals)]
    max_ris_residual = float(np.max(finite_ris)) if finite_ris.size else float("nan")

    assignment_good = float(config["reliability_assignment_good"])
    clock_good_ns = float(config["reliability_clock_good_ns"])
    ris_good = float(config["reliability_ris_good"])
    assignment_low = float(config["reliability_assignment_low"])
    clock_bad_ns = float(config["reliability_clock_bad_ns"])
    ris_bad = float(config["reliability_ris_bad"])
    use_ris_residual_gate = bool(
        config.get(
            "stage1_ris_residual_gate",
            str(config.get("stage1_ris_geometry_mode", "coarse_correlation"))
            == "coarse_correlation",
        )
    )

    bad_score = 0
    trigger_reasons = []
    if _finite_less(assignment_margin, assignment_good):
        bad_score += 1
        trigger_reasons.append("low_assignment_margin")
    if _finite_greater(sigma_delta_t_ns, clock_good_ns):
        bad_score += 1
        trigger_reasons.append("poor_clock_consistency")
    if use_ris_residual_gate and _finite_greater(max_ris_residual, ris_good):
        bad_score += 1
        trigger_reasons.append("large_ris_residual")

    severe_unreliable = (
        _finite_less(assignment_margin, assignment_low)
        or _finite_greater(sigma_delta_t_ns, clock_bad_ns)
        or (use_ris_residual_gate and _finite_greater(max_ris_residual, ris_bad))
    )
    decision = "direct_vp" if bad_score == 0 else "ris_only_stage2_then_vp"
    return {
        "assignment_margin": assignment_margin,
        "sigma_delta_t": sigma_delta_t,
        "sigma_delta_t_ns": sigma_delta_t_ns,
        "delta_t_k_ns": delta_t_k_ns.tolist(),
        "reliability_used_panel_order": bool(used_panel_order),
        "reliability_panel_to_column": panel_to_column,
        "max_ris_residual": max_ris_residual,
        "stage1_ris_residual_type": stage1_estimate.get(
            "stage1_ris_residual_type", "unknown"
        ),
        "stage1_ris_residual_gate": bool(use_ris_residual_gate),
        "bad_score": int(bad_score),
        "decision": decision,
        "trigger_reasons": trigger_reasons,
        "severe_unreliable": bool(severe_unreliable),
    }


def _paper_balanced_good_snr_triggered_stage2(config: dict, reliability: dict) -> bool:
    """Return True when default paper preset unexpectedly asks for Stage-II."""
    return (
        str(config.get("diagnostic_mode", "performance")).lower() == "performance"
        and float(config.get("SNR_dB", float("nan"))) >= 0.0
        and str(config.get("stage1_init_mode", "paper_balanced")) == "paper_balanced"
        and reliability.get("decision") != "direct_vp"
    )


def _max_ris_residual_display(reliability: dict):
    """Return the log value for max RIS residual consistent with gate usage."""
    if bool(reliability.get("stage1_ris_residual_gate", False)):
        return reliability.get("max_ris_residual")
    if reliability.get("stage1_ris_residual_type") == "coarse_proxy":
        return "coarse_proxy_not_used"
    return "NA"


def _print_stage1_initialization_diagnostics(results: dict) -> None:
    """Print the Stage-I initialization mode and RIS search strength."""
    ris_search = results["stage1_initialization"]["ris_search"]
    print(f"stage1_init_mode = {results['stage1_initialization']['mode']}")
    print(
        f"range/elev/az = ({ris_search['num_range']}, "
        f"{ris_search['num_elev']}, {ris_search['num_az']})"
    )
    print(f"exact_refine_starts = {ris_search['num_exact_refine_starts']}")
    print(f"lift_candidates = {ris_search['num_lift_candidates']}")
    print(f"lift_steps = {ris_search['num_lift_steps']}")


def _print_run_configuration(config: dict, results: dict) -> None:
    """Print the structured run configuration block."""
    scene = results["scene"]
    ris_search = results["stage1_initialization"]["ris_search"]
    final_method = str(
        config.get("final_refinement_method", "global_exact_spherical_vp")
    ).lower()
    global_vp_options = dict(config.get("global_vp", {}))
    global_vp_solver = str(global_vp_options.get("solver", "least_squares"))
    global_vp_mode = str(global_vp_options.get("mode", "jones_regularized"))
    if not config.get("enable_global_vp", True) or final_method == "none":
        vp_solver_type = "skipped_global_vp"
        vp_backend = "skipped"
    elif (
        final_method == "global_exact_spherical_vp"
        and global_vp_mode in {"adaptive_jones", "jones_regularized", "jones_free"}
        and scipy_is_available()
    ):
        vp_solver_type = "scipy.optimize.minimize:L-BFGS-B"
        vp_backend = "scipy.optimize"
    elif (
        final_method == "global_exact_spherical_vp"
        and global_vp_mode in {"adaptive_jones", "jones_regularized", "jones_free"}
    ):
        vp_solver_type = "bounded_coordinate_search"
        vp_backend = "fallback"
    elif (
        final_method == "global_exact_spherical_vp"
        and global_vp_solver == "least_squares"
        and scipy_is_available()
    ):
        vp_solver_type = "scipy.optimize.least_squares"
        vp_backend = "scipy.optimize"
    elif final_method == "global_exact_spherical_vp" and global_vp_solver == "least_squares":
        vp_solver_type = "bounded_coordinate_search"
        vp_backend = "fallback"
    elif (
        final_method == "global_exact_spherical_vp"
        and global_vp_solver == "lbfgsb_reduced"
        and scipy_is_available()
    ):
        vp_solver_type = "scipy.optimize.minimize:L-BFGS-B"
        vp_backend = "scipy.optimize"
    elif final_method == "global_exact_spherical_vp" and global_vp_solver == "lbfgsb_reduced":
        vp_solver_type = "bounded_coordinate_search"
        vp_backend = "fallback"
    elif final_method == "legacy_raw_vp" and scipy_is_available():
        vp_solver_type = "scipy.optimize.least_squares"
        vp_backend = "scipy.optimize"
    else:
        vp_solver_type = "bounded_coordinate_search"
        vp_backend = "fallback"

    print("=== Run configuration ===")
    print(f"diagnostic_mode = {config.get('diagnostic_mode', 'performance')}")
    print(f"seed = {config['seed']}")
    print(f"git_commit = {_git_commit()}")
    print(f"SNR_dB = {config['SNR_dB']:.1f}")
    print(f"fc_Hz = {config['fc']:.6e}")
    print(f"fc_GHz = {config['fc'] / 1.0e9:.3f}")
    print(f"delta_f_Hz = {config['delta_f']:.6e}")
    print(f"delta_f_MHz = {config['delta_f'] / 1.0e6:.3f}")
    print(
        f"K={scene['K']}, I={scene['I']}, N={scene['N']}, "
        f"P={scene['P']}, L={scene['L']}, T={scene['T']}"
    )
    print(
        f"RIS_shape = ({scene['M_Rx']}, {scene['M_Ry']}), "
        f"M_R = {scene['M_R']}"
    )
    print(f"stage1_init_mode = {results['stage1_initialization']['mode']}")
    print(
        "diagnostic_fast_problem_size = "
        f"{config.get('diagnostic_fast_problem_size', True)}"
    )
    print(
        "diagnostic_fast_stage1_search = "
        f"{config.get('diagnostic_fast_stage1_search', True)}"
    )
    print(
        "stage1_grid = "
        f"range={ris_search['num_range']}, "
        f"elev={ris_search['num_elev']}, az={ris_search['num_az']}"
    )
    print(f"exact_refine_starts = {ris_search['num_exact_refine_starts']}")
    print(f"lift_candidates = {ris_search['num_lift_candidates']}")
    print(f"lift_steps = {ris_search['num_lift_steps']}")
    print(
        "configured_stage2_enabled_flags = "
        f"EVS={config.get('stage2_enable_evs', True)}, "
        f"delay={config.get('stage2_enable_delay', True)}, "
        f"RIS={config.get('stage2_enable_ris', True)}"
    )
    print("proposed_ris_only_stage2_flags = EVS=False, delay=False, RIS=True")
    print(f"stage2_guarded = {config.get('stage2_guarded', False)}")
    print(f"configured_stage2_mode = {config.get('stage2_mode', 'none')}")
    print(
        "proposed_stage2_policy = "
        f"{config.get('proposed_stage2_policy', FINAL_PROPOSED_STAGE2_POLICY)}"
    )
    print(
        "stage2_ris_rescue_impl = "
        f"{config.get('stage2_ris_rescue_impl', FINAL_PROPOSED_RIS_RESCUE_IMPL)}"
    )
    print(f"ngc_lambda_ris = {config.get('ngc_lambda_ris', 1.0)}")
    print(
        "ngc_clock_green_quantile = "
        f"{config.get('ngc_clock_green_quantile', 0.99)}"
    )
    print(
        "ngc_clock_red_quantile = "
        f"{config.get('ngc_clock_red_quantile', 0.999)}"
    )
    print(
        "rescue_accept_min_rel_improvement = "
        f"{config.get('rescue_accept_min_rel_improvement', 0.0)}"
    )
    print(
        "rescue_accept_min_abs_improvement = "
        f"{config.get('rescue_accept_min_abs_improvement', 1.0e-8)}"
    )
    if str(config.get("proposed_stage2_policy", "")).lower() == "ngc_certified_ris_only":
        print(
            "ngc_clock_sigma_floor_ns = "
            f"{config.get('ngc_clock_sigma_floor_ns', 0.5)}"
        )
        print("ngc_ris_availability_mode = ris_geometry_consistency_score_if_C_available")
    print(f"run_full_legacy_comparison = {config.get('run_full_legacy_comparison', False)}")
    print(f"num_structured_iters = {config['num_structured_iters']}")
    print(f"enable_global_vp = {config.get('enable_global_vp', True)}")
    print(f"final_refinement_method = {config.get('final_refinement_method', 'global_exact_spherical_vp')}")
    print(f"global_vp_solver = {global_vp_solver}")
    print(f"global_vp_mode = {global_vp_mode}")
    print(f"vp_solver_type = {vp_solver_type}")
    print(f"vp_solver_backend = {vp_backend}")
    if str(config.get("diagnostic_mode", "performance")).lower() == "smoke":
        print("WARNING_SMOKE_TEST_NOT_FOR_PERFORMANCE")
    if bool(config.get("diagnostic_fast_problem_size", False)):
        print(
            "WARNING: diagnostic_fast_problem_size=True changes the estimation problem."
        )
        print("Results are not comparable with full-size paper performance.")
    if bool(config.get("diagnostic_fast_stage1_search", False)):
        print(
            "WARNING: diagnostic_fast_stage1_search=True weakens Stage-I search."
        )
        print("Results are not comparable with full-size paper performance.")


def _print_assignment_diagnostics(results: dict) -> None:
    """Print Stage-I and Stage-II assignment mappings with correct orientations."""
    print("\n=== Assignment diagnostics ===")
    column_to_panel = [int(item) for item in results["estimate_initial"]["assignment"]]
    panel_to_column = _inverse_assignment(column_to_panel)
    print(f"column_to_panel_assignment = {column_to_panel}")
    print(f"panel_to_column_assignment = {panel_to_column}")
    costs = results["estimate_initial"].get("assignment_costs")
    if costs is not None:
        print(f"stage1_assignment_cost_matrix_col_by_panel = {_format_matrix(costs)}")
        margin = _permutation_margin(costs, "column_to_panel")
        if margin is not None:
            best, second, rel_margin = margin
            print(
                "stage1_assignment_margin = "
                f"best={best:.6e}, second={second:.6e}, relative={rel_margin:.6e}"
            )
            if rel_margin < 1.0e-3:
                print(
                    "WARNING ASSIGNMENT_AMBIGUOUS: Stage-I column-to-panel "
                    f"assignment has relative margin {rel_margin:.3e}."
                )

    for iter_idx, update in enumerate(results["structured_diag"]["updates"], start=1):
        mode4_panel_order = update.get("mode4_assignment_order")
        print(f"iter_{iter_idx}_mode4_panel_order = {mode4_panel_order}")
        assignment_costs = update.get("mode4_assignment_costs")
        if assignment_costs is not None:
            print(
                f"iter_{iter_idx}_mode4_assignment_cost_matrix_col_by_panel = "
                f"{_format_matrix(assignment_costs)}"
            )
            margin = _permutation_margin(assignment_costs, "panel_to_column")
            if margin is not None:
                best, second, rel_margin = margin
                print(
                    f"iter_{iter_idx}_mode4_assignment_margin = "
                    f"best={best:.6e}, second={second:.6e}, relative={rel_margin:.6e}"
                )
                if rel_margin < 1.0e-3:
                    print(
                        "WARNING ASSIGNMENT_AMBIGUOUS: Stage-II mode-4 "
                        f"panel order has relative margin {rel_margin:.3e}."
                    )


def _empty_structured_diag() -> dict:
    return {
        "z_hat_history": [],
        "residuals_noisy_rmse": [],
        "updates": [],
        "ris_projection_total_s": 0.0,
    }


def _common_timing_total(timing: dict) -> float:
    return float(
        timing.get("data_generation", 0.0)
        + timing.get("hankelization", 0.0)
        + timing.get("stage1", 0.0)
    )


def _make_branch_result(
    *,
    data: dict,
    estimate_initial: dict,
    estimate_used: dict,
    structured_diag: dict,
    final: dict,
    timing: dict,
    stage1_config: dict,
    branch_name: str,
    reliability: dict,
) -> dict:
    return {
        **data,
        "estimate_initial": estimate_initial,
        "estimate_used": estimate_used,
        "structured_diag": structured_diag,
        "stage1_initialization": {
            "mode": stage1_config.get("stage1_init_mode", "normal"),
            "ris_search": dict(stage1_config["ris_search"]),
        },
        "stage1_config": stage1_config,
        "final": final,
        "timing": timing,
        "branch_name": branch_name,
        "reliability": reliability,
    }


def _run_global_vp_branch(
    y_noisy: np.ndarray,
    estimate_used: dict,
    scene: dict,
    config: dict,
    stage2_mode: str,
) -> tuple[dict, float]:
    vp_start = time.perf_counter()
    final = global_exact_spherical_vp_refinement(
        y_noisy, estimate_used, scene, config
    )
    vp_s = time.perf_counter() - vp_start
    final["vp_enabled"] = True
    final["stage2_mode"] = stage2_mode
    final["final_refinement_method"] = "global_exact_spherical_vp"
    return final, vp_s


def _run_ris_only_stage2(
    z_noisy: np.ndarray,
    scene: dict,
    config: dict,
    estimate_initial: dict,
) -> tuple[dict, dict, dict, float]:
    ris_config = copy.deepcopy(config)
    ris_config["stage2_mode"] = "full_legacy"
    ris_config["num_structured_iters"] = int(
        config.get("stage2_ris_rescue_max_iters", 1)
    )
    # Do not enable EVS/delay projections in the default rescue branch.
    # Low-SNR logs show that EVS/delay updates can drift; RIS projection is the
    # physically meaningful basin-recovery step.
    ris_config["stage2_enable_evs"] = False
    ris_config["stage2_enable_delay"] = False
    ris_config["stage2_enable_ris"] = True
    ris_config["stage2_ris_use_current_eta"] = True
    ris_config["stage2_ris_skip_assignment"] = True
    ris_config["_stage2_ris_projection_cache"] = {}
    if not bool(config.get("stage2_precise_ablation", False)):
        ris_config["stage2_damping_grid"] = tuple(
            config.get("stage2_ris_rescue_damping_grid", (0.0, 1.0))
        )
    impl = str(config.get("stage2_ris_rescue_impl", FINAL_PROPOSED_RIS_RESCUE_IMPL))
    stage2_start = time.perf_counter()
    if impl == "robust_jnpp":
        estimate_used, structured_diag = robust_jnpp_basin_recovery(
            estimate_initial, scene, ris_config
        )
    elif impl in ("local_ris_projection", "fast"):
        estimate_used, structured_diag = ris_only_basin_recovery_fast(
            estimate_initial, z_noisy, scene, ris_config
        )
    elif impl == "legacy_structured":
        estimate_used, structured_diag = structured_refinement(
            z_noisy, scene, ris_config, copy.deepcopy(estimate_initial)
        )
    else:
        raise ValueError(f"unknown stage2_ris_rescue_impl {impl!r}")
    return (
        estimate_used,
        structured_diag,
        ris_config,
        time.perf_counter() - stage2_start,
    )


def refine_stage2_ris_factors(
    z_noisy: np.ndarray,
    scene: dict,
    config: dict,
    stage1_estimate: dict,
    *,
    efim_context: dict | None = None,
) -> Stage2CommonState:
    """Run the sole common RIS-only Stage-II preprocessing step."""
    start = time.perf_counter()
    refined, structured_diag, rescue_config, refinement_runtime = _run_ris_only_stage2(
        z_noisy, scene, config, copy.deepcopy(stage1_estimate)
    )
    tau_hat_s = np.asarray(
        [tau_from_pole(pole, scene["delta_f"]) for pole in refined["poles"]],
        dtype=float,
    )
    uncertainty = build_stage2_delay_uncertainty(
        stage1_estimate,
        scene,
        config,
        efim_context=efim_context,
    )
    local_records = build_local_fix_records(refined, scene, rescue_config)
    valid_local = int(sum(bool(record.get("valid", False)) for record in local_records))
    diagnostics = dict(structured_diag)
    diagnostics.update(
        {
            "common_ris_refinement_success": True,
            "common_ris_refinement_impl": str(
                config.get("stage2_ris_rescue_impl", FINAL_PROPOSED_RIS_RESCUE_IMPL)
            ),
            "common_ris_refinement_runtime_s": float(refinement_runtime),
            "common_ris_refinement_num_valid_local_fixes": valid_local,
            "local_fix_records": local_records,
            "num_valid_local_fixes": valid_local,
            "local_weight_source": "uniform_fallback",
            "delay_sigma_source": uncertainty["source"],
            "delay_variance_source": uncertainty["source"],
            "sigma_tau_source": uncertainty["source"],
            "delay_sigma_used_floor": bool(uncertainty["used_floor"]),
            "delay_sigma_values": np.asarray(uncertainty["sigma_tau_s"], dtype=float).copy(),
            "stage2_common_total_runtime_s": float(time.perf_counter() - start),
        }
    )
    return Stage2CommonState(
        stage1_estimate=copy.deepcopy(stage1_estimate),
        refined_estimate=refined,
        rescue_config=rescue_config,
        tau_hat_s=tau_hat_s,
        sigma_tau_s=np.asarray(uncertainty["sigma_tau_s"], dtype=float),
        sigma_tau_sq_s2=np.asarray(uncertainty["sigma_tau_sq_s2"], dtype=float),
        sigma_tau_source=str(uncertainty["source"]),
        sigma_tau_used_floor=bool(uncertainty["used_floor"]),
        local_fix_records=local_records,
        common_refinement_success=True,
        common_refinement_runtime_s=float(refinement_runtime),
        common_refinement_diagnostics=diagnostics,
    )


def _stage2_solution_from_polish(
    state: Stage2CommonState,
    scene: dict,
    polish: dict,
    *,
    geometry_seed_impl: str,
    seed_failure_reason: str = "",
) -> dict:
    diagnostics = dict(state.common_refinement_diagnostics)
    diagnostics.update(polish.get("diagnostics", {}))
    diagnostics["stage2_rescue_impl"] = geometry_seed_impl
    diagnostics["geometry_seed_impl"] = geometry_seed_impl
    diagnostics["stage2_failure_reason"] = seed_failure_reason
    polish_available = bool(
        polish.get(
            "rescue_available",
            polish.get("diagnostics", {}).get("rescue_available", False),
        )
    )
    diagnostics["rescue_available"] = polish_available and not bool(seed_failure_reason)
    estimate_used = copy.deepcopy(state.refined_estimate)
    if diagnostics["rescue_available"]:
        estimate_used["_global_vp_initial_p_u"] = np.asarray(polish["position"], dtype=float).copy()
        estimate_used["_global_vp_initial_delta_t"] = float(polish["clock_s"])
    return {
        "estimate": estimate_used,
        "position": np.asarray(polish.get("position", np.full(3, np.nan)), dtype=float).copy(),
        "clock": float(polish.get("clock_s", np.nan)),
        "chi": np.concatenate(
            [
                np.asarray(state.refined_estimate.get("gamma", []), dtype=float),
                np.asarray(state.refined_estimate.get("eta_pol", []), dtype=float),
            ]
        ),
        "objective": float(diagnostics.get("after_phi_stage2_normalized", np.nan)),
        "rescue_available": bool(diagnostics["rescue_available"]),
        "failure_reason": seed_failure_reason or str(diagnostics.get("polish_failure_reason", "")),
        "runtime_s": float(state.common_refinement_runtime_s + polish.get("runtime_s", 0.0)),
        "diagnostics": diagnostics,
        "rescue_config": state.rescue_config,
    }


def solve_stage2_legacy_multistart(state: Stage2CommonState, scene: dict, config: dict) -> dict:
    """Compute the existing legacy geometry seed, then use common polish."""
    valid_records = [
        record for record in state.local_fix_records if bool(record.get("valid", False))
    ]
    if not valid_records:
        position = np.full(3, np.nan)
        clock = float("nan")
        failure = "no_valid_local_fixes"
    else:
        try:
            position = np.mean(
                np.asarray([record["position"] for record in valid_records], dtype=float),
                axis=0,
            ).reshape(3)
            panels = np.asarray(
                [int(record["panel_index"]) for record in valid_records], dtype=int
            )
            ranges = np.asarray(state.refined_estimate["ris_eta"], dtype=float)[
                panels, 0
            ]
            values = state.tau_hat_s[panels] - (
                ranges + np.asarray(scene["d_RB"], dtype=float)[panels]
            ) / float(scene["c0"])
            clock = float(np.median(values))
            failure = ""
        except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
            position = np.full(3, np.nan)
            clock = float("nan")
            failure = "invalid_position_or_clock"
    polish = polish_stage2_seed(
        position, clock, state, scene, config, geometry_seed_impl="legacy_multistart"
    )
    return _stage2_solution_from_polish(
        state, scene, polish, geometry_seed_impl="legacy_multistart", seed_failure_reason=failure
    )


def solve_stage2_pllg(state: Stage2CommonState, scene: dict, config: dict) -> dict:
    """Public proposed-estimator entry point for the PLLG geometry seed."""
    pllg = _solve_stage2_pllg_backend(state, scene, config)
    return _stage2_solution_from_polish(
        state,
        scene,
        pllg,
        geometry_seed_impl="pllg",
        seed_failure_reason=str(pllg.get("failure_reason", "")),
    )


def solve_stage2_rescue(
    state: Stage2CommonState,
    scene: dict,
    config: dict,
    impl: str | None = None,
) -> dict:
    """Dispatch only the geometry seed solver after common preprocessing."""
    selected_impl = str(impl or config.get("stage2_rescue_impl", "legacy_multistart")).lower()
    if selected_impl == "legacy_multistart":
        return solve_stage2_legacy_multistart(state, scene, config)
    if selected_impl != "pllg":
        raise ValueError(f"unknown stage2_rescue_impl {selected_impl!r}")
    pllg = _solve_stage2_pllg_backend(state, scene, config)
    failure = str(pllg.get("failure_reason", ""))
    max_projection = float(config.get("stage2_pllg_max_projection_distance_m", 0.05))
    if np.isfinite(float(pllg.get("diagnostics", {}).get("pllg_projection_distance_m", np.nan))) and float(pllg["diagnostics"].get("pllg_projection_distance_m", 0.0)) > max_projection:
        failure = "invalid_projected_seed"
        pllg["diagnostics"]["pllg_failure_reason"] = failure
    if not failure and bool(pllg.get("rescue_available", False)):
        return _stage2_solution_from_polish(state, scene, pllg, geometry_seed_impl="pllg")
    if bool(config.get("stage2_pllg_legacy_fallback", True)):
        fallback_start = time.perf_counter()
        legacy = solve_stage2_legacy_multistart(state, scene, config)
        fallback_runtime = time.perf_counter() - fallback_start
        legacy["diagnostics"].update(
            {
                "stage2_rescue_impl": "pllg",
                "pllg_success": False,
                "pllg_failure_reason": failure or "pllg_failure",
                "legacy_fallback_used": True,
                "legacy_fallback_reason": failure or "pllg_failure",
                "legacy_fallback_runtime_s": float(fallback_runtime),
            }
        )
        legacy["failure_reason"] = failure or "pllg_failure"
        return legacy
    return _stage2_solution_from_polish(
        state,
        scene,
        {"position": np.full(3, np.nan), "clock_s": np.nan, "rescue_available": False, "diagnostics": pllg.get("diagnostics", {}), "runtime_s": pllg.get("runtime_s", 0.0)},
        geometry_seed_impl="pllg",
        seed_failure_reason=failure or "pllg_failure",
    )


def _run_full_legacy_stage2(
    z_noisy: np.ndarray,
    scene: dict,
    config: dict,
    estimate_initial: dict,
) -> tuple[dict, dict, dict, float]:
    legacy_config = copy.deepcopy(config)
    legacy_config["stage2_mode"] = "full_legacy"
    legacy_config["stage2_enable_evs"] = True
    legacy_config["stage2_enable_delay"] = True
    legacy_config["stage2_enable_ris"] = True
    stage2_start = time.perf_counter()
    estimate_used, structured_diag = structured_refinement(
        z_noisy, scene, legacy_config, copy.deepcopy(estimate_initial)
    )
    return (
        estimate_used,
        structured_diag,
        legacy_config,
        time.perf_counter() - stage2_start,
    )


def _final_raw_objective(final: dict) -> float:
    for key in ("raw_objective_final", "raw_objective"):
        value = final.get(key)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_float):
            return value_float
    return float("nan")


def run_stage1_only(data: dict, config: dict) -> dict:
    """Run the Stage-I tensor/subspace initialization for one realization."""
    stage1_start = time.perf_counter()
    estimate = initialize_from_hankel(data["Z_noisy"], data["scene"], config)
    timing = {"stage1": time.perf_counter() - stage1_start}
    for key, value in estimate.items():
        if key.startswith("stage1_time_"):
            try:
                timing[key] = float(value)
            except (TypeError, ValueError):
                pass
    return {
        "estimate": estimate,
        "timing": timing,
        "stage1_config": config,
    }


def run_direct_vp_branch(
    data: dict,
    stage1_estimate: dict,
    config: dict,
    base_timing: dict,
    reliability: dict,
) -> dict:
    """Run the direct Stage-I initialized raw-domain VP-WNLS branch."""
    scene = data["scene"]
    final, vp_s = _run_global_vp_branch(
        data["Y_noisy"], copy.deepcopy(stage1_estimate), scene, config, "none"
    )
    timing = dict(base_timing)
    timing.update(
        {
            "stage2": 0.0,
            "ris_projection_total": 0.0,
            "vp": vp_s,
            "total": _common_timing_total(base_timing) + vp_s,
        }
    )
    return _make_branch_result(
        data=data,
        estimate_initial=stage1_estimate,
        estimate_used=copy.deepcopy(stage1_estimate),
        structured_diag=_empty_structured_diag(),
        final=final,
        timing=timing,
        stage1_config=config,
        branch_name="direct_vp",
        reliability=reliability,
    )


def run_ris_only_stage2_branch(
    data: dict,
    stage1_estimate: dict,
    config: dict,
    base_timing: dict,
    reliability: dict,
    *,
    common_state: Stage2CommonState | None = None,
) -> dict:
    """
    Run RIS-only Stage-II basin recovery followed by raw-domain VP-WNLS.

    RIS-only Stage-II is used only to move the initialization closer to the
    exact spherical RIS manifold before the final raw-domain VP-WNLS.
    """
    scene = data["scene"]
    if common_state is None:
        common_state = refine_stage2_ris_factors(
            data["Z_noisy"], scene, config, stage1_estimate
        )
    stage2_solution = solve_stage2_rescue(
        common_state,
        scene,
        config,
        impl=str(config.get("stage2_rescue_impl", "legacy_multistart")),
    )
    ris_estimate = stage2_solution["estimate"]
    structured_diag = dict(stage2_solution.get("diagnostics", {}))
    ris_config = stage2_solution.get("rescue_config", config)
    stage2_s = float(stage2_solution.get("runtime_s", 0.0))
    structured_diag.setdefault(
        "common_ris_refinement_runtime_s",
        float(common_state.common_refinement_runtime_s),
    )
    structured_diag.setdefault(
        "common_ris_refinement_success",
        bool(common_state.common_refinement_success),
    )
    structured_diag.setdefault(
        "common_ris_refinement_impl",
        str(config.get("stage2_ris_rescue_impl", FINAL_PROPOSED_RIS_RESCUE_IMPL)),
    )
    structured_diag.setdefault(
        "common_ris_refinement_num_valid_local_fixes",
        int(sum(bool(record.get("valid", False)) for record in common_state.local_fix_records)),
    )
    structured_diag["stage2_rescue_available"] = bool(
        stage2_solution.get("rescue_available", False)
    )
    structured_diag["stage2_total_runtime_s"] = float(stage2_s)
    structured_diag["stage2_failure_reason"] = str(
        stage2_solution.get("failure_reason", "")
    )
    if "p_u_true" in scene and "delta_t_true" in scene:
        seed_position = np.asarray(stage2_solution.get("position"), dtype=float).reshape(-1)
        if seed_position.size == 3 and np.all(np.isfinite(seed_position)):
            truth_position = np.asarray(scene["p_u_true"], dtype=float).reshape(3)
            structured_diag["seed_position_error_m"] = float(
                np.linalg.norm(seed_position - truth_position)
            )
            structured_diag["seed_z_error_m"] = float(seed_position[2] - truth_position[2])
        seed_clock = float(stage2_solution.get("clock", np.nan))
        if np.isfinite(seed_clock):
            structured_diag["seed_clock_error_s"] = float(
                seed_clock - float(scene["delta_t_true"])
            )
    final, vp_s = _run_global_vp_branch(
        data["Y_noisy"], ris_estimate, scene, ris_config, "ris_only"
    )
    timing = dict(base_timing)
    timing.update(
        {
            "stage2": stage2_s,
            "ris_projection_total": float(
                structured_diag.get("ris_projection_total_s", 0.0)
            ),
            "stage2_time_ris_codebook_build": float(
                structured_diag.get("stage2_time_ris_codebook_build", 0.0)
            ),
            "stage2_time_ris_correlation": float(
                structured_diag.get("stage2_time_ris_correlation", 0.0)
            ),
            "stage2_time_ris_warm_start": float(
                structured_diag.get("stage2_time_ris_warm_start", 0.0)
            ),
            "stage2_time_ris_fresnel_lift": float(
                structured_diag.get("stage2_time_ris_fresnel_lift", 0.0)
            ),
            "stage2_time_ris_refine": float(
                structured_diag.get("stage2_time_ris_refine", 0.0)
            ),
            "structured_refinement_total": float(
                structured_diag.get("structured_refinement_total", stage2_s)
            ),
            "per_iteration_total": float(
                structured_diag.get("per_iteration_total", 0.0)
            ),
            "projection_per_path_total": float(
                structured_diag.get("projection_per_path_total", 0.0)
            ),
            "global_Z_reconstruction_time": float(
                structured_diag.get("global_Z_reconstruction_time", 0.0)
            ),
            "guarded_SSE_time": float(structured_diag.get("guarded_SSE_time", 0.0)),
            "damping_grid_time": float(structured_diag.get("damping_grid_time", 0.0)),
            "factor_copy_time": float(structured_diag.get("factor_copy_time", 0.0)),
            "pseudo_inverse_time": float(
                structured_diag.get("pseudo_inverse_time", 0.0)
            ),
            "logging_time": float(structured_diag.get("logging_time", 0.0)),
            "deepcopy_time": float(structured_diag.get("deepcopy_time", 0.0)),
            "vp": vp_s,
            "total": _common_timing_total(base_timing) + stage2_s + vp_s,
        }
    )
    return _make_branch_result(
        data=data,
        estimate_initial=stage1_estimate,
        estimate_used=ris_estimate,
        structured_diag=structured_diag,
        final=final,
        timing=timing,
        stage1_config=ris_config,
        branch_name="ris_only_stage2_then_vp",
        reliability=reliability,
    )


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return default
    return value_float if np.isfinite(value_float) else default


def stage2_severe_unreliable(stage1_estimate: dict, reliability: dict, config: dict) -> dict:
    """Classify whether local RIS rescue is too narrow for the observed failure."""
    assignment_margin = _safe_float(
        reliability.get(
            "assignment_margin",
            stage1_estimate.get("stage1_assignment_margin"),
        )
    )
    rank1_ratio = _safe_float(stage1_estimate.get("stage1_max_rank1_ratio"))
    z_residual = _safe_float(stage1_estimate.get("initial_z_residual"))
    margin_bad = bool(
        np.isfinite(assignment_margin)
        and assignment_margin < float(config.get("mhr_assignment_margin_threshold", 0.3))
    )
    rank1_bad = bool(
        np.isfinite(rank1_ratio)
        and rank1_ratio > float(config.get("mhr_rank1_ratio_threshold", 0.9))
    )
    z_bad = bool(
        np.isfinite(z_residual)
        and z_residual > float(config.get("mhr_z_residual_threshold", 0.98))
    )
    severe = bool(margin_bad or rank1_bad or z_bad)
    return {
        "severe_unreliable": severe,
        "assignment_margin": assignment_margin,
        "rank1_ratio": rank1_ratio,
        "initial_z_residual": z_residual,
        "margin_bad": margin_bad,
        "rank1_bad": rank1_bad,
        "z_residual_bad": z_bad,
    }


def enumerate_top_assignment_hypotheses(stage1_estimate: dict, config: dict) -> list[dict]:
    """Enumerate candidate panel-to-column assignments from Stage-I costs."""
    costs = np.asarray(
        stage1_estimate.get(
            "assignment_costs_col_by_panel",
            stage1_estimate.get("assignment_costs", np.empty((0, 0))),
        ),
        dtype=float,
    )
    k_paths = int(np.asarray(stage1_estimate["poles"]).size)
    if costs.shape != (k_paths, k_paths) or not np.any(np.isfinite(costs)):
        costs = np.zeros((k_paths, k_paths), dtype=float)
    if np.any(np.isfinite(costs)):
        missing_cost = float(np.nanmax(costs[np.isfinite(costs)]) + 1.0)
    else:
        missing_cost = 1.0
    finite_costs = np.where(np.isfinite(costs), costs, missing_cost)
    if k_paths <= 6:
        permutations = itertools.permutations(range(k_paths))
    else:
        permutations = [tuple(range(k_paths))]

    poles = np.asarray(stage1_estimate["poles"])
    ris_eta = np.asarray(
        stage1_estimate.get("ris_eta", np.zeros((k_paths, 3))), dtype=float
    )
    c0 = float(config.get("c0", 299792458.0))
    delta_f = float(config.get("delta_f", 1.0))
    d_rb = np.asarray(config.get("d_RB", []), dtype=float)
    if d_rb.size != k_paths:
        d_rb = np.zeros(k_paths, dtype=float)
    clock_scale = float(
        config.get("mhr_clock_scale_s", config.get("assignment_clock_scale_s", 1.0e-9))
    )
    clock_weight = float(
        config.get("mhr_clock_weight", config.get("assignment_clock_weight", 0.5))
    )
    hypotheses = []
    for perm in permutations:
        # perm[panel] = current column used for that physical panel.
        assignment_score = float(
            sum(finite_costs[int(col), panel] for panel, col in enumerate(perm))
        )
        delta_t_k = []
        for panel, col in enumerate(perm):
            tau_hat = tau_from_pole(poles[int(col)], delta_f)
            range_hat = float(ris_eta[int(col), 0]) if ris_eta.ndim == 2 else 0.0
            delta_t_k.append(tau_hat - (range_hat + d_rb[panel]) / c0)
        delta_t_k = np.asarray(delta_t_k, dtype=float)
        clock_std = float(np.std(delta_t_k)) if delta_t_k.size else float("nan")
        clock_score = clock_std / max(clock_scale, config.get("eps", 1.0e-10))
        total_score = assignment_score + clock_weight * clock_score
        hypotheses.append(
            {
                "assignment": tuple(int(v) for v in perm),
                "assignment_score": assignment_score,
                "clock_std": clock_std,
                "delta_t_k": delta_t_k,
                "score": float(total_score),
            }
        )
    hypotheses.sort(key=lambda item: item["score"])
    max_keep = int(config.get("mhr_top_assignments", min(6, len(hypotheses))))
    return hypotheses[: max(1, min(max_keep, len(hypotheses)))]


def _mhr_eta_grid(
    center_eta: np.ndarray,
    panel: int,
    rank1_bad: bool,
    scene: dict,
    config: dict,
) -> np.ndarray:
    """Build a semi-global or widened local RIS reacquisition grid."""
    search = local_ris_search_config(scene, config, panel)
    num_range, num_elev, num_az = (
        int(v) for v in config.get("mhr_ris_grid", (7, 5, 9))
    )
    use_center = bool(config.get("mhr_use_current_eta_as_center", True))
    use_global = bool(config.get("mhr_allow_global_if_rank1_bad", True)) and rank1_bad
    if use_center and not use_global and np.all(np.isfinite(center_eta)):
        r_span = float(config.get("mhr_range_span", 1.5))
        a_span = float(config.get("mhr_angle_span", 0.35))
        r_bounds = (
            max(search["range_bounds"][0], float(center_eta[0]) - 0.5 * r_span),
            min(search["range_bounds"][1], float(center_eta[0]) + 0.5 * r_span),
        )
        e_bounds = (
            max(search["elev_bounds"][0], float(center_eta[1]) - 0.5 * a_span),
            min(search["elev_bounds"][1], float(center_eta[1]) + 0.5 * a_span),
        )
        a_bounds = (
            float(center_eta[2]) - 0.5 * a_span,
            float(center_eta[2]) + 0.5 * a_span,
        )
    else:
        r_bounds = search["range_bounds"]
        e_bounds = search["elev_bounds"]
        a_bounds = search["az_bounds"]
    ranges = np.linspace(r_bounds[0], r_bounds[1], max(num_range, 1))
    elevs = np.linspace(e_bounds[0], e_bounds[1], max(num_elev, 1))
    azs = np.linspace(a_bounds[0], a_bounds[1], max(num_az, 1))
    return np.asarray(list(itertools.product(ranges, elevs, azs)), dtype=float)


def generate_ris_reacquisition_candidates(
    stage1_estimate: dict,
    scene: dict,
    assignment: dict,
    config: dict,
) -> tuple[list[list[dict]], dict]:
    """Generate top exact-response RIS candidates for one assignment hypothesis."""
    start = time.perf_counter()
    k_paths = scene["K"]
    top_q = int(config.get("mhr_top_ris_candidates_per_path", 3))
    eps = float(config.get("eps", 1.0e-10))
    c_pool = np.asarray(stage1_estimate["C"], dtype=complex)
    eta_pool = np.asarray(stage1_estimate["ris_eta"], dtype=float)
    rank1_ratios = np.asarray(
        stage1_estimate.get("stage1_rank1_ratios", np.zeros(k_paths)),
        dtype=float,
    )
    rank1_threshold = float(config.get("mhr_rank1_ratio_threshold", 0.9))
    candidates_by_panel: list[list[dict]] = []
    num_candidates = 0
    for panel, col in enumerate(assignment["assignment"]):
        col = int(col)
        proxy = c_pool[:, col]
        center_eta = eta_pool[col] if eta_pool.ndim == 2 else np.full(3, np.nan)
        rank1_bad = bool(
            rank1_ratios.size > col
            and np.isfinite(rank1_ratios[col])
            and rank1_ratios[col] > rank1_threshold
        )
        grid = _mhr_eta_grid(center_eta, panel, rank1_bad, scene, config)
        panel_candidates = []
        tau_hat = tau_from_pole(stage1_estimate["poles"][col], scene["delta_f"])
        for eta in grid:
            h_vec = compressed_exact_response(
                eta,
                scene["Omega"][panel],
                scene["a_RB"][panel],
                scene["ris_grid"],
                scene["wavelength"],
            )
            residual, alpha = scaled_residual(proxy, h_vec, eps)
            relative = float(np.sqrt(residual / (np.linalg.norm(proxy) ** 2 + eps)))
            delta_t = tau_hat - (float(eta[0]) + scene["d_RB"][panel]) / scene["c0"]
            panel_candidates.append(
                {
                    "panel": panel,
                    "column": col,
                    "eta_local": eta.copy(),
                    "c": alpha * h_vec,
                    "alpha": alpha,
                    "relative_residual": relative,
                    "score": relative,
                    "delta_t": float(delta_t),
                    "rank1_bad_global_grid": rank1_bad,
                }
            )
        panel_candidates.sort(key=lambda item: item["score"])
        kept = panel_candidates[: max(1, top_q)]
        num_candidates += len(kept)
        candidates_by_panel.append(kept)
    return candidates_by_panel, {
        "mhr_candidate_generation_time": float(time.perf_counter() - start),
        "mhr_num_ris_candidates": int(num_candidates),
    }


def _mhr_build_estimate_from_hypothesis(
    stage1_estimate: dict,
    assignment: dict,
    candidate_tuple: tuple[dict, ...],
    scene: dict,
    config: dict,
) -> dict:
    """Build a physical-panel-ordered estimate for one MHRR hypothesis."""
    estimate = copy.deepcopy(stage1_estimate)
    k_paths = scene["K"]
    columns = np.asarray(assignment["assignment"], dtype=int)
    for key in ("A", "B", "Q", "C"):
        mat = np.asarray(stage1_estimate[key])
        estimate[key] = mat[:, columns].copy()
    for key in ("poles", "beta_z", "gamma", "eta_pol"):
        if key in stage1_estimate:
            estimate[key] = np.asarray(stage1_estimate[key])[columns].copy()
    ris_eta = np.empty((k_paths, 3), dtype=float)
    c_mat = np.asarray(estimate["C"], dtype=complex).copy()
    for panel, candidate in enumerate(candidate_tuple):
        ris_eta[panel] = candidate["eta_local"]
        c_mat[:, panel] = candidate["c"]
    estimate["C"] = c_mat
    estimate["ris_eta"] = ris_eta
    estimate["columns_are_panel_ordered"] = True
    estimate["mhr_assignment"] = tuple(int(v) for v in assignment["assignment"])
    estimate["mhr_ris_candidate_scores"] = np.array(
        [float(candidate["relative_residual"]) for candidate in candidate_tuple],
        dtype=float,
    )
    return estimate


def _mhr_build_global_hypotheses(
    stage1_estimate: dict,
    scene: dict,
    config: dict,
    assignments: list[dict],
) -> tuple[list[dict], dict]:
    """Combine assignment and per-panel RIS candidates into global hypotheses."""
    start = time.perf_counter()
    global_hypotheses = []
    total_ris_candidates = 0
    candidate_time = 0.0
    clock_scale = float(config.get("mhr_clock_scale_s", 1.0e-9))
    for assignment in assignments:
        candidates_by_panel, cand_diag = generate_ris_reacquisition_candidates(
            stage1_estimate, scene, assignment, config
        )
        total_ris_candidates += int(cand_diag["mhr_num_ris_candidates"])
        candidate_time += float(cand_diag["mhr_candidate_generation_time"])
        for candidate_tuple in itertools.product(*candidates_by_panel):
            clock_values = np.array(
                [float(candidate["delta_t"]) for candidate in candidate_tuple],
                dtype=float,
            )
            clock_std = float(np.std(clock_values)) if clock_values.size else float("nan")
            ris_score = float(
                np.mean([candidate["relative_residual"] for candidate in candidate_tuple])
            )
            score = (
                float(assignment["score"])
                + ris_score
                + clock_std / max(clock_scale, config.get("eps", 1.0e-10))
            )
            estimate = _mhr_build_estimate_from_hypothesis(
                stage1_estimate, assignment, candidate_tuple, scene, config
            )
            global_hypotheses.append(
                {
                    "estimate": estimate,
                    "assignment": assignment,
                    "ris_candidates": candidate_tuple,
                    "clock_std": clock_std,
                    "ris_score": ris_score,
                    "score": float(score),
                }
            )
    global_hypotheses.sort(key=lambda item: item["score"])
    max_hyp = int(config.get("mhr_max_global_hypotheses", 8))
    return global_hypotheses[: max(1, min(max_hyp, len(global_hypotheses)))], {
        "mhr_num_ris_candidates": int(total_ris_candidates),
        "mhr_candidate_generation_time": float(candidate_time),
        "mhr_hypothesis_build_time": float(time.perf_counter() - start),
    }


def run_short_vp_for_hypotheses(
    hypotheses: list[dict],
    y_noisy: np.ndarray,
    scene: dict,
    config: dict,
) -> tuple[list[dict], dict]:
    """Run short raw-domain VP probes and return the best full-VP candidates."""
    start = time.perf_counter()
    short_config = copy.deepcopy(config)
    short_config["vp_least_squares_max_nfev"] = int(
        config.get("mhr_short_vp_max_nfev", 5)
    )
    for hyp in hypotheses:
        hyp_start = time.perf_counter()
        final = global_exact_spherical_vp_refinement(
            y_noisy, hyp["estimate"], scene, short_config
        )
        hyp["short_final"] = final
        hyp["short_raw_objective"] = _final_raw_objective(final)
        hyp["short_success"] = bool(
            final.get("global_vp_success", final.get("optimizer", {}).get("success", False))
        )
        hyp["short_nfev"] = int(final.get("optimizer", {}).get("n_eval", 0))
        hyp["short_vp_time"] = float(time.perf_counter() - hyp_start)
    hypotheses.sort(
        key=lambda item: (
            not np.isfinite(item.get("short_raw_objective", float("nan"))),
            item.get("short_raw_objective", float("inf")),
            item["score"],
        )
    )
    keep = int(config.get("mhr_num_full_vp_candidates", 1))
    return hypotheses[: max(1, min(keep, len(hypotheses)))], {
        "mhr_short_vp_time": float(time.perf_counter() - start),
    }


def run_multi_hypothesis_ris_reacquisition_branch(
    data: dict,
    stage1_estimate: dict,
    config: dict,
    base_timing: dict,
    reliability: dict,
) -> dict:
    """Run severe-case multi-hypothesis RIS re-acquisition followed by VP."""
    total_start = time.perf_counter()
    scene = data["scene"]
    assignment_start = time.perf_counter()
    assignments = enumerate_top_assignment_hypotheses(stage1_estimate, config)
    assignment_time = time.perf_counter() - assignment_start
    hypotheses, hyp_diag = _mhr_build_global_hypotheses(
        stage1_estimate, scene, config, assignments
    )
    selected_hypotheses, short_diag = run_short_vp_for_hypotheses(
        hypotheses, data["Y_noisy"], scene, config
    )
    full_start = time.perf_counter()
    best_hypothesis = None
    best_final = None
    for hyp in selected_hypotheses:
        full = global_exact_spherical_vp_refinement(
            data["Y_noisy"], hyp["estimate"], scene, config
        )
        hyp["full_final"] = full
        hyp["full_raw_objective"] = _final_raw_objective(full)
        if best_final is None or hyp["full_raw_objective"] < _final_raw_objective(best_final):
            best_hypothesis = hyp
            best_final = full
    full_time = time.perf_counter() - full_start
    assert best_hypothesis is not None and best_final is not None
    total_time = time.perf_counter() - total_start
    structured_diag = {
        "stage2_rescue_mode": "multi_hypothesis_ris_reacquisition",
        "mhr_num_assignment_hypotheses": int(len(assignments)),
        "mhr_num_ris_candidates": int(hyp_diag["mhr_num_ris_candidates"]),
        "mhr_num_global_hypotheses": int(len(hypotheses)),
        "mhr_best_assignment": list(best_hypothesis["assignment"]["assignment"]),
        "mhr_best_clock_std_ns": float(best_hypothesis["clock_std"] * 1.0e9),
        "mhr_best_short_vp_objective": float(best_hypothesis["short_raw_objective"]),
        "mhr_best_full_vp_objective": float(best_hypothesis["full_raw_objective"]),
        "mhr_accepted": False,
        "mhr_runtime_total": float(total_time),
        "mhr_assignment_time": float(assignment_time),
        "mhr_candidate_generation_time": float(hyp_diag["mhr_candidate_generation_time"]),
        "mhr_short_vp_time": float(short_diag["mhr_short_vp_time"]),
        "mhr_full_vp_time": float(full_time),
        "updates": [],
        "z_hat_history": [],
        "residuals_noisy_rmse": [],
        "ris_projection_total_s": 0.0,
    }
    timing = dict(base_timing)
    timing.update(
        {
            "stage2": total_time,
            "ris_projection_total": 0.0,
            "vp": full_time,
            "total": _common_timing_total(base_timing) + total_time,
            "mhr_runtime_total": total_time,
            "mhr_assignment_time": assignment_time,
            "mhr_candidate_generation_time": hyp_diag["mhr_candidate_generation_time"],
            "mhr_short_vp_time": short_diag["mhr_short_vp_time"],
            "mhr_full_vp_time": full_time,
        }
    )
    return _make_branch_result(
        data=data,
        estimate_initial=stage1_estimate,
        estimate_used=best_hypothesis["estimate"],
        structured_diag=structured_diag,
        final=best_final,
        timing=timing,
        stage1_config=config,
        branch_name="multi_hypothesis_ris_reacquisition_then_vp",
        reliability=reliability,
    )


def run_full_legacy_comparison_branch(
    data: dict,
    stage1_estimate: dict,
    config: dict,
    base_timing: dict,
    reliability: dict,
) -> dict:
    """Run the full legacy EVS/delay/RIS Stage-II only as an explicit ablation."""
    scene = data["scene"]
    estimate, structured_diag, legacy_config, stage2_s = _run_full_legacy_stage2(
        data["Z_noisy"], scene, config, stage1_estimate
    )
    final, vp_s = _run_global_vp_branch(
        data["Y_noisy"], estimate, scene, legacy_config, "full_legacy"
    )
    timing = dict(base_timing)
    timing.update(
        {
            "stage2": stage2_s,
            "ris_projection_total": float(
                structured_diag.get("ris_projection_total_s", 0.0)
            ),
            "vp": vp_s,
            "total": _common_timing_total(base_timing) + stage2_s + vp_s,
        }
    )
    return _make_branch_result(
        data=data,
        estimate_initial=stage1_estimate,
        estimate_used=estimate,
        structured_diag=structured_diag,
        final=final,
        timing=timing,
        stage1_config=legacy_config,
        branch_name="full_legacy_comparison",
        reliability=reliability,
    )


def _branch_y_nmse(branch: dict) -> float:
    return float(relative_nmse(branch["final"]["Y_hat"], branch["Y_true"]))


def _candidate_metric_float(*values: Any) -> float:
    for value in values:
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_float):
            return value_float
    return float("nan")


def _candidate_position_error_from_estimate(
    result: dict,
    data: dict | None,
) -> float:
    final = result.get("final", {})
    p_hat = final.get("p_u", result.get("p_u"))
    p_true = None
    for source in (result, data or {}):
        if not isinstance(source, dict):
            continue
        scene = source.get("scene", {})
        if isinstance(scene, dict) and scene.get("p_u_true") is not None:
            p_true = scene.get("p_u_true")
            break
        if source.get("p_u_true") is not None:
            p_true = source.get("p_u_true")
            break
    if p_hat is None or p_true is None:
        return float("nan")
    try:
        return float(
            np.linalg.norm(
                np.asarray(p_hat, dtype=float) - np.asarray(p_true, dtype=float)
            )
        )
    except (TypeError, ValueError, FloatingPointError):
        return float("nan")


def _extract_branch_candidate_diagnostics(
    prefix: str,
    result: dict | None,
    data: dict | None = None,
) -> dict:
    """Return offline diagnostics for a candidate branch without affecting selection."""
    diagnostics = {
        f"{prefix}_candidate_position_error_m": float("nan"),
        f"{prefix}_candidate_y_nmse": float("nan"),
        f"{prefix}_candidate_raw_objective": float("nan"),
        f"{prefix}_candidate_lambda_jones_per_path": None,
        f"{prefix}_candidate_snr_eff_per_path": None,
        f"{prefix}_candidate_jones_leakage_per_path": None,
        f"{prefix}_candidate_data_only_scaled_efim_lambda_min": float("nan"),
        f"{prefix}_candidate_data_only_scaled_efim_condition_number": float("nan"),
    }
    if result is None:
        return diagnostics

    final = result.get("final", {})
    position_error = _candidate_metric_float(
        result.get("position_error_m"),
        result.get("debug_main_position_error_m"),
    )
    if not np.isfinite(position_error):
        position_error = _candidate_position_error_from_estimate(result, data)

    y_nmse = _candidate_metric_float(
        result.get("y_nmse"),
        result.get("debug_main_y_nmse"),
    )
    if not np.isfinite(y_nmse):
        y_hat = final.get("Y_hat")
        y_true = result.get("Y_true")
        if y_true is None and isinstance(data, dict):
            y_true = data.get("Y_true")
        if y_hat is not None and y_true is not None:
            try:
                y_nmse = float(relative_nmse(y_hat, y_true))
            except (TypeError, ValueError, FloatingPointError):
                y_nmse = float("nan")

    raw_objective = _candidate_metric_float(
        result.get("raw_objective_final"),
        result.get("debug_main_raw_objective"),
        final.get("raw_objective_final"),
        final.get("raw_objective"),
    )

    diagnostics[f"{prefix}_candidate_position_error_m"] = position_error
    diagnostics[f"{prefix}_candidate_y_nmse"] = y_nmse
    diagnostics[f"{prefix}_candidate_raw_objective"] = raw_objective
    diagnostics[f"{prefix}_candidate_lambda_jones_per_path"] = final.get(
        "lambda_jones_per_path"
    )
    diagnostics[f"{prefix}_candidate_snr_eff_per_path"] = final.get(
        "snr_eff_per_path"
    )
    diagnostics[f"{prefix}_candidate_jones_leakage_per_path"] = final.get(
        "jones_leakage_per_path"
    )
    diagnostics.update(_candidate_data_only_efim_diagnostics(prefix, result))
    return diagnostics


def _candidate_data_only_efim_diagnostics(prefix: str, result: dict) -> dict:
    """Return optional candidate EFIM diagnostics, best-effort only."""
    lambda_key = f"{prefix}_candidate_data_only_scaled_efim_lambda_min"
    cond_key = f"{prefix}_candidate_data_only_scaled_efim_condition_number"
    diagnostics = {
        lambda_key: float("nan"),
        cond_key: float("nan"),
    }
    for source in (
        result.get("direct_vp_quality", {}),
        result.get("final", {}),
        result,
    ):
        if not isinstance(source, dict):
            continue
        lambda_min = source.get("data_only_scaled_efim_lambda_min")
        condition = source.get("data_only_scaled_efim_condition_number")
        try:
            lambda_float = float(lambda_min)
            condition_float = float(condition)
        except (TypeError, ValueError):
            continue
        if np.isfinite(lambda_float) and np.isfinite(condition_float):
            diagnostics[lambda_key] = lambda_float
            diagnostics[cond_key] = condition_float
            return diagnostics

    try:
        final = result["final"]
        scene = result["scene"]
        config = result.get("stage1_config", {})
        sigma2 = result.get("noise_variance", _final_raw_objective(final))
        efim_diag = data_only_efim_diagnostic(
            result["Y_noisy"],
            final["p_u"],
            final["delta_t"],
            result["estimate_initial"],
            scene,
            config,
            sigma2=float(sigma2),
        )
        diagnostics[lambda_key] = float(
            efim_diag.get("data_only_scaled_efim_lambda_min", float("nan"))
        )
        diagnostics[cond_key] = float(
            efim_diag.get(
                "data_only_scaled_efim_condition_number", float("nan")
            )
        )
    except Exception:
        pass
    return diagnostics


def _rescue_candidate_selection_diagnostics(
    rescue_result: dict | None,
    selected_branch: str,
    reliability: dict,
    *,
    rescue_requested: bool,
) -> dict:
    if rescue_result is None:
        return {
            "rescue_candidate_available": False,
            "rescue_accept_decision": "not_run",
            "rescue_reject_reason": (
                "no_rescue_candidate" if rescue_requested else "not_requested"
            ),
        }

    rescue_branch = str(rescue_result.get("branch_name", "ris_only_stage2_then_vp"))
    rescue_branches = {
        rescue_branch,
        "ris_only_stage2_then_vp",
        "multi_hypothesis_ris_reacquisition_then_vp",
    }
    if selected_branch in rescue_branches:
        accept_decision = "accepted"
        reject_reason = "accepted"
    elif selected_branch == "direct_vp_rollback":
        accept_decision = "rollback"
        reject_reason = "existing_selector_rollback"
    elif str(reliability.get("decision", "")) == "direct_vp":
        accept_decision = "rollback"
        reject_reason = "gate_direct_vp_hard_reject"
    else:
        accept_decision = "unknown"
        reject_reason = "unknown"
    return {
        "rescue_candidate_available": True,
        "rescue_accept_decision": accept_decision,
        "rescue_reject_reason": reject_reason,
    }


def _normal_quantile(p: float) -> float:
    """Return a standard-normal quantile with scipy or a fixed fallback."""
    if scipy_is_available():
        from scipy.stats import norm

        return float(norm.ppf(p))
    # Acklam-style constants are overkill here; GOF only needs a stable fallback
    # for the default p=0.95 used by the asymptotic chi-square approximation.
    table = {
        0.90: 1.2815515655446004,
        0.95: 1.6448536269514722,
        0.975: 1.959963984540054,
        0.99: 2.3263478740408408,
    }
    nearest = min(table, key=lambda key: abs(key - p))
    return table[nearest]


def _chi_square_gate_threshold(p_fa: float, dof: int) -> float:
    """Return chi-square 1-p_fa threshold or a normal approximation."""
    q = 1.0 - float(p_fa)
    if scipy_is_available():
        from scipy.stats import chi2

        return float(chi2.ppf(q, int(dof)))
    z = _normal_quantile(q)
    return float(dof + np.sqrt(2.0 * max(dof, 1)) * z)


def _chi_square_quantile(q: float, dof: int) -> float:
    """Return chi-square q quantile for NGC clock certification."""
    dof = int(dof)
    if dof <= 0:
        return 0.0
    q = float(np.clip(q, 0.0, 1.0))
    if scipy_is_available():
        from scipy.stats import chi2

        return float(chi2.ppf(q, dof))
    z = _normal_quantile(q)
    return float(max(0.0, dof + np.sqrt(2.0 * dof) * z))


def _ngc_tau_crb_from_efim(
    branch_result: dict | None,
    scene: dict,
    config: dict,
) -> tuple[np.ndarray | None, str]:
    if branch_result is None or "final" not in branch_result:
        return None, ""
    final = branch_result["final"]
    if "p_u" not in final:
        return None, ""
    efim_diag = {}
    for source in (
        branch_result.get("direct_vp_quality", {}),
        branch_result.get("data_only_efim_diagnostic", {}),
        final,
        branch_result,
    ):
        if isinstance(source, dict) and "data_only_efim" in source:
            efim_diag = source
            break
    if not efim_diag:
        try:
            sigma2 = branch_result.get("noise_variance", _final_raw_objective(final))
            efim_diag = data_only_efim_diagnostic(
                branch_result["Y_noisy"],
                final["p_u"],
                final["delta_t"],
                branch_result["estimate_initial"],
                scene,
                config,
                sigma2=float(sigma2),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ):
            return None, ""
    try:
        j_tau = np.asarray(efim_diag["data_only_efim"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None, ""
    if j_tau.shape != (4, 4) or not np.all(np.isfinite(j_tau)):
        return None, ""
    cov = np.linalg.pinv((j_tau + j_tau.T) * 0.5)
    p_u = np.asarray(final["p_u"], dtype=float).reshape(3)
    sigmas = []
    for k in range(int(scene["K"])):
        diff = p_u - np.asarray(scene["ris_centers"][k], dtype=float)
        range_m = float(np.linalg.norm(diff))
        if not np.isfinite(range_m) or range_m <= 0.0:
            return None, ""
        grad = np.empty(4, dtype=float)
        grad[:3] = diff / (range_m * float(scene["c0"]))
        grad[3] = 1.0
        variance = float(grad @ cov @ grad)
        if not np.isfinite(variance) or variance < 0.0:
            return None, ""
        sigmas.append(float(np.sqrt(max(variance, 0.0))))
    arr = np.asarray(sigmas, dtype=float)
    if arr.size != int(scene["K"]) or not np.all(np.isfinite(arr)):
        return None, ""
    return arr, "data_only_efim_tau_crb"


def build_stage2_delay_uncertainty(
    stage1_estimate: dict,
    scene: dict,
    config: dict,
    *,
    efim_context: dict | None = None,
) -> dict:
    """Build explicit Stage-II delay uncertainty without mutating Stage-I."""
    k_paths = int(scene["K"])
    sigma, efim_source = _ngc_tau_crb_from_efim(efim_context, scene, config)
    floor_ns = float(config.get("stage2_delay_sigma_floor_ns", 0.5))
    floor_s = max(floor_ns * 1.0e-9, 1.0e-15)
    used_floor = False
    if sigma is None:
        sigma = np.full(k_paths, floor_s, dtype=float)
        source = "configured_floor"
        used_floor = True
    else:
        sigma = np.asarray(sigma, dtype=float).reshape(-1)
        if sigma.size != k_paths:
            sigma = np.full(k_paths, floor_s, dtype=float)
            source = "configured_floor"
            used_floor = True
        else:
            # Keep the specific provenance reported by the EFIM helper rather
            # than collapsing every EFIM-derived sigma to one generic label.
            source = efim_source or "ngc_efim"
            invalid = ~np.isfinite(sigma) | (sigma <= 0.0)
            if np.any(invalid):
                sigma = sigma.copy()
                sigma[invalid] = floor_s
                used_floor = True
    sigma = np.maximum(sigma, floor_s)
    used_floor = bool(used_floor or np.any(sigma <= floor_s))
    return {
        "sigma_tau_s": sigma,
        "sigma_tau_sq_s2": sigma * sigma,
        "source": source,
        "used_floor": used_floor,
    }


def _ngc_clock_sigmas_s(
    stage1_estimate: dict,
    k_paths: int,
    config: dict,
    branch_result: dict | None = None,
    scene: dict | None = None,
) -> tuple[np.ndarray, str]:
    if scene is None:
        floor_s = max(float(config.get("ngc_clock_sigma_floor_ns", 0.5)) * 1.0e-9, 1.0e-15)
        return np.full(k_paths, floor_s, dtype=float), "fallback_floor"
    ngc_config = dict(config)
    ngc_config["stage2_delay_sigma_floor_ns"] = float(
        config.get("ngc_clock_sigma_floor_ns", 0.5)
    )
    uncertainty = build_stage2_delay_uncertainty(
        stage1_estimate,
        scene,
        ngc_config,
        efim_context=branch_result,
    )
    return np.asarray(uncertainty["sigma_tau_s"], dtype=float), str(uncertainty["source"])


def _ngc_optional_threshold(config: dict, key: str) -> float:
    value = config.get(key)
    if value is None or value == "":
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _ngc_certificate(
    prefix: str,
    branch_result: dict | None,
    stage1_estimate: dict,
    scene: dict,
    config: dict,
) -> dict:
    k_paths = int(scene["K"])
    clock_dof = k_paths - 1
    q_green = _chi_square_quantile(
        float(config.get("ngc_clock_green_quantile", 0.99)), clock_dof
    )
    q_red = _chi_square_quantile(
        float(config.get("ngc_clock_red_quantile", 0.999)), clock_dof
    )
    base = {
        f"ngc_{prefix}_clock_score": float("nan"),
        f"ngc_{prefix}_clock_score_norm": float("nan"),
        f"ngc_{prefix}_clock_dof": int(clock_dof),
        f"ngc_{prefix}_clock_sigma_source": "",
        f"ngc_{prefix}_clock_std_ns": float("nan"),
        f"ngc_{prefix}_ris_score": float("nan"),
        f"ngc_{prefix}_ris_score_norm": float("nan"),
        f"ngc_{prefix}_ris_available": False,
        f"ngc_{prefix}_total_score": float("nan"),
        f"ngc_{prefix}_cert_status": "",
        f"ngc_{prefix}_cert_reason": "candidate_unavailable",
        "ngc_threshold_clock_green": float(q_green),
        "ngc_threshold_clock_red": float(q_red),
    }
    if branch_result is None or "final" not in branch_result:
        return base
    final = branch_result["final"]
    if "p_u" not in final:
        return base
    if clock_dof <= 0:
        base.update(
            {
                f"ngc_{prefix}_clock_dof": int(clock_dof),
                f"ngc_{prefix}_cert_status": "not_applicable",
                f"ngc_{prefix}_cert_reason": "clock_not_applicable_k_lt_2",
            }
        )
        return base

    p_u = np.asarray(final["p_u"], dtype=float).reshape(3)
    tau_stage1, _, _, _ = _stage1_clock_panel_order(stage1_estimate, scene)
    sigmas_s, sigma_source = _ngc_clock_sigmas_s(
        stage1_estimate,
        k_paths,
        config,
        branch_result=branch_result,
        scene=scene,
    )
    weights = 1.0 / np.maximum(sigmas_s**2, 1.0e-30)
    delta_t_hat = np.empty(k_paths, dtype=float)
    for k in range(k_paths):
        d_ur = float(np.linalg.norm(p_u - np.asarray(scene["ris_centers"][k], dtype=float)))
        delta_t_hat[k] = float(tau_stage1[k] - (d_ur + scene["d_RB"][k]) / scene["c0"])
    weight_sum = float(np.sum(weights))
    if weight_sum > 0.0 and np.all(np.isfinite(delta_t_hat)):
        delta_bar = float(np.sum(weights * delta_t_hat) / weight_sum)
        residual = delta_t_hat - delta_bar
        clock_score = float(np.sum(weights * residual**2))
        clock_std_ns = float(np.std(delta_t_hat) * 1.0e9)
    else:
        clock_score = float("nan")
        clock_std_ns = float("nan")

    ris_diag = robust_jnpp_geometry_consistency_score(
        p_u, stage1_estimate, scene, config
    )
    ris_available = bool(ris_diag.get("available", False))
    ris_score = float(ris_diag.get("score", float("nan")))
    ris_score_norm = float(ris_diag.get("score_norm", float("nan")))
    lambda_ris = float(config.get("ngc_lambda_ris", 1.0))
    total_score = clock_score
    reasons = []
    if ris_available and np.isfinite(ris_score_norm):
        total_score = float(total_score + lambda_ris * ris_score_norm)
    else:
        reasons.append("clock_only_no_ris_score")

    ris_green_threshold = _ngc_optional_threshold(config, "ngc_ris_green_threshold")
    ris_red_threshold = _ngc_optional_threshold(config, "ngc_ris_red_threshold")
    clock_green = bool(np.isfinite(clock_score) and clock_score <= q_green)
    clock_red = bool(np.isfinite(clock_score) and clock_score >= q_red)
    if not np.isfinite(clock_score):
        status = "gray"
        reasons.append("clock_score_unavailable")
    else:
        ris_green = True
        ris_red = False
        if ris_available and np.isfinite(ris_score_norm):
            if np.isfinite(ris_green_threshold):
                ris_green = bool(ris_score_norm <= ris_green_threshold)
            else:
                reasons.append("ris_green_threshold_unset")
            if np.isfinite(ris_red_threshold):
                ris_red = bool(ris_score_norm > ris_red_threshold)
            else:
                reasons.append("ris_red_threshold_unset")
        if clock_red or ris_red:
            status = "red"
        elif clock_green and ris_green:
            status = "green"
        else:
            status = "gray"
    if not reasons:
        reasons.append(f"clock_{status}")
    base.update(
        {
            f"ngc_{prefix}_clock_score": clock_score,
            f"ngc_{prefix}_clock_score_norm": clock_score,
            f"ngc_{prefix}_clock_dof": int(clock_dof),
            f"ngc_{prefix}_clock_sigma_source": sigma_source,
            f"ngc_{prefix}_clock_std_ns": clock_std_ns,
            f"ngc_{prefix}_ris_score": ris_score,
            f"ngc_{prefix}_ris_score_norm": ris_score_norm,
            f"ngc_{prefix}_ris_available": ris_available,
            f"ngc_{prefix}_total_score": total_score,
            f"ngc_{prefix}_cert_status": status,
            f"ngc_{prefix}_cert_reason": ",".join(reasons),
        }
    )
    return base


def _ngc_base_diagnostics(config: dict) -> dict:
    return {
        "ngc_policy_active": False,
        "ngc_lambda_ris": float(config.get("ngc_lambda_ris", 1.0)),
        "ngc_rescue_requested": False,
        "ngc_rescue_request_reason": "",
        "ngc_selected_by": "",
        "ngc_final_unreliable": False,
    }


def evaluate_direct_vp_quality(
    direct_result: dict,
    stage1_result: dict,
    scene: dict,
    config: dict,
) -> dict:
    """Return observable direct-VP quality diagnostics for rescue gating."""
    _ = stage1_result
    final = direct_result["final"]
    optimizer = final.get("optimizer", {})
    success = bool(final.get("global_vp_success", optimizer.get("success", False)))
    raw_final = _final_raw_objective(final)
    raw_initial = final.get("raw_objective_initial")
    try:
        raw_initial_float = float(raw_initial)
    except (TypeError, ValueError):
        raw_initial_float = float("nan")
    finite_objective = bool(np.isfinite(raw_final))
    if np.isfinite(raw_initial_float) and raw_initial_float > 0.0:
        rel_decrease = float((raw_initial_float - raw_final) / raw_initial_float)
        objective_decreased = rel_decrease >= float(
            config.get("direct_vp_min_rel_residual_decrease", 1.0e-4)
        )
    else:
        rel_decrease = float("nan")
        objective_decreased = finite_objective

    nfev = int(optimizer.get("n_eval", final.get("global_vp_num_iter", 10**9)))
    nfev_good = nfev <= int(config.get("direct_vp_max_good_nfev", 12))
    noise_variance = direct_result.get("noise_variance")
    if noise_variance is None:
        noise_floor_good = True
        noise_floor_threshold = float("nan")
    else:
        noise_floor_threshold = float(config.get("direct_vp_noise_floor_factor", 1.5)) * float(
            noise_variance
        )
        noise_floor_good = bool(raw_final <= noise_floor_threshold)
    global_vp_options = dict(config.get("global_vp", {}))
    k_paths = int(scene["K"])
    sigma2_hat = float(noise_variance) if noise_variance is not None else raw_final
    p_fa = float(global_vp_options.get("gof_pfa", 0.05))
    gof_dof = max(1, 2 * int(scene["I"]) * int(scene["N"]) * int(scene["T"]) - 4 * k_paths - 4)
    gof_stat = float(2.0 * raw_final * final["Y_hat"].size / max(sigma2_hat, config.get("eps", 1.0e-10)))
    gof_threshold = _chi_square_gate_threshold(p_fa, gof_dof)
    gof_pass = bool(np.isfinite(gof_stat) and gof_stat <= gof_threshold)

    efim_diag = {}
    if bool(global_vp_options.get("use_data_only_efim_gate", True)):
        try:
            efim_diag = data_only_efim_diagnostic(
                direct_result["Y_noisy"],
                final["p_u"],
                final["delta_t"],
                direct_result["estimate_initial"],
                scene,
                config,
                sigma2=sigma2_hat,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            efim_diag = {
                "data_only_efim_lambda_min": 0.0,
                "data_only_efim_condition_number": float("inf"),
                "data_only_scaled_efim_lambda_min": 0.0,
                "data_only_scaled_efim_condition_number": float("inf"),
                "data_only_efim_error": str(exc),
            }
    lambda_min = float(efim_diag.get("data_only_scaled_efim_lambda_min", efim_diag.get("data_only_efim_lambda_min", float("nan"))))
    efim_cond = float(efim_diag.get("data_only_scaled_efim_condition_number", efim_diag.get("data_only_efim_condition_number", float("inf"))))
    well_conditioned = bool(
        np.isfinite(lambda_min)
        and lambda_min >= float(global_vp_options.get("efim_lambda_min_threshold", 1.0e-8))
        and np.isfinite(efim_cond)
        and efim_cond <= float(global_vp_options.get("efim_cond_threshold", 1.0e12))
    )
    if gof_pass:
        revised_decision = "direct_vp"
    elif well_conditioned:
        revised_decision = "jnpp_then_vp"
    else:
        revised_decision = "ill_conditioned"

    good = bool(
        success
        and finite_objective
        and gof_pass
    )
    return {
        "good": good,
        "success": success,
        "finite_objective": finite_objective,
        "raw_objective_initial": raw_initial_float,
        "raw_objective_final": raw_final,
        "relative_objective_decrease": rel_decrease,
        "objective_decreased": bool(objective_decreased),
        "nfev": nfev,
        "nfev_good": bool(nfev_good),
        "noise_floor_good": bool(noise_floor_good),
        "noise_floor_threshold": noise_floor_threshold,
        "gof_stat": gof_stat,
        "gof_dof": int(gof_dof),
        "gof_threshold": gof_threshold,
        "gof_pass": gof_pass,
        "data_only_efim_lambda_min": lambda_min,
        "data_only_efim_condition_number": efim_cond,
        "data_only_scaled_efim_lambda_min": lambda_min,
        "data_only_scaled_efim_condition_number": efim_cond,
        "data_only_efim_well_conditioned": well_conditioned,
        "reliability_decision": revised_decision,
        **efim_diag,
    }


def select_proposed_branch(
    direct_result: dict,
    rescue_result: dict | None,
    reliability: dict,
    config: dict,
) -> tuple[dict, bool]:
    """Select the proposed output using the final raw-domain objective."""
    bounds = np.asarray(config["ue_bounds"], dtype=float)
    vp_options = dict(config.get("global_vp", {}))
    boundary_tol = float(vp_options.get("boundary_tol_m", 0.02))
    boundary_rel_tol = float(vp_options.get("boundary_accept_rel_tol", 1.0e-3))
    direct_boundary = (
        distance_to_box_boundary(
            direct_result["final"]["p_u"], bounds, boundary_tol
        )
        if "p_u" in direct_result["final"]
        else {"boundary_hit": False}
    )
    rescue_boundary = (
        distance_to_box_boundary(
            rescue_result["final"]["p_u"], bounds, boundary_tol
        )
        if rescue_result is not None and "p_u" in rescue_result["final"]
        else {"boundary_hit": False}
    )
    boundary_rule_used = False
    warning = ""
    branch_score_margin = float("nan")

    if rescue_result is None:
        selected = dict(direct_result)
        selected_branch = "direct_vp"
        no_gain = False
    else:
        rel_gain = float(config.get("rescue_accept_min_rel_improvement", 1.0e-3))
        abs_gain = float(config.get("rescue_accept_min_abs_improvement", 1.0e-8))
        direct_raw = _final_raw_objective(direct_result["final"])
        rescue_raw = _final_raw_objective(rescue_result["final"])
        branch_score_margin = float(rescue_raw - direct_raw)
        if np.isfinite(direct_raw) and np.isfinite(rescue_raw):
            improvement = direct_raw - rescue_raw
            accept_rescue = (
                improvement >= abs_gain
                and rescue_raw < direct_raw * (1.0 - rel_gain)
            )
        else:
            direct_nmse = _branch_y_nmse(direct_result)
            rescue_nmse = _branch_y_nmse(rescue_result)
            improvement = direct_nmse - rescue_nmse
            accept_rescue = (
                improvement >= abs_gain
                and rescue_nmse < direct_nmse * (1.0 - rel_gain)
            )

        if direct_boundary["boundary_hit"] and not rescue_boundary["boundary_hit"]:
            boundary_rule_used = True
            accept_rescue = bool(
                np.isfinite(direct_raw)
                and np.isfinite(rescue_raw)
                and rescue_raw <= direct_raw * (1.0 + boundary_rel_tol) + 1.0e-15
            )
            if not accept_rescue:
                warning = "boundary_solution_retained_due_to_lower_residual"

        if reliability["decision"] == "direct_vp" and not boundary_rule_used:
            accept_rescue = False

        # Stage-II is accepted only if it improves the final raw-domain objective.
        # This keeps the diagnostic consistent with the final estimator objective.
        if accept_rescue:
            selected = dict(rescue_result)
            selected_branch = str(
                rescue_result.get("branch_name", "ris_only_stage2_then_vp")
            )
            no_gain = False
        else:
            selected = dict(direct_result)
            selected_branch = "direct_vp_rollback"
            no_gain = True

    selected["selected_branch"] = selected_branch
    selected["reliability"] = reliability
    selected["ris_stage2_no_gain"] = bool(no_gain)
    selected["direct_boundary_hit"] = bool(direct_boundary["boundary_hit"])
    selected["rescue_boundary_hit"] = bool(rescue_boundary["boundary_hit"])
    selected["branch_score_margin"] = branch_score_margin
    selected["boundary_selection_rule_used"] = bool(boundary_rule_used)
    selected["warning"] = warning
    if rescue_result is not None:
        rescue_diag = rescue_result.get("structured_diag", {})
        if "mhr_accepted" in rescue_diag:
            rescue_diag["mhr_accepted"] = bool(
                selected_branch == "multi_hypothesis_ris_reacquisition_then_vp"
            )
        if "jnpp_accepted_by_raw_vp" in rescue_diag:
            rescue_diag["jnpp_accepted_by_raw_vp"] = bool(
                selected_branch == rescue_result.get("branch_name")
            )
    selected["final"] = dict(selected["final"])
    selected["final"]["selected_branch"] = selected_branch
    selected["final"]["reliability"] = reliability
    selected["final"].update(
        {
            "direct_boundary_hit": bool(direct_boundary["boundary_hit"]),
            "rescue_boundary_hit": bool(rescue_boundary["boundary_hit"]),
            "branch_score_margin": branch_score_margin,
            "boundary_selection_rule_used": bool(boundary_rule_used),
            "warning": warning,
        }
    )
    return selected, bool(no_gain)


def select_ngc_branch(
    direct_result: dict,
    rescue_result: dict | None,
    reliability: dict,
    ngc_diagnostics: dict,
    config: dict | None = None,
) -> tuple[dict, bool]:
    """Select direct/rescue candidate using the NGC certification policy."""
    config = config or {}
    direct_status = str(ngc_diagnostics.get("ngc_direct_cert_status", ""))
    rescue_status = str(ngc_diagnostics.get("ngc_rescue_cert_status", ""))
    direct_green = direct_status == "green"
    rescue_green = rescue_status == "green"
    direct_raw = _final_raw_objective(direct_result.get("final", {}))
    rescue_raw = (
        _final_raw_objective(rescue_result.get("final", {}))
        if rescue_result is not None
        else float("nan")
    )
    branch_score_margin = (
        float(rescue_raw - direct_raw)
        if np.isfinite(direct_raw) and np.isfinite(rescue_raw)
        else float("nan")
    )
    selector_guard_enabled = bool(config.get("stage2_selector_guard", True))
    selector_raw_degradation = (
        float(rescue_raw - direct_raw)
        if np.isfinite(direct_raw) and np.isfinite(rescue_raw)
        else float("nan")
    )
    selector_raw_relative_improvement = (
        float((direct_raw - rescue_raw) / max(abs(direct_raw), 1.0e-12))
        if np.isfinite(direct_raw) and np.isfinite(rescue_raw)
        else float("nan")
    )
    selector_boundary_guard_used = False
    selector_boundary_override_used = False
    rescue_candidate_admissible = True
    selector_guard_reject_reason = ""
    if selector_guard_enabled and rescue_result is not None:
        rescue_available = bool(rescue_result.get("structured_diag", {}).get(
            "stage2_rescue_available", rescue_result.get("rescue_available", True)
        ))
        if not rescue_available or not np.isfinite(rescue_raw):
            rescue_candidate_admissible = False
            selector_guard_reject_reason = "rescue_unavailable_or_nonfinite"
        elif np.isfinite(direct_raw):
            raw_tol = float(config.get("stage2_selector_raw_degradation_abs_tol", 1.0e-8))
            raw_tol += float(config.get("stage2_selector_raw_degradation_rel_tol", 1.0e-4)) * max(abs(direct_raw), 1.0e-12)
            if rescue_raw > direct_raw + raw_tol:
                rescue_candidate_admissible = False
                selector_guard_reject_reason = "raw_objective_degradation"
        direct_boundary = bool(
            direct_result.get("final", {}).get(
                "boundary_hit", direct_result.get("direct_boundary_hit", False)
            )
        )
        rescue_boundary = bool(
            rescue_result.get("final", {}).get(
                "boundary_hit", rescue_result.get("rescue_boundary_hit", False)
            )
        )
        if rescue_candidate_admissible and not direct_boundary and rescue_boundary:
            selector_boundary_guard_used = True
            min_improvement = float(
                config.get(
                    "stage2_selector_boundary_override_min_rel_improvement", 1.0e-3
                )
            )
            if not np.isfinite(selector_raw_relative_improvement) or selector_raw_relative_improvement < min_improvement:
                rescue_candidate_admissible = False
                selector_guard_reject_reason = "boundary_without_required_raw_improvement"
            else:
                selector_boundary_override_used = True
        if not rescue_candidate_admissible:
            rescue_result = None
    ngc_diagnostics.update(
        {
            "rescue_candidate_admissible": bool(rescue_candidate_admissible),
            "selector_guard_reject_reason": selector_guard_reject_reason,
            "selector_raw_degradation": selector_raw_degradation,
            "selector_raw_relative_improvement": selector_raw_relative_improvement,
            "selector_boundary_guard_used": bool(selector_boundary_guard_used),
            "selector_boundary_override_used": bool(selector_boundary_override_used),
        }
    )
    final_unreliable = False

    if rescue_result is None:
        selected = dict(direct_result)
        selected_branch = "direct_vp"
        selected_by = str(
            ngc_diagnostics.get(
                "ngc_selected_by",
                "rescue_unavailable_direct_fallback",
            )
        )
        if not selected_by:
            selected_by = "rescue_unavailable_direct_fallback"
        final_unreliable = direct_status != "green"
        no_gain = False
    elif direct_green and not rescue_green:
        selected = dict(direct_result)
        selected_branch = "direct_vp_rollback"
        selected_by = "ngc_certified_candidate"
        no_gain = True
    elif rescue_green and not direct_green:
        selected = dict(rescue_result)
        selected_branch = str(rescue_result.get("branch_name", "ris_only_stage2_then_vp"))
        selected_by = "ngc_certified_candidate"
        no_gain = False
    elif direct_green and rescue_green:
        if np.isfinite(rescue_raw) and (
            not np.isfinite(direct_raw) or rescue_raw < direct_raw
        ):
            selected = dict(rescue_result)
            selected_branch = str(rescue_result.get("branch_name", "ris_only_stage2_then_vp"))
            no_gain = False
        else:
            selected = dict(direct_result)
            selected_branch = "direct_vp_rollback"
            no_gain = True
        selected_by = "both_green_lower_raw"
    else:
        final_unreliable = True
        if np.isfinite(rescue_raw) and (
            not np.isfinite(direct_raw) or rescue_raw < direct_raw
        ):
            selected = dict(rescue_result)
            selected_branch = str(rescue_result.get("branch_name", "ris_only_stage2_then_vp"))
            no_gain = False
        else:
            selected = dict(direct_result)
            selected_branch = "direct_vp_rollback"
            no_gain = True
        selected_by = "both_uncertified_lower_raw"

    ngc_diagnostics["ngc_selected_by"] = selected_by
    ngc_diagnostics["ngc_final_unreliable"] = bool(final_unreliable)
    selected["selected_branch"] = selected_branch
    selected["reliability"] = reliability
    selected["ris_stage2_no_gain"] = bool(no_gain)
    selected["branch_score_margin"] = branch_score_margin
    selected["boundary_selection_rule_used"] = False
    selected["warning"] = selected.get("warning", "")
    selected["final"] = dict(selected["final"])
    selected["final"]["selected_branch"] = selected_branch
    selected["final"]["reliability"] = reliability
    selected["final"].update(
        {
            "branch_score_margin": branch_score_margin,
            "boundary_selection_rule_used": False,
            "warning": selected["warning"],
            "rescue_candidate_admissible": bool(rescue_candidate_admissible),
            "selector_guard_reject_reason": selector_guard_reject_reason,
            "selector_raw_degradation": selector_raw_degradation,
            "selector_raw_relative_improvement": selector_raw_relative_improvement,
            "selector_boundary_guard_used": bool(selector_boundary_guard_used),
            "selector_boundary_override_used": bool(selector_boundary_override_used),
        }
    )
    return selected, bool(no_gain)


def run_from_existing_stage1(
    data: dict,
    stage1: dict,
    config: dict,
    allow_stage2: bool = True,
    *,
    total_start: float | None = None,
) -> dict:
    """Run the proposed post-Stage-I pipeline for an existing realization."""
    if total_start is None:
        total_start = time.perf_counter()
    config = _apply_main_single_defaults(copy.deepcopy(config))
    try:
        noise_variance = float(data.get("noise_variance", np.nan))
    except (TypeError, ValueError):
        noise_variance = float("nan")
    if np.isfinite(noise_variance) and noise_variance > 0.0:
        config.setdefault("noise_variance", noise_variance)
    base_timing = dict(data.get("timing", {}))
    stage1_estimate = stage1["estimate"]
    base_timing.update(stage1["timing"])
    stage1_profile_reference = None
    if (
        bool(config.get("verbose_timing", False))
        and str(config.get("stage1_init_mode")) == "paper_balanced"
    ):
        heavy_config = copy.deepcopy(config)
        apply_stage1_init_preset(heavy_config, "normal_heavy")
        heavy_stage1 = run_stage1_only(data, heavy_config)
        stage1_profile_reference = {
            "mode": "normal_heavy",
            "stage1_s": float(heavy_stage1["timing"]["stage1"]),
        }
        base_timing["stage1_normal_heavy_reference"] = stage1_profile_reference[
            "stage1_s"
        ]
    reliability = compute_stage1_reliability(stage1_estimate, data["scene"], config)
    progress_printed = bool(config.get("print_progress", True))
    if progress_printed:
        _print_reliability_progress(reliability)
        if _paper_balanced_good_snr_triggered_stage2(config, reliability):
            print("WARNING_PAPER_BALANCED_TRIGGERED_STAGE2_AT_GOOD_SNR", flush=True)
        print("running_direct_vp_branch = True", flush=True)

    geometry_trigger_names = {
        "low_assignment_margin",
        "poor_clock_consistency",
        "large_ris_residual",
    }
    stage1_geometry_trigger_reasons = [
        str(reason)
        for reason in reliability.get("trigger_reasons", [])
        if str(reason) in geometry_trigger_names
    ]
    stage1_geometry_trigger = bool(stage1_geometry_trigger_reasons)
    stage2_rescue_enabled = (
        allow_stage2
        and bool(config.get("stage2_adaptive", True))
        and str(config.get("stage2_rescue_type", "ris_only")) == "ris_only"
    )
    force_stage2_for_diagnostics = bool(
        config.get("stage2_force_run_for_diagnostics", False)
        and stage2_rescue_enabled
    )
    branch_timing = {
        "direct_probe_branch_s": 0.0,
        "direct_probe_vp_s": 0.0,
        "direct_forced_z_rescue_branch_s": 0.0,
        "direct_forced_z_rescue_vp_s": 0.0,
        "rescue_branch_s": 0.0,
        "rescue_stage2_s": 0.0,
        "rescue_vp_s": 0.0,
    }
    direct_z_rescue_rerun_executed = False
    direct_z_rescue_rerun_skipped = False
    direct_z_rescue_skip_reason = ""
    direct_probe_z_rescue_disabled = bool(stage1_geometry_trigger and stage2_rescue_enabled)
    direct_probe_config = config
    if direct_probe_z_rescue_disabled:
        direct_probe_config = copy.deepcopy(config)
        direct_probe_config["global_vp"] = dict(direct_probe_config.get("global_vp", {}))
        direct_probe_config["global_vp"]["enable_z_rescue_multistart"] = False
        if progress_printed:
            print(
                "direct_probe_z_rescue_disabled = True "
                "reason=stage1_geometry_trigger_rescue_path",
                flush=True,
            )

    # Good initialization: avoid unnecessary factor-domain projection.
    direct_probe_start = time.perf_counter()
    direct_result = run_direct_vp_branch(
        data, stage1_estimate, direct_probe_config, base_timing, reliability
    )
    branch_timing["direct_probe_branch_s"] = time.perf_counter() - direct_probe_start
    branch_timing["direct_probe_vp_s"] = float(
        direct_result.get("timing", {}).get("vp", 0.0)
    )
    direct_vp_quality = evaluate_direct_vp_quality(
        direct_result, stage1, data["scene"], config
    )
    vp_options = dict(config.get("global_vp", {}))
    trigger_mode = str(
        vp_options.get("z_rescue_trigger", "boundary_or_unreliable")
    ).lower()
    direct_z_rescue_rerun_requested = (
        bool(vp_options.get("enable_z_rescue_multistart", True))
        and "unreliable" in trigger_mode
        and not bool(direct_result["final"].get("z_rescue_triggered", False))
        and not bool(direct_vp_quality.get("good", False))
    )
    skip_direct_z_rescue_for_geometry = bool(
        direct_z_rescue_rerun_requested
        and stage2_rescue_enabled
        and stage1_geometry_trigger
    )
    if direct_z_rescue_rerun_requested and skip_direct_z_rescue_for_geometry:
        direct_z_rescue_rerun_skipped = True
        direct_z_rescue_skip_reason = "stage1_geometry_trigger_rescue_path"
        if progress_printed:
            print(
                "skipping_direct_forced_z_rescue_rerun = True "
                f"reason={direct_z_rescue_skip_reason}",
                flush=True,
            )
    elif direct_z_rescue_rerun_requested:
        rescue_config = copy.deepcopy(config)
        rescue_config["_global_vp_force_z_rescue"] = True
        forced_direct_start = time.perf_counter()
        direct_result = run_direct_vp_branch(
            data, stage1_estimate, rescue_config, base_timing, reliability
        )
        direct_z_rescue_rerun_executed = True
        branch_timing["direct_forced_z_rescue_branch_s"] = (
            time.perf_counter() - forced_direct_start
        )
        branch_timing["direct_forced_z_rescue_vp_s"] = float(
            direct_result.get("timing", {}).get("vp", 0.0)
        )
        direct_vp_quality = evaluate_direct_vp_quality(
            direct_result, stage1, data["scene"], config
        )
    reliability = dict(reliability)
    legacy_stage1_decision = str(reliability.get("decision", "unknown"))
    gof_reliability_decision = str(
        direct_vp_quality.get("reliability_decision", "direct_vp")
    )
    reliability["legacy_stage1_decision"] = legacy_stage1_decision
    reliability["gof_reliability_decision"] = gof_reliability_decision
    reliability["stage1_geometry_trigger"] = stage1_geometry_trigger
    reliability["stage1_geometry_trigger_reasons"] = stage1_geometry_trigger_reasons
    reliability["direct_probe_z_rescue_disabled"] = direct_probe_z_rescue_disabled
    reliability["direct_z_rescue_rerun_executed"] = direct_z_rescue_rerun_executed
    reliability["direct_z_rescue_rerun_skipped"] = direct_z_rescue_rerun_skipped
    reliability["direct_z_rescue_skip_reason"] = direct_z_rescue_skip_reason
    reliability["decision"] = gof_reliability_decision
    reliability.update(
        {
            "gof_stat": direct_vp_quality.get("gof_stat"),
            "gof_dof": direct_vp_quality.get("gof_dof"),
            "gof_threshold": direct_vp_quality.get("gof_threshold"),
            "gof_pass": direct_vp_quality.get("gof_pass"),
            "data_only_efim_lambda_min": direct_vp_quality.get(
                "data_only_efim_lambda_min"
            ),
            "data_only_efim_condition_number": direct_vp_quality.get(
                "data_only_efim_condition_number"
            ),
            "data_only_efim_well_conditioned": direct_vp_quality.get(
                "data_only_efim_well_conditioned"
            ),
            "data_only_scaled_efim_lambda_min": direct_vp_quality.get(
                "data_only_scaled_efim_lambda_min"
            ),
            "data_only_scaled_efim_condition_number": direct_vp_quality.get(
                "data_only_scaled_efim_condition_number"
            ),
        }
    )
    direct_result["reliability"] = reliability
    direct_result["final"] = dict(direct_result["final"])
    direct_result["final"]["reliability"] = reliability
    direct_result["direct_vp_quality"] = direct_vp_quality
    branches = {"direct_vp": direct_result}
    rescue_result = None
    severe_diag = stage2_severe_unreliable(stage1_estimate, reliability, config)
    direct_vp_first = bool(config.get("direct_vp_first", True))
    direct_vp_override = bool(
        direct_vp_first
        and direct_vp_quality["good"]
        and reliability["decision"] != "direct_vp"
    )
    stage2_policy = str(
        config.get("proposed_stage2_policy", "ngc_certified_ris_only")
    ).lower()
    valid_stage2_policies = {
        "reliability_gated",
        "reliability_gated_ris_only",
        "force_ris_only",
        "geometry_gated_ris_only",
        "ngc_certified_ris_only",
    }
    if stage2_policy not in valid_stage2_policies:
        raise ValueError(f"unknown proposed_stage2_policy {stage2_policy!r}")
    ngc_active = stage2_policy == "ngc_certified_ris_only"
    ngc_diagnostics = _ngc_base_diagnostics(config)
    if ngc_active:
        ngc_diagnostics["ngc_policy_active"] = True
        ngc_diagnostics.update(
            _ngc_certificate(
                "direct", direct_result, stage1_estimate, data["scene"], config
            )
        )
        direct_status = str(ngc_diagnostics.get("ngc_direct_cert_status", ""))
        reliability["proposed_stage2_policy"] = stage2_policy
        reliability["stage2_policy_forced"] = False
        if stage2_rescue_enabled and direct_status == "green" and not force_stage2_for_diagnostics:
            reliability["decision"] = "direct_vp"
            ngc_diagnostics["ngc_rescue_requested"] = False
            ngc_diagnostics[
                "ngc_rescue_request_reason"
            ] = "ngc_green_skip_rescue"
            ngc_diagnostics["ngc_selected_by"] = "ngc_green_skip_rescue"
            direct_vp_override = False
        elif stage2_rescue_enabled and direct_status == "green":
            reliability["decision"] = "direct_vp"
            ngc_diagnostics["ngc_rescue_requested"] = True
            ngc_diagnostics[
                "ngc_rescue_request_reason"
            ] = "stage2_force_run_for_diagnostics"
            ngc_diagnostics["ngc_selected_by"] = "stage2_force_run_for_diagnostics"
            direct_vp_override = False
        elif stage2_rescue_enabled:
            reliability["decision"] = "jnpp_then_vp"
            ngc_diagnostics["ngc_rescue_requested"] = True
            ngc_diagnostics["ngc_rescue_request_reason"] = (
                "ngc_red_run_rescue"
                if direct_status == "red"
                else "ngc_gray_run_rescue"
            )
            direct_vp_override = False
        else:
            reliability["proposed_stage2_policy"] = stage2_policy
            ngc_diagnostics[
                "ngc_rescue_request_reason"
            ] = "ngc_stage2_disabled_direct_fallback"
            ngc_diagnostics[
                "ngc_selected_by"
            ] = "rescue_unavailable_direct_fallback"
    elif stage2_policy == "force_ris_only" and stage2_rescue_enabled:
        reliability["decision"] = "jnpp_then_vp"
        reliability["proposed_stage2_policy"] = stage2_policy
        reliability["stage2_policy_forced"] = True
        direct_vp_override = False
    elif stage2_policy == "geometry_gated_ris_only" and stage2_rescue_enabled:
        if gof_reliability_decision == "jnpp_then_vp" or stage1_geometry_trigger:
            reliability["decision"] = "jnpp_then_vp"
        else:
            reliability["decision"] = gof_reliability_decision
        reliability["proposed_stage2_policy"] = stage2_policy
        reliability["stage2_policy_forced"] = False
    else:
        reliability["proposed_stage2_policy"] = stage2_policy
        reliability["stage2_policy_forced"] = False
    if direct_vp_override and progress_printed:
        print("GATE_OVERRIDE_DIRECT_VP_GOOD", flush=True)

    rescue_requested = (
        stage2_rescue_enabled
        and (
            reliability["decision"] == "jnpp_then_vp"
            or force_stage2_for_diagnostics
        )
        and not direct_vp_override
    )
    common_stage2_state = None
    if rescue_requested:
        rescue_impl = str(
            config.get("stage2_ris_rescue_impl", FINAL_PROPOSED_RIS_RESCUE_IMPL)
        )
        if rescue_impl == "robust_jnpp":
            rescue_mode = "robust_jnpp"
        elif severe_diag["severe_unreliable"] and bool(
            config.get("jnpp_assignment_aware", False)
        ):
            rescue_mode = "multi_hypothesis_ris_reacquisition"
        else:
            rescue_mode = "local_ris_rescue"
        if progress_printed:
            print("running_ris_only_stage2_rescue = True", flush=True)
            print(f"stage2_rescue_mode = {rescue_mode}", flush=True)
        # Poor but potentially recoverable initialization: use RIS-only projection
        # to improve the VP basin, then return to the same raw-domain VP-WNLS objective.
        if rescue_mode == "multi_hypothesis_ris_reacquisition":
            rescue_branch_start = time.perf_counter()
            rescue_result = run_multi_hypothesis_ris_reacquisition_branch(
                data, stage1_estimate, config, base_timing, reliability
            )
            branch_timing["rescue_branch_s"] = time.perf_counter() - rescue_branch_start
            branch_timing["rescue_stage2_s"] = float(
                rescue_result.get("timing", {}).get("stage2", 0.0)
            )
            branch_timing["rescue_vp_s"] = float(
                rescue_result.get("timing", {}).get("vp", 0.0)
            )
            branches["multi_hypothesis_ris_reacquisition_then_vp"] = rescue_result
        else:
            rescue_branch_start = time.perf_counter()
            common_stage2_state = refine_stage2_ris_factors(
                data["Z_noisy"],
                data["scene"],
                config,
                stage1_estimate,
                efim_context=direct_result,
            )
            rescue_result = run_ris_only_stage2_branch(
                data,
                stage1_estimate,
                config,
                base_timing,
                reliability,
                common_state=common_stage2_state,
            )
            branch_timing["rescue_branch_s"] = time.perf_counter() - rescue_branch_start
            branch_timing["rescue_stage2_s"] = float(
                rescue_result.get("timing", {}).get("stage2", 0.0)
            )
            branch_timing["rescue_vp_s"] = float(
                rescue_result.get("timing", {}).get("vp", 0.0)
            )
            rescue_result["structured_diag"]["stage2_rescue_mode"] = rescue_mode
            branches["ris_only_stage2_then_vp"] = rescue_result

    if ngc_active:
        ngc_diagnostics.update(
            _ngc_certificate(
                "rescue", rescue_result, stage1_estimate, data["scene"], config
            )
        )
        selector_rescue_result = rescue_result
        if force_stage2_for_diagnostics and str(
            ngc_diagnostics.get("ngc_direct_cert_status", "")
        ) == "green":
            selector_rescue_result = None
        selected, no_gain = select_ngc_branch(
            direct_result,
            selector_rescue_result,
            reliability,
            ngc_diagnostics,
            config,
        )
    else:
        selected, no_gain = select_proposed_branch(
            direct_result, rescue_result, reliability, config
        )
    selected_branch = str(
        selected.get(
            "selected_branch",
            selected.get("final", {}).get("selected_branch", ""),
        )
    )
    candidate_diagnostics = {}
    candidate_diagnostics.update(
        _extract_branch_candidate_diagnostics("direct", direct_result, data)
    )
    candidate_diagnostics.update(
        _extract_branch_candidate_diagnostics("rescue", rescue_result, data)
    )
    candidate_diagnostics.update(
        _rescue_candidate_selection_diagnostics(
            rescue_result,
            selected_branch,
            reliability,
            rescue_requested=bool(rescue_requested),
        )
    )
    candidate_diagnostics.update(ngc_diagnostics)
    selected.update(candidate_diagnostics)
    if bool(config.get("run_full_legacy_comparison", False)):
        if progress_printed:
            print("running_full_legacy_comparison = True", flush=True)
        branches["full_legacy_comparison"] = run_full_legacy_comparison_branch(
            data, stage1_estimate, config, base_timing, reliability
        )

    result = dict(selected)
    result["branches"] = branches
    result["stage1_config"] = config
    result["direct_vp_quality"] = direct_vp_quality
    if stage1_profile_reference is not None:
        result["stage1_profile_reference"] = stage1_profile_reference
        result["timing"] = dict(result["timing"])
        result["timing"]["stage1_normal_heavy_reference"] = stage1_profile_reference[
            "stage1_s"
        ]
    result["requested_reliability_decision"] = reliability["decision"]
    result["stage2_severe_unreliable"] = severe_diag
    result["gate_override_direct_vp_good"] = bool(direct_vp_override)
    result["ris_stage2_no_gain"] = bool(no_gain)
    stage2_diagnostics = dict(
        rescue_result.get("structured_diag", {}) if rescue_result is not None else {}
    )
    result["stage2_diagnostics"] = stage2_diagnostics
    result["stage2_rescue_triggered"] = bool(rescue_requested)
    result["stage2_force_run_for_diagnostics"] = bool(
        force_stage2_for_diagnostics
    )
    result["stage2_rescue_impl"] = str(
        config.get("stage2_rescue_impl", "legacy_multistart")
    )
    result["stage2_rescue_available"] = bool(
        rescue_result is not None
        and rescue_result.get("structured_diag", {}).get(
            "stage2_rescue_available", True
        )
    )
    result["progress_printed"] = progress_printed
    result["direct_probe_z_rescue_disabled"] = bool(direct_probe_z_rescue_disabled)
    result["direct_z_rescue_rerun_executed"] = bool(direct_z_rescue_rerun_executed)
    result["direct_z_rescue_rerun_skipped"] = bool(direct_z_rescue_rerun_skipped)
    result["direct_z_rescue_skip_reason"] = direct_z_rescue_skip_reason
    result["timing"] = dict(result["timing"])
    result["timing"].update(branch_timing)
    result["timing"]["diagnostic_total"] = time.perf_counter() - total_start
    return result


def run_single_proposed_diagnostic(
    config: dict,
    allow_stage2: bool = True,
    data_override: dict | None = None,
) -> dict:
    """Run one reliability-gated proposed diagnostic realization."""
    total_start = time.perf_counter()
    config = _apply_main_single_defaults(copy.deepcopy(config))
    data = _make_data(config) if data_override is None else data_override
    stage1 = run_stage1_only(data, config)
    return run_from_existing_stage1(
        data,
        stage1,
        config,
        allow_stage2=allow_stage2,
        total_start=total_start,
    )


def _run_single_pipeline(config: dict, use_structured: bool) -> dict:
    """Run the pre-gated single pipeline for tests and older diagnostics."""
    total_start = time.perf_counter()
    data = _make_data(config)
    timing = dict(data.get("timing", {}))
    scene = data["scene"]

    stage1_start = time.perf_counter()
    estimate_initial = initialize_from_hankel(data["Z_noisy"], scene, config)
    timing["stage1"] = time.perf_counter() - stage1_start

    requested_stage2_mode = str(config.get("stage2_mode", "none")).lower()
    stage2_mode = requested_stage2_mode if use_structured else "none"
    if stage2_mode == "full_legacy":
        stage2_start = time.perf_counter()
        estimate_used, structured_diag = structured_refinement(
            data["Z_noisy"], scene, config, copy.deepcopy(estimate_initial)
        )
        timing["stage2"] = time.perf_counter() - stage2_start
    elif stage2_mode == "ris_only":
        raise NotImplementedError("stage2_mode='ris_only' is not a standalone pipeline")
    elif stage2_mode == "none":
        estimate_used = copy.deepcopy(estimate_initial)
        structured_diag = _empty_structured_diag()
        timing["stage2"] = 0.0
    else:
        raise ValueError(f"unknown stage2_mode {stage2_mode!r}")
    timing["ris_projection_total"] = float(
        structured_diag.get("ris_projection_total_s", 0.0)
    )

    final_method = str(
        config.get("final_refinement_method", "global_exact_spherical_vp")
    ).lower()
    if not config.get("enable_global_vp", True):
        final_method = "none"

    vp_start = time.perf_counter()
    if final_method == "global_exact_spherical_vp":
        final = global_exact_spherical_vp_refinement(
            data["Y_noisy"], estimate_used, scene, config
        )
        final["vp_enabled"] = True
        final["stage2_mode"] = stage2_mode
        final["final_refinement_method"] = "global_exact_spherical_vp"
    elif final_method == "legacy_raw_vp":
        final = refine_global_raw(data["Y_noisy"], scene, config, estimate_used)
        final["vp_enabled"] = True
        final["stage2_mode"] = stage2_mode
        final["final_refinement_method"] = "legacy_raw_vp"
    elif final_method == "none":
        y_hat = reconstruct_raw_tensor_from_structured_estimate(estimate_used, scene)
        raw_residual = y_hat - data["Y_noisy"]
        raw_objective = float(
            np.vdot(raw_residual.reshape(-1), raw_residual.reshape(-1)).real
            / data["Y_noisy"].size
        )
        tau_hat = np.array(
            [tau_from_pole(pole, scene["delta_f"]) for pole in estimate_used["poles"]]
        )
        final = {
            "Y_hat": y_hat,
            "p_u": estimate_position_from_ris_eta(scene, estimate_used),
            "components": {
                "taus": tau_hat,
                "ranges": estimate_used["ris_eta"][:, 0],
            },
            "raw_residual_rmse_noisy": float(
                np.linalg.norm(raw_residual) / np.sqrt(data["Y_noisy"].size)
            ),
            "raw_objective_initial": raw_objective,
            "raw_objective_final": raw_objective,
            "optimizer": {
                "success": True,
                "message": "global VP disabled by config",
                "n_eval": 0,
                "method": "skipped_global_vp",
                "solver_backend": "skipped",
            },
            "vp_enabled": False,
            "stage2_mode": stage2_mode,
            "final_refinement_method": "none",
        }
    else:
        raise ValueError(f"unknown final_refinement_method {final_method!r}")

    timing["vp"] = time.perf_counter() - vp_start
    timing["total"] = time.perf_counter() - total_start
    return {
        **data,
        "estimate_initial": estimate_initial,
        "estimate_used": estimate_used,
        "structured_diag": structured_diag,
        "stage1_initialization": {
            "mode": config.get("stage1_init_mode", "normal"),
            "ris_search": dict(config["ris_search"]),
        },
        "stage1_config": config,
        "final": final,
        "timing": timing,
    }


def _print_global_vp_diagnostics(results: dict) -> None:
    """Print the compact final-refinement diagnostics requested for the demo."""
    scene = results["scene"]
    final = results["final"]
    y_noisy = results["Y_noisy"]
    stage1_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_initial"], scene
    )
    stage1_residual = float(
        np.linalg.norm(stage1_y_hat - y_noisy) / np.sqrt(y_noisy.size)
    )
    timing = results.get("timing", {})

    print("\n=== Global VP diagnostics ===")
    print(f"stage1_raw_residual_rmse_noisy = {stage1_residual:.6e}")
    print(f"global_vp_initial_residual = {_fmt(final.get('raw_residual_initial'))}")
    print(f"global_vp_final_residual = {_fmt(final.get('raw_residual_final'))}")
    print(f"global_vp_raw_objective = {_fmt(final.get('raw_objective'))}")
    print(f"global_vp_delay_prior_objective = {_fmt(final.get('delay_prior_objective'))}")
    print(f"global_vp_total_objective = {_fmt(final.get('total_objective'))}")
    print(f"global_vp_solver = {final.get('global_vp_solver', 'unknown')}")
    print(f"global_vp_mode = {final.get('global_vp_mode', final.get('vp_mode', 'unknown'))}")
    print(f"selected_vp_family_branch = {final.get('selected_vp_family_branch', 'NA')}")
    print(f"fixed_pol_score = {_fmt(final.get('fixed_pol_score'))}")
    print(f"jones_score = {_fmt(final.get('jones_score'))}")
    print(f"snr_eff_per_path = {_fmt_vector(final.get('snr_eff_per_path', []))}")
    print(f"lambda_jones_per_path = {_fmt_vector(final.get('lambda_jones_per_path', []))}")
    print(f"jones_leakage_per_path = {_fmt_vector(final.get('jones_leakage_per_path', []))}")
    print(f"linear_nuisance_dim = {final.get('linear_nuisance_dim', 'NA')}")
    print(f"nonlinear_dim = {final.get('nonlinear_dim', 'NA')}")
    rho = np.asarray(final.get("jones_rho", []), dtype=float)
    if rho.size:
        print(
            "jones_rho_summary = "
            f"min={_fmt(np.min(rho))}, median={_fmt(np.median(rho))}, max={_fmt(np.max(rho))}"
        )
    else:
        print("jones_rho_summary = NA")
    print(f"jones_prior_status = {final.get('jones_prior_status', [])}")
    print(f"condition_number_gram = {_fmt(final.get('condition_number_gram'))}")
    print(f"rank_gram = {final.get('rank_gram', 'NA')}")
    print(f"global_vp_evs_mode = {final.get('global_vp_evs_mode', 'unknown')}")
    print(f"global_vp_use_delay_prior = {final.get('global_vp_use_delay_prior', 'NA')}")
    print(f"global_vp_trust_region_used = {final.get('global_vp_trust_region_used', 'NA')}")
    print(
        "global_vp_columns_are_panel_ordered = "
        f"{final.get('global_vp_columns_are_panel_ordered', 'NA')}"
    )
    print(
        "global_vp_used_panel_to_column = "
        f"{final.get('global_vp_used_panel_to_column', 'NA')}"
    )
    print(f"global_vp_panel_to_column = {final.get('global_vp_panel_to_column', 'NA')}")
    print(f"tau_stage1_ns = {_fmt_vector(final.get('tau_stage1', []), scale=1e9)}")
    print(
        "tau_after_global_vp_ns = "
        f"{_fmt_vector(final.get('tau_after_global_vp', []), scale=1e9)}"
    )
    print(f"global_vp_init_method = {final.get('global_vp_init_method', 'unknown')}")
    print(
        "global_vp_init_selected_candidate = "
        f"{final.get('global_vp_init_selected_candidate', 'unknown')}"
    )
    print(f"estimated_p_u_m = {_fmt_vector(final.get('p_u', []), precision=5)}")
    delta_t = final.get("delta_t")
    delta_t_ns = None if delta_t is None else float(delta_t) * 1.0e9
    print(f"estimated_Delta_t_ns = {_fmt(delta_t_ns, precision=6)}")
    print(f"stage1_runtime_s = {_fmt(timing.get('stage1'))}")
    print(f"legacy_stage2_runtime_s = {_fmt(timing.get('stage2'))}")
    print(f"global_vp_runtime_s = {_fmt(timing.get('vp'))}")
    print(f"total_runtime_s = {_fmt(timing.get('total'))}")


def _print_reliability_gate_diagnostics(results: dict) -> None:
    reliability = results["reliability"]
    config = results.get("stage1_config", {})
    reasons = reliability.get("trigger_reasons", [])
    print("\n=== Stage-I reliability gate ===")
    print(f"reliability_decision = {reliability['decision']}")
    print(f"legacy_stage1_decision = {reliability.get('legacy_stage1_decision', 'NA')}")
    print(f"selected_proposed_branch = {results.get('selected_branch', 'unknown')}")
    print(f"bad_score = {reliability['bad_score']}")
    print(f"trigger_reasons = {reasons if reasons else ['none']}")
    print(f"assignment_margin = {_fmt(reliability.get('assignment_margin'))}")
    print(f"sigma_delta_t_ns = {_fmt(reliability.get('sigma_delta_t_ns'))}")
    print(f"delta_t_k_ns = {_fmt_vector(reliability.get('delta_t_k_ns', []))}")
    print(
        "reliability_used_panel_order = "
        f"{reliability.get('reliability_used_panel_order', 'NA')}"
    )
    print(
        "reliability_panel_to_column = "
        f"{reliability.get('reliability_panel_to_column', 'NA')}"
    )
    max_ris_display = _max_ris_residual_display(reliability)
    print(
        f"max_ris_residual = {_fmt(max_ris_display) if not isinstance(max_ris_display, str) else max_ris_display}"
    )
    print(f"severe_unreliable = {reliability.get('severe_unreliable', False)}")
    print(f"gof_stat = {_fmt(reliability.get('gof_stat'))}")
    print(f"gof_dof = {reliability.get('gof_dof', 'NA')}")
    print(f"gof_pass = {reliability.get('gof_pass', 'NA')}")
    print(
        "data_only_efim_lambda_min = "
        f"{_fmt(reliability.get('data_only_efim_lambda_min'))}"
    )
    print(
        "data_only_efim_condition_number = "
        f"{_fmt(reliability.get('data_only_efim_condition_number'))}"
    )
    print(
        "data_only_scaled_efim_lambda_min = "
        f"{_fmt(reliability.get('data_only_scaled_efim_lambda_min'))}"
    )
    print(
        "data_only_scaled_efim_condition_number = "
        f"{_fmt(reliability.get('data_only_scaled_efim_condition_number'))}"
    )
    if reliability.get("severe_unreliable", False):
        print("WARNING_STAGE1_SEVERE_UNRELIABLE: Stage-I reliability gate crossed a severe threshold.")
    if _paper_balanced_good_snr_triggered_stage2(config, reliability):
        print("WARNING_PAPER_BALANCED_TRIGGERED_STAGE2_AT_GOOD_SNR")
    if results.get("gate_override_direct_vp_good", False):
        print("GATE_OVERRIDE_DIRECT_VP_GOOD")
    if results.get("ris_stage2_no_gain", False):
        rescue_mode = results.get("structured_diag", {}).get("stage2_rescue_mode")
        if rescue_mode == "multi_hypothesis_ris_reacquisition":
            print(
                "MHRR_NO_SIGNIFICANT_RAW_GAIN: MHRR+VP did not significantly "
                "improve the final raw-domain objective; selected direct VP."
            )
        elif rescue_mode == "robust_jnpp":
            print(
                "WARNING_JNPP_NO_SIGNIFICANT_RAW_GAIN: JNPP+VP did not significantly "
                "improve the final raw-domain objective; selected direct VP."
            )
        else:
            print(
                "WARNING_RIS_STAGE2_NO_SIGNIFICANT_RAW_GAIN: RIS-only Stage-II+VP did "
                "not significantly improve the final raw-domain objective; selected direct VP."
            )


def _print_reliability_warnings(results: dict) -> None:
    reliability = results["reliability"]
    config = results.get("stage1_config", {})
    if reliability.get("severe_unreliable", False):
        print(
            "WARNING_STAGE1_SEVERE_UNRELIABLE: Stage-I reliability gate crossed a severe threshold."
        )
    if results.get("ris_stage2_no_gain", False):
        rescue_mode = results.get("structured_diag", {}).get("stage2_rescue_mode")
        if rescue_mode != "multi_hypothesis_ris_reacquisition":
            for branch in results.get("branches", {}).values():
                mode = branch.get("structured_diag", {}).get("stage2_rescue_mode")
                if mode == "multi_hypothesis_ris_reacquisition":
                    rescue_mode = mode
                    break
        if rescue_mode == "multi_hypothesis_ris_reacquisition":
            print(
                "MHRR_NO_SIGNIFICANT_RAW_GAIN: MHRR+VP did not significantly "
                "improve the final raw-domain objective; selected direct VP."
            )
        elif rescue_mode == "robust_jnpp":
            print(
                "WARNING_JNPP_NO_SIGNIFICANT_RAW_GAIN: JNPP+VP did not significantly "
                "improve the final raw-domain objective; selected direct VP."
            )
        else:
            print(
                "WARNING_RIS_STAGE2_NO_SIGNIFICANT_RAW_GAIN: RIS-only Stage-II+VP did "
                "not significantly improve the final raw-domain objective; selected direct VP."
            )
    if _paper_balanced_good_snr_triggered_stage2(config, reliability):
        print("WARNING_PAPER_BALANCED_TRIGGERED_STAGE2_AT_GOOD_SNR")
    if results.get("gate_override_direct_vp_good", False):
        print("GATE_OVERRIDE_DIRECT_VP_GOOD")


def _print_reliability_progress(reliability: dict) -> None:
    print("\n=== Stage-I reliability gate ===", flush=True)
    print(f"reliability_decision = {reliability['decision']}", flush=True)
    print(f"legacy_stage1_decision = {reliability.get('legacy_stage1_decision', 'NA')}", flush=True)
    print(f"bad_score = {reliability['bad_score']}", flush=True)
    print(
        f"trigger_reasons = {reliability.get('trigger_reasons', []) or ['none']}",
        flush=True,
    )
    print(f"assignment_margin = {_fmt(reliability.get('assignment_margin'))}", flush=True)
    print(f"sigma_delta_t_ns = {_fmt(reliability.get('sigma_delta_t_ns'))}", flush=True)
    print(f"delta_t_k_ns = {_fmt_vector(reliability.get('delta_t_k_ns', []))}", flush=True)
    print(
        "reliability_used_panel_order = "
        f"{reliability.get('reliability_used_panel_order', 'NA')}",
        flush=True,
    )
    print(
        "reliability_panel_to_column = "
        f"{reliability.get('reliability_panel_to_column', 'NA')}",
        flush=True,
    )
    max_ris_display = _max_ris_residual_display(reliability)
    print(
        "max_ris_residual = "
        f"{_fmt(max_ris_display) if not isinstance(max_ris_display, str) else max_ris_display}",
        flush=True,
    )
    print(f"severe_unreliable = {reliability.get('severe_unreliable', False)}", flush=True)
    print(f"gof_stat = {_fmt(reliability.get('gof_stat'))}", flush=True)
    print(f"gof_dof = {reliability.get('gof_dof', 'NA')}", flush=True)
    print(f"gof_pass = {reliability.get('gof_pass', 'NA')}", flush=True)
    print(
        "data_only_efim_lambda_min = "
        f"{_fmt(reliability.get('data_only_efim_lambda_min'))}",
        flush=True,
    )
    print(
        "data_only_efim_condition_number = "
        f"{_fmt(reliability.get('data_only_efim_condition_number'))}",
        flush=True,
    )
    print(
        "data_only_scaled_efim_lambda_min = "
        f"{_fmt(reliability.get('data_only_scaled_efim_lambda_min'))}",
        flush=True,
    )
    print(
        "data_only_scaled_efim_condition_number = "
        f"{_fmt(reliability.get('data_only_scaled_efim_condition_number'))}",
        flush=True,
    )


def _print_vp_branch_metrics(results: dict) -> None:
    """Print direct, RIS-only if run, optional full legacy, and selected metrics."""
    branches = results.get("branches", {})
    ordered_names = ["direct_vp", "ris_only_stage2_then_vp", "full_legacy_comparison"]
    print("\n=== Reliability branch metrics ===")
    print(
        "branch | raw_objective_final | Y_NMSE_true | position_RMSE_m | "
        "range_RMSE_m | tau_RMSE_s | nfev | success"
    )
    printed_selected = False
    for name in ordered_names:
        branch = branches.get(name)
        if branch is None:
            continue
        final = branch["final"]
        y_metrics = y_metric_summary(final["Y_hat"], branch["Y_true"])
        geom = parameter_errors_for_vp(branch["scene"], final, branch["true_components"])
        optimizer = final.get("optimizer", {})
        print(
            f"{name} | {_fmt(_final_raw_objective(final))} | "
            f"{_fmt(y_metrics['nmse'])} | {_fmt(geom['position_rmse'])} | "
            f"{_fmt(geom['range_rmse'])} | {_fmt(geom['tau_rmse'])} | "
            f"{optimizer.get('n_eval', 'NA')} | {optimizer.get('success', 'NA')}"
        )
        if name == results.get("selected_branch"):
            printed_selected = True
    if not printed_selected:
        final = results["final"]
        y_metrics = y_metric_summary(final["Y_hat"], results["Y_true"])
        geom = parameter_errors_for_vp(results["scene"], final, results["true_components"])
        optimizer = final.get("optimizer", {})
        print(
            f"{results.get('selected_branch', 'selected_proposed')} | "
            f"{_fmt(_final_raw_objective(final))} | {_fmt(y_metrics['nmse'])} | "
            f"{_fmt(geom['position_rmse'])} | {_fmt(geom['range_rmse'])} | "
            f"{_fmt(geom['tau_rmse'])} | {optimizer.get('n_eval', 'NA')} | "
            f"{optimizer.get('success', 'NA')}"
        )
    selected_final = results["final"]
    selected_y = y_metric_summary(selected_final["Y_hat"], results["Y_true"])
    selected_geom = parameter_errors_for_vp(
        results["scene"], selected_final, results["true_components"]
    )
    print(
        "selected_proposed_metrics = "
        f"branch={results.get('selected_branch', 'unknown')}, "
        f"raw_objective_final={_fmt(_final_raw_objective(selected_final))}, "
        f"Y_NMSE_true={_fmt(selected_y['nmse'])}, "
        f"position_RMSE_m={_fmt(selected_geom['position_rmse'])}"
    )
    quality = results.get("direct_vp_quality")
    if quality is not None:
        print(
            "direct_vp_quality = "
            f"good={quality.get('good')}, "
            f"success={quality.get('success')}, "
            f"nfev={quality.get('nfev')}, "
            f"noise_floor_good={quality.get('noise_floor_good')}, "
            f"rel_decrease={_fmt(quality.get('relative_objective_decrease'))}, "
            f"gof_stat={_fmt(quality.get('gof_stat'))}, "
            f"gof_dof={quality.get('gof_dof')}, "
            f"gof_pass={quality.get('gof_pass')}, "
            f"data_only_efim_lambda_min={_fmt(quality.get('data_only_efim_lambda_min'))}, "
            f"data_only_efim_condition_number={_fmt(quality.get('data_only_efim_condition_number'))}, "
            f"data_only_scaled_efim_condition_number={_fmt(quality.get('data_only_scaled_efim_condition_number'))}, "
            f"reliability_decision={quality.get('reliability_decision')}"
        )


def _print_noise_and_y_metrics(results: dict, direct_results: dict, snr_db: float) -> dict:
    """Print noise and raw-domain metrics for default diagnostics."""
    scene = results["scene"]
    y_true = results["Y_true"]
    y_noisy = results["Y_noisy"]
    vp_enabled = bool(results["final"].get("vp_enabled", True))
    initial_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_initial"], scene
    )
    structured_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_used"], scene
    )
    final_y_hat = results["final"]["Y_hat"]

    noise_metrics = noise_metric_summary(y_true, y_noisy, snr_db)
    print("\n=== Noise and Y-domain metrics ===")
    for key in (
        "norm_Y_true",
        "norm_noise",
        "signal_power_Y",
        "noise_power_Y",
        "target_SNR_dB",
        "empirical_SNR_dB",
        "RMSE_Y_noisy_abs",
        "NMSE_Y_noisy",
    ):
        print(f"{key} = {noise_metrics[key]:.6e}")

    initial_metrics = y_metric_summary(initial_y_hat, y_true)
    structured_metrics = y_metric_summary(structured_y_hat, y_true)
    final_metrics = y_metric_summary(final_y_hat, y_true)
    direct_final_metrics = y_metric_summary(direct_results["final"]["Y_hat"], y_true)

    print(f"RMSE_Y_hat_initial_abs = {initial_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat_initial = {initial_metrics['nmse']:.6e}")
    print(f"RMSE_Y_hat_after_structured_abs = {structured_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat_after_structured = {structured_metrics['nmse']:.6e}")
    if vp_enabled:
        print(f"RMSE_Y_hat_after_VP_abs = {final_metrics['rmse_abs']:.6e}")
        print(f"NMSE_Y_hat_after_VP = {final_metrics['nmse']:.6e}")
    else:
        print(f"RMSE_Y_hat_final_stage2_only_abs = {final_metrics['rmse_abs']:.6e}")
        print(f"NMSE_Y_hat_final_stage2_only = {final_metrics['nmse']:.6e}")
    print(f"RMSE_Y_hat_abs = {final_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat = {final_metrics['nmse']:.6e}")
    print(f"after_structured_Y_RMSE_abs = {structured_metrics['rmse_abs']:.6e}")
    print(f"after_structured_Y_NMSE = {structured_metrics['nmse']:.6e}")
    if vp_enabled:
        print(f"after_VP_Y_RMSE_abs = {final_metrics['rmse_abs']:.6e}")
        print(f"after_VP_Y_NMSE = {final_metrics['nmse']:.6e}")

    if not 0.70 <= noise_metrics["NMSE_Y_noisy"] <= 1.30:
        print("WARNING OBJECTIVE_MISMATCH: NMSE_Y_noisy is not close to 1 at 0 dB; check AWGN scaling.")
    if vp_enabled and final_metrics["nmse"] > structured_metrics["nmse"]:
        print(
            "WARNING VP_NO_GAIN: Raw VP-WNLS worsened true-domain Y NMSE after Stage-II in this run; "
            "likely cause is noisy-domain fitting or weak nonlinear initialization."
        )

    return {
        "initial": initial_metrics,
        "structured": structured_metrics,
        "final": final_metrics,
        "direct_final": direct_final_metrics,
        "vp": final_metrics,
        "direct_vp": direct_final_metrics,
        "noise": noise_metrics,
    }


def _print_z_stage_metrics(results: dict) -> list[dict]:
    """Print true-domain and noisy-domain Z residuals for Stage II."""
    print("\n=== Z-domain structured-stage diagnostics ===")
    z_true = results["Z_true"]
    z_noisy = results["Z_noisy"]
    initial_metrics = z_metric_summary(results["estimate_initial"]["Z_hat"], z_true, z_noisy)
    print(f"initial_Z_RMSE_noisy = {initial_metrics['rmse_noisy']:.6e}")
    print(f"initial_Z_RMSE_true = {initial_metrics['rmse_true']:.6e}")
    print(f"initial_Z_NMSE_noisy = {initial_metrics['nmse_noisy']:.6e}")
    print(f"initial_Z_NMSE_true = {initial_metrics['nmse_true']:.6e}")

    history_metrics = []
    for idx, z_hat in enumerate(results["structured_diag"]["z_hat_history"], start=1):
        metrics = z_metric_summary(z_hat, z_true, z_noisy)
        history_metrics.append(metrics)
        print(f"structured_iter_{idx}_Z_RMSE_noisy = {metrics['rmse_noisy']:.6e}")
        print(f"structured_iter_{idx}_Z_RMSE_true = {metrics['rmse_true']:.6e}")
        print(f"structured_iter_{idx}_Z_NMSE_noisy = {metrics['nmse_noisy']:.6e}")
        print(f"structured_iter_{idx}_Z_NMSE_true = {metrics['nmse_true']:.6e}")

    if history_metrics and history_metrics[-1]["nmse_true"] >= initial_metrics["nmse_true"]:
        print("WARNING OBJECTIVE_MISMATCH: Stage-II did not reduce true-domain Z NMSE in this run.")
    return [initial_metrics] + history_metrics


def _structured_parameter_arrays(scene: dict, estimate: dict) -> dict:
    tau_hat = np.array([tau_from_pole(pole, scene["delta_f"]) for pole in estimate["poles"]])
    return {
        "tau": tau_hat,
        "range": np.asarray(estimate["ris_eta"][:, 0], dtype=float),
        "elev": np.asarray(estimate["ris_eta"][:, 1], dtype=float),
        "az": np.asarray(estimate["ris_eta"][:, 2], dtype=float),
        "gamma": np.asarray(estimate.get("gamma", []), dtype=float),
        "eta_pol": np.asarray(estimate.get("eta_pol", []), dtype=float),
    }


def _vp_parameter_arrays(final: dict) -> dict:
    components = final["components"]
    return {
        "tau": np.asarray(components["taus"], dtype=float),
        "range": np.asarray(components["ranges"], dtype=float),
        "elev": np.asarray(components.get("elevations", []), dtype=float),
        "az": np.asarray(components.get("azimuths", []), dtype=float),
        "gamma": np.asarray(final.get("gamma", []), dtype=float),
        "eta_pol": np.asarray(final.get("eta_pol", []), dtype=float),
    }


def _geometry_error_metrics(scene: dict, arrays: dict, p_hat: np.ndarray, true_components: dict) -> dict:
    az_err = np.array(
        [
            _wrap_angle_rad(arrays["az"][k] - true_components["azimuths"][k])
            for k in range(scene["K"])
        ]
    )
    return {
        "tau_RMSE_s": float(
            np.linalg.norm(arrays["tau"] - true_components["taus"]) / np.sqrt(scene["K"])
        ),
        "range_RMSE_m": float(
            np.linalg.norm(arrays["range"] - true_components["ranges"]) / np.sqrt(scene["K"])
        ),
        "elev_RMSE_deg": float(
            np.rad2deg(
                np.linalg.norm(arrays["elev"] - true_components["elevations"])
                / np.sqrt(scene["K"])
            )
        ),
        "az_RMSE_deg": float(np.rad2deg(np.linalg.norm(az_err) / np.sqrt(scene["K"]))),
        "position_RMSE_m": position_rmse(p_hat, scene["p_u_true"]),
    }


def _structured_geometry_metrics(scene: dict, estimate: dict, true_components: dict) -> dict:
    arrays = _structured_parameter_arrays(scene, estimate)
    p_hat = estimate_position_from_ris_eta(scene, estimate)
    return _geometry_error_metrics(scene, arrays, p_hat, true_components)


def _vp_geometry_metrics(scene: dict, final: dict, true_components: dict) -> dict:
    arrays = _vp_parameter_arrays(final)
    return _geometry_error_metrics(scene, arrays, final["p_u"], true_components)


def _parameter_table_rows(
    scene: dict,
    arrays: dict,
    c_rows: np.ndarray | None,
    a_rows: np.ndarray | None,
    true_components: dict | None,
) -> list[dict]:
    rows = []
    for k in range(scene["K"]):
        tau = arrays["tau"][k]
        range_m = arrays["range"][k]
        elev = arrays["elev"][k] if arrays["elev"].size else float("nan")
        az = arrays["az"][k] if arrays["az"].size else float("nan")
        gamma = arrays["gamma"][k] if arrays["gamma"].size else float("nan")
        eta_pol = arrays["eta_pol"][k] if arrays["eta_pol"].size else float("nan")
        if true_components is None:
            tau_err_ps = range_err_m = elev_err_deg = az_err_deg = float("nan")
        else:
            tau_err_ps = (tau - true_components["taus"][k]) * 1.0e12
            range_err_m = range_m - true_components["ranges"][k]
            elev_err_deg = np.rad2deg(elev - true_components["elevations"][k])
            az_err_deg = np.rad2deg(_wrap_angle_rad(az - true_components["azimuths"][k]))

        ris_residual = float("nan")
        if c_rows is not None and np.isfinite([range_m, elev, az]).all():
            c_vec = c_rows[k] if c_rows.shape[0] == scene["K"] else c_rows[:, k]
            ris_residual = _ris_local_residual(
                scene, k, np.asarray(c_vec), np.array([range_m, elev, az])
            )

        evs_residual = float("nan")
        if (
            a_rows is not None
            and np.isfinite(gamma)
            and np.isfinite(eta_pol)
        ):
            a_vec = a_rows[k] if a_rows.shape[0] == scene["K"] else a_rows[:, k]
            evs_residual = _evs_local_residual(
                scene, k, np.asarray(a_vec), gamma, eta_pol
            )

        rows.append(
            {
                "path": k,
                "panel": k,
                "tau_ns": tau * 1.0e9,
                "tau_err_ps": tau_err_ps,
                "range_m": range_m,
                "range_err_m": range_err_m,
                "elev_deg": np.rad2deg(elev),
                "elev_err_deg": elev_err_deg,
                "az_deg": np.rad2deg(az),
                "az_err_deg": az_err_deg,
                "gamma_deg": np.rad2deg(gamma),
                "eta_pol_deg": np.rad2deg(eta_pol),
                "RIS_local_residual": ris_residual,
                "EVS_local_residual": evs_residual,
            }
        )
    return rows


def _print_parameter_table(title: str, rows: list[dict]) -> None:
    columns = [
        "path",
        "panel",
        "tau_ns",
        "tau_err_ps",
        "range_m",
        "range_err_m",
        "elev_deg",
        "elev_err_deg",
        "az_deg",
        "az_err_deg",
        "gamma_deg",
        "eta_pol_deg",
        "RIS_local_residual",
        "EVS_local_residual",
    ]
    print(f"\n=== {title} ===")
    print(" | ".join(columns))
    for row in rows:
        print(
            " | ".join(
                str(row[col]) if col in ("path", "panel") else _fmt(row[col], 5)
                for col in columns
            )
        )


def _print_per_path_parameter_tables(results: dict) -> None:
    scene = results["scene"]
    true_components = results.get("true_components")
    initial_arrays = _structured_parameter_arrays(scene, results["estimate_initial"])
    structured_arrays = _structured_parameter_arrays(scene, results["estimate_used"])
    _print_parameter_table(
        "Per-path parameters after Stage-I",
        _parameter_table_rows(
            scene,
            initial_arrays,
            results["estimate_initial"]["C"],
            results["estimate_initial"]["A"],
            true_components,
        ),
    )
    _print_parameter_table(
        "Per-path parameters after Stage-II",
        _parameter_table_rows(
            scene,
            structured_arrays,
            results["estimate_used"]["C"],
            results["estimate_used"]["A"],
            true_components,
        ),
    )
    if bool(results["final"].get("vp_enabled", True)):
        final_arrays = _vp_parameter_arrays(results["final"])
        _print_parameter_table(
            "Per-path parameters after VP",
            _parameter_table_rows(
                scene,
                final_arrays,
                results["final"]["components"].get("c"),
                results["final"]["components"].get("a_EVS"),
                true_components,
            ),
        )


def _relative_change_scalar(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before) or abs(before) <= 1.0e-300:
        return float("nan")
    return float((after - before) / abs(before))


def _print_stage2_summary_table(results: dict) -> dict:
    """Print Stage-I versus after Stage-II summary metrics."""
    scene = results["scene"]
    true_components = results["true_components"]
    initial_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_initial"], scene
    )
    structured_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_used"], scene
    )
    initial_geom = _structured_geometry_metrics(
        scene, results["estimate_initial"], true_components
    )
    structured_geom = _structured_geometry_metrics(
        scene, results["estimate_used"], true_components
    )
    initial_values = {
        "Y_NMSE_true": relative_nmse(initial_y_hat, results["Y_true"]),
        "Z_NMSE_true": relative_nmse(results["estimate_initial"]["Z_hat"], results["Z_true"]),
        "global_Z_SSE": float(np.linalg.norm(results["estimate_initial"]["Z_hat"] - results["Z_noisy"]) ** 2),
        **initial_geom,
        "num_EVS_accepted": 0.0,
        "num_delay_accepted": 0.0,
        "num_RIS_accepted": 0.0,
        "num_iteration_rollbacks": 0.0,
    }
    updates = results["structured_diag"]["updates"]
    structured_values = {
        "Y_NMSE_true": relative_nmse(structured_y_hat, results["Y_true"]),
        "Z_NMSE_true": relative_nmse(results["estimate_used"]["Z_hat"], results["Z_true"]),
        "global_Z_SSE": float(np.linalg.norm(results["estimate_used"]["Z_hat"] - results["Z_noisy"]) ** 2),
        **structured_geom,
        "num_EVS_accepted": float(
            sum(
                bool(detail.get("accepted", False))
                for update in updates
                for detail in update.get("evs_projection_details", [])
            )
        ),
        "num_delay_accepted": float(
            sum(bool(update.get("delay_projection_details", {}).get("accepted", False)) for update in updates)
        ),
        "num_RIS_accepted": float(
            sum(
                bool(detail.get("accepted", False))
                for update in updates
                for detail in update.get("ris_projection_details", [])
            )
        ),
        "num_iteration_rollbacks": float(
            sum(not bool(update.get("iteration_accepted", True)) for update in updates)
        ),
    }
    rows = [
        "Y_NMSE_true",
        "Z_NMSE_true",
        "tau_RMSE_s",
        "range_RMSE_m",
        "elev_RMSE_deg",
        "az_RMSE_deg",
        "position_RMSE_m",
        "global_Z_SSE",
        "num_EVS_accepted",
        "num_delay_accepted",
        "num_RIS_accepted",
        "num_iteration_rollbacks",
    ]
    print("\n=== Stage-II summary ===")
    print("metric | Stage-I | After Stage-II | Relative change")
    for key in rows:
        before = initial_values[key]
        after = structured_values[key]
        print(
            f"{key} | {_fmt(before)} | {_fmt(after)} | "
            f"{_fmt(_relative_change_scalar(after, before))}"
        )

    if (
        structured_values["global_Z_SSE"] < initial_values["global_Z_SSE"]
        and structured_values["Z_NMSE_true"] > initial_values["Z_NMSE_true"]
    ):
        print(
            "WARNING OBJECTIVE_MISMATCH: Stage-II reduced noisy-domain global_Z_SSE "
            "but increased true-domain Z_NMSE_true."
        )
    for key in ("range_RMSE_m", "elev_RMSE_deg", "az_RMSE_deg", "position_RMSE_m"):
        if structured_values[key] > initial_values[key] + 1.0e-12:
            print(
                f"WARNING GEOM_DEGRADE: {key} increased from "
                f"{initial_values[key]:.6e} to {structured_values[key]:.6e}."
            )

    return {"initial": initial_values, "structured": structured_values}


def _print_parameter_diagnostics(results: dict) -> dict:
    """Print tau, range, position, and compact per-path diagnostics."""
    print("\n=== Parameter diagnostics ===")
    scene = results["scene"]
    true_components = results["true_components"]
    vp_enabled = bool(results["final"].get("vp_enabled", True))
    initial = parameter_errors_for_structured(scene, results["estimate_initial"], true_components)
    structured = parameter_errors_for_structured(scene, results["estimate_used"], true_components)
    final = parameter_errors_for_vp(scene, results["final"], true_components)

    print(f"tau_RMSE_initial = {initial['tau_rmse']:.6e}")
    print(f"tau_RMSE_after_structured = {structured['tau_rmse']:.6e}")
    if vp_enabled:
        print(f"tau_RMSE_after_VP = {final['tau_rmse']:.6e}")
    else:
        print(f"tau_RMSE_final_stage2_only = {final['tau_rmse']:.6e}")
    print(f"range_RMSE_initial = {initial['range_rmse']:.6e}")
    print(f"range_RMSE_after_structured = {structured['range_rmse']:.6e}")
    if vp_enabled:
        print(f"range_RMSE_after_VP = {final['range_rmse']:.6e}")
    else:
        print(f"range_RMSE_final_stage2_only = {final['range_rmse']:.6e}")
    print(f"position_RMSE_initial = {initial['position_rmse']:.6e}")
    print(f"position_RMSE_after_structured = {structured['position_rmse']:.6e}")
    if vp_enabled:
        print(f"position_RMSE_after_VP = {final['position_rmse']:.6e}")
    else:
        print(f"position_RMSE_final_stage2_only = {final['position_rmse']:.6e}")

    print(f"true_tau_ns = {format_float_list(true_components['taus'], scale=1e9)}")
    print(f"initial_tau_ns = {format_float_list(initial['tau_hat'], scale=1e9)}")
    print(f"structured_tau_ns = {format_float_list(structured['tau_hat'], scale=1e9)}")
    if vp_enabled:
        print(f"VP_tau_ns = {format_float_list(final['tau_hat'], scale=1e9)}")
    else:
        print(f"final_stage2_tau_ns = {format_float_list(final['tau_hat'], scale=1e9)}")
    print(f"true_range_m = {format_float_list(true_components['ranges'])}")
    print(f"initial_range_m = {format_float_list(initial['range_hat'])}")
    print(f"structured_range_m = {format_float_list(structured['range_hat'])}")
    if vp_enabled:
        print(f"VP_range_m = {format_float_list(final['range_hat'])}")
    else:
        print(f"final_stage2_range_m = {format_float_list(final['range_hat'])}")
    print(f"true_RIS_panel_assignment = {list(range(scene['K']))}")
    print(f"estimated_col_to_panel_assignment = {results['estimate_initial']['assignment']}")
    _print_per_path_parameter_tables(results)
    return {"initial": initial, "structured": structured, "final": final, "vp": final}


def _fmt_eta(eta: np.ndarray | None) -> str:
    if eta is None:
        return "range_m=NA,elev_deg=NA,az_deg=NA"
    arr = np.asarray(eta, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return "range_m=NA,elev_deg=NA,az_deg=NA"
    return (
        f"range_m={arr[0]:.6e},"
        f"elev_deg={np.rad2deg(arr[1]):.6e},"
        f"az_deg={np.rad2deg(arr[2]):.6e}"
    )


def _print_stage_two_update_diagnostics(results: dict) -> None:
    """Print whether Stage-II variables and projections are changing."""
    print("\n=== Stage-II update diagnostics ===")
    unchanged_ris_count = 0
    for idx, update in enumerate(results["structured_diag"]["updates"], start=1):
        print(
            f"iter {idx}: "
            f"delta_A={update['delta_A']:.3e}, "
            f"delta_B={update['delta_B']:.3e}, "
            f"delta_Q={update['delta_Q']:.3e}, "
            f"delta_C={update['delta_C']:.3e}, "
            f"delta_beta={update['delta_beta']:.3e}, "
            f"nonfinite(A,B,Q,C,beta)="
            f"({update['nonfinite_A']},{update['nonfinite_B']},"
            f"{update['nonfinite_Q']},{update['nonfinite_C']},"
            f"{update['nonfinite_beta']})"
        )
        if any(
            update[key] > 0
            for key in (
                "nonfinite_A",
                "nonfinite_B",
                "nonfinite_Q",
                "nonfinite_C",
                "nonfinite_beta",
            )
        ):
            print("  WARNING NONFINITE: Stage-II iterate contains nonfinite entries.")
        print(
            "  iteration_guard: "
            f"accepted={update.get('iteration_accepted', True)}, "
            f"global_SSE_before={_fmt(update.get('iteration_sse_before'))}, "
            f"global_SSE_proposed={_fmt(update.get('iteration_sse_proposed'))}, "
            f"global_SSE_after={_fmt(update.get('iteration_sse_after'))}, "
            f"relative_change={_fmt(update.get('relative_residual_change'))}"
        )

        for detail in update["evs_projection_details"]:
            path = detail.get("path", "?")
            gamma_before = np.rad2deg(detail.get("gamma_before", np.nan))
            gamma_after = np.rad2deg(detail.get("gamma_after", np.nan))
            eta_before = np.rad2deg(detail.get("eta_pol_before", np.nan))
            eta_after = np.rad2deg(detail.get("eta_pol_after", np.nan))
            print(
                f"  EVS path {path}: "
                f"accepted={detail.get('accepted', False)}, "
                f"reason={detail.get('reason', 'unknown')}, "
                f"local_res_before={_fmt(detail.get('local_res_before'))}, "
                f"local_res_after={_fmt(detail.get('local_res_after'))}, "
                f"relative_improvement={_fmt(detail.get('relative_improvement'))}, "
                f"global_SSE_before={_fmt(detail.get('global_sse_before'))}, "
                f"global_SSE_after={_fmt(detail.get('global_sse_after'))}, "
                f"damping_rho={_fmt(detail.get('best_rho'))}, "
                f"gamma_deg_before={_fmt(gamma_before)}, "
                f"gamma_deg_after={_fmt(gamma_after)}, "
                f"eta_pol_deg_before={_fmt(eta_before)}, "
                f"eta_pol_deg_after={_fmt(eta_after)}"
            )

        delay_detail = update["delay_projection_details"]
        print(
            "  delay structured LS: "
            f"skipped={delay_detail.get('skipped', False)}, "
            f"accepted={delay_detail.get('accepted', False)}, "
            f"reason={delay_detail.get('reason', 'unknown')}, "
            f"local_res_before={_fmt(delay_detail.get('local_res_before'))}, "
            f"local_res_after={_fmt(delay_detail.get('local_res_after'))}, "
            f"relative_improvement={_fmt(delay_detail.get('relative_improvement'))}, "
            f"global_SSE_before={_fmt(delay_detail.get('global_sse_before'))}, "
            f"global_SSE_after={_fmt(delay_detail.get('global_sse_after'))}, "
            f"damping_rho={_fmt(delay_detail.get('damping'))}, "
            f"tau_ns_before={_fmt_vector(delay_detail.get('tau_before', []), scale=1e9)}, "
            f"tau_ns_candidate={_fmt_vector(delay_detail.get('tau_candidate', []), scale=1e9)}, "
            f"tau_ns_after={_fmt_vector(delay_detail.get('tau_after', []), scale=1e9)}, "
            f"geom_accepted={delay_detail.get('geometry_correction_accepted', False)}"
        )
        print(f"  mode4_panel_order = {update.get('mode4_assignment_order')}")
        ris_accept = [
            "skipped" if detail.get("skipped", False) else detail.get("accepted", False)
            for detail in update["ris_projection_details"]
        ]
        print(f"  RIS projection accepted = {ris_accept}")
        for detail in update["ris_projection_details"]:
            eta = detail.get("selected_eta")
            if detail.get("skipped", False):
                print(
                    f"  RIS path {detail['path']}: "
                    f"skipped=True, "
                    f"accepted={detail.get('accepted', False)}, "
                    f"reason={detail.get('reason', 'unknown')}, "
                    f"geometry_after=({_fmt_eta(eta)})"
                )
                continue
            print(
                f"  RIS path {detail['path']}: "
                f"accepted={detail.get('accepted', False)}, "
                f"reason={detail.get('reason', 'unknown')}, "
                f"local_res_before={_fmt(detail.get('residual_before'))}, "
                f"local_res_after={_fmt(detail.get('residual_after'))}, "
                f"relative_improvement={_fmt(detail.get('relative_improvement'))}, "
                f"global_SSE_before={_fmt(detail.get('global_sse_before'))}, "
                f"global_SSE_after={_fmt(detail.get('global_sse_after'))}, "
                f"damping_rho={_fmt(detail.get('best_rho'))}, "
                f"projection_time_s={_fmt(detail.get('projection_time_s'))}, "
                f"selected_model={detail.get('selected_model')}, "
                f"lifted_used={detail.get('lifted_used', False)}, "
                f"c_delta={_fmt(detail.get('c_relative_change'))}, "
                f"geometry_before=({_fmt_eta(detail.get('eta_before'))}), "
                f"geometry_candidate=({_fmt_eta(detail.get('candidate_eta'))}), "
                f"geometry_after=({_fmt_eta(eta)})"
            )
            for candidate in detail.get("candidate_ranking", [])[:3]:
                print(
                    f"    RIS candidate rank {candidate.get('rank')}: "
                    f"model={candidate.get('model')}, "
                    f"range_m={_fmt(candidate.get('range_m'))}, "
                    f"elev_deg={_fmt(candidate.get('elev_deg'))}, "
                    f"az_deg={_fmt(candidate.get('az_deg'))}, "
                    f"local_residual={_fmt(candidate.get('local_residual'))}, "
                    f"exact_refined={candidate.get('exact_refined')}, "
                    f"selected={candidate.get('selected')}"
                )
            if detail.get("c_relative_change", 0.0) < 1e-8:
                unchanged_ris_count += 1
                print(
                    "  WARNING RIS_STAGNATION: RIS Mode-4 projection returned "
                    "an almost unchanged c_k."
                )
    if unchanged_ris_count:
        print(
            "WARNING RIS_STAGNATION: RIS Mode-4 projection stagnated in at least one path/iteration; "
            "likely cause is the compressed RIS projection selecting the same local grid optimum."
        )


def _print_structured_comparison(results: dict, direct_results: dict, y_metrics: dict) -> None:
    """Print final estimates with versus without the structured stage."""
    if str(results["final"].get("stage2_mode", "none")).lower() == "none":
        print("\n=== With vs without structured stage ===")
        print("legacy_structured_stage_enabled = False")
        print("comparison_note = default pipeline bypasses legacy factor-domain Stage-II")
        return

    y_true = results["Y_true"]
    _ = y_true
    vp_enabled = bool(results["final"].get("vp_enabled", True))
    direct_nmse = y_metrics["direct_final"]["nmse"]
    with_nmse = y_metrics["final"]["nmse"]
    direct_pos = position_rmse(direct_results["final"]["p_u"], results["scene"]["p_u_true"])
    with_pos = position_rmse(results["final"]["p_u"], results["scene"]["p_u_true"])
    improvement = direct_nmse - with_nmse

    print("\n=== With vs without structured stage ===")
    if vp_enabled:
        print(f"NMSE_Y_after_VP_without_structured = {direct_nmse:.6e}")
    else:
        print(f"NMSE_Y_final_without_structured_no_VP = {direct_nmse:.6e}")
    print(f"position_RMSE_without_structured = {direct_pos:.6e}")
    if vp_enabled:
        print(f"NMSE_Y_after_VP_with_structured = {with_nmse:.6e}")
    else:
        print(f"NMSE_Y_final_with_structured_no_VP = {with_nmse:.6e}")
    print(f"position_RMSE_with_structured = {with_pos:.6e}")
    print(f"improvement_from_structured = {improvement:.6e}")
    if abs(improvement) < 1e-4 and abs(direct_pos - with_pos) < 1e-3:
        if vp_enabled:
            print(
                "WARNING VP_NO_GAIN: Structured HP-R1P-CPD stage currently gives little improvement "
                "over direct VP-WNLS."
            )
        else:
            print(
                "WARNING OBJECTIVE_MISMATCH: Structured HP-R1P-CPD stage currently gives little improvement "
                "over initialization-only output."
            )


def _print_vp_branch_comparison(results: dict, direct_results: dict) -> None:
    """Print direct, gated RIS-only, optional full legacy, and selected VP diagnostics."""
    if not bool(results["final"].get("vp_enabled", True)):
        return
    scene = results["scene"]
    true_components = results["true_components"]
    branches = [("Stage-I+VP", direct_results["final"])]
    branch_results = results.get("branches", {})
    if branch_results.get("ris_only_stage2_then_vp") is not None:
        branches.append(
            ("RIS-only Stage-II+VP", branch_results["ris_only_stage2_then_vp"]["final"])
        )
    if branch_results.get("full_legacy_comparison") is not None:
        branches.append(
            (
                "Full legacy Stage-II+VP comparison",
                branch_results["full_legacy_comparison"]["final"],
            )
        )
    branches.append(("Selected proposed", results["final"]))
    print("\n=== VP branch comparison ===")
    print(
        "branch | solver_type | solver_backend | raw_objective_initial | "
        "raw_objective_final | nfev | success | position_RMSE_m | "
        "range_RMSE_m | tau_RMSE_s | message"
    )
    for label, final in branches:
        optimizer = final.get("optimizer", {})
        metrics = _vp_geometry_metrics(scene, final, true_components)
        raw_initial = final.get("raw_objective_initial")
        raw_final = final.get("raw_objective_final")
        print(
            f"{label} | {optimizer.get('method', 'unknown')} | "
            f"{optimizer.get('solver_backend', 'unknown')} | "
            f"{_fmt(raw_initial)} | {_fmt(raw_final)} | "
            f"{optimizer.get('n_eval', 'NA')} | {optimizer.get('success', 'NA')} | "
            f"{_fmt(metrics['position_RMSE_m'])} | {_fmt(metrics['range_RMSE_m'])} | "
            f"{_fmt(metrics['tau_RMSE_s'])} | {optimizer.get('message', '')}"
        )
        if (
            raw_initial is not None
            and raw_final is not None
            and np.isfinite(raw_initial)
            and np.isfinite(raw_final)
            and raw_final >= raw_initial - 1.0e-12
        ):
            print(
                f"WARNING VP_NO_GAIN: {label} raw objective did not decrease "
                f"({_fmt(raw_initial)} -> {_fmt(raw_final)})."
            )


def _print_runtime_profile(results: dict, direct_results: dict) -> None:
    """Print always-on runtime profiling collected with time.perf_counter."""
    timing = results.get("timing", {})
    print("\n=== Runtime profile ===")
    print(f"data_generation_s = {_fmt(timing.get('data_generation'))}")
    print(f"hankelization_s = {_fmt(timing.get('hankelization'))}")
    print(f"stage1_s = {_fmt(timing.get('stage1'))}")
    print(f"stage2_s = {_fmt(timing.get('stage2'))}")
    print(f"legacy_stage2_s = {_fmt(timing.get('stage2'))}")
    print(f"ris_projection_total_s = {_fmt(timing.get('ris_projection_total'))}")
    for iter_idx, update in enumerate(results["structured_diag"]["updates"], start=1):
        print(
            f"stage2_iter_{iter_idx}_mode4_assignment_time_s = "
            f"{_fmt(update.get('mode4_assignment_time_s'))}"
        )
        for detail in update.get("ris_projection_details", []):
            print(
                f"stage2_iter_{iter_idx}_ris_path_{detail.get('path')}_projection_time_s = "
                f"{_fmt(detail.get('projection_time_s'))}"
            )
    print(f"vp_s = {_fmt(timing.get('vp'))}")
    print(f"global_vp_s = {_fmt(timing.get('vp'))}")
    print(f"total_s = {_fmt(timing.get('total'))}")
    print(f"diagnostic_total_s = {_fmt(timing.get('diagnostic_total'))}")


def _print_stage1_summary(results: dict) -> None:
    """Print compact Stage-I initialization information."""
    print("\n=== Stage-I summary ===")
    _print_stage1_initialization_diagnostics(results)
    estimate = results["estimate_initial"]
    column_to_panel = estimate.get(
        "column_to_panel_assignment", estimate.get("assignment")
    )
    print(f"column_to_panel_assignment = {column_to_panel}")
    print(f"panel_to_column_assignment = {estimate.get('panel_to_column_assignment', 'NA')}")
    reliability = results.get("reliability", {})
    selected_clock_std = estimate.get("selected_clock_std")
    try:
        selected_clock_std_ns = float(selected_clock_std) * 1.0e9
    except (TypeError, ValueError):
        selected_clock_std_ns = float("nan")
    print(f"stage1_assignment_margin = {_fmt(estimate.get('stage1_assignment_margin'))}")
    print(f"stage1_selected_clock_std_ns = {_fmt(selected_clock_std_ns)}")
    print(f"delta_t_k_ns = {_fmt_vector(reliability.get('delta_t_k_ns', []))}")
    print(f"stage1_max_rank1_ratio = {_fmt(estimate.get('stage1_max_rank1_ratio'))}")
    max_ris_display = _max_ris_residual_display(reliability)
    print(
        "max_ris_residual = "
        f"{_fmt(max_ris_display) if not isinstance(max_ris_display, str) else max_ris_display}"
    )
    print(f"stage1_ris_residual_type = {estimate.get('stage1_ris_residual_type', 'NA')}")
    print(f"initial_z_residual = {_fmt(estimate.get('initial_z_residual'))}")


def _print_selected_runtime(results: dict) -> None:
    timing = results.get("timing", {})
    stage2_timing = timing
    stage2_structured = results.get("structured_diag", {})
    branches = results.get("branches", {})
    rescue_branch = branches.get("ris_only_stage2_then_vp")
    if rescue_branch is None:
        rescue_branch = branches.get("multi_hypothesis_ris_reacquisition_then_vp")
    if rescue_branch is not None and results.get("ris_stage2_no_gain", False):
        stage2_timing = rescue_branch.get("timing", timing)
        stage2_structured = rescue_branch.get("structured_diag", stage2_structured)
    print("\n=== Runtime profile ===")
    print(f"selected_branch = {results.get('selected_branch', 'unknown')}")
    print(f"data_generation_s = {_fmt(timing.get('data_generation'))}")
    print(f"hankelization_s = {_fmt(timing.get('hankelization'))}")
    print(f"stage1_s = {_fmt(timing.get('stage1'))}")
    print(
        "direct_probe_z_rescue_disabled = "
        f"{results.get('direct_probe_z_rescue_disabled', 'NA')}"
    )
    print(f"direct_probe_branch_s = {_fmt(timing.get('direct_probe_branch_s'))}")
    print(f"direct_probe_vp_s = {_fmt(timing.get('direct_probe_vp_s'))}")
    print(
        "direct_forced_z_rescue_branch_s = "
        f"{_fmt(timing.get('direct_forced_z_rescue_branch_s'))}"
    )
    print(
        "direct_forced_z_rescue_vp_s = "
        f"{_fmt(timing.get('direct_forced_z_rescue_vp_s'))}"
    )
    print(
        "direct_z_rescue_rerun_executed = "
        f"{results.get('direct_z_rescue_rerun_executed', 'NA')}"
    )
    print(
        "direct_z_rescue_rerun_skipped = "
        f"{results.get('direct_z_rescue_rerun_skipped', 'NA')}"
    )
    print(
        "direct_z_rescue_skip_reason = "
        f"{results.get('direct_z_rescue_skip_reason', '') or 'NA'}"
    )
    print(f"rescue_branch_s = {_fmt(timing.get('rescue_branch_s'))}")
    print(f"rescue_stage2_s = {_fmt(timing.get('rescue_stage2_s'))}")
    print(f"rescue_vp_s = {_fmt(timing.get('rescue_vp_s'))}")
    print(f"stage2_ris_rescue_time = {_fmt(stage2_timing.get('stage2'))}")
    print(f"ris_only_stage2_s = {_fmt(stage2_timing.get('stage2'))}")
    print(f"ris_projection_total_s = {_fmt(stage2_timing.get('ris_projection_total'))}")
    print(
        "stage2_ris_codebook_time = "
        f"{_fmt(stage2_timing.get('stage2_time_ris_codebook_build'))}"
    )
    print(
        "stage2_ris_correlation_time = "
        f"{_fmt(stage2_timing.get('stage2_time_ris_correlation'))}"
    )
    print(
        "stage2_ris_warm_start_time = "
        f"{_fmt(stage2_timing.get('stage2_time_ris_warm_start'))}"
    )
    print(
        "stage2_ris_fresnel_lift_time = "
        f"{_fmt(stage2_timing.get('stage2_time_ris_fresnel_lift'))}"
    )
    print(
        "stage2_ris_refine_time = "
        f"{_fmt(stage2_timing.get('stage2_time_ris_refine'))}"
    )
    print(f"stage2_ris_grid_used = {stage2_structured.get('stage2_ris_grid_used', 'NA')}")
    print(
        f"stage2_ris_local_window = {stage2_structured.get('stage2_ris_local_window', 'NA')}"
    )
    print(f"stage2_rescue_mode = {stage2_structured.get('stage2_rescue_mode', 'none')}")
    if stage2_structured.get("stage2_rescue_mode") == "multi_hypothesis_ris_reacquisition":
        for key in (
            "mhr_num_assignment_hypotheses",
            "mhr_num_ris_candidates",
            "mhr_num_global_hypotheses",
            "mhr_best_assignment",
            "mhr_best_clock_std_ns",
            "mhr_best_short_vp_objective",
            "mhr_best_full_vp_objective",
            "mhr_accepted",
            "mhr_runtime_total",
            "mhr_assignment_time",
            "mhr_candidate_generation_time",
            "mhr_short_vp_time",
            "mhr_full_vp_time",
        ):
            print(f"{key} = {stage2_structured.get(key, 'NA')}")
    if stage2_structured.get("stage2_rescue_mode") == "robust_jnpp":
        for key in (
            "jnpp_num_starts",
            "jnpp_num_candidates",
            "jnpp_num_subsets",
            "jnpp_weight_mode",
            "jnpp_use_confidence_weights",
            "jnpp_weights",
            "jnpp_rank1_ratios",
            "jnpp_rank1_ratios_used",
            "jnpp_use_leave_one_out",
            "jnpp_leave_one_out_effective",
            "jnpp_best_objective",
            "jnpp_best_position",
            "jnpp_best_clock_std_ns",
            "jnpp_best_delta_t_ns",
            "jnpp_gradient_mode",
            "jnpp_runtime_total",
            "jnpp_objective_eval_count",
            "jnpp_gradient_eval_count",
            "jnpp_accepted_by_raw_vp",
        ):
            print(f"{key} = {stage2_structured.get(key, 'NA')}")
        branches = results.get("branches", {})
        direct_branch = branches.get("direct_vp")
        rescue_branch = branches.get("ris_only_stage2_then_vp")
        if direct_branch is not None:
            print(
                "direct_vp_raw_objective = "
                f"{_fmt(_final_raw_objective(direct_branch.get('final', {})))}"
            )
        if rescue_branch is not None:
            print(
                "jnpp_vp_raw_objective = "
                f"{_fmt(_final_raw_objective(rescue_branch.get('final', {})))}"
            )
        print(f"selected_branch = {results.get('selected_branch', 'unknown')}")
    final = results.get("final", {})
    print(f"global_vp_cache_build_time = {_fmt(final.get('global_vp_cache_build_time'))}")
    print(
        "global_vp_residual_eval_time_mean = "
        f"{_fmt(final.get('global_vp_residual_eval_time_mean'))}"
    )
    print(
        "global_vp_residual_eval_time_total = "
        f"{_fmt(final.get('global_vp_residual_eval_time_total'))}"
    )
    print(f"global_vp_num_residual_calls = {final.get('global_vp_num_residual_calls', 'NA')}")
    print(f"global_vp_optimizer_nfev = {final.get('global_vp_optimizer_nfev', 'NA')}")
    print(
        "global_vp_fixed_anchor_runtime_s = "
        f"{_fmt(final.get('global_vp_fixed_anchor_runtime_s'))}"
    )
    print(
        "global_vp_jones_runtime_s = "
        f"{_fmt(final.get('global_vp_jones_runtime_s'))}"
    )
    print(f"adaptive_jones_triggered = {final.get('adaptive_jones_triggered', 'NA')}")
    print(
        "adaptive_jones_trigger_reason = "
        f"{final.get('adaptive_jones_trigger_reason', 'NA')}"
    )
    print(f"z_rescue_strategy = {final.get('z_rescue_strategy', 'NA')}")
    print(f"z_rescue_num_probes = {final.get('z_rescue_num_probes', 'NA')}")
    print(
        "z_rescue_num_full_refines = "
        f"{final.get('z_rescue_num_full_refines', 'NA')}"
    )
    print(
        "z_rescue_probe_runtime_s = "
        f"{_fmt(final.get('z_rescue_probe_runtime_s'))}"
    )
    print(
        "z_rescue_full_refine_runtime_s = "
        f"{_fmt(final.get('z_rescue_full_refine_runtime_s'))}"
    )
    print(f"global_vp_jacobian_mode = {final.get('global_vp_jacobian_mode', 'NA')}")
    print(f"global_vp_s = {_fmt(timing.get('vp'))}")
    if bool(results.get("stage1_config", {}).get("verbose_timing", False)):
        print("\n=== Stage-II timing detail ===")
        for key in (
            "structured_refinement_total",
            "per_iteration_total",
            "projection_per_path_total",
            "global_Z_reconstruction_time",
            "guarded_SSE_time",
            "damping_grid_time",
            "factor_copy_time",
            "pseudo_inverse_time",
            "logging_time",
            "deepcopy_time",
        ):
            print(f"{key}_s = {_fmt(stage2_timing.get(key, stage2_structured.get(key)))}")
    print(f"selected_branch_total_s = {_fmt(timing.get('total'))}")
    print(f"diagnostic_total_s = {_fmt(timing.get('diagnostic_total'))}")
    if bool(results.get("stage1_config", {}).get("verbose_timing", False)):
        print("\n=== Stage-I timing detail ===")
        for key in (
            "stage1_time_delay_estimation",
            "stage1_time_vandermonde_reconstruction",
            "stage1_time_coupled_ls",
            "stage1_time_rank1_svd_split",
            "stage1_time_assignment_total",
            "stage1_time_assignment_evs",
            "stage1_time_assignment_ris",
            "stage1_time_ris_codebook_build",
            "stage1_time_ris_projection_refine",
            "stage1_time_reliability_diagnostics",
            "stage1_time_other",
        ):
            print(f"{key}_s = {_fmt(timing.get(key, results['estimate_initial'].get(key)))}")
        reference = results.get("stage1_profile_reference")
        if reference is not None:
            paper_s = timing.get("stage1")
            heavy_s = reference.get("stage1_s")
            print(f"stage1_normal_heavy_reference_s = {_fmt(heavy_s)}")
            if (
                np.isfinite(float(paper_s))
                and np.isfinite(float(heavy_s))
                and float(paper_s) > float(heavy_s)
            ):
                print("WARNING_PAPER_BALANCED_SLOWER_THAN_HEAVY")


def print_run_summary(results: dict, config: dict) -> dict:
    """Print the concise proposed-method diagnostic report."""
    scene = results["scene"]
    true_components = results["true_components"]
    y_metrics = y_metric_summary(results["final"]["Y_hat"], results["Y_true"])
    param_metrics = parameter_errors_for_vp(scene, results["final"], true_components)

    mode = str(config.get("diagnostic_mode", "performance")).lower()
    result_label = (
        "SMOKE-TEST RESULT" if mode == "smoke" else "FULL-SIZE PERFORMANCE RESULT"
    )
    print(f"=== Single proposed diagnostic run: {result_label} ===")
    _print_run_configuration(config, results)
    _print_stage1_summary(results)
    if not results.get("progress_printed", False):
        _print_reliability_gate_diagnostics(results)
    else:
        _print_reliability_warnings(results)
    _print_vp_branch_metrics(results)
    _print_global_vp_diagnostics(results)
    if not scipy_is_available():
        print("optimizer_note = scipy.optimize not found; using deterministic fallback optimizer")
    if bool(config.get("verbose_stage2", False)) and results["structured_diag"]["updates"]:
        _print_stage_two_update_diagnostics(results)
    _print_selected_runtime(results)

    print(f"\n=== Final result: {result_label} ===")
    print(f"Y_true shape = {results['Y_true'].shape}")
    print(f"Y_noisy shape = {results['Y_noisy'].shape}")
    print(f"Y_hat shape = {results['final']['Y_hat'].shape}")
    print(f"global_VP_enabled = {results['final'].get('vp_enabled', True)}")
    print(f"selected_branch = {results.get('selected_branch', 'unknown')}")
    print(f"raw_objective_final = {_fmt(_final_raw_objective(results['final']))}")
    print(f"RMSE_Y_abs = {y_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat = {y_metrics['nmse']:.6e}")
    print(f"UE_position_RMSE_m = {param_metrics['position_rmse']:.6e}")
    return {"y": y_metrics, "parameters": param_metrics}


def _repeat_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _repeat_position(value: Any) -> np.ndarray:
    try:
        position = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.full(3, np.nan)
    if position.size != 3 or not np.all(np.isfinite(position)):
        return np.full(3, np.nan)
    return position


def _repeat_estimate_position(scene: dict, estimate: Any) -> float:
    if not isinstance(estimate, dict):
        return float("nan")
    try:
        p_hat = estimate.get("p_u")
        if p_hat is None:
            p_hat = estimate_position_from_ris_eta(scene, estimate)
        return float(np.linalg.norm(_repeat_position(p_hat) - _repeat_position(scene["p_u_true"])))
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return float("nan")


def _repeat_array3(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.full(3, np.nan)
    return array.copy() if array.size == 3 and np.all(np.isfinite(array)) else np.full(3, np.nan)


def _stage2_repeat_sidecars(
    result: dict,
    *,
    trial_id: int,
    seed: int,
    config: dict,
) -> tuple[dict, list[dict]]:
    scene = result.get("scene", {})
    diag = dict(result.get("stage2_diagnostics", {}))
    final = result.get("final", {})
    p_true = _repeat_array3(scene.get("p_u_true"))
    p_final = _repeat_array3(final.get("p_u"))
    c0 = _repeat_float(scene.get("c0", config.get("c0")))
    stage2_position = _repeat_array3(diag.get("seed_position"))
    final_clock = _repeat_float(final.get("delta_t"))
    true_clock = _repeat_float(scene.get("delta_t_true"))
    sigma_values = np.asarray(diag.get("delay_sigma_values", []), dtype=float).reshape(-1)
    linear_position = _repeat_array3(diag.get("pllg_linear_x_m"))
    projected_position = _repeat_array3(diag.get("projected_position"))
    sidecar = {
        "trial_id": int(trial_id), "seed": int(seed), "snr_db": _repeat_float(config.get("SNR_dB")),
        "true_k": int(scene.get("K", config.get("K", 0))),
        "stage2_rescue_impl": str(result.get("stage2_rescue_impl", config.get("stage2_rescue_impl", "legacy_multistart"))),
        "ngc_direct_status": str(result.get("ngc_direct_cert_status", "")),
        "ngc_direct_score": _repeat_float(result.get("ngc_direct_total_score")),
        "rescue_triggered": bool(result.get("stage2_rescue_triggered", False)),
        "stage2_force_run_for_diagnostics": bool(result.get("stage2_force_run_for_diagnostics", False)),
        "rescue_available": bool(result.get("stage2_rescue_available", False)),
        "pllg_success": bool(diag.get("pllg_success", False)),
        "pllg_failure_reason": str(diag.get("pllg_failure_reason", "")),
        "legacy_fallback_used": bool(diag.get("legacy_fallback_used", False)),
        "legacy_fallback_reason": str(diag.get("legacy_fallback_reason", "")),
        "num_valid_local_fixes": int(diag.get("num_valid_local_fixes", 0)),
        "local_weight_source": str(diag.get("local_weight_source", "")),
        "delay_variance_source": str(diag.get("delay_sigma_source", diag.get("sigma_tau_source", ""))),
        "pllg_rank": diag.get("pllg_rank", ""), "pllg_condition_number": diag.get("pllg_condition_number", ""),
        "pllg_reweight_steps": diag.get("pllg_reweight_steps", config.get("stage2_pllg_reweight_steps", 1)),
        "pllg_linear_x_m": linear_position[0], "pllg_linear_y_m": linear_position[1], "pllg_linear_z_m": linear_position[2],
        "pllg_linear_s_m": diag.get("pllg_linear_s_m", ""), "pllg_linear_clock_s": diag.get("pllg_linear_clock_s", ""),
        "pllg_projected_x_m": projected_position[0], "pllg_projected_y_m": projected_position[1], "pllg_projected_z_m": projected_position[2],
        "pllg_projection_distance_m": diag.get("pllg_projection_distance_m", ""),
        "pllg_phi_before_polish": diag.get("before_phi_stage2_normalized", ""),
        "pllg_phi_after_polish": diag.get("after_phi_stage2_normalized", ""),
        "pllg_polish_success": bool(diag.get("polish_accepted", False)),
        "pllg_linear_runtime_s": diag.get("pllg_linear_runtime_s", ""),
        "pllg_polish_runtime_s": diag.get("polish_runtime_s", ""),
        "legacy_fallback_runtime_s": diag.get("legacy_fallback_runtime_s", ""),
        "stage2_total_runtime_s": diag.get("stage2_total_runtime_s", result.get("timing", {}).get("stage2", "")),
        "seed_position_error_m": float(np.linalg.norm(stage2_position - p_true)) if np.all(np.isfinite(stage2_position)) and np.all(np.isfinite(p_true)) else "",
        "seed_z_error_m": float(stage2_position[2] - p_true[2]) if np.all(np.isfinite(stage2_position)) and np.all(np.isfinite(p_true)) else "",
        "seed_clock_error_s": _repeat_float(diag.get("seed_clock_s")) - true_clock if np.isfinite(_repeat_float(diag.get("seed_clock_s"))) and np.isfinite(true_clock) else "",
        "final_position_error_m": float(np.linalg.norm(p_final - p_true)) if np.all(np.isfinite(p_final)) and np.all(np.isfinite(p_true)) else "",
        "final_clock_error_s": final_clock - true_clock if np.isfinite(final_clock) and np.isfinite(true_clock) else "",
        "z_boundary_hit": bool(final.get("boundary_hit", False)),
        "common_ris_refinement_success": bool(diag.get("common_ris_refinement_success", False)),
        "common_ris_refinement_impl": str(diag.get("common_ris_refinement_impl", "")),
        "common_ris_refinement_runtime_s": diag.get("common_ris_refinement_runtime_s", ""),
        "common_ris_refinement_num_valid_local_fixes": diag.get("common_ris_refinement_num_valid_local_fixes", ""),
        "geometry_seed_impl": str(diag.get("geometry_seed_impl", "")),
        "pllg_pseudorange_block_weight": diag.get("pllg_pseudorange_block_weight", config.get("stage2_pllg_pseudorange_block_weight", 1.0)),
        "delay_sigma_source": str(diag.get("delay_sigma_source", diag.get("sigma_tau_source", ""))),
        "delay_sigma_used_floor": bool(diag.get("delay_sigma_used_floor", diag.get("sigma_tau_used_floor", False))),
        "delay_sigma_min_s": float(np.min(sigma_values)) if sigma_values.size else "",
        "delay_sigma_max_s": float(np.max(sigma_values)) if sigma_values.size else "",
        "delay_sigma_values_json": json.dumps(sigma_values.tolist()),
        "stage2_clock_term_raw_s2_before": diag.get("before_clock_term_raw_s2", ""),
        "stage2_clock_term_normalized_before": diag.get("before_clock_term_normalized", ""),
        "stage2_ris_term_raw_before": diag.get("before_ris_term_raw", ""),
        "stage2_ris_term_mean_before": diag.get("before_ris_term_mean", ""),
        "stage2_ris_term_normalized_before": diag.get("before_ris_term_normalized", ""),
        "stage2_phi_normalized_before": diag.get("before_phi_stage2_normalized", ""),
        "stage2_clock_term_raw_s2_after": diag.get("after_clock_term_raw_s2", ""),
        "stage2_clock_term_normalized_after": diag.get("after_clock_term_normalized", ""),
        "stage2_ris_term_raw_after": diag.get("after_ris_term_raw", ""),
        "stage2_ris_term_mean_after": diag.get("after_ris_term_mean", ""),
        "stage2_ris_term_normalized_after": diag.get("after_ris_term_normalized", ""),
        "stage2_phi_normalized_after": diag.get("after_phi_stage2_normalized", ""),
        "stage2_ris_normalization_scale": config.get("stage2_ris_normalization_scale", 1.0e-4),
        "stage2_lambda_ris_normalized": config.get("stage2_lambda_ris_normalized", 1.0),
        "polish_accepted": bool(diag.get("polish_accepted", False)),
        "stage2_clock_estimator": str(diag.get("clock_estimator", "")),
        "stage2_clock_weighted_mean_s": diag.get("clock_weighted_mean_s", ""),
        "stage2_clock_decoupled_s": diag.get("clock_decoupled_s", ""),
        "stage2_clock_decoupled_available": bool(diag.get("clock_decoupled_available", False)),
        "stage2_clock_decoupled_reason": str(diag.get("clock_decoupled_reason", "")),
        "stage2_clock_decoupled_num_inliers": diag.get("clock_decoupled_num_inliers", ""),
        "stage2_clock_decoupled_scale_m": diag.get("clock_decoupled_scale_m", ""),
        "rescue_candidate_admissible": result.get("rescue_candidate_admissible", ""),
        "selector_guard_reject_reason": str(result.get("selector_guard_reject_reason", "")),
        "selector_raw_degradation": result.get("selector_raw_degradation", ""),
        "selector_raw_relative_improvement": result.get("selector_raw_relative_improvement", ""),
        "selector_boundary_guard_used": result.get("selector_boundary_guard_used", ""),
        "selector_boundary_override_used": result.get("selector_boundary_override_used", ""),
    }
    try:
        stage1_records = build_local_fix_records(result.get("estimate_initial", {}), scene, config, source_stage="stage1")
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        stage1_records = []
    refined_records = diag.get("local_fix_records", [])
    stage1_by_panel = {int(record.get("panel_index", -1)): record for record in stage1_records}
    local_rows = []
    for record in refined_records:
        panel = int(record.get("panel_index", -1))
        position = _repeat_array3(record.get("position"))
        true_local = p_true
        err = position - true_local
        stage1_position = _repeat_array3(stage1_by_panel.get(panel, {}).get("position"))
        eta = np.asarray(record.get("eta", [np.nan] * 3), dtype=float).reshape(-1)
        local_rows.append({
            "trial_id": int(trial_id), "seed": int(seed), "snr_db": _repeat_float(config.get("SNR_dB")),
            "panel_index": panel, "assigned_panel_index": record.get("assigned_column_index", ""),
            "panel_match_correct": "", "local_fix_valid": bool(record.get("valid", False)),
            "local_fix_reject_reason": str(record.get("reject_reason", "")),
            "local_fix_x_m": position[0], "local_fix_y_m": position[1], "local_fix_z_m": position[2],
            "true_x_m": true_local[0], "true_y_m": true_local[1], "true_z_m": true_local[2],
            "local_error_x_m": err[0], "local_error_y_m": err[1], "local_error_z_m": err[2],
            "local_error_norm_m": float(np.linalg.norm(err)) if np.all(np.isfinite(err)) else "",
            "range_hat_m": eta[0] if eta.size >= 1 else "", "theta_hat_rad": eta[1] if eta.size >= 2 else "", "phi_hat_rad": eta[2] if eta.size >= 3 else "",
            "projection_residual_before": stage1_by_panel.get(panel, {}).get("residual_after", ""),
            "projection_residual_after": record.get("residual_after", ""),
            "assignment_margin": result.get("reliability", {}).get("assignment_margin", ""),
            "local_weight_source": record.get("weight_source", ""), "local_weight_scalar": record.get("weight_scalar", ""),
            "local_fix_source_stage": str(record.get("source_stage", "refined")),
            "common_refinement_impl": str(diag.get("common_ris_refinement_impl", "")),
            "stage1_local_fix_x_m": stage1_position[0], "stage1_local_fix_y_m": stage1_position[1], "stage1_local_fix_z_m": stage1_position[2],
            "refined_local_fix_x_m": position[0], "refined_local_fix_y_m": position[1], "refined_local_fix_z_m": position[2],
        })
    return sidecar, local_rows


def _empty_repeat_trial_row(
    trial_id: int,
    seed: int,
    config: dict,
) -> dict[str, Any]:
    row: dict[str, Any] = {field: float("nan") for field in REPEAT_TRIAL_FIELDS}
    row.update(
        {
            "trial_id": int(trial_id),
            "seed": int(seed),
            "snr_db": _repeat_float(config.get("SNR_dB")),
            "K": int(config.get("K", 0)),
            "failed": False,
            "error": "",
            "selected_branch": "",
            "global_vp_mode": "",
            "global_vp_backend": "",
            "global_vp_gpu_used": False,
            "global_vp_gpu_num_objective_calls": 0,
            "jones_mode": "",
            "adaptive_enabled": False,
            "boundary_hit": False,
            "boundary_hit_axis": "",
            "z_rescue_triggered": False,
            "z_rescue_num_starts": 0,
            "z_rescue_strategy": "",
            "z_rescue_num_probes": 0,
            "z_rescue_num_full_refines": 0,
            "z_rescue_refine_vp_mode": "",
            "z_rescue_selected_reason": "",
            "stage2_warm_start_mode": "",
            "adaptive_jones_triggered": False,
            "adaptive_jones_trigger_reason": "",
            "direct_boundary_hit": False,
            "rescue_boundary_hit": False,
            "boundary_selection_rule_used": False,
            "warning": "",
        }
    )
    return row


def _extract_repeat_trial_metrics(
    result: dict,
    *,
    trial_id: int,
    seed: int,
    config: dict,
    runtime_s: float,
) -> dict[str, Any]:
    row = _empty_repeat_trial_row(trial_id, seed, config)
    scene = result.get("scene", {})
    final = result.get("final", {})
    p_true = _repeat_position(scene.get("p_u_true"))
    p_hat = _repeat_position(final.get("p_u"))
    error = p_hat - p_true
    delta_t_true = _repeat_float(scene.get("delta_t_true"))
    delta_t_hat = _repeat_float(final.get("delta_t"))
    delta_t_error = delta_t_hat - delta_t_true
    y_true = result.get("Y_true")
    y_noisy = result.get("Y_noisy")
    y_hat = final.get("Y_hat")
    y_nmse = float("nan")
    residual_norm = float("nan")
    if y_true is not None and y_hat is not None:
        y_nmse = float(relative_nmse(np.asarray(y_hat), np.asarray(y_true)))
    if y_noisy is not None and y_hat is not None:
        residual_norm = float(np.linalg.norm(np.asarray(y_hat) - np.asarray(y_noisy)))
    stage_config = result.get("stage1_config", config)
    global_vp = stage_config.get("global_vp", {}) if isinstance(stage_config, dict) else {}
    global_vp_mode = str(
        final.get("global_vp_mode", final.get("vp_mode", global_vp.get("mode", "")))
    )
    jones_mode = str(
        final.get(
            "jones_mode",
            final.get("selected_vp_family_branch", global_vp_mode),
        )
    )
    boundary = distance_to_box_boundary(
        p_hat,
        np.asarray(config.get("ue_bounds"), dtype=float),
        float(global_vp.get("boundary_tol_m", 0.02)),
    )
    boundary_axis = boundary["boundary_hit_axis"]
    if isinstance(boundary_axis, list):
        boundary_axis = ",".join(boundary_axis)
    timing = result.get("timing", {})
    structured_diag = result.get("structured_diag", {})
    ris_details = (
        structured_diag.get("updates", [{}])[0].get("ris_projection_details", [])
        if structured_diag.get("updates")
        else []
    )
    stage2_warm_start_mode = next(
        (
            str(detail.get("stage2_warm_start_mode"))
            for detail in ris_details
            if detail.get("stage2_warm_start_mode")
        ),
        "",
    )
    row.update(
        {
            "runtime_s": float(runtime_s),
            "p_true_x": p_true[0],
            "p_true_y": p_true[1],
            "p_true_z": p_true[2],
            "p_hat_x": p_hat[0],
            "p_hat_y": p_hat[1],
            "p_hat_z": p_hat[2],
            "err_x_m": error[0],
            "err_y_m": error[1],
            "err_z_m": error[2],
            "position_error_m": float(np.linalg.norm(error)),
            "delta_t_true_s": delta_t_true,
            "delta_t_hat_s": delta_t_hat,
            "delta_t_error_s": delta_t_error,
            "clock_error_ns": abs(delta_t_error) * 1.0e9,
            "clock_range_error_m": _repeat_float(scene.get("c0", config.get("c0")))
            * abs(delta_t_error),
            "y_nmse": y_nmse,
            "raw_objective_final": _repeat_float(
                final.get("raw_objective_final", final.get("raw_objective"))
            ),
            "residual_norm": residual_norm,
            "noise_variance": _repeat_float(result.get("noise_variance")),
            "stage1_position_error_m": _repeat_estimate_position(
                scene, result.get("estimate_initial")
            ),
            "stage2_position_error_m": _repeat_estimate_position(
                scene, result.get("estimate_used")
            ),
            "final_position_error_m": float(np.linalg.norm(error)),
            "selected_branch": str(
                result.get("selected_branch", final.get("selected_branch", ""))
            ),
            "global_vp_mode": global_vp_mode,
            "global_vp_backend": str(final.get("global_vp_backend", "cpu")),
            "global_vp_gpu_used": bool(final.get("global_vp_gpu_used", False)),
            "global_vp_gpu_num_objective_calls": int(
                final.get("global_vp_gpu_num_objective_calls", 0)
            ),
            "global_vp_cpu_gpu_objective_rel_diff": _repeat_float(
                final.get("global_vp_cpu_gpu_objective_rel_diff")
            ),
            "global_vp_cpu_gpu_gradient_rel_diff": _repeat_float(
                final.get("global_vp_cpu_gpu_gradient_rel_diff")
            ),
            "global_vp_cpu_gpu_xhat_rel_diff": _repeat_float(
                final.get("global_vp_cpu_gpu_xhat_rel_diff")
            ),
            "jones_mode": jones_mode,
            "adaptive_enabled": bool(
                global_vp.get("mode") == "adaptive_jones"
                or "adaptive" in global_vp_mode.lower()
                or "adaptive" in jones_mode.lower()
            ),
            "boundary_hit": bool(final.get("boundary_hit", boundary["boundary_hit"])),
            "boundary_hit_axis": str(
                final.get("boundary_hit_axis", boundary_axis)
            ),
            "distance_to_position_box_boundary_m": _repeat_float(
                final.get(
                    "distance_to_position_box_boundary_m",
                    boundary["distance_to_position_box_boundary_m"],
                )
            ),
            "z_rescue_triggered": bool(final.get("z_rescue_triggered", False)),
            "z_rescue_num_starts": int(final.get("z_rescue_num_starts", 0)),
            "z_rescue_strategy": str(final.get("z_rescue_strategy", "")),
            "z_rescue_num_probes": int(final.get("z_rescue_num_probes", 0)),
            "z_rescue_num_full_refines": int(
                final.get("z_rescue_num_full_refines", 0)
            ),
            "z_rescue_probe_runtime_s": _repeat_float(
                final.get("z_rescue_probe_runtime_s")
            ),
            "z_rescue_full_refine_runtime_s": _repeat_float(
                final.get("z_rescue_full_refine_runtime_s")
            ),
            "z_rescue_refine_vp_mode": str(
                final.get("z_rescue_refine_vp_mode", "")
            ),
            "z_rescue_best_z": _repeat_float(final.get("z_rescue_best_z")),
            "z_rescue_selected_reason": str(
                final.get("z_rescue_selected_reason", "")
            ),
            "stage2_warm_start_mode": stage2_warm_start_mode,
            "stage2_ris_warm_start_runtime_s": _repeat_float(
                timing.get("stage2_time_ris_warm_start")
            ),
            "stage2_ris_fresnel_lift_runtime_s": _repeat_float(
                timing.get("stage2_time_ris_fresnel_lift")
            ),
            "stage2_ris_refine_runtime_s": _repeat_float(
                timing.get("stage2_time_ris_refine")
            ),
            "adaptive_jones_triggered": bool(
                final.get("adaptive_jones_triggered", False)
            ),
            "adaptive_jones_trigger_reason": str(
                final.get("adaptive_jones_trigger_reason", "")
            ),
            "global_vp_fixed_anchor_runtime_s": _repeat_float(
                final.get("global_vp_fixed_anchor_runtime_s")
            ),
            "global_vp_jones_runtime_s": _repeat_float(
                final.get("global_vp_jones_runtime_s")
            ),
            "global_vp_optimizer_nfev": _repeat_float(
                final.get("global_vp_optimizer_nfev")
            ),
            "global_vp_actual_residual_calls": _repeat_float(
                final.get("global_vp_num_residual_calls")
            ),
            "direct_boundary_hit": bool(
                result.get("direct_boundary_hit", final.get("direct_boundary_hit", False))
            ),
            "rescue_boundary_hit": bool(
                result.get("rescue_boundary_hit", final.get("rescue_boundary_hit", False))
            ),
            "branch_score_margin": _repeat_float(
                result.get("branch_score_margin", final.get("branch_score_margin"))
            ),
            "boundary_selection_rule_used": bool(
                result.get(
                    "boundary_selection_rule_used",
                    final.get("boundary_selection_rule_used", False),
                )
            ),
            "warning": str(result.get("warning", final.get("warning", ""))),
        }
    )
    try:
        stage2_sidecar, local_sidecar = _stage2_repeat_sidecars(
            result, trial_id=trial_id, seed=seed, config=config
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        stage2_sidecar, local_sidecar = {}, []
    row["_stage2_sidecar"] = stage2_sidecar
    row["_stage2_local_fix_sidecar"] = local_sidecar
    return row


def _apply_repeat_overrides(config: dict, overrides: dict | None) -> dict:
    if not overrides:
        return config
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = _apply_repeat_overrides(copy.deepcopy(config[key]), value)
        else:
            config[key] = copy.deepcopy(value)
    return config


def _repeat_worker_init(blas_threads: int) -> None:
    threads = str(int(blas_threads))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = threads


def _run_repeat_trial(task: dict[str, Any]) -> dict[str, Any]:
    trial_id = int(task["trial_id"])
    seed = int(task["seed"])
    config = default_config()
    config["seed"] = seed
    config = _apply_repeat_overrides(config, task.get("config_overrides"))
    row = _empty_repeat_trial_row(trial_id, seed, config)
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_single_proposed_diagnostic(config)
        row = _extract_repeat_trial_metrics(
            result,
            trial_id=trial_id,
            seed=seed,
            config=config,
            runtime_s=time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - failed trials belong in the output.
        row["failed"] = True
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["runtime_s"] = float(time.perf_counter() - start)
    return row


def _repeat_stat_values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    values = np.asarray([_repeat_float(row.get(metric)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def summarize_repeated_main_single(
    rows: list[dict[str, Any]],
    outlier_threshold_m: float = 0.1,
) -> dict[str, Any]:
    successful = [row for row in rows if not bool(row.get("failed"))]
    summary: dict[str, Any] = {
        "n_runs": len(rows),
        "n_success": len(successful),
        "n_failed": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
    }
    for metric in REPEAT_NUMERIC_METRICS:
        values = _repeat_stat_values(successful, metric)
        stats = (
            {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p10": float(np.percentile(values, 10.0)),
                "p90": float(np.percentile(values, 90.0)),
            }
            if values.size
            else {name: float("nan") for name in ("mean", "median", "std", "min", "max", "p10", "p90")}
        )
        for name, value in stats.items():
            summary[f"{metric}_{name}"] = value

    def rmse(metric: str) -> float:
        values = _repeat_stat_values(successful, metric)
        return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")

    summary.update(
        {
            "position_rmse_m": rmse("position_error_m"),
            "err_x_rmse_m": rmse("err_x_m"),
            "err_y_rmse_m": rmse("err_y_m"),
            "err_z_rmse_m": rmse("err_z_m"),
            "clock_error_ns_rmse": rmse("clock_error_ns"),
            "clock_range_error_m_rmse": rmse("clock_range_error_m"),
            "y_nmse_mean": summary.get("y_nmse_mean", float("nan")),
            "raw_objective_final_mean": summary.get(
                "raw_objective_final_mean", float("nan")
            ),
        }
    )
    valid_errors = _repeat_stat_values(successful, "position_error_m")
    inlier_errors = valid_errors[valid_errors <= float(outlier_threshold_m)]
    summary["outlier_threshold_m"] = float(outlier_threshold_m)
    summary["outlier_rate"] = (
        float(np.mean(valid_errors > float(outlier_threshold_m)))
        if valid_errors.size
        else float("nan")
    )
    summary["success_only_position_rmse_m"] = (
        float(np.sqrt(np.mean(inlier_errors**2)))
        if inlier_errors.size
        else float("nan")
    )
    return summary


def _write_repeat_csv(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _repeat_log_lines(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    started_at: str,
    n_runs: int,
    jobs: int,
    base_seed: int,
) -> list[str]:
    lines = [
        f"start_time_utc = {started_at}",
        f"command_line = {' '.join(sys.argv)}",
        f"n_runs = {n_runs}",
        f"jobs = {jobs}",
        f"seed = {base_seed}",
    ]
    for row in sorted(rows, key=lambda item: int(item["trial_id"])):
        lines.append(
            "trial "
            f"trial_id={row['trial_id']} seed={row['seed']} "
            f"position_error_m={_fmt(row.get('position_error_m'))} "
            f"clock_error_ns={_fmt(row.get('clock_error_ns'))} "
            f"y_nmse={_fmt(row.get('y_nmse'))} "
            f"runtime_s={_fmt(row.get('runtime_s'))} "
            f"failed={row.get('failed')}"
        )
    lines.extend(
        [
            "final_summary",
            f"success_rate = {_fmt(summary.get('success_rate'))}",
            f"position_rmse_m = {_fmt(summary.get('position_rmse_m'))}",
            f"err_x_rmse_m = {_fmt(summary.get('err_x_rmse_m'))}",
            f"err_y_rmse_m = {_fmt(summary.get('err_y_rmse_m'))}",
            f"err_z_rmse_m = {_fmt(summary.get('err_z_rmse_m'))}",
            f"clock_error_ns_rmse = {_fmt(summary.get('clock_error_ns_rmse'))}",
            f"clock_range_error_m_rmse = {_fmt(summary.get('clock_range_error_m_rmse'))}",
            f"y_nmse_mean = {_fmt(summary.get('y_nmse_mean'))}",
            f"runtime_s_mean = {_fmt(summary.get('runtime_s_mean'))}",
            f"runtime_s_std = {_fmt(summary.get('runtime_s_std'))}",
        ]
    )
    return lines


def run_repeated_main_single(
    n_runs: int,
    jobs: int,
    base_seed: int,
    out_dir: pathlib.Path,
    config_overrides: dict | None = None,
    blas_threads: int = 1,
    rerun_seeds: list[int] | None = None,
    outlier_threshold_m: float = 0.1,
    resource_options: dict[str, Any] | None = None,
) -> dict:
    if rerun_seeds is not None:
        rerun_seeds = [int(seed) for seed in rerun_seeds]
        n_runs = len(rerun_seeds)
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if blas_threads <= 0:
        raise ValueError("blas_threads must be positive")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    if rerun_seeds is None:
        seed_sequence = np.random.SeedSequence(int(base_seed))
        seeds = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in seed_sequence.spawn(int(n_runs))
        ]
    else:
        seeds = list(rerun_seeds)
    tasks = [
        {
            "trial_id": trial_id,
            "seed": seed,
            "config_overrides": copy.deepcopy(config_overrides or {}),
        }
        for trial_id, seed in enumerate(seeds)
    ]
    process_count = min(int(jobs), int(n_runs))
    rows: list[dict[str, Any]] = []
    with mp.Pool(
        processes=process_count,
        initializer=_repeat_worker_init,
        initargs=(int(blas_threads),),
    ) as pool:
        for row in pool.imap_unordered(_run_repeat_trial, tasks, chunksize=1):
            rows.append(row)
            print(
                "repeat trial completed: "
                f"trial_id={row['trial_id']} seed={row['seed']} "
                f"position_error_m={_fmt(row.get('position_error_m'))} "
                f"clock_error_ns={_fmt(row.get('clock_error_ns'))} "
                f"y_nmse={_fmt(row.get('y_nmse'))} "
                f"runtime_s={_fmt(row.get('runtime_s'))}"
            )
    rows.sort(key=lambda item: int(item["trial_id"]))
    summary = summarize_repeated_main_single(rows, outlier_threshold_m)
    trial_path = out_dir / "main_single_repeat_trials.csv"
    outlier_path = out_dir / "main_single_repeat_outliers.csv"
    summary_path = out_dir / "main_single_repeat_summary.csv"
    log_path = out_dir / "main_single_repeat.log"
    metadata_path = out_dir / "main_single_repeat_metadata.json"
    _write_repeat_csv(trial_path, rows, REPEAT_TRIAL_FIELDS)
    stage2_rows = [row.get("_stage2_sidecar", {}) for row in rows if row.get("_stage2_sidecar")]
    local_rows = [local for row in rows for local in row.get("_stage2_local_fix_sidecar", [])]
    _write_repeat_csv(out_dir / "stage2_diagnostics.csv", stage2_rows, STAGE2_DIAGNOSTIC_FIELDS)
    _write_repeat_csv(
        out_dir / "stage2_local_fix_diagnostics.csv",
        local_rows,
        STAGE2_LOCAL_FIX_DIAGNOSTIC_FIELDS,
    )
    outlier_rows = [
        row
        for row in rows
        if bool(row.get("boundary_hit"))
        or _repeat_float(row.get("position_error_m")) > float(outlier_threshold_m)
    ]
    _write_repeat_csv(outlier_path, outlier_rows, REPEAT_TRIAL_FIELDS)
    _write_repeat_csv(summary_path, [summary], list(summary))
    log_path.write_text(
        "\n".join(
            _repeat_log_lines(
                rows,
                summary,
                started_at=started_at,
                n_runs=n_runs,
                jobs=process_count,
                base_seed=base_seed,
            )
        )
        + "\n"
    )
    metadata = {
        "start_time_utc": started_at,
        "finish_time_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(sys.argv),
        "n_runs": int(n_runs),
        "jobs": process_count,
        "base_seed": int(base_seed),
        "trial_seeds": seeds,
        "rerun_seeds_provided": rerun_seeds is not None,
        "outlier_threshold_m": float(outlier_threshold_m),
        "blas_threads": int(blas_threads),
        "config_overrides": config_overrides or {},
        "resource_options": resource_options or {},
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return {
        "rows": rows,
        "summary": summary,
        "metadata": metadata,
        "out_dir": out_dir,
        "outlier_rows": outlier_rows,
    }


def _print_repeated_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print("Repeated proposed diagnostic finished.")
    print(
        f"n_runs = {summary['n_runs']}, "
        f"success = {summary['n_success']}/{summary['n_runs']}"
    )
    print(f"Position RMSE = {_fmt(summary.get('position_rmse_m'))} m")
    print(f"Outlier rate = {_fmt(summary.get('outlier_rate'))}")
    print(
        "Success-only position RMSE = "
        f"{_fmt(summary.get('success_only_position_rmse_m'))} m"
    )
    print(
        "x/y/z RMSE = "
        f"{_fmt(summary.get('err_x_rmse_m'))} / "
        f"{_fmt(summary.get('err_y_rmse_m'))} / "
        f"{_fmt(summary.get('err_z_rmse_m'))} m"
    )
    print(f"Clock RMSE = {_fmt(summary.get('clock_error_ns_rmse'))} ns")
    print(
        "Clock range RMSE = "
        f"{_fmt(summary.get('clock_range_error_m_rmse'))} m"
    )
    print(f"Channel NMSE mean = {_fmt(summary.get('y_nmse_mean'))}")
    print(f"Results written to: {result['out_dir']}")


def _parse_snr_db_values(value: str | float | int | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    values = [
        float(token.strip())
        for token in str(value).split(",")
        if token.strip()
    ]
    if not values:
        raise ValueError("--snr-db must contain at least one numeric value")
    return values


def _resolve_alias_int(
    parser: argparse.ArgumentParser,
    *,
    canonical: str,
    values: list[tuple[str, int | None]],
    default: int,
) -> tuple[int, bool]:
    provided = [(name, int(value)) for name, value in values if value is not None]
    if provided:
        first_name, first_value = provided[0]
        conflicts = [
            f"{name}={value}"
            for name, value in provided[1:]
            if value != first_value
        ]
        if conflicts:
            all_values = ", ".join([f"{first_name}={first_value}", *conflicts])
            parser.error(f"conflicting {canonical} aliases: {all_values}")
        return first_value, True
    return int(default), False


def _snr_out_dir_name(snr_db: float) -> str:
    text = f"{float(snr_db):g}".replace(".", "p")
    return f"snr_{text}dB"


def _repeat_requested(args: argparse.Namespace) -> bool:
    # A run count or worker count of one describes the single-run path, so it
    # must not pull in the repeat machinery and its CSV/metadata side effects.
    return bool(
        (args.repeat_runs_was_set and args.repeat_runs > 1)
        or (args.repeat_jobs_was_set and args.repeat_jobs > 1)
        or args.repeat_out_dir_was_set
        or args.rerun_seeds is not None
        or (args.snr_db is not None and len(args.snr_db) > 1)
    )


def run_default_diagnostic(config: dict | None = None) -> None:
    """Run and print the default proposed diagnostic report."""
    if config is None:
        config = default_config()
    config = _apply_main_single_defaults(config)
    results = run_single_proposed_diagnostic(config)
    print_run_summary(results, config)


def _run_compact(config: dict) -> dict:
    """Run the full pipeline once and return compact sweep metrics."""
    config = _apply_main_single_defaults(config)
    results = _run_single_pipeline(config, use_structured=True)
    y_true = results["Y_true"]
    y_noisy = results["Y_noisy"]
    final_y = results["final"]["Y_hat"]
    true_components = results["true_components"]
    final_params = parameter_errors_for_vp(results["scene"], results["final"], true_components)
    structured_params = parameter_errors_for_structured(
        results["scene"], results["estimate_used"], true_components
    )
    return {
        "NMSE_Y_noisy": relative_nmse(y_noisy, y_true),
        "NMSE_Y_hat_final": relative_nmse(final_y, y_true),
        "position_RMSE_final": final_params["position_rmse"],
        "global_VP_enabled": bool(results["final"].get("vp_enabled", True)),
        "range_RMSE_after_structured": structured_params["range_rmse"],
        "selected_branch": results.get("selected_branch", "unknown"),
    }


def run_snr_sweep() -> None:
    """Run a small one-seed SNR diagnostic sweep."""
    print("=== Diagnostic SNR sweep ===")
    snrs = [-10.0, 0.0, 10.0, 20.0, 30.0]
    position_errors = []
    for snr in snrs:
        config = default_config()
        config["SNR_dB"] = snr
        metrics = _run_compact(config)
        position_errors.append(metrics["position_RMSE_final"])
        print(
            f"SNR_dB={snr:.1f}, "
            f"NMSE_Y_noisy={metrics['NMSE_Y_noisy']:.6e}, "
            f"NMSE_Y_hat_final={metrics['NMSE_Y_hat_final']:.6e}, "
            f"position_RMSE_final={metrics['position_RMSE_final']:.6e}, "
            f"global_VP_enabled={metrics['global_VP_enabled']}, "
            f"selected_branch={metrics['selected_branch']}"
        )
    if position_errors[-1] > position_errors[0]:
        print("WARNING GEOM_DEGRADE: UE position RMSE did not improve from -10 dB to 30 dB.")


def run_mr_sweep() -> None:
    """Run a small RIS-size diagnostic sweep."""
    print("=== Diagnostic M_R sweep ===")
    cases = [((4, 4), 18), ((8, 8), 32), ((16, 16), 64)]
    for ris_shape, t_dim in cases:
        config = default_config()
        config["ris_shape"] = ris_shape
        config["T"] = t_dim
        metrics = _run_compact(config)
        print(
            f"M_Rx={ris_shape[0]}, M_Ry={ris_shape[1]}, M_R={ris_shape[0] * ris_shape[1]}, "
            f"T={t_dim}, "
            f"range_RMSE_after_structured={metrics['range_RMSE_after_structured']:.6e}, "
            f"position_RMSE_final={metrics['position_RMSE_final']:.6e}, "
            f"global_VP_enabled={metrics['global_VP_enabled']}, "
            f"selected_branch={metrics['selected_branch']}"
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for default diagnostics and optional sweeps."""
    argv_list = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-snr-sweep", action="store_true")
    parser.add_argument("--diagnostic-mr-sweep", action="store_true")
    parser.add_argument("--run-full-legacy-comparison", action="store_true")
    parser.add_argument("--verbose-stage2", action="store_true")
    parser.add_argument(
        "--diagnostic-mode",
        choices=("smoke", "performance"),
        default=None,
        help="Run a quick smoke diagnostic or the full-size performance diagnostic.",
    )
    parser.add_argument("--full-size-diagnostic", action="store_true")
    parser.add_argument("--full-stage1-search", action="store_true")
    parser.add_argument("--repeat-runs", type=int, default=None)
    parser.add_argument("--mc", type=int, default=None, help="Alias for --repeat-runs.")
    parser.add_argument(
        "--rerun-seeds",
        type=str,
        default=None,
        help="Comma-separated uint32 seeds; bypasses SeedSequence generation.",
    )
    parser.add_argument("--repeat-jobs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None, help="Alias for --repeat-jobs.")
    parser.add_argument(
        "--process-workers",
        type=int,
        default=None,
        help="Alias for --repeat-jobs.",
    )
    parser.add_argument(
        "--repeat-out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/main_single_repeat"),
    )
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument(
        "--snr-db",
        default=None,
        help=(
            "Single SNR or comma-separated list. For negative lists, prefer "
            "--snr-db=-30,-25,-20 to avoid argparse treating values as options."
        ),
    )
    parser.add_argument("--paper-k", type=int, default=None)
    parser.add_argument(
        "--stage2-rescue-impl",
        choices=("legacy_multistart", "pllg"),
        default=None,
    )
    parser.add_argument("--stage2-pllg-reweight-steps", type=int, choices=(0, 1), default=None)
    parser.add_argument("--stage2-pllg-cond-max", type=float, default=None)
    parser.add_argument("--stage2-pllg-pseudorange-block-weight", type=float, default=None)
    parser.add_argument(
        "--stage2-clock-estimator",
        choices=("decoupled_robust", "weighted_mean"),
        default=None,
    )
    parser.add_argument("--stage2-clock-sigma-range-m", type=float, default=None)
    parser.add_argument("--stage2-clock-outlier-kappa", type=float, default=None)
    parser.add_argument("--stage2-delay-sigma-floor-ns", type=float, default=None)
    parser.add_argument("--stage2-ris-normalization-scale", type=float, default=None)
    parser.add_argument("--stage2-lambda-ris-normalized", type=float, default=None)
    parser.add_argument("--stage2-pllg-max-projection-distance-m", type=float, default=None)
    parser.add_argument(
        "--stage2-pllg-local-weight-mode",
        choices=("auto", "uniform"),
        default=None,
    )
    parser.add_argument(
        "--stage2-pllg-legacy-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--stage2-force-run-for-diagnostics", action="store_true")
    parser.add_argument(
        "--stage2-selector-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--stage2-selector-raw-degradation-abs-tol", type=float, default=None)
    parser.add_argument("--stage2-selector-raw-degradation-rel-tol", type=float, default=None)
    parser.add_argument(
        "--stage2-selector-boundary-override-min-rel-improvement",
        type=float,
        default=None,
    )
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--global-vp-backend",
        choices=("cpu", "cupy", "auto"),
        default=None,
    )
    parser.add_argument("--global-vp-gpu-device", type=int, default=None)
    parser.add_argument(
        "--global-vp-validate-gpu-against-cpu",
        action="store_true",
    )
    parser.add_argument("--memory-budget-gb", type=float, default=None)
    parser.add_argument("--memory-per-worker-gb", type=float, default=None)
    parser.add_argument(
        "--trim-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    args = parser.parse_args(argv_list)

    args.repeat_runs, args.repeat_runs_was_set = _resolve_alias_int(
        parser,
        canonical="repeat run count",
        values=(("--repeat-runs", args.repeat_runs), ("--mc", args.mc)),
        default=1,
    )
    args.repeat_jobs, args.repeat_jobs_was_set = _resolve_alias_int(
        parser,
        canonical="repeat job count",
        values=(
            ("--repeat-jobs", args.repeat_jobs),
            ("--jobs", args.jobs),
            ("--process-workers", args.process_workers),
        ),
        default=1,
    )
    args.snr_db = _parse_snr_db_values(args.snr_db)
    args.repeat_out_dir_was_set = any(
        token == "--repeat-out-dir" or token.startswith("--repeat-out-dir=")
        for token in argv_list
    )
    args.seed_was_set = any(
        token == "--seed" or token.startswith("--seed=") for token in argv_list
    )

    rerun_seeds = None
    if args.rerun_seeds is not None:
        rerun_seeds = [
            int(token.strip())
            for token in args.rerun_seeds.split(",")
            if token.strip()
        ]
        if not rerun_seeds:
            raise ValueError("--rerun-seeds must contain at least one seed")
        if any(seed < 0 or seed > np.iinfo(np.uint32).max for seed in rerun_seeds):
            raise ValueError("--rerun-seeds values must be uint32 integers")

    if _repeat_requested(args):
        if args.repeat_jobs <= 0:
            raise ValueError("--repeat-jobs must be positive")
        if args.blas_threads <= 0:
            raise ValueError("--blas-threads must be positive")
        if args.paper_k is not None and args.paper_k <= 0:
            raise ValueError("--paper-k must be positive")
        if args.memory_budget_gb is not None and args.memory_budget_gb <= 0:
            raise ValueError("--memory-budget-gb must be positive")
        if args.memory_per_worker_gb is not None and args.memory_per_worker_gb <= 0:
            raise ValueError("--memory-per-worker-gb must be positive")

        snr_values: list[float | None] = (
            [None] if args.snr_db is None else [float(value) for value in args.snr_db]
        )
        multi_snr = len(snr_values) > 1
        base_overrides: dict[str, Any] = {}
        if args.paper_k is not None:
            base_overrides["K"] = int(args.paper_k)
        if args.stage2_rescue_impl is not None:
            base_overrides["stage2_rescue_impl"] = str(args.stage2_rescue_impl)
        if args.stage2_pllg_reweight_steps is not None:
            base_overrides["stage2_pllg_reweight_steps"] = int(args.stage2_pllg_reweight_steps)
        if args.stage2_pllg_cond_max is not None:
            base_overrides["stage2_pllg_cond_max"] = float(args.stage2_pllg_cond_max)
        for argument, key in (
            ("stage2_pllg_pseudorange_block_weight", "stage2_pllg_pseudorange_block_weight"),
            ("stage2_clock_estimator", "stage2_clock_estimator"),
            ("stage2_clock_sigma_range_m", "stage2_clock_sigma_range_m"),
            ("stage2_clock_outlier_kappa", "stage2_clock_outlier_kappa"),
            ("stage2_delay_sigma_floor_ns", "stage2_delay_sigma_floor_ns"),
            ("stage2_ris_normalization_scale", "stage2_ris_normalization_scale"),
            ("stage2_lambda_ris_normalized", "stage2_lambda_ris_normalized"),
            ("stage2_pllg_max_projection_distance_m", "stage2_pllg_max_projection_distance_m"),
            ("stage2_selector_guard", "stage2_selector_guard"),
            ("stage2_selector_raw_degradation_abs_tol", "stage2_selector_raw_degradation_abs_tol"),
            ("stage2_selector_raw_degradation_rel_tol", "stage2_selector_raw_degradation_rel_tol"),
            ("stage2_selector_boundary_override_min_rel_improvement", "stage2_selector_boundary_override_min_rel_improvement"),
        ):
            value = getattr(args, argument)
            if value is not None:
                base_overrides[key] = value
        if args.stage2_pllg_local_weight_mode is not None:
            base_overrides["stage2_pllg_local_weight_mode"] = str(args.stage2_pllg_local_weight_mode)
        if args.stage2_pllg_legacy_fallback is not None:
            base_overrides["stage2_pllg_legacy_fallback"] = bool(args.stage2_pllg_legacy_fallback)
        if args.stage2_force_run_for_diagnostics:
            base_overrides["stage2_force_run_for_diagnostics"] = True
        global_vp_overrides = {}
        if args.global_vp_backend is not None:
            global_vp_overrides["backend"] = args.global_vp_backend
        if args.global_vp_gpu_device is not None:
            global_vp_overrides["gpu_device"] = int(args.global_vp_gpu_device)
        if args.global_vp_validate_gpu_against_cpu:
            global_vp_overrides["validate_gpu_against_cpu"] = True
        if global_vp_overrides:
            base_overrides["global_vp"] = global_vp_overrides
        resource_options = {
            "memory_budget_gb": args.memory_budget_gb,
            "memory_per_worker_gb": args.memory_per_worker_gb,
            "trim_memory": bool(args.trim_memory),
        }
        backend_label = (
            args.global_vp_backend
            if args.global_vp_backend is not None
            else default_config().get("global_vp", {}).get("backend", "cpu")
        )
        gpu_device_label = (
            int(args.global_vp_gpu_device)
            if args.global_vp_gpu_device is not None
            else default_config().get("global_vp", {}).get("gpu_device", 0)
        )
        for snr_db in snr_values:
            out_dir = (
                args.repeat_out_dir / _snr_out_dir_name(float(snr_db))
                if multi_snr and snr_db is not None
                else args.repeat_out_dir
            )
            trial_path = out_dir / "main_single_repeat_trials.csv"
            if trial_path.exists() and not args.force_rerun:
                raise FileExistsError(
                    f"{trial_path} already exists; use --force-rerun to overwrite"
                )
            overrides = copy.deepcopy(base_overrides)
            if snr_db is not None:
                overrides["SNR_dB"] = float(snr_db)
            print(
                "repeat setup: "
                f"snr_db={snr_db if snr_db is not None else 'default'} "
                f"repeat_runs={len(rerun_seeds) if rerun_seeds is not None else int(args.repeat_runs)} "
                f"repeat_jobs={int(args.repeat_jobs)} "
                f"backend={backend_label} "
                f"gpu_device={gpu_device_label} "
                f"out_dir={out_dir}"
            )
            result = run_repeated_main_single(
                n_runs=len(rerun_seeds) if rerun_seeds is not None else int(args.repeat_runs),
                jobs=int(args.repeat_jobs),
                base_seed=int(args.seed),
                out_dir=out_dir,
                config_overrides=overrides,
                blas_threads=int(args.blas_threads),
                rerun_seeds=rerun_seeds,
                outlier_threshold_m=float(args.outlier_threshold_m),
                resource_options=resource_options,
            )
            _print_repeated_summary(result)
    elif args.diagnostic_snr_sweep:
        run_snr_sweep()
    elif args.diagnostic_mr_sweep:
        run_mr_sweep()
    else:
        config = default_config()
        if args.seed_was_set:
            # Previously --seed only reached the repeat path, so single runs
            # silently reused config["seed"] and every --seed value produced an
            # identical realization.
            config["seed"] = int(args.seed)
        if args.snr_db is not None:
            config["SNR_dB"] = float(args.snr_db[0])
        if args.stage2_rescue_impl is not None:
            config["stage2_rescue_impl"] = str(args.stage2_rescue_impl)
        if args.stage2_pllg_reweight_steps is not None:
            config["stage2_pllg_reweight_steps"] = int(args.stage2_pllg_reweight_steps)
        if args.stage2_pllg_cond_max is not None:
            config["stage2_pllg_cond_max"] = float(args.stage2_pllg_cond_max)
        for argument, key in (
            ("stage2_pllg_pseudorange_block_weight", "stage2_pllg_pseudorange_block_weight"),
            ("stage2_clock_estimator", "stage2_clock_estimator"),
            ("stage2_clock_sigma_range_m", "stage2_clock_sigma_range_m"),
            ("stage2_clock_outlier_kappa", "stage2_clock_outlier_kappa"),
            ("stage2_delay_sigma_floor_ns", "stage2_delay_sigma_floor_ns"),
            ("stage2_ris_normalization_scale", "stage2_ris_normalization_scale"),
            ("stage2_lambda_ris_normalized", "stage2_lambda_ris_normalized"),
            ("stage2_pllg_max_projection_distance_m", "stage2_pllg_max_projection_distance_m"),
            ("stage2_selector_guard", "stage2_selector_guard"),
            ("stage2_selector_raw_degradation_abs_tol", "stage2_selector_raw_degradation_abs_tol"),
            ("stage2_selector_raw_degradation_rel_tol", "stage2_selector_raw_degradation_rel_tol"),
            ("stage2_selector_boundary_override_min_rel_improvement", "stage2_selector_boundary_override_min_rel_improvement"),
        ):
            value = getattr(args, argument)
            if value is not None:
                config[key] = value
        if args.stage2_pllg_local_weight_mode is not None:
            config["stage2_pllg_local_weight_mode"] = str(args.stage2_pllg_local_weight_mode)
        if args.stage2_pllg_legacy_fallback is not None:
            config["stage2_pllg_legacy_fallback"] = bool(args.stage2_pllg_legacy_fallback)
        if args.stage2_force_run_for_diagnostics:
            config["stage2_force_run_for_diagnostics"] = True
        if args.global_vp_backend is not None:
            config["global_vp"]["backend"] = args.global_vp_backend
        if args.global_vp_gpu_device is not None:
            config["global_vp"]["gpu_device"] = int(args.global_vp_gpu_device)
        if args.global_vp_validate_gpu_against_cpu:
            config["global_vp"]["validate_gpu_against_cpu"] = True
        if args.diagnostic_mode is not None:
            config["diagnostic_mode"] = args.diagnostic_mode
            if args.diagnostic_mode == "smoke":
                config["diagnostic_fast_problem_size"] = True
                config["diagnostic_fast_stage1_search"] = True
        if args.run_full_legacy_comparison:
            config["run_full_legacy_comparison"] = True
        if args.verbose_stage2:
            config["verbose_stage2"] = True
        if args.full_size_diagnostic:
            config["diagnostic_fast_problem_size"] = False
        if args.full_stage1_search:
            config["diagnostic_fast_stage1_search"] = False
        run_default_diagnostic(config)


if __name__ == "__main__":
    main()
