"""Paired Monte Carlo comparison of old 4-D Jones-VP and CCOP-JVP.

Each trial generates the raw observation and Stage-I estimate exactly once.
Deep copies of that same Stage-I estimate are then passed to the frozen 4-D
Jones-VP objective and the experimental common-clock-profiled 3-D objective.

The runner deliberately does not activate CP-NGC, LG rescue, RDC, or assignment
rescue.  It isolates the Stage-III optimization change validated by
``check_ccop_jvp_equivalence``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import pathlib
import platform
import shlex
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.ccop_jvp import refine_ccop_jvp
from src.config import default_config
from ..cp_ngc import cp_ngc_stage1_vector
from src.global_vp import (
    _global_exact_spherical_vp_refinement_lbfgsb_reduced,
    build_jones_vp_dictionary,
    distance_to_box_boundary,
    extract_stage1_jones_directions,
)
from src.main_single_proposed import (
    _apply_main_single_defaults,
    _make_data,
    build_stage2_delay_uncertainty,
    run_stage1_only,
)
from src.metrics import relative_nmse, rmse_abs
from src.utils import scipy_is_available
from src.validation_artifacts import (
    array_sha256,
    canonical_hash,
    deterministic_stage1_output,
    validation_environment,
)
from src.experiments.resource_control import apply_thread_limits, resolve_hybrid_resources


ROUTE_OLD = "old_4d_jones_vp"
ROUTE_CCOP = "ccop_jvp_profiled_3d"

TRIAL_FIELDS = [
    "trial_id",
    "seed",
    "snr_db",
    "route",
    "failed",
    "error",
    "shared_data_hash",
    "diagnostic_mode",
    "K",
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
    "range_rmse_m",
    "tau_rmse_s",
    "channel_y_nmse",
    "channel_y_rmse_abs",
    "raw_objective_final",
    "regularized_objective_final",
    "outlier_flag",
    "boundary_hit",
    "outer_success",
    "outer_message",
    "outer_n_iter",
    "outer_n_eval",
    "nonlinear_dim",
    "linear_nuisance_dim",
    "clock_certified",
    "clock_certificate_gap_objective",
    "clock_fft_peak_gap_objective",
    "clock_bnb_splits",
    "ccop_position_evaluations",
    "ccop_clock_interval_evaluations",
    "old_incumbent_used",
    "incumbent_non_degradation",
    "data_runtime_s",
    "stage1_runtime_s",
    "route_refinement_runtime_s",
    "metric_reconstruction_runtime_s",
    "route_runtime_s",
    "end_to_end_incremental_runtime_s",
    "incumbent_generation_runtime_s",
    "deployment_runtime_s",
    "paired_trial_wall_runtime_s",
    "noise_variance",
    "vp_backend",
]

DIFFERENCE_FIELDS = [
    "trial_id",
    "seed",
    "snr_db",
    "failed",
    "error",
    "shared_data_hash",
    "delta_position_error_m",
    "delta_clock_error_ns",
    "delta_range_rmse_m",
    "delta_tau_rmse_s",
    "delta_channel_y_nmse",
    "delta_raw_objective",
    "delta_regularized_objective",
    "delta_route_runtime_s",
    "delta_deployment_runtime_s",
    "ccop_position_win",
    "ccop_clock_win",
    "ccop_channel_win",
    "ccop_raw_objective_win",
    "ccop_regularized_objective_non_degradation",
    "ccop_incremental_runtime_win",
    "old_outlier",
    "ccop_outlier",
    "outlier_rescued",
    "outlier_introduced",
]

ARTIFACT_MANIFEST_FIELDS = [
    "trial_id",
    "seed",
    "resolved_config_hash",
    "y_noisy_hash",
    "stage1_input_hash",
    "stage1_output_hash",
    "old_candidate_hash",
    "ccop_candidate_hash",
    "artifact_payload_hash",
    "artifact_json",
    "artifact_npz",
    "artifact_runtime_s",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _git_value(arguments: list[str], default: str = "unavailable") -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=pathlib.Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return default
    return result.stdout.strip() or default


def _environment(command: str) -> dict:
    environment = validation_environment(
        command, repo_root=pathlib.Path(__file__).resolve().parents[3]
    )
    environment["scipy_optimizer_available"] = bool(scipy_is_available())
    return environment


def _trial_seed(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _build_config(spec: dict, seed: int) -> dict:
    config = default_config()
    config["seed"] = int(seed)
    config["SNR_dB"] = float(spec["snr_db"])
    config["print_progress"] = False
    config["verbose_stage2"] = False
    config["run_full_legacy_comparison"] = False
    mode = str(spec["diagnostic_mode"])
    config["diagnostic_mode"] = "smoke" if mode == "fast" else "performance"
    config["diagnostic_fast_problem_size"] = mode == "fast"
    config["diagnostic_fast_stage1_search"] = mode == "fast"
    config = _apply_main_single_defaults(config)
    config["global_vp"] = dict(config.get("global_vp", {}))
    config["global_vp"].update(
        {
            "solver": "lbfgsb_reduced",
            "mode": str(spec["jones_mode"]),
            "backend": str(spec["old_vp_backend"]),
            "gpu_device": int(spec["gpu_device"]),
            "vp_dictionary_mode": "matrix_free",
            "use_weight": False,
            "use_delay_prior": False,
            "jones_diagonal_loading": 0.0,
            "enable_z_rescue_multistart": False,
            "use_multistart": False,
            "max_iter": int(spec["old_max_iter"]),
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
        }
    )
    config["ccop_jvp"] = {
        "clock_fft_size": int(spec["clock_fft_size"]),
        "clock_abs_tol_objective": float(spec["clock_abs_tol"]),
        "clock_rel_tol": float(spec["clock_rel_tol"]),
        "clock_max_intervals": int(spec["clock_max_intervals"]),
        "outer_max_iter": int(spec["ccop_outer_max_iter"]),
        "outer_ftol": 1.0e-12,
        "outer_gtol": 1.0e-8,
    }
    return config


def _empty_trial_row(
    trial_id: int,
    seed: int,
    route: str,
    spec: dict,
) -> dict:
    row = {field: "" for field in TRIAL_FIELDS}
    row.update(
        {
            "trial_id": int(trial_id),
            "seed": int(seed),
            "snr_db": float(spec["snr_db"]),
            "route": route,
            "failed": True,
            "error": "not_run",
            "diagnostic_mode": str(spec["diagnostic_mode"]),
            "old_incumbent_used": bool(
                route == ROUTE_CCOP and spec["use_old_incumbent"]
            ),
            "vp_backend": str(spec["old_vp_backend"])
            if route == ROUTE_OLD
            else "numpy_cpu",
        }
    )
    return row


def _estimated_ranges_taus(p_u: np.ndarray, delta_t: float, scene: dict) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(p_u, dtype=float).reshape(3)
    ranges = np.linalg.norm(
        position[None, :] - np.asarray(scene["ris_centers"], dtype=float), axis=1
    )
    taus = (
        ranges + np.asarray(scene["d_RB"], dtype=float)
    ) / float(scene["c0"]) + float(delta_t)
    return ranges, taus


def _route_row(
    *,
    trial_id: int,
    seed: int,
    route: str,
    estimate: dict,
    y_hat: np.ndarray,
    data: dict,
    config: dict,
    spec: dict,
    data_hash: str,
    data_runtime_s: float,
    stage1_runtime_s: float,
    refinement_runtime_s: float,
    reconstruction_runtime_s: float,
    old_runtime_s: float,
    paired_wall_runtime_s: float,
) -> dict:
    scene = data["scene"]
    p_true = np.asarray(scene["p_u_true"], dtype=float).reshape(3)
    p_hat = np.asarray(estimate["p_u"], dtype=float).reshape(3)
    error = p_hat - p_true
    delta_t_true = float(scene["delta_t_true"])
    delta_t_hat = float(estimate["delta_t"])
    delta_t_error = delta_t_hat - delta_t_true
    ranges_hat, taus_hat = _estimated_ranges_taus(p_hat, delta_t_hat, scene)
    ranges_true = np.asarray(data["true_components"]["ranges"], dtype=float)
    taus_true = np.asarray(data["true_components"]["taus"], dtype=float)
    optimizer = dict(estimate.get("optimizer", {}))
    bounds = distance_to_box_boundary(
        p_hat,
        np.asarray(config["ue_bounds"], dtype=float),
        float(config["global_vp"].get("boundary_tol_m", 0.02)),
    )
    route_runtime = float(refinement_runtime_s + reconstruction_runtime_s)
    incremental_end_to_end = float(data_runtime_s + stage1_runtime_s + route_runtime)
    incumbent_generation = float(
        old_runtime_s
        if route == ROUTE_CCOP and bool(spec["use_old_incumbent"])
        else 0.0
    )
    deployment_runtime = float(incremental_end_to_end + incumbent_generation)
    position_error = float(np.linalg.norm(error))
    row = _empty_trial_row(trial_id, seed, route, spec)
    row.update(
        {
            "failed": False,
            "error": "",
            "shared_data_hash": data_hash,
            "K": int(scene["K"]),
            "p_true_x": float(p_true[0]),
            "p_true_y": float(p_true[1]),
            "p_true_z": float(p_true[2]),
            "p_hat_x": float(p_hat[0]),
            "p_hat_y": float(p_hat[1]),
            "p_hat_z": float(p_hat[2]),
            "err_x_m": float(error[0]),
            "err_y_m": float(error[1]),
            "err_z_m": float(error[2]),
            "position_error_m": position_error,
            "delta_t_true_s": delta_t_true,
            "delta_t_hat_s": delta_t_hat,
            "delta_t_error_s": float(delta_t_error),
            "clock_error_ns": float(abs(delta_t_error) * 1.0e9),
            "range_rmse_m": float(np.sqrt(np.mean((ranges_hat - ranges_true) ** 2))),
            "tau_rmse_s": float(np.sqrt(np.mean((taus_hat - taus_true) ** 2))),
            "channel_y_nmse": float(relative_nmse(y_hat, data["Y_true"])),
            "channel_y_rmse_abs": float(rmse_abs(y_hat, data["Y_true"])),
            "raw_objective_final": float(
                estimate.get("raw_objective_final", estimate.get("raw_objective", np.nan))
            ),
            "regularized_objective_final": float(
                estimate.get("total_objective_final", estimate.get("total_objective", np.nan))
            ),
            "outlier_flag": bool(position_error > float(spec["outlier_threshold_m"])),
            "boundary_hit": bool(bounds["boundary_hit"]),
            "outer_success": bool(
                optimizer.get("success", estimate.get("global_vp_success", False))
            ),
            "outer_message": str(
                optimizer.get("message", estimate.get("global_vp_message", ""))
            ),
            "outer_n_iter": int(
                optimizer.get("n_iter", estimate.get("global_vp_num_iter", 0))
            ),
            "outer_n_eval": int(
                optimizer.get("n_eval", len(estimate.get("objective_history", [])))
            ),
            "nonlinear_dim": int(estimate.get("nonlinear_dim", 4 if route == ROUTE_OLD else 3)),
            "linear_nuisance_dim": int(estimate.get("linear_nuisance_dim", 2 * scene["K"])),
            "clock_certified": bool(estimate.get("clock_certified", False)),
            "clock_certificate_gap_objective": float(
                estimate.get("clock_certificate_gap_objective", np.nan)
            ),
            "clock_fft_peak_gap_objective": float(
                estimate.get("clock_fft_peak_gap_objective", np.nan)
            ),
            "clock_bnb_splits": int(estimate.get("clock_bnb_splits", 0)),
            "ccop_position_evaluations": int(estimate.get("ccop_position_evaluations", 0)),
            "ccop_clock_interval_evaluations": int(
                estimate.get("ccop_clock_interval_evaluations", 0)
            ),
            "incumbent_non_degradation": bool(
                estimate.get("incumbent_non_degradation", False)
            ),
            "data_runtime_s": float(data_runtime_s),
            "stage1_runtime_s": float(stage1_runtime_s),
            "route_refinement_runtime_s": float(refinement_runtime_s),
            "metric_reconstruction_runtime_s": float(reconstruction_runtime_s),
            "route_runtime_s": route_runtime,
            "end_to_end_incremental_runtime_s": incremental_end_to_end,
            "incumbent_generation_runtime_s": incumbent_generation,
            "deployment_runtime_s": deployment_runtime,
            "paired_trial_wall_runtime_s": float(paired_wall_runtime_s),
            "noise_variance": float(data["noise_variance"]),
            "vp_backend": str(estimate.get("global_vp_backend", row["vp_backend"])),
        }
    )
    return row


def _difference_row(old: dict, ccop: dict) -> dict:
    row = {field: "" for field in DIFFERENCE_FIELDS}
    row.update(
        {
            "trial_id": int(old["trial_id"]),
            "seed": int(old["seed"]),
            "snr_db": float(old["snr_db"]),
            "failed": bool(old["failed"] or ccop["failed"]),
            "error": "; ".join(
                value for value in (str(old["error"]), str(ccop["error"])) if value
            ),
            "shared_data_hash": str(old.get("shared_data_hash", "")),
        }
    )
    if row["failed"]:
        return row

    def delta(field: str) -> float:
        return float(ccop[field]) - float(old[field])

    old_outlier = bool(old["outlier_flag"])
    ccop_outlier = bool(ccop["outlier_flag"])
    row.update(
        {
            "delta_position_error_m": delta("position_error_m"),
            "delta_clock_error_ns": delta("clock_error_ns"),
            "delta_range_rmse_m": delta("range_rmse_m"),
            "delta_tau_rmse_s": delta("tau_rmse_s"),
            "delta_channel_y_nmse": delta("channel_y_nmse"),
            "delta_raw_objective": delta("raw_objective_final"),
            "delta_regularized_objective": delta("regularized_objective_final"),
            "delta_route_runtime_s": delta("route_runtime_s"),
            "delta_deployment_runtime_s": delta("deployment_runtime_s"),
            "ccop_position_win": bool(float(ccop["position_error_m"]) < float(old["position_error_m"])),
            "ccop_clock_win": bool(float(ccop["clock_error_ns"]) < float(old["clock_error_ns"])),
            "ccop_channel_win": bool(float(ccop["channel_y_nmse"]) < float(old["channel_y_nmse"])),
            "ccop_raw_objective_win": bool(
                float(ccop["raw_objective_final"]) <= float(old["raw_objective_final"])
            ),
            "ccop_regularized_objective_non_degradation": bool(
                float(ccop["regularized_objective_final"])
                <= float(old["regularized_objective_final"]) + 1.0e-12
            ),
            "ccop_incremental_runtime_win": bool(
                float(ccop["route_runtime_s"]) < float(old["route_runtime_s"])
            ),
            "old_outlier": old_outlier,
            "ccop_outlier": ccop_outlier,
            "outlier_rescued": bool(old_outlier and not ccop_outlier),
            "outlier_introduced": bool(not old_outlier and ccop_outlier),
        }
    )
    return row


def _failure_rows(
    trial_id: int,
    seed: int,
    spec: dict,
    error: BaseException,
) -> dict:
    message = f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=8)}"
    old = _empty_trial_row(trial_id, seed, ROUTE_OLD, spec)
    ccop = _empty_trial_row(trial_id, seed, ROUTE_CCOP, spec)
    old["error"] = message
    ccop["error"] = message
    return {"trial_rows": [old, ccop], "difference_row": _difference_row(old, ccop)}


def _candidate_repro_payload(estimate: dict) -> dict:
    """Return deterministic final-candidate state, excluding runtime/history."""
    return {
        "p_u": np.asarray(estimate["p_u"], dtype=float).reshape(3),
        "delta_t": float(estimate["delta_t"]),
        "x_hat": np.asarray(estimate.get("x_hat", np.array([], dtype=complex))),
        "raw_objective_final": float(
            estimate.get("raw_objective_final", estimate.get("raw_objective", np.nan))
        ),
        "regularized_objective_final": float(
            estimate.get(
                "total_objective_final", estimate.get("total_objective", np.nan)
            )
        ),
        "selected_candidate": str(estimate.get("selected_candidate", "")),
        "optimizer_success": bool(
            dict(estimate.get("optimizer", {})).get(
                "success", estimate.get("global_vp_success", False)
            )
        ),
    }


def _boundary_artifact(position: np.ndarray, config: dict) -> dict:
    return distance_to_box_boundary(
        np.asarray(position, dtype=float).reshape(3),
        np.asarray(config["ue_bounds"], dtype=float),
        float(config["global_vp"].get("boundary_tol_m", 0.02)),
    )


def _save_validation_artifact(
    *,
    artifact_dir: pathlib.Path,
    trial_id: int,
    seed: int,
    environment: dict,
    config: dict,
    data: dict,
    stage1_estimate: dict,
    old_estimate: dict,
    ccop_estimate: dict,
    old_row: dict,
    ccop_row: dict,
) -> dict:
    """Persist small exact sidecars without changing the frozen CSV schemas."""
    artifact_start = time.perf_counter()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scene = data["scene"]
    k_paths = int(scene["K"])
    resolved_config_hash = canonical_hash(config)
    y_noisy_hash = array_sha256(data["Y_noisy"])
    stage1_input_hash = canonical_hash(
        {
            "Z_noisy": data["Z_noisy"],
            "scene": scene,
            "resolved_config_hash": resolved_config_hash,
        }
    )
    stage1_output_hash = canonical_hash(
        deterministic_stage1_output(stage1_estimate)
    )
    old_payload = _candidate_repro_payload(old_estimate)
    ccop_payload = _candidate_repro_payload(ccop_estimate)
    old_candidate_hash = canonical_hash(old_payload)
    ccop_candidate_hash = canonical_hash(ccop_payload)

    z_hat = cp_ngc_stage1_vector(stage1_estimate, scene)
    stage1_delays = z_hat[:k_paths].copy()
    ris_eta = np.asarray(stage1_estimate["ris_eta"], dtype=float).reshape(k_paths, 3)
    anchors = extract_stage1_jones_directions(stage1_estimate, scene)
    top_l_raw = list(stage1_estimate.get("stage1_shortlisted_assignments", []))
    top_l = (
        np.asarray(top_l_raw, dtype=int).reshape(-1, k_paths)
        if top_l_raw
        else np.empty((0, k_paths), dtype=int)
    )
    delay_uncertainty = build_stage2_delay_uncertainty(
        stage1_estimate, scene, config
    )
    old_boundary = _boundary_artifact(old_estimate["p_u"], config)
    ccop_boundary = _boundary_artifact(ccop_estimate["p_u"], config)
    deterministic_payload = {
        "trial_id": int(trial_id),
        "seed": int(seed),
        "resolved_config_hash": resolved_config_hash,
        "y_noisy_hash": y_noisy_hash,
        "stage1_input_hash": stage1_input_hash,
        "stage1_output_hash": stage1_output_hash,
        "old_candidate_hash": old_candidate_hash,
        "ccop_candidate_hash": ccop_candidate_hash,
        "z_hat": z_hat,
        "stage1_delays": stage1_delays,
        "ris_eta": ris_eta,
        "jones_direction_anchors": anchors,
        "assignment": np.asarray(stage1_estimate.get("assignment", []), dtype=int),
        "panel_to_column_assignment": np.asarray(
            stage1_estimate.get("panel_to_column_assignment", []), dtype=int
        ),
        "top_l_assignments": top_l,
        "rank_one_ratios": np.asarray(
            stage1_estimate.get("stage1_rank1_ratios", []), dtype=float
        ),
        "delay_uncertainty_s": np.asarray(
            delay_uncertainty["sigma_tau_s"], dtype=float
        ),
        "ris_residuals": np.asarray(
            stage1_estimate.get("stage1_ris_residuals", []), dtype=float
        ),
        "clock_replicas_s": np.asarray(
            stage1_estimate.get("selected_clock_offsets", []), dtype=float
        ),
        "old_candidate": old_payload,
        "ccop_candidate": ccop_payload,
    }
    artifact_payload_hash = canonical_hash(deterministic_payload)
    stem = f"trial_{int(trial_id):06d}_seed_{int(seed)}"
    json_path = artifact_dir / f"{stem}.json"
    npz_path = artifact_dir / f"{stem}.npz"
    np.savez(
        npz_path,
        z_hat=z_hat,
        stage1_delay_estimates_s=stage1_delays,
        ris_eta=ris_eta,
        jones_direction_anchors=anchors,
        assignment=np.asarray(stage1_estimate.get("assignment", []), dtype=int),
        panel_to_column_assignment=np.asarray(
            stage1_estimate.get("panel_to_column_assignment", []), dtype=int
        ),
        top_l_assignments=top_l,
        rank_one_ratios=np.asarray(
            stage1_estimate.get("stage1_rank1_ratios", []), dtype=float
        ),
        delay_singular_values=np.asarray(
            stage1_estimate.get("stage1_delay_singular_values", []), dtype=float
        ),
        delay_uncertainty_s=np.asarray(delay_uncertainty["sigma_tau_s"], dtype=float),
        ris_residuals=np.asarray(
            stage1_estimate.get("stage1_ris_residuals", []), dtype=float
        ),
        clock_replicas_s=np.asarray(
            stage1_estimate.get("selected_clock_offsets", []), dtype=float
        ),
        old_p_u=np.asarray(old_estimate["p_u"], dtype=float),
        old_delta_t_s=np.asarray(float(old_estimate["delta_t"])),
        old_x_hat=np.asarray(old_estimate.get("x_hat", np.array([], dtype=complex))),
        ccop_p_u=np.asarray(ccop_estimate["p_u"], dtype=float),
        ccop_delta_t_s=np.asarray(float(ccop_estimate["delta_t"])),
        ccop_x_hat=np.asarray(ccop_estimate.get("x_hat", np.array([], dtype=complex))),
        covariance_z=np.empty((0, 0), dtype=float),
    )
    metadata = {
        "trial_id": int(trial_id),
        "seed": int(seed),
        "environment": environment,
        "dimensions": {
            key: int(scene[key]) for key in ("K", "M_A", "I", "N", "P", "L", "T", "M_R")
        },
        "snr_db": float(config["SNR_dB"]),
        "resolved_config_hash": resolved_config_hash,
        "y_noisy_hash": y_noisy_hash,
        "stage1_input_hash": stage1_input_hash,
        "stage1_output_hash": stage1_output_hash,
        "old_candidate_hash": old_candidate_hash,
        "ccop_candidate_hash": ccop_candidate_hash,
        "artifact_payload_hash": artifact_payload_hash,
        "stage1": {
            "assignment": stage1_estimate.get("assignment", []),
            "panel_to_column_assignment": stage1_estimate.get(
                "panel_to_column_assignment", []
            ),
            "top_l_assignment_hypotheses": top_l_raw,
            "all_assignment_scores": stage1_estimate.get("all_assignment_scores", []),
            "assignment_margin": float(
                stage1_estimate.get("assignment_margin", np.nan)
            ),
            "rank_one_ratios": stage1_estimate.get("stage1_rank1_ratios", []),
            "delay_uncertainty_source": str(delay_uncertainty["source"]),
            "delay_uncertainty_used_floor": bool(delay_uncertainty["used_floor"]),
            "ris_residual_type": str(
                stage1_estimate.get("stage1_ris_residual_type", "")
            ),
            "boundary_diagnostics": stage1_estimate.get("stage1_boundary_hit", []),
            "validity": {
                "delay": stage1_estimate.get("stage1_delay_valid", []),
                "local_geometry": stage1_estimate.get(
                    "stage1_local_geometry_valid", []
                ),
                "assignment_confident": stage1_estimate.get(
                    "stage1_assignment_confident", []
                ),
            },
        },
        "candidates": {
            "old_4d": {
                **old_payload,
                "boundary": old_boundary,
                "metrics": old_row,
            },
            "ccop": {
                **ccop_payload,
                "boundary": ccop_boundary,
                "metrics": ccop_row,
            },
        },
        "covariance_z": {
            "available": False,
            "source": "not_yet_implemented_reproducibility_gate",
        },
        "runtime_buckets_s": {
            "data": float(old_row["data_runtime_s"]),
            "stage1": float(old_row["stage1_runtime_s"]),
            "old_route": float(old_row["route_runtime_s"]),
            "ccop_route": float(ccop_row["route_runtime_s"]),
            "paired_wall": float(old_row["paired_trial_wall_runtime_s"]),
        },
        "npz_file": npz_path.name,
    }
    json_path.write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "trial_id": int(trial_id),
        "seed": int(seed),
        "resolved_config_hash": resolved_config_hash,
        "y_noisy_hash": y_noisy_hash,
        "stage1_input_hash": stage1_input_hash,
        "stage1_output_hash": stage1_output_hash,
        "old_candidate_hash": old_candidate_hash,
        "ccop_candidate_hash": ccop_candidate_hash,
        "artifact_payload_hash": artifact_payload_hash,
        "artifact_json": str(json_path),
        "artifact_npz": str(npz_path),
        "artifact_runtime_s": float(time.perf_counter() - artifact_start),
    }


def _run_trial(task: dict) -> dict:
    trial_id = int(task["trial_id"])
    seed = int(task["seed"])
    spec = dict(task["spec"])
    apply_thread_limits(int(task["blas_threads"]))
    trial_start = time.perf_counter()
    try:
        config = _build_config(spec, seed)
        data_start = time.perf_counter()
        data = _make_data(config)
        data_runtime = time.perf_counter() - data_start
        config["noise_variance"] = float(data["noise_variance"])
        data_hash = hashlib.sha256(
            np.ascontiguousarray(data["Y_noisy"]).view(np.uint8)
        ).hexdigest()[:20]

        stage1_start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            stage1_record = run_stage1_only(data, config)
        stage1_runtime = time.perf_counter() - stage1_start
        stage1_estimate = stage1_record["estimate"]

        old_start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            old_estimate = _global_exact_spherical_vp_refinement_lbfgsb_reduced(
                data["Y_noisy"], copy.deepcopy(stage1_estimate), data["scene"], config
            )
        old_runtime = time.perf_counter() - old_start
        old_y_hat = np.asarray(old_estimate["Y_hat"], dtype=complex)

        ccop_start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            ccop_estimate = refine_ccop_jvp(
                data["Y_noisy"],
                copy.deepcopy(stage1_estimate),
                data["scene"],
                config,
                incumbent=old_estimate if bool(spec["use_old_incumbent"]) else None,
            )
        ccop_refinement_runtime = time.perf_counter() - ccop_start
        reconstruction_start = time.perf_counter()
        dictionary = build_jones_vp_dictionary(
            ccop_estimate["p_u"],
            ccop_estimate["delta_t"],
            data["scene"],
            config,
        )
        ccop_y_hat = (
            dictionary @ np.asarray(ccop_estimate["x_hat"], dtype=complex)
        ).reshape(data["Y_noisy"].shape)
        reconstruction_runtime = time.perf_counter() - reconstruction_start
        paired_wall = time.perf_counter() - trial_start

        old_row = _route_row(
            trial_id=trial_id,
            seed=seed,
            route=ROUTE_OLD,
            estimate=old_estimate,
            y_hat=old_y_hat,
            data=data,
            config=config,
            spec=spec,
            data_hash=data_hash,
            data_runtime_s=data_runtime,
            stage1_runtime_s=stage1_runtime,
            refinement_runtime_s=old_runtime,
            reconstruction_runtime_s=0.0,
            old_runtime_s=old_runtime,
            paired_wall_runtime_s=paired_wall,
        )
        ccop_row = _route_row(
            trial_id=trial_id,
            seed=seed,
            route=ROUTE_CCOP,
            estimate=ccop_estimate,
            y_hat=ccop_y_hat,
            data=data,
            config=config,
            spec=spec,
            data_hash=data_hash,
            data_runtime_s=data_runtime,
            stage1_runtime_s=stage1_runtime,
            refinement_runtime_s=ccop_refinement_runtime,
            reconstruction_runtime_s=reconstruction_runtime,
            old_runtime_s=old_runtime,
            paired_wall_runtime_s=paired_wall,
        )
        artifact_manifest = None
        artifact_dir = task.get("artifact_dir")
        if artifact_dir:
            artifact_manifest = _save_validation_artifact(
                artifact_dir=pathlib.Path(artifact_dir),
                trial_id=trial_id,
                seed=seed,
                environment=dict(task.get("environment", {})),
                config=config,
                data=data,
                stage1_estimate=stage1_estimate,
                old_estimate=old_estimate,
                ccop_estimate=ccop_estimate,
                old_row=old_row,
                ccop_row=ccop_row,
            )
        return {
            "trial_rows": [old_row, ccop_row],
            "difference_row": _difference_row(old_row, ccop_row),
            "artifact_manifest": artifact_manifest,
        }
    except Exception as error:  # noqa: BLE001 - failed trials are data.
        return _failure_rows(trial_id, seed, spec, error)


def _write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerows(rows)


def _finite(rows: list[dict], field: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _metric_stats(values: np.ndarray, prefix: str) -> dict:
    if values.size == 0:
        return {
            f"{prefix}_{name}": float("nan")
            for name in ("mean", "median", "p90", "p95", "rmse")
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90.0)),
        f"{prefix}_p95": float(np.percentile(values, 95.0)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
    }


def _route_summaries(rows: list[dict]) -> list[dict]:
    summaries = []
    for route in (ROUTE_OLD, ROUTE_CCOP):
        selected = [row for row in rows if row["route"] == route]
        successful = [row for row in selected if not bool(row["failed"])]
        summary = {
            "route": route,
            "n_trials": len(selected),
            "n_success": len(successful),
            "n_failed": len(selected) - len(successful),
            "failure_rate": float(1.0 - len(successful) / len(selected)) if selected else float("nan"),
            "outlier_rate": float(np.mean([bool(row["outlier_flag"]) for row in successful]))
            if successful
            else float("nan"),
            "outer_success_rate": float(np.mean([bool(row["outer_success"]) for row in successful]))
            if successful
            else float("nan"),
            "clock_certificate_rate": float(np.mean([bool(row["clock_certified"]) for row in successful]))
            if route == ROUTE_CCOP and successful
            else float("nan"),
        }
        for field in (
            "position_error_m",
            "clock_error_ns",
            "range_rmse_m",
            "tau_rmse_s",
            "channel_y_nmse",
            "raw_objective_final",
            "regularized_objective_final",
            "route_runtime_s",
            "deployment_runtime_s",
            "outer_n_iter",
            "outer_n_eval",
        ):
            summary.update(_metric_stats(_finite(successful, field), field))
        summaries.append(summary)
    return summaries


def _paired_summary(rows: list[dict]) -> dict:
    successful = [row for row in rows if not bool(row["failed"])]
    summary = {
        "n_trials": len(rows),
        "n_paired_success": len(successful),
        "paired_failure_rate": float(1.0 - len(successful) / len(rows)) if rows else float("nan"),
    }
    for field in (
        "delta_position_error_m",
        "delta_clock_error_ns",
        "delta_range_rmse_m",
        "delta_tau_rmse_s",
        "delta_channel_y_nmse",
        "delta_raw_objective",
        "delta_regularized_objective",
        "delta_route_runtime_s",
        "delta_deployment_runtime_s",
    ):
        summary.update(_metric_stats(_finite(successful, field), field))
    for field in (
        "ccop_position_win",
        "ccop_clock_win",
        "ccop_channel_win",
        "ccop_raw_objective_win",
        "ccop_regularized_objective_non_degradation",
        "ccop_incremental_runtime_win",
        "outlier_rescued",
        "outlier_introduced",
    ):
        summary[f"{field}_rate"] = (
            float(np.mean([bool(row[field]) for row in successful]))
            if successful
            else float("nan")
        )
    return summary


def _summary_markdown(
    route_summaries: list[dict],
    paired: dict,
    *,
    resource_plan: dict,
    old_backend: str,
    use_old_incumbent: bool,
) -> str:
    lookup = {row["route"]: row for row in route_summaries}
    old = lookup[ROUTE_OLD]
    ccop = lookup[ROUTE_CCOP]

    def fmt(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "NA"
        return f"{number:.6g}" if np.isfinite(number) else "NA"

    lines = [
        "# Paired old 4-D Jones-VP vs CCOP-JVP",
        "",
        "Each seed shares the same `Y_noisy` and Stage-I estimate.",
        "",
        "| Metric | Old 4-D | CCOP-JVP |",
        "|---|---:|---:|",
        f"| Successful trials | {old['n_success']} | {ccop['n_success']} |",
        f"| Position RMSE (m) | {fmt(old['position_error_m_rmse'])} | {fmt(ccop['position_error_m_rmse'])} |",
        f"| Position median (m) | {fmt(old['position_error_m_median'])} | {fmt(ccop['position_error_m_median'])} |",
        f"| Position p95 (m) | {fmt(old['position_error_m_p95'])} | {fmt(ccop['position_error_m_p95'])} |",
        f"| Clock RMSE (ns) | {fmt(old['clock_error_ns_rmse'])} | {fmt(ccop['clock_error_ns_rmse'])} |",
        f"| Channel Y-NMSE median | {fmt(old['channel_y_nmse_median'])} | {fmt(ccop['channel_y_nmse_median'])} |",
        f"| Outlier rate | {fmt(old['outlier_rate'])} | {fmt(ccop['outlier_rate'])} |",
        f"| Incremental route runtime median (s) | {fmt(old['route_runtime_s_median'])} | {fmt(ccop['route_runtime_s_median'])} |",
        f"| Deployment runtime median (s) | {fmt(old['deployment_runtime_s_median'])} | {fmt(ccop['deployment_runtime_s_median'])} |",
        "",
        "## Paired differences",
        "",
        "Differences are CCOP minus old; negative error/objective values favor CCOP.",
        "",
        f"- Position win rate: {fmt(paired['ccop_position_win_rate'])}",
        f"- Channel-NMSE win rate: {fmt(paired['ccop_channel_win_rate'])}",
        f"- Regularized-objective non-degradation rate: {fmt(paired['ccop_regularized_objective_non_degradation_rate'])}",
        f"- Outlier rescued rate: {fmt(paired['outlier_rescued_rate'])}",
        f"- Outlier introduced rate: {fmt(paired['outlier_introduced_rate'])}",
        "",
    ]
    if int(resource_plan["process_workers"]) != 1:
        lines.append(
            "Runtime warning: multiple worker processes were used; runtime values are throughput diagnostics, not a clean wall-clock comparison."
        )
    if old_backend != "cpu":
        lines.append(
            "Runtime warning: old VP used a non-CPU backend while CCOP remains NumPy/CPU, so route runtimes are not backend-fair."
        )
    if use_old_incumbent:
        lines.append(
            "CCOP deployment runtime includes old-VP incumbent generation; `route_runtime_s` reports only the CCOP increment plus its Y-hat reconstruction."
        )
    lines.extend(
        [
            "",
            "This runner isolates Stage-III CCOP profiling. It does not test CP-NGC covariance calibration, LG rescue, RDC fallback, or top-L assignment rescue.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_blas_threads(value: str) -> int | str:
    if str(value).lower() == "auto":
        return "auto"
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--blas-threads must be positive or 'auto'")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired old 4-D Jones-VP versus CCOP-JVP Monte Carlo runner."
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument(
        "--diagnostic-mode", choices=("performance", "fast"), default="performance"
    )
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    parser.add_argument(
        "--jones-mode",
        choices=("jones_regularized", "jones_free"),
        default="jones_regularized",
    )
    parser.add_argument("--old-max-iter", type=int, default=80)
    parser.add_argument("--ccop-outer-max-iter", type=int, default=20)
    parser.add_argument("--clock-fft-size", type=int, default=4096)
    parser.add_argument("--clock-abs-tol", type=float, default=1.0e-12)
    parser.add_argument("--clock-rel-tol", type=float, default=1.0e-10)
    parser.add_argument("--clock-max-intervals", type=int, default=20000)
    parser.add_argument(
        "--use-old-incumbent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--old-vp-backend", choices=("cpu", "cupy", "auto"), default="cpu"
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--process-workers", type=int, default=None)
    parser.add_argument("--blas-threads", default="auto")
    parser.add_argument("--memory-budget-gb", type=float, default=None)
    parser.add_argument("--memory-per-worker-gb", type=float, default=None)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/ccop_paired_mc"),
    )
    parser.add_argument(
        "--save-validation-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save per-realization reproducibility sidecars without changing "
            "paired_trials.csv or paired_differences.csv schemas."
        ),
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_trials <= 0 or args.jobs <= 0:
        raise ValueError("--n-trials and --jobs must be positive")
    if args.outlier_threshold_m <= 0.0:
        raise ValueError("--outlier-threshold-m must be positive")
    blas_threads = _parse_blas_threads(args.blas_threads)
    resource_plan = resolve_hybrid_resources(
        jobs=int(args.jobs),
        process_workers=args.process_workers,
        blas_threads=blas_threads,
        n_tasks=int(args.n_trials),
        memory_budget_gb=args.memory_budget_gb,
        memory_per_worker_gb=args.memory_per_worker_gb,
    )
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "oldcode.ccop_validation.experiments.run_ccop_paired_mc",
            *(argv or sys.argv[1:]),
        ]
    )
    environment = _environment(command)
    print(f"command = {command}", flush=True)
    print(f"resource_plan = {json.dumps(resource_plan, sort_keys=True)}", flush=True)
    if int(resource_plan["process_workers"]) != 1:
        print(
            "WARNING: parallel workers make runtime a throughput diagnostic; use --jobs 1 for timing claims.",
            flush=True,
        )
    if args.old_vp_backend != "cpu":
        print(
            "WARNING: CCOP is NumPy/CPU; a non-CPU old backend is not a fair runtime comparison.",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trial_path = args.out_dir / "paired_trials.csv"
    difference_path = args.out_dir / "paired_differences.csv"
    artifact_dir = args.out_dir / "artifacts"
    artifact_manifest_path = args.out_dir / "artifact_manifest.csv"
    protected = [
        trial_path,
        difference_path,
        args.out_dir / "route_summary.csv",
        args.out_dir / "paired_summary.csv",
        args.out_dir / "config.json",
        args.out_dir / "summary.md",
    ]
    if bool(args.save_validation_artifacts):
        protected.append(artifact_manifest_path)
    if not args.force_rerun and any(path.exists() for path in protected):
        raise FileExistsError(
            f"outputs already exist under {args.out_dir}; use --force-rerun to overwrite"
        )
    _write_csv(trial_path, [], TRIAL_FIELDS)
    _write_csv(difference_path, [], DIFFERENCE_FIELDS)
    if bool(args.save_validation_artifacts):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(artifact_manifest_path, [], ARTIFACT_MANIFEST_FIELDS)

    seed_sequence = np.random.SeedSequence(int(args.seed))
    seeds = [_trial_seed(child) for child in seed_sequence.spawn(int(args.n_trials))]
    spec = {
        "snr_db": float(args.snr_db),
        "diagnostic_mode": str(args.diagnostic_mode),
        "outlier_threshold_m": float(args.outlier_threshold_m),
        "jones_mode": str(args.jones_mode),
        "old_max_iter": int(args.old_max_iter),
        "ccop_outer_max_iter": int(args.ccop_outer_max_iter),
        "clock_fft_size": int(args.clock_fft_size),
        "clock_abs_tol": float(args.clock_abs_tol),
        "clock_rel_tol": float(args.clock_rel_tol),
        "clock_max_intervals": int(args.clock_max_intervals),
        "use_old_incumbent": bool(args.use_old_incumbent),
        "old_vp_backend": str(args.old_vp_backend),
        "gpu_device": int(args.gpu_device),
    }
    tasks = [
        {
            "trial_id": trial_id,
            "seed": seed,
            "spec": spec,
            "blas_threads": int(resource_plan["blas_threads"]),
            "artifact_dir": str(artifact_dir)
            if bool(args.save_validation_artifacts)
            else "",
            "environment": environment,
        }
        for trial_id, seed in enumerate(seeds)
    ]
    representative_config = _build_config(spec, seeds[0])
    config_record = {
        "experiment": "paired old 4-D Jones-VP versus CCOP-JVP",
        "environment": environment,
        "arguments": vars(args),
        "resource_plan": resource_plan,
        "trial_seeds": seeds,
        "shared_trial_contract": "one Y_noisy and one Stage-I estimate per seed",
        "runtime_contract": {
            "route_runtime_s": "route refinement plus CCOP-only Y_hat reconstruction",
            "deployment_runtime_s": (
                "data + Stage-I + old incumbent generation + CCOP route"
                if args.use_old_incumbent
                else "data + Stage-I + route"
            ),
        },
        "representative_config": representative_config,
        "representative_config_hash": canonical_hash(representative_config),
        "artifact_contract": {
            "enabled": bool(args.save_validation_artifacts),
            "directory": str(artifact_dir)
            if bool(args.save_validation_artifacts)
            else "",
            "csv_schema_unchanged": True,
            "stage1_output_hash_excludes_timing_only": True,
            "covariance_z_status": "not_yet_implemented_reproducibility_gate",
        },
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(_jsonable(config_record), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trial_rows: list[dict] = []
    difference_rows: list[dict] = []
    artifact_manifest_rows: list[dict] = []

    def record(result: dict) -> None:
        rows = list(result["trial_rows"])
        difference = dict(result["difference_row"])
        trial_rows.extend(rows)
        difference_rows.append(difference)
        artifact = result.get("artifact_manifest")
        if artifact is not None:
            artifact_manifest_rows.append(dict(artifact))
            _append_csv(
                artifact_manifest_path,
                [dict(artifact)],
                ARTIFACT_MANIFEST_FIELDS,
            )
        _append_csv(trial_path, rows, TRIAL_FIELDS)
        _append_csv(difference_path, [difference], DIFFERENCE_FIELDS)
        status = "failed" if difference["failed"] else "ok"
        print(
            f"trial {int(difference['trial_id']) + 1}/{args.n_trials} "
            f"seed={difference['seed']} status={status}",
            flush=True,
        )
        if difference["failed"] and not args.continue_on_error:
            raise RuntimeError(str(difference["error"]))

    if int(resource_plan["process_workers"]) == 1:
        for task in tasks:
            record(_run_trial(task))
    else:
        with ProcessPoolExecutor(
            max_workers=int(resource_plan["process_workers"])
        ) as executor:
            future_to_task = {
                executor.submit(_run_trial, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                record(future.result())

    trial_rows.sort(key=lambda row: (int(row["trial_id"]), str(row["route"])))
    difference_rows.sort(key=lambda row: int(row["trial_id"]))
    artifact_manifest_rows.sort(key=lambda row: int(row["trial_id"]))
    _write_csv(trial_path, trial_rows, TRIAL_FIELDS)
    _write_csv(difference_path, difference_rows, DIFFERENCE_FIELDS)
    if bool(args.save_validation_artifacts):
        _write_csv(
            artifact_manifest_path,
            artifact_manifest_rows,
            ARTIFACT_MANIFEST_FIELDS,
        )
    route_summary = _route_summaries(trial_rows)
    paired_summary = _paired_summary(difference_rows)
    route_fields = sorted({key for row in route_summary for key in row})
    paired_fields = sorted(paired_summary)
    _write_csv(args.out_dir / "route_summary.csv", route_summary, route_fields)
    _write_csv(
        args.out_dir / "paired_summary.csv", [paired_summary], paired_fields
    )
    (args.out_dir / "summary.md").write_text(
        _summary_markdown(
            route_summary,
            paired_summary,
            resource_plan=resource_plan,
            old_backend=str(args.old_vp_backend),
            use_old_incumbent=bool(args.use_old_incumbent),
        ),
        encoding="utf-8",
    )
    print(f"outputs = {args.out_dir}", flush=True)
    if any(bool(row["failed"]) for row in difference_rows) and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
