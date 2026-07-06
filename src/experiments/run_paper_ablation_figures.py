"""Generate paper ablation CSVs and PDF figures for the revised estimator."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import pathlib
import platform
import subprocess
import sys
import time
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.channel_model import channel_components, evs_component_selection
    from src.config import default_config
    from src.diagnostics import estimate_position_from_ris_eta
    from src.geometry import polarization_vector
    from src.estimators import (
        reconstruct_raw_tensor_from_structured_estimate,
        estimate_position_from_local_ris,
        global_exact_spherical_vp_refinement,
    )
    from src.global_vp import (
        _build_global_dictionary,
        _solve_linear_vp_regularized,
        data_only_efim_diagnostic,
    )
    from src.main_single_proposed import (
        _apply_main_single_defaults,
        _make_data,
        run_from_existing_stage1,
        run_single_proposed_diagnostic,
        run_stage1_only,
    )
    from src.metrics import position_rmse, relative_nmse
    from src.projections_delay import tau_from_pole
    from src.tensor_utils import hankelize_frequency
    from src.experiments.resource_control import (
        apply_thread_limits,
        assert_row_is_light,
        memory_snapshot_mb,
        resolve_hybrid_resources,
        thread_limit_context,
        trim_memory,
    )
    from src.experiments.progress_logger import ProgressLogger
    from src.experiments.cli_common import (
        add_io_args,
        add_mc_args,
        add_progress_args,
        add_resource_args,
        normalize_blas_threads,
    )
    from src.utils import scipy_is_available
else:
    from ..channel_model import channel_components, evs_component_selection
    from ..config import default_config
    from ..diagnostics import estimate_position_from_ris_eta
    from ..geometry import polarization_vector
    from ..estimators import (
        reconstruct_raw_tensor_from_structured_estimate,
        estimate_position_from_local_ris,
        global_exact_spherical_vp_refinement,
    )
    from ..global_vp import (
        _build_global_dictionary,
        _solve_linear_vp_regularized,
        data_only_efim_diagnostic,
    )
    from ..main_single_proposed import (
        _apply_main_single_defaults,
        _make_data,
        run_from_existing_stage1,
        run_single_proposed_diagnostic,
        run_stage1_only,
    )
    from ..metrics import position_rmse, relative_nmse
    from ..projections_delay import tau_from_pole
    from ..tensor_utils import hankelize_frequency
    from .resource_control import (
        apply_thread_limits,
        assert_row_is_light,
        memory_snapshot_mb,
        resolve_hybrid_resources,
        thread_limit_context,
        trim_memory,
    )
    from .progress_logger import ProgressLogger
    from .cli_common import (
        add_io_args,
        add_mc_args,
        add_progress_args,
        add_resource_args,
        normalize_blas_threads,
    )
    from ..utils import scipy_is_available


DEFAULT_SNR_GRID = "-30,-25,-20,-15,-10,-5,0,5,10"
DEFAULT_PAPER_K = 3
DEFAULT_BLAS_THREADS = "auto"
LARGE_ARRAY_ELEMENT_THRESHOLD = 1_000_000
WORKER_ROW_ARRAY_JSON_THRESHOLD = 16
RECEIVER_NOISE_CONVENTION = (
    "AWGN is added only on active EVS observation components with variance set "
    "by active-component signal power."
)
RECEIVER_MODE_CONVENTION = (
    "receiver_mode selects the EVS component observation mask before noise "
    "generation and estimator evaluation."
)
NESTED_RECEIVER_NOISE_CONVENTION = "fixed_full6d_reference_sigma2"
FIGURE6_K_GRID = [1, 2, 3, 4]
FIGURE_ORDER = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]
FIG1_FIG2_SHARED_FIGURE = "fig1_fig2"
FIG1_FIG2_SHARED_TRIAL_CSV = "fig1_fig2_vp_family_trials.csv"
FIG1_FIG2_SHARED_SUMMARY_CSV = "fig1_fig2_vp_family_summary.csv"
FIGURE_PDFS = {
    "fig1": "fig1_vp_family_rmse_vs_snr.pdf",
    "fig2": "fig2_vp_family_nmse_vs_snr.pdf",
    "fig3": "fig3_evs_sensing_rmse_vs_snr.pdf",
    "fig4": "fig4_evs_peb_vs_snr.pdf",
    "fig5": "fig5_ngc_rescue_outlier_trigger_vs_snr.pdf",
    "fig6": "fig6_rmse_vs_K_snr0.pdf",
}
FIGURE_METRICS = {
    "fig1": "position_rmse_m",
    "fig2": "y_nmse",
    "fig3": "position_rmse_m",
    "fig4": "peb_position_m",
    "fig5": "outlier_flag",
    "fig6": "position_rmse_m",
}

VARIANT_LABELS = {
    "direct_vp": "Direct VP (w/o rescue)",
    "old_gated": "GoF-gated rescue",
    "force_rescue": "Always-run rescue",
    "oracle_init_vp": "Oracle init VP",
    "adaptive_jones_vp_proposed": "NGC proposed",
    "adaptive_jones_no_rescue": "Adaptive-Jones VP (w/o rescue)",
    "PEB": "Data-only Free-Jones PEB",
    "proposed_peb": "Data-only Free-Jones PEB",
    "full_6d_evs_peb": "Full-6D Free-Jones PEB",
    "scalar_peb": "Scalar Free-Jones PEB",
    "dual_pol_peb": "Dual-pol Free-Jones PEB",
    "constrained_jones_peb": "Constrained-Jones PEB",
    "full_6d_constrained_jones_peb": "Full-6D Constrained-Jones PEB",
    "adaptive_jones_vp_proposed_force_lower_raw": "Proposed w/ always-run rescue",
    "adaptive_jones_vp_proposed_old_gated": "Proposed w/ GoF-gated rescue",
}


def _proposed_ngc_spec(*, allow_stage2: bool = True) -> dict[str, Any]:
    return {
        "enable_global_vp": True,
        "global_vp": {"mode": "adaptive_jones"},
        "_runner": "main_single_proposed",
        "_allow_stage2": bool(allow_stage2),
        "stage2_adaptive": True,
        "stage2_rescue_type": "ris_only",
        "proposed_stage2_policy": "ngc_certified_ris_only",
        "rescue_accept_min_rel_improvement": 0.0,
        "rescue_accept_min_abs_improvement": 1.0e-8,
    }


def _proposed_force_lower_raw_spec() -> dict[str, Any]:
    spec = _proposed_ngc_spec(allow_stage2=True)
    spec["proposed_stage2_policy"] = "force_ris_only"
    return spec


def _proposed_old_gated_spec() -> dict[str, Any]:
    spec = _proposed_ngc_spec(allow_stage2=True)
    spec["proposed_stage2_policy"] = "reliability_gated_ris_only"
    spec["rescue_accept_min_rel_improvement"] = 1.0e-3
    return spec
RAW_SUMMARY_METRICS = [
    "position_rmse_m",
    "y_nmse",
    "range_rmse_m",
    "tau_rmse_s",
    "raw_objective_final",
    "outlier_flag",
    "peb_position_m",
    "peb_scalar_m",
    "peb_dual_m",
    "peb_evs_m",
    "peb_free_jones_m",
    "peb_constrained_jones_m",
    "peb_anchored_jones_m",
]

PEB_EXTRA_FIELDS = [
    "peb_free_jones_m",
    "peb_constrained_jones_m",
    "peb_anchored_jones_m",
    "peb_variant",
    "jones_bound_type",
    "constrained_jones_peb_m",
    "anchored_jones_peb_m",
    "free_jones_peb_m",
    "peb_fim_rank_chi_free",
    "peb_fim_rank_chi_constrained",
    "peb_fim_rank_chi_anchored",
    "peb_fim_cond_chi_free",
    "peb_fim_cond_chi_constrained",
    "peb_fim_cond_chi_anchored",
    "peb_clock_schur_used",
    "peb_rank_deficient",
    "anchored_prior_scaling",
    "anchored_prior_lambda",
    "anchored_prior_precision_norm",
    "peb_free_projection_schur_relerr",
    "peb_con_minus_free_min_eig",
    "peb_hyb_minus_free_min_eig",
    "peb_ordering_ok",
]

FIELDNAMES = [
    "figure",
    "variant",
    "trial_id",
    "seed",
    "snr_db",
    "x_name",
    "x_value",
    "K",
    "paper_k",
    "effective_K",
    "num_ris_paths",
    "receiver_mode",
    "config_seed",
    "nested_receiver_noise_convention",
    "reference_receiver_mode",
    "reference_sigma2",
    "nested_base_y_noisy_hash",
    "failed",
    "error",
    "runtime_s",
    "position_rmse_m",
    "position_error_m",
    "y_nmse",
    "range_rmse_m",
    "tau_rmse_s",
    "raw_objective_final",
    "outlier_flag",
    "selected_branch",
    "final_refinement_method",
    "final_runner_name",
    "used_main_single_proposed_path",
    "variant_config_hash",
    "global_vp_mode",
    "global_vp_backend",
    "global_vp_gpu_used",
    "global_vp_gpu_device",
    "global_vp_objective_backend",
    "global_vp_linear_solve_backend",
    "jones_mode",
    "adaptive_enabled",
    "adaptive_policy_name",
    "selected_vp_family_branch",
    "lambda_path_min",
    "lambda_path_max",
    "lambda_path_mean",
    "lambda_clipped_fraction",
    "used_stage1_regularizer",
    "linear_nuisance_dim",
    "nonlinear_dim",
    "fixed_pol_score",
    "jones_score",
    "lambda_jones_per_path",
    "snr_eff_per_path",
    "jones_leakage_per_path",
    "reliability_decision",
    "trigger_reasons",
    "gof_stat",
    "gof_dof",
    "gof_pass",
    "data_only_scaled_efim_lambda_min",
    "data_only_scaled_efim_condition_number",
    "stage1_assignment_margin",
    "stage1_selected_clock_std_ns",
    "delta_t_k_ns",
    "stage1_runtime_s",
    "global_vp_runtime_s",
    "total_runtime_s",
    "peb_position_m",
    "peb_scalar_m",
    "peb_dual_m",
    "peb_evs_m",
    "peb_free_jones_m",
    "peb_constrained_jones_m",
    "peb_anchored_jones_m",
    "peb_variant",
    "jones_bound_type",
    "constrained_jones_peb_m",
    "anchored_jones_peb_m",
    "free_jones_peb_m",
    "peb_fim_rank_chi_free",
    "peb_fim_rank_chi_constrained",
    "peb_fim_rank_chi_anchored",
    "peb_fim_cond_chi_free",
    "peb_fim_cond_chi_constrained",
    "peb_fim_cond_chi_anchored",
    "peb_clock_schur_used",
    "peb_rank_deficient",
    "anchored_prior_scaling",
    "anchored_prior_lambda",
    "anchored_prior_precision_norm",
    "peb_free_projection_schur_relerr",
    "peb_con_minus_free_min_eig",
    "peb_hyb_minus_free_min_eig",
    "peb_ordering_ok",
    "efim_unscaled_cache_hit",
    "efim_unscaled_cache_key",
    "efim_sigma2",
    "efim_reuse_mode",
    "peb_backend",
    "rss_mb_before",
    "rss_mb_after",
    "rss_mb_delta",
    "warning",
    "debug_main_position_error_m",
    "debug_main_y_nmse",
    "debug_main_raw_objective",
    "debug_main_global_vp_mode",
    "debug_main_adaptive_enabled",
    "debug_config_diff_summary",
    "direct_candidate_position_error_m",
    "direct_candidate_y_nmse",
    "direct_candidate_raw_objective",
    "rescue_candidate_position_error_m",
    "rescue_candidate_y_nmse",
    "rescue_candidate_raw_objective",
    "rescue_candidate_available",
    "rescue_accept_decision",
    "rescue_reject_reason",
    "direct_candidate_lambda_jones_per_path",
    "direct_candidate_snr_eff_per_path",
    "direct_candidate_jones_leakage_per_path",
    "rescue_candidate_lambda_jones_per_path",
    "rescue_candidate_snr_eff_per_path",
    "rescue_candidate_jones_leakage_per_path",
    "direct_candidate_data_only_scaled_efim_lambda_min",
    "direct_candidate_data_only_scaled_efim_condition_number",
    "rescue_candidate_data_only_scaled_efim_lambda_min",
    "rescue_candidate_data_only_scaled_efim_condition_number",
    "legacy_stage1_decision",
    "gof_reliability_decision",
    "stage1_geometry_trigger",
    "stage1_geometry_trigger_reasons",
    "ngc_policy_active",
    "ngc_lambda_ris",
    "ngc_direct_clock_score",
    "ngc_direct_clock_score_norm",
    "ngc_direct_clock_dof",
    "ngc_direct_clock_sigma_source",
    "ngc_direct_clock_std_ns",
    "ngc_direct_ris_score",
    "ngc_direct_ris_score_norm",
    "ngc_direct_ris_available",
    "ngc_direct_total_score",
    "ngc_direct_cert_status",
    "ngc_direct_cert_reason",
    "ngc_rescue_requested",
    "ngc_rescue_request_reason",
    "ngc_rescue_clock_score",
    "ngc_rescue_clock_score_norm",
    "ngc_rescue_clock_dof",
    "ngc_rescue_clock_sigma_source",
    "ngc_rescue_clock_std_ns",
    "ngc_rescue_ris_score",
    "ngc_rescue_ris_score_norm",
    "ngc_rescue_ris_available",
    "ngc_rescue_total_score",
    "ngc_rescue_cert_status",
    "ngc_rescue_cert_reason",
    "ngc_selected_by",
    "ngc_final_unreliable",
    "ngc_threshold_clock_green",
    "ngc_threshold_clock_red",
    "proposed_stage2_policy",
]

_PEB_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_WORKER_BASE_CONFIG: dict[str, Any] | None = None
_WORKER_OUT_DIR: pathlib.Path | None = None
_WORKER_BLAS_THREADS = 1
_WORKER_RESPECT_EXISTING_BLAS_ENV = False
_WORKER_TRIM_MEMORY = True
_PEB_CACHE_DISABLED_PATHS: set[str] = set()

PEB_CACHE_TIMEOUT_S = 60.0
PEB_CACHE_BUSY_TIMEOUT_MS = 60_000
PEB_CACHE_LOCK_RETRIES = 5
PEB_CACHE_RETRY_BASE_S = 0.05


class GroupedResultReuseError(RuntimeError):
    """Raised when grouped variants accidentally share one final result object."""


def _is_fig1_fig2(figure: str) -> bool:
    return figure in {"fig1", "fig2", FIG1_FIG2_SHARED_FIGURE}


def parse_snr_grid(value: str | Iterable[float] = DEFAULT_SNR_GRID) -> list[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def parse_k_grid(value: str | Iterable[int] = "1,2,3,4") -> list[int]:
    if isinstance(value, str):
        grid = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        grid = [int(item) for item in value]
    if not grid:
        raise ValueError("--k-grid must contain at least one positive integer")
    if any(k <= 0 for k in grid):
        raise ValueError("--k-grid entries must be positive")
    return grid


def _apply_blas_thread_env(
    blas_threads: int,
    *,
    respect_existing_blas_env: bool = False,
) -> None:
    apply_thread_limits(
        int(blas_threads),
        respect_existing=bool(respect_existing_blas_env),
    )


def parse_figures(value: str) -> list[str]:
    if value == "all":
        return list(FIGURE_ORDER)
    figures = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in figures if item not in FIGURE_ORDER]
    if unknown:
        raise ValueError(f"unknown figure ids: {unknown}")
    requested = set(figures)
    return [figure for figure in FIGURE_ORDER if figure in requested]


def parse_variant_filter(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    if not variants:
        raise ValueError("--variant-filter must contain at least one variant name")
    return variants


def apply_nested_update(config: dict, update_dict: dict) -> dict:
    result = copy.deepcopy(config)
    for key, value in update_dict.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = apply_nested_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def global_vp_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Return global-VP backend overrides requested by the ablation CLI."""
    if getattr(args, "global_vp_backend", None) is None:
        return {}
    overrides: dict[str, Any] = {
        "backend": str(args.global_vp_backend),
        "gpu_device": int(getattr(args, "global_vp_gpu_device", 0)),
        "validate_gpu_against_cpu": bool(
            getattr(args, "global_vp_validate_gpu_against_cpu", False)
        ),
        "gpu_dtype": str(getattr(args, "global_vp_gpu_dtype", "complex128")),
    }
    if bool(getattr(args, "global_vp_gpu_keep_arrays_on_device", False)):
        overrides["gpu_keep_arrays_on_device"] = True
    return overrides


def apply_global_vp_cli_overrides(
    config: dict,
    args: argparse.Namespace,
) -> dict:
    """Apply global-VP backend CLI overrides in-place and return config."""
    overrides = global_vp_cli_overrides(args)
    if not overrides:
        return config
    global_vp = config.setdefault("global_vp", {})
    global_vp.update(overrides)
    return config


def apply_peb_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    crb = config.setdefault("crb", {})
    crb["include_constrained_jones_peb"] = bool(
        getattr(args, "include_constrained_jones_peb", True)
    )
    crb["include_anchored_jones_peb"] = bool(
        getattr(args, "include_anchored_jones_peb", False)
    )
    crb["jones_anchor_prior_mode"] = str(
        getattr(args, "jones_anchor_prior_mode", "disabled")
    )
    crb["jones_anchor_prior_scale"] = float(
        getattr(args, "jones_anchor_prior_scale", 1.0)
    )
    return config


def set_number_of_ris_paths(config: dict, k_paths: int) -> dict:
    """Set the physical path/RIS count used by scene generation and estimators."""
    k_paths = int(k_paths)
    if k_paths <= 0:
        raise ValueError("K must be positive")
    config["K"] = k_paths
    ris_centers = np.asarray(config.get("ris_centers"), dtype=float)
    if ris_centers.ndim != 2 or ris_centers.shape[1] != 3:
        raise ValueError("config['ris_centers'] must have shape (num_ris, 3)")
    if ris_centers.shape[0] < k_paths:
        extra = []
        base_z = float(np.mean(ris_centers[:, 2])) if ris_centers.size else 1.2
        while ris_centers.shape[0] + len(extra) < k_paths:
            idx = ris_centers.shape[0] + len(extra)
            side = -1.0 if idx % 2 else 1.0
            y_offset = side * (2.8 + 0.35 * idx)
            x_offset = 4.6 + 0.35 * idx
            z_offset = base_z + 0.05 * ((idx % 3) - 1)
            extra.append([x_offset, y_offset, z_offset])
        ris_centers = np.vstack([ris_centers, np.asarray(extra, dtype=float)])
    config["ris_centers"] = ris_centers
    config["jnpp_max_candidates"] = max(int(config.get("jnpp_max_candidates", 1)), 1 + k_paths)
    config["jnpp_top_assignments"] = min(3, math.factorial(k_paths))
    return config


def make_base_config(seed: int, snr_db: float, overrides: dict | None = None) -> dict:
    config = default_config()
    config["seed"] = int(seed)
    config["SNR_dB"] = float(snr_db)
    config["print_progress"] = False
    config["verbose_stage2"] = False
    config["run_full_legacy_comparison"] = False
    if overrides:
        config = apply_nested_update(config, overrides)
    set_number_of_ris_paths(config, int(config.get("K", 1)))
    return config


def make_nested_receiver_mode_data(
    base_full6d_data: dict,
    receiver_mode: str,
    config: dict,
) -> dict:
    """Mask one shared full-6D realization without regenerating its noise."""
    mode = str(receiver_mode)
    if str(base_full6d_data["scene"].get("receiver_mode")) != "full_6d":
        raise ValueError("nested receiver data requires a full_6d base realization")
    scene = copy.deepcopy(base_full6d_data["scene"])
    component_mask = evs_component_selection(mode)
    observation_mask = np.tile(component_mask, int(scene["M_A"])).astype(bool)
    tensor_mask = observation_mask[:, None, None]
    scene["receiver_mode"] = mode
    scene["evs_component_mask"] = component_mask
    scene["evs_observation_mask"] = observation_mask

    data = copy.deepcopy(base_full6d_data)
    data["scene"] = scene
    data["Y_true"] = np.asarray(base_full6d_data["Y_true"]) * tensor_mask
    data["Y_noisy"] = np.asarray(base_full6d_data["Y_noisy"]) * tensor_mask
    data["Z_true"] = hankelize_frequency(data["Y_true"], scene["P"])
    data["Z_noisy"] = hankelize_frequency(data["Y_noisy"], scene["P"])
    true_components = copy.deepcopy(base_full6d_data["true_components"])
    if "a_EVS" in true_components:
        true_components["a_EVS"] = (
            np.asarray(true_components["a_EVS"]) * observation_mask[None, :]
        )
    data["true_components"] = true_components
    data["noise_variance"] = float(base_full6d_data["noise_variance"])
    data["nested_receiver_noise_convention"] = NESTED_RECEIVER_NOISE_CONVENTION
    data["reference_receiver_mode"] = "full_6d"
    data["receiver_mode"] = mode
    data["reference_sigma2"] = float(base_full6d_data["noise_variance"])
    data["nested_base_y_noisy_hash"] = _hash_array(
        base_full6d_data["Y_noisy"]
    )
    data["config_receiver_mode"] = str(config.get("receiver_mode", mode))
    return data


def _variant_specs(figure: str) -> dict[str, dict[str, Any]]:
    if _is_fig1_fig2(figure):
        return {
            "stage1_only": {
                "enable_global_vp": False,
                "stage2_adaptive": False,
                "proposed_stage2_policy": "reliability_gated",
                "_runner": "stage1_only",
                "_allow_stage2": False,
            },
            "fixed_pol_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "fixed_pol"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "free_jones_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_free"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "regularized_jones_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_regularized"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "adaptive_jones_vp_proposed": {
                **_proposed_ngc_spec(allow_stage2=True),
            },
        }
    if figure == "fig3":
        return {
            "scalar_receiver": {"receiver_mode": "scalar", "global_vp": {"mode": "adaptive_jones"}},
            "dual_pol_receiver": {"receiver_mode": "dual_pol", "global_vp": {"mode": "adaptive_jones"}},
            "full_6d_evs": {"receiver_mode": "full_6d", "global_vp": {"mode": "adaptive_jones"}},
        }
    if figure == "fig4":
        return {
            "scalar_peb": {"receiver_mode": "scalar", "_runner": "peb_only"},
            "dual_pol_peb": {"receiver_mode": "dual_pol", "_runner": "peb_only"},
            "full_6d_evs_peb": {"receiver_mode": "full_6d", "_runner": "peb_only"},
            "full_6d_constrained_jones_peb": {
                "receiver_mode": "full_6d",
                "_runner": "peb_only",
            },
        }
    if figure == "fig5":
        return {
            "direct_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "adaptive_jones"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "old_gated": {
                **_proposed_old_gated_spec(),
            },
            "adaptive_jones_vp_proposed": {
                **_proposed_ngc_spec(allow_stage2=True),
            },
            "force_rescue": {
                **_proposed_force_lower_raw_spec(),
            },
            "oracle_init_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "adaptive_jones"},
                "_runner": "oracle_init_vp",
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
        }
    if figure == "fig6":
        return {
            "fixed_pol_vp": {
                "global_vp": {"mode": "fixed_pol"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "free_jones_vp": {
                "global_vp": {"mode": "jones_free"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "adaptive_jones_no_rescue": {
                "global_vp": {"mode": "adaptive_jones"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "adaptive_jones_vp_proposed": {
                **_proposed_ngc_spec(allow_stage2=True),
            },
            "proposed_peb": {"global_vp": {"mode": "adaptive_jones"}, "_runner": "peb_only"},
            "constrained_jones_peb": {
                "global_vp": {"mode": "adaptive_jones"},
                "_runner": "peb_only",
            },
        }
    raise ValueError(f"unknown figure {figure!r}")


def _diagnostic_variant_specs(figure: str) -> dict[str, dict[str, Any]]:
    if _is_fig1_fig2(figure):
        return {
            "free_jones_vp_gated_rescue": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_free"},
                "_allow_stage2": True,
                "stage2_adaptive": True,
                "stage2_rescue_type": "ris_only",
            },
            "free_jones_vp_geometry_gated_rescue": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_free"},
                "_allow_stage2": True,
                "stage2_adaptive": True,
                "stage2_rescue_type": "ris_only",
                "proposed_stage2_policy": "geometry_gated_ris_only",
            },
            "free_jones_vp_force_rescue": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_free"},
                "_allow_stage2": True,
                "stage2_adaptive": True,
                "stage2_rescue_type": "ris_only",
                "proposed_stage2_policy": "force_ris_only",
            },
            "free_jones_vp_force_rescue_lower_raw": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_free"},
                "_allow_stage2": True,
                "stage2_adaptive": True,
                "stage2_rescue_type": "ris_only",
                "proposed_stage2_policy": "force_ris_only",
                "rescue_accept_min_rel_improvement": 0.0,
            },
            "free_jones_vp_geometry_gated_lower_raw": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_free"},
                "_allow_stage2": True,
                "stage2_adaptive": True,
                "stage2_rescue_type": "ris_only",
                "proposed_stage2_policy": "geometry_gated_ris_only",
                "rescue_accept_min_rel_improvement": 0.0,
            },
            "adaptive_jones_vp_proposed_old_gated": _proposed_old_gated_spec(),
            "adaptive_jones_vp_proposed_force_lower_raw": _proposed_force_lower_raw_spec(),
        }
    return {}


def _extra_peb_specs(figure: str) -> dict[str, dict[str, Any]]:
    if _is_fig1_fig2(figure):
        return {
            "PEB": {"receiver_mode": "full_6d", "_runner": "peb_only"},
            "constrained_jones_peb": {
                "receiver_mode": "full_6d",
                "_runner": "peb_only",
            },
        }
    if figure == "fig3":
        return {
            "full_6d_evs_peb": {
                "receiver_mode": "full_6d",
                "_runner": "peb_only",
            },
            "full_6d_constrained_jones_peb": {
                "receiver_mode": "full_6d",
                "_runner": "peb_only",
            },
        }
    return {}


def _variant_filter_aliases(
    variant: str,
    spec: dict[str, Any],
) -> set[str]:
    aliases = {str(variant).lower()}
    runner = str(spec.get("_runner", "")).lower()
    if runner:
        aliases.add(runner)
    if variant == "PEB" and runner == "peb_only":
        aliases.update({"peb", "peb_only", "data_only_peb"})
    if "constrained_jones_peb" in str(variant).lower() and runner == "peb_only":
        aliases.update({"peb", "constrained_peb", "constrained_jones_peb"})
    return aliases


def _filter_variants(
    figure: str,
    variants: dict[str, dict[str, Any]],
    variant_filter: tuple[str, ...] | None,
) -> dict[str, dict[str, Any]]:
    if variant_filter is None or not _is_fig1_fig2(figure):
        return variants
    requested = {name.lower() for name in variant_filter}
    diagnostic_requested = requested & {
        name.lower() for name in _diagnostic_variant_specs(figure)
    }
    if diagnostic_requested and not any(
        name in variants for name in _diagnostic_variant_specs(figure)
    ):
        requested_text = ", ".join(sorted(diagnostic_requested))
        raise ValueError(
            "--variant-filter requested diagnostic Fig.1/Fig.2 variant(s) "
            f"{requested_text}; add --include-diagnostic-variants to enable them"
        )
    filtered = {
        name: spec
        for name, spec in variants.items()
        if requested & _variant_filter_aliases(name, spec)
    }
    if not filtered:
        available = ", ".join(variants)
        raise ValueError(
            "--variant-filter selected no Fig.1/Fig.2 variants; "
            f"available variants: {available} (PEB aliases: PEB, peb_only, data_only_peb)"
        )
    return filtered


def _variants_for_figure(
    figure: str,
    variant_filter: tuple[str, ...] | None = None,
    *,
    include_diagnostic_variants: bool = False,
) -> dict[str, dict[str, Any]]:
    canonical = FIG1_FIG2_SHARED_FIGURE if _is_fig1_fig2(figure) else figure
    variants = {
        **_variant_specs(canonical),
        **(
            _diagnostic_variant_specs(canonical)
            if include_diagnostic_variants
            else {}
        ),
        **_extra_peb_specs(canonical),
    }
    return _filter_variants(canonical, variants, variant_filter)


def _apply_peb_cli_variant_filter(
    variants: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if bool(getattr(args, "include_constrained_jones_peb", True)):
        return variants
    return {
        name: spec
        for name, spec in variants.items()
        if "constrained_jones_peb" not in str(name).lower()
    }


def _trial_seed(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _path_get(container: Any, path: tuple[Any, ...], default: Any) -> Any:
    current = container
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        elif isinstance(current, (list, tuple)) and isinstance(key, int):
            if key >= len(current):
                return default
            current = current[key]
        else:
            return default
    return current


def get_nested(result: dict, possible_paths: Iterable[str | tuple[Any, ...]], default: Any = np.nan) -> Any:
    for path in possible_paths:
        parts = tuple(path.split(".")) if isinstance(path, str) else tuple(path)
        value = _path_get(result, parts, default)
        if value is not default:
            return value
    return default


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


def _vector_string(value: Any) -> str:
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    if np.iscomplexobj(arr):
        payload = [[float(np.real(item)), float(np.imag(item))] for item in arr.reshape(-1)]
    else:
        payload = [float(item) for item in np.asarray(arr, dtype=float).reshape(-1)]
    return json.dumps(payload, separators=(",", ":"))


def _list_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    return str(value)


def _empty_row() -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    numeric_fields = [
        "runtime_s",
        "position_rmse_m",
        "position_error_m",
        "y_nmse",
        "range_rmse_m",
        "tau_rmse_s",
        "raw_objective_final",
        "lambda_path_min",
        "lambda_path_max",
        "lambda_path_mean",
        "lambda_clipped_fraction",
        "gof_stat",
        "data_only_scaled_efim_lambda_min",
        "data_only_scaled_efim_condition_number",
        "stage1_assignment_margin",
        "stage1_selected_clock_std_ns",
        "stage1_runtime_s",
        "global_vp_runtime_s",
        "total_runtime_s",
        "peb_position_m",
        "peb_scalar_m",
        "peb_dual_m",
        "peb_evs_m",
        "peb_free_jones_m",
        "peb_constrained_jones_m",
        "peb_anchored_jones_m",
        "constrained_jones_peb_m",
        "anchored_jones_peb_m",
        "free_jones_peb_m",
        "peb_fim_rank_chi_free",
        "peb_fim_rank_chi_constrained",
        "peb_fim_rank_chi_anchored",
        "peb_fim_cond_chi_free",
        "peb_fim_cond_chi_constrained",
        "peb_fim_cond_chi_anchored",
        "anchored_prior_lambda",
        "anchored_prior_precision_norm",
        "peb_free_projection_schur_relerr",
        "peb_con_minus_free_min_eig",
        "peb_hyb_minus_free_min_eig",
        "rss_mb_before",
        "rss_mb_after",
        "rss_mb_delta",
        "direct_candidate_data_only_scaled_efim_lambda_min",
        "direct_candidate_data_only_scaled_efim_condition_number",
        "rescue_candidate_data_only_scaled_efim_lambda_min",
        "rescue_candidate_data_only_scaled_efim_condition_number",
    ]
    for field in numeric_fields:
        row[field] = float("nan")
    return row


def _num_ris_paths(config: dict[str, Any]) -> int | str:
    ris_centers = np.asarray(config.get("ris_centers", []))
    if ris_centers.ndim == 2:
        return int(ris_centers.shape[0])
    return ""


def _assert_effective_k(
    *,
    figure: str,
    effective_k: int,
    paper_k: int,
    x_value: float,
) -> None:
    expected_k = int(x_value) if figure == "fig6" else int(paper_k)
    if int(effective_k) != expected_k:
        expectation = "x_value" if figure == "fig6" else "paper_k"
        raise AssertionError(
            f"{figure}: effective_K={effective_k} must equal "
            f"{expectation}={expected_k}"
        )


def _set_row_k_metadata(
    row: dict[str, Any],
    *,
    figure: str,
    effective_k: int,
    paper_k: int,
    x_value: float,
    receiver_mode: str,
    config_seed: int,
    num_ris_paths: int | str,
) -> None:
    _assert_effective_k(
        figure=figure,
        effective_k=int(effective_k),
        paper_k=int(paper_k),
        x_value=float(x_value),
    )
    row.update(
        {
            "K": int(effective_k),
            "paper_k": int(paper_k),
            "effective_K": int(effective_k),
            "num_ris_paths": num_ris_paths,
            "receiver_mode": str(receiver_mode),
            "config_seed": int(config_seed),
        }
    )


def _rss_mb() -> float:
    return memory_snapshot_mb()


def compact_experiment_result(
    result: Any,
    keep_large_arrays: bool = False,
    *,
    large_array_threshold: int = LARGE_ARRAY_ELEMENT_THRESHOLD,
) -> Any:
    """Drop large tensors from a diagnostic result after metrics are extracted."""
    if keep_large_arrays:
        return result
    large_names = {
        "Y_true",
        "Y_noisy",
        "Y_hat",
        "Z_true",
        "Z_noisy",
        "Z_hat",
        "tensor",
        "hankel",
        "raw_tensor",
        "raw_tensors",
    }

    def _compact(value: Any, key: str | None = None) -> Any:
        if isinstance(value, np.ndarray):
            key_lower = "" if key is None else key.lower()
            if key in large_names or key_lower in large_names or value.size > large_array_threshold:
                return None
            return value
        if isinstance(value, dict):
            for child_key in list(value.keys()):
                value[child_key] = _compact(value[child_key], str(child_key))
            return value
        if isinstance(value, list):
            for idx, item in enumerate(value):
                value[idx] = _compact(item)
            return value
        if isinstance(value, tuple):
            return tuple(_compact(item) for item in value)
        return value

    return _compact(result)


def _truth_init_estimate(scene: dict, true_components: dict) -> dict:
    k_paths = int(scene["K"])
    return {
        "A": true_components["a_EVS"].T.copy(),
        "D": true_components["d"].T.copy(),
        "C": true_components["c"].T.copy(),
        "poles": true_components["poles"].copy(),
        "ris_eta": np.column_stack(
            [
                true_components["ranges"],
                true_components["elevations"],
                true_components["azimuths"],
            ]
        ),
        "gamma": scene["gamma_true"].copy(),
        "eta_pol": scene["eta_true"].copy(),
        "assignment": list(range(k_paths)),
        "panel_to_column_assignment": list(range(k_paths)),
        "columns_are_panel_ordered": True,
        "Z_hat": np.zeros((scene["I"], scene["P"], scene["L"], scene["T"]), dtype=complex),
        "beta_z": np.ones(k_paths, dtype=complex),
    }


def _hash_array(value: Any) -> str:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(float), decimals=12)
    payload = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha256(payload).hexdigest()[:16]


def peb_cache_key(config: dict) -> tuple[Any, ...]:
    """Return a deterministic key for data-only EFIM PEB computations."""
    return (
        float(config.get("SNR_dB")),
        int(config.get("K")),
        str(config.get("receiver_mode", config.get("evs_selection", "full_6d"))),
        int(config.get("seed")),
        float(config.get("fc")),
        tuple(int(item) for item in config.get("ris_shape", ())),
        int(config.get("M_A")),
        int(config.get("N")),
        int(config.get("P")),
        int(config.get("T")),
        _hash_array(config.get("p_B")),
        _hash_array(config.get("p_u_true")),
        _hash_array(config.get("ris_centers")),
        float(config.get("delta_t_true")),
        _hash_config_for_peb(config),
    )


def _hash_config_for_peb(config: dict) -> str:
    payload = {
        "global_vp": config.get("global_vp", {}),
        "crb": config.get("crb", {}),
        "eps": config.get("eps", None),
        "delta_f": config.get("delta_f", None),
        "wavelength": config.get("wavelength", None),
        "nested_receiver_noise_convention": config.get(
            "nested_receiver_noise_convention", None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _variant_config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            default=lambda value: np.asarray(value).tolist()
            if isinstance(value, np.ndarray)
            else str(value),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]


def _annotate_variant_result(
    result: dict,
    config: dict,
    *,
    final_runner_name: str,
    used_main_single_proposed_path: bool,
) -> dict:
    final = result.get("final", {})
    global_vp = dict(config.get("global_vp", {}))
    mode = str(final.get("global_vp_mode", final.get("vp_mode", global_vp.get("mode", ""))))
    selected_family = str(final.get("selected_vp_family_branch", ""))
    lambdas = np.asarray(final.get("lambda_jones_per_path", []), dtype=float).reshape(-1)
    finite_lambdas = lambdas[np.isfinite(lambdas)]
    lambda_min_bound = float(global_vp.get("jones_lambda_min", 1.0e-4))
    lambda_max_bound = float(global_vp.get("jones_lambda_max", 1.0e8))
    clipped_fraction = (
        float(
            np.mean(
                np.isclose(finite_lambdas, lambda_min_bound)
                | np.isclose(finite_lambdas, lambda_max_bound)
            )
        )
        if finite_lambdas.size
        else float("nan")
    )
    result["variant_diagnostics"] = {
        "variant_config_hash": _variant_config_hash(config),
        "final_runner_name": str(final_runner_name),
        "used_main_single_proposed_path": bool(used_main_single_proposed_path),
        "jones_mode": selected_family or mode,
        "adaptive_enabled": mode == "adaptive_jones",
        "adaptive_policy_name": (
            "fixed_anchor_then_adaptive_jones_score_selection"
            if mode == "adaptive_jones"
            else ""
        ),
        "lambda_path_min": float(np.min(finite_lambdas))
        if finite_lambdas.size
        else float("nan"),
        "lambda_path_max": float(np.max(finite_lambdas))
        if finite_lambdas.size
        else float("nan"),
        "lambda_path_mean": float(np.mean(finite_lambdas))
        if finite_lambdas.size
        else float("nan"),
        "lambda_clipped_fraction": clipped_fraction,
        "used_stage1_regularizer": bool(
            float(config.get("stage1_factor_reg", 0.0)) > 0.0
            or float(config.get("stage1_factor_reg_rel", 0.0)) > 0.0
        ),
    }
    return result


def _json_safe_float(value: Any) -> float | None:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if np.isfinite(value_float) else None


def _restore_cached_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _peb_cache_path(out_dir: pathlib.Path | None) -> pathlib.Path | None:
    if out_dir is None:
        return None
    return pathlib.Path(out_dir) / ".cache" / "peb_cache.sqlite"


def _peb_cache_path_key(db_path: pathlib.Path) -> str:
    return str(pathlib.Path(db_path).resolve())


def _peb_cache_is_enabled(db_path: pathlib.Path) -> bool:
    return _peb_cache_path_key(db_path) not in _PEB_CACHE_DISABLED_PATHS


def _disable_peb_cache(db_path: pathlib.Path, reason: BaseException | str) -> None:
    path_key = _peb_cache_path_key(db_path)
    if path_key in _PEB_CACHE_DISABLED_PATHS:
        return
    _PEB_CACHE_DISABLED_PATHS.add(path_key)
    print(
        f"WARNING: disabling persistent PEB cache at {db_path}: {reason}",
        file=sys.stderr,
    )


def _configure_peb_cache_connection(conn: sqlite3.Connection) -> None:
    conn.execute(f"PRAGMA busy_timeout={PEB_CACHE_BUSY_TIMEOUT_MS}")


def _is_sqlite_locked_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()


def _init_peb_cache(db_path: pathlib.Path) -> None:
    """Initialize the cache schema and WAL mode from the parent process."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(PEB_CACHE_LOCK_RETRIES):
        try:
            with sqlite3.connect(db_path, timeout=PEB_CACHE_TIMEOUT_S) as conn:
                _configure_peb_cache_connection(conn)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS peb_cache (
                        cache_key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            _PEB_CACHE_DISABLED_PATHS.discard(_peb_cache_path_key(db_path))
            return
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            last_error = exc
            if attempt + 1 < PEB_CACHE_LOCK_RETRIES:
                time.sleep(PEB_CACHE_RETRY_BASE_S * (2**attempt))
    raise RuntimeError(
        f"PEB cache initialization remained locked after "
        f"{PEB_CACHE_LOCK_RETRIES} attempts: {db_path}"
    ) from last_error


def _prepare_peb_cache(out_dir: pathlib.Path | None) -> bool:
    """Initialize persistent cache once in the parent, or disable it safely."""
    db_path = _peb_cache_path(out_dir)
    if db_path is None:
        return False
    try:
        _init_peb_cache(db_path)
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        _disable_peb_cache(db_path, exc)
        return False
    return True


def _open_peb_cache(db_path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=PEB_CACHE_TIMEOUT_S)
    try:
        _configure_peb_cache_connection(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _peb_cache_key_string(config: dict) -> str:
    return json.dumps(peb_cache_key(config), sort_keys=True, default=str, separators=(",", ":"))


def _read_persistent_peb_cache(config: dict, out_dir: pathlib.Path | None) -> dict[str, Any] | None:
    db_path = _peb_cache_path(out_dir)
    if (
        db_path is None
        or not db_path.exists()
        or not _peb_cache_is_enabled(db_path)
    ):
        return None
    cache_key = _peb_cache_key_string(config)
    try:
        with _open_peb_cache(db_path) as conn:
            row = conn.execute(
                "SELECT value_json FROM peb_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    payload = json.loads(row[0])
    result = {
        "peb_position_m": _restore_cached_float(payload.get("peb_position_m")),
        "peb_scalar_m": _restore_cached_float(payload.get("peb_scalar_m")),
        "peb_dual_m": _restore_cached_float(payload.get("peb_dual_m")),
        "peb_evs_m": _restore_cached_float(payload.get("peb_evs_m")),
        "warning": payload.get("warning", ""),
        "peb_is_data_only": bool(payload.get("peb_is_data_only", True)),
        "peb_uses_regularization": bool(
            payload.get("peb_uses_regularization", False)
        ),
        "nuisance_model": str(payload.get("nuisance_model", "jones_linear")),
        "clock_eliminated": bool(payload.get("clock_eliminated", True)),
        "efim_condition_number": _restore_cached_float(
            payload.get("efim_condition_number")
        ),
        "efim_parameter_order": payload.get(
            "efim_parameter_order",
            ["p_x_m", "p_y_m", "p_z_m", "c_delta_t_m"],
        ),
        "peb_reference_type": str(
            payload.get("peb_reference_type", "matched_model")
        ),
        "efim_unscaled_cache_hit": bool(
            payload.get("efim_unscaled_cache_hit", False)
        ),
        "efim_unscaled_cache_key": str(
            payload.get("efim_unscaled_cache_key", "")
        ),
        "efim_sigma2": _restore_cached_float(payload.get("efim_sigma2")),
        "efim_reuse_mode": str(payload.get("efim_reuse_mode", "")),
        "peb_backend": str(payload.get("peb_backend", "cpu")),
    }
    for field in PEB_EXTRA_FIELDS:
        value = payload.get(field)
        if field in {
            "peb_variant",
            "jones_bound_type",
            "anchored_prior_scaling",
            "peb_ordering_ok",
        }:
            result[field] = "" if value is None else value
        elif field in {"peb_clock_schur_used", "peb_rank_deficient"}:
            result[field] = bool(value) if value is not None else False
        else:
            result[field] = _restore_cached_float(value)
    return result


def _write_persistent_peb_cache(
    config: dict,
    out_dir: pathlib.Path | None,
    value: dict[str, Any],
) -> None:
    db_path = _peb_cache_path(out_dir)
    if db_path is None or not _peb_cache_is_enabled(db_path):
        return
    if not db_path.exists() and not _prepare_peb_cache(out_dir):
        return
    payload = {
        "peb_position_m": _json_safe_float(value.get("peb_position_m")),
        "peb_scalar_m": _json_safe_float(value.get("peb_scalar_m")),
        "peb_dual_m": _json_safe_float(value.get("peb_dual_m")),
        "peb_evs_m": _json_safe_float(value.get("peb_evs_m")),
        "warning": str(value.get("warning", "")),
        "peb_is_data_only": bool(value.get("peb_is_data_only", True)),
        "peb_uses_regularization": bool(value.get("peb_uses_regularization", False)),
        "nuisance_model": str(value.get("nuisance_model", "jones_linear")),
        "clock_eliminated": bool(value.get("clock_eliminated", True)),
        "efim_condition_number": _json_safe_float(
            value.get("efim_condition_number")
        ),
        "efim_parameter_order": value.get("efim_parameter_order", []),
        "peb_reference_type": str(
            value.get("peb_reference_type", "matched_model")
        ),
        "efim_unscaled_cache_hit": bool(
            value.get("efim_unscaled_cache_hit", False)
        ),
        "efim_unscaled_cache_key": str(
            value.get("efim_unscaled_cache_key", "")
        ),
        "efim_sigma2": _json_safe_float(value.get("efim_sigma2")),
        "efim_reuse_mode": str(value.get("efim_reuse_mode", "")),
        "peb_backend": str(value.get("peb_backend", "cpu")),
    }
    for field in PEB_EXTRA_FIELDS:
        field_value = value.get(field)
        if isinstance(field_value, (bool, np.bool_)):
            payload[field] = bool(field_value)
        elif isinstance(field_value, str):
            payload[field] = field_value
        else:
            payload[field] = _json_safe_float(field_value)
    cache_key = _peb_cache_key_string(config)
    value_json = json.dumps(payload, sort_keys=True)
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(PEB_CACHE_LOCK_RETRIES):
        try:
            with _open_peb_cache(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO peb_cache(cache_key, value_json) VALUES (?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET value_json=excluded.value_json
                    """,
                    (cache_key, value_json),
                )
                conn.commit()
            return
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                _disable_peb_cache(db_path, exc)
                return
            last_error = exc
            if attempt + 1 < PEB_CACHE_LOCK_RETRIES:
                time.sleep(PEB_CACHE_RETRY_BASE_S * (2**attempt))
    _disable_peb_cache(
        db_path,
        RuntimeError(
            f"cache write remained locked after {PEB_CACHE_LOCK_RETRIES} attempts"
        ),
    )


def position_peb_from_global_efim(
    efim: np.ndarray,
    parameter_order: Iterable[str],
    already_clock_eliminated: bool = False,
    *,
    condition_threshold: float = 1.0e12,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """Return position PEB after explicit clock Schur elimination.

    A 4x4 global EFIM must describe position followed by clock (either seconds
    or range-equivalent c*Delta_t). A 3x3 EFIM is accepted only when the caller
    explicitly states that clock has already been eliminated.
    """
    matrix = np.asarray(efim, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("global EFIM must be a square matrix")
    order = [str(item) for item in parameter_order]
    if len(order) != matrix.shape[0]:
        raise ValueError("parameter_order length must match EFIM dimension")
    matrix = (matrix + matrix.T) * 0.5
    warning = ""
    clock_singular = False

    if matrix.shape == (3, 3):
        if not already_clock_eliminated:
            raise ValueError(
                "3x3 EFIM requires already_clock_eliminated=True"
            )
        position_efim = matrix
    elif matrix.shape == (4, 4):
        if already_clock_eliminated:
            raise ValueError(
                "4x4 EFIM cannot be marked already clock-eliminated"
            )
        normalized = [
            item.lower().replace(" ", "").replace("-", "_") for item in order
        ]
        if not all(
            token in normalized[index]
            for index, token in enumerate(("p_x", "p_y", "p_z"))
        ):
            raise ValueError(
                "4x4 EFIM parameter_order must start with p_x, p_y, p_z"
            )
        clock_name = normalized[3]
        if not any(
            token in clock_name
            for token in ("clock", "delta_t", "deltat", "c_delta_t", "cdeltat")
        ):
            raise ValueError(
                "fourth EFIM parameter must be clock or cDelta_t"
            )
        j_pp = matrix[:3, :3]
        j_pc = matrix[:3, 3:4]
        j_cc = matrix[3:4, 3:4]
        if np.linalg.matrix_rank(j_cc) < 1:
            warning = "data_only_efim_clock_schur_singular"
            clock_singular = True
        position_efim = j_pp - j_pc @ np.linalg.pinv(j_cc) @ j_pc.T
    else:
        raise ValueError(
            "global EFIM must be 4x4, or 3x3 when clock is already eliminated"
        )

    position_efim = (position_efim + position_efim.T) * 0.5
    singular_values = np.linalg.svd(position_efim, compute_uv=False)
    tolerance = (
        max(position_efim.shape)
        * np.finfo(float).eps
        * singular_values[0]
        if singular_values.size
        else float("inf")
    )
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    if (
        clock_singular
        or rank < 3
        or not np.isfinite(condition)
        or condition > float(condition_threshold)
    ):
        peb = float("nan")
        warning = warning or "data_only_efim_position_singular_or_ill_conditioned"
    else:
        covariance = np.linalg.pinv(position_efim)
        trace_value = float(np.trace(covariance).real)
        peb = (
            float(np.sqrt(max(trace_value, 0.0)))
            if np.isfinite(trace_value)
            else float("nan")
        )
        if not np.isfinite(peb):
            warning = warning or "data_only_efim_peb_nonfinite"
    diagnostics = {
        "clock_eliminated": True,
        "efim_condition_number": condition,
        "warning": warning,
        "position_efim": position_efim,
        "efim_parameter_order": order,
    }
    if return_diagnostics:
        return peb, diagnostics
    return peb


def _symmetrize_fim(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    return 0.5 * (arr + arr.T)


def _fim_rank_condition(matrix: np.ndarray, *, rcond: float = 1.0e-10) -> tuple[int, float]:
    arr = _symmetrize_fim(matrix)
    singular = np.linalg.svd(arr, compute_uv=False)
    if not singular.size:
        return 0, float("inf")
    tol = float(rcond) * max(float(singular[0]), 1.0)
    rank = int(np.sum(singular > tol))
    positive = singular[singular > tol]
    condition = (
        float(singular[0] / positive[-1])
        if positive.size
        else float("inf")
    )
    return rank, condition


def _scale_seconds_clock_efim_to_range_clock(efim: np.ndarray, scene: dict) -> np.ndarray:
    scale = np.diag([1.0, 1.0, 1.0, 1.0 / float(scene["c0"])])
    return _symmetrize_fim(scale.T @ np.asarray(efim, dtype=float) @ scale)


def _projection_efim_from_design(
    design: np.ndarray,
    d_model: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    try:
        nuisance_coeff = np.linalg.lstsq(design, d_model, rcond=None)[0]
    except np.linalg.LinAlgError:
        nuisance_coeff = np.linalg.pinv(design, rcond=1.0e-10) @ d_model
    projected = d_model - design @ nuisance_coeff
    return _symmetrize_fim((2.0 / float(sigma2)) * np.real(projected.conj().T @ projected))


def _real_schur_efim_from_design(
    design: np.ndarray,
    d_model: np.ndarray,
    sigma2: float,
    *,
    prior_precision: np.ndarray | None = None,
) -> np.ndarray:
    nuisance = np.column_stack([design, 1j * design])
    scale = 2.0 / float(sigma2)
    j_chichi = scale * np.real(d_model.conj().T @ d_model)
    j_chixi = scale * np.real(d_model.conj().T @ nuisance)
    j_xixi = scale * np.real(nuisance.conj().T @ nuisance)
    if prior_precision is not None:
        j_xixi = j_xixi + np.asarray(prior_precision, dtype=float)
    try:
        schur = j_chichi - j_chixi @ np.linalg.solve(j_xixi, j_chixi.T)
    except np.linalg.LinAlgError:
        schur = j_chichi - j_chixi @ np.linalg.pinv(j_xixi, rcond=1.0e-10) @ j_chixi.T
    return _symmetrize_fim(schur)


def _true_jones_coefficients(scene: dict) -> np.ndarray:
    k_paths = int(scene["K"])
    beta_true = np.asarray(scene.get("beta_true"), dtype=complex).reshape(-1)
    gamma = np.asarray(scene.get("gamma_true"), dtype=float).reshape(-1)
    eta = np.asarray(scene.get("eta_true"), dtype=float).reshape(-1)
    if beta_true.size < k_paths or gamma.size < k_paths or eta.size < k_paths:
        raise KeyError("missing beta_true/gamma_true/eta_true for Jones PEB")
    coeff = np.empty(2 * k_paths, dtype=complex)
    for k in range(k_paths):
        coeff[2 * k : 2 * k + 2] = beta_true[k] * polarization_vector(
            float(gamma[k]), float(eta[k])
        )
    return coeff


def _constrained_jones_design(
    phi: np.ndarray,
    dphi_dx: list[np.ndarray],
    x_true: np.ndarray,
    k_paths: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1.0e-12
    block = np.zeros((2 * k_paths, k_paths), dtype=complex)
    beta = np.empty(k_paths, dtype=complex)
    for k in range(k_paths):
        x_k = np.asarray(x_true[2 * k : 2 * k + 2], dtype=complex)
        norm = float(np.linalg.norm(x_k))
        if not np.isfinite(norm) or norm <= eps:
            block[2 * k, k] = 1.0
            beta[k] = 0.0
        else:
            block[2 * k : 2 * k + 2, k] = x_k / norm
            beta[k] = norm
    design = phi @ block
    d_model = np.column_stack([(dphi @ block) @ beta for dphi in dphi_dx])
    return design, d_model, beta


def _peb_value_and_diagnostics(
    efim_scaled: np.ndarray,
    parameter_order: list[str],
    cond_threshold: float,
) -> tuple[float, dict[str, Any]]:
    peb, diag = position_peb_from_global_efim(
        efim_scaled,
        parameter_order,
        already_clock_eliminated=False,
        condition_threshold=cond_threshold,
        return_diagnostics=True,
    )
    rank, cond = _fim_rank_condition(efim_scaled)
    return peb, {
        "rank_chi": rank,
        "cond_chi": cond,
        "rank_deficient": bool(rank < min(4, efim_scaled.shape[0])),
        "clock_schur_used": bool(diag.get("clock_eliminated", False)),
        "warning": str(diag.get("warning", "")),
    }


def _anchored_prior_from_config(
    config: dict,
    scene: dict,
    sigma2: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    crb = dict(config.get("crb", {}))
    mode = str(crb.get("jones_anchor_prior_mode", "disabled"))
    scale = float(crb.get("jones_anchor_prior_scale", 1.0))
    k_paths = int(scene["K"])
    if mode == "disabled":
        return None, {
            "anchored_prior_scaling": "disabled",
            "anchored_prior_lambda": float("nan"),
            "anchored_prior_precision_norm": float("nan"),
        }
    if mode == "manual":
        lam = float(scale)
        scaling = "manual_real_precision"
        precision_value = lam
    elif mode == "lambda_from_adaptive":
        global_vp = dict(config.get("global_vp", {}))
        lam = float(global_vp.get("jones_lambda0", 1.0)) * scale
        scaling = "unnormalized_residual_lambda_over_sigma2"
        precision_value = 2.0 * lam / max(float(sigma2), float(config.get("eps", 1.0e-10)))
    else:
        raise ValueError(f"unknown jones_anchor_prior_mode {mode!r}")
    precision = precision_value * np.eye(4 * k_paths, dtype=float)
    return precision, {
        "anchored_prior_scaling": scaling,
        "anchored_prior_lambda": lam,
        "anchored_prior_precision_norm": float(np.linalg.norm(precision)),
    }


def _jones_bound_options(config: dict) -> dict[str, Any]:
    crb = dict(config.get("crb", {}))
    return {
        "include_constrained": bool(crb.get("include_constrained_jones_peb", True)),
        "include_anchored": bool(crb.get("include_anchored_jones_peb", False)),
    }


def compute_constrained_jones_peb(data: dict, config: dict) -> dict[str, Any]:
    scene = data["scene"]
    init = _truth_init_estimate(scene, data["true_components"])
    efim_config = copy.deepcopy(config)
    efim_config["global_vp"] = dict(efim_config.get("global_vp", {}))
    efim_config["global_vp"]["mode"] = "jones_free"
    xi = np.r_[
        np.asarray(scene["p_u_true"], dtype=float).reshape(3),
        float(scene["delta_t_true"]),
    ]
    phi, aux = _build_global_dictionary(xi, init, scene, efim_config, need_jacobian=True)
    x_true = _true_jones_coefficients(scene)
    design, d_model, _ = _constrained_jones_design(
        phi, aux["dPhi_dx"], x_true, int(scene["K"])
    )
    sigma2 = max(float(data.get("noise_variance")), float(config.get("eps", 1.0e-10)))
    efim_seconds = _projection_efim_from_design(design, d_model, sigma2)
    efim_scaled = _scale_seconds_clock_efim_to_range_clock(efim_seconds, scene)
    cond_threshold = float(
        config.get("global_vp", {}).get(
            "efim_cond_threshold", config.get("efim_cond_threshold", 1.0e12)
        )
    )
    peb, diag = _peb_value_and_diagnostics(
        efim_scaled,
        ["p_x_m", "p_y_m", "p_z_m", "c_delta_t_m"],
        cond_threshold,
    )
    return {
        "peb_constrained_jones_m": peb,
        "constrained_jones_peb_m": peb,
        "peb_fim_rank_chi_constrained": diag["rank_chi"],
        "peb_fim_cond_chi_constrained": diag["cond_chi"],
        "peb_constrained_rank_deficient": diag["rank_deficient"],
        "peb_constrained_warning": diag["warning"],
        "J_chi_constrained_scaled": efim_scaled,
    }


def compute_anchored_jones_peb(data: dict, config: dict) -> dict[str, Any]:
    scene = data["scene"]
    init = _truth_init_estimate(scene, data["true_components"])
    efim_config = copy.deepcopy(config)
    efim_config["global_vp"] = dict(efim_config.get("global_vp", {}))
    efim_config["global_vp"]["mode"] = "jones_free"
    xi = np.r_[
        np.asarray(scene["p_u_true"], dtype=float).reshape(3),
        float(scene["delta_t_true"]),
    ]
    phi, aux = _build_global_dictionary(xi, init, scene, efim_config, need_jacobian=True)
    y_vec = np.asarray(data["Y_true"], dtype=complex).reshape(-1)
    coeff, _ = _solve_linear_vp_regularized(phi, y_vec, None, 0.0)
    d_model = np.column_stack([dphi @ coeff for dphi in aux["dPhi_dx"]])
    sigma2 = max(float(data.get("noise_variance")), float(config.get("eps", 1.0e-10)))
    prior, prior_diag = _anchored_prior_from_config(config, scene, sigma2)
    if prior is None:
        return {
            "peb_anchored_jones_m": float("nan"),
            "anchored_jones_peb_m": float("nan"),
            "peb_fim_rank_chi_anchored": 0,
            "peb_fim_cond_chi_anchored": float("inf"),
            "peb_anchored_rank_deficient": True,
            "peb_anchored_warning": "anchored_jones_peb_disabled",
            "J_chi_anchored_scaled": None,
            **prior_diag,
        }
    efim_seconds = _real_schur_efim_from_design(
        phi, d_model, sigma2, prior_precision=prior
    )
    efim_scaled = _scale_seconds_clock_efim_to_range_clock(efim_seconds, scene)
    cond_threshold = float(
        config.get("global_vp", {}).get(
            "efim_cond_threshold", config.get("efim_cond_threshold", 1.0e12)
        )
    )
    peb, diag = _peb_value_and_diagnostics(
        efim_scaled,
        ["p_x_m", "p_y_m", "p_z_m", "c_delta_t_m"],
        cond_threshold,
    )
    return {
        "peb_anchored_jones_m": peb,
        "anchored_jones_peb_m": peb,
        "peb_fim_rank_chi_anchored": diag["rank_chi"],
        "peb_fim_cond_chi_anchored": diag["cond_chi"],
        "peb_anchored_rank_deficient": diag["rank_deficient"],
        "peb_anchored_warning": diag["warning"],
        "J_chi_anchored_scaled": efim_scaled,
        **prior_diag,
    }


def _free_schur_relerr_from_truth_dictionary(
    data: dict,
    config: dict,
    free_efim_scaled: np.ndarray,
) -> float:
    scene = data["scene"]
    init = _truth_init_estimate(scene, data["true_components"])
    efim_config = copy.deepcopy(config)
    efim_config["global_vp"] = dict(efim_config.get("global_vp", {}))
    efim_config["global_vp"]["mode"] = "jones_free"
    xi = np.r_[
        np.asarray(scene["p_u_true"], dtype=float).reshape(3),
        float(scene["delta_t_true"]),
    ]
    phi, aux = _build_global_dictionary(xi, init, scene, efim_config, need_jacobian=True)
    y_vec = np.asarray(data["Y_true"], dtype=complex).reshape(-1)
    coeff, _ = _solve_linear_vp_regularized(phi, y_vec, None, 0.0)
    d_model = np.column_stack([dphi @ coeff for dphi in aux["dPhi_dx"]])
    sigma2 = max(float(data.get("noise_variance")), float(config.get("eps", 1.0e-10)))
    schur_seconds = _real_schur_efim_from_design(phi, d_model, sigma2)
    schur_scaled = _scale_seconds_clock_efim_to_range_clock(schur_seconds, scene)
    denom = max(1.0, float(np.linalg.norm(free_efim_scaled)))
    return float(np.linalg.norm(np.asarray(free_efim_scaled) - schur_scaled) / denom)


def _min_eig_difference(candidate: np.ndarray | None, free: np.ndarray | None) -> float:
    if candidate is None or free is None:
        return float("nan")
    diff = _symmetrize_fim(np.asarray(candidate, dtype=float) - np.asarray(free, dtype=float))
    eig = np.linalg.eigvalsh(diff)
    return float(np.min(eig)) if eig.size else float("nan")


def _peb_from_efim(data: dict, config: dict) -> dict[str, Any]:
    scene = data["scene"]
    init = _truth_init_estimate(scene, data["true_components"])
    warning = ""
    condition = float("inf")
    free_rank = 0
    free_rank_deficient = True
    free_efim_scaled = None
    free_schur_relerr = float("nan")
    try:
        diag = data_only_efim_diagnostic(
            data["Y_true"],
            scene["p_u_true"],
            scene["delta_t_true"],
            init,
            scene,
            config,
            sigma2=data.get("noise_variance"),
        )
        efim = np.asarray(diag["data_only_scaled_efim"], dtype=float)
        free_efim_scaled = efim.copy()
        parameter_order = diag.get(
            "data_only_scaled_efim_parameter_order",
            ["p_x_m", "p_y_m", "p_z_m", "c_delta_t_m"],
        )
        cond_threshold = float(
            config.get("global_vp", {}).get("efim_cond_threshold", config.get("efim_cond_threshold", 1.0e12))
        )
        peb, peb_diag = position_peb_from_global_efim(
            efim,
            parameter_order,
            already_clock_eliminated=bool(
                diag.get("data_only_scaled_efim_clock_eliminated", False)
            ),
            condition_threshold=cond_threshold,
            return_diagnostics=True,
        )
        warning = str(peb_diag["warning"])
        condition = float(peb_diag["efim_condition_number"])
        free_rank, free_condition_chi = _fim_rank_condition(efim)
        free_rank_deficient = bool(free_rank < min(4, efim.shape[0]))
        try:
            free_schur_relerr = _free_schur_relerr_from_truth_dictionary(
                data, config, efim
            )
        except (KeyError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            free_schur_relerr = float("nan")
    except (KeyError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        peb = float("nan")
        warning = f"data_only_efim_peb_failed: {type(exc).__name__}: {exc}"
        parameter_order = ["p_x_m", "p_y_m", "p_z_m", "c_delta_t_m"]
        diag = {}
        free_condition_chi = float("inf")
    options = _jones_bound_options(config)
    constrained: dict[str, Any] = {
        "peb_constrained_jones_m": float("nan"),
        "constrained_jones_peb_m": float("nan"),
        "peb_fim_rank_chi_constrained": 0,
        "peb_fim_cond_chi_constrained": float("inf"),
        "peb_constrained_rank_deficient": True,
        "peb_constrained_warning": "constrained_jones_peb_disabled",
        "J_chi_constrained_scaled": None,
    }
    if options["include_constrained"]:
        try:
            constrained = compute_constrained_jones_peb(data, config)
        except (KeyError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            constrained["peb_constrained_warning"] = (
                f"constrained_jones_peb_failed: {type(exc).__name__}: {exc}"
            )
    anchored: dict[str, Any] = {
        "peb_anchored_jones_m": float("nan"),
        "anchored_jones_peb_m": float("nan"),
        "peb_fim_rank_chi_anchored": 0,
        "peb_fim_cond_chi_anchored": float("inf"),
        "peb_anchored_rank_deficient": True,
        "peb_anchored_warning": "anchored_jones_peb_disabled",
        "J_chi_anchored_scaled": None,
        "anchored_prior_scaling": "disabled",
        "anchored_prior_lambda": float("nan"),
        "anchored_prior_precision_norm": float("nan"),
    }
    if options["include_anchored"]:
        try:
            anchored = compute_anchored_jones_peb(data, config)
        except (KeyError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            anchored["peb_anchored_warning"] = (
                f"anchored_jones_peb_failed: {type(exc).__name__}: {exc}"
            )
    con_minus_free = _min_eig_difference(
        constrained.get("J_chi_constrained_scaled"), free_efim_scaled
    )
    hyb_minus_free = _min_eig_difference(
        anchored.get("J_chi_anchored_scaled"), free_efim_scaled
    )
    constrained_peb = float(constrained.get("peb_constrained_jones_m", float("nan")))
    anchored_peb = float(anchored.get("peb_anchored_jones_m", float("nan")))
    ordering_checks = []
    if np.isfinite(constrained_peb) and np.isfinite(peb):
        ordering_checks.append(constrained_peb <= peb * (1.0 + 1.0e-7))
    if options["include_anchored"] and np.isfinite(anchored_peb) and np.isfinite(peb):
        ordering_checks.append(anchored_peb <= peb * (1.0 + 1.0e-7))
    ordering_ok = bool(all(ordering_checks)) if ordering_checks else ""
    extra_warnings = [
        str(constrained.get("peb_constrained_warning", "")),
        str(anchored.get("peb_anchored_warning", "")),
    ]
    extra_warnings = [item for item in extra_warnings if item and not item.endswith("_disabled")]
    if extra_warnings:
        warning = "; ".join([item for item in [warning, *extra_warnings] if item])
    mode = str(config.get("receiver_mode", "full_6d"))
    return {
        "peb_position_m": peb,
        "peb_scalar_m": peb if mode == "scalar" else float("nan"),
        "peb_dual_m": peb if mode == "dual_pol" else float("nan"),
        "peb_evs_m": peb if mode == "full_6d" else float("nan"),
        "peb_free_jones_m": peb,
        "peb_constrained_jones_m": constrained_peb,
        "peb_anchored_jones_m": anchored_peb,
        "peb_variant": "free_jones_peb",
        "jones_bound_type": "free",
        "constrained_jones_peb_m": constrained_peb,
        "anchored_jones_peb_m": anchored_peb,
        "free_jones_peb_m": peb,
        "peb_fim_rank_chi_free": free_rank,
        "peb_fim_rank_chi_constrained": int(
            constrained.get("peb_fim_rank_chi_constrained", 0)
        ),
        "peb_fim_rank_chi_anchored": int(
            anchored.get("peb_fim_rank_chi_anchored", 0)
        ),
        "peb_fim_cond_chi_free": free_condition_chi,
        "peb_fim_cond_chi_constrained": float(
            constrained.get("peb_fim_cond_chi_constrained", float("inf"))
        ),
        "peb_fim_cond_chi_anchored": float(
            anchored.get("peb_fim_cond_chi_anchored", float("inf"))
        ),
        "peb_clock_schur_used": True,
        "peb_rank_deficient": bool(
            free_rank_deficient
            or constrained.get("peb_constrained_rank_deficient", False)
            or (
                options["include_anchored"]
                and anchored.get("peb_anchored_rank_deficient", False)
            )
        ),
        "anchored_prior_scaling": str(
            anchored.get("anchored_prior_scaling", "disabled")
        ),
        "anchored_prior_lambda": float(
            anchored.get("anchored_prior_lambda", float("nan"))
        ),
        "anchored_prior_precision_norm": float(
            anchored.get("anchored_prior_precision_norm", float("nan"))
        ),
        "peb_free_projection_schur_relerr": free_schur_relerr,
        "peb_con_minus_free_min_eig": con_minus_free,
        "peb_hyb_minus_free_min_eig": hyb_minus_free,
        "peb_ordering_ok": ordering_ok,
        "warning": warning,
        "peb_is_data_only": True,
        "peb_uses_regularization": False,
        "nuisance_model": "jones_linear",
        "clock_eliminated": True,
        "efim_condition_number": condition,
        "efim_parameter_order": list(parameter_order),
        "peb_reference_type": "matched_model",
        "efim_unscaled_cache_hit": bool(
            diag.get("efim_unscaled_cache_hit", False)
        ),
        "efim_unscaled_cache_key": str(
            diag.get("efim_unscaled_cache_key", "")
        ),
        "efim_sigma2": float(diag.get("efim_sigma2", float("nan"))),
        "efim_reuse_mode": str(diag.get("efim_reuse_mode", "")),
        "peb_backend": str(diag.get("peb_backend", "cpu")),
    }


def extract_metrics(result: dict, outlier_threshold_m: float) -> dict[str, Any]:
    final = result.get("final", {})
    scene = result.get("scene", {})
    true_components = result.get("true_components", {})
    timing = result.get("timing", {})
    reliability = result.get("reliability", final.get("reliability", {}))
    variant_diagnostics = result.get("variant_diagnostics", {})

    y_nmse = float("nan")
    if final.get("Y_hat") is not None and result.get("Y_true") is not None:
        y_nmse = float(relative_nmse(final["Y_hat"], result["Y_true"]))

    pos_rmse = float("nan")
    if final.get("p_u") is not None and scene.get("p_u_true") is not None:
        pos_rmse = float(position_rmse(np.asarray(final["p_u"]), np.asarray(scene["p_u_true"])))

    components = final.get("components", {})
    metrics = {
        "K": scene.get("K", ""),
        "receiver_mode": scene.get("receiver_mode", ""),
        "nested_receiver_noise_convention": result.get(
            "nested_receiver_noise_convention", ""
        ),
        "reference_receiver_mode": result.get("reference_receiver_mode", ""),
        "reference_sigma2": _finite_float(result.get("reference_sigma2")),
        "nested_base_y_noisy_hash": result.get(
            "nested_base_y_noisy_hash", ""
        ),
        "warning": result.get("warning", ""),
        "position_rmse_m": pos_rmse,
        "y_nmse": y_nmse,
        "range_rmse_m": _rmse_array(components.get("ranges"), true_components.get("ranges")),
        "tau_rmse_s": _rmse_array(components.get("taus"), true_components.get("taus")),
        "raw_objective_final": _finite_float(
            get_nested(result, ["final.raw_objective_final", "final.raw_objective"], np.nan)
        ),
        "outlier_flag": bool(np.isfinite(pos_rmse) and pos_rmse > outlier_threshold_m),
        "position_error_m": pos_rmse,
        "selected_branch": get_nested(result, ["selected_branch", "final.selected_branch"], ""),
        "final_refinement_method": get_nested(result, ["final.final_refinement_method"], ""),
        "final_runner_name": variant_diagnostics.get("final_runner_name", ""),
        "used_main_single_proposed_path": variant_diagnostics.get(
            "used_main_single_proposed_path", False
        ),
        "variant_config_hash": variant_diagnostics.get("variant_config_hash", ""),
        "global_vp_mode": get_nested(result, ["final.global_vp_mode", "final.vp_mode"], ""),
        "global_vp_backend": get_nested(result, ["final.global_vp_backend"], ""),
        "global_vp_gpu_used": get_nested(result, ["final.global_vp_gpu_used"], ""),
        "global_vp_gpu_device": get_nested(result, ["final.global_vp_gpu_device"], ""),
        "global_vp_objective_backend": get_nested(
            result,
            ["final.global_vp_objective_backend"],
            "",
        ),
        "global_vp_linear_solve_backend": get_nested(
            result,
            ["final.global_vp_linear_solve_backend", "final.global_vp_lstsq_backend"],
            "",
        ),
        "jones_mode": variant_diagnostics.get(
            "jones_mode",
            get_nested(result, ["final.selected_vp_family_branch", "final.global_vp_mode"], ""),
        ),
        "adaptive_enabled": variant_diagnostics.get("adaptive_enabled", False),
        "adaptive_policy_name": variant_diagnostics.get("adaptive_policy_name", ""),
        "selected_vp_family_branch": get_nested(result, ["final.selected_vp_family_branch"], ""),
        "lambda_path_min": _finite_float(variant_diagnostics.get("lambda_path_min")),
        "lambda_path_max": _finite_float(variant_diagnostics.get("lambda_path_max")),
        "lambda_path_mean": _finite_float(variant_diagnostics.get("lambda_path_mean")),
        "lambda_clipped_fraction": _finite_float(
            variant_diagnostics.get("lambda_clipped_fraction")
        ),
        "used_stage1_regularizer": variant_diagnostics.get(
            "used_stage1_regularizer", False
        ),
        "linear_nuisance_dim": get_nested(result, ["final.linear_nuisance_dim"], ""),
        "nonlinear_dim": get_nested(result, ["final.nonlinear_dim"], ""),
        "fixed_pol_score": _finite_float(get_nested(result, ["final.fixed_pol_score"], np.nan)),
        "jones_score": _finite_float(get_nested(result, ["final.jones_score"], np.nan)),
        "lambda_jones_per_path": _vector_string(
            get_nested(result, ["final.lambda_jones_per_path"], None)
        ),
        "snr_eff_per_path": _vector_string(get_nested(result, ["final.snr_eff_per_path"], None)),
        "jones_leakage_per_path": _vector_string(
            get_nested(result, ["final.jones_leakage_per_path"], None)
        ),
        "reliability_decision": reliability.get("decision", ""),
        "proposed_stage2_policy": reliability.get(
            "proposed_stage2_policy",
            get_nested(result, ["stage1_config.proposed_stage2_policy"], ""),
        ),
        "trigger_reasons": _list_string(reliability.get("trigger_reasons", [])),
        "gof_stat": _finite_float(reliability.get("gof_stat")),
        "gof_dof": reliability.get("gof_dof", ""),
        "gof_pass": reliability.get("gof_pass", ""),
        "data_only_scaled_efim_lambda_min": _finite_float(
            reliability.get("data_only_scaled_efim_lambda_min")
        ),
        "data_only_scaled_efim_condition_number": _finite_float(
            reliability.get("data_only_scaled_efim_condition_number")
        ),
        "stage1_assignment_margin": _finite_float(reliability.get("assignment_margin")),
        "stage1_selected_clock_std_ns": _finite_float(reliability.get("sigma_delta_t_ns")),
        "delta_t_k_ns": _vector_string(reliability.get("delta_t_k_ns")),
        "stage1_runtime_s": _finite_float(timing.get("stage1")),
        "global_vp_runtime_s": _finite_float(timing.get("vp")),
        "total_runtime_s": _finite_float(timing.get("total", timing.get("diagnostic_total"))),
    }
    for key in (
        "peb_position_m",
        "peb_scalar_m",
        "peb_dual_m",
        "peb_evs_m",
        "peb_free_jones_m",
        "peb_constrained_jones_m",
        "peb_anchored_jones_m",
        "constrained_jones_peb_m",
        "anchored_jones_peb_m",
        "free_jones_peb_m",
        "peb_fim_rank_chi_free",
        "peb_fim_rank_chi_constrained",
        "peb_fim_rank_chi_anchored",
        "peb_fim_cond_chi_free",
        "peb_fim_cond_chi_constrained",
        "peb_fim_cond_chi_anchored",
        "anchored_prior_lambda",
        "anchored_prior_precision_norm",
        "peb_free_projection_schur_relerr",
        "peb_con_minus_free_min_eig",
        "peb_hyb_minus_free_min_eig",
    ):
        metrics[key] = _finite_float(result.get(key))
    metrics["peb_variant"] = str(result.get("peb_variant", ""))
    metrics["jones_bound_type"] = str(result.get("jones_bound_type", ""))
    metrics["peb_clock_schur_used"] = result.get("peb_clock_schur_used", "")
    metrics["peb_rank_deficient"] = result.get("peb_rank_deficient", "")
    metrics["anchored_prior_scaling"] = str(
        result.get("anchored_prior_scaling", "")
    )
    metrics["peb_ordering_ok"] = result.get("peb_ordering_ok", "")
    metrics.update(
        {
            "efim_unscaled_cache_hit": bool(
                result.get("efim_unscaled_cache_hit", False)
            ),
            "efim_unscaled_cache_key": str(
                result.get("efim_unscaled_cache_key", "")
            ),
            "efim_sigma2": _finite_float(result.get("efim_sigma2")),
            "efim_reuse_mode": str(result.get("efim_reuse_mode", "")),
            "peb_backend": str(result.get("peb_backend", "")),
        }
    )
    metrics.update(
        {
            "direct_candidate_position_error_m": _finite_float(
                result.get("direct_candidate_position_error_m")
            ),
            "direct_candidate_y_nmse": _finite_float(
                result.get("direct_candidate_y_nmse")
            ),
            "direct_candidate_raw_objective": _finite_float(
                result.get("direct_candidate_raw_objective")
            ),
            "rescue_candidate_position_error_m": _finite_float(
                result.get("rescue_candidate_position_error_m")
            ),
            "rescue_candidate_y_nmse": _finite_float(
                result.get("rescue_candidate_y_nmse")
            ),
            "rescue_candidate_raw_objective": _finite_float(
                result.get("rescue_candidate_raw_objective")
            ),
            "rescue_candidate_available": bool(
                result.get("rescue_candidate_available", False)
            ),
            "rescue_accept_decision": str(
                result.get("rescue_accept_decision", "")
            ),
            "rescue_reject_reason": str(result.get("rescue_reject_reason", "")),
            "direct_candidate_lambda_jones_per_path": _vector_string(
                result.get("direct_candidate_lambda_jones_per_path")
            ),
            "direct_candidate_snr_eff_per_path": _vector_string(
                result.get("direct_candidate_snr_eff_per_path")
            ),
            "direct_candidate_jones_leakage_per_path": _vector_string(
                result.get("direct_candidate_jones_leakage_per_path")
            ),
            "rescue_candidate_lambda_jones_per_path": _vector_string(
                result.get("rescue_candidate_lambda_jones_per_path")
            ),
            "rescue_candidate_snr_eff_per_path": _vector_string(
                result.get("rescue_candidate_snr_eff_per_path")
            ),
            "rescue_candidate_jones_leakage_per_path": _vector_string(
                result.get("rescue_candidate_jones_leakage_per_path")
            ),
            "direct_candidate_data_only_scaled_efim_lambda_min": _finite_float(
                result.get("direct_candidate_data_only_scaled_efim_lambda_min")
            ),
            "direct_candidate_data_only_scaled_efim_condition_number": _finite_float(
                result.get("direct_candidate_data_only_scaled_efim_condition_number")
            ),
            "rescue_candidate_data_only_scaled_efim_lambda_min": _finite_float(
                result.get("rescue_candidate_data_only_scaled_efim_lambda_min")
            ),
            "rescue_candidate_data_only_scaled_efim_condition_number": _finite_float(
                result.get("rescue_candidate_data_only_scaled_efim_condition_number")
            ),
            "legacy_stage1_decision": reliability.get("legacy_stage1_decision", ""),
            "gof_reliability_decision": reliability.get(
                "gof_reliability_decision", ""
            ),
            "stage1_geometry_trigger": reliability.get(
                "stage1_geometry_trigger", ""
            ),
            "stage1_geometry_trigger_reasons": _list_string(
                reliability.get("stage1_geometry_trigger_reasons", [])
            ),
        }
    )
    metrics.update(
        {
            "ngc_policy_active": bool(result.get("ngc_policy_active", False)),
            "ngc_lambda_ris": _finite_float(result.get("ngc_lambda_ris")),
            "ngc_direct_clock_score": _finite_float(
                result.get("ngc_direct_clock_score")
            ),
            "ngc_direct_clock_score_norm": _finite_float(
                result.get("ngc_direct_clock_score_norm")
            ),
            "ngc_direct_clock_dof": result.get("ngc_direct_clock_dof", ""),
            "ngc_direct_clock_sigma_source": str(
                result.get("ngc_direct_clock_sigma_source", "")
            ),
            "ngc_direct_clock_std_ns": _finite_float(
                result.get("ngc_direct_clock_std_ns")
            ),
            "ngc_direct_ris_score": _finite_float(
                result.get("ngc_direct_ris_score")
            ),
            "ngc_direct_ris_score_norm": _finite_float(
                result.get("ngc_direct_ris_score_norm")
            ),
            "ngc_direct_ris_available": bool(
                result.get("ngc_direct_ris_available", False)
            ),
            "ngc_direct_total_score": _finite_float(
                result.get("ngc_direct_total_score")
            ),
            "ngc_direct_cert_status": str(
                result.get("ngc_direct_cert_status", "")
            ),
            "ngc_direct_cert_reason": str(
                result.get("ngc_direct_cert_reason", "")
            ),
            "ngc_rescue_requested": bool(result.get("ngc_rescue_requested", False)),
            "ngc_rescue_request_reason": str(
                result.get("ngc_rescue_request_reason", "")
            ),
            "ngc_rescue_clock_score": _finite_float(
                result.get("ngc_rescue_clock_score")
            ),
            "ngc_rescue_clock_score_norm": _finite_float(
                result.get("ngc_rescue_clock_score_norm")
            ),
            "ngc_rescue_clock_dof": result.get("ngc_rescue_clock_dof", ""),
            "ngc_rescue_clock_sigma_source": str(
                result.get("ngc_rescue_clock_sigma_source", "")
            ),
            "ngc_rescue_clock_std_ns": _finite_float(
                result.get("ngc_rescue_clock_std_ns")
            ),
            "ngc_rescue_ris_score": _finite_float(
                result.get("ngc_rescue_ris_score")
            ),
            "ngc_rescue_ris_score_norm": _finite_float(
                result.get("ngc_rescue_ris_score_norm")
            ),
            "ngc_rescue_ris_available": bool(
                result.get("ngc_rescue_ris_available", False)
            ),
            "ngc_rescue_total_score": _finite_float(
                result.get("ngc_rescue_total_score")
            ),
            "ngc_rescue_cert_status": str(
                result.get("ngc_rescue_cert_status", "")
            ),
            "ngc_rescue_cert_reason": str(
                result.get("ngc_rescue_cert_reason", "")
            ),
            "ngc_selected_by": str(result.get("ngc_selected_by", "")),
            "ngc_final_unreliable": bool(
                result.get("ngc_final_unreliable", False)
            ),
            "ngc_threshold_clock_green": _finite_float(
                result.get("ngc_threshold_clock_green")
            ),
            "ngc_threshold_clock_red": _finite_float(
                result.get("ngc_threshold_clock_red")
            ),
        }
    )
    return metrics


def _stage1_only_result(config: dict) -> dict:
    data = _make_data(config)
    stage1 = run_stage1_only(data, config)
    estimate = stage1["estimate"]
    y_hat = reconstruct_raw_tensor_from_structured_estimate(estimate, data["scene"])
    try:
        p_hat = estimate_position_from_local_ris(data["scene"], estimate, config)
    except (KeyError, ValueError, np.linalg.LinAlgError):
        p_hat = estimate_position_from_ris_eta(data["scene"], estimate)
    tau_hat = np.array([tau_from_pole(pole, data["scene"]["delta_f"]) for pole in estimate["poles"]])
    ranges = np.asarray(estimate["ris_eta"], dtype=float)[:, 0]
    raw_residual = y_hat - data["Y_noisy"]
    final = {
        "Y_hat": y_hat,
        "p_u": p_hat,
        "components": {"taus": tau_hat, "ranges": ranges},
        "raw_objective_final": float(np.vdot(raw_residual.reshape(-1), raw_residual.reshape(-1)).real / data["Y_noisy"].size),
        "final_refinement_method": "stage1_only",
        "vp_mode": "none",
        "global_vp_mode": "none",
        "linear_nuisance_dim": 0,
        "nonlinear_dim": 0,
    }
    return {
        **data,
        "estimate_initial": estimate,
        "estimate_used": estimate,
        "final": final,
        "timing": {
            **data.get("timing", {}),
            **stage1["timing"],
            "stage2": 0.0,
            "vp": 0.0,
            "total": float(sum(data.get("timing", {}).values()) + stage1["timing"].get("stage1", 0.0)),
        },
        "reliability": {},
        "selected_branch": "stage1_only",
    }


def _oracle_result(config: dict) -> dict:
    data = _make_data(config)
    init = _truth_init_estimate(data["scene"], data["true_components"])
    vp_start = time.perf_counter()
    final = global_exact_spherical_vp_refinement(data["Y_noisy"], init, data["scene"], config)
    vp_s = time.perf_counter() - vp_start
    final["selected_branch"] = "oracle_init_vp"
    final["final_refinement_method"] = "global_exact_spherical_vp"
    return {
        **data,
        "estimate_initial": init,
        "estimate_used": init,
        "final": final,
        "timing": {**data.get("timing", {}), "stage1": 0.0, "stage2": 0.0, "vp": vp_s, "total": vp_s},
        "reliability": {"decision": "oracle_init_vp", "trigger_reasons": ["oracle_truth_initialization"]},
        "selected_branch": "oracle_init_vp",
    }


def _stage1_only_result_from_shared(data: dict, stage1: dict, config: dict) -> dict:
    estimate = copy.deepcopy(stage1["estimate"])
    y_hat = reconstruct_raw_tensor_from_structured_estimate(estimate, data["scene"])
    try:
        p_hat = estimate_position_from_local_ris(data["scene"], estimate, config)
    except (KeyError, ValueError, np.linalg.LinAlgError):
        p_hat = estimate_position_from_ris_eta(data["scene"], estimate)
    tau_hat = np.array([tau_from_pole(pole, data["scene"]["delta_f"]) for pole in estimate["poles"]])
    ranges = np.asarray(estimate["ris_eta"], dtype=float)[:, 0]
    raw_residual = y_hat - data["Y_noisy"]
    final = {
        "Y_hat": y_hat,
        "p_u": p_hat,
        "components": {"taus": tau_hat, "ranges": ranges},
        "raw_objective_final": float(
            np.vdot(raw_residual.reshape(-1), raw_residual.reshape(-1)).real
            / data["Y_noisy"].size
        ),
        "final_refinement_method": "stage1_only",
        "vp_mode": "none",
        "global_vp_mode": "none",
        "linear_nuisance_dim": 0,
        "nonlinear_dim": 0,
    }
    return {
        **data,
        "estimate_initial": estimate,
        "estimate_used": estimate,
        "final": final,
        "timing": {
            **data.get("timing", {}),
            **stage1["timing"],
            "stage2": 0.0,
            "vp": 0.0,
            "total": float(
                sum(data.get("timing", {}).values()) + stage1["timing"].get("stage1", 0.0)
            ),
        },
        "reliability": {},
        "selected_branch": "stage1_only",
    }


def _oracle_result_from_shared(data: dict, config: dict) -> dict:
    init = _truth_init_estimate(data["scene"], data["true_components"])
    vp_start = time.perf_counter()
    final = global_exact_spherical_vp_refinement(data["Y_noisy"], init, data["scene"], config)
    vp_s = time.perf_counter() - vp_start
    final["selected_branch"] = "oracle_init_vp"
    final["final_refinement_method"] = "global_exact_spherical_vp"
    return {
        **data,
        "estimate_initial": init,
        "estimate_used": init,
        "final": final,
        "timing": {**data.get("timing", {}), "stage1": 0.0, "stage2": 0.0, "vp": vp_s, "total": vp_s},
        "reliability": {"decision": "oracle_init_vp", "trigger_reasons": ["oracle_truth_initialization"]},
        "selected_branch": "oracle_init_vp",
    }


def run_stage1_shared_trial(data: dict, config: dict) -> tuple[dict, dict, dict]:
    stage1 = run_stage1_only(data, config)
    reliability_base: dict[str, Any] = {}
    return stage1["estimate"], stage1["timing"], reliability_base


def _result_to_row(
    result: dict,
    *,
    figure: str,
    variant: str,
    trial_id: int,
    trial_seed: int,
    snr_db: float,
    x_name: str,
    x_value: float,
    outlier_threshold_m: float,
    runtime_s: float,
    store_large_arrays: bool,
    rss_before: float,
    profile_memory: bool,
    compact_result: bool = True,
) -> dict[str, Any]:
    row = _empty_row()
    row.update(
        {
            "figure": figure,
            "variant": variant,
            "trial_id": int(trial_id),
            "seed": int(trial_seed),
            "snr_db": float(snr_db),
            "x_name": x_name,
            "x_value": float(x_value),
            "failed": False,
            "error": "",
        }
    )
    row.update(extract_metrics(result, outlier_threshold_m))
    if compact_result:
        compact_experiment_result(result, keep_large_arrays=store_large_arrays)
    rss_after = _rss_mb() if profile_memory else float("nan")
    row["rss_mb_before"] = rss_before
    row["rss_mb_after"] = rss_after
    row["rss_mb_delta"] = (
        rss_after - rss_before
        if np.isfinite(rss_before) and np.isfinite(rss_after)
        else float("nan")
    )
    row["runtime_s"] = float(runtime_s)
    return row


def _failure_row_from_payload(
    *,
    figure: str,
    variant: str,
    trial_id: int,
    trial_seed: int,
    snr_db: float,
    x_name: str,
    x_value: float,
    k_paths: int,
    receiver_mode: str,
    exc: BaseException,
) -> dict[str, Any]:
    row = _empty_row()
    row.update(
        {
            "figure": figure,
            "variant": variant,
            "trial_id": int(trial_id),
            "seed": int(trial_seed),
            "snr_db": float(snr_db),
            "x_name": x_name,
            "x_value": float(x_value),
            "K": int(k_paths),
            "receiver_mode": receiver_mode,
            "failed": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    return row


def run_final_vp_from_shared_stage1(
    data: dict,
    stage1: dict,
    config: dict,
    variant_spec: dict[str, Any],
    allow_stage2: bool,
) -> dict:
    runner_name = str(variant_spec.get("_runner", "proposed_post_stage1"))
    result = run_from_existing_stage1(
        data,
        {"estimate": copy.deepcopy(stage1["estimate"]), "timing": dict(stage1["timing"])},
        config,
        allow_stage2=allow_stage2,
    )
    return _annotate_variant_result(
        result,
        config,
        final_runner_name=runner_name,
        used_main_single_proposed_path=runner_name == "main_single_proposed",
    )


def run_fig1_adaptive_from_shared(
    data: dict,
    stage1: dict,
    config: dict,
) -> dict:
    """Run the Fig.1 Proposed path using the same post-Stage-I logic as main_single."""
    spec = _variant_specs(FIG1_FIG2_SHARED_FIGURE)[
        "adaptive_jones_vp_proposed"
    ]
    proposed_config = apply_nested_update(copy.deepcopy(config), spec)
    return run_final_vp_from_shared_stage1(
        copy.deepcopy(data),
        copy.deepcopy(stage1),
        proposed_config,
        spec,
        allow_stage2=True,
    )


def _config_diff_summary(left: dict, right: dict) -> str:
    keys = sorted(set(left) | set(right))
    differences = []
    for key in keys:
        left_value = json.dumps(left.get(key), sort_keys=True, default=str)
        right_value = json.dumps(right.get(key), sort_keys=True, default=str)
        if left_value != right_value:
            differences.append(key)
    return ",".join(differences)


def _main_single_debug_metrics(result: dict) -> dict[str, Any]:
    metrics = extract_metrics(result, outlier_threshold_m=float("inf"))
    mode = str(metrics.get("global_vp_mode", ""))
    return {
        "position_error_m": _finite_float(metrics.get("position_error_m")),
        "y_nmse": _finite_float(metrics.get("y_nmse")),
        "raw_objective": _finite_float(metrics.get("raw_objective_final")),
        "global_vp_mode": mode,
        "adaptive_enabled": mode == "adaptive_jones",
    }


def _peb_result(config: dict, out_dir: pathlib.Path | None = None) -> dict:
    key = peb_cache_key(config)
    cached = _PEB_CACHE.get(key)
    if cached is None:
        cached = _read_persistent_peb_cache(config, out_dir)
        if cached is not None:
            _PEB_CACHE[key] = copy.deepcopy(cached)
    if cached is not None:
        return {
            "scene": {
                "K": int(config.get("K", "")),
                "receiver_mode": str(
                    config.get("receiver_mode", config.get("evs_selection", "full_6d"))
                ),
            },
            **copy.deepcopy(cached),
            "final": {},
            "timing": {},
        }
    data = _make_data(config)
    peb_metrics = _peb_from_efim(data, config)
    _PEB_CACHE[key] = copy.deepcopy(peb_metrics)
    _write_persistent_peb_cache(config, out_dir, peb_metrics)
    return {**data, **peb_metrics, "final": {}, "timing": data.get("timing", {})}


def _peb_bound_type_for_variant(variant: str) -> str:
    name = str(variant).lower()
    if "constrained" in name:
        return "constrained"
    if "anchored" in name:
        return "anchored"
    return "free"


def _apply_peb_bound_variant(result: dict, variant: str) -> dict:
    bound_type = _peb_bound_type_for_variant(variant)
    if bound_type == "constrained":
        peb = result.get("peb_constrained_jones_m", float("nan"))
        variant_name = "constrained_jones_peb"
    elif bound_type == "anchored":
        peb = result.get("peb_anchored_jones_m", float("nan"))
        variant_name = "anchored_jones_peb"
    else:
        peb = result.get("peb_free_jones_m", result.get("peb_position_m", float("nan")))
        variant_name = "free_jones_peb"
    result = copy.deepcopy(result)
    result["peb_position_m"] = peb
    mode = str(result.get("scene", {}).get("receiver_mode", "full_6d"))
    result["peb_scalar_m"] = peb if mode == "scalar" else float("nan")
    result["peb_dual_m"] = peb if mode == "dual_pol" else float("nan")
    result["peb_evs_m"] = peb if mode == "full_6d" else float("nan")
    result["peb_variant"] = variant_name
    result["jones_bound_type"] = bound_type
    return result


def run_one_trial(
    config: dict,
    trial_seed: int,
    variant_name: str,
    figure_name: str,
    *,
    trial_id: int,
    x_name: str,
    x_value: float,
    paper_k: int,
    outlier_threshold_m: float,
    verbose: bool = False,
    runner: str = "proposed",
    allow_stage2: bool = True,
    store_large_arrays: bool = False,
    profile_memory: bool = False,
    blas_threads: int = 1,
    respect_existing_blas_env: bool = False,
    trim_memory_enabled: bool = True,
    out_dir: pathlib.Path | None = None,
) -> tuple[dict[str, Any], str]:
    _apply_blas_thread_env(
        blas_threads,
        respect_existing_blas_env=respect_existing_blas_env,
    )
    config = copy.deepcopy(config)
    config["seed"] = int(trial_seed)
    config["experiment"] = dict(config.get("experiment", {}))
    config["experiment"]["store_large_arrays"] = bool(store_large_arrays)
    log_buffer = io.StringIO()
    start = time.perf_counter()
    rss_before = _rss_mb() if profile_memory else float("nan")
    row = _empty_row()
    row.update(
        {
            "figure": figure_name,
            "variant": variant_name,
            "trial_id": trial_id,
            "seed": int(trial_seed),
            "snr_db": float(config["SNR_dB"]),
            "x_name": x_name,
            "x_value": float(x_value),
            "failed": False,
            "error": "",
        }
    )
    _set_row_k_metadata(
        row,
        figure=figure_name,
        effective_k=int(config["K"]),
        paper_k=int(paper_k),
        x_value=float(x_value),
        receiver_mode=str(config.get("receiver_mode", "full_6d")),
        config_seed=int(config["seed"]),
        num_ris_paths=_num_ris_paths(config),
    )
    try:
        stream = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(log_buffer)
        with stream, thread_limit_context(blas_threads):
            if runner == "stage1_only":
                result = _stage1_only_result(config)
            elif runner == "oracle_init_vp":
                result = _oracle_result(config)
            elif runner == "peb_only":
                result = _peb_result(config, out_dir)
                result = _apply_peb_bound_variant(result, variant_name)
            else:
                result = run_single_proposed_diagnostic(config, allow_stage2=allow_stage2)
        result = _annotate_variant_result(
            result,
            config,
            final_runner_name=runner,
            used_main_single_proposed_path=runner == "main_single_proposed",
        )
        row.update(extract_metrics(result, outlier_threshold_m))
        compact_experiment_result(result, keep_large_arrays=store_large_arrays)
        del result
    except Exception as exc:  # noqa: BLE001 - failed trials must be logged as rows.
        row["failed"] = True
        row["error"] = f"{type(exc).__name__}: {exc}"
        log_buffer.write(f"\nERROR {type(exc).__name__}: {exc}\n")
    if trim_memory_enabled:
        trim_memory()
    else:
        gc.collect()
    rss_after = _rss_mb() if profile_memory else float("nan")
    row["rss_mb_before"] = rss_before
    row["rss_mb_after"] = rss_after
    row["rss_mb_delta"] = (
        rss_after - rss_before
        if np.isfinite(rss_before) and np.isfinite(rss_after)
        else float("nan")
    )
    row["runtime_s"] = float(time.perf_counter() - start)
    return row, log_buffer.getvalue()


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_atomic_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with StreamingCsvWriter(path, fieldnames) as writer:
        for row in rows:
            writer.writerow(row)


class StreamingCsvWriter:
    """Write trial rows to a temporary CSV and atomically publish on completion."""

    def __init__(
        self,
        final_path: pathlib.Path,
        fieldnames: list[str],
        *,
        flush_every: int = 1,
    ):
        self.final_path = pathlib.Path(final_path)
        self.tmp_path = self.final_path.with_name(f"{self.final_path.name}.tmp")
        self.fieldnames = fieldnames
        self.flush_every = int(flush_every)
        self.rows_since_flush = 0
        self.handle: Any | None = None
        self.writer: csv.DictWriter | None = None

    def __enter__(self) -> "StreamingCsvWriter":
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.tmp_path.open("w", newline="")
        self.writer = csv.DictWriter(
            self.handle, fieldnames=self.fieldnames, extrasaction="ignore"
        )
        self.writer.writeheader()
        self.handle.flush()
        return self

    def writerow(self, row: dict[str, Any]) -> None:
        if self.writer is None or self.handle is None:
            raise RuntimeError("StreamingCsvWriter is not open")
        self.writer.writerow(row)
        self.rows_since_flush += 1
        if self.flush_every <= 1 or self.rows_since_flush >= self.flush_every:
            self.handle.flush()
            self.rows_since_flush = 0

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
        if exc_type is None:
            os.replace(self.tmp_path, self.final_path)
        return False


def _read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


RESCUE_POLICY_VARIANTS = {
    "adaptive_jones_vp_proposed_old_gated",
    "adaptive_jones_vp_proposed_force_lower_raw",
    "free_jones_vp",
    "free_jones_vp_gated_rescue",
    "free_jones_vp_geometry_gated_rescue",
    "free_jones_vp_force_rescue",
    "free_jones_vp_force_rescue_lower_raw",
    "free_jones_vp_geometry_gated_lower_raw",
}

RESCUE_POLICY_PAIRED_FIELDS = [
    "figure",
    "variant",
    "snr_db",
    "trial_id",
    "seed",
    "receiver_mode",
    "K",
    "paper_k",
    "effective_K",
    "selected_branch",
    "reliability_decision",
    "trigger_reasons",
    "gof_pass",
    "position_error_m",
    "peb_position_m",
    "error_over_peb",
    "relative_outlier_5peb",
    "y_nmse",
    "raw_objective_final",
    "direct_candidate_position_error_m",
    "direct_candidate_y_nmse",
    "direct_candidate_raw_objective",
    "rescue_candidate_position_error_m",
    "rescue_candidate_y_nmse",
    "rescue_candidate_raw_objective",
    "rescue_candidate_available",
    "rescue_accept_decision",
    "rescue_reject_reason",
    "direct_candidate_lambda_jones_per_path",
    "direct_candidate_snr_eff_per_path",
    "direct_candidate_jones_leakage_per_path",
    "rescue_candidate_lambda_jones_per_path",
    "rescue_candidate_snr_eff_per_path",
    "rescue_candidate_jones_leakage_per_path",
    "direct_candidate_data_only_scaled_efim_lambda_min",
    "direct_candidate_data_only_scaled_efim_condition_number",
    "rescue_candidate_data_only_scaled_efim_lambda_min",
    "rescue_candidate_data_only_scaled_efim_condition_number",
    "legacy_stage1_decision",
    "gof_reliability_decision",
    "stage1_geometry_trigger",
    "stage1_geometry_trigger_reasons",
    "ngc_policy_active",
    "ngc_lambda_ris",
    "ngc_direct_clock_score",
    "ngc_direct_clock_score_norm",
    "ngc_direct_clock_dof",
    "ngc_direct_clock_sigma_source",
    "ngc_direct_clock_std_ns",
    "ngc_direct_ris_score",
    "ngc_direct_ris_score_norm",
    "ngc_direct_ris_available",
    "ngc_direct_total_score",
    "ngc_direct_cert_status",
    "ngc_direct_cert_reason",
    "ngc_rescue_requested",
    "ngc_rescue_request_reason",
    "ngc_rescue_clock_score",
    "ngc_rescue_clock_score_norm",
    "ngc_rescue_clock_dof",
    "ngc_rescue_clock_sigma_source",
    "ngc_rescue_clock_std_ns",
    "ngc_rescue_ris_score",
    "ngc_rescue_ris_score_norm",
    "ngc_rescue_ris_available",
    "ngc_rescue_total_score",
    "ngc_rescue_cert_status",
    "ngc_rescue_cert_reason",
    "ngc_selected_by",
    "ngc_final_unreliable",
    "ngc_threshold_clock_green",
    "ngc_threshold_clock_red",
    "proposed_stage2_policy",
]

RESCUE_POLICY_SUMMARY_FIELDS = [
    "variant",
    "snr_db",
    "n",
    "rescue_run_rate",
    "mean_error_over_peb",
    "median_error_over_peb",
    "p90_error_over_peb",
    "max_error_over_peb",
    "outlier_count_5peb",
    "outlier_rate_5peb",
    "direct_vp_count",
    "ris_only_stage2_then_vp_count",
    "direct_vp_rollback_count",
    "jnpp_decision_count",
    "gof_fail_count",
]


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        if lowered == "":
            return None
    numeric = _to_float(value)
    if np.isfinite(numeric):
        return bool(numeric)
    return None


def _has_finite_metadata(row: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(np.isfinite(_to_float(row.get(field))) for field in fields)


def _rescue_policy_valid_key_values(row: dict[str, Any]) -> bool:
    return (
        bool(str(row.get("figure", "")))
        and bool(str(row.get("receiver_mode", "")))
        and np.isfinite(_to_float(row.get("snr_db")))
        and np.isfinite(_to_float(row.get("trial_id")))
        and np.isfinite(_to_float(row.get("seed")))
    )


def _rescue_policy_key(
    row: dict[str, Any],
    *,
    include_k_metadata: bool,
) -> tuple[Any, ...]:
    base = (
        str(row.get("figure", "")),
        float(_to_float(row.get("snr_db"))),
        int(_to_float(row.get("trial_id"))),
        int(_to_float(row.get("seed"))),
        str(row.get("receiver_mode", "")),
    )
    if not include_k_metadata:
        return base
    return (
        *base,
        int(_to_float(row.get("K"))),
        int(_to_float(row.get("paper_k"))),
        int(_to_float(row.get("effective_K"))),
    )


def _write_rescue_policy_ablation_csvs(
    out_dir: pathlib.Path,
    rows: list[dict[str, Any]],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    present_variants = {str(row.get("variant", "")) for row in rows}
    target_variants = RESCUE_POLICY_VARIANTS & present_variants
    if not target_variants or "PEB" not in present_variants:
        return

    peb_by_full_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    peb_by_fallback_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    skipped_peb = 0
    duplicate_peb = 0
    for row in rows:
        if str(row.get("variant", "")) != "PEB":
            continue
        if _to_bool(row.get("failed")) is True:
            continue
        peb = _to_float(row.get("peb_position_m"))
        if not np.isfinite(peb) or peb <= 0.0 or not _rescue_policy_valid_key_values(row):
            skipped_peb += 1
            continue
        fallback_key = _rescue_policy_key(row, include_k_metadata=False)
        if fallback_key in peb_by_fallback_key:
            duplicate_peb += 1
        else:
            peb_by_fallback_key[fallback_key] = row
        if _has_finite_metadata(row, ("K", "paper_k", "effective_K")):
            full_key = _rescue_policy_key(row, include_k_metadata=True)
            if full_key in peb_by_full_key:
                duplicate_peb += 1
            else:
                peb_by_full_key[full_key] = row

    paired_rows: list[dict[str, Any]] = []
    unpaired = 0
    skipped_data = 0
    for row in rows:
        variant = str(row.get("variant", ""))
        if variant not in RESCUE_POLICY_VARIANTS:
            continue
        if _to_bool(row.get("failed")) is True:
            skipped_data += 1
            continue
        position_error = _to_float(row.get("position_error_m"))
        if not np.isfinite(position_error) or not _rescue_policy_valid_key_values(row):
            skipped_data += 1
            continue
        if _has_finite_metadata(row, ("K", "paper_k", "effective_K")):
            peb_row = peb_by_full_key.get(
                _rescue_policy_key(row, include_k_metadata=True)
            )
            if peb_row is None:
                peb_row = peb_by_fallback_key.get(
                    _rescue_policy_key(row, include_k_metadata=False)
                )
        else:
            peb_row = peb_by_fallback_key.get(
                _rescue_policy_key(row, include_k_metadata=False)
            )
        if peb_row is None:
            unpaired += 1
            continue
        peb = _to_float(peb_row.get("peb_position_m"))
        if not np.isfinite(peb) or peb <= 0.0:
            skipped_data += 1
            continue
        error_over_peb = position_error / peb
        paired_row = {
                "figure": row.get("figure"),
                "variant": variant,
                "snr_db": row.get("snr_db"),
                "trial_id": row.get("trial_id"),
                "seed": row.get("seed"),
                "receiver_mode": row.get("receiver_mode"),
                "K": row.get("K"),
                "paper_k": row.get("paper_k"),
                "effective_K": row.get("effective_K"),
                "selected_branch": row.get("selected_branch"),
                "reliability_decision": row.get("reliability_decision"),
                "trigger_reasons": row.get("trigger_reasons"),
                "gof_pass": row.get("gof_pass"),
                "position_error_m": position_error,
                "peb_position_m": peb,
                "error_over_peb": error_over_peb,
                "relative_outlier_5peb": bool(error_over_peb > 5.0),
                "y_nmse": row.get("y_nmse"),
                "raw_objective_final": row.get("raw_objective_final"),
                "direct_candidate_position_error_m": row.get(
                    "direct_candidate_position_error_m"
                ),
                "direct_candidate_y_nmse": row.get("direct_candidate_y_nmse"),
                "direct_candidate_raw_objective": row.get(
                    "direct_candidate_raw_objective"
                ),
                "rescue_candidate_position_error_m": row.get(
                    "rescue_candidate_position_error_m"
                ),
                "rescue_candidate_y_nmse": row.get("rescue_candidate_y_nmse"),
                "rescue_candidate_raw_objective": row.get(
                    "rescue_candidate_raw_objective"
                ),
                "rescue_candidate_available": row.get(
                    "rescue_candidate_available"
                ),
                "rescue_accept_decision": row.get("rescue_accept_decision"),
                "rescue_reject_reason": row.get("rescue_reject_reason"),
                "direct_candidate_lambda_jones_per_path": row.get(
                    "direct_candidate_lambda_jones_per_path"
                ),
                "direct_candidate_snr_eff_per_path": row.get(
                    "direct_candidate_snr_eff_per_path"
                ),
                "direct_candidate_jones_leakage_per_path": row.get(
                    "direct_candidate_jones_leakage_per_path"
                ),
                "rescue_candidate_lambda_jones_per_path": row.get(
                    "rescue_candidate_lambda_jones_per_path"
                ),
                "rescue_candidate_snr_eff_per_path": row.get(
                    "rescue_candidate_snr_eff_per_path"
                ),
                "rescue_candidate_jones_leakage_per_path": row.get(
                    "rescue_candidate_jones_leakage_per_path"
                ),
                "direct_candidate_data_only_scaled_efim_lambda_min": row.get(
                    "direct_candidate_data_only_scaled_efim_lambda_min"
                ),
                "direct_candidate_data_only_scaled_efim_condition_number": row.get(
                    "direct_candidate_data_only_scaled_efim_condition_number"
                ),
                "rescue_candidate_data_only_scaled_efim_lambda_min": row.get(
                    "rescue_candidate_data_only_scaled_efim_lambda_min"
                ),
                "rescue_candidate_data_only_scaled_efim_condition_number": row.get(
                    "rescue_candidate_data_only_scaled_efim_condition_number"
                ),
                "legacy_stage1_decision": row.get("legacy_stage1_decision"),
                "gof_reliability_decision": row.get("gof_reliability_decision"),
                "stage1_geometry_trigger": row.get("stage1_geometry_trigger"),
                "stage1_geometry_trigger_reasons": row.get(
                    "stage1_geometry_trigger_reasons"
                ),
        }
        for field in RESCUE_POLICY_PAIRED_FIELDS:
            if field.startswith("ngc_"):
                paired_row[field] = row.get(field)
        paired_rows.append(paired_row)

    if unpaired or skipped_data or skipped_peb or duplicate_peb:
        print(
            "WARNING rescue_policy_ablation pairing: "
            f"unpaired_rows={unpaired} skipped_data_rows={skipped_data} "
            f"skipped_peb_rows={skipped_peb} duplicate_peb_keys={duplicate_peb}"
        )

    _write_csv(
        out_dir / "rescue_policy_ablation_paired.csv",
        paired_rows,
        RESCUE_POLICY_PAIRED_FIELDS,
    )
    _write_csv(
        out_dir / "rescue_policy_ablation_summary.csv",
        _summarize_rescue_policy_ablation(paired_rows),
        RESCUE_POLICY_SUMMARY_FIELDS,
    )


def _summarize_rescue_policy_ablation(
    paired_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in paired_rows:
        snr_db = _to_float(row.get("snr_db"))
        if not np.isfinite(snr_db):
            continue
        key = (str(row.get("variant", "")), float(snr_db))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (variant, snr_db), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        values = np.asarray(
            [
                _to_float(row.get("error_over_peb"))
                for row in group
                if np.isfinite(_to_float(row.get("error_over_peb")))
            ],
            dtype=float,
        )
        n = int(values.size)
        outlier_count = int(
            sum(_to_bool(row.get("relative_outlier_5peb")) is True for row in group)
        )
        rescue_run_rate = _ngc_rescue_run_rate(group)
        summary_rows.append(
            {
                "variant": variant,
                "snr_db": snr_db,
                "n": n,
                "rescue_run_rate": rescue_run_rate,
                "mean_error_over_peb": float(np.mean(values)) if n else float("nan"),
                "median_error_over_peb": float(np.median(values)) if n else float("nan"),
                "p90_error_over_peb": float(np.percentile(values, 90)) if n else float("nan"),
                "max_error_over_peb": float(np.max(values)) if n else float("nan"),
                "outlier_count_5peb": outlier_count,
                "outlier_rate_5peb": float(outlier_count / n) if n else float("nan"),
                "direct_vp_count": sum(
                    str(row.get("selected_branch", "")) == "direct_vp"
                    for row in group
                ),
                "ris_only_stage2_then_vp_count": sum(
                    str(row.get("selected_branch", ""))
                    == "ris_only_stage2_then_vp"
                    for row in group
                ),
                "direct_vp_rollback_count": sum(
                    str(row.get("selected_branch", "")) == "direct_vp_rollback"
                    for row in group
                ),
                "jnpp_decision_count": sum(
                    str(row.get("reliability_decision", "")) == "jnpp_then_vp"
                    for row in group
                ),
                "gof_fail_count": sum(
                    _to_bool(row.get("gof_pass")) is False for row in group
                ),
            }
        )
    return summary_rows


def _finite_metric_values(group: list[dict[str, Any]], metric: str) -> np.ndarray:
    values = np.asarray(
        [
            _to_float(row.get(metric))
            for row in group
            if str(row.get("failed")) != "True"
        ],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _metric_available(group: list[dict[str, Any]], metric: str) -> bool:
    return bool(_finite_metric_values(group, metric).size)


def get_plot_metric(row_or_group: dict[str, Any] | list[dict[str, Any]], figure: str, variant: str) -> str | None:
    """Return the source metric column to summarize and plot for a variant."""
    group = row_or_group if isinstance(row_or_group, list) else [row_or_group]
    if figure == "fig1":
        return "peb_position_m" if "peb" in variant.lower() else "position_rmse_m"
    if figure == "fig2":
        return None if "peb" in variant.lower() else "y_nmse"
    if figure == "fig3":
        return "peb_position_m" if "peb" in variant.lower() else "position_rmse_m"
    if figure == "fig4":
        preferred = {
            "scalar_peb": "peb_scalar_m",
            "dual_pol_peb": "peb_dual_m",
            "full_6d_evs_peb": "peb_evs_m",
            "full_6d_constrained_jones_peb": "peb_evs_m",
        }.get(variant, "peb_position_m")
        return preferred if _metric_available(group, preferred) else "peb_position_m"
    if figure == "fig5":
        return "outlier_flag"
    if figure == "fig6":
        return "peb_position_m" if "peb" in variant.lower() else "position_rmse_m"
    raise ValueError(f"unknown figure {figure!r}")


def _plot_metric_name(metric: str) -> str:
    return "outlier_flag_mean" if metric == "outlier_flag" else metric


def _variant_linestyle(variant: str) -> str:
    lower = str(variant).lower()
    if "constrained_jones_peb" in lower:
        return "-."
    if "anchored_jones_peb" in lower:
        return ":"
    if "peb" in lower:
        return "--"
    return "-"


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p10": float(np.percentile(values, 10.0)),
            "p90": float(np.percentile(values, 90.0)),
        }
    return {name: float("nan") for name in ("mean", "median", "std", "p10", "p90")}


def _ngc_rescue_run_rate(rows: list[dict[str, Any]]) -> float:
    active_rows = [
        row for row in rows if _to_bool(row.get("ngc_policy_active")) is True
    ]
    if not active_rows:
        return float("nan")
    requested = sum(
        _to_bool(row.get("ngc_rescue_requested")) is True for row in active_rows
    )
    return float(requested / len(active_rows))


def _rescue_trigger_rate(rows: list[dict[str, Any]]) -> float:
    valid_rows = [
        row for row in rows if _to_bool(row.get("failed")) is not True
    ]
    if not valid_rows:
        return float("nan")

    rescue_branches = {
        "ris_only_stage2_then_vp",
        "multi_hypothesis_ris_reacquisition_then_vp",
        "direct_vp_rollback",
    }
    triggered = 0
    for row in valid_rows:
        if _to_bool(row.get("ngc_policy_active")) is True:
            was_triggered = _to_bool(row.get("ngc_rescue_requested")) is True
        elif str(row.get("proposed_stage2_policy", "")) == "force_ris_only":
            was_triggered = True
        else:
            candidate_available = _to_bool(row.get("rescue_candidate_available"))
            if candidate_available is not None:
                was_triggered = candidate_available
            else:
                selected_branch = str(row.get("selected_branch", ""))
                was_triggered = (
                    selected_branch in rescue_branches
                    or "stage2" in selected_branch
                    or "rescue" in selected_branch
                )
        triggered += int(was_triggered)
    return float(triggered / len(valid_rows))


def summarize_rows(rows: list[dict[str, Any]], figure: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["variant"]), _to_float(row["x_value"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (variant, x_value), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        metric = get_plot_metric(group, figure, variant)
        if metric is None:
            continue
        values = _finite_metric_values(group, metric)
        failed_count = sum(str(row.get("failed")) == "True" for row in group)
        outliers = np.asarray([str(row.get("outlier_flag")) == "True" for row in group], dtype=float)
        stats = _summary_stats(values)
        row_summary = {
            "figure": figure,
            "variant": variant,
            "x_value": x_value,
            "metric": metric,
            "plot_metric_name": _plot_metric_name(metric),
            "plot_y_mean": stats["mean"],
            "plot_y_median": stats["median"],
            "plot_y_std": stats["std"],
            "plot_y_p10": stats["p10"],
            "plot_y_p90": stats["p90"],
            **stats,
            "success_rate": float((len(group) - failed_count) / max(len(group), 1)),
            "outlier_rate": float(np.mean(outliers)) if outliers.size else float("nan"),
            "rescue_run_rate": _ngc_rescue_run_rate(group),
            "rescue_trigger_rate": _rescue_trigger_rate(group),
            "n": len(group),
        }
        for metadata_field in ("K", "paper_k", "effective_K"):
            unique_values = {
                int(value)
                for row in group
                if np.isfinite(value := _to_float(row.get(metadata_field)))
            }
            if len(unique_values) == 1:
                row_summary[metadata_field] = unique_values.pop()
        for raw_metric in RAW_SUMMARY_METRICS:
            raw_stats = _summary_stats(_finite_metric_values(group, raw_metric))
            for name, value in raw_stats.items():
                row_summary[f"{raw_metric}_{name}"] = value
        summary.append(row_summary)
    return summary


def summarize_fig1_fig2_shared_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shared_summary: list[dict[str, Any]] = []
    for plot_name in ("fig1", "fig2"):
        for row in summarize_rows(rows, plot_name):
            shared_summary.append({"plot_name": plot_name, **row})
    return shared_summary


def _plot_figure(figure: str, summary_rows: list[dict[str, Any]], out_dir: pathlib.Path) -> None:
    mpl_config = out_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib.pyplot as plt

    metric = FIGURE_METRICS[figure]
    xlabel = "K" if figure == "fig6" else "SNR (dB)"
    markers = ["o", "s", "^", "D", "v", "P"]
    variants = list(dict.fromkeys(row["variant"] for row in summary_rows))
    proposed_variant = "adaptive_jones_vp_proposed"
    if figure == "fig5":
        preferred = [
            "direct_vp",
            "old_gated",
            proposed_variant,
            "force_rescue",
            "oracle_init_vp",
        ]
        variants = [
            variant for variant in preferred if variant in variants
        ] + [
            variant for variant in variants if variant not in preferred
        ]
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
        panels = [
            (axes[0], "plot_y_mean", "Outlier probability"),
            (axes[1], "rescue_trigger_rate", "Rescue trigger rate"),
        ]
        for ax, field, ylabel in panels:
            for idx, variant in enumerate(variants):
                rows = [row for row in summary_rows if row["variant"] == variant]
                xs = np.asarray([_to_float(row["x_value"]) for row in rows], dtype=float)
                ys = np.asarray([_to_float(row.get(field)) for row in rows], dtype=float)
                finite = np.isfinite(xs) & np.isfinite(ys)
                if not np.any(finite):
                    continue
                order = np.argsort(xs[finite])
                is_proposed = variant == proposed_variant
                ax.plot(
                    xs[finite][order],
                    ys[finite][order],
                    marker=markers[idx % len(markers)],
                    linestyle=_variant_linestyle(variant),
                    linewidth=2.5 if is_proposed else 1.5,
                    label=VARIANT_LABELS.get(variant, variant),
                    zorder=10 if is_proposed else 2,
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=3)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        else:
            fig.tight_layout()
        fig.savefig(out_dir / FIGURE_PDFS[figure])
        plt.close(fig)
        return

    if figure == "fig6":
        if proposed_variant in variants:
            variants = [
                variant for variant in variants if variant != proposed_variant
            ] + [proposed_variant]
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
        panels = [
            (axes[0], "plot_y_median", "Median position error / PEB (m)", True),
            (axes[1], "outlier_rate", "Outlier probability", False),
        ]
        for ax, field, ylabel, log_y in panels:
            for idx, variant in enumerate(variants):
                if field == "outlier_rate" and variant == "proposed_peb":
                    continue
                rows = [row for row in summary_rows if row["variant"] == variant]
                xs = np.asarray([_to_float(row["x_value"]) for row in rows], dtype=float)
                ys = np.asarray([_to_float(row.get(field)) for row in rows], dtype=float)
                finite = np.isfinite(xs) & np.isfinite(ys)
                if not np.any(finite):
                    continue
                order = np.argsort(xs[finite])
                is_proposed = variant == proposed_variant
                ax.plot(
                    xs[finite][order],
                    ys[finite][order],
                    marker=markers[idx % len(markers)],
                    linestyle=_variant_linestyle(variant),
                    linewidth=2.5 if is_proposed else 1.5,
                    label=VARIANT_LABELS.get(variant, variant),
                    zorder=10 if is_proposed else 2,
                )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if log_y:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.02, 1.02)
            ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=3)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        else:
            fig.tight_layout()
        fig.savefig(out_dir / FIGURE_PDFS[figure])
        plt.close(fig)
        return

    ylabel = {
        "position_rmse_m": "Position RMSE (m)",
        "y_nmse": "Channel NMSE",
        "peb_position_m": "PEB (m)",
        "outlier_flag": "Outlier probability",
    }[metric]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if proposed_variant in variants:
        variants = [
            variant for variant in variants if variant != proposed_variant
        ] + [proposed_variant]
    for idx, variant in enumerate(variants):
        rows = [row for row in summary_rows if row["variant"] == variant]
        xs = np.asarray([_to_float(row["x_value"]) for row in rows], dtype=float)
        ys = np.asarray([_to_float(row["plot_y_mean"]) for row in rows], dtype=float)
        order = np.argsort(xs)
        is_proposed = variant == proposed_variant
        ax.plot(
            xs[order],
            ys[order],
            marker=markers[idx % len(markers)],
            linestyle=_variant_linestyle(variant),
            linewidth=2.5 if is_proposed else 1.5,
            label=VARIANT_LABELS.get(variant, variant),
            zorder=10 if is_proposed else 2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if figure == "fig5":
        ax.set_ylim(-0.02, 1.02)
    else:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / FIGURE_PDFS[figure])
    plt.close(fig)


def _write_duplicate_curves_report(
    figure: str,
    summary_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    out_dir: pathlib.Path,
) -> dict[str, Any]:
    proposed_variant = "adaptive_jones_vp_proposed"
    proposed = sorted(
        [row for row in summary_rows if row.get("variant") == proposed_variant],
        key=lambda row: _to_float(row.get("x_value")),
    )
    duplicates = []
    for variant in sorted(
        {
            str(row.get("variant"))
            for row in summary_rows
            if row.get("variant") != proposed_variant
        }
    ):
        candidate = sorted(
            [row for row in summary_rows if row.get("variant") == variant],
            key=lambda row: _to_float(row.get("x_value")),
        )
        if len(candidate) != len(proposed) or not candidate:
            continue
        proposed_points = [
            (_to_float(row.get("x_value")), _to_float(row.get("plot_y_mean")))
            for row in proposed
        ]
        candidate_points = [
            (_to_float(row.get("x_value")), _to_float(row.get("plot_y_mean")))
            for row in candidate
        ]
        if proposed_points == candidate_points:
            duplicates.append(variant)
    proposed_trials = [
        row for row in trial_rows if row.get("variant") == proposed_variant
    ]
    report = {
        "figure": figure,
        "proposed_variant": proposed_variant,
        "exact_duplicate_variants": duplicates,
        "overlap_explanation": (
            "Adaptive policy selected the fixed-pol anchor for every reported point."
            if duplicates
            and all(
                str(row.get("selected_vp_family_branch")) == "fixed_pol_anchor"
                for row in proposed_trials
            )
            else "No exact duplicate curve detected."
            if not duplicates
            else "Exact overlap detected; inspect per-trial dispatch diagnostics."
        ),
        "lambda_diagnostics": [
            {
                "trial_id": row.get("trial_id"),
                "seed": row.get("seed"),
                "snr_db": row.get("snr_db"),
                "lambda_path_min": row.get("lambda_path_min"),
                "lambda_path_max": row.get("lambda_path_max"),
                "lambda_path_mean": row.get("lambda_path_mean"),
                "lambda_clipped_fraction": row.get("lambda_clipped_fraction"),
                "selected_vp_family_branch": row.get(
                    "selected_vp_family_branch"
                ),
                "variant_config_hash": row.get("variant_config_hash"),
            }
            for row in proposed_trials
        ],
    }
    (out_dir / "duplicate_curves_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    return report


def _cache_signature(args: argparse.Namespace, snr_grid: list[float], figures: list[str]) -> dict[str, Any]:
    global_vp_overrides = global_vp_cli_overrides(args)
    signature = {
        "n_trials": int(args.n_trials),
        "snr_grid": [float(value) for value in snr_grid],
        "paper_k": int(args.paper_k),
        "k_grid": [int(value) for value in args.k_grid_values],
        "figures": list(figures),
        "seed": int(args.seed),
        "variant_list": {
            figure: _expected_variant_names(
                figure,
                getattr(args, "variant_filter_values", None),
                include_diagnostic_variants=bool(
                    getattr(args, "include_diagnostic_variants", False)
                ),
            )
            for figure in figures
        },
        "include_diagnostic_variants": bool(
            getattr(args, "include_diagnostic_variants", False)
        ),
        "git_commit": _git_commit_hash(),
        "receiver_noise_convention": RECEIVER_NOISE_CONVENTION,
        "receiver_mode_convention": RECEIVER_MODE_CONVENTION,
        "debug_compare_main_single_proposed": bool(
            getattr(args, "debug_compare_main_single_proposed", False)
        ),
        "global_vp_backend": global_vp_overrides.get("backend", "cpu"),
        "global_vp_gpu_device": global_vp_overrides.get("gpu_device", 0),
        "global_vp_validate_gpu_against_cpu": bool(
            global_vp_overrides.get("validate_gpu_against_cpu", False)
        ),
        "fig1_proposed_dispatch_version": 2,
        "include_constrained_jones_peb": bool(
            getattr(args, "include_constrained_jones_peb", True)
        ),
        "include_anchored_jones_peb": bool(
            getattr(args, "include_anchored_jones_peb", False)
        ),
        "jones_anchor_prior_mode": str(
            getattr(args, "jones_anchor_prior_mode", "disabled")
        ),
        "jones_anchor_prior_scale": float(
            getattr(args, "jones_anchor_prior_scale", 1.0)
        ),
    }
    if global_vp_overrides:
        signature["global_vp_overrides"] = global_vp_overrides
    if any(figure in {"fig3", "fig4"} for figure in figures):
        signature["nested_receiver_noise_convention"] = (
            NESTED_RECEIVER_NOISE_CONVENTION
        )
    return signature


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _metadata(args: argparse.Namespace, snr_grid: list[float], figures: list[str]) -> dict[str, Any]:
    commit = _git_commit_hash()
    timestamp = datetime.now(timezone.utc).isoformat()
    global_vp_overrides = global_vp_cli_overrides(args)
    metadata = {
        "git_commit": commit,
        "timestamp": timestamp,
        "timestamp_utc": timestamp,
        "command_line": " ".join(sys.argv),
        "n_trials": int(args.n_trials),
        "jobs": int(args.jobs),
        "process_workers": int(args.process_workers),
        "max_workers": int(args.process_workers),
        "maxtasksperchild": int(args.maxtasksperchild),
        "task_grouping": str(args.task_grouping),
        "streaming_csv": bool(args.streaming_csv),
        "csv_flush_every": int(args.csv_flush_every),
        "store_large_arrays": bool(args.store_large_arrays),
        "blas_threads": int(args.blas_threads),
        "estimated_cpu_slots": int(args.resource_plan["estimated_cpu_slots"]),
        "memory_budget_gb": args.memory_budget_gb,
        "memory_per_worker_gb": args.memory_per_worker_gb,
        "trim_memory": bool(args.trim_memory),
        "respect_existing_blas_env": bool(args.respect_existing_blas_env),
        "profile_memory": bool(args.profile_memory),
        "snr_grid": snr_grid,
        "paper_k": int(args.paper_k),
        "k_grid": [int(value) for value in args.k_grid_values],
        "figures": figures,
        "fig6_interpretation": (
            "complete_ngc_proposed_system_vs_vp_only_polarization_variants;"
            "adaptive_jones_no_rescue isolates adaptive Jones VP without rescue"
        ),
        "seed": int(args.seed),
        "include_diagnostic_variants": bool(
            getattr(args, "include_diagnostic_variants", False)
        ),
        "diagnostic_variants": _enabled_diagnostic_variant_names(args, figures),
        "global_vp_backend": global_vp_overrides.get("backend", "cpu"),
        "global_vp_gpu_device": global_vp_overrides.get("gpu_device", 0),
        "global_vp_validate_gpu_against_cpu": bool(
            global_vp_overrides.get("validate_gpu_against_cpu", False)
        ),
        "receiver_noise_convention": RECEIVER_NOISE_CONVENTION,
        "receiver_mode_convention": RECEIVER_MODE_CONVENTION,
        "config_overrides": {
            "seed": int(args.seed),
            "outlier_threshold_m": float(args.outlier_threshold_m),
            "paper_k_for_fig1_to_fig5": int(args.paper_k),
            "k_grid_for_fig6": [int(value) for value in args.k_grid_values],
            "global_vp": global_vp_overrides,
            "crb": {
                "include_constrained_jones_peb": bool(
                    getattr(args, "include_constrained_jones_peb", True)
                ),
                "include_anchored_jones_peb": bool(
                    getattr(args, "include_anchored_jones_peb", False)
                ),
                "jones_anchor_prior_mode": str(
                    getattr(args, "jones_anchor_prior_mode", "disabled")
                ),
                "jones_anchor_prior_scale": float(
                    getattr(args, "jones_anchor_prior_scale", 1.0)
                ),
            },
        },
        "shared_cache_signatures": {
            FIG1_FIG2_SHARED_FIGURE: _cache_signature(
                args, snr_grid, [FIG1_FIG2_SHARED_FIGURE]
            )
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy_available": bool(scipy_is_available()),
    }
    if args.variant_filter_values is not None:
        metadata["variant_filter"] = list(args.variant_filter_values)
    if any(figure in {"fig3", "fig4"} for figure in figures):
        metadata["nested_receiver_noise_convention"] = (
            NESTED_RECEIVER_NOISE_CONVENTION
        )
        metadata["nested_receiver_reference_mode"] = "full_6d"
        metadata["snr_interpretation_fig3_fig4"] = "full_6d_reference_snr"
    return metadata


def _read_metadata(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _metadata_matches_request(
    metadata: dict[str, Any] | None,
    args: argparse.Namespace,
    snr_grid: list[float],
    figures: list[str],
) -> bool:
    if not metadata:
        return False
    expected = _cache_signature(args, snr_grid, figures)
    if figures == [FIG1_FIG2_SHARED_FIGURE]:
        shared = metadata.get("shared_cache_signatures", {})
        if shared.get(FIG1_FIG2_SHARED_FIGURE) == expected:
            return True
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            return False
    return True


def _expected_variant_names(
    figure: str,
    variant_filter: tuple[str, ...] | None = None,
    *,
    include_diagnostic_variants: bool = False,
) -> list[str]:
    return list(
        _variants_for_figure(
            figure,
            variant_filter,
            include_diagnostic_variants=include_diagnostic_variants,
        )
    )


def _csv_matches_request(
    rows: list[dict[str, Any]],
    figure: str,
    args: argparse.Namespace,
    snr_grid: list[float],
) -> bool:
    if not rows:
        return False
    if not _rows_have_k_cache_columns(rows):
        return False
    canonical_figure = FIG1_FIG2_SHARED_FIGURE if _is_fig1_fig2(figure) else figure
    x_name, x_values = _figure_x_grid(canonical_figure, snr_grid, args.k_grid_values)
    expected_variants = set(
        _expected_variant_names(
            figure,
            getattr(args, "variant_filter_values", None),
            include_diagnostic_variants=bool(
                getattr(args, "include_diagnostic_variants", False)
            ),
        )
    )
    expected_x_values = {float(value) for value in x_values}
    groups: dict[tuple[str, float], int] = {}
    for row in rows:
        row_figure = row.get("figure")
        if _is_fig1_fig2(figure):
            if row_figure not in {"fig1", "fig2", FIG1_FIG2_SHARED_FIGURE}:
                return False
        elif row_figure != figure:
            return False
        variant = str(row.get("variant"))
        x_value = _to_float(row.get("x_value"))
        if variant not in expected_variants or x_value not in expected_x_values:
            return False
        row_k_value = _to_float(row.get("K"))
        effective_k_value = _to_float(row.get("effective_K"))
        paper_k_value = _to_float(row.get("paper_k"))
        row_k = int(row_k_value) if np.isfinite(row_k_value) else None
        effective_k = (
            int(effective_k_value) if np.isfinite(effective_k_value) else None
        )
        row_paper_k = (
            int(paper_k_value) if np.isfinite(paper_k_value) else None
        )
        expected_k = int(x_value) if canonical_figure == "fig6" else int(args.paper_k)
        if (
            row_k != expected_k
            or effective_k != expected_k
            or row_paper_k != int(args.paper_k)
        ):
            return False
        if row.get("x_name") != x_name:
            return False
        groups[(variant, x_value)] = groups.get((variant, x_value), 0) + 1
    expected_groups = {
        (variant, x_value)
        for variant in expected_variants
        for x_value in expected_x_values
    }
    return set(groups) == expected_groups and all(
        count == int(args.n_trials) for count in groups.values()
    )


def _rows_have_k_cache_columns(rows: list[dict[str, Any]]) -> bool:
    required = {"K", "paper_k", "effective_K"}
    return bool(rows) and required.issubset(rows[0])


def _can_reuse_csv(
    trial_csv: pathlib.Path,
    figure: str,
    args: argparse.Namespace,
    snr_grid: list[float],
    figures: list[str],
    existing_metadata: dict[str, Any] | None,
) -> tuple[bool, list[dict[str, Any]]]:
    if not trial_csv.exists() or args.force_rerun:
        return False, []
    rows = _read_csv(trial_csv)
    if not _rows_have_k_cache_columns(rows):
        print(
            f"{trial_csv}: stale cache missing K/paper_k/effective_K columns; "
            "recomputing trials"
        )
        return False, rows
    if args.reuse_existing:
        return True, rows
    metadata_ok = _metadata_matches_request(existing_metadata, args, snr_grid, figures)
    csv_ok = _csv_matches_request(rows, figure, args, snr_grid)
    return bool(metadata_ok and csv_ok), rows


def _figure_x_grid(figure: str, snr_grid: list[float], k_grid: list[int]) -> tuple[str, list[float]]:
    if figure == "fig6":
        return "K", [float(k) for k in k_grid]
    return "snr_db", snr_grid


def _config_for_point(
    *,
    figure: str,
    variant_updates: dict[str, Any],
    seed: int,
    snr_db: float,
    x_value: float,
    paper_k: int,
) -> dict:
    overrides = copy.deepcopy(variant_updates)
    if figure == "fig6":
        overrides["K"] = int(x_value)
        snr_db = 0.0
    else:
        overrides["K"] = int(paper_k)
    return make_base_config(seed, snr_db, overrides)


def _failure_row_from_task(task: dict[str, Any], exc: BaseException) -> tuple[list[dict[str, Any]], str]:
    row = _empty_row()
    row.update(
        {
            "figure": task["figure"],
            "variant": task["variant"],
            "trial_id": int(task["trial_id"]),
            "seed": int(task["trial_seed"]),
            "snr_db": float(task["snr_db"]),
            "x_name": task["x_name"],
            "x_value": float(task["x_value"]),
            "failed": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    effective_k = (
        int(task["x_value"])
        if task["figure"] == "fig6"
        else int(task["paper_k"])
    )
    _set_row_k_metadata(
        row,
        figure=str(task["figure"]),
        effective_k=effective_k,
        paper_k=int(task["paper_k"]),
        x_value=float(task["x_value"]),
        receiver_mode=str(task.get("receiver_mode", "full_6d")),
        config_seed=int(task["trial_seed"]),
        num_ris_paths=task.get("num_ris_paths", ""),
    )
    return _ensure_worker_result_rows_safe([row]), f"\nERROR {type(exc).__name__}: {exc}\n"


def _run_trial_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    try:
        k_paths = int(task["x_value"]) if task["figure"] == "fig6" else int(task["paper_k"])
        config = _config_from_base(
            figure=task["figure"],
            seed=task["trial_seed"],
            snr_db=task["snr_db"],
            k_paths=k_paths,
            updates=task["updates"],
        )
        row, log_text = run_one_trial(
            config,
            task["trial_seed"],
            task["variant"],
            task["figure"],
            trial_id=task["trial_id"],
            x_name=task["x_name"],
            x_value=task["x_value"],
            paper_k=task["paper_k"],
            outlier_threshold_m=task["outlier_threshold_m"],
            verbose=task["verbose"],
            runner=task["runner"],
            allow_stage2=task["allow_stage2"],
            store_large_arrays=task["store_large_arrays"],
            profile_memory=task["profile_memory"],
            blas_threads=task["blas_threads"],
            respect_existing_blas_env=bool(task.get("respect_existing_blas_env", False)),
            trim_memory_enabled=bool(task.get("trim_memory", True)),
            out_dir=pathlib.Path(task["out_dir"]) if task.get("out_dir") else None,
        )
    except Exception as exc:  # noqa: BLE001 - preserve failed-trial logging.
        return _failure_row_from_task(task, exc)
    return _ensure_worker_result_rows_safe([row]), log_text


def _trial_tasks(
    *,
    figure: str,
    x_name: str,
    x_values: list[float],
    variants: dict[str, dict[str, Any]],
    trial_seeds: list[int],
    outlier_threshold_m: float,
    verbose: bool,
    paper_k: int,
    store_large_arrays: bool,
    profile_memory: bool,
    blas_threads: int,
    respect_existing_blas_env: bool,
    trim_memory_enabled: bool,
    out_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    tasks = []
    for x_value in x_values:
        snr_db = float(x_value) if x_name == "snr_db" else 0.0
        effective_k = int(x_value) if figure == "fig6" else int(paper_k)
        for variant, updates in variants.items():
            for trial_id, trial_seed in enumerate(trial_seeds):
                tasks.append(
                    {
                        "figure": figure,
                        "variant": variant,
                        "updates": updates,
                        "trial_id": trial_id,
                        "trial_seed": int(trial_seed),
                        "snr_db": snr_db,
                        "x_name": x_name,
                        "x_value": float(x_value),
                        "K": effective_k,
                        "paper_k": int(paper_k),
                        "effective_K": effective_k,
                        "num_ris_paths": "",
                        "receiver_mode": str(
                            updates.get("receiver_mode", "full_6d")
                        ),
                        "config_seed": int(trial_seed),
                        "outlier_threshold_m": float(outlier_threshold_m),
                        "verbose": bool(verbose),
                        "runner": str(updates.get("_runner", "proposed")),
                        "allow_stage2": bool(updates.get("_allow_stage2", True)),
                        "store_large_arrays": bool(store_large_arrays),
                        "profile_memory": bool(profile_memory),
                        "blas_threads": int(blas_threads),
                        "respect_existing_blas_env": bool(respect_existing_blas_env),
                        "trim_memory": bool(trim_memory_enabled),
                        "out_dir": str(out_dir),
                    }
                )
    return tasks


def _grouped_tasks(
    *,
    figure: str,
    group: str,
    x_name: str,
    x_values: list[float],
    variants: dict[str, dict[str, Any]],
    trial_seeds: list[int],
    outlier_threshold_m: float,
    paper_k: int,
    store_large_arrays: bool,
    profile_memory: bool,
    blas_threads: int,
    respect_existing_blas_env: bool,
    trim_memory_enabled: bool,
    out_dir: pathlib.Path,
    debug_compare_main_single_proposed: bool = False,
    include_diagnostic_variants: bool = False,
) -> list[dict[str, Any]]:
    tasks = []
    for x_value in x_values:
        snr_db = float(x_value) if x_name == "snr_db" else 0.0
        k_paths = int(x_value) if figure == "fig6" else int(paper_k)
        for trial_id, trial_seed in enumerate(trial_seeds):
            tasks.append(
                {
                    "task_kind": "grouped",
                    "figure": figure,
                    "group": group,
                    "selected_variants": list(variants),
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "snr_db": float(snr_db),
                    "x_name": x_name,
                    "x_value": float(x_value),
                    "K": int(k_paths),
                    "paper_k": int(paper_k),
                    "effective_K": int(k_paths),
                    "config_seed": int(trial_seed),
                    "outlier_threshold_m": float(outlier_threshold_m),
                    "store_large_arrays": bool(store_large_arrays),
                    "profile_memory": bool(profile_memory),
                    "blas_threads": int(blas_threads),
                    "respect_existing_blas_env": bool(respect_existing_blas_env),
                    "trim_memory": bool(trim_memory_enabled),
                    "out_dir": str(out_dir),
                    "debug_compare_main_single_proposed": bool(
                        debug_compare_main_single_proposed
                    ),
                    "include_diagnostic_variants": bool(
                        include_diagnostic_variants
                    ),
                }
            )
    return tasks


def _tasks_for_figure(
    *,
    figure: str,
    x_name: str,
    x_values: list[float],
    variants: dict[str, dict[str, Any]],
    trial_seeds: list[int],
    args: argparse.Namespace,
    grouped_group: str | None = None,
) -> list[dict[str, Any]]:
    task_blas_threads = (
        int(args.resource_plan["blas_threads"])
        if hasattr(args, "resource_plan")
        else (1 if str(args.blas_threads).lower() == "auto" else int(args.blas_threads))
    )
    if args.task_grouping == "grouped" and grouped_group is not None:
        return _grouped_tasks(
            figure=figure,
            group=grouped_group,
            x_name=x_name,
            x_values=x_values,
            variants=variants,
            trial_seeds=trial_seeds,
            outlier_threshold_m=float(args.outlier_threshold_m),
            paper_k=int(args.paper_k),
            store_large_arrays=bool(args.store_large_arrays),
            profile_memory=bool(args.profile_memory),
            blas_threads=task_blas_threads,
            respect_existing_blas_env=bool(args.respect_existing_blas_env),
            trim_memory_enabled=bool(args.trim_memory),
            out_dir=pathlib.Path(args.out_dir),
            debug_compare_main_single_proposed=bool(
                getattr(args, "debug_compare_main_single_proposed", False)
            ),
            include_diagnostic_variants=bool(
                getattr(args, "include_diagnostic_variants", False)
            ),
        )
    tasks = _trial_tasks(
        figure=figure,
        x_name=x_name,
        x_values=x_values,
        variants=variants,
        trial_seeds=trial_seeds,
        outlier_threshold_m=float(args.outlier_threshold_m),
        verbose=bool(args.verbose),
        paper_k=int(args.paper_k),
        store_large_arrays=bool(args.store_large_arrays),
        profile_memory=bool(args.profile_memory),
        blas_threads=task_blas_threads,
        respect_existing_blas_env=bool(args.respect_existing_blas_env),
        trim_memory_enabled=bool(args.trim_memory),
        out_dir=pathlib.Path(args.out_dir),
    )
    return tasks


def _print_task_summary(
    tasks: list[dict[str, Any]],
    variants: dict[str, dict[str, Any]],
) -> None:
    print(
        f"Task summary: execution_tasks={len(tasks)} "
        f"variant_runs={sum(1 for task in tasks for _ in task.get('selected_variants', [task.get('variant')]))}"
    )
    snr_values = sorted({float(task["snr_db"]) for task in tasks})
    trial_ids = sorted({int(task["trial_id"]) for task in tasks})
    snr_text = ",".join(f"{value:g}" for value in snr_values)
    trial_text = ",".join(str(value) for value in trial_ids)
    for variant, spec in variants.items():
        receiver_mode = str(spec.get("receiver_mode", "full_6d"))
        print(
            f"  variant={variant} receiver_mode={receiver_mode} "
            f"snr_db=[{snr_text}] trial_id=[{trial_text}] "
            f"proposed_stage2_policy={spec.get('proposed_stage2_policy', '')} "
            f"ngc_lambda_ris={spec.get('ngc_lambda_ris', 1.0)} "
            f"ngc_clock_green_quantile={spec.get('ngc_clock_green_quantile', 0.99)} "
            f"ngc_clock_red_quantile={spec.get('ngc_clock_red_quantile', 0.999)} "
            "rescue_accept_min_rel_improvement="
            f"{spec.get('rescue_accept_min_rel_improvement', '')} "
            "rescue_accept_min_abs_improvement="
            f"{spec.get('rescue_accept_min_abs_improvement', '')}"
        )


def _init_worker(
    base_config: dict,
    out_dir: str,
    blas_threads: int,
    respect_existing_blas_env: bool = False,
    trim_memory_enabled: bool = True,
    peb_cache_enabled: bool = True,
) -> None:
    global _WORKER_BASE_CONFIG, _WORKER_OUT_DIR, _WORKER_BLAS_THREADS
    global _WORKER_RESPECT_EXISTING_BLAS_ENV, _WORKER_TRIM_MEMORY
    _WORKER_BASE_CONFIG = base_config
    _WORKER_OUT_DIR = pathlib.Path(out_dir)
    _WORKER_BLAS_THREADS = int(blas_threads)
    _WORKER_RESPECT_EXISTING_BLAS_ENV = bool(respect_existing_blas_env)
    _WORKER_TRIM_MEMORY = bool(trim_memory_enabled)
    _apply_blas_thread_env(
        _WORKER_BLAS_THREADS,
        respect_existing_blas_env=_WORKER_RESPECT_EXISTING_BLAS_ENV,
    )
    cache_path = _peb_cache_path(_WORKER_OUT_DIR)
    if cache_path is not None:
        path_key = _peb_cache_path_key(cache_path)
        if peb_cache_enabled:
            _PEB_CACHE_DISABLED_PATHS.discard(path_key)
        else:
            _PEB_CACHE_DISABLED_PATHS.add(path_key)


def _base_worker_config() -> dict:
    if _WORKER_BASE_CONFIG is not None:
        return copy.deepcopy(_WORKER_BASE_CONFIG)
    return default_config()


def _config_from_base(
    *,
    figure: str,
    seed: int,
    snr_db: float,
    k_paths: int,
    updates: dict[str, Any] | None = None,
) -> dict:
    config = copy.deepcopy(_base_worker_config())
    config["seed"] = int(seed)
    config["SNR_dB"] = float(0.0 if figure == "fig6" else snr_db)
    config["print_progress"] = False
    config["verbose_stage2"] = False
    config["run_full_legacy_comparison"] = False
    config = apply_nested_update(config, updates or {})
    set_number_of_ris_paths(config, int(k_paths))
    return _apply_main_single_defaults(config)


def _worker_row_value(value: Any, *, row_index: int, key: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.size > WORKER_ROW_ARRAY_JSON_THRESHOLD:
            raise ValueError(
                f"worker row {row_index} field {key!r} contains ndarray with "
                f"{value.size} elements"
            )
        return _vector_string(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ensure_worker_result_rows_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [assert_row_is_light(row) for row in rows]


def _peb_metrics_result_for_config(
    config: dict,
    out_dir: pathlib.Path | None,
    data: dict | None = None,
) -> dict:
    key = peb_cache_key(config)
    cached = _PEB_CACHE.get(key)
    if cached is None:
        cached = _read_persistent_peb_cache(config, out_dir)
        if cached is not None:
            _PEB_CACHE[key] = copy.deepcopy(cached)
    if cached is None:
        if data is None:
            result = _peb_result(config, out_dir)
            return result
        cached = _peb_from_efim(data, config)
        _PEB_CACHE[key] = copy.deepcopy(cached)
        _write_persistent_peb_cache(config, out_dir, cached)
    return {
        "scene": {
            "K": int(config.get("K", "")),
            "receiver_mode": str(config.get("receiver_mode", config.get("evs_selection", "full_6d"))),
        },
        "nested_receiver_noise_convention": (
            data.get("nested_receiver_noise_convention", "")
            if data is not None
            else ""
        ),
        "reference_receiver_mode": (
            data.get("reference_receiver_mode", "") if data is not None else ""
        ),
        "reference_sigma2": (
            data.get("reference_sigma2", float("nan"))
            if data is not None
            else float("nan")
        ),
        "nested_base_y_noisy_hash": (
            data.get("nested_base_y_noisy_hash", "")
            if data is not None
            else ""
        ),
        **copy.deepcopy(cached),
        "final": {},
        "timing": {},
    }


def _peb_bound_metrics_result_for_config(
    config: dict,
    out_dir: pathlib.Path | None,
    data: dict | None,
    variant: str,
) -> dict:
    result = _peb_metrics_result_for_config(config, out_dir, data)
    return _apply_peb_bound_variant(result, variant)


def _row_for_result_or_failure(
    *,
    result_factory,
    figure: str,
    variant: str,
    trial_id: int,
    trial_seed: int,
    snr_db: float,
    x_name: str,
    x_value: float,
    k_paths: int,
    paper_k: int,
    num_ris_paths: int | str,
    receiver_mode: str,
    outlier_threshold_m: float,
    store_large_arrays: bool,
    profile_memory: bool,
    result_identity_guard: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    start = time.perf_counter()
    rss_before = _rss_mb() if profile_memory else float("nan")
    try:
        with thread_limit_context(_WORKER_BLAS_THREADS):
            result = result_factory()
        if result_identity_guard is not None:
            token = result.get("_grouped_result_identity_token")
            if token is None:
                token = f"{variant}:{time.perf_counter_ns()}:{id(result)}"
                result["_grouped_result_identity_token"] = token
            prior_variant = result_identity_guard["tokens"].get(token)
            if prior_variant is not None:
                raise GroupedResultReuseError(
                    "grouped execution reused one final result object for "
                    f"{prior_variant!r} and {variant!r}"
                )
            result_identity_guard["tokens"][token] = variant
        row = _result_to_row(
            result,
            figure=figure,
            variant=variant,
            trial_id=trial_id,
            trial_seed=trial_seed,
            snr_db=snr_db,
            x_name=x_name,
            x_value=x_value,
            outlier_threshold_m=outlier_threshold_m,
            runtime_s=time.perf_counter() - start,
            store_large_arrays=store_large_arrays,
            rss_before=rss_before,
            profile_memory=profile_memory,
            compact_result=False,
        )
        result_effective_k = _to_float(row.get("K"))
        effective_k = (
            int(result_effective_k)
            if np.isfinite(result_effective_k)
            else int(k_paths)
        )
        _set_row_k_metadata(
            row,
            figure=figure,
            effective_k=effective_k,
            paper_k=int(paper_k),
            x_value=float(x_value),
            receiver_mode=str(row.get("receiver_mode") or receiver_mode),
            config_seed=int(trial_seed),
            num_ris_paths=num_ris_paths,
        )
        del result
    except GroupedResultReuseError:
        raise
    except Exception as exc:  # noqa: BLE001 - failed variants must be logged as rows.
        row = _failure_row_from_payload(
            figure=figure,
            variant=variant,
            trial_id=trial_id,
            trial_seed=trial_seed,
            snr_db=snr_db,
            x_name=x_name,
            x_value=x_value,
            k_paths=k_paths,
            receiver_mode=receiver_mode,
            exc=exc,
        )
        _set_row_k_metadata(
            row,
            figure=figure,
            effective_k=int(k_paths),
            paper_k=int(paper_k),
            x_value=float(x_value),
            receiver_mode=receiver_mode,
            config_seed=int(trial_seed),
            num_ris_paths=num_ris_paths,
        )
        if _WORKER_TRIM_MEMORY:
            trim_memory()
        return (
            _ensure_worker_result_rows_safe([row])[0],
            f"\nERROR {variant} {type(exc).__name__}: {exc}\n",
        )
    if _WORKER_TRIM_MEMORY:
        trim_memory()
    return _ensure_worker_result_rows_safe([row])[0], ""


def _run_grouped_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    _apply_blas_thread_env(
        int(task.get("blas_threads", _WORKER_BLAS_THREADS)),
        respect_existing_blas_env=bool(
            task.get("respect_existing_blas_env", _WORKER_RESPECT_EXISTING_BLAS_ENV)
        ),
    )
    figure = str(task["figure"])
    group = str(task["group"])
    trial_id = int(task["trial_id"])
    trial_seed = int(task["trial_seed"])
    snr_db = float(task["snr_db"])
    x_name = str(task["x_name"])
    x_value = float(task["x_value"])
    k_paths = int(task["K"])
    paper_k = int(task["paper_k"])
    out_dir = pathlib.Path(task["out_dir"]) if task.get("out_dir") else _WORKER_OUT_DIR
    outlier_threshold_m = float(task["outlier_threshold_m"])
    store_large_arrays = bool(task["store_large_arrays"])
    profile_memory = bool(task["profile_memory"])
    trim_memory_enabled = bool(task.get("trim_memory", _WORKER_TRIM_MEMORY))
    rows: list[dict[str, Any]] = []
    logs: list[str] = []
    data = None
    stage1 = None
    try:
        base_config = _config_from_base(
            figure=figure,
            seed=trial_seed,
            snr_db=snr_db,
            k_paths=k_paths,
        )
        receiver_mode = str(
            base_config.get("receiver_mode", base_config.get("evs_selection", "full_6d"))
        )
        num_ris_paths = _num_ris_paths(base_config)
        if group in {"fig1_fig2", "fig5", "fig6"}:
            data = _make_data(base_config)
            if group != "fig4":
                stage1 = run_stage1_only(data, base_config)
        elif group == "nested_receiver":
            base_config["receiver_mode"] = "full_6d"
            base_config["nested_receiver_noise_convention"] = (
                NESTED_RECEIVER_NOISE_CONVENTION
            )
            data = _make_data(base_config)
        if group == "nested_receiver":
            variants = _variant_specs(figure)
            for variant, updates in variants.items():
                config = apply_nested_update(copy.deepcopy(base_config), updates)
                mode = str(config.get("receiver_mode", "full_6d"))
                nested_data = make_nested_receiver_mode_data(data, mode, config)
                if figure == "fig3":
                    factory = (
                        lambda nested_data=nested_data, config=config: (
                            run_single_proposed_diagnostic(
                                config,
                                allow_stage2=bool(
                                    updates.get("_allow_stage2", True)
                                ),
                                data_override=nested_data,
                            )
                        )
                    )
                else:
                    factory = (
                        lambda nested_data=nested_data,
                        config=config,
                        variant=variant: (
                            _peb_bound_metrics_result_for_config(
                                config, out_dir, nested_data, variant
                            )
                        )
                    )
                row, log = _row_for_result_or_failure(
                    result_factory=factory,
                    figure=figure,
                    variant=variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=snr_db,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    paper_k=paper_k,
                    num_ris_paths=num_ris_paths,
                    receiver_mode=mode,
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                )
                rows.append(row)
                if log:
                    logs.append(log)
            if figure == "fig3":
                for variant, updates in _extra_peb_specs("fig3").items():
                    if (
                        "constrained_jones_peb" in variant
                        and not bool(_jones_bound_options(base_config)["include_constrained"])
                    ):
                        continue
                    config = apply_nested_update(copy.deepcopy(base_config), updates)
                    nested_data = make_nested_receiver_mode_data(
                        data, "full_6d", config
                    )
                    row, log = _row_for_result_or_failure(
                        result_factory=lambda nested_data=nested_data,
                        config=config,
                        variant=variant: (
                            _peb_bound_metrics_result_for_config(
                                config, out_dir, nested_data, variant
                            )
                        ),
                        figure=figure,
                        variant=variant,
                        trial_id=trial_id,
                        trial_seed=trial_seed,
                        snr_db=snr_db,
                        x_name=x_name,
                        x_value=x_value,
                        k_paths=k_paths,
                        paper_k=paper_k,
                        num_ris_paths=num_ris_paths,
                        receiver_mode="full_6d",
                        outlier_threshold_m=outlier_threshold_m,
                        store_large_arrays=store_large_arrays,
                        profile_memory=profile_memory,
                    )
                    rows.append(row)
                    if log:
                        logs.append(log)
        elif group == "fig1_fig2":
            variants = {
                **_variant_specs(FIG1_FIG2_SHARED_FIGURE),
                **(
                    _diagnostic_variant_specs(FIG1_FIG2_SHARED_FIGURE)
                    if bool(task.get("include_diagnostic_variants", False))
                    else {}
                ),
            }
            result_identity_guard: dict[str, Any] = {"tokens": {}}
            selected_variants = {
                str(name)
                for name in task.get(
                    "selected_variants",
                    task.get("validation_variants", []),
                )
            }
            if selected_variants:
                variants = {
                    name: spec
                    for name, spec in variants.items()
                    if name in selected_variants
                }
            for variant, updates in variants.items():
                config = apply_nested_update(copy.deepcopy(base_config), updates)
                allow_stage2 = bool(updates.get("_allow_stage2", True))
                runner = str(updates.get("_runner", "proposed"))
                if runner == "stage1_only":
                    factory = (
                        lambda data=data, stage1=stage1, config=config: (
                            _annotate_variant_result(
                                _stage1_only_result_from_shared(
                                    data, copy.deepcopy(stage1), config
                                ),
                                config,
                                final_runner_name="stage1_only",
                                used_main_single_proposed_path=False,
                            )
                        )
                    )
                else:
                    factory = (
                        lambda data=data,
                        stage1=stage1,
                        config=config,
                        updates=updates,
                        allow_stage2=allow_stage2: run_final_vp_from_shared_stage1(
                            data, stage1, config, updates, allow_stage2
                        )
                    )
                row, log = _row_for_result_or_failure(
                    result_factory=factory,
                    figure=figure,
                    variant=variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=snr_db,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    paper_k=paper_k,
                    num_ris_paths=num_ris_paths,
                    receiver_mode=receiver_mode,
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                    result_identity_guard=result_identity_guard,
                )
                if (
                    variant == "adaptive_jones_vp_proposed"
                    and bool(task.get("debug_compare_main_single_proposed", False))
                    and not bool(row.get("failed"))
                ):
                    main_result = run_single_proposed_diagnostic(
                        copy.deepcopy(config),
                        data_override=copy.deepcopy(data),
                    )
                    main_metrics = _main_single_debug_metrics(main_result)
                    row.update(
                        {
                            "debug_main_position_error_m": main_metrics[
                                "position_error_m"
                            ],
                            "debug_main_y_nmse": main_metrics["y_nmse"],
                            "debug_main_raw_objective": main_metrics[
                                "raw_objective"
                            ],
                            "debug_main_global_vp_mode": main_metrics[
                                "global_vp_mode"
                            ],
                            "debug_main_adaptive_enabled": main_metrics[
                                "adaptive_enabled"
                            ],
                            "debug_config_diff_summary": _config_diff_summary(
                                config, main_result.get("stage1_config", {})
                            ),
                        }
                    )
                    position_diff = abs(
                        _to_float(row.get("position_error_m"))
                        - _to_float(main_metrics["position_error_m"])
                    )
                    if np.isfinite(position_diff) and position_diff > 1.0e-6:
                        logs.append(
                            "\nWARNING FIG1_MAIN_SINGLE_MISMATCH "
                            f"trial_id={trial_id} snr_db={snr_db} "
                            f"position_diff_m={position_diff:.6e}\n"
                        )
                    compact_experiment_result(main_result, keep_large_arrays=False)
                    del main_result
                rows.append(row)
                if log:
                    logs.append(log)
            peb_variants = []
            if not selected_variants or "PEB" in selected_variants:
                peb_variants.append("PEB")
            if (
                bool(_jones_bound_options(base_config)["include_constrained"])
                and (
                    not selected_variants
                    or "constrained_jones_peb" in selected_variants
                    or "PEB" in selected_variants
                )
            ):
                peb_variants.append("constrained_jones_peb")
            for peb_variant in peb_variants:
                peb_config = apply_nested_update(
                    copy.deepcopy(base_config),
                    _extra_peb_specs(FIG1_FIG2_SHARED_FIGURE)["PEB"],
                )
                row, log = _row_for_result_or_failure(
                    result_factory=lambda config=peb_config, data=data: (
                        _annotate_variant_result(
                            _peb_bound_metrics_result_for_config(
                                config, out_dir, data, peb_variant
                            ),
                            config,
                            final_runner_name="peb_only",
                            used_main_single_proposed_path=False,
                        )
                    ),
                    figure=figure,
                    variant=peb_variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=snr_db,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    paper_k=paper_k,
                    num_ris_paths=num_ris_paths,
                    receiver_mode="full_6d",
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                    result_identity_guard=result_identity_guard,
                )
                rows.append(row)
                if log:
                    logs.append(log)
        elif group == "fig5":
            for variant, updates in _variant_specs("fig5").items():
                config = apply_nested_update(copy.deepcopy(base_config), updates)
                allow_stage2 = bool(updates.get("_allow_stage2", True))
                runner = str(updates.get("_runner", "proposed"))
                if runner == "oracle_init_vp":
                    factory = lambda data=data, config=config: _oracle_result_from_shared(data, config)
                else:
                    factory = lambda data=data, stage1=stage1, config=config, updates=updates, allow_stage2=allow_stage2: run_final_vp_from_shared_stage1(
                        data, stage1, config, updates, allow_stage2
                    )
                row, log = _row_for_result_or_failure(
                    result_factory=factory,
                    figure=figure,
                    variant=variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=snr_db,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    paper_k=paper_k,
                    num_ris_paths=num_ris_paths,
                    receiver_mode=receiver_mode,
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                )
                rows.append(row)
                if log:
                    logs.append(log)
        elif group == "fig6":
            for variant, updates in _variant_specs("fig6").items():
                if str(updates.get("_runner", "proposed")) == "peb_only":
                    continue
                config = apply_nested_update(copy.deepcopy(base_config), updates)
                row, log = _row_for_result_or_failure(
                    result_factory=lambda data=data, stage1=stage1, config=config, updates=updates: run_final_vp_from_shared_stage1(
                        data, stage1, config, updates, bool(updates.get("_allow_stage2", True))
                    ),
                    figure=figure,
                    variant=variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=0.0,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    paper_k=paper_k,
                    num_ris_paths=num_ris_paths,
                    receiver_mode=receiver_mode,
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                )
                rows.append(row)
                if log:
                    logs.append(log)
            for variant, updates in _variant_specs("fig6").items():
                if str(updates.get("_runner", "proposed")) != "peb_only":
                    continue
                if (
                    "constrained_jones_peb" in variant
                    and not bool(_jones_bound_options(base_config)["include_constrained"])
                ):
                    continue
                peb_config = apply_nested_update(copy.deepcopy(base_config), updates)
                row, log = _row_for_result_or_failure(
                    result_factory=lambda config=peb_config,
                    data=data,
                    variant=variant: _peb_bound_metrics_result_for_config(
                        config, out_dir, data, variant
                    ),
                    figure=figure,
                    variant=variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=0.0,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    paper_k=paper_k,
                    num_ris_paths=num_ris_paths,
                    receiver_mode=receiver_mode,
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                )
                rows.append(row)
                if log:
                    logs.append(log)
        else:
            raise ValueError(f"unsupported grouped task group {group!r}")
    finally:
        del data, stage1
        if trim_memory_enabled:
            trim_memory()
        else:
            gc.collect()
    return _ensure_worker_result_rows_safe(rows), "".join(logs)


def _iter_task_results(
    tasks: list[dict[str, Any]],
    *,
    process_workers: int,
    maxtasksperchild: int,
    base_config: dict,
    out_dir: pathlib.Path,
    blas_threads: int,
    respect_existing_blas_env: bool,
    trim_memory_enabled: bool,
) -> Iterable[tuple[list[dict[str, Any]], str]]:
    peb_cache_enabled = _prepare_peb_cache(out_dir)
    if process_workers == 1:
        _init_worker(
            base_config,
            str(out_dir),
            blas_threads,
            respect_existing_blas_env,
            trim_memory_enabled,
            peb_cache_enabled,
        )
        for task in tasks:
            if task.get("task_kind") == "grouped":
                yield _run_grouped_task(task)
            else:
                yield _run_trial_task(task)
        return
    processes = int(process_workers)
    worker = _run_grouped_task if tasks and tasks[0].get("task_kind") == "grouped" else _run_trial_task
    with mp.Pool(
        processes=processes,
        maxtasksperchild=int(maxtasksperchild),
        initializer=_init_worker,
        initargs=(
            base_config,
            str(out_dir),
            int(blas_threads),
            bool(respect_existing_blas_env),
            bool(trim_memory_enabled),
            bool(peb_cache_enabled),
        ),
    ) as pool:
        for result in pool.imap_unordered(worker, tasks, chunksize=1):
            yield result


def _write_trial_results(
    trial_csv: pathlib.Path,
    tasks: list[dict[str, Any]],
    log_path: pathlib.Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    resource_plan = getattr(args, "resource_plan", None) or resolve_hybrid_resources(
        jobs=int(args.jobs),
        process_workers=getattr(args, "process_workers", None),
        blas_threads=args.blas_threads,
        n_tasks=max(len(tasks), 1),
        memory_budget_gb=getattr(args, "memory_budget_gb", None),
        memory_per_worker_gb=getattr(args, "memory_per_worker_gb", None),
    )
    process_workers = int(resource_plan["process_workers"])
    return_trial_rows = bool(getattr(args, "return_trial_rows", True))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_config = default_config()
    apply_global_vp_cli_overrides(base_config, args)
    apply_peb_cli_overrides(base_config, args)
    out_dir = pathlib.Path(args.out_dir)
    writer_context = (
        StreamingCsvWriter(
            trial_csv,
            FIELDNAMES,
            flush_every=int(args.csv_flush_every),
        )
        if args.streaming_csv
        else contextlib.nullcontext()
    )
    buffered_rows: list[dict[str, Any]] = []
    with log_path.open("w") as log_handle, writer_context as csv_writer:
        log_handle.write(f"jobs={int(resource_plan['jobs'])}\n")
        log_handle.write(f"process_workers={process_workers}\n")
        log_handle.write(f"blas_threads={int(resource_plan['blas_threads'])}\n")
        log_handle.write(f"task_grouping={args.task_grouping}\n")
        log_handle.write(f"streaming_csv={bool(args.streaming_csv)}\n")
        log_handle.write(f"store_large_arrays={bool(args.store_large_arrays)}\n")
        if tasks and tasks[0].get("task_grouping_warning"):
            log_handle.write(f"WARNING: {tasks[0]['task_grouping_warning']}\n")
        for row_batch, log_text in _iter_task_results(
            tasks,
            process_workers=process_workers,
            maxtasksperchild=int(args.maxtasksperchild),
            base_config=base_config,
            out_dir=out_dir,
            blas_threads=int(resource_plan["blas_threads"]),
            respect_existing_blas_env=bool(args.respect_existing_blas_env),
            trim_memory_enabled=bool(args.trim_memory),
        ):
            progress_logger = getattr(args, "progress_logger", None)
            representative = row_batch[0] if row_batch else {}
            failed_rows = [
                row
                for row in row_batch
                if str(row.get("failed")).lower() == "true"
            ]
            if progress_logger is not None:
                progress_logger.log(
                    "task_failed" if failed_rows else "task_done",
                    "failed" if failed_rows else "completed",
                    figure=representative.get("figure", ""),
                    baseline_or_variant=",".join(
                        str(row.get("variant", "")) for row in row_batch
                    ),
                    snr_db=representative.get("snr_db", ""),
                    trial_id=representative.get("trial_id", ""),
                    seed=representative.get("seed", ""),
                    K=representative.get("K", ""),
                    message="paper ablation trial batch completed",
                    error="; ".join(
                        str(row.get("error", "")) for row in failed_rows
                    ),
                )
            for row in row_batch:
                if args.streaming_csv:
                    csv_writer.writerow(row)
                else:
                    buffered_rows.append(row)
                log_handle.write(
                    f"figure={row['figure']} variant={row['variant']} trial={row['trial_id']} "
                    f"seed={row['seed']} x={row['x_value']} failed={row['failed']}\n"
                )
                log_handle.flush()
            if log_text:
                log_handle.write(log_text)
                if not log_text.endswith("\n"):
                    log_handle.write("\n")
                log_handle.flush()
    if not args.streaming_csv:
        with StreamingCsvWriter(
            trial_csv,
            FIELDNAMES,
            flush_every=int(args.csv_flush_every),
        ) as writer:
            for row in buffered_rows:
                writer.writerow(row)
        if return_trial_rows:
            return buffered_rows
        buffered_rows.clear()
        return []
    if return_trial_rows:
        return _read_csv(trial_csv)
    return []


def _args_without_trial_row_return(args: argparse.Namespace) -> argparse.Namespace:
    no_rows_args = copy.copy(args)
    no_rows_args.return_trial_rows = False
    return no_rows_args


def _fig1_fig2_shared_trial_csv(out_dir: pathlib.Path) -> pathlib.Path:
    return out_dir / FIG1_FIG2_SHARED_TRIAL_CSV


def _fig1_fig2_shared_summary_csv(out_dir: pathlib.Path) -> pathlib.Path:
    return out_dir / FIG1_FIG2_SHARED_SUMMARY_CSV


def _figure_trial_csv(out_dir: pathlib.Path, figure: str) -> pathlib.Path:
    return out_dir / f"{figure}_trials.csv"


def _figure_summary_csv(out_dir: pathlib.Path, figure: str) -> pathlib.Path:
    return out_dir / f"{figure}_summary.csv"


def _summarize_trial_csv(trial_csv: pathlib.Path, figure: str) -> list[dict[str, Any]]:
    return summarize_rows(_read_csv(trial_csv), figure)


def _summarize_fig1_fig2_shared_csv(shared_trial_csv: pathlib.Path) -> list[dict[str, Any]]:
    return summarize_fig1_fig2_shared_rows(_read_csv(shared_trial_csv))


def _write_fig1_fig2_derived_outputs(
    out_dir: pathlib.Path,
    rows: list[dict[str, Any]],
    *,
    include_diagnostic_variants: bool = False,
) -> None:
    fig1_rows = [{**row, "figure": "fig1"} for row in rows]
    fig2_rows = [{**row, "figure": "fig2"} for row in rows]
    _write_rows_atomic_csv(_figure_trial_csv(out_dir, "fig1"), fig1_rows, FIELDNAMES)
    _write_rows_atomic_csv(_figure_trial_csv(out_dir, "fig2"), fig2_rows, FIELDNAMES)
    _write_rescue_policy_ablation_csvs(
        out_dir,
        rows,
        enabled=include_diagnostic_variants,
    )
    fig1_summary = summarize_rows(rows, "fig1")
    fig2_summary = summarize_rows(rows, "fig2")
    _write_csv(
        _figure_summary_csv(out_dir, "fig1"),
        fig1_summary,
        list(fig1_summary[0].keys()) if fig1_summary else [],
    )
    _write_csv(
        _figure_summary_csv(out_dir, "fig2"),
        fig2_summary,
        list(fig2_summary[0].keys()) if fig2_summary else [],
    )
    _write_duplicate_curves_report("fig1", fig1_summary, rows, out_dir)


FIG1_MAIN_SINGLE_CONSISTENCY_FIELDS = [
    "trial_id",
    "seed",
    "snr_db",
    "fig1_position_error_m",
    "main_single_position_error_m",
    "abs_position_diff_m",
    "fig1_y_nmse",
    "main_single_y_nmse",
    "fig1_raw_objective",
    "main_single_raw_objective",
    "fig1_global_vp_mode",
    "main_single_global_vp_mode",
    "fig1_adaptive_enabled",
    "main_single_adaptive_enabled",
    "config_diff_summary",
]


def _write_fig1_main_single_consistency(
    out_dir: pathlib.Path,
    rows: list[dict[str, Any]],
) -> None:
    consistency_rows = []
    for row in rows:
        if row.get("variant") != "adaptive_jones_vp_proposed":
            continue
        main_position = _to_float(row.get("debug_main_position_error_m"))
        fig_position = _to_float(row.get("position_error_m"))
        if not np.isfinite(main_position):
            continue
        consistency_rows.append(
            {
                "trial_id": row.get("trial_id"),
                "seed": row.get("seed"),
                "snr_db": row.get("snr_db"),
                "fig1_position_error_m": fig_position,
                "main_single_position_error_m": main_position,
                "abs_position_diff_m": abs(fig_position - main_position),
                "fig1_y_nmse": row.get("y_nmse"),
                "main_single_y_nmse": row.get("debug_main_y_nmse"),
                "fig1_raw_objective": row.get("raw_objective_final"),
                "main_single_raw_objective": row.get(
                    "debug_main_raw_objective"
                ),
                "fig1_global_vp_mode": row.get("global_vp_mode"),
                "main_single_global_vp_mode": row.get(
                    "debug_main_global_vp_mode"
                ),
                "fig1_adaptive_enabled": row.get("adaptive_enabled"),
                "main_single_adaptive_enabled": row.get(
                    "debug_main_adaptive_enabled"
                ),
                "config_diff_summary": row.get("debug_config_diff_summary"),
            }
        )
    _write_csv(
        out_dir / "fig1_main_single_consistency.csv",
        consistency_rows,
        FIG1_MAIN_SINGLE_CONSISTENCY_FIELDS,
    )


def _write_fig1_fig2_derived_outputs_from_csv(
    out_dir: pathlib.Path,
    shared_trial_csv: pathlib.Path,
    *,
    include_diagnostic_variants: bool = False,
) -> None:
    rows = _read_csv(shared_trial_csv)
    try:
        _write_fig1_fig2_derived_outputs(
            out_dir,
            rows,
            include_diagnostic_variants=include_diagnostic_variants,
        )
    finally:
        del rows


def _run_fig1_fig2_shared_trials(
    *,
    args: argparse.Namespace,
    snr_grid: list[float],
    trial_seeds: list[int],
    existing_metadata: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    out_dir = pathlib.Path(args.out_dir)
    shared_trial_csv = _fig1_fig2_shared_trial_csv(out_dir)
    shared_summary_csv = _fig1_fig2_shared_summary_csv(out_dir)
    log_path = out_dir / "fig1_fig2_vp_family_raw.log"
    can_reuse, rows = _can_reuse_csv(
        shared_trial_csv,
        FIG1_FIG2_SHARED_FIGURE,
        args,
        snr_grid,
        [FIG1_FIG2_SHARED_FIGURE],
        existing_metadata,
    )
    if not can_reuse:
        x_name, x_values = _figure_x_grid(FIG1_FIG2_SHARED_FIGURE, snr_grid, args.k_grid_values)
        variants = _variants_for_figure(
            FIG1_FIG2_SHARED_FIGURE,
            args.variant_filter_values,
            include_diagnostic_variants=bool(
                getattr(args, "include_diagnostic_variants", False)
            ),
        )
        variants = _apply_peb_cli_variant_filter(variants, args)
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks = _tasks_for_figure(
            figure=FIG1_FIG2_SHARED_FIGURE,
            grouped_group="fig1_fig2",
            x_name=x_name,
            x_values=x_values,
            variants=variants,
            trial_seeds=trial_seeds,
            args=args,
        )
        _print_task_summary(tasks, variants)
        rows = _write_trial_results(shared_trial_csv, tasks, log_path, args)
    shared_summary = summarize_fig1_fig2_shared_rows(rows)
    _write_csv(
        shared_summary_csv,
        shared_summary,
        list(shared_summary[0].keys()) if shared_summary else [],
    )
    _write_fig1_fig2_derived_outputs(
        out_dir,
        rows,
        include_diagnostic_variants=bool(
            getattr(args, "include_diagnostic_variants", False)
        ),
    )
    if bool(getattr(args, "debug_compare_main_single_proposed", False)):
        _write_fig1_main_single_consistency(out_dir, rows)
    return rows


def _ensure_fig1_fig2_shared_outputs(
    *,
    args: argparse.Namespace,
    snr_grid: list[float],
    trial_seeds: list[int],
    existing_metadata: dict[str, Any] | None,
) -> None:
    out_dir = pathlib.Path(args.out_dir)
    shared_trial_csv = _fig1_fig2_shared_trial_csv(out_dir)
    shared_summary_csv = _fig1_fig2_shared_summary_csv(out_dir)
    log_path = out_dir / "fig1_fig2_vp_family_raw.log"
    can_reuse, cached_rows = _can_reuse_csv(
        shared_trial_csv,
        FIG1_FIG2_SHARED_FIGURE,
        args,
        snr_grid,
        [FIG1_FIG2_SHARED_FIGURE],
        existing_metadata,
    )
    if not can_reuse:
        x_name, x_values = _figure_x_grid(FIG1_FIG2_SHARED_FIGURE, snr_grid, args.k_grid_values)
        variants = _variants_for_figure(
            FIG1_FIG2_SHARED_FIGURE,
            args.variant_filter_values,
            include_diagnostic_variants=bool(
                getattr(args, "include_diagnostic_variants", False)
            ),
        )
        variants = _apply_peb_cli_variant_filter(variants, args)
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks = _tasks_for_figure(
            figure=FIG1_FIG2_SHARED_FIGURE,
            grouped_group="fig1_fig2",
            x_name=x_name,
            x_values=x_values,
            variants=variants,
            trial_seeds=trial_seeds,
            args=args,
        )
        _print_task_summary(tasks, variants)
        _write_trial_results(
            shared_trial_csv,
            tasks,
            log_path,
            _args_without_trial_row_return(args),
        )
        shared_summary = _summarize_fig1_fig2_shared_csv(shared_trial_csv)
    else:
        shared_summary = summarize_fig1_fig2_shared_rows(cached_rows)
        del cached_rows
    _write_csv(
        shared_summary_csv,
        shared_summary,
        list(shared_summary[0].keys()) if shared_summary else [],
    )
    _write_fig1_fig2_derived_outputs_from_csv(
        out_dir,
        shared_trial_csv,
        include_diagnostic_variants=bool(
            getattr(args, "include_diagnostic_variants", False)
        ),
    )
    if bool(getattr(args, "debug_compare_main_single_proposed", False)):
        _write_fig1_main_single_consistency(
            out_dir, _read_csv(shared_trial_csv)
        )


def _ensure_figure_outputs(
    figure: str,
    *,
    args: argparse.Namespace,
    snr_grid: list[float],
    trial_seeds: list[int],
    figures: list[str],
    existing_metadata: dict[str, Any] | None,
    completed_figures: set[str],
) -> list[dict[str, Any]]:
    if _is_fig1_fig2(figure):
        _ = completed_figures
        _ensure_fig1_fig2_shared_outputs(
            args=args,
            snr_grid=snr_grid,
            trial_seeds=trial_seeds,
            existing_metadata=existing_metadata,
        )
        return _read_csv(_figure_summary_csv(pathlib.Path(args.out_dir), figure))

    out_dir = pathlib.Path(args.out_dir)
    trial_csv = _figure_trial_csv(out_dir, figure)
    summary_csv = _figure_summary_csv(out_dir, figure)
    log_path = out_dir / f"{figure}_raw.log"
    can_reuse, cached_rows = _can_reuse_csv(
        trial_csv, figure, args, snr_grid, figures, existing_metadata
    )
    if not can_reuse:
        x_name, x_values = _figure_x_grid(figure, snr_grid, args.k_grid_values)
        variants = _variants_for_figure(
            figure,
            args.variant_filter_values,
            include_diagnostic_variants=bool(
                getattr(args, "include_diagnostic_variants", False)
            ),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        grouped_group = {
            "fig3": "nested_receiver",
            "fig4": "nested_receiver",
            "fig5": "fig5",
            "fig6": "fig6",
        }.get(figure)
        tasks = _tasks_for_figure(
            figure=figure,
            grouped_group=grouped_group,
            x_name=x_name,
            x_values=x_values,
            variants=variants,
            trial_seeds=trial_seeds,
            args=args,
        )
        _print_task_summary(tasks, variants)
        _write_trial_results(
            trial_csv,
            tasks,
            log_path,
            _args_without_trial_row_return(args),
        )
        summary = _summarize_trial_csv(trial_csv, figure)
    else:
        summary = summarize_rows(cached_rows, figure)
        del cached_rows
    _write_csv(summary_csv, summary, list(summary[0].keys()) if summary else [])
    return summary


def _run_figure(
    figure: str,
    *,
    args: argparse.Namespace,
    snr_grid: list[float],
    trial_seeds: list[int],
    figures: list[str],
    existing_metadata: dict[str, Any] | None,
    completed_figures: set[str],
) -> list[dict[str, Any]]:
    if _is_fig1_fig2(figure):
        return _run_fig1_fig2_shared_trials(
            args=args,
            snr_grid=snr_grid,
            trial_seeds=trial_seeds,
            existing_metadata=existing_metadata,
        )
    out_dir = pathlib.Path(args.out_dir)
    trial_csv = out_dir / f"{figure}_trials.csv"
    summary_csv = out_dir / f"{figure}_summary.csv"
    log_path = out_dir / f"{figure}_raw.log"
    can_reuse, cached_rows = _can_reuse_csv(
        trial_csv, figure, args, snr_grid, figures, existing_metadata
    )
    if can_reuse:
        rows = cached_rows
        summary = summarize_rows(rows, figure)
        _write_csv(summary_csv, summary, list(summary[0].keys()) if summary else [])
        return rows

    x_name, x_values = _figure_x_grid(figure, snr_grid, args.k_grid_values)
    variants = _variants_for_figure(
        figure,
        args.variant_filter_values,
        include_diagnostic_variants=bool(
            getattr(args, "include_diagnostic_variants", False)
        ),
    )
    variants = _apply_peb_cli_variant_filter(variants, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped_group = {
        "fig3": "nested_receiver",
        "fig4": "nested_receiver",
        "fig5": "fig5",
        "fig6": "fig6",
    }.get(figure)
    tasks = _tasks_for_figure(
        figure=figure,
        grouped_group=grouped_group,
        x_name=x_name,
        x_values=x_values,
        variants=variants,
        trial_seeds=trial_seeds,
        args=args,
    )
    _print_task_summary(tasks, variants)
    rows = _write_trial_results(trial_csv, tasks, log_path, args)
    summary = summarize_rows(rows, figure)
    _write_csv(summary_csv, summary, list(summary[0].keys()) if summary else [])
    return rows


def _clone_args_for_validation(
    args: argparse.Namespace,
    *,
    task_grouping: str,
    out_dir: pathlib.Path,
) -> argparse.Namespace:
    validation_args = copy.copy(args)
    validation_args.task_grouping = task_grouping
    validation_args.out_dir = out_dir
    validation_args.n_trials = 1
    validation_args.paper_k = 1
    validation_args.no_plots = True
    validation_args.force_rerun = True
    validation_args.reuse_existing = False
    return validation_args


def _validation_rows_for_grouping(
    args: argparse.Namespace,
    *,
    task_grouping: str,
    snr_db: float,
    trial_seed: int,
) -> list[dict[str, Any]]:
    validation_out_dir = pathlib.Path(args.out_dir) / f"grouped_equivalence_{task_grouping}"
    validation_args = _clone_args_for_validation(
        args,
        task_grouping=task_grouping,
        out_dir=validation_out_dir,
    )
    variants = {
        name: spec
        for name, spec in _variant_specs(FIG1_FIG2_SHARED_FIGURE).items()
        if name
        in {
            "fixed_pol_vp",
            "free_jones_vp",
            "regularized_jones_vp",
            "adaptive_jones_vp_proposed",
        }
    }
    tasks = _tasks_for_figure(
        figure=FIG1_FIG2_SHARED_FIGURE,
        grouped_group="fig1_fig2",
        x_name="snr_db",
        x_values=[float(snr_db)],
        variants=variants,
        trial_seeds=[int(trial_seed)],
        args=validation_args,
    )
    if task_grouping == "grouped":
        for task in tasks:
            task["validation_variants"] = list(variants)
    rows: list[dict[str, Any]] = []
    validation_base_config = default_config()
    apply_global_vp_cli_overrides(validation_base_config, validation_args)
    apply_peb_cli_overrides(validation_base_config, validation_args)
    for row_batch, log_text in _iter_task_results(
        tasks,
        process_workers=min(
            int(validation_args.process_workers),
            max(len(tasks), 1),
        ),
        maxtasksperchild=int(validation_args.maxtasksperchild),
        base_config=validation_base_config,
        out_dir=validation_out_dir,
        blas_threads=int(validation_args.blas_threads),
        respect_existing_blas_env=bool(validation_args.respect_existing_blas_env),
        trim_memory_enabled=bool(validation_args.trim_memory),
    ):
        rows.extend(row_batch)
        if log_text:
            sys.stderr.write(log_text)
            if not log_text.endswith("\n"):
                sys.stderr.write("\n")
    return rows


def _single_row_by_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get("variant")) == variant]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one validation row for {variant}, got {len(matches)}")
    return matches[0]


def _raise_grouped_equivalence_mismatch(
    *,
    variant: str,
    metric: str,
    grouped_row: dict[str, Any],
    variant_row: dict[str, Any],
    reason: str,
) -> None:
    payload = {
        "variant": variant,
        "metric": metric,
        "reason": reason,
        "grouped_row": grouped_row,
        "variant_row": variant_row,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), file=sys.stderr)
    raise RuntimeError(
        f"grouped equivalence validation failed for {variant} {metric}: {reason}"
    )


def validate_grouped_equivalence(args: argparse.Namespace, snr_grid: list[float]) -> None:
    _ = snr_grid
    variants = [
        "fixed_pol_vp",
        "free_jones_vp",
        "regularized_jones_vp",
        "adaptive_jones_vp_proposed",
    ]
    metrics = ["position_rmse_m", "y_nmse", "raw_objective_final"]
    snr_db = 0.0
    trial_seed = int(args.seed)
    grouped_rows = _validation_rows_for_grouping(
        args,
        task_grouping="grouped",
        snr_db=snr_db,
        trial_seed=trial_seed,
    )
    variant_rows = _validation_rows_for_grouping(
        args,
        task_grouping="variant",
        snr_db=snr_db,
        trial_seed=trial_seed,
    )
    for variant in variants:
        grouped_row = _single_row_by_variant(grouped_rows, variant)
        variant_row = _single_row_by_variant(variant_rows, variant)
        if str(grouped_row.get("failed")) == "True" or str(variant_row.get("failed")) == "True":
            _raise_grouped_equivalence_mismatch(
                variant=variant,
                metric="failed",
                grouped_row=grouped_row,
                variant_row=variant_row,
                reason="one or both validation rows failed",
            )
        for metric in metrics:
            grouped_value = _to_float(grouped_row.get(metric))
            variant_value = _to_float(variant_row.get(metric))
            if not np.isclose(
                grouped_value,
                variant_value,
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            ):
                _raise_grouped_equivalence_mismatch(
                    variant=variant,
                    metric=metric,
                    grouped_row=grouped_row,
                    variant_row=variant_row,
                    reason=f"grouped={grouped_value!r} variant={variant_value!r}",
                )
    print(
        "Grouped equivalence validation passed "
        f"(fig1_fig2, seed={trial_seed}, snr_db={snr_db})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_list = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description="Generate paper ablation figures.")
    parser.add_argument("--figures", default="all")
    parser.add_argument(
        "--variant-filter",
        default=None,
        help=(
            "Comma-separated Fig.1/Fig.2 variant names. "
            "PEB accepts PEB, peb_only, or data_only_peb."
        ),
    )
    parser.add_argument(
        "--include-diagnostic-variants",
        action="store_true",
        help="Expose opt-in Fig.1/Fig.2 diagnostic variants before filtering.",
    )
    add_mc_args(parser, n_trials_default=50, paper_k_default=DEFAULT_PAPER_K, outlier_threshold_default=0.1)
    parser.add_argument("--snr-grid", default=DEFAULT_SNR_GRID)
    parser.add_argument("--k-grid", default="1,2,3,4")
    add_io_args(parser, default_out_dir="results/ablation_paper")
    parser.add_argument("--reuse-existing", action="store_true")
    add_resource_args(parser, jobs_default=10, blas_threads_default=DEFAULT_BLAS_THREADS)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--maxtasksperchild", type=int, default=1)
    parser.add_argument(
        "--task-grouping",
        choices=("grouped", "variant"),
        default="grouped",
    )
    parser.add_argument(
        "--streaming-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--store-large-arrays",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--global-vp-backend",
        choices=("cpu", "cupy", "auto"),
        default=None,
    )
    parser.add_argument("--global-vp-gpu-device", type=int, default=0)
    parser.add_argument(
        "--global-vp-validate-gpu-against-cpu",
        action="store_true",
    )
    parser.add_argument("--global-vp-gpu-dtype", default="complex128")
    parser.add_argument(
        "--global-vp-gpu-keep-arrays-on-device",
        action="store_true",
    )
    parser.add_argument(
        "--include-constrained-jones-peb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--no-constrained-jones-peb",
        dest="include_constrained_jones_peb",
        action="store_false",
    )
    parser.add_argument("--include-anchored-jones-peb", action="store_true")
    parser.add_argument(
        "--jones-anchor-prior-mode",
        choices=("disabled", "manual", "lambda_from_adaptive"),
        default="disabled",
    )
    parser.add_argument("--jones-anchor-prior-scale", type=float, default=1.0)
    parser.add_argument("--csv-flush-every", type=int, default=10)
    parser.add_argument("--validate-grouped-equivalence", action="store_true")
    add_progress_args(parser)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--debug-compare-main-single-proposed",
        action="store_true",
        help=(
            "Re-run main_single proposed on each Fig.1 shared realization and "
            "write a numerical consistency CSV."
        ),
    )
    args = parser.parse_args(argv_list)
    if args.max_workers is not None:
        if (
            args.process_workers is not None
            and int(args.max_workers) != int(args.process_workers)
        ):
            raise ValueError(
                "--max-workers and --process-workers specify different process counts"
            )
        args.process_workers = int(args.max_workers)
    args.max_workers = args.process_workers
    normalize_blas_threads(args)
    args.k_grid_values = parse_k_grid(args.k_grid)
    args.variant_filter_values = parse_variant_filter(args.variant_filter)
    return args


def _progress_task_count(
    args: argparse.Namespace,
    figures: list[str],
    snr_grid: list[float],
) -> int:
    total = 0
    counted_shared = False
    for figure in figures:
        if _is_fig1_fig2(figure):
            if counted_shared:
                continue
            counted_shared = True
            figure_key = FIG1_FIG2_SHARED_FIGURE
            x_count = len(snr_grid)
        else:
            figure_key = figure
            x_count = (
                len(args.k_grid_values) if figure == "fig6" else len(snr_grid)
            )
        multiplier = 1
        if args.task_grouping == "variant":
            multiplier = len(
                _variants_for_figure(
                    figure_key,
                    args.variant_filter_values,
                    include_diagnostic_variants=bool(
                        getattr(args, "include_diagnostic_variants", False)
                    ),
                )
            )
        total += int(args.n_trials) * x_count * multiplier
    return max(total, 1)


def _enabled_diagnostic_variant_names(
    args: argparse.Namespace,
    figures: list[str],
) -> list[str]:
    if not bool(getattr(args, "include_diagnostic_variants", False)):
        return []
    names: list[str] = []
    for figure in figures:
        for name in _diagnostic_variant_specs(figure):
            if name not in names:
                names.append(name)
    return names


def _print_diagnostic_comparison_note(
    args: argparse.Namespace,
    figures: list[str],
) -> None:
    if not _enabled_diagnostic_variant_names(args, figures):
        return
    print(
        "Diagnostic comparison: match free_jones_vp, "
        "free_jones_vp_gated_rescue, free_jones_vp_force_rescue, and PEB rows on "
        "snr_db, trial_id, seed, receiver_mode; compute "
        "error_over_peb = position_error_m / matched_peb_position_m."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.maxtasksperchild <= 0:
        raise ValueError("--maxtasksperchild must be positive")
    if args.process_workers is not None and args.process_workers <= 0:
        raise ValueError("--process-workers must be positive")
    if args.blas_threads != "auto" and int(args.blas_threads) <= 0:
        raise ValueError("--blas-threads must be positive or 'auto'")
    if args.paper_k <= 0:
        raise ValueError("--paper-k must be positive")
    figures = parse_figures(args.figures)
    snr_grid = parse_snr_grid(args.snr_grid)
    print(f"Running Fig.1-Fig.5 with paper_k = {args.paper_k}")
    print(f"Running Fig.6 K-grid = {args.k_grid_values}")
    diagnostic_variants = _enabled_diagnostic_variant_names(args, figures)
    print(f"include_diagnostic_variants={bool(args.include_diagnostic_variants)}")
    print(f"diagnostic_variants={','.join(diagnostic_variants)}")
    _print_diagnostic_comparison_note(args, figures)
    progress_path = (
        pathlib.Path(args.progress_log)
        if args.progress_log is not None
        else pathlib.Path(args.out_dir) / "progress.jsonl"
    )
    progress = ProgressLogger(
        progress_path,
        _progress_task_count(args, figures, snr_grid),
        "run_paper_ablation_figures",
    )
    args.progress_logger = progress
    if not args.quiet_progress:
        print(f"Progress log: {progress_path}")
        print("Monitor with:")
        print(f"    tail -f {progress_path}")
        print(
            "    python -m src.experiments.monitor_progress "
            f"--progress-log {progress_path}"
        )
    progress.log("start", "running", message="paper ablation experiment started")
    n_tasks = max(
        1,
        int(args.n_trials)
        * max(
            len(snr_grid),
            len(args.k_grid_values) if "fig6" in figures else 0,
        ),
    )
    args.resource_plan = resolve_hybrid_resources(
        jobs=args.jobs,
        process_workers=args.process_workers,
        blas_threads=args.blas_threads,
        n_tasks=n_tasks,
        memory_budget_gb=args.memory_budget_gb,
        memory_per_worker_gb=args.memory_per_worker_gb,
    )
    args.process_workers = int(args.resource_plan["process_workers"])
    args.max_workers = args.process_workers
    args.blas_threads = int(args.resource_plan["blas_threads"])
    if (
        str(getattr(args, "global_vp_backend", None)) in {"cupy", "auto"}
        and int(args.process_workers) > 2
    ):
        print(
            "WARNING: --global-vp-backend "
            f"{args.global_vp_backend} with --process-workers "
            f"{args.process_workers} runs that many CuPy processes on one GPU; "
            "each keeps its own memory pool and can exhaust GPU memory. "
            "Prefer --process-workers 1 or 2 for the GPU backend, or use "
            "--global-vp-backend cpu with more workers."
        )
    _apply_blas_thread_env(
        args.blas_threads,
        respect_existing_blas_env=bool(args.respect_existing_blas_env),
    )
    print(
        "Resource plan: "
        f"jobs={args.jobs} "
        f"process_workers={args.process_workers} "
        f"blas_threads={args.blas_threads} "
        f"estimated_cpu_slots={args.resource_plan['estimated_cpu_slots']} "
        f"memory_budget_gb={args.memory_budget_gb} "
        f"memory_per_worker_gb={args.memory_per_worker_gb}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_grouped_equivalence:
        validate_grouped_equivalence(args, snr_grid)
        progress.log(
            "finished",
            "completed",
            message="grouped equivalence validation finished",
        )
        progress.close()
        return
    metadata_path = args.out_dir / "experiment_metadata.json"
    existing_metadata = _read_metadata(metadata_path)
    seed_sequence = np.random.SeedSequence(args.seed)
    trial_seeds = [_trial_seed(child) for child in seed_sequence.spawn(args.n_trials)]

    completed_figures: set[str] = set()
    fig1_fig2_shared_done = False
    for figure in figures:
        if _is_fig1_fig2(figure):
            if not fig1_fig2_shared_done:
                _ensure_fig1_fig2_shared_outputs(
                    args=args,
                    snr_grid=snr_grid,
                    trial_seeds=trial_seeds,
                    existing_metadata=existing_metadata,
                )
                fig1_fig2_shared_done = True
            summary = _read_csv(_figure_summary_csv(args.out_dir, figure))
        else:
            summary = _ensure_figure_outputs(
                figure,
                args=args,
                snr_grid=snr_grid,
                trial_seeds=trial_seeds,
                figures=figures,
                existing_metadata=existing_metadata,
                completed_figures=completed_figures,
            )
        if not args.no_plots:
            _plot_figure(figure, summary, args.out_dir)
        completed_figures.add(figure)
    with metadata_path.open("w") as handle:
        json.dump(_metadata(args, snr_grid, figures), handle, indent=2)
    progress.log("finished", "completed", message="paper ablation experiment finished")
    progress.close()
    print(f"Wrote paper ablation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
