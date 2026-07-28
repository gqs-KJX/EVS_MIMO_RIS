"""Run benchmark comparison baselines for EVS-MIMO-RIS paper figures."""

from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import json
import multiprocessing as mp
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.baselines.als_cpd import run_als_cpd_baseline
    from src.baselines.common import REFINEMENT_TIERS, BaselineResult, data_hash, make_baseline_row, proposed_trace_diagnostics, y_noisy_hash
    from src.baselines.nf_ris_groupomp_localgrid_wls import run_nf_ris_groupomp_localgrid_wls_baseline
    from src.baselines.ris_vbi_sbl import run_ris_vbi_sbl_baseline
    from src.channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from src.config import default_config
    from src.main_single_proposed import _make_data, run_single_proposed_diagnostic
    from src.metrics import relative_nmse, position_rmse
    from src.tensor_utils import hankelize_frequency
    from src.experiments.run_paper_ablation_figures import _peb_from_efim, set_number_of_ris_paths
    from src.experiments.final_mksc_ccop_common import (
        Stage1Cache,
        apply_final_paper_options,
        run_paper_variant,
    )
    from src.experiments.resource_control import (
        PeakRssSampler,
        apply_thread_limits,
        assert_row_is_light,
        memory_snapshot_mb,
        resolve_hybrid_resources,
        thread_limit_context,
        trim_memory,
    )
    from src.experiments.progress_logger import ProgressLogger
    from src.utils import scipy_is_available
    from src.experiments.cli_common import (
        add_io_args,
        add_global_vp_args,
        add_mc_args,
        add_progress_args,
        add_resource_args,
        normalize_blas_threads,
        global_vp_cli_overrides,
    )
else:
    from ..baselines.als_cpd import run_als_cpd_baseline
    from ..baselines.common import REFINEMENT_TIERS, BaselineResult, data_hash, make_baseline_row, proposed_trace_diagnostics, y_noisy_hash
    from ..baselines.nf_ris_groupomp_localgrid_wls import run_nf_ris_groupomp_localgrid_wls_baseline
    from ..baselines.ris_vbi_sbl import run_ris_vbi_sbl_baseline
    from ..channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from ..config import default_config
    from ..main_single_proposed import _make_data, run_single_proposed_diagnostic
    from ..metrics import relative_nmse, position_rmse
    from ..tensor_utils import hankelize_frequency
    from .run_paper_ablation_figures import _peb_from_efim, set_number_of_ris_paths
    from .final_mksc_ccop_common import (
        Stage1Cache,
        apply_final_paper_options,
        run_paper_variant,
    )
    from .resource_control import (
        PeakRssSampler,
        apply_thread_limits,
        assert_row_is_light,
        memory_snapshot_mb,
        resolve_hybrid_resources,
        thread_limit_context,
        trim_memory,
    )
    from .progress_logger import ProgressLogger
    from ..utils import scipy_is_available
    from .cli_common import (
        add_io_args,
        add_global_vp_args,
        add_mc_args,
        add_progress_args,
        add_resource_args,
        normalize_blas_threads,
        global_vp_cli_overrides,
    )


DEFAULT_SNR_GRID = "-30,-25,-20,-15,-10,-5,0,5,10,15,20"
DEFAULT_BASELINES = (
    "als_cpd,scaled_4d,nf_ris_groupomp_localgrid_wls,ris_vbi_sbl,"
    "mksc_ccop,peb,"
    "constrained_jones_peb"
)
TRIAL_CSV = "benchmark_trials.csv"
SUMMARY_CSV = "benchmark_summary.csv"
RMSE_PDF = "fig7_benchmark_rmse_vs_snr.pdf"
CONDITIONAL_RMSE_PDF = "fig7_benchmark_conditional_rmse_vs_snr.pdf"
NMSE_PDF = "fig7_benchmark_nmse_vs_snr.pdf"
OUTLIER_PDF = "fig7_benchmark_outlier_vs_snr.pdf"
POSITION_P95_PDF = "fig7_benchmark_position_p95_vs_snr.pdf"
NMSE_P95_PDF = "fig7_benchmark_nmse_p95_vs_snr.pdf"
RUNTIME_MEMORY_CSV = "table1_runtime_memory.csv"
SUMMARY_MD = "benchmark_summary.md"
FIELDNAMES = [
    "baseline",
    "trial_id",
    "seed",
    "snr_db",
    "K",
    "data_hash",
    "y_noisy_hash",
    "failed",
    "error",
    "runtime_s",
    "position_error_m",
    "position_rmse_m",
    "y_nmse",
    "range_rmse_m",
    "tau_rmse_s",
    "clock_estimate_ns",
    "clock_native_estimate_ns",
    "clock_error_ns",
    "clock_panel_mad_ns",
    "clock_num_panels",
    "clock_expected_panels",
    "clock_complete_panel_set",
    "clock_invalid",
    "clock_invalid_reason",
    "clock_catastrophic",
    "clock_catastrophic_threshold_ns",
    "clock_extraction_rule",
    "clock_delay_source",
    "clock_panel_estimates_ns",
    "clock_certified",
    "raw_objective_final",
    "stage1_runtime_s",
    "global_vp_runtime_s",
    "total_runtime_s",
    "stage1_output_hash",
    "candidate_hash",
    "support_size",
    "grid_size",
    "dictionary_mode",
    "group_omp",
    "offgrid_refinement",
    "refinement_objective",
    "model_variant",
    "selected_support",
    "selected_group_count",
    "selected_panel_count",
    "selected_panels",
    "unique_panel_constraint",
    "expanded_support_count",
    "active_coefficient_count",
    "active_panel_count",
    "als_geometry_mapping",
    "als_geometry_assignment",
    "als_geometry_unique_panel_count",
    "als_geometry_coarse_score",
    "als_geometry_refined_score",
    "als_geometry_refined_factor_score",
    "als_geometry_refined_clock_std_ns",
    "als_geometry_refinement_used",
    "als_geometry_refinement_success",
    "als_geometry_refinement_evals",
    "peb_position_m",
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
    "peb_con_minus_free_relative_frobenius",
    "peb_hyb_minus_free_min_eig",
    "peb_ordering_ok",
    "peb_is_data_only",
    "peb_uses_regularization",
    "nuisance_model",
    "clock_eliminated",
    "efim_condition_number",
    "efim_parameter_order",
    "peb_reference_type",
    "peb_reference_data_hash",
    "batch_size",
    "max_batch_memory_mb",
    "num_batches",
    "baseline_backend",
    "gpu_used",
    "gpu_device",
    "gpu_num_batches",
    "gpu_batch_size",
    "cache_enabled",
    "cache_hits",
    "cache_misses",
    "cache_estimated_bytes",
    "scoring_time_s",
    "backend_warning",
    "factorized_scoring",
    "score_mode",
    "coarse_backend",
    "coarse_gpu_used",
    "local_refinement_backend",
    "local_refinement_gpu_used",
    "wls_backend",
    "mixed_backend",
    "selected_grid_index",
    "momp_group_omp_enabled",
    "momp_score_mode",
    "momp_group_size",
    "momp_max_groups",
    "momp_selected_groups",
    "momp_local_refinement_used",
    "momp_refinement_levels",
    "momp_refinement_num_evals",
    "momp_coordinate_sweeps",
    "momp_coordinate_evaluations",
    "momp_source_competitions",
    "cartesian_dictionary_materialized",
    "range_dictionary_used",
    "nf_mmpsr_cc_metric",
    "nf_mmpsr_top_candidates",
    "nf_mmpsr_local_refinement_used",
    "nf_mmpsr_refinement_levels",
    "nf_mmpsr_refinement_num_evals",
    "nf_mmpsr_coarse_best_score",
    "nf_mmpsr_refined_best_score",
    "reference_algorithm",
    "cpd_omp_adapted_used",
    "cpd_rank1_sequential",
    "near_field_l1_refinement_used",
    "sage_enabled",
    "sage_iterations",
    "sage_num_evals",
    "local_grid_enabled",
    "local_grid_iterations",
    "local_grid_num_evals",
    "wls_enabled",
    "wls_final_cost",
    "wls_weight_model",
    "subris_mode",
    "subris_shape",
    "subris_fallback_used",
    "adaptation_note",
    "rss_mb_before",
    "rss_mb_peak",
    "rss_mb_after",
    "warning",
    "selected_branch",
    "proposed_stage2_policy",
    "ngc_policy_active",
    "ngc_rescue_requested",
    "rescue_requested",
    "ngc_selected_by",
    "ngc_final_unreliable",
    "global_vp_backend",
    "global_vp_gpu_used",
    "global_vp_gpu_device",
    "global_vp_objective_backend",
    "global_vp_linear_solve_backend",
    "vp_dictionary_mode",
    "vp_dictionary_mode_requested",
    "vp_jacobian_mode",
    "vp_matrix_free_enabled",
    "vp_matrix_free_fallback_reason",
    "vp_precontract_static_modes",
    "vp_factor_cache_hits",
    "vp_factor_cache_misses",
    "vp_matrix_free_num_objective_calls",
    "vp_matrix_free_debug_num_compares",
    "vp_matrix_free_debug_rel_G_diff",
    "vp_matrix_free_debug_rel_b_diff",
    "vp_matrix_free_debug_rel_x_hat_diff",
    "vp_matrix_free_debug_rel_regularized_objective_diff",
    "vp_matrix_free_debug_rel_gradient_diff",
]
BASELINE_LABELS = {
    "als_cpd": "ALS-CPD + Joint Mapping (Adapted)",
    "ris_vbi_sbl": "VBI/SBL joint localization + channel reconstruction",
    "nf_ris_groupomp_localgrid_wls": "NF-RIS CPD-OMP-SAGE-WLS adaptation",
    "proposed": "Legacy NGC–LG-RDC (archived comparator)",
    "scaled_4d": "Scale-normalized 4-D Jones-VP",
    "mksc_ccop": "Proposed MKSC-GI-balanced + CCOP-JVP",
    "constrained_jones_peb": "Constrained-Jones PEB",
    "peb": "Data-only Free-Jones PEB",
}

ESTIMATOR_BASELINES = tuple(
    baseline
    for baseline in BASELINE_LABELS
    if baseline not in {"peb", "constrained_jones_peb"}
)

# These are the only standalone comparison methods whose implementation uses
# ``baselines.backend_config`` and can therefore enter the dedicated CuPy lane.
# Keep this explicit: a method must not be moved to the GPU lane merely because
# it happens to run in a process where a CUDA device is visible.
GPU_EXTERNAL_BASELINES = frozenset(
    set()
)


_WORKER_BLAS_THREADS = 1
_WORKER_RESPECT_EXISTING_BLAS_ENV = False
_WORKER_TRIM_MEMORY = True
_WORKER_PROFILE_MEMORY = False


def _benchmark_memory_snapshot_mb() -> float:
    """Return RSS in MiB, with a Linux /proc fallback when psutil is absent."""
    rss_mb = memory_snapshot_mb()
    if np.isfinite(rss_mb):
        return float(rss_mb)
    try:
        with pathlib.Path("/proc/self/status").open() as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, IndexError, TypeError, ValueError):
        pass
    return float("nan")


def _init_worker(
    blas_threads: int,
    respect_existing_blas_env: bool = False,
    trim_memory_enabled: bool = True,
    profile_memory: bool = False,
) -> None:
    global _WORKER_BLAS_THREADS, _WORKER_RESPECT_EXISTING_BLAS_ENV
    global _WORKER_TRIM_MEMORY, _WORKER_PROFILE_MEMORY
    _WORKER_BLAS_THREADS = int(blas_threads)
    _WORKER_RESPECT_EXISTING_BLAS_ENV = bool(respect_existing_blas_env)
    _WORKER_TRIM_MEMORY = bool(trim_memory_enabled)
    _WORKER_PROFILE_MEMORY = bool(profile_memory)
    apply_thread_limits(
        _WORKER_BLAS_THREADS,
        respect_existing=_WORKER_RESPECT_EXISTING_BLAS_ENV,
    )


def parse_snr_grid(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def parse_baselines(value: str) -> list[str]:
    baselines = [item.strip() for item in value.split(",") if item.strip()]
    allowed = set(BASELINE_LABELS)
    unknown = [item for item in baselines if item not in allowed]
    if unknown:
        raise ValueError(f"unknown baselines: {unknown}")
    return baselines


def _trial_seed(seed: int, trial_id: int) -> int:
    sequence = np.random.SeedSequence(int(seed))
    return int(sequence.spawn(int(trial_id) + 1)[int(trial_id)].generate_state(1, dtype=np.uint32)[0])


def _proposed_policy_log_fragment(config: dict) -> str:
    return (
        f"proposed_stage2_policy={config.get('proposed_stage2_policy', '')} "
        f"ngc_lambda_ris={config.get('ngc_lambda_ris', 1.0)} "
        f"ngc_clock_green_quantile={config.get('ngc_clock_green_quantile', 0.99)} "
        f"ngc_clock_red_quantile={config.get('ngc_clock_red_quantile', 0.999)} "
        "rescue_accept_min_rel_improvement="
        f"{config.get('rescue_accept_min_rel_improvement', '')} "
        "rescue_accept_min_abs_improvement="
        f"{config.get('rescue_accept_min_abs_improvement', '')}"
    )


def _apply_grid_profile(config: dict, profile: str) -> dict:
    baselines = dict(config.get("baselines", {}))
    if profile == "coarse":
        baselines.update(
            {
                "als_cpd": {
                    "position_grid_shape": (3, 3, 2),
                    "geometry_refinement_starts": 2,
                    "geometry_refinement_maxiter": 30,
                },
                "ris_vbi_sbl": {"nf_grid_x": 7, "nf_grid_y": 7, "nf_grid_z": 5, "delay_grid_size": 61, "vbi_max_iter": 20, "vbi_refine_maxiter": 80},
                "nf_ris_groupomp_localgrid_wls": {"direction_grid_size": 5, "range_grid_size": 5, "delay_grid_size": 5, "max_groups": config["K"], "cpd_max_iter": 10, "sage_enabled": True, "sage_iterations": 1, "sage_maxiter": 5, "wls_enabled": True, "wls_max_nfev": 20},
            }
        )
    elif profile == "medium":
        baselines.update(
            {
                "als_cpd": {
                    "position_grid_shape": (5, 5, 3),
                    "geometry_refinement_starts": 4,
                    "geometry_refinement_maxiter": 60,
                },
                "ris_vbi_sbl": {"nf_grid_x": 9, "nf_grid_y": 9, "nf_grid_z": 7, "delay_grid_size": 121, "vbi_max_iter": 40, "vbi_refine_maxiter": 200},
                "nf_ris_groupomp_localgrid_wls": {"direction_grid_size": 31, "range_grid_size": 31, "delay_grid_size": 41, "max_groups": config["K"], "cpd_max_iter": 80, "sage_enabled": True, "sage_iterations": 2, "sage_maxiter": 30, "wls_enabled": True, "wls_max_nfev": 100},
            }
        )
    elif profile == "fine":
        baselines.update(
            {
                "als_cpd": {
                    "position_grid_shape": (7, 7, 5),
                    "geometry_refinement_starts": 8,
                    "geometry_refinement_maxiter": 80,
                },
                "ris_vbi_sbl": {"nf_grid_x": 11, "nf_grid_y": 11, "nf_grid_z": 9, "delay_grid_size": 161, "vbi_max_iter": 60, "vbi_refine_maxiter": 300},
                "nf_ris_groupomp_localgrid_wls": {"direction_grid_size": 45, "range_grid_size": 45, "delay_grid_size": 61, "max_groups": config["K"], "cpd_max_iter": 120, "sage_enabled": True, "sage_iterations": 3, "sage_maxiter": 50, "wls_enabled": True, "wls_max_nfev": 150},
            }
        )
    else:
        raise ValueError(f"unknown grid profile {profile!r}")
    config["baselines"] = baselines
    return config


def make_config(
    seed: int,
    snr_db: float,
    paper_k: int,
    grid_profile: str,
    *,
    strict_ris_geometry: bool = False,
) -> dict:
    config = default_config()
    config["seed"] = int(seed)
    config["SNR_dB"] = float(snr_db)
    config["receiver_mode"] = "full_6d"
    config["print_progress"] = False
    config["verbose_stage2"] = False
    config["run_full_legacy_comparison"] = False
    config["stage2_adaptive"] = True
    config["stage2_rescue_type"] = "ris_only"
    config["proposed_stage2_policy"] = "ngc_certified_ris_only"
    config["rescue_accept_min_rel_improvement"] = 0.0
    config["rescue_accept_min_abs_improvement"] = 1.0e-8
    # Drop the legacy fixed-count coarse RIS dictionary from acquisition.  After
    # the Nyquist beam-space fix it is redundant in this configuration: over 128
    # paired trials (32 per SNR at -20/-15/-10/0 dB) it won 9 of 1152 argmins
    # (0.78%), changed zero outcomes, moved the position estimate by a median
    # 8.7e-10 m, and cost 2.32 s of a 4.23 s Stage-I.  It also does not rescue
    # low SNR: at -20 dB both arms sit at an identical 46.88% outlier rate.
    # Scoped here rather than in default_config() because the redundancy is a
    # property of this array size and refine-start budget, not of the estimator.
    config["ris_search"] = dict(config["ris_search"])
    config["ris_search"]["coarse_codebook_mode"] = "beamspace_only"
    if strict_ris_geometry and np.asarray(config.get("ris_centers", [])).shape[0] < int(paper_k):
        raise ValueError(
            f"--strict-ris-geometry requested K={paper_k}, but config has only "
            f"{np.asarray(config.get('ris_centers', [])).shape[0]} RIS centers"
        )
    set_number_of_ris_paths(config, int(paper_k))
    _apply_grid_profile(config, grid_profile)
    return config


def _proposed_row(data: dict, config: dict, trial_id: int, baseline: str) -> dict[str, Any]:
    start = time.perf_counter()
    result = run_single_proposed_diagnostic(config, data_override=data)
    runtime_s = time.perf_counter() - start
    final = result.get("final", {})
    y_hat = final.get("Y_hat")
    p_hat = final.get("p_u")
    raw_obj = final.get("raw_objective_final", final.get("raw_objective", float("nan")))
    baseline_result = BaselineResult(
        name=baseline,
        p_u=None if p_hat is None else np.asarray(p_hat, dtype=float),
        delta_t=final.get("delta_t"),
        Y_hat=y_hat,
        raw_objective_final=float(raw_obj) if raw_obj is not None else float("nan"),
        components=final.get("components", {}),
        selected_support=[],
        runtime_s=runtime_s,
        diagnostics={
            "dictionary_mode": "proposed_ngc_adaptive_jones_vp",
            "clock_output_semantics": "native_joint_common_clock",
            **proposed_trace_diagnostics(result),
        },
    )
    return make_baseline_row(
        baseline_result,
        data,
        config,
        baseline=baseline,
        trial_id=trial_id,
        seed=int(config["seed"]),
        snr_db=float(config["SNR_dB"]),
    )


def _final_mksc_ccop_row(
    data: dict,
    config: dict,
    trial_id: int,
    baseline: str,
    cache: Stage1Cache | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen final-paper route on the shared benchmark data."""
    final_config = apply_final_paper_options(config)
    variant = "scaled_4d" if baseline == "scaled_4d" else "proposed"
    route_row = run_paper_variant(
        variant,
        data=data,
        config=final_config,
        cache=cache,
        suite="external_benchmark",
        x_name="snr_db",
        x_value=float(config["SNR_dB"]),
        trial_id=int(trial_id),
    )
    if bool(route_row["failed"]):
        raise RuntimeError(str(route_row["error"]))
    clock_error_ns = float(route_row["clock_error_ns"])
    clock_threshold_ns = float(
        config.get("benchmark_clock_catastrophic_threshold_ns", 1.0)
    )
    clock_invalid = not np.isfinite(clock_error_ns)
    return {
        "baseline": baseline,
        "trial_id": int(trial_id),
        "seed": int(config["seed"]),
        "snr_db": float(config["SNR_dB"]),
        "K": int(config["K"]),
        "data_hash": data_hash(data),
        "y_noisy_hash": y_noisy_hash(data),
        "failed": False,
        "error": "",
        "runtime_s": float(route_row["deployment_runtime_s"]),
        "position_error_m": float(route_row["position_error_m"]),
        "position_rmse_m": float(route_row["position_error_m"]),
        "y_nmse": float(route_row["channel_nmse"]),
        "range_rmse_m": float("nan"),
        "tau_rmse_s": float("nan"),
        "raw_objective_final": float(route_row["raw_objective_final"]),
        "support_size": int(2 * config["K"]),
        "grid_size": "",
        "dictionary_mode": (
            "scaled_4d_jones_vp"
            if baseline == "scaled_4d"
            else "mksc_gi_balanced_ccop_jvp"
        ),
        "selected_support": "",
        "reference_algorithm": "frozen_final_paper_route",
        "global_vp_backend": "numpy_cpu",
        "global_vp_gpu_used": False,
        "clock_error_ns": clock_error_ns,
        "clock_invalid": clock_invalid,
        "clock_invalid_reason": "native_clock_unavailable" if clock_invalid else "",
        "clock_catastrophic": bool(
            not clock_invalid and clock_error_ns > clock_threshold_ns
        ),
        "clock_catastrophic_threshold_ns": clock_threshold_ns,
        "clock_extraction_rule": "native_common_clock_parameter",
        "clock_delay_source": "baseline_native_delta_t",
        "clock_certified": route_row["clock_certified"],
        "stage1_runtime_s": float(route_row["stage1_runtime_s"]),
        "global_vp_runtime_s": float(route_row["stage3_runtime_s"]),
        "total_runtime_s": float(route_row["deployment_runtime_s"]),
        "stage1_output_hash": str(route_row["stage1_output_hash"]),
        "candidate_hash": str(route_row["candidate_hash"]),
        "warning": "",
    }


def _peb_row(
    data: dict,
    config: dict,
    trial_id: int,
    *,
    baseline: str = "peb",
) -> dict[str, Any]:
    start = time.perf_counter()
    metrics = _peb_from_efim(data, config)
    if baseline == "constrained_jones_peb":
        peb_value = metrics.get("peb_constrained_jones_m", float("nan"))
        peb_variant = "constrained_jones_peb"
        bound_type = "constrained"
    else:
        peb_value = metrics.get("peb_free_jones_m", metrics.get("peb_position_m", float("nan")))
        peb_variant = "free_jones_peb"
        bound_type = "free"
    runtime_s = time.perf_counter() - start
    return {
        "baseline": baseline,
        "trial_id": int(trial_id),
        "seed": int(config["seed"]),
        "snr_db": float(config["SNR_dB"]),
        "K": int(config["K"]),
        "data_hash": data_hash(data),
        "y_noisy_hash": y_noisy_hash(data),
        "failed": False,
        "error": "",
        "runtime_s": runtime_s,
        "position_error_m": float("nan"),
        "position_rmse_m": float("nan"),
        "y_nmse": float("nan"),
        "range_rmse_m": float("nan"),
        "tau_rmse_s": float("nan"),
        "raw_objective_final": float("nan"),
        "support_size": 0,
        "grid_size": "",
        "dictionary_mode": (
            "constrained_jones_efim_peb"
            if baseline == "constrained_jones_peb"
            else "data_only_free_jones_efim_peb"
        ),
        "selected_support": "",
        "peb_position_m": peb_value,
        "peb_free_jones_m": metrics.get("peb_free_jones_m", float("nan")),
        "peb_constrained_jones_m": metrics.get("peb_constrained_jones_m", float("nan")),
        "peb_anchored_jones_m": metrics.get("peb_anchored_jones_m", float("nan")),
        "peb_variant": peb_variant,
        "jones_bound_type": bound_type,
        "constrained_jones_peb_m": metrics.get("constrained_jones_peb_m", float("nan")),
        "anchored_jones_peb_m": metrics.get("anchored_jones_peb_m", float("nan")),
        "free_jones_peb_m": metrics.get("free_jones_peb_m", float("nan")),
        "peb_fim_rank_chi_free": metrics.get("peb_fim_rank_chi_free", ""),
        "peb_fim_rank_chi_constrained": metrics.get("peb_fim_rank_chi_constrained", ""),
        "peb_fim_rank_chi_anchored": metrics.get("peb_fim_rank_chi_anchored", ""),
        "peb_fim_cond_chi_free": metrics.get("peb_fim_cond_chi_free", float("nan")),
        "peb_fim_cond_chi_constrained": metrics.get("peb_fim_cond_chi_constrained", float("nan")),
        "peb_fim_cond_chi_anchored": metrics.get("peb_fim_cond_chi_anchored", float("nan")),
        "peb_clock_schur_used": metrics.get("peb_clock_schur_used", ""),
        "peb_rank_deficient": metrics.get("peb_rank_deficient", ""),
        "anchored_prior_scaling": metrics.get("anchored_prior_scaling", ""),
        "anchored_prior_lambda": metrics.get("anchored_prior_lambda", float("nan")),
        "anchored_prior_precision_norm": metrics.get("anchored_prior_precision_norm", float("nan")),
        "peb_free_projection_schur_relerr": metrics.get("peb_free_projection_schur_relerr", float("nan")),
        "peb_con_minus_free_min_eig": metrics.get("peb_con_minus_free_min_eig", float("nan")),
        "peb_con_minus_free_relative_frobenius": metrics.get(
            "peb_con_minus_free_relative_frobenius", float("nan")
        ),
        "peb_hyb_minus_free_min_eig": metrics.get("peb_hyb_minus_free_min_eig", float("nan")),
        "peb_ordering_ok": metrics.get("peb_ordering_ok", ""),
        "peb_is_data_only": bool(metrics.get("peb_is_data_only", True)),
        "peb_uses_regularization": bool(metrics.get("peb_uses_regularization", False)),
        "nuisance_model": str(metrics.get("nuisance_model", "jones_linear")),
        "clock_eliminated": bool(metrics.get("clock_eliminated", True)),
        "efim_condition_number": metrics.get("efim_condition_number", float("inf")),
        "efim_parameter_order": metrics.get("efim_parameter_order", []),
        "peb_reference_type": metrics.get(
            "peb_reference_type", "matched_model"
        ),
        "peb_reference_data_hash": data_hash(data),
        "warning": metrics.get("warning", ""),
    }


BASELINE_RUNNERS = {
    "als_cpd": run_als_cpd_baseline,
    "ris_vbi_sbl": run_ris_vbi_sbl_baseline,
    "nf_ris_groupomp_localgrid_wls": run_nf_ris_groupomp_localgrid_wls_baseline,
}


def _failure_row(
    baseline: str,
    trial_id: int,
    config: dict,
    exc: BaseException,
    data: dict | None = None,
) -> dict[str, Any]:
    clock_evaluated = baseline in ESTIMATOR_BASELINES
    return {
        "baseline": baseline,
        "trial_id": int(trial_id),
        "seed": int(config.get("seed", 0)),
        "snr_db": float(config.get("SNR_dB", float("nan"))),
        "K": int(config.get("K", 0)),
        "data_hash": data_hash(data) if data is not None else "",
        "y_noisy_hash": y_noisy_hash(data) if data is not None else "",
        "failed": True,
        "error": f"{type(exc).__name__}: {exc}",
        "runtime_s": float("nan"),
        "position_error_m": float("nan"),
        "position_rmse_m": float("nan"),
        "y_nmse": float("nan"),
        "range_rmse_m": float("nan"),
        "tau_rmse_s": float("nan"),
        "clock_error_ns": float("nan"),
        "clock_invalid": True if clock_evaluated else "",
        "clock_invalid_reason": "baseline_failed" if clock_evaluated else "",
        "clock_catastrophic": False if clock_evaluated else "",
        "clock_catastrophic_threshold_ns": config.get(
            "benchmark_clock_catastrophic_threshold_ns", 1.0
        ),
        "raw_objective_final": float("nan"),
        "support_size": 0,
        "grid_size": "",
        "dictionary_mode": "",
        "selected_support": "",
        "peb_position_m": float("nan"),
        "peb_is_data_only": "",
        "peb_uses_regularization": "",
        "nuisance_model": "",
        "clock_eliminated": "",
        "efim_condition_number": float("nan"),
        "efim_parameter_order": "",
        "peb_reference_type": "",
        "peb_reference_data_hash": "",
        "warning": "",
    }


def _config_for_trial_snr(task: dict[str, Any], snr_db: float) -> dict:
    """Build one SNR-specific config without changing trial-static settings."""
    config = make_config(
        seed=int(task["seed"]),
        snr_db=float(snr_db),
        paper_k=int(task["paper_k"]),
        grid_profile=str(task["grid_profile"]),
        strict_ris_geometry=bool(task.get("strict_ris_geometry", False)),
    )
    config["crb"] = dict(task.get("crb", {}))
    config["baselines"]["backend_config"] = dict(task.get("backend_config", {}))
    config["baselines"]["trim_memory"] = bool(
        task.get("trim_memory", _WORKER_TRIM_MEMORY)
    )
    tier = task.get("baseline_refinement_tier")
    if tier:
        config["baselines"]["refinement_tier"] = str(tier)
    config["benchmark_clock_catastrophic_threshold_ns"] = float(
        task.get("clock_catastrophic_threshold_ns", 1.0)
    )
    config.setdefault("global_vp", {}).update(dict(task.get("global_vp", {})))
    return config


def _iter_shared_scene_snr_data(configs: list[dict[str, Any]]):
    """Yield the legacy per-SNR data while generating trial-static arrays once."""
    if not configs:
        return
    reference = configs[0]
    data_start = time.perf_counter()
    rng = np.random.default_rng(int(reference["seed"]))
    scene = generate_scene(reference, rng)
    true_components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(true_components, scene["beta_true"])
    noise_rng_state = copy.deepcopy(rng.bit_generator.state)
    static_generation_s = time.perf_counter() - data_start

    hankel_start = time.perf_counter()
    z_true = hankelize_frequency(y_true, scene["P"])
    static_hankelization_s = time.perf_counter() - hankel_start
    for index, config in enumerate(configs):
        noise_start = time.perf_counter()
        noise_rng = np.random.default_rng()
        noise_rng.bit_generator.state = copy.deepcopy(noise_rng_state)
        y_noisy, noise_variance = add_awgn(
            y_true,
            float(config["SNR_dB"]),
            noise_rng,
            active_mask=scene.get("evs_observation_mask"),
        )
        noise_generation_s = time.perf_counter() - noise_start
        noisy_hankel_start = time.perf_counter()
        z_noisy = hankelize_frequency(y_noisy, scene["P"])
        noisy_hankelization_s = time.perf_counter() - noisy_hankel_start
        yield {
            "scene": scene,
            "true_components": true_components,
            "Y_true": y_true,
            "Y_noisy": y_noisy,
            "Z_true": z_true,
            "Z_noisy": z_noisy,
            "noise_variance": noise_variance,
            "timing": {
                "data_generation": noise_generation_s
                + (static_generation_s if index == 0 else 0.0),
                "hankelization": noisy_hankelization_s
                + (static_hankelization_s if index == 0 else 0.0),
            },
        }


def _run_trial_methods(
    task: dict[str, Any],
    config: dict,
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate all requested methods on one SNR-specific realization."""
    rows: list[dict[str, Any]] = []
    final_cache = (
        Stage1Cache(data, apply_final_paper_options(config))
        if any(name in {"scaled_4d", "mksc_ccop"} for name in task["baselines"])
        else None
    )
    for baseline in task["baselines"]:
        profile_memory = bool(
            task.get("profile_memory", _WORKER_PROFILE_MEMORY)
        )
        rss_before = (
            _benchmark_memory_snapshot_mb()
            if profile_memory
            else float("nan")
        )
        with PeakRssSampler(profile_memory) as memory_sampler:
            try:
                with thread_limit_context(
                    int(task.get("blas_threads", _WORKER_BLAS_THREADS))
                ):
                    if baseline in BASELINE_RUNNERS:
                        result = BASELINE_RUNNERS[baseline](data, config)
                        row = make_baseline_row(
                            result,
                            data,
                            config,
                            baseline=baseline,
                            trial_id=int(task["trial_id"]),
                            seed=int(config["seed"]),
                            snr_db=float(config["SNR_dB"]),
                        )
                        del result
                    elif baseline == "proposed":
                        row = _proposed_row(
                            data, config, int(task["trial_id"]), baseline
                        )
                    elif baseline in {"scaled_4d", "mksc_ccop"}:
                        row = _final_mksc_ccop_row(
                            data,
                            config,
                            int(task["trial_id"]),
                            baseline,
                            cache=final_cache,
                        )
                    elif baseline in {"peb", "constrained_jones_peb"}:
                        row = _peb_row(
                            data, config, int(task["trial_id"]), baseline=baseline
                        )
                    else:
                        raise ValueError(f"unknown baseline {baseline!r}")
            except Exception as exc:  # noqa: BLE001 - failure is an MC outcome.
                row = _failure_row(
                    baseline, int(task["trial_id"]), config, exc, data
                )
        if bool(task.get("trim_memory", _WORKER_TRIM_MEMORY)):
            trim_memory(
                release_gpu=bool(task.get("trim_gpu_memory_pool", True))
            )
        rss_after = (
            _benchmark_memory_snapshot_mb()
            if profile_memory
            else float("nan")
        )
        row["rss_mb_before"] = rss_before
        row["rss_mb_peak"] = memory_sampler.peak_mb
        row["rss_mb_after"] = rss_after
        rows.append(assert_row_is_light(row))
    return rows


def _run_trial_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    apply_thread_limits(
        int(task.get("blas_threads", _WORKER_BLAS_THREADS)),
        respect_existing=bool(
            task.get(
                "respect_existing_blas_env",
                _WORKER_RESPECT_EXISTING_BLAS_ENV,
            )
        ),
    )
    snr_grid = [float(value) for value in task.get("snr_grid", [task["snr_db"]])]
    configs = [_config_for_trial_snr(task, snr_db) for snr_db in snr_grid]
    if "snr_grid" in task:
        data_iter = _iter_shared_scene_snr_data(configs)
    else:
        data_iter = (_make_data(configs[0]),)

    rows: list[dict[str, Any]] = []
    for config, data in zip(configs, data_iter):
        rows.extend(_run_trial_methods(task, config, data))
        del data
        if bool(task.get("trim_memory", _WORKER_TRIM_MEMORY)):
            trim_memory(
                release_gpu=bool(task.get("trim_gpu_memory_pool", True))
            )
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class StreamingCsvWriter:
    """Write row batches to a temporary CSV and atomically publish it."""

    def __init__(self, final_path: pathlib.Path, fieldnames: list[str]):
        self.final_path = pathlib.Path(final_path)
        self.tmp_path = self.final_path.with_name(f"{self.final_path.name}.tmp")
        self.fieldnames = fieldnames
        self.handle: Any | None = None
        self.writer: csv.DictWriter | None = None

    def __enter__(self) -> "StreamingCsvWriter":
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.tmp_path.open("w", newline="")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=self.fieldnames,
            extrasaction="ignore",
        )
        self.writer.writeheader()
        self.handle.flush()
        return self

    def writerows(self, rows: Iterable[dict[str, Any]]) -> None:
        if self.writer is None or self.handle is None:
            raise RuntimeError("StreamingCsvWriter is not open")
        self.writer.writerows(rows)
        self.handle.flush()

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


def summarize_csv(
    path: pathlib.Path,
    outlier_threshold_m: float | None = None,
) -> list[dict[str, Any]]:
    """Summarize a trial CSV without retaining complete rows."""
    groups: dict[tuple[str, float], dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["baseline"]), _to_float(row["snr_db"]))
            group = groups.setdefault(
                key,
                {
                    "n": 0,
                    "success": 0,
                    "position_error": [],
                    "nmse": [],
                    "clock": [],
                    "clock_certificate": [],
                    "clock_invalid": [],
                    "clock_catastrophic": [],
                    "peb": [],
                    "runtime": [],
                },
            )
            group["n"] += 1
            clock_invalid = _to_bool_or_none(row.get("clock_invalid"))
            clock_catastrophic = _to_bool_or_none(
                row.get("clock_catastrophic")
            )
            if clock_invalid is not None:
                group["clock_invalid"].append(clock_invalid)
                if not clock_invalid and clock_catastrophic is not None:
                    group["clock_catastrophic"].append(clock_catastrophic)
            if str(row.get("failed")).lower() == "true":
                continue
            group["success"] += 1
            group["position_error"].append(_position_error_from_row(row))
            group["nmse"].append(_to_float(row.get("y_nmse")))
            group["clock"].append(_to_float(row.get("clock_error_ns")))
            certificate = str(row.get("clock_certified", "")).lower()
            if certificate in {"true", "false"}:
                group["clock_certificate"].append(certificate == "true")
            group["peb"].append(_to_float(row.get("peb_position_m")))
            group["runtime"].append(_to_float(row.get("runtime_s")))

    summary: list[dict[str, Any]] = []
    for (baseline, snr_db), group in sorted(groups.items()):
        position_stats = _summary_stats(group["position_error"])
        nmse_stats = _summary_stats(group["nmse"])
        clock_stats = _summary_stats(group["clock"])
        clock_rates = _clock_rate_summary(
            group["clock_invalid"], group["clock_catastrophic"]
        )
        peb_stats = _summary_stats(group["peb"])
        runtime_stats = _summary_stats(group["runtime"], percentiles=False)
        finite_position = np.asarray(group["position_error"], dtype=float)
        finite_position = finite_position[np.isfinite(finite_position)]
        finite_peb = np.asarray(group["peb"], dtype=float)
        finite_peb = finite_peb[np.isfinite(finite_peb)]
        outlier_rate = float("nan")
        outlier_count = 0
        if outlier_threshold_m is not None and finite_position.size:
            outlier_count = int(
                np.sum(finite_position > float(outlier_threshold_m))
            )
            outlier_rate = float(
                outlier_count / finite_position.size
            )
        outlier_ci_low, outlier_ci_high, outlier_ci_method = _binomial_interval(
            outlier_count, int(finite_position.size)
        )
        conditional_position = (
            finite_position[finite_position <= float(outlier_threshold_m)]
            if outlier_threshold_m is not None
            else np.asarray([], dtype=float)
        )
        catastrophic_count = int(group["n"] - group["success"] + outlier_count)
        catastrophic_low, catastrophic_high, catastrophic_method = _binomial_interval(
            catastrophic_count, int(group["n"])
        )
        row_summary = {
            "baseline": baseline,
            "snr_db": snr_db,
            "n": int(group["n"]),
            "success_rate": float(group["success"] / max(group["n"], 1)),
            "outlier_rate": outlier_rate,
            "outlier_ci_low": outlier_ci_low,
            "outlier_ci_high": outlier_ci_high,
            "outlier_ci_method": outlier_ci_method,
            "catastrophic_rate": float(catastrophic_count / max(group["n"], 1)),
            "catastrophic_ci_low": catastrophic_low,
            "catastrophic_ci_high": catastrophic_high,
            "catastrophic_ci_method": catastrophic_method,
            "runtime_s_mean": runtime_stats["mean"],
            "position_rmse_m": _rms(finite_position),
            "position_mean_error_m": position_stats["mean"],
            "position_conditional_rmse_m": _rms(conditional_position),
            "peb_position_m_rms": _rms(finite_peb),
            "clock_rmse_ns": _rms(group["clock"]),
            "clock_median_abs_error_ns": clock_stats["median"],
            "clock_p95_abs_error_ns": clock_stats["p95"],
            "clock_certificate_rate": (
                float(np.mean(group["clock_certificate"]))
                if group["clock_certificate"]
                else float("nan")
            ),
            **clock_rates,
        }
        for name, value in position_stats.items():
            row_summary[f"position_error_m_{name}"] = value
            # Retained as a legacy alias; these are distribution statistics
            # of single-realization errors, not Monte Carlo RMSE statistics.
            row_summary[f"position_rmse_m_{name}"] = value
        for name, value in nmse_stats.items():
            row_summary[f"y_nmse_{name}"] = value
        for name, value in clock_stats.items():
            row_summary[f"clock_error_ns_{name}"] = value
        for name, value in peb_stats.items():
            row_summary[f"peb_position_m_{name}"] = value
        summary.append(row_summary)
    return summary


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _position_error_from_row(row: Mapping[str, Any]) -> float:
    value = _to_float(row.get("position_error_m"))
    if np.isfinite(value):
        return value
    return _to_float(row.get("position_rmse_m"))


def _rms(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return (
        float(np.sqrt(np.mean(array**2)))
        if array.size
        else float("nan")
    )


def _binomial_interval(successes: int, total: int, alpha: float = 0.05):
    if total <= 0:
        return float("nan"), float("nan"), "unavailable"
    try:
        from scipy.stats import beta

        low = 0.0 if successes == 0 else float(
            beta.ppf(alpha / 2.0, successes, total - successes + 1)
        )
        high = 1.0 if successes == total else float(
            beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes)
        )
        return low, high, "Clopper-Pearson exact"
    except ImportError:
        return float("nan"), float("nan"), "scipy_unavailable"


def _clock_rate_summary(
    invalid_flags: list[bool],
    catastrophic_flags: list[bool],
) -> dict[str, Any]:
    evaluated_count = len(invalid_flags)
    invalid_count = int(sum(invalid_flags))
    valid_count = int(evaluated_count - invalid_count)
    catastrophic_count = int(sum(catastrophic_flags))
    catastrophic_total = len(catastrophic_flags)
    combined_count = int(invalid_count + catastrophic_count)
    invalid_low, invalid_high, invalid_method = _binomial_interval(
        invalid_count, evaluated_count
    )
    catastrophic_low, catastrophic_high, catastrophic_method = _binomial_interval(
        catastrophic_count, catastrophic_total
    )
    combined_low, combined_high, combined_method = _binomial_interval(
        combined_count, evaluated_count
    )
    return {
        "clock_evaluated_count": evaluated_count,
        "clock_valid_count": valid_count,
        "clock_invalid_count": invalid_count,
        "clock_invalid_rate": (
            float(invalid_count / evaluated_count)
            if evaluated_count
            else float("nan")
        ),
        "clock_invalid_ci_low": invalid_low,
        "clock_invalid_ci_high": invalid_high,
        "clock_invalid_ci_method": invalid_method,
        "clock_catastrophic_count": catastrophic_count,
        "clock_catastrophic_rate": (
            float(catastrophic_count / catastrophic_total)
            if catastrophic_total
            else float("nan")
        ),
        "clock_catastrophic_ci_low": catastrophic_low,
        "clock_catastrophic_ci_high": catastrophic_high,
        "clock_catastrophic_ci_method": catastrophic_method,
        "clock_catastrophic_or_invalid_count": combined_count,
        "clock_catastrophic_or_invalid_rate": (
            float(combined_count / evaluated_count)
            if evaluated_count
            else float("nan")
        ),
        "clock_catastrophic_or_invalid_ci_low": combined_low,
        "clock_catastrophic_or_invalid_ci_high": combined_high,
        "clock_catastrophic_or_invalid_ci_method": combined_method,
    }


def _summary_stats(values: list[float], percentiles: bool = True) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        stats = {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
        if percentiles:
            stats.update(
                {
                    "p10": float("nan"),
                    "p90": float("nan"),
                    "p95": float("nan"),
                }
            )
        return stats
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }
    if percentiles:
        stats.update(
            {
                "p10": float(np.percentile(arr, 10.0)),
                "p90": float(np.percentile(arr, 90.0)),
                "p95": float(np.percentile(arr, 95.0)),
            }
        )
    return stats


def get_plot_metric(baseline: str, plot_kind: str) -> str | None:
    if baseline in {"peb", "constrained_jones_peb"}:
        return (
            "peb_position_m"
            if plot_kind in {"rmse", "conditional_rmse"}
            else None
        )
    if plot_kind in {"rmse", "conditional_rmse"}:
        return "position_rmse_m"
    if plot_kind == "nmse":
        return "y_nmse"
    if plot_kind == "outlier":
        return "outlier_rate"
    if plot_kind == "position_p95":
        return "position_rmse_m"
    if plot_kind == "nmse_p95":
        return "y_nmse"
    raise ValueError(f"unknown plot kind {plot_kind!r}")


def summarize_rows(rows: list[dict[str, Any]], outlier_threshold_m: float | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["baseline"]), _to_float(row["snr_db"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (baseline, snr_db), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        success = [row for row in group if str(row.get("failed")).lower() != "true"]
        position_values = [_position_error_from_row(row) for row in success]
        position_stats = _summary_stats(position_values)
        nmse_stats = _summary_stats(
            [_to_float(row.get("y_nmse")) for row in success]
        )
        clock_stats = _summary_stats(
            [_to_float(row.get("clock_error_ns")) for row in success]
        )
        clock_invalid_flags: list[bool] = []
        clock_catastrophic_flags: list[bool] = []
        for row in group:
            clock_invalid = _to_bool_or_none(row.get("clock_invalid"))
            clock_catastrophic = _to_bool_or_none(
                row.get("clock_catastrophic")
            )
            if clock_invalid is None:
                continue
            clock_invalid_flags.append(clock_invalid)
            if not clock_invalid and clock_catastrophic is not None:
                clock_catastrophic_flags.append(clock_catastrophic)
        clock_rates = _clock_rate_summary(
            clock_invalid_flags, clock_catastrophic_flags
        )
        clock_certificates = [
            str(row.get("clock_certified", "")).lower() == "true"
            for row in success
            if str(row.get("clock_certified", "")).lower() in {"true", "false"}
        ]
        peb_stats = _summary_stats([_to_float(row.get("peb_position_m")) for row in success])
        runtime_stats = _summary_stats([_to_float(row.get("runtime_s")) for row in success], percentiles=False)
        outlier_rate = float("nan")
        outlier_count = 0
        finite_count = 0
        if outlier_threshold_m is not None:
            position = np.asarray(position_values, dtype=float)
            finite = position[np.isfinite(position)]
            finite_count = int(finite.size)
            outlier_count = int(np.sum(finite > float(outlier_threshold_m)))
            outlier_rate = float(outlier_count / finite.size) if finite.size else float("nan")
        outlier_ci_low, outlier_ci_high, outlier_ci_method = _binomial_interval(
            outlier_count, finite_count
        )
        catastrophic_count = int(len(group) - len(success) + outlier_count)
        catastrophic_low, catastrophic_high, catastrophic_method = _binomial_interval(
            catastrophic_count, len(group)
        )
        finite_position = np.asarray(position_values, dtype=float)
        finite_position = finite_position[np.isfinite(finite_position)]
        conditional_position = (
            finite_position[finite_position <= float(outlier_threshold_m)]
            if outlier_threshold_m is not None
            else np.asarray([], dtype=float)
        )
        finite_peb = np.asarray(
            [_to_float(row.get("peb_position_m")) for row in success],
            dtype=float,
        )
        finite_peb = finite_peb[np.isfinite(finite_peb)]
        row_summary = {
            "baseline": baseline,
            "snr_db": snr_db,
            "n": len(group),
            "success_rate": float(len(success) / max(len(group), 1)),
            "outlier_rate": outlier_rate,
            "outlier_ci_low": outlier_ci_low,
            "outlier_ci_high": outlier_ci_high,
            "outlier_ci_method": outlier_ci_method,
            "catastrophic_rate": float(catastrophic_count / max(len(group), 1)),
            "catastrophic_ci_low": catastrophic_low,
            "catastrophic_ci_high": catastrophic_high,
            "catastrophic_ci_method": catastrophic_method,
            "runtime_s_mean": runtime_stats["mean"],
            "position_rmse_m": _rms(finite_position),
            "position_mean_error_m": position_stats["mean"],
            "position_conditional_rmse_m": _rms(conditional_position),
            "peb_position_m_rms": _rms(finite_peb),
            "clock_rmse_ns": _rms(
                [_to_float(row.get("clock_error_ns")) for row in success]
            ),
            "clock_median_abs_error_ns": clock_stats["median"],
            "clock_p95_abs_error_ns": clock_stats["p95"],
            "clock_certificate_rate": (
                float(np.mean(clock_certificates))
                if clock_certificates
                else float("nan")
            ),
            **clock_rates,
        }
        for name, value in position_stats.items():
            row_summary[f"position_error_m_{name}"] = value
            # Retained as a legacy alias for existing CSV consumers.
            row_summary[f"position_rmse_m_{name}"] = value
        for name, value in nmse_stats.items():
            row_summary[f"y_nmse_{name}"] = value
        for name, value in clock_stats.items():
            row_summary[f"clock_error_ns_{name}"] = value
        for name, value in peb_stats.items():
            row_summary[f"peb_position_m_{name}"] = value
        summary.append(row_summary)
    return summary


def validate_same_data_hashes(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[int, float, int], set[str]] = {}
    for row in rows:
        if str(row.get("failed")).lower() == "true":
            continue
        key = (int(row["trial_id"]), _to_float(row["snr_db"]), int(row["K"]))
        groups.setdefault(key, set()).add(str(row.get("y_noisy_hash", "")))
    mismatches = {key: values for key, values in groups.items() if len(values) > 1}
    if mismatches:
        raise RuntimeError(f"benchmark same-data hash mismatch: {mismatches}")


def _plot(summary_rows: list[dict[str, Any]], out_dir: pathlib.Path, plot_kind: str) -> None:
    mpl_config = out_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    baselines = [name for name in BASELINE_LABELS if any(row["baseline"] == name for row in summary_rows)]
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, baseline in enumerate(baselines):
        metric = get_plot_metric(baseline, plot_kind)
        if metric is None:
            continue
        rows = [row for row in summary_rows if row["baseline"] == baseline]
        xs = np.asarray([_to_float(row["snr_db"]) for row in rows], dtype=float)
        if plot_kind == "outlier":
            summary_field = metric
        elif plot_kind in {"rmse", "conditional_rmse"}:
            summary_field = (
                "peb_position_m_rms"
                if baseline in {"peb", "constrained_jones_peb"}
                else (
                    "position_conditional_rmse_m"
                    if plot_kind == "conditional_rmse"
                    else "position_rmse_m"
                )
            )
        elif plot_kind in {"position_p95", "nmse_p95"}:
            summary_field = (
                "position_error_m_p95"
                if plot_kind == "position_p95"
                else f"{metric}_p95"
            )
        else:
            summary_field = f"{metric}_mean"
        ys = np.asarray(
            [_to_float(row.get(summary_field)) for row in rows], dtype=float
        )
        if plot_kind in {"nmse", "nmse_p95"}:
            positive = ys > 0.0
            ys[positive] = 10.0 * np.log10(ys[positive])
            ys[~positive] = np.nan
        finite = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(finite):
            continue
        order = np.argsort(xs[finite])
        linestyle = "-." if baseline == "constrained_jones_peb" else ("--" if baseline == "peb" else "-")
        is_proposed = baseline in {"proposed", "mksc_ccop"}
        ax.plot(
            xs[finite][order],
            ys[finite][order],
            marker=markers[idx % len(markers)],
            linestyle=linestyle,
            linewidth=2.5 if is_proposed else 1.5,
            label=BASELINE_LABELS[baseline],
            zorder=10 if is_proposed else 2,
        )
    ax.set_xlabel("SNR (dB)")
    ylabel = {
        "rmse": "Position RMSE (m)",
        "conditional_rmse": "Correct-basin conditional RMSE / PEB (m)",
        "nmse": "Channel NMSE (dB)",
        "outlier": "Outlier probability",
        "position_p95": "Position-error p95 (m)",
        "nmse_p95": "Channel-NMSE p95 (dB)",
    }[plot_kind]
    ax.set_ylabel(ylabel)
    if plot_kind in {"rmse", "conditional_rmse", "position_p95"}:
        ax.set_yscale("log")
    elif plot_kind == "outlier":
        ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    output_name = {
        "rmse": RMSE_PDF,
        "conditional_rmse": CONDITIONAL_RMSE_PDF,
        "nmse": NMSE_PDF,
        "outlier": OUTLIER_PDF,
        "position_p95": POSITION_P95_PDF,
        "nmse_p95": NMSE_P95_PDF,
    }[plot_kind]
    fig.savefig(out_dir / output_name)
    plt.close(fig)


def _runtime_memory_table(trial_csv: pathlib.Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {
        baseline: {
            "seen": [],
            "runtime_minus10": [],
            "runtime_0": [],
            "rss": [],
            "rss_peak": [],
        }
        for baseline in ESTIMATOR_BASELINES
    }
    with trial_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            baseline = str(row.get("baseline", ""))
            if baseline not in grouped:
                continue
            grouped[baseline]["seen"].append(1.0)
            if str(row.get("failed", "")).lower() == "true":
                continue
            snr_db = _to_float(row.get("snr_db"))
            runtime_s = _to_float(row.get("runtime_s"))
            if np.isfinite(runtime_s):
                if np.isclose(snr_db, -10.0):
                    grouped[baseline]["runtime_minus10"].append(runtime_s)
                if np.isclose(snr_db, 0.0):
                    grouped[baseline]["runtime_0"].append(runtime_s)
            rss_peak = _to_float(row.get("rss_mb_peak"))
            if np.isfinite(rss_peak):
                grouped[baseline]["rss_peak"].append(rss_peak)
            else:
                for field in ("rss_mb_before", "rss_mb_after"):
                    rss_mb = _to_float(row.get(field))
                    if np.isfinite(rss_mb):
                        grouped[baseline]["rss"].append(rss_mb)

    table_rows = []
    for baseline in ESTIMATOR_BASELINES:
        values = grouped[baseline]
        if not values["seen"]:
            continue
        runtime_minus10 = np.asarray(values["runtime_minus10"], dtype=float)
        runtime_0 = np.asarray(values["runtime_0"], dtype=float)
        rss_peak = np.asarray(values["rss_peak"], dtype=float)
        rss = (
            rss_peak
            if rss_peak.size
            else np.asarray(values["rss"], dtype=float)
        )
        table_rows.append(
            {
                "baseline": baseline,
                "algorithm": BASELINE_LABELS[baseline],
                "mean_runtime_s_at_minus10_db": (
                    float(np.mean(runtime_minus10))
                    if runtime_minus10.size
                    else float("nan")
                ),
                "mean_runtime_s_at_0_db": (
                    float(np.mean(runtime_0))
                    if runtime_0.size
                    else float("nan")
                ),
                "peak_memory_mb": (
                    float(np.max(rss)) if rss.size else float("nan")
                ),
                "memory_measurement": (
                    "sampled_process_peak_rss_10ms"
                    if rss_peak.size
                    else "max_pre_post_rss_snapshot"
                ),
                "n_runtime_at_minus10_db": int(runtime_minus10.size),
                "n_runtime_at_0_db": int(runtime_0.size),
            }
        )
    return table_rows


def _markdown_value(value: Any) -> str:
    numeric = _to_float(value)
    if np.isfinite(numeric):
        return f"{numeric:.6g}"
    text = str(value or "")
    return text.replace("|", "\\|").replace("\n", " ")


def _write_summary_markdown(
    out_dir: pathlib.Path,
    command_line: str,
    outlier_threshold_m: float,
    clock_catastrophic_threshold_ns: float,
    runtime_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# External benchmark summary",
        "",
        f"Command: `{command_line}`",
        "",
        (
            "All algorithms at a given trial and SNR use the same noisy "
            "realization, verified by `y_noisy_hash`."
        ),
        "",
        (
            "Fig.7(b) plots `10 log10(mean linear NMSE)`. The trial and "
            "summary CSVs retain linear-domain NMSE values."
        ),
        "",
        (
            "Fig.7(a) uses `sqrt(mean(position_error_m**2))`; PEB curves use "
            "`sqrt(mean(peb_position_m**2))`. The arithmetic mean position "
            "error is retained separately as `position_mean_error_m`."
        ),
        "",
        (
            "Fig.7(c) defines an outlier as position error greater than "
            f"{outlier_threshold_m:g} m."
        ),
        "",
        (
            "External clocks use each baseline's estimated position and raw "
            "panel delays: `median_k(tau_hat_k - "
            "(||p_hat-r_k||+d_RB,k)/c0)`, with one unique delay required for "
            "every physical panel and no clock-bound clipping."
        ),
        "",
        (
            "Clock catastrophic means absolute clock error greater than "
            f"{clock_catastrophic_threshold_ns:g} ns. Invalid clocks are "
            "reported separately and in the combined catastrophic-or-invalid "
            "rate."
        ),
        "",
        "Per-baseline clock delay sources:",
        "",
        "- `als_cpd`: CP-ALS delay factor after joint unique-panel mapping.",
        "",
        (
            "- `nf_ris_groupomp_localgrid_wls`: SAGE-refined panel delay with "
            "that baseline's WLS position."
        ),
        "",
        (
            "- `ris_vbi_sbl`: per-panel VBI/SBL delay with that baseline's "
            "fused position."
        ),
        "",
        "## Table I: runtime and memory comparison",
        "",
        (
            "Memory is the 10-ms sampled peak process RSS while each method "
            "runs. It is available only with `--profile-memory`; short-lived "
            "allocation spikes below the sampling interval may be missed."
        ),
        "",
    ]
    columns = [
        ("algorithm", "Algorithm"),
        ("mean_runtime_s_at_minus10_db", "Mean runtime at -10 dB (s)"),
        ("mean_runtime_s_at_0_db", "Mean runtime at 0 dB (s)"),
        ("peak_memory_mb", "Peak memory (MB)"),
    ]
    if runtime_rows:
        lines.append("| " + " | ".join(label for _, label in columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in runtime_rows:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_value(row.get(field, "")) for field, _ in columns
                )
                + " |"
            )
    else:
        lines.append(
            "Runtime table omitted: use `--runtime-profile` in a dedicated "
            "single-process run."
        )
    lines.extend(
        [
            "",
            (
                "Per-SNR mean, median, standard deviation, p10, p90, p95, "
                "success rate, and outlier rate are stored in "
                f"`{SUMMARY_CSV}`."
            ),
            "",
        ]
    )
    (out_dir / SUMMARY_MD).write_text("\n".join(lines) + "\n")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            cwd=pathlib.Path(__file__).resolve().parents[2],
        )
        return bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return True


def _cache_signature(args: argparse.Namespace, snr_grid: list[float], baselines: list[str]) -> dict[str, Any]:
    return {
        "benchmark_layout_version": 3,
        "n_trials": int(args.n_trials),
        "snr_grid": [float(value) for value in snr_grid],
        "paper_k": int(args.paper_k),
        "baselines": list(baselines),
        "grid_profile": str(args.grid_profile),
        "seed": int(args.seed),
        "outlier_threshold_m": float(args.outlier_threshold_m),
        "clock_catastrophic_threshold_ns": float(
            args.clock_catastrophic_threshold_ns
        ),
        "profile_memory": bool(args.profile_memory),
        "runtime_profile": bool(args.runtime_profile),
        "memory_snapshot_fallback": "proc_self_status_vmrss",
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "strict_ris_geometry": bool(args.strict_ris_geometry),
        "baseline_refinement_tier": str(args.baseline_refinement_tier),
        "baseline_backend": str(args.baseline_backend),
        "allow_shared_gpu_workers": bool(args.allow_shared_gpu_workers),
        "hybrid_single_gpu": bool(args.hybrid_single_gpu),
        "hybrid_retain_gpu_memory_pool": bool(
            args.hybrid_retain_gpu_memory_pool
        ),
        "hybrid_gpu_baselines": list(
            getattr(args, "hybrid_gpu_baselines", [])
        ),
        "hybrid_cpu_baselines": list(
            getattr(args, "hybrid_cpu_baselines", [])
        ),
        "gpu_owner_blas_threads": int(args.gpu_owner_blas_threads),
        "progress_heartbeat_s": float(args.progress_heartbeat_s),
        "gpu_device": args.gpu_device,
        "gpu_batch_size": args.gpu_batch_size,
        "cpu_batch_size": args.cpu_batch_size,
        "cache_baseline_grids": bool(args.cache_baseline_grids),
        "cache_memory_budget_gb": args.cache_memory_budget_gb,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "include_constrained_jones_peb": bool(args.include_constrained_jones_peb),
        "include_anchored_jones_peb": bool(args.include_anchored_jones_peb),
        "jones_anchor_prior_mode": str(args.jones_anchor_prior_mode),
        "jones_anchor_prior_scale": float(args.jones_anchor_prior_scale),
        "global_vp_overrides": global_vp_cli_overrides(args),
    }


def _metadata(args: argparse.Namespace, snr_grid: list[float], baselines: list[str]) -> dict[str, Any]:
    signature = _cache_signature(args, snr_grid, baselines)
    return {
        **signature,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": str(getattr(args, "command_line", " ".join(sys.argv))),
        "jobs": int(args.jobs),
        "process_workers": int(args.process_workers),
        "blas_threads": int(args.blas_threads),
        "gpu_process_workers": int(
            args.resource_plan.get("gpu_process_workers", 0)
        ),
        "gpu_owner_blas_threads": int(args.gpu_owner_blas_threads),
        "estimated_cpu_slots": int(args.resource_plan["estimated_cpu_slots"]),
        "memory_budget_gb": args.memory_budget_gb,
        "memory_per_worker_gb": args.memory_per_worker_gb,
        "trim_memory": bool(args.trim_memory),
        "profile_memory": bool(args.profile_memory),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy_optimizer_available": bool(scipy_is_available()),
        "baseline_backend": str(args.baseline_backend),
        "gpu_device": args.gpu_device,
        "gpu_batch_size": args.gpu_batch_size,
        "cpu_batch_size": args.cpu_batch_size,
        "cache_baseline_grids": bool(args.cache_baseline_grids),
        "cache_memory_budget_gb": args.cache_memory_budget_gb,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "include_constrained_jones_peb": bool(args.include_constrained_jones_peb),
        "include_anchored_jones_peb": bool(args.include_anchored_jones_peb),
        "jones_anchor_prior_mode": str(args.jones_anchor_prior_mode),
        "jones_anchor_prior_scale": float(args.jones_anchor_prior_scale),
    }


def _read_metadata(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _metadata_matches(metadata: dict[str, Any] | None, args: argparse.Namespace, snr_grid: list[float], baselines: list[str]) -> bool:
    if not metadata:
        return False
    expected = _cache_signature(args, snr_grid, baselines)
    return all(metadata.get(key) == value for key, value in expected.items())


def _tasks(args: argparse.Namespace, snr_grid: list[float], baselines: list[str]) -> list[dict[str, Any]]:
    tasks = []
    for trial_id in range(int(args.n_trials)):
        tasks.append(
            {
                "trial_id": int(trial_id),
                "seed": _trial_seed(int(args.seed), int(trial_id)),
                "snr_db": float(snr_grid[0]),
                "snr_grid": [float(value) for value in snr_grid],
                "paper_k": int(args.paper_k),
                "baselines": list(baselines),
                "grid_profile": str(args.grid_profile),
                "blas_threads": (
                    1
                    if str(args.blas_threads).lower() == "auto"
                    else int(args.blas_threads)
                ),
                "respect_existing_blas_env": bool(args.respect_existing_blas_env),
                "trim_memory": bool(args.trim_memory),
                "profile_memory": bool(args.profile_memory),
                "clock_catastrophic_threshold_ns": float(
                    args.clock_catastrophic_threshold_ns
                ),
                "strict_ris_geometry": bool(args.strict_ris_geometry),
                "baseline_refinement_tier": str(args.baseline_refinement_tier),
                "backend_config": {
                    "backend": str(args.baseline_backend),
                    "gpu_device": args.gpu_device,
                    "gpu_batch_size": args.gpu_batch_size,
                    "cpu_batch_size": args.cpu_batch_size,
                    "cache_enabled": bool(args.cache_baseline_grids),
                    "cache_memory_budget_gb": args.cache_memory_budget_gb,
                    "gpu_memory_fraction": args.gpu_memory_fraction,
                    "dtype": "complex128",
                },
                "crb": {
                    "include_constrained_jones_peb": bool(
                        args.include_constrained_jones_peb
                    ),
                    "include_anchored_jones_peb": bool(
                        args.include_anchored_jones_peb
                    ),
                    "jones_anchor_prior_mode": str(
                        args.jones_anchor_prior_mode
                    ),
                    "jones_anchor_prior_scale": float(
                        args.jones_anchor_prior_scale
                    ),
                },
                "global_vp": global_vp_cli_overrides(args),
            }
        )
    return tasks


def _partition_hybrid_baselines(
    args: argparse.Namespace,
    baselines: list[str],
) -> tuple[list[str], list[str]]:
    """Split methods without ever exposing CPU-only workers to CUDA."""
    if args.baseline_backend not in {"cupy", "auto"}:
        raise ValueError(
            "--hybrid-single-gpu requires --baseline-backend cupy or auto"
        )
    if args.global_vp_backend not in {None, "cpu"}:
        raise ValueError(
            "--hybrid-single-gpu currently requires --global-vp-backend cpu; "
            "this keeps every global-VP route in the CPU-only worker pool"
        )
    gpu_baselines = [
        baseline for baseline in baselines if baseline in GPU_EXTERNAL_BASELINES
    ]
    cpu_baselines = [
        baseline for baseline in baselines if baseline not in GPU_EXTERNAL_BASELINES
    ]
    if not gpu_baselines or not cpu_baselines:
        raise ValueError(
            "--hybrid-single-gpu requires at least one GPU-capable external "
            "baseline and at least one CPU-only baseline"
        )
    return gpu_baselines, cpu_baselines


def _hybrid_resource_plan(
    args: argparse.Namespace,
    *,
    n_cpu_tasks: int,
) -> dict[str, int]:
    """Reserve disjoint CPU slots for one GPU owner and the CPU worker pool."""
    if args.process_workers is None:
        raise ValueError(
            "--hybrid-single-gpu requires an explicit --process-workers count "
            "for the CPU-only pool"
        )
    if isinstance(args.blas_threads, str):
        raise ValueError(
            "--hybrid-single-gpu requires an explicit integer --blas-threads"
        )
    gpu_threads = int(args.gpu_owner_blas_threads)
    if gpu_threads <= 0:
        raise ValueError("--gpu-owner-blas-threads must be positive")
    cpu_job_budget = int(args.jobs) - gpu_threads
    if cpu_job_budget <= 0:
        raise ValueError(
            "--jobs must exceed --gpu-owner-blas-threads so the CPU pool has "
            "at least one CPU slot"
        )
    cpu_plan = resolve_hybrid_resources(
        jobs=cpu_job_budget,
        process_workers=int(args.process_workers),
        blas_threads=int(args.blas_threads),
        n_tasks=max(int(n_cpu_tasks), 1),
        memory_budget_gb=args.memory_budget_gb,
        memory_per_worker_gb=args.memory_per_worker_gb,
    )
    estimated_slots = int(cpu_plan["estimated_cpu_slots"]) + gpu_threads
    if estimated_slots > int(args.jobs):
        raise RuntimeError("hybrid resource plan exceeds --jobs")
    return {
        "jobs": int(args.jobs),
        "process_workers": int(cpu_plan["process_workers"]),
        "blas_threads": int(cpu_plan["blas_threads"]),
        "gpu_process_workers": 1,
        "gpu_owner_blas_threads": gpu_threads,
        "n_tasks": int(n_cpu_tasks),
        "estimated_cpu_slots": estimated_slots,
    }


def _task_for_hybrid_lane(
    task: dict[str, Any],
    *,
    baselines: list[str],
    lane: str,
    blas_threads: int,
    retain_gpu_memory_pool: bool,
) -> dict[str, Any]:
    lane_task = dict(task)
    lane_task["baselines"] = list(baselines)
    lane_task["worker_lane"] = str(lane)
    lane_task["blas_threads"] = int(blas_threads)
    lane_task["backend_config"] = dict(task["backend_config"])
    lane_task["trim_gpu_memory_pool"] = not (
        lane == "gpu" and bool(retain_gpu_memory_pool)
    )
    if lane == "cpu":
        # This is the hard isolation boundary: CPU workers never import CuPy or
        # create a CUDA context, even though the overall benchmark uses CuPy.
        lane_task["backend_config"]["backend"] = "cpu"
        lane_task["backend_config"]["gpu_device"] = None
    return lane_task


def _hybrid_row_batches(
    *,
    cpu_tasks: list[dict[str, Any]],
    gpu_tasks: list[dict[str, Any]],
    cpu_workers: int,
    cpu_blas_threads: int,
    gpu_blas_threads: int,
    respect_existing_blas_env: bool,
    trim_memory_enabled: bool,
    profile_memory: bool,
    progress: ProgressLogger,
    heartbeat_s: float,
):
    """Yield row batches from disjoint CPU and single-owner GPU pools."""
    cpu_initargs = (
        int(cpu_blas_threads),
        bool(respect_existing_blas_env),
        bool(trim_memory_enabled),
        bool(profile_memory),
    )
    gpu_initargs = (
        int(gpu_blas_threads),
        bool(respect_existing_blas_env),
        bool(trim_memory_enabled),
        bool(profile_memory),
    )
    with mp.Pool(
        processes=int(cpu_workers),
        initializer=_init_worker,
        initargs=cpu_initargs,
    ) as cpu_pool, mp.Pool(
        processes=1,
        initializer=_init_worker,
        initargs=gpu_initargs,
    ) as gpu_pool:
        active = {
            "cpu": cpu_pool.imap_unordered(
                _run_trial_task, cpu_tasks, chunksize=1
            ),
            "gpu": gpu_pool.imap_unordered(
                _run_trial_task, gpu_tasks, chunksize=1
            ),
        }
        last_heartbeat = time.monotonic()
        while active:
            yielded = False
            for lane in tuple(active):
                try:
                    row_batch = active[lane].next(timeout=0.25)
                except mp.TimeoutError:
                    continue
                except StopIteration:
                    del active[lane]
                    continue
                yielded = True
                yield lane, row_batch
                last_heartbeat = time.monotonic()
            now = time.monotonic()
            if not yielded and now - last_heartbeat >= float(heartbeat_s):
                progress.log(
                    "heartbeat",
                    "running",
                    message=(
                        "hybrid workers active; waiting for the next complete "
                        "trial batch"
                    ),
                    active_lanes=sorted(active),
                )
                last_heartbeat = now


def _record_row_batch(
    row_batch: list[dict[str, Any]],
    *,
    lane: str,
    writer: StreamingCsvWriter,
    progress: ProgressLogger,
    hash_groups: dict[tuple[int, float, int], set[str]],
    coverage_groups: dict[tuple[int, float, int], list[str]],
) -> None:
    writer.writerows(row_batch)
    progress_groups: dict[float, list[dict[str, Any]]] = {}
    for row in row_batch:
        progress_groups.setdefault(_to_float(row.get("snr_db")), []).append(row)
    for snr_db, progress_rows in progress_groups.items():
        representative = progress_rows[0] if progress_rows else {}
        failed_rows = [
            row
            for row in progress_rows
            if str(row.get("failed")).lower() == "true"
        ]
        progress.log(
            "task_failed" if failed_rows else "task_done",
            "failed" if failed_rows else "completed",
            figure="fig7",
            baseline_or_variant=",".join(
                str(row.get("baseline", "")) for row in progress_rows
            ),
            snr_db=snr_db,
            trial_id=representative.get("trial_id", ""),
            seed=representative.get("seed", ""),
            K=representative.get("K", ""),
            message="benchmark trial lane batch completed",
            error="; ".join(
                str(row.get("error", "")) for row in failed_rows
            ),
            scheduler_lane=str(lane),
        )
    for row in row_batch:
        key = (
            int(row["trial_id"]),
            _to_float(row["snr_db"]),
            int(row["K"]),
        )
        coverage_groups.setdefault(key, []).append(str(row.get("baseline", "")))
        observation_hash = str(row.get("y_noisy_hash", ""))
        if observation_hash:
            hash_groups.setdefault(key, set()).add(observation_hash)


def _validate_benchmark_coverage(
    coverage_groups: dict[tuple[int, float, int], list[str]],
    *,
    n_trials: int,
    snr_grid: list[float],
    paper_k: int,
    baselines: list[str],
) -> None:
    expected_baselines = sorted(str(value) for value in baselines)
    errors: list[str] = []
    for trial_id in range(int(n_trials)):
        for snr_db in snr_grid:
            key = (int(trial_id), float(snr_db), int(paper_k))
            observed = sorted(coverage_groups.get(key, []))
            if observed != expected_baselines:
                errors.append(
                    f"key={key} expected={expected_baselines} observed={observed}"
                )
                if len(errors) >= 5:
                    break
        if len(errors) >= 5:
            break
    if errors:
        raise RuntimeError(
            "benchmark method coverage mismatch: " + "; ".join(errors)
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EVS-MIMO-RIS benchmark comparisons.")
    add_mc_args(
        parser,
        n_trials_default=100,
        paper_k_default=3,
        outlier_threshold_default=0.1,
    )
    parser.add_argument("--snr-grid", default=DEFAULT_SNR_GRID)
    add_io_args(parser, default_out_dir="results/benchmark_comparison")
    add_resource_args(parser, jobs_default=10, blas_threads_default="auto")
    add_progress_args(parser)
    parser.add_argument(
        "--baseline-backend",
        choices=("cpu", "cupy", "auto"),
        default="cpu",
    )
    parser.add_argument(
        "--allow-shared-gpu-workers",
        action="store_true",
        help=(
            "allow multiple worker processes to share --gpu-device for "
            "accuracy/throughput runs; disabled by default and incompatible "
            "with --runtime-profile"
        ),
    )
    parser.add_argument(
        "--hybrid-single-gpu",
        action="store_true",
        help=(
            "run GPU-capable external baselines in one dedicated CUDA owner "
            "process while CPU-only baselines run in --process-workers CPU "
            "processes; accuracy/throughput only"
        ),
    )
    parser.add_argument(
        "--gpu-owner-blas-threads",
        type=int,
        default=1,
        help=(
            "native CPU threads reserved for the dedicated GPU owner in "
            "--hybrid-single-gpu mode"
        ),
    )
    parser.add_argument(
        "--hybrid-retain-gpu-memory-pool",
        action="store_true",
        help=(
            "retain unused CuPy memory-pool blocks between methods in the "
            "single GPU-owner process; requires a GPU-memory pilot"
        ),
    )
    parser.add_argument(
        "--progress-heartbeat-s",
        type=float,
        default=60.0,
        help="parent-process heartbeat interval while worker batches are pending",
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-batch-size", type=int, default=None)
    parser.add_argument("--cpu-batch-size", type=int, default=None)
    parser.add_argument(
        "--cache-baseline-grids",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cache-memory-budget-gb", type=float, default=None)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--reuse-incompatible-cache", action="store_true")
    parser.add_argument(
        "--runtime-profile",
        action="store_true",
        help="publish runtime/memory tables from a dedicated single-process run",
    )
    parser.add_argument("--baselines", default=DEFAULT_BASELINES)
    parser.add_argument("--strict-ris-geometry", action="store_true")
    parser.add_argument(
        "--clock-catastrophic-threshold-ns",
        type=float,
        default=1.0,
        help=(
            "frozen absolute clock-error threshold for the clock "
            "catastrophic rate (default: 1 ns)"
        ),
    )
    add_global_vp_args(parser)
    parser.add_argument("--grid-profile", choices=("coarse", "medium", "fine"), default="medium")
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
    parser.add_argument(
        "--baseline-refinement-tier",
        choices=REFINEMENT_TIERS,
        default="refinement_matched",
        help=(
            "declared comparison policy for the external baselines. "
            "'refinement_matched' (default) grants every route the same final "
            "continuous exact-model polish of (p_u, Delta_t) from its own "
            "seed; 'as_published' stops each baseline where its own reference "
            "stops. Only ris_vbi_sbl is tier-sensitive."
        ),
    )
    args = parser.parse_args(argv)
    if args.global_vp_backend in {"cupy", "auto"} and args.process_workers is None:
        args.process_workers = 1
    normalize_blas_threads(args)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    command_argv = [
        sys.executable,
        sys.argv[0],
        *(sys.argv[1:] if argv is None else argv),
    ]
    args.command_line = shlex.join(command_argv)
    print(f"Command: {args.command_line}")
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.process_workers is not None and args.process_workers <= 0:
        raise ValueError("--process-workers must be positive")
    if args.gpu_owner_blas_threads <= 0:
        raise ValueError("--gpu-owner-blas-threads must be positive")
    if args.progress_heartbeat_s <= 0.0:
        raise ValueError("--progress-heartbeat-s must be positive")
    if args.paper_k <= 0:
        raise ValueError("--paper-k must be positive")
    if args.clock_catastrophic_threshold_ns <= 0.0:
        raise ValueError("--clock-catastrophic-threshold-ns must be positive")
    if args.blas_threads != "auto" and int(args.blas_threads) <= 0:
        raise ValueError("--blas-threads must be positive or 'auto'")
    if args.gpu_batch_size is not None and args.gpu_batch_size <= 0:
        raise ValueError("--gpu-batch-size must be positive")
    if args.cpu_batch_size is not None and args.cpu_batch_size <= 0:
        raise ValueError("--cpu-batch-size must be positive")
    if args.cache_memory_budget_gb is not None and args.cache_memory_budget_gb <= 0:
        raise ValueError("--cache-memory-budget-gb must be positive")
    if args.gpu_memory_fraction is not None and not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0, 1]")
    if args.hybrid_single_gpu and args.allow_shared_gpu_workers:
        raise ValueError(
            "--hybrid-single-gpu and --allow-shared-gpu-workers are mutually "
            "exclusive"
        )
    if args.hybrid_retain_gpu_memory_pool and not args.hybrid_single_gpu:
        raise ValueError(
            "--hybrid-retain-gpu-memory-pool requires --hybrid-single-gpu"
        )
    if args.hybrid_single_gpu and args.runtime_profile:
        raise ValueError(
            "--hybrid-single-gpu is an accuracy/throughput mode and cannot be "
            "used with --runtime-profile"
        )
    if args.hybrid_single_gpu and args.process_workers is None:
        raise ValueError(
            "--hybrid-single-gpu requires an explicit --process-workers count "
            "for the CPU-only pool"
        )
    if args.hybrid_single_gpu and isinstance(args.blas_threads, str):
        raise ValueError(
            "--hybrid-single-gpu requires an explicit integer --blas-threads"
        )
    if (
        args.baseline_backend in {"cupy", "auto"}
        and args.process_workers is None
        and not args.hybrid_single_gpu
    ):
        args.process_workers = 1
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = (
        pathlib.Path(args.progress_log)
        if args.progress_log is not None
        else out_dir / "progress.jsonl"
    )
    trial_csv = out_dir / TRIAL_CSV
    summary_csv = out_dir / SUMMARY_CSV
    metadata_path = out_dir / "benchmark_metadata.json"
    baselines = parse_baselines(args.baselines)
    if not bool(args.include_constrained_jones_peb):
        baselines = [
            baseline
            for baseline in baselines
            if baseline != "constrained_jones_peb"
        ]
    elif (
        "peb" in baselines
        and "constrained_jones_peb" not in baselines
    ):
        baselines.append("constrained_jones_peb")
    snr_grid = parse_snr_grid(args.snr_grid)
    if not snr_grid:
        raise ValueError("--snr-grid must contain at least one value")
    if args.hybrid_single_gpu:
        gpu_baselines, cpu_baselines = _partition_hybrid_baselines(
            args, baselines
        )
        progress_lane_count = 2
    else:
        gpu_baselines, cpu_baselines = [], []
        progress_lane_count = 1
    args.hybrid_gpu_baselines = list(gpu_baselines)
    args.hybrid_cpu_baselines = list(cpu_baselines)
    if "proposed" in baselines:
        preview_config = make_config(
            seed=int(args.seed),
            snr_db=float(snr_grid[0]),
            paper_k=int(args.paper_k),
            grid_profile=str(args.grid_profile),
            strict_ris_geometry=bool(args.strict_ris_geometry),
        )
        print(f"baseline=proposed {_proposed_policy_log_fragment(preview_config)}")
    tasks = _tasks(args, snr_grid, baselines)
    progress = ProgressLogger(
        progress_path,
        len(tasks) * len(snr_grid) * progress_lane_count,
        "run_benchmark_comparison",
    )
    if not args.quiet_progress:
        print(f"Progress log: {progress_path}")
        print("Monitor with:")
        print(f"    tail -f {progress_path}")
        print(
            "    python -m src.experiments.monitor_progress "
            f"--progress-log {progress_path}"
        )
    progress.log("start", "running", message="benchmark experiment started")
    if args.hybrid_single_gpu:
        args.resource_plan = _hybrid_resource_plan(
            args,
            n_cpu_tasks=max(len(tasks), 1),
        )
    else:
        args.resource_plan = resolve_hybrid_resources(
            jobs=args.jobs,
            process_workers=args.process_workers,
            blas_threads=args.blas_threads,
            n_tasks=max(len(tasks), 1),
            memory_budget_gb=args.memory_budget_gb,
            memory_per_worker_gb=args.memory_per_worker_gb,
        )
        args.resource_plan["gpu_process_workers"] = (
            1
            if (
                args.baseline_backend in {"cupy", "auto"}
                or args.global_vp_backend in {"cupy", "auto"}
            )
            else 0
        )
        args.resource_plan["gpu_owner_blas_threads"] = int(
            args.resource_plan["blas_threads"]
        )
    args.process_workers = int(args.resource_plan["process_workers"])
    args.blas_threads = int(args.resource_plan["blas_threads"])
    args.gpu_owner_blas_threads = int(
        args.resource_plan["gpu_owner_blas_threads"]
    )
    gpu_requested = (
        args.baseline_backend in {"cupy", "auto"}
        or args.global_vp_backend in {"cupy", "auto"}
    )
    if (
        not args.hybrid_single_gpu
        and gpu_requested
        and args.process_workers != 1
        and not bool(args.allow_shared_gpu_workers)
    ):
        raise ValueError(
            "sharing one --gpu-device across multiple worker processes is "
            "disabled by default; use --process-workers 1, split jobs across "
            "GPU devices, or explicitly pass --allow-shared-gpu-workers for "
            "an accuracy/throughput run"
        )
    if args.runtime_profile and args.process_workers != 1:
        raise ValueError("--runtime-profile requires --process-workers 1")
    if (
        not args.hybrid_single_gpu
        and gpu_requested
        and args.process_workers > 1
    ):
        print(
            "WARNING: multiple worker processes are sharing one GPU; use this "
            "run for accuracy/throughput only, not paper runtime or memory claims"
        )
    if args.hybrid_single_gpu:
        cpu_tasks = [
            _task_for_hybrid_lane(
                task,
                baselines=cpu_baselines,
                lane="cpu",
                blas_threads=args.blas_threads,
                retain_gpu_memory_pool=False,
            )
            for task in tasks
        ]
        gpu_tasks = [
            _task_for_hybrid_lane(
                task,
                baselines=gpu_baselines,
                lane="gpu",
                blas_threads=args.gpu_owner_blas_threads,
                retain_gpu_memory_pool=bool(
                    args.hybrid_retain_gpu_memory_pool
                ),
            )
            for task in tasks
        ]
    else:
        cpu_tasks, gpu_tasks = [], []
        for task in tasks:
            task["blas_threads"] = args.blas_threads
    apply_thread_limits(
        args.blas_threads,
        respect_existing=bool(args.respect_existing_blas_env),
    )
    print(
        "Resource plan: "
        f"jobs={args.jobs} "
        f"process_workers={args.process_workers} "
        f"blas_threads={args.blas_threads} "
        f"hybrid_single_gpu={bool(args.hybrid_single_gpu)} "
        f"gpu_process_workers={args.resource_plan['gpu_process_workers']} "
        f"gpu_owner_blas_threads={args.gpu_owner_blas_threads} "
        f"estimated_cpu_slots={args.resource_plan['estimated_cpu_slots']} "
        f"allow_shared_gpu_workers={bool(args.allow_shared_gpu_workers)} "
        f"memory_budget_gb={args.memory_budget_gb} "
        f"memory_per_worker_gb={args.memory_per_worker_gb}"
    )
    metadata = _read_metadata(metadata_path)
    cache_compatible = _metadata_matches(metadata, args, snr_grid, baselines)
    can_reuse = (
        trial_csv.exists()
        and summary_csv.exists()
        and not args.force_rerun
        and (cache_compatible or bool(args.reuse_incompatible_cache))
    )
    if can_reuse:
        summary = _read_csv(summary_csv)
    else:
        hash_groups: dict[tuple[int, float, int], set[str]] = {}
        coverage_groups: dict[tuple[int, float, int], list[str]] = {}
        with StreamingCsvWriter(trial_csv, FIELDNAMES) as writer:
            if args.hybrid_single_gpu:
                for lane, row_batch in _hybrid_row_batches(
                    cpu_tasks=cpu_tasks,
                    gpu_tasks=gpu_tasks,
                    cpu_workers=args.process_workers,
                    cpu_blas_threads=args.blas_threads,
                    gpu_blas_threads=args.gpu_owner_blas_threads,
                    respect_existing_blas_env=bool(
                        args.respect_existing_blas_env
                    ),
                    trim_memory_enabled=bool(args.trim_memory),
                    profile_memory=bool(args.profile_memory),
                    progress=progress,
                    heartbeat_s=float(args.progress_heartbeat_s),
                ):
                    _record_row_batch(
                        row_batch,
                        lane=lane,
                        writer=writer,
                        progress=progress,
                        hash_groups=hash_groups,
                        coverage_groups=coverage_groups,
                    )
            else:
                initargs = (
                    args.blas_threads,
                    bool(args.respect_existing_blas_env),
                    bool(args.trim_memory),
                    bool(args.profile_memory),
                )
                if args.process_workers == 1:
                    _init_worker(*initargs)
                    row_batches = map(_run_trial_task, tasks)
                    pool_context = contextlib.nullcontext()
                else:
                    pool_context = mp.Pool(
                        processes=args.process_workers,
                        initializer=_init_worker,
                        initargs=initargs,
                    )
                    pool = pool_context.__enter__()
                    row_batches = pool.imap_unordered(
                        _run_trial_task,
                        tasks,
                        chunksize=1,
                    )
                try:
                    for row_batch in row_batches:
                        _record_row_batch(
                            row_batch,
                            lane="standard",
                            writer=writer,
                            progress=progress,
                            hash_groups=hash_groups,
                            coverage_groups=coverage_groups,
                        )
                finally:
                    if args.process_workers != 1:
                        pool_context.__exit__(None, None, None)
        _validate_benchmark_coverage(
            coverage_groups,
            n_trials=int(args.n_trials),
            snr_grid=snr_grid,
            paper_k=int(args.paper_k),
            baselines=baselines,
        )
        mismatches = {
            key: values for key, values in hash_groups.items() if len(values) > 1
        }
        if mismatches:
            raise RuntimeError(f"benchmark same-data hash mismatch: {mismatches}")
        summary = summarize_csv(
            trial_csv,
            outlier_threshold_m=float(args.outlier_threshold_m),
        )
        _write_csv(
            summary_csv,
            summary,
            list(summary[0].keys()) if summary else [],
        )
    summary = _read_csv(summary_csv)
    with metadata_path.open("w") as handle:
        json.dump(_metadata(args, snr_grid, baselines), handle, indent=2)
    runtime_rows = _runtime_memory_table(trial_csv) if args.runtime_profile else []
    runtime_fields = [
        "baseline",
        "algorithm",
        "mean_runtime_s_at_minus10_db",
        "mean_runtime_s_at_0_db",
        "peak_memory_mb",
        "memory_measurement",
        "n_runtime_at_minus10_db",
        "n_runtime_at_0_db",
    ]
    if args.runtime_profile:
        _write_csv(out_dir / RUNTIME_MEMORY_CSV, runtime_rows, runtime_fields)
    _write_summary_markdown(
        out_dir,
        args.command_line,
        float(args.outlier_threshold_m),
        float(args.clock_catastrophic_threshold_ns),
        runtime_rows,
    )
    if not args.no_plots:
        _plot(summary, out_dir, "rmse")
        _plot(summary, out_dir, "conditional_rmse")
        _plot(summary, out_dir, "nmse")
        _plot(summary, out_dir, "outlier")
        _plot(summary, out_dir, "position_p95")
        _plot(summary, out_dir, "nmse_p95")
    progress.log("finished", "completed", message="benchmark experiment finished")
    progress.close()
    print(f"Wrote benchmark comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
