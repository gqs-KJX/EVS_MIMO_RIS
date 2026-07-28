"""Shared, paired experiment helpers for the frozen MKSC-GI--CCOP route.

This module intentionally calls the validated estimator implementations.  It
does not duplicate or alter the channel model, Stage-I algebra, or CCOP
objective.  All variants evaluated through :func:`run_paired_variants` see the
same immutable ``data`` dictionary and therefore the same noisy observation.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import itertools
import json
import pathlib
import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ..ccop_jvp import refine_ccop_jvp, refine_four_dimensional_jvp_experimental
from ..ccop_stage1_initializer import (
    apply_ccop_stage1_preset,
    initialize_ccop_stage1,
    refresh_ccop_stage1_jones_anchor,
    refine_ccop_stage1_joint_geometry,
)
from ..global_vp import _initial_xi_from_stage1, build_jones_vp_dictionary
from ..main_single_proposed import _make_data, run_stage1_only
from ..metrics import relative_nmse
from ..projections_delay import tau_from_pole
from ..robust_jnpp import _confidence_weights, _objective_and_grad_factory
from ..validation_artifacts import (
    array_sha256,
    canonical_hash,
    deterministic_stage1_output,
)
from .ccop_experiment_config import build_ccop_experiment_config
from .resource_control import PeakRssSampler, memory_snapshot_mb


PAPER_VARIANTS = (
    "old_4d",
    "scaled_4d",
    "old_stage1_ccop",
    "raw_delay_gi_ccop",
    "mksc_delay_ccop",
    "mksc_gi_1_no_refresh_ccop",
    "mksc_gi_4_no_refresh_ccop",
    "proposed",
    "mksc_gi_7_refresh_ccop",
    "oracle_position_start_ccop",
    "mksc_gi_refresh_4d_seconds",
    "mksc_gi_refresh_4d_nanoseconds",
    "mksc_gi_refresh_4d_distance_m",
    "mksc_gi_refresh_ccop_seconds",
    "mksc_gi_refresh_ccop_nanoseconds",
    "mksc_gi_refresh_ccop_distance_m",
)

VARIANT_LABELS = {
    "old_4d": "Frozen Stage-I + 4-D Jones-VP",
    "scaled_4d": "Frozen Stage-I + scaled 4-D Jones-VP",
    "old_stage1_ccop": "Frozen Stage-I + CCOP-JVP",
    "raw_delay_gi_ccop": "Raw-delay Stage-I + common GI + refresh + CCOP-JVP",
    "mksc_delay_ccop": "MKSC delay only + CCOP-JVP",
    "mksc_gi_1_no_refresh_ccop": "MKSC-GI (1 start, no refresh) + CCOP-JVP",
    "mksc_gi_4_no_refresh_ccop": "MKSC-GI (4 starts, no refresh) + CCOP-JVP",
    "proposed": "MKSC-GI-balanced + CCOP-JVP",
    "mksc_gi_7_refresh_ccop": "MKSC-GI (7 starts + refresh) + CCOP-JVP",
    "oracle_position_start_ccop": "Oracle-position-start CCOP (reference)",
    "mksc_gi_refresh_4d_seconds": (
        "MKSC-GI (4 starts + refresh) + 4-D VP [seconds]"
    ),
    "mksc_gi_refresh_4d_nanoseconds": (
        "MKSC-GI (4 starts + refresh) + 4-D VP [nanoseconds]"
    ),
    "mksc_gi_refresh_4d_distance_m": (
        "MKSC-GI (4 starts + refresh) + 4-D VP [clock range, m]"
    ),
    "mksc_gi_refresh_ccop_seconds": (
        "MKSC-GI (4 starts + refresh) + CCOP [seconds label]"
    ),
    "mksc_gi_refresh_ccop_nanoseconds": (
        "MKSC-GI (4 starts + refresh) + CCOP [nanoseconds label]"
    ),
    "mksc_gi_refresh_ccop_distance_m": (
        "MKSC-GI (4 starts + refresh) + CCOP [clock range label, m]"
    ),
    "free_jones_peb": "Data-only free-Jones PEB",
    "constrained_jones_peb": "Constrained-Jones PEB (reference)",
}

FOUR_D_CLOCK_COORDINATES = {
    "old_4d": "seconds",
    "scaled_4d": "distance_m",
    "mksc_gi_refresh_4d_seconds": "seconds",
    "mksc_gi_refresh_4d_nanoseconds": "nanoseconds",
    "mksc_gi_refresh_4d_distance_m": "distance_m",
}

CCOP_CLOCK_UNIT_VARIANTS = {
    "mksc_gi_refresh_ccop_seconds": "seconds",
    "mksc_gi_refresh_ccop_nanoseconds": "nanoseconds",
    "mksc_gi_refresh_ccop_distance_m": "distance_m",
}

GOOD_INITIALIZER_VARIANTS = frozenset(
    {
        "proposed",
        *(
            variant
            for variant in FOUR_D_CLOCK_COORDINATES
            if variant.startswith("mksc_gi_refresh_")
        ),
        *CCOP_CLOCK_UNIT_VARIANTS,
    }
)

TRIAL_FIELDS = (
    "suite",
    "x_name",
    "x_value",
    "trial_id",
    "seed",
    "snr_db",
    "variant",
    "variant_label",
    "initializer_family",
    "inner_solver",
    "clock_parameterization",
    "clock_coordinate_scale_per_second",
    "clock_parameterization_affects_solver",
    "nonlinear_dim",
    "failed",
    "error",
    "resolved_config_hash",
    "y_noisy_hash",
    "base_y_noisy_hash",
    "nested_receiver_noise_convention",
    "reference_sigma2",
    "stage1_output_hash",
    "candidate_hash",
    "reference_type",
    "peb_position_m",
    "position_error_m",
    "clock_error_ns",
    "channel_nmse",
    "noisy_fit_nmse",
    "outlier",
    "raw_objective_final",
    "total_objective_final",
    "stage1_position_error_m",
    "stage1_delay_rmse_ns",
    "stage1_delay_max_abs_error_ns",
    "stage1_min_delay_separation_ns",
    "stage1_panel_pairing_correct",
    "stage1_runtime_s",
    "mksc_projection_runtime_s",
    "common_geometry_runtime_s",
    "anchor_refresh_runtime_s",
    "stage3_runtime_s",
    "deployment_runtime_s",
    "rss_mb_before",
    "rss_mb_peak",
    "rss_mb_after",
    "memory_measurement",
    "optimizer_evaluations",
    "optimizer_iterations",
    "clock_interval_evaluations",
    "clock_profile_evaluations",
    "clock_profile_certified_count",
    "clock_profiles_all_certified",
    "clock_certified",
    "clock_certificate_gap",
    "clock_certificate_tolerance",
    "clock_certificate_gap_ratio",
    "clock_certificate_gap_max_over_positions",
    "clock_certificate_gap_ratio_max_over_positions",
    "clock_fft_peak_gap",
    "clock_branch_gap_tolerance",
    "clock_bnb_splits",
    "clock_branch_ambiguous",
    "outer_safeguard_used",
    "selected_candidate",
    "stage1_evs_union_rank",
    "stage1_evs_retained_energy_fraction",
    "stage1_delay_sigma_k",
    "stage1_delay_sigma_kplus1_over_k",
    "stage1_delay_subspace_condition",
    "stage1_assignment_margin",
    "stage1_max_rank1_ratio",
    "common_geometry_hessian_min_eig",
    "common_geometry_hessian_condition",
    "stage1_basin_acquired",
    "receiver_mode",
    "N",
    "T",
    "K",
    "M_A",
    "M_R",
    "p_true_x",
    "p_true_y",
    "p_true_z",
)

SUMMARY_FIELDS = (
    "suite",
    "x_name",
    "x_value",
    "variant",
    "variant_label",
    "initializer_family",
    "inner_solver",
    "clock_parameterization",
    "clock_coordinate_scale_per_second",
    "clock_parameterization_affects_solver",
    "n",
    "n_failed",
    "n_success",
    "success_rate",
    "failure_rate",
    "position_rmse_m",
    "position_mean_error_m",
    "position_conditional_rmse_m",
    "position_median_m",
    "position_p90_m",
    "position_p95_m",
    "peb_position_m_mean",
    "peb_position_m_rms",
    "clock_rmse_ns",
    "clock_median_ns",
    "clock_p90_ns",
    "clock_p95_ns",
    "channel_nmse_mean",
    "channel_nmse_median",
    "channel_nmse_p90",
    "channel_nmse_p95",
    "conditional_outlier_rate",
    "outlier_rate",
    "catastrophic_rate",
    "outlier_ci_low",
    "outlier_ci_high",
    "outlier_ci_method",
    "runtime_mean_s",
    "runtime_median_s",
    "runtime_p95_s",
    "peak_rss_median_mb",
    "peak_rss_max_mb",
    "stage1_runtime_median_s",
    "mksc_projection_runtime_median_s",
    "common_geometry_runtime_median_s",
    "anchor_refresh_runtime_median_s",
    "stage3_runtime_median_s",
    "nonlinear_dim_median",
    "optimizer_evaluations_median",
    "optimizer_iterations_median",
    "clock_profile_evaluations_median",
    "clock_all_positions_certified_rate",
    "clock_certificate_rate",
    "clock_certificate_gap_p95",
    "clock_certificate_tolerance_p95",
    "clock_certificate_gap_ratio_p95",
    "clock_certificate_gap_max_over_positions_max",
    "clock_certificate_gap_ratio_max_over_positions_p95",
    "clock_certificate_gap_ratio_max_over_positions_max",
    "clock_fft_peak_gap_median",
    "stage1_basin_acquisition_rate",
    "stage1_delay_rmse_ns_median",
    "stage1_delay_rmse_ns_p95",
    "common_geometry_hessian_min_eig_median",
)

PAIRED_FIELDS = (
    "suite",
    "x_name",
    "x_value",
    "reference_variant",
    "candidate_variant",
    "n_pairs",
    "n_metric_pairs",
    "position_win_rate",
    "clock_win_rate",
    "channel_win_rate",
    "raw_objective_non_degradation_rate",
    "rescued_outliers",
    "introduced_outliers",
    "mcnemar_exact_p",
    "paired_position_rmse_difference_m",
    "paired_position_rmse_difference_ci_low_m",
    "paired_position_rmse_difference_ci_high_m",
    "paired_runtime_ratio_median",
    "paired_runtime_ratio_ci_low",
    "paired_runtime_ratio_ci_high",
    "bootstrap_replicates",
)


def _paper_spec(snr_db: float, *, diagnostic_mode: str = "performance") -> dict:
    return {
        "snr_db": float(snr_db),
        "diagnostic_mode": str(diagnostic_mode),
        "outlier_threshold_m": 0.1,
        "jones_mode": "jones_regularized",
        "old_max_iter": 80,
        "ccop_outer_max_iter": 20,
        "clock_fft_size": 4096,
        "clock_abs_tol": 1.0e-12,
        "clock_rel_tol": 1.0e-10,
        "clock_max_intervals": 20000,
        "use_old_incumbent": False,
        "old_vp_backend": "cpu",
        "gpu_device": 0,
    }


def make_paper_config(
    seed: int,
    snr_db: float,
    *,
    diagnostic_mode: str = "performance",
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Build the frozen paper configuration without changing any defaults."""
    config = build_ccop_experiment_config(
        _paper_spec(snr_db, diagnostic_mode=diagnostic_mode), int(seed)
    )
    config = apply_ccop_stage1_preset(config, "balanced")
    if overrides:
        for key, value in overrides.items():
            # ``ris_search`` is merged rather than replaced: a caller that wants
            # to change one acquisition switch must not silently drop the grid
            # sizes, bounds and refine-start budget that sit beside it.
            if key in {
                "global_vp",
                "ccop_jvp",
                "ccop_stage1_joint_geometry",
                "ris_search",
            }:
                merged = copy.deepcopy(config.get(key, {}))
                merged.update(copy.deepcopy(value))
                config[key] = merged
            else:
                config[key] = copy.deepcopy(value)
    return config


def apply_final_paper_options(config: dict) -> dict:
    """Overlay only the frozen route options on an existing experiment config.

    This is used by external-comparison and robustness runners after they have
    changed SNR, training overhead, receiver mode, or geometry.  It preserves
    those scenario settings and changes no physical quantity.
    """
    resolved = apply_ccop_stage1_preset(copy.deepcopy(config), "balanced")
    resolved["print_progress"] = False
    resolved["verbose_stage2"] = False
    resolved["global_vp"] = dict(resolved.get("global_vp", {}))
    resolved["global_vp"].update(
        {
            "solver": "lbfgsb_reduced",
            "mode": "jones_regularized",
            "backend": "cpu",
            "vp_dictionary_mode": "matrix_free",
            "use_weight": False,
            "use_delay_prior": False,
            "jones_diagonal_loading": 0.0,
            "enable_z_rescue_multistart": False,
            "use_multistart": False,
            "max_iter": 80,
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
        }
    )
    resolved["ccop_jvp"] = {
        **dict(resolved.get("ccop_jvp", {})),
        "clock_fft_size": 4096,
        "clock_abs_tol_objective": 1.0e-12,
        "clock_rel_tol": 1.0e-10,
        "clock_max_intervals": 20000,
        "outer_max_iter": 20,
        "outer_ftol": 1.0e-12,
        "outer_gtol": 1.0e-8,
    }
    return resolved


def make_shared_data(config: dict) -> dict:
    """Generate one immutable realization for all paired routes."""
    data = _make_data(config)
    config["noise_variance"] = float(data["noise_variance"])
    return data


def _quiet_call(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class Stage1Cache:
    """Lazily compute shared Stage-I intermediates for one realization."""

    def __init__(self, data: dict, config: dict):
        self.data = data
        self.config = config
        self._cache: dict[str, tuple[dict, dict[str, float]]] = {}

    def frozen(self) -> tuple[dict, dict[str, float]]:
        if "frozen" not in self._cache:
            start = time.perf_counter()
            estimate = _quiet_call(run_stage1_only, self.data, self.config)["estimate"]
            elapsed = time.perf_counter() - start
            self._cache["frozen"] = (
                estimate,
                {
                    "stage1_runtime_s": float(elapsed),
                    "mksc_projection_runtime_s": 0.0,
                    "common_geometry_runtime_s": 0.0,
                    "anchor_refresh_runtime_s": 0.0,
                },
            )
        return self._cache["frozen"]

    def mksc_delay(self) -> tuple[dict, dict[str, float]]:
        if "mksc_delay" not in self._cache:
            start = time.perf_counter()
            estimate = initialize_ccop_stage1(
                self.data["Z_noisy"], self.data["scene"], self.config
            )
            elapsed = time.perf_counter() - start
            self._cache["mksc_delay"] = (
                estimate,
                {
                    "stage1_runtime_s": float(elapsed),
                    "mksc_projection_runtime_s": float(
                        estimate.get("stage1_time_evs_sufficient_projection", 0.0)
                    ),
                    "common_geometry_runtime_s": 0.0,
                    "anchor_refresh_runtime_s": 0.0,
                },
            )
        return self._cache["mksc_delay"]

    def joint(self, starts: int, refresh: bool) -> tuple[dict, dict[str, float]]:
        key = f"joint_{int(starts)}_{int(bool(refresh))}"
        if key not in self._cache:
            base, base_times = self.mksc_delay()
            local_config = copy.deepcopy(self.config)
            local_config["ccop_stage1_joint_geometry"] = {
                **dict(local_config.get("ccop_stage1_joint_geometry", {})),
                "num_starts": int(starts),
                "max_iter": 30,
                "use_leave_one_out": False,
                "enable_z_starts": True,
                "z_start_grid_size": 7,
                "use_coarse_grid": False,
            }
            geometry_start = time.perf_counter()
            estimate = refine_ccop_stage1_joint_geometry(
                copy.deepcopy(base), self.data["scene"], local_config
            )
            geometry_time = time.perf_counter() - geometry_start
            estimate.update(
                _common_geometry_hessian_diagnostics(
                    base,
                    np.asarray(estimate["p_u"], dtype=float),
                    self.data["scene"],
                    local_config,
                )
            )
            refresh_time = 0.0
            if refresh:
                refresh_start = time.perf_counter()
                estimate = refresh_ccop_stage1_jones_anchor(
                    self.data["Y_noisy"], estimate, self.data["scene"], local_config
                )
                refresh_time = time.perf_counter() - refresh_start
            self._cache[key] = (
                estimate,
                {
                    **base_times,
                    "stage1_runtime_s": float(
                        base_times["stage1_runtime_s"] + geometry_time + refresh_time
                    ),
                    "common_geometry_runtime_s": float(geometry_time),
                    "anchor_refresh_runtime_s": float(refresh_time),
                },
            )
        return self._cache[key]


def _stage1_for_variant(
    variant: str, cache: Stage1Cache
) -> tuple[dict, dict[str, float]]:
    if variant in {"old_4d", "scaled_4d", "old_stage1_ccop"}:
        return cache.frozen()
    if variant == "raw_delay_gi_ccop":
        key = "raw_delay_joint_4_1"
        if key not in cache._cache:
            base, base_times = cache.frozen()
            local_config = copy.deepcopy(cache.config)
            local_config["ccop_stage1_joint_geometry"] = {
                **dict(local_config.get("ccop_stage1_joint_geometry", {})),
                "num_starts": 4,
                "max_iter": 30,
                "use_leave_one_out": False,
                "enable_z_starts": True,
                "z_start_grid_size": 7,
                "use_coarse_grid": False,
            }
            start = time.perf_counter()
            estimate = refine_ccop_stage1_joint_geometry(
                copy.deepcopy(base), cache.data["scene"], local_config
            )
            geometry_time = time.perf_counter() - start
            estimate.update(
                _common_geometry_hessian_diagnostics(
                    base,
                    np.asarray(estimate["p_u"], dtype=float),
                    cache.data["scene"],
                    local_config,
                )
            )
            start = time.perf_counter()
            estimate = refresh_ccop_stage1_jones_anchor(
                cache.data["Y_noisy"], estimate, cache.data["scene"], local_config
            )
            refresh_time = time.perf_counter() - start
            cache._cache[key] = (
                estimate,
                {
                    **base_times,
                    "stage1_runtime_s": float(
                        base_times["stage1_runtime_s"]
                        + geometry_time
                        + refresh_time
                    ),
                    "common_geometry_runtime_s": float(geometry_time),
                    "anchor_refresh_runtime_s": float(refresh_time),
                },
            )
        return cache._cache[key]
    if variant == "mksc_delay_ccop":
        return cache.mksc_delay()
    if variant == "mksc_gi_1_no_refresh_ccop":
        return cache.joint(1, False)
    if variant == "mksc_gi_4_no_refresh_ccop":
        return cache.joint(4, False)
    if variant in GOOD_INITIALIZER_VARIANTS:
        return cache.joint(4, True)
    if variant == "mksc_gi_7_refresh_ccop":
        return cache.joint(7, True)
    if variant == "oracle_position_start_ccop":
        estimate, times = cache.joint(4, False)
        oracle = copy.deepcopy(estimate)
        oracle["p_u"] = np.asarray(cache.data["scene"]["p_u_true"], dtype=float).copy()
        start = time.perf_counter()
        oracle = refresh_ccop_stage1_jones_anchor(
            cache.data["Y_noisy"], oracle, cache.data["scene"], cache.config
        )
        oracle_times = dict(times)
        oracle_times["anchor_refresh_runtime_s"] = float(time.perf_counter() - start)
        oracle_times["stage1_runtime_s"] += oracle_times["anchor_refresh_runtime_s"]
        return oracle, oracle_times
    raise ValueError(f"unknown final-paper variant {variant!r}")


def _common_geometry_hessian_diagnostics(
    unprojected_stage1: dict,
    position: np.ndarray,
    scene: dict,
    config: dict,
) -> dict[str, float]:
    """Numerically differentiate the analytic GI gradient for diagnostics."""
    try:
        c_tilde = np.asarray(unprojected_stage1["C"], dtype=complex)
        weights, _, _, _ = _confidence_weights(
            unprojected_stage1, config, int(scene["K"])
        )
        value_and_grad, _ = _objective_and_grad_factory(
            c_tilde,
            scene,
            weights,
            tuple(range(int(scene["K"]))),
            float(config.get("eps", 1.0e-10)),
        )
        p0 = np.asarray(position, dtype=float).reshape(3)
        bounds = np.asarray(config["ue_bounds"], dtype=float)
        hessian = np.zeros((3, 3), dtype=float)
        for dim in range(3):
            step = 1.0e-4 * max(1.0, abs(float(p0[dim])))
            plus = p0.copy()
            minus = p0.copy()
            plus[dim] = min(bounds[dim, 1], plus[dim] + step)
            minus[dim] = max(bounds[dim, 0], minus[dim] - step)
            denominator = float(plus[dim] - minus[dim])
            if denominator <= 0.0:
                return {
                    "common_geometry_hessian_min_eig": float("nan"),
                    "common_geometry_hessian_condition": float("inf"),
                }
            _, gradient_plus = value_and_grad(plus, True)
            _, gradient_minus = value_and_grad(minus, True)
            hessian[:, dim] = (gradient_plus - gradient_minus) / denominator
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues = np.linalg.eigvalsh(hessian)
        abs_eigenvalues = np.abs(eigenvalues)
        condition = float(
            np.max(abs_eigenvalues) / max(np.min(abs_eigenvalues), 1.0e-300)
        )
        return {
            "common_geometry_hessian_min_eig": float(eigenvalues[0]),
            "common_geometry_hessian_condition": condition,
        }
    except (KeyError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        return {
            "common_geometry_hessian_min_eig": float("nan"),
            "common_geometry_hessian_condition": float("inf"),
        }


def _reconstruct(final: dict, data: dict, config: dict) -> np.ndarray:
    dictionary = build_jones_vp_dictionary(
        np.asarray(final["p_u"], dtype=float),
        float(final["delta_t"]),
        data["scene"],
        config,
    )
    return (dictionary @ np.asarray(final["x_hat"], dtype=complex)).reshape(
        data["Y_noisy"].shape
    )


def _stage1_delay_diagnostics(stage1: dict, data: dict) -> dict[str, Any]:
    """Return panel-ordered delay and pairing diagnostics for one Stage-I result."""
    try:
        poles = np.asarray(stage1["poles"], dtype=complex).reshape(-1)
        truth = np.asarray(data["true_components"]["taus"], dtype=float).reshape(-1)
        if poles.size != truth.size or poles.size == 0:
            raise ValueError("Stage-I and truth delay dimensions differ")
        delta_f = float(data["scene"]["delta_f"])
        estimated = np.asarray(
            [tau_from_pole(pole, delta_f) for pole in poles], dtype=float
        )
        direct_error_ns = (estimated - truth) * 1.0e9
        permutations = list(itertools.permutations(range(truth.size)))
        costs = [
            float(np.sum(np.abs(estimated - truth[list(permutation)])))
            for permutation in permutations
        ]
        best_permutation = permutations[int(np.argmin(costs))]
        pairwise = [
            abs(float(estimated[left] - estimated[right])) * 1.0e9
            for left in range(estimated.size)
            for right in range(left + 1, estimated.size)
        ]
        return {
            "stage1_delay_rmse_ns": float(np.sqrt(np.mean(direct_error_ns**2))),
            "stage1_delay_max_abs_error_ns": float(np.max(np.abs(direct_error_ns))),
            "stage1_min_delay_separation_ns": (
                float(min(pairwise)) if pairwise else float("inf")
            ),
            "stage1_panel_pairing_correct": bool(
                best_permutation == tuple(range(truth.size))
            ),
        }
    except (KeyError, TypeError, ValueError, FloatingPointError):
        return {
            "stage1_delay_rmse_ns": float("nan"),
            "stage1_delay_max_abs_error_ns": float("nan"),
            "stage1_min_delay_separation_ns": float("nan"),
            "stage1_panel_pairing_correct": False,
        }


def _delay_subspace_scalars(stage1: dict, k_paths: int) -> dict[str, float]:
    """Scalar summaries of the delay-subspace singular spectrum (rank-K monitoring)."""
    try:
        sigma = np.sort(
            np.abs(
                np.asarray(
                    stage1["stage1_delay_singular_values"], dtype=float
                ).ravel()
            )
        )[::-1]
    except (KeyError, TypeError, ValueError):
        sigma = np.array([], dtype=float)
    if k_paths <= 0 or sigma.size < k_paths or sigma[k_paths - 1] <= 0.0:
        return {
            "stage1_delay_sigma_k": float("nan"),
            "stage1_delay_sigma_kplus1_over_k": float("nan"),
            "stage1_delay_subspace_condition": float("nan"),
        }
    sigma_k = float(sigma[k_paths - 1])
    return {
        "stage1_delay_sigma_k": sigma_k,
        "stage1_delay_sigma_kplus1_over_k": (
            float(sigma[k_paths] / sigma_k) if sigma.size > k_paths else float("nan")
        ),
        "stage1_delay_subspace_condition": float(sigma[0] / sigma_k),
    }


def _variant_execution_metadata(variant: str, scene: dict) -> dict[str, Any]:
    if variant in {"old_4d", "scaled_4d", "old_stage1_ccop"}:
        initializer_family = "frozen_stage1"
    elif variant in GOOD_INITIALIZER_VARIANTS:
        initializer_family = "mksc_gi_4_starts_refresh"
    else:
        initializer_family = f"component_specific:{variant}"

    if variant in FOUR_D_CLOCK_COORDINATES:
        parameterization = FOUR_D_CLOCK_COORDINATES[variant]
        scale = {
            "seconds": 1.0,
            "nanoseconds": 1.0e9,
            "distance_m": float(scene["c0"]),
        }[parameterization]
        inner_solver = "explicit_4d_lbfgsb"
        affects_solver = True
        nonlinear_dim = 4
    else:
        parameterization = CCOP_CLOCK_UNIT_VARIANTS.get(
            variant, "profiled_unit_free"
        )
        scale = {
            "seconds": 1.0,
            "nanoseconds": 1.0e9,
            "distance_m": float(scene["c0"]),
        }.get(parameterization, float("nan"))
        inner_solver = "ccop_profiled_3d"
        affects_solver = False
        nonlinear_dim = 3
    return {
        "initializer_family": initializer_family,
        "inner_solver": inner_solver,
        "clock_parameterization": parameterization,
        "clock_coordinate_scale_per_second": scale,
        "clock_parameterization_affects_solver": affects_solver,
        "nonlinear_dim": nonlinear_dim,
    }


def run_paper_variant(
    variant: str,
    *,
    data: dict,
    config: dict,
    cache: Stage1Cache | None = None,
    suite: str = "",
    x_name: str = "",
    x_value: Any = "",
    trial_id: int = 0,
    outlier_threshold_m: float = 0.1,
) -> dict[str, Any]:
    """Run one route and return a light, publication-oriented CSV row."""
    if variant not in PAPER_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    stage1_cache = cache if cache is not None else Stage1Cache(data, config)
    execution_metadata = _variant_execution_metadata(variant, data["scene"])
    row = {field: "" for field in TRIAL_FIELDS}
    row.update(
        {
            "suite": str(suite),
            "x_name": str(x_name),
            "x_value": x_value,
            "trial_id": int(trial_id),
            "seed": int(config["seed"]),
            "snr_db": float(config["SNR_dB"]),
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            **execution_metadata,
            "failed": False,
            "error": "",
            "resolved_config_hash": canonical_hash(config),
            "y_noisy_hash": array_sha256(data["Y_noisy"]),
            "base_y_noisy_hash": str(
                data.get("nested_base_y_noisy_hash", array_sha256(data["Y_noisy"]))
            ),
            "nested_receiver_noise_convention": str(
                data.get("nested_receiver_noise_convention", "")
            ),
            "reference_sigma2": float(
                data.get("reference_sigma2", data.get("noise_variance", np.nan))
            ),
        }
    )
    try:
        stage1, stage1_times = _stage1_for_variant(variant, stage1_cache)
        xi0 = _initial_xi_from_stage1(stage1, data["scene"], config)
        stage3_start = time.perf_counter()
        if variant in FOUR_D_CLOCK_COORDINATES:
            final = _quiet_call(
                refine_four_dimensional_jvp_experimental,
                data["Y_noisy"],
                copy.deepcopy(stage1),
                data["scene"],
                config,
                clock_coordinate=FOUR_D_CLOCK_COORDINATES[variant],
                max_iter=80,
            )
        else:
            final = _quiet_call(
                refine_ccop_jvp,
                data["Y_noisy"],
                copy.deepcopy(stage1),
                data["scene"],
                config,
                incumbent=None,
            )
        stage3_runtime = time.perf_counter() - stage3_start
        y_hat = _reconstruct(final, data, config)
        p_hat = np.asarray(final["p_u"], dtype=float).reshape(3)
        p_true = np.asarray(data["scene"]["p_u_true"], dtype=float).reshape(3)
        position_error = float(np.linalg.norm(p_hat - p_true))
        clock_error = float(
            abs(float(final["delta_t"]) - float(data["scene"]["delta_t_true"]))
            * 1.0e9
        )
        optimizer = dict(final.get("optimizer", {}))
        certificate_gap = final.get(
            "clock_certificate_gap_objective",
            final.get(
                "clock_certificate_gap",
                final.get("clock_global_gap", final.get("clock_objective_gap", np.nan)),
            ),
        )
        certificate_tolerance = (
            float(final.get("clock_certificate_tolerance_score", np.nan))
            / max(np.size(data["Y_noisy"]), 1)
        )
        certificate_gap_ratio = (
            float(certificate_gap) / certificate_tolerance
            if np.isfinite(float(certificate_gap))
            and np.isfinite(certificate_tolerance)
            and certificate_tolerance > 0.0
            else float("nan")
        )
        is_four_dimensional = variant in FOUR_D_CLOCK_COORDINATES
        row.update(
            {
                "stage1_output_hash": canonical_hash(
                    deterministic_stage1_output(stage1)
                ),
                "candidate_hash": canonical_hash(
                    {
                        "p_u": p_hat,
                        "delta_t": float(final["delta_t"]),
                        "x_hat": np.asarray(final["x_hat"], dtype=complex),
                        "raw_objective_final": float(final["raw_objective_final"]),
                    }
                ),
                "position_error_m": position_error,
                "clock_error_ns": clock_error,
                "channel_nmse": float(relative_nmse(y_hat, data["Y_true"])),
                "noisy_fit_nmse": float(relative_nmse(y_hat, data["Y_noisy"])),
                "outlier": bool(position_error > float(outlier_threshold_m)),
                "raw_objective_final": float(final["raw_objective_final"]),
                "total_objective_final": float(final["total_objective_final"]),
                "stage1_position_error_m": float(np.linalg.norm(xi0[:3] - p_true)),
                **_stage1_delay_diagnostics(stage1, data),
                **stage1_times,
                "stage3_runtime_s": float(stage3_runtime),
                "deployment_runtime_s": float(
                    stage1_times["stage1_runtime_s"] + stage3_runtime
                ),
                "optimizer_evaluations": int(
                    optimizer.get("n_eval", final.get("ccop_position_evaluations", 0))
                ),
                "optimizer_iterations": int(optimizer.get("n_iter", 0)),
                "clock_interval_evaluations": int(
                    final.get("ccop_clock_interval_evaluations", 0)
                ),
                "clock_profile_evaluations": (
                    int(final.get("ccop_clock_profile_evaluations", 0))
                    if not is_four_dimensional
                    else 0
                ),
                "clock_profile_certified_count": (
                    int(final.get("ccop_clock_profile_certified_count", 0))
                    if not is_four_dimensional
                    else 0
                ),
                "clock_profiles_all_certified": (
                    bool(final.get("ccop_clock_profiles_all_certified", False))
                    if not is_four_dimensional
                    else ""
                ),
                "clock_certified": (
                    bool(final.get("clock_certified", False))
                    if not is_four_dimensional
                    else ""
                ),
                "clock_certificate_gap": (
                    float(certificate_gap)
                    if not is_four_dimensional
                    else float("nan")
                ),
                "clock_certificate_tolerance": (
                    certificate_tolerance
                    if not is_four_dimensional
                    else float("nan")
                ),
                "clock_certificate_gap_ratio": (
                    certificate_gap_ratio
                    if not is_four_dimensional
                    else float("nan")
                ),
                "clock_certificate_gap_max_over_positions": (
                    float(
                        final.get(
                            "ccop_clock_profile_max_certificate_gap_objective",
                            np.nan,
                        )
                    )
                    if not is_four_dimensional
                    else float("nan")
                ),
                "clock_certificate_gap_ratio_max_over_positions": (
                    float(
                        final.get(
                            "ccop_clock_profile_max_certificate_gap_ratio",
                            np.nan,
                        )
                    )
                    if not is_four_dimensional
                    else float("nan")
                ),
                "clock_fft_peak_gap": (
                    float(final.get("clock_fft_peak_gap_objective", np.nan))
                    if not is_four_dimensional
                    else float("nan")
                ),
                "clock_branch_gap_tolerance": (
                    float(
                        final.get(
                            "clock_branch_gap_tolerance_objective", np.nan
                        )
                    )
                    if not is_four_dimensional
                    else float("nan")
                ),
                "clock_bnb_splits": (
                    int(final.get("clock_bnb_splits", 0))
                    if not is_four_dimensional
                    else 0
                ),
                "clock_branch_ambiguous": bool(
                    final.get("clock_branch_ambiguous_seen", False)
                ),
                "outer_safeguard_used": bool(
                    final.get("outer_branch_safeguard_used", False)
                ),
                "selected_candidate": str(final.get("selected_candidate", "")),
                "stage1_evs_union_rank": int(
                    stage1.get("stage1_evs_union_rank", data["scene"]["I"])
                ),
                "stage1_evs_retained_energy_fraction": float(
                    stage1.get("stage1_evs_retained_energy_fraction", 1.0)
                ),
                **_delay_subspace_scalars(stage1, int(data["scene"]["K"])),
                "stage1_assignment_margin": float(
                    stage1.get("stage1_assignment_margin", np.nan)
                ),
                "stage1_max_rank1_ratio": float(
                    stage1.get("stage1_max_rank1_ratio", np.nan)
                ),
                "common_geometry_hessian_min_eig": float(
                    stage1.get("common_geometry_hessian_min_eig", np.nan)
                ),
                "common_geometry_hessian_condition": float(
                    stage1.get("common_geometry_hessian_condition", np.nan)
                ),
                "stage1_basin_acquired": bool(
                    np.linalg.norm(xi0[:3] - p_true) <= float(outlier_threshold_m)
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - failures are first-class MC results.
        row.update(
            {
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
                "outlier": True,
            }
        )
    scene = data["scene"]
    p_true = np.asarray(scene["p_u_true"], dtype=float)
    row.update(
        {
            "receiver_mode": str(scene.get("receiver_mode", config.get("receiver_mode", ""))),
            "N": int(scene["N"]),
            "T": int(scene["T"]),
            "K": int(scene["K"]),
            "M_A": int(scene["M_A"]),
            "M_R": int(scene["M_R"]),
            "p_true_x": float(p_true[0]),
            "p_true_y": float(p_true[1]),
            "p_true_z": float(p_true[2]),
        }
    )
    return row


def run_paired_variants(
    variants: Iterable[str],
    *,
    data: dict,
    config: dict,
    suite: str,
    x_name: str,
    x_value: Any,
    trial_id: int,
    outlier_threshold_m: float = 0.1,
    profile_memory: bool = False,
) -> list[dict[str, Any]]:
    cache = Stage1Cache(data, config)
    rows = []
    for variant in variants:
        rss_before = memory_snapshot_mb() if profile_memory else float("nan")
        with PeakRssSampler(profile_memory) as memory_sampler:
            row = run_paper_variant(
                variant,
                data=data,
                config=config,
                cache=cache,
                suite=suite,
                x_name=x_name,
                x_value=x_value,
                trial_id=trial_id,
                outlier_threshold_m=outlier_threshold_m,
            )
        row.update(
            {
                "rss_mb_before": rss_before,
                "rss_mb_peak": memory_sampler.peak_mb,
                "rss_mb_after": (
                    memory_snapshot_mb() if profile_memory else float("nan")
                ),
                "memory_measurement": (
                    "sampled_process_peak_rss_10ms" if profile_memory else ""
                ),
            }
        )
        rows.append(row)
    hashes = {str(row["y_noisy_hash"]) for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"paired variants did not share Y_noisy: {sorted(hashes)}")
    for initializer_family in {
        str(row.get("initializer_family", "")) for row in rows
    }:
        stage1_hashes = {
            str(row["stage1_output_hash"])
            for row in rows
            if str(row.get("initializer_family", "")) == initializer_family
            and str(row.get("failed", "False")).lower() != "true"
            and str(row.get("stage1_output_hash", ""))
        }
        if len(stage1_hashes) > 1:
            raise RuntimeError(
                "paired variants did not share the declared initializer: "
                f"family={initializer_family} hashes={sorted(stage1_hashes)}"
            )
    ccop_unit_rows = [
        row
        for row in rows
        if str(row.get("variant", "")) in CCOP_CLOCK_UNIT_VARIANTS
        and str(row.get("failed", "False")).lower() != "true"
    ]
    if len(ccop_unit_rows) > 1:
        candidate_hashes = {
            str(row.get("candidate_hash", "")) for row in ccop_unit_rows
        }
        if len(candidate_hashes) != 1:
            raise RuntimeError(
                "CCOP changed under a clock-unit label even though the clock "
                f"is profiled: hashes={sorted(candidate_hashes)}"
            )
    return rows


def write_csv(path: pathlib.Path, rows: Iterable[dict[str, Any]], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _finite(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        if str(row.get("failed", "False")).lower() == "true":
            continue
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _binomial_interval(successes: int, total: int, alpha: float = 0.05):
    if total <= 0:
        return float("nan"), float("nan"), "unavailable"
    try:
        from scipy.stats import beta

        low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
        high = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
        return low, high, "Clopper-Pearson exact"
    except (ImportError, ValueError):
        p = successes / total
        z = 1.959963984540054
        denom = 1.0 + z * z / total
        center = (p + z * z / (2 * total)) / denom
        half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
        return max(0.0, center - half), min(1.0, center + half), "Wilson"


def summarize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("suite", "")),
            str(row.get("x_name", "")),
            str(row.get("x_value", "")),
            str(row.get("variant", "")),
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for (suite, x_name, x_value, variant), selected in sorted(grouped.items()):
        variant_label = next(
            (
                str(row.get("variant_label"))
                for row in selected
                if str(row.get("variant_label", "")).strip()
            ),
            VARIANT_LABELS.get(variant, variant),
        )
        successful_rows = [
            row
            for row in selected
            if str(row.get("failed", "False")).lower() != "true"
        ]
        position = _finite(successful_rows, "position_error_m")
        conditional_position = _finite(
            [
                row
                for row in selected
                if str(row.get("failed", "False")).lower() != "true"
                and str(row.get("outlier", "False")).lower() != "true"
            ],
            "position_error_m",
        )
        peb_position = _finite(successful_rows, "peb_position_m")
        clock = _finite(selected, "clock_error_ns")
        channel = _finite(selected, "channel_nmse")
        runtime = _finite(selected, "deployment_runtime_s")
        peak_rss = _finite(selected, "rss_mb_peak")
        stage1_runtime = _finite(selected, "stage1_runtime_s")
        mksc_runtime = _finite(selected, "mksc_projection_runtime_s")
        geometry_runtime = _finite(selected, "common_geometry_runtime_s")
        refresh_runtime = _finite(selected, "anchor_refresh_runtime_s")
        stage3_runtime = _finite(selected, "stage3_runtime_s")
        nonlinear_dim = _finite(selected, "nonlinear_dim")
        optimizer_evaluations = _finite(selected, "optimizer_evaluations")
        optimizer_iterations = _finite(selected, "optimizer_iterations")
        clock_profile_evaluations = _finite(
            selected, "clock_profile_evaluations"
        )
        certificate_gap = _finite(selected, "clock_certificate_gap")
        certificate_tolerance = _finite(
            selected, "clock_certificate_tolerance"
        )
        certificate_gap_ratio = _finite(
            selected, "clock_certificate_gap_ratio"
        )
        trajectory_certificate_gap = _finite(
            selected, "clock_certificate_gap_max_over_positions"
        )
        trajectory_certificate_gap_ratio = _finite(
            selected, "clock_certificate_gap_ratio_max_over_positions"
        )
        fft_peak_gap = _finite(selected, "clock_fft_peak_gap")
        certificate_values = [
            str(row.get("clock_certified", "")).lower() == "true"
            for row in selected
            if str(row.get("clock_certified", "")).lower() in {"true", "false"}
        ]
        all_positions_certificate_values = [
            str(row.get("clock_profiles_all_certified", "")).lower() == "true"
            for row in selected
            if str(row.get("clock_profiles_all_certified", "")) != ""
        ]
        stage1_delay_rmse = _finite(selected, "stage1_delay_rmse_ns")
        hessian_minimum = _finite(selected, "common_geometry_hessian_min_eig")
        failed = sum(str(row.get("failed", "False")).lower() == "true" for row in selected)
        successful = len(selected) - failed
        conditional_outliers = sum(
            str(row.get("failed", "False")).lower() != "true"
            and str(row.get("outlier", "False")).lower() == "true"
            for row in selected
        )
        outliers = sum(
            str(row.get("failed", "False")).lower() == "true"
            or str(row.get("outlier", "False")).lower() == "true"
            for row in selected
        )
        is_reference = variant.startswith("free_jones_peb") or variant.startswith(
            "constrained_jones_peb"
        )
        if is_reference:
            low, high, method = float("nan"), float("nan"), "not_applicable"
        else:
            low, high, method = _binomial_interval(outliers, len(selected))

        def percentile(values: np.ndarray, q: float) -> float:
            return float(np.percentile(values, q)) if values.size else float("nan")

        summary.append(
            {
                "suite": suite,
                "x_name": x_name,
                "x_value": x_value,
                "variant": variant,
                "variant_label": variant_label,
                "initializer_family": str(
                    selected[0].get("initializer_family", "")
                ),
                "inner_solver": str(selected[0].get("inner_solver", "")),
                "clock_parameterization": str(
                    selected[0].get("clock_parameterization", "")
                ),
                "clock_coordinate_scale_per_second": selected[0].get(
                    "clock_coordinate_scale_per_second", float("nan")
                ),
                "clock_parameterization_affects_solver": selected[0].get(
                    "clock_parameterization_affects_solver", ""
                ),
                "n": len(selected),
                "n_failed": int(failed),
                "n_success": int(successful),
                "success_rate": float(successful / len(selected)) if selected else float("nan"),
                "failure_rate": float(failed / len(selected)) if selected else float("nan"),
                "position_rmse_m": float(np.sqrt(np.mean(position**2))) if position.size else float("nan"),
                "position_mean_error_m": float(np.mean(position))
                if position.size
                else float("nan"),
                "position_conditional_rmse_m": float(
                    np.sqrt(np.mean(conditional_position**2))
                )
                if conditional_position.size
                else float("nan"),
                "position_median_m": percentile(position, 50),
                "position_p90_m": percentile(position, 90),
                "position_p95_m": percentile(position, 95),
                "peb_position_m_mean": float(np.mean(peb_position))
                if peb_position.size
                else float("nan"),
                "peb_position_m_rms": float(np.sqrt(np.mean(peb_position**2)))
                if peb_position.size
                else float("nan"),
                "clock_rmse_ns": float(np.sqrt(np.mean(clock**2))) if clock.size else float("nan"),
                "clock_median_ns": percentile(clock, 50),
                "clock_p90_ns": percentile(clock, 90),
                "clock_p95_ns": percentile(clock, 95),
                "channel_nmse_mean": float(np.mean(channel)) if channel.size else float("nan"),
                "channel_nmse_median": percentile(channel, 50),
                "channel_nmse_p90": percentile(channel, 90),
                "channel_nmse_p95": percentile(channel, 95),
                "conditional_outlier_rate": (
                    float(conditional_outliers / successful)
                    if successful and not is_reference
                    else float("nan")
                ),
                "outlier_rate": (
                    float("nan")
                    if is_reference
                    else float(outliers / len(selected)) if selected else float("nan")
                ),
                "catastrophic_rate": (
                    float("nan")
                    if is_reference
                    else float(outliers / len(selected)) if selected else float("nan")
                ),
                "outlier_ci_low": low,
                "outlier_ci_high": high,
                "outlier_ci_method": method,
                "runtime_mean_s": float(np.mean(runtime)) if runtime.size else float("nan"),
                "runtime_median_s": percentile(runtime, 50),
                "runtime_p95_s": percentile(runtime, 95),
                "peak_rss_median_mb": percentile(peak_rss, 50),
                "peak_rss_max_mb": (
                    float(np.max(peak_rss)) if peak_rss.size else float("nan")
                ),
                "stage1_runtime_median_s": percentile(stage1_runtime, 50),
                "mksc_projection_runtime_median_s": percentile(mksc_runtime, 50),
                "common_geometry_runtime_median_s": percentile(geometry_runtime, 50),
                "anchor_refresh_runtime_median_s": percentile(refresh_runtime, 50),
                "stage3_runtime_median_s": percentile(stage3_runtime, 50),
                "nonlinear_dim_median": percentile(nonlinear_dim, 50),
                "optimizer_evaluations_median": percentile(
                    optimizer_evaluations, 50
                ),
                "optimizer_iterations_median": percentile(
                    optimizer_iterations, 50
                ),
                "clock_profile_evaluations_median": percentile(
                    clock_profile_evaluations, 50
                ),
                "clock_all_positions_certified_rate": (
                    float(np.mean(all_positions_certificate_values))
                    if all_positions_certificate_values
                    else float("nan")
                ),
                "clock_certificate_rate": (
                    float(np.mean(certificate_values))
                    if certificate_values
                    else float("nan")
                ),
                "clock_certificate_gap_p95": percentile(certificate_gap, 95),
                "clock_certificate_tolerance_p95": percentile(
                    certificate_tolerance, 95
                ),
                "clock_certificate_gap_ratio_p95": percentile(
                    certificate_gap_ratio, 95
                ),
                "clock_certificate_gap_max_over_positions_max": percentile(
                    trajectory_certificate_gap, 100
                ),
                "clock_certificate_gap_ratio_max_over_positions_p95": percentile(
                    trajectory_certificate_gap_ratio, 95
                ),
                "clock_certificate_gap_ratio_max_over_positions_max": percentile(
                    trajectory_certificate_gap_ratio, 100
                ),
                "clock_fft_peak_gap_median": percentile(fft_peak_gap, 50),
                "stage1_basin_acquisition_rate": float(
                    sum(
                        str(row.get("stage1_basin_acquired", "False")).lower()
                        == "true"
                        for row in selected
                    )
                    / len(selected)
                ) if selected else float("nan"),
                "stage1_delay_rmse_ns_median": percentile(stage1_delay_rmse, 50),
                "stage1_delay_rmse_ns_p95": percentile(stage1_delay_rmse, 95),
                "common_geometry_hessian_min_eig_median": percentile(
                    hessian_minimum, 50
                ),
            }
        )
    return summary


def paired_summary(
    rows: Iterable[dict[str, Any]],
    *,
    reference_variant: str = "old_4d",
    candidate_variant: str = "proposed",
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260716,
) -> list[dict[str, Any]]:
    """Return paired tail/runtime statistics with deterministic bootstrap CIs."""
    grouped: dict[tuple[str, str, str], dict[tuple[int, int], dict[str, dict]]] = {}
    for row in rows:
        if str(row.get("variant", "")) not in {reference_variant, candidate_variant}:
            continue
        key = (
            str(row.get("suite", "")),
            str(row.get("x_name", "")),
            str(row.get("x_value", "")),
        )
        trial_key = (int(row.get("trial_id", 0)), int(row.get("seed", 0)))
        grouped.setdefault(key, {}).setdefault(trial_key, {})[
            str(row["variant"])
        ] = row

    rng = np.random.default_rng(int(bootstrap_seed))
    result = []
    for (suite, x_name, x_value), trials in sorted(grouped.items()):
        all_pairs = [
            (by_variant[reference_variant], by_variant[candidate_variant])
            for by_variant in trials.values()
            if reference_variant in by_variant
            and candidate_variant in by_variant
        ]
        if not all_pairs:
            continue
        pairs = [
            pair
            for pair in all_pairs
            if str(pair[0].get("failed", "False")).lower() != "true"
            and str(pair[1].get("failed", "False")).lower() != "true"
        ]
        ref_position = np.asarray([float(pair[0]["position_error_m"]) for pair in pairs])
        cand_position = np.asarray([float(pair[1]["position_error_m"]) for pair in pairs])
        ref_clock = np.asarray([float(pair[0]["clock_error_ns"]) for pair in pairs])
        cand_clock = np.asarray([float(pair[1]["clock_error_ns"]) for pair in pairs])
        ref_channel = np.asarray([float(pair[0]["channel_nmse"]) for pair in pairs])
        cand_channel = np.asarray([float(pair[1]["channel_nmse"]) for pair in pairs])
        ref_raw = np.asarray([float(pair[0]["raw_objective_final"]) for pair in pairs])
        cand_raw = np.asarray([float(pair[1]["raw_objective_final"]) for pair in pairs])
        ref_runtime = np.asarray([float(pair[0]["deployment_runtime_s"]) for pair in pairs])
        cand_runtime = np.asarray([float(pair[1]["deployment_runtime_s"]) for pair in pairs])
        ref_outlier = np.asarray(
            [
                str(pair[0].get("failed", "False")).lower() == "true"
                or str(pair[0].get("outlier", "False")).lower() == "true"
                for pair in all_pairs
            ],
            dtype=bool,
        )
        cand_outlier = np.asarray(
            [
                str(pair[1].get("failed", "False")).lower() == "true"
                or str(pair[1].get("outlier", "False")).lower() == "true"
                for pair in all_pairs
            ],
            dtype=bool,
        )
        rescued = int(np.sum(ref_outlier & ~cand_outlier))
        introduced = int(np.sum(~ref_outlier & cand_outlier))
        discordant = rescued + introduced
        try:
            from scipy.stats import binomtest

            mcnemar_p = (
                float(binomtest(min(rescued, introduced), discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
        except ImportError:
            mcnemar_p = float("nan")

        rmse_difference = (
            float(np.sqrt(np.mean(cand_position**2)) - np.sqrt(np.mean(ref_position**2)))
            if pairs
            else float("nan")
        )
        runtime_ratio = (
            float(np.median(cand_runtime / np.maximum(ref_runtime, 1.0e-300)))
            if pairs
            else float("nan")
        )
        boot_rmse = []
        boot_runtime = []
        n_pairs = len(all_pairs)
        n_metric_pairs = len(pairs)
        for _ in range(max(0, int(bootstrap_replicates))):
            if n_metric_pairs == 0:
                break
            indices = rng.integers(0, n_metric_pairs, size=n_metric_pairs)
            boot_rmse.append(
                float(
                    np.sqrt(np.mean(cand_position[indices] ** 2))
                    - np.sqrt(np.mean(ref_position[indices] ** 2))
                )
            )
            boot_runtime.append(
                float(
                    np.median(
                        cand_runtime[indices]
                        / np.maximum(ref_runtime[indices], 1.0e-300)
                    )
                )
            )
        if boot_rmse:
            rmse_low, rmse_high = np.percentile(boot_rmse, [2.5, 97.5])
            runtime_low, runtime_high = np.percentile(boot_runtime, [2.5, 97.5])
        else:
            rmse_low = rmse_high = runtime_low = runtime_high = float("nan")
        result.append(
            {
                "suite": suite,
                "x_name": x_name,
                "x_value": x_value,
                "reference_variant": reference_variant,
                "candidate_variant": candidate_variant,
                "n_pairs": n_pairs,
                "n_metric_pairs": n_metric_pairs,
                "position_win_rate": (
                    float(np.mean(cand_position < ref_position)) if pairs else float("nan")
                ),
                "clock_win_rate": (
                    float(np.mean(cand_clock < ref_clock)) if pairs else float("nan")
                ),
                "channel_win_rate": (
                    float(np.mean(cand_channel < ref_channel)) if pairs else float("nan")
                ),
                "raw_objective_non_degradation_rate": float(
                    np.mean(cand_raw <= ref_raw + 1.0e-12)
                ) if pairs else float("nan"),
                "rescued_outliers": rescued,
                "introduced_outliers": introduced,
                "mcnemar_exact_p": mcnemar_p,
                "paired_position_rmse_difference_m": rmse_difference,
                "paired_position_rmse_difference_ci_low_m": float(rmse_low),
                "paired_position_rmse_difference_ci_high_m": float(rmse_high),
                "paired_runtime_ratio_median": runtime_ratio,
                "paired_runtime_ratio_ci_low": float(runtime_low),
                "paired_runtime_ratio_ci_high": float(runtime_high),
                "bootstrap_replicates": int(bootstrap_replicates),
            }
        )
    return result


def save_run_manifest(
    out_dir: pathlib.Path,
    *,
    command: str,
    arguments: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(dict(arguments), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (out_dir / "environment.json").write_text(
        json.dumps(dict(environment), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (out_dir / "command.txt").write_text(str(command) + "\n", encoding="utf-8")


def _json_safe(value: Any):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "real": np.asarray(value.real).tolist(),
                "imag": np.asarray(value.imag).tolist(),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def save_resolved_config(out_dir: pathlib.Path, config: dict, filename: str) -> None:
    """Save a human-readable resolved config together with its canonical hash."""
    payload = {
        "resolved_config_hash": canonical_hash(config),
        "resolved_config": _json_safe(config),
    }
    (out_dir / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
