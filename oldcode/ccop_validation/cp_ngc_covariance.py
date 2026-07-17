"""Experimental covariance and held-out calibration helpers for CP-NGC.

The routines here do not alter Stage-I or the frozen selector.  C1 is a
conditional local linearization of the fitted raw-domain factor model; C2 is
an optional per-realization parametric bootstrap; C3 contains validation-only
empirical calibration helpers; and C4 implements a one-way RIS-training-block
split with full bandwidth in each fold.
"""

from __future__ import annotations

import contextlib
import copy
import io
from typing import Any

import numpy as np

from src.ccop_jvp import CommonClockJonesProfiler, refine_ccop_jvp
from .cp_ngc import (
    cp_ngc_stage1_vector,
    cp_ngc_statistic,
    regularize_cp_ngc_covariance,
)
from src.estimators import initialize_from_hankel
from src.projections_delay import tau_from_pole
from src.projections_ris import exact_spherical_response_and_jacobian
from src.tensor_utils import dehankelize_frequency, hankelize_frequency


def _active_evs_indices(scene: dict) -> np.ndarray:
    mask = scene.get("evs_observation_mask")
    if mask is None:
        return np.arange(int(scene["I"]), dtype=int)
    active = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
    if active.size == 0:
        raise ValueError("scene contains no active EVS observations")
    return active


def linearized_stage1_covariance(
    y_raw: np.ndarray,
    stage1_estimate: dict,
    scene: dict,
    noise_variance: float,
    *,
    shrinkage: float = 0.02,
    eigenvalue_floor_relative: float = 1.0e-7,
) -> dict:
    """C1: conditional covariance of ``[tau, RIS range/elev/az]``.

    The fitted Stage-I EVS factors are conditioned on, while complex path
    weights are eliminated as nuisance parameters through a Fisher Schur
    complement.  Delay derivatives use one raw-frequency mother factor, so B
    and Q are never treated as independent observations.  RIS derivatives use
    ``c_k = Omega_k g_k`` through the exact compressed spherical Jacobian.
    """
    expected = (int(scene["I"]), int(scene["N"]), int(scene["T"]))
    if tuple(np.asarray(y_raw).shape) != expected:
        raise ValueError(f"y_raw must have shape {expected}")
    sigma2 = float(noise_variance)
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("noise_variance must be positive")
    k_paths = int(scene["K"])
    a_mat = np.asarray(stage1_estimate["A"], dtype=complex)
    c_mat = np.asarray(stage1_estimate["C"], dtype=complex)
    beta = np.asarray(stage1_estimate.get("beta_z"), dtype=complex).reshape(k_paths)
    eta = np.asarray(stage1_estimate["ris_eta"], dtype=float).reshape(k_paths, 3)
    poles = np.asarray(stage1_estimate["poles"], dtype=complex).reshape(k_paths)
    if a_mat.shape != (int(scene["I"]), k_paths):
        raise ValueError("Stage-I A shape is incompatible with scene")
    if c_mat.shape != (int(scene["T"]), k_paths):
        raise ValueError("Stage-I C shape is incompatible with scene")

    active = _active_evs_indices(scene)
    a_active = a_mat[active]
    n_index = np.arange(int(scene["N"]), dtype=float)
    dimension = 4 * k_paths
    interest = np.empty(
        (active.size * int(scene["N"]) * int(scene["T"]), dimension),
        dtype=complex,
    )
    nuisance = np.empty((interest.shape[0], 2 * k_paths), dtype=complex)
    interest.fill(0.0)
    nuisance.fill(0.0)
    ris_alpha = np.empty(k_paths, dtype=complex)
    for path in range(k_paths):
        tau = tau_from_pole(poles[path], float(scene["delta_f"]))
        delay = np.exp(
            -1j * 2.0 * np.pi * float(scene["delta_f"]) * tau * n_index
        )
        delay_derivative = (
            -1j * 2.0 * np.pi * float(scene["delta_f"]) * n_index * delay
        )
        atom = (
            a_active[:, path, None, None]
            * delay[None, :, None]
            * c_mat[None, :, path]
        ).reshape(-1)
        nuisance[:, 2 * path] = atom
        nuisance[:, 2 * path + 1] = 1j * atom
        interest[:, path] = (
            beta[path]
            * a_active[:, path, None, None]
            * delay_derivative[None, :, None]
            * c_mat[None, :, path]
        ).reshape(-1)

        response, response_jacobian = exact_spherical_response_and_jacobian(
            eta[path],
            np.asarray(scene["Omega"][path], dtype=complex),
            np.asarray(scene["a_RB"][path], dtype=complex),
            np.asarray(scene["ris_grid"], dtype=float),
            float(scene["wavelength"]),
        )
        denominator = float(np.vdot(response, response).real)
        alpha = (
            np.vdot(response, c_mat[:, path]) / denominator
            if denominator > 0.0
            else 0.0j
        )
        ris_alpha[path] = alpha
        for component in range(3):
            interest[:, k_paths + 3 * path + component] = (
                beta[path]
                * a_active[:, path, None, None]
                * delay[None, :, None]
                * (alpha * response_jacobian[:, component])[None, None, :]
            ).reshape(-1)

    # Parameter scaling is numerical only: tau in ns, range in m, angles in rad.
    physical_scales = np.concatenate(
        [np.full(k_paths, 1.0e-9), np.tile([1.0, 1.0, 1.0], k_paths)]
    )
    scaled_interest = interest * physical_scales[None, :]
    fisher_gg = (2.0 / sigma2) * np.real(scaled_interest.conj().T @ scaled_interest)
    fisher_gn = (2.0 / sigma2) * np.real(scaled_interest.conj().T @ nuisance)
    fisher_nn = (2.0 / sigma2) * np.real(nuisance.conj().T @ nuisance)
    fisher_effective = fisher_gg - fisher_gn @ np.linalg.pinv(
        fisher_nn, rcond=1.0e-12
    ) @ fisher_gn.T
    fisher_effective = 0.5 * (fisher_effective + fisher_effective.T)
    eigvals, eigvecs = np.linalg.eigh(fisher_effective)
    largest = max(float(np.max(eigvals)), np.finfo(float).tiny)
    information_floor = max(
        largest * float(eigenvalue_floor_relative), np.finfo(float).tiny
    )
    floored_information = np.maximum(eigvals, information_floor)
    covariance_scaled = eigvecs @ np.diag(1.0 / floored_information) @ eigvecs.T
    covariance = (
        physical_scales[:, None]
        * covariance_scaled
        * physical_scales[None, :]
    )
    covariance, regularization = regularize_cp_ngc_covariance(
        covariance,
        shrinkage=shrinkage,
        eigenvalue_floor_relative=eigenvalue_floor_relative,
    )
    fitted = dehankelize_frequency(
        np.asarray(stage1_estimate["Z_hat"], dtype=complex), int(scene["N"])
    )
    residual = np.asarray(y_raw, dtype=complex)[active] - fitted[active]
    fisher_rank = int(
        np.sum(eigvals > largest * float(eigenvalue_floor_relative))
    )
    return {
        "covariance_z": covariance,
        "C_tau": covariance[:k_paths, :k_paths],
        "C_xi": covariance[k_paths:, k_paths:],
        "C_tau_xi": covariance[:k_paths, k_paths:],
        "source": "C1_conditional_raw_factor_linearization",
        "conditional_on_stage1_A_and_ris_alpha": True,
        "delay_uses_single_mother_factor": True,
        "ris_uses_compressed_exact_geometry": True,
        "ris_alpha": ris_alpha,
        "fisher_effective_scaled": fisher_effective,
        "fisher_eigenvalues_before_floor": eigvals,
        "fisher_rank_before_floor": fisher_rank,
        "full_fisher_rank": bool(fisher_rank == dimension),
        "covariance_reliable_for_hard_certificate": bool(fisher_rank == dimension),
        "fisher_condition_after_floor": float(
            np.max(floored_information) / np.min(floored_information)
        ),
        "regularization": regularization,
        "raw_fit_residual_to_noise_ratio": float(
            np.vdot(residual, residual).real / (residual.size * sigma2)
        ),
    }


def parametric_bootstrap_stage1_covariance(
    stage1_estimate: dict,
    scene: dict,
    config: dict,
    noise_variance: float,
    *,
    n_bootstrap: int = 24,
    bootstrap_seed: int = 0,
    shrinkage: float = 0.05,
    eigenvalue_floor_relative: float = 1.0e-6,
) -> dict:
    """C2: per-realization Stage-I parametric bootstrap covariance."""
    count = int(n_bootstrap)
    if count < 2:
        raise ValueError("n_bootstrap must be at least two")
    sigma2 = float(noise_variance)
    if sigma2 <= 0.0:
        raise ValueError("noise_variance must be positive")
    fitted_y = dehankelize_frequency(
        np.asarray(stage1_estimate["Z_hat"], dtype=complex), int(scene["N"])
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    active = _active_evs_indices(scene)
    reference = cp_ngc_stage1_vector(stage1_estimate, scene)
    samples: list[np.ndarray] = []
    failures: list[str] = []
    for _ in range(count):
        noise = np.zeros_like(fitted_y, dtype=complex)
        shape = fitted_y[active].shape
        noise[active] = np.sqrt(sigma2 / 2.0) * (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        )
        z_boot = hankelize_frequency(fitted_y + noise, int(scene["P"]))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                estimate = initialize_from_hankel(z_boot, scene, copy.deepcopy(config))
            sample = cp_ngc_stage1_vector(estimate, scene)
            period = 1.0 / float(scene["delta_f"])
            k_paths = int(scene["K"])
            delay_difference = (
                sample[:k_paths] - reference[:k_paths] + 0.5 * period
            ) % period - 0.5 * period
            sample[:k_paths] = reference[:k_paths] + delay_difference
            for path in range(k_paths):
                azimuth_index = k_paths + 3 * path + 2
                difference = np.angle(
                    np.exp(1j * (sample[azimuth_index] - reference[azimuth_index]))
                )
                sample[azimuth_index] = reference[azimuth_index] + difference
            if np.all(np.isfinite(sample)):
                samples.append(sample)
            else:
                failures.append("nonfinite_stage1_vector")
        except Exception as error:  # noqa: BLE001 - failures are bootstrap diagnostics.
            failures.append(f"{type(error).__name__}: {error}")
    if len(samples) < 2:
        raise RuntimeError("fewer than two valid Stage-I bootstrap replicates")
    sample_matrix = np.asarray(samples, dtype=float)
    covariance_empirical = np.cov(sample_matrix, rowvar=False, ddof=1)
    covariance, regularization = regularize_cp_ngc_covariance(
        covariance_empirical,
        shrinkage=shrinkage,
        eigenvalue_floor_relative=eigenvalue_floor_relative,
    )
    k_paths = int(scene["K"])
    return {
        "covariance_z": covariance,
        "C_tau": covariance[:k_paths, :k_paths],
        "C_xi": covariance[k_paths:, k_paths:],
        "C_tau_xi": covariance[:k_paths, k_paths:],
        "source": "C2_conditional_parametric_bootstrap",
        "n_requested": count,
        "n_valid": len(samples),
        "n_failed": len(failures),
        "failure_messages": failures,
        "sample_mean": np.mean(sample_matrix, axis=0),
        "sample_bias_from_fit": np.mean(sample_matrix, axis=0) - reference,
        "regularization": regularization,
    }


def reliability_stratum(diagnostics: dict, thresholds: dict) -> str:
    """C3 deterministic stratum assignment; thresholds come from validation."""
    if not bool(diagnostics.get("stage1_valid", True)) or bool(
        diagnostics.get("boundary_hit", False)
    ):
        return "invalid_or_boundary"
    margin = float(diagnostics.get("assignment_margin", np.nan))
    rank_one = float(diagnostics.get("max_rank_one_ratio", np.nan))
    clock_dispersion = float(diagnostics.get("clock_dispersion_ns", np.nan))
    ris_residual = float(diagnostics.get("max_ris_residual", np.nan))
    low = (
        not np.isfinite(margin)
        or margin < float(thresholds["assignment_margin_min"])
        or not np.isfinite(rank_one)
        or rank_one > float(thresholds["rank_one_ratio_max"])
        or not np.isfinite(clock_dispersion)
        or clock_dispersion > float(thresholds["clock_dispersion_ns_max"])
        or not np.isfinite(ris_residual)
        or ris_residual > float(thresholds["ris_residual_max"])
    )
    return "low_reliability" if low else "high_reliability"


def fit_empirical_cp_ngc_calibration(
    records: list[dict],
    *,
    correct_trigger_rate: float = 0.10,
    correct_red_rate: float = 0.02,
    minimum_stratum_size: int = 20,
) -> dict:
    """Fit C3 thresholds using validation records only.

    Each record must contain ``statistic``, ``correct_basin`` and ``stratum``.
    Sparse strata fall back to the pooled correct-basin distribution and are
    explicitly marked; no wrong-basin label is used to set thresholds.
    """
    trigger = float(correct_trigger_rate)
    red = float(correct_red_rate)
    if not 0.0 < red < trigger < 1.0:
        raise ValueError("require 0 < correct_red_rate < correct_trigger_rate < 1")
    correct = [
        record
        for record in records
        if bool(record["correct_basin"]) and np.isfinite(float(record["statistic"]))
    ]
    if not correct:
        raise ValueError("calibration requires correct-basin validation records")
    pooled = np.asarray([float(record["statistic"]) for record in correct])
    strata = sorted({str(record["stratum"]) for record in records})
    calibrated = {}
    for stratum in strata:
        values = np.asarray(
            [
                float(record["statistic"])
                for record in correct
                if str(record["stratum"]) == stratum
            ],
            dtype=float,
        )
        fallback = values.size < int(minimum_stratum_size)
        reference = pooled if fallback else values
        calibrated[stratum] = {
            "green_max": float(np.quantile(reference, 1.0 - trigger, method="higher")),
            "red_min": float(np.quantile(reference, 1.0 - red, method="higher")),
            "n_correct_stratum": int(values.size),
            "n_reference": int(reference.size),
            "pooled_fallback": bool(fallback),
        }
    return {
        "source": "C3_validation_correct_basin_empirical_quantiles",
        "correct_trigger_rate_target": trigger,
        "correct_red_rate_target": red,
        "minimum_stratum_size": int(minimum_stratum_size),
        "strata": calibrated,
    }


def apply_empirical_cp_ngc_calibration(
    statistic: float, stratum: str, calibration: dict
) -> str:
    thresholds = calibration["strata"][str(stratum)]
    value = float(statistic)
    if value <= float(thresholds["green_max"]):
        return "green"
    if value >= float(thresholds["red_min"]):
        return "red"
    return "gray"


def ris_training_fold(data: dict, config: dict, indices: np.ndarray) -> tuple[dict, dict]:
    """Return one independent training-block fold while preserving bandwidth."""
    selected = np.asarray(indices, dtype=int).reshape(-1)
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= int(data["scene"]["T"])):
        raise ValueError("invalid RIS training fold indices")
    if np.unique(selected).size != selected.size:
        raise ValueError("RIS training fold indices must be unique")
    scene = copy.deepcopy(data["scene"])
    scene["T"] = int(selected.size)
    scene["Omega"] = np.asarray(scene["Omega"])[:, selected, :].copy()
    fold_config = copy.deepcopy(config)
    fold_config["T"] = int(selected.size)
    y_noisy = np.asarray(data["Y_noisy"])[:, :, selected].copy()
    y_true = np.asarray(data["Y_true"])[:, :, selected].copy()
    return {
        "scene": scene,
        "Y_noisy": y_noisy,
        "Y_true": y_true,
        "Z_noisy": hankelize_frequency(y_noisy, int(scene["P"])),
        "Z_true": hankelize_frequency(y_true, int(scene["P"])),
        "noise_variance": float(data["noise_variance"]),
        "training_indices": selected,
    }, fold_config


def profiled_position_covariance(
    y_raw: np.ndarray,
    stage1_estimate: dict,
    candidate: dict,
    scene: dict,
    config: dict,
    noise_variance: float,
    *,
    step_m: float = 2.0e-4,
    eigenvalue_floor_relative: float = 1.0e-7,
) -> dict:
    """Local Laplace covariance of a fold-A profiled position candidate."""
    profiler = CommonClockJonesProfiler(y_raw, stage1_estimate, scene, config)
    position = np.asarray(candidate["p_u"], dtype=float).reshape(3)
    step = float(step_m)
    hessian = np.empty((3, 3), dtype=float)
    for dim in range(3):
        direction = np.zeros(3, dtype=float)
        direction[dim] = step
        plus = profiler.profile_clock(position + direction)
        minus = profiler.profile_clock(position - direction)
        hessian[:, dim] = (
            np.asarray(plus["gradient_p"]) - np.asarray(minus["gradient_p"])
        ) / (2.0 * step)
    hessian = 0.5 * (hessian + hessian.T)
    information = hessian * (np.asarray(y_raw).size / float(noise_variance))
    eigvals, eigvecs = np.linalg.eigh(information)
    spectral_scale = max(float(np.max(np.abs(eigvals))), 1.0e-8)
    floor = max(
        spectral_scale * float(eigenvalue_floor_relative), 1.0e-12
    )
    floored = np.maximum(eigvals, floor)
    covariance = eigvecs @ np.diag(1.0 / floored) @ eigvecs.T
    return {
        "covariance_p": 0.5 * (covariance + covariance.T),
        "source": "fold_A_profile_objective_local_laplace",
        "information_eigenvalues": eigvals,
        "information_rank": int(np.sum(eigvals > floor)),
        "information_floor": float(floor),
        "regularized": bool(np.any(eigvals <= floor)),
        "valid_local_minimum": bool(np.all(eigvals > floor)),
    }


def one_way_heldout_cp_ngc(
    data: dict,
    config: dict,
    *,
    fold_a_indices: np.ndarray | None = None,
    covariance_regularization: dict | None = None,
) -> dict:
    """C4: A-fit/B-test held-out CP-NGC with candidate uncertainty."""
    t_dim = int(data["scene"]["T"])
    if fold_a_indices is None:
        fold_a_indices = np.arange(0, t_dim, 2, dtype=int)
    fold_a_indices = np.asarray(fold_a_indices, dtype=int)
    fold_b_indices = np.setdiff1d(np.arange(t_dim, dtype=int), fold_a_indices)
    fold_a, config_a = ris_training_fold(data, config, fold_a_indices)
    fold_b, config_b = ris_training_fold(data, config, fold_b_indices)
    with contextlib.redirect_stdout(io.StringIO()):
        stage1_a = initialize_from_hankel(fold_a["Z_noisy"], fold_a["scene"], config_a)
        candidate_a = refine_ccop_jvp(
            fold_a["Y_noisy"], stage1_a, fold_a["scene"], config_a, incumbent=None
        )
        stage1_b = initialize_from_hankel(fold_b["Z_noisy"], fold_b["scene"], config_b)
    covariance_b = linearized_stage1_covariance(
        fold_b["Y_noisy"],
        stage1_b,
        fold_b["scene"],
        fold_b["noise_variance"],
    )
    covariance_p = profiled_position_covariance(
        fold_a["Y_noisy"],
        stage1_a,
        candidate_a,
        fold_a["scene"],
        config_a,
        fold_a["noise_variance"],
    )
    statistic = cp_ngc_statistic(
        cp_ngc_stage1_vector(stage1_b, fold_b["scene"]),
        candidate_a["p_u"],
        covariance_b["covariance_z"],
        fold_b["scene"],
        covariance_p=covariance_p["covariance_p"],
        covariance_regularization=covariance_regularization,
    )
    return {
        "statistic": statistic,
        "candidate_a": candidate_a,
        "stage1_a": stage1_a,
        "stage1_b": stage1_b,
        "covariance_b": covariance_b,
        "covariance_p_a": covariance_p,
        "fold_a_indices": fold_a_indices,
        "fold_b_indices": fold_b_indices,
        "independent_training_blocks": True,
        "full_bandwidth_each_fold": True,
        "direction": "A_fit_B_test",
        "heldout_certificate_valid": bool(
            covariance_p["valid_local_minimum"]
            and statistic["full_3d_certificate"]
        ),
    }
