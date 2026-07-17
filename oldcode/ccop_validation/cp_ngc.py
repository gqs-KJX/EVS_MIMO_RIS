"""Covariance-projected normalized geometry consistency diagnostics.

The statistic combines common-clock delays and exact UE-to-RIS local geometry
in one covariance-weighted vector.  It is an experimental certification module
and does not replace the frozen NGC selector in ``main_single_proposed``.
"""

from __future__ import annotations

import numpy as np

from src.geometry import elev_az_from_unit_vector
from src.projections_delay import tau_from_pole


def cp_ngc_geometry(p_u: np.ndarray, scene: dict) -> np.ndarray:
    """Return h0(p)=[geometric delays, flattened RIS local geometry]."""
    position = np.asarray(p_u, dtype=float).reshape(3)
    k_paths = int(scene["K"])
    delays = np.empty(k_paths, dtype=float)
    ris_eta = np.empty((k_paths, 3), dtype=float)
    eps = 1.0e-15
    for path in range(k_paths):
        difference_global = position - np.asarray(
            scene["ris_centers"][path], dtype=float
        )
        difference_local = np.asarray(scene["rotations"][path], dtype=float) @ difference_global
        range_m = float(np.linalg.norm(difference_local))
        safe_range = max(range_m, eps)
        elevation, azimuth = elev_az_from_unit_vector(
            difference_local / safe_range
        )
        delays[path] = (
            range_m + float(scene["d_RB"][path])
        ) / float(scene["c0"])
        ris_eta[path] = (range_m, elevation, azimuth)
    return np.concatenate([delays, ris_eta.reshape(-1)])


def cp_ngc_clock_vector(scene: dict) -> np.ndarray:
    """Return a=[1_K,0] for the common-clock nuisance direction."""
    k_paths = int(scene["K"])
    vector = np.zeros(4 * k_paths, dtype=float)
    vector[:k_paths] = 1.0
    return vector


def cp_ngc_stage1_vector(stage1_estimate: dict, scene: dict) -> np.ndarray:
    """Return Stage-I [delay, RIS geometry] in physical panel order."""
    k_paths = int(scene["K"])
    poles = np.asarray(stage1_estimate["poles"], dtype=complex).reshape(k_paths)
    ris_eta = np.asarray(stage1_estimate["ris_eta"], dtype=float).reshape(k_paths, 3)
    tau = np.asarray(
        [tau_from_pole(pole, scene["delta_f"]) for pole in poles], dtype=float
    )
    if not bool(stage1_estimate.get("columns_are_panel_ordered", False)):
        panel_to_column = stage1_estimate.get("panel_to_column_assignment")
        if panel_to_column is not None:
            order = np.asarray(panel_to_column, dtype=int).reshape(k_paths)
            tau = tau[order]
            ris_eta = ris_eta[order]
    return np.concatenate([tau, ris_eta.reshape(-1)])


def _wrap_azimuth_residuals(residual: np.ndarray, scene: dict) -> np.ndarray:
    result = np.asarray(residual, dtype=float).copy()
    k_paths = int(scene["K"])
    offset = k_paths
    for path in range(k_paths):
        azimuth_index = offset + 3 * path + 2
        result[azimuth_index] = float(
            np.angle(np.exp(1j * result[azimuth_index]))
        )
    return result


def cp_ngc_geometry_jacobian(
    p_u: np.ndarray,
    scene: dict,
    *,
    step_m: float = 1.0e-5,
) -> np.ndarray:
    """Return a central-difference Jacobian of h0(p), with wrapped azimuths."""
    position = np.asarray(p_u, dtype=float).reshape(3)
    step = float(step_m)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_m must be positive")
    columns = []
    for dim in range(3):
        direction = np.zeros(3, dtype=float)
        direction[dim] = step
        plus = cp_ngc_geometry(position + direction, scene)
        minus = cp_ngc_geometry(position - direction, scene)
        difference = _wrap_azimuth_residuals(plus - minus, scene)
        columns.append(difference / (2.0 * step))
    return np.column_stack(columns)


def _whiten_covariance(covariance: np.ndarray) -> tuple[np.ndarray, str]:
    """Return L such that C=L L^T, preserving mixed physical units."""
    cov = np.asarray(covariance, dtype=float)
    cov = 0.5 * (cov + cov.T)
    try:
        return np.linalg.cholesky(cov), "cholesky"
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals.size == 0 or float(np.max(eigvals)) <= 0.0:
            raise np.linalg.LinAlgError("CP-NGC covariance is not positive semidefinite")
        tolerance = np.finfo(float).eps * max(cov.shape) * float(np.max(eigvals))
        if np.any(eigvals <= tolerance):
            raise np.linalg.LinAlgError(
                "CP-NGC chi-square calibration requires a positive-definite covariance"
            )
        return eigvecs @ np.diag(np.sqrt(eigvals)), "eigendecomposition"


def regularize_cp_ngc_covariance(
    covariance: np.ndarray,
    *,
    shrinkage: float = 0.0,
    eigenvalue_floor_relative: float = 1.0e-8,
    scale_floor_relative: float = 1.0e-12,
) -> tuple[np.ndarray, dict]:
    """Regularize a mixed-unit covariance in correlation coordinates.

    Delay, range, and angle components have incompatible physical units.  The
    shrinkage and eigenvalue floor are therefore applied to the dimensionless
    correlation matrix and then mapped back to the original units.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite values")
    alpha = float(shrinkage)
    floor = float(eigenvalue_floor_relative)
    scale_floor = float(scale_floor_relative)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("shrinkage must lie in [0,1]")
    if floor <= 0.0 or scale_floor <= 0.0:
        raise ValueError("covariance floors must be positive")
    cov = 0.5 * (cov + cov.T)
    diagonal = np.diag(cov).copy()
    positive = diagonal[np.isfinite(diagonal) & (diagonal > 0.0)]
    if positive.size == 0:
        raise np.linalg.LinAlgError("covariance has no positive marginal variance")
    reference_variance = float(np.median(positive))
    minimum_variance = max(
        np.finfo(float).tiny,
        scale_floor * reference_variance,
    )
    repaired_diagonal = np.maximum(diagonal, minimum_variance)
    scales = np.sqrt(repaired_diagonal)
    correlation = cov / np.outer(scales, scales)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    correlation_shrunk = (1.0 - alpha) * correlation + alpha * np.eye(cov.shape[0])
    eigvals, eigvecs = np.linalg.eigh(
        0.5 * (correlation_shrunk + correlation_shrunk.T)
    )
    max_eigenvalue = max(float(np.max(eigvals)), 1.0)
    eigenvalue_floor = floor * max_eigenvalue
    floored = np.maximum(eigvals, eigenvalue_floor)
    regularized_correlation = eigvecs @ np.diag(floored) @ eigvecs.T
    regularized_correlation = 0.5 * (
        regularized_correlation + regularized_correlation.T
    )
    regularized = (
        scales[:, None] * regularized_correlation * scales[None, :]
    )
    regularized = 0.5 * (regularized + regularized.T)
    diagnostics = {
        "shrinkage": alpha,
        "eigenvalue_floor_relative": floor,
        "scale_floor_relative": scale_floor,
        "repaired_marginal_count": int(np.sum(diagonal < minimum_variance)),
        "correlation_min_eigenvalue_before": float(np.min(eigvals)),
        "correlation_min_eigenvalue_after": float(np.min(floored)),
        "correlation_condition_number_before": float(
            np.linalg.cond(correlation_shrunk)
        ),
        "correlation_condition_number_after": float(
            np.max(floored) / np.min(floored)
        ),
        "positive_definite": True,
    }
    np.linalg.cholesky(regularized)
    return regularized, diagnostics


def cp_ngc_statistic(
    z_hat: np.ndarray,
    p_candidate: np.ndarray,
    covariance_z: np.ndarray,
    scene: dict,
    *,
    covariance_p: np.ndarray | None = None,
    covariance_regularization: dict | None = None,
    jacobian_step_m: float = 1.0e-5,
    rank_relative_tolerance: float = 1.0e-8,
) -> dict:
    """Evaluate the clock-nulled CP-NGC statistic at one fixed candidate.

    If ``covariance_p`` is supplied, the held-out effective covariance
    ``C_z + J_h C_p J_h^T`` is used.  This is the one-way cross-fit correction;
    it does not assert that two fitted-fold statistics may be averaged as a
    chi-square random variable.
    """
    geometry = cp_ngc_geometry(p_candidate, scene)
    observation = np.asarray(z_hat, dtype=float).reshape(-1)
    if observation.shape != geometry.shape:
        raise ValueError(
            f"z_hat has length {observation.size}; expected {geometry.size}"
        )
    covariance = np.asarray(covariance_z, dtype=float)
    if covariance.shape != (geometry.size, geometry.size):
        raise ValueError("covariance_z shape does not match the CP-NGC vector")
    jacobian = cp_ngc_geometry_jacobian(
        p_candidate, scene, step_m=jacobian_step_m
    )
    covariance_source = "C_z"
    if covariance_p is not None:
        cov_p = np.asarray(covariance_p, dtype=float)
        if cov_p.shape != (3, 3):
            raise ValueError("covariance_p must have shape (3, 3)")
        covariance = covariance + jacobian @ cov_p @ jacobian.T
        covariance_source = "C_z_plus_J_Cp_JT"
    covariance = 0.5 * (covariance + covariance.T)
    regularization_diagnostics = None
    if covariance_regularization is not None:
        covariance, regularization_diagnostics = regularize_cp_ngc_covariance(
            covariance, **dict(covariance_regularization)
        )
        covariance_source += "_regularized"

    clock_vector = cp_ngc_clock_vector(scene)
    residual = _wrap_azimuth_residuals(observation - geometry, scene)
    factor, factorization = _whiten_covariance(covariance)
    whitened_residual = np.linalg.solve(factor, residual)
    whitened_clock = np.linalg.solve(factor, clock_vector)
    clock_information = float(np.vdot(whitened_clock, whitened_clock).real)
    if not np.isfinite(clock_information) or clock_information <= 0.0:
        raise np.linalg.LinAlgError("common clock is not observable under covariance_z")
    delta_t_gls = float(
        np.vdot(whitened_clock, whitened_residual).real / clock_information
    )
    projected_whitened = (
        whitened_residual - whitened_clock * delta_t_gls
    )
    statistic = float(
        np.vdot(projected_whitened, projected_whitened).real
    )
    dof = int(observation.size - 1)

    inverse_factor = np.linalg.solve(factor, np.eye(factor.shape[0]))
    whitened_projector = np.eye(observation.size) - np.outer(
        whitened_clock, whitened_clock
    ) / clock_information
    projector = inverse_factor.T @ whitened_projector @ inverse_factor
    projector = 0.5 * (projector + projector.T)
    projected_jacobian = whitened_projector @ np.linalg.solve(factor, jacobian)
    _, singular_values, right_vectors_h = np.linalg.svd(
        projected_jacobian, full_matrices=False
    )
    relative_rank_tolerance = float(rank_relative_tolerance)
    if relative_rank_tolerance <= 0.0:
        raise ValueError("rank_relative_tolerance must be positive")
    rank_tolerance = (
        max(
            max(projected_jacobian.shape) * np.finfo(float).eps,
            relative_rank_tolerance,
        )
        * singular_values[0]
        if singular_values.size
        else 0.0
    )
    geometry_rank = int(np.sum(singular_values > rank_tolerance))
    uncertifiable_directions = (
        right_vectors_h[geometry_rank:].conj().T
        if geometry_rank < right_vectors_h.shape[0]
        else np.empty((3, 0), dtype=float)
    )
    cert_information = projected_jacobian.T @ projected_jacobian
    cert_information = 0.5 * (cert_information + cert_information.T)
    return {
        "statistic": statistic,
        "dof": dof,
        "delta_t_gls": delta_t_gls,
        "residual": residual,
        "projected_whitened_residual": projected_whitened,
        "projector": projector,
        "covariance_effective": covariance,
        "covariance_source": covariance_source,
        "covariance_factorization": factorization,
        "covariance_regularization": regularization_diagnostics,
        "clock_vector": clock_vector,
        "whitened_clock_vector": whitened_clock,
        "whitened_projector": whitened_projector,
        "projector_clock_null_norm": float(np.linalg.norm(projector @ clock_vector)),
        "whitened_projector_idempotence_error": float(
            np.linalg.norm(whitened_projector @ whitened_projector - whitened_projector)
        ),
        "geometry_jacobian": jacobian,
        "projected_geometry_jacobian": projected_jacobian,
        "projected_geometry_rank": geometry_rank,
        "projected_geometry_singular_values": singular_values,
        "projected_geometry_rank_tolerance": float(rank_tolerance),
        "uncertifiable_position_directions": np.asarray(
            uncertifiable_directions, dtype=float
        ),
        "full_3d_certificate": bool(geometry_rank == 3),
        "cert_information": cert_information,
        "cert_information_eigenvalues": np.linalg.eigvalsh(cert_information),
    }
