"""Compressed exact-spherical RIS near-field projection."""

from __future__ import annotations

import numpy as np

from .geometry import near_field_spherical_response
from .utils import bounded_coordinate_search, scipy_is_available


def compressed_exact_response(
    eta_local: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """Return h_ex(eta) = Omega @ (a_RB * a_UR^NF_exact(eta))."""
    range_m, elevation, azimuth = eta_local
    a_ur = near_field_spherical_response(range_m, elevation, azimuth, ris_grid, wavelength)
    g_elem = a_rb * a_ur
    return omega @ g_elem


def scaled_residual(c_tilde: np.ndarray, h_model: np.ndarray, eps: float) -> tuple[float, complex]:
    """Return min_alpha ||c_tilde - alpha h_model||^2 and alpha."""
    denom = np.vdot(h_model, h_model) + eps
    alpha = np.vdot(h_model, c_tilde) / denom
    residual = np.linalg.norm(c_tilde - alpha * h_model) ** 2
    return float(residual), alpha


def project_ris_factor(
    c_tilde: np.ndarray,
    omega: np.ndarray,
    a_rb: np.ndarray,
    ris_grid: np.ndarray,
    wavelength: float,
    search_config: dict,
    eps: float = 1e-10,
) -> dict:
    """Fit c_tilde by the compressed exact-spherical near-field model."""
    assert c_tilde.ndim == 1, "c_tilde must be a vector"
    assert omega.shape[0] == c_tilde.size, "Omega rows must match c_tilde length"
    assert omega.shape[1] == a_rb.size, "Omega columns must match RIS response length"

    r_grid = np.linspace(*search_config["range_bounds"], search_config["num_range"])
    e_grid = np.linspace(*search_config["elev_bounds"], search_config["num_elev"])
    a_grid = np.linspace(*search_config["az_bounds"], search_config["num_az"])

    best_eta = None
    best_value = np.inf
    best_alpha = 0.0j
    for range_m in r_grid:
        for elevation in e_grid:
            for azimuth in a_grid:
                eta_local = np.array([range_m, elevation, azimuth])
                h_model = compressed_exact_response(
                    eta_local, omega, a_rb, ris_grid, wavelength
                )
                value, alpha = scaled_residual(c_tilde, h_model, eps)
                if value < best_value:
                    best_value = value
                    best_eta = eta_local
                    best_alpha = alpha

    lower = np.array(
        [
            search_config["range_bounds"][0],
            search_config["elev_bounds"][0],
            search_config["az_bounds"][0],
        ],
        dtype=float,
    )
    upper = np.array(
        [
            search_config["range_bounds"][1],
            search_config["elev_bounds"][1],
            search_config["az_bounds"][1],
        ],
        dtype=float,
    )

    def objective(eta_local: np.ndarray) -> float:
        h_model = compressed_exact_response(eta_local, omega, a_rb, ris_grid, wavelength)
        value, _ = scaled_residual(c_tilde, h_model, eps)
        return value / (np.linalg.norm(c_tilde) ** 2 + eps)

    optimizer_message = "coarse grid only"
    if scipy_is_available():
        from scipy.optimize import minimize

        result = minimize(
            objective,
            best_eta,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": 80, "ftol": 1e-10},
        )
        if result.fun <= objective(best_eta):
            best_eta = np.asarray(result.x, dtype=float)
            optimizer_message = f"scipy L-BFGS-B success={result.success}"
    else:
        x0_scaled = (best_eta - lower) / (upper - lower)

        def scaled_objective(x_scaled: np.ndarray) -> float:
            eta_local = lower + np.clip(x_scaled, 0.0, 1.0) * (upper - lower)
            return objective(eta_local)

        x_best, _, info = bounded_coordinate_search(
            scaled_objective,
            x0_scaled,
            np.zeros(3),
            np.ones(3),
            step0=0.07,
            max_iter=35,
            tol=2e-4,
        )
        best_eta = lower + x_best * (upper - lower)
        optimizer_message = info["message"]

    h_best = compressed_exact_response(best_eta, omega, a_rb, ris_grid, wavelength)
    final_value, best_alpha = scaled_residual(c_tilde, h_best, eps)
    relative_residual = np.sqrt(final_value / (np.linalg.norm(c_tilde) ** 2 + eps))
    return {
        "c": h_best,
        "eta_local": best_eta,
        "alpha": best_alpha,
        "relative_residual": float(relative_residual),
        "optimizer_message": optimizer_message,
    }

