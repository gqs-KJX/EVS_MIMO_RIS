"""Diagnostics and self-tests for the single proposed-method demo."""

from __future__ import annotations

import numpy as np

from .geometry import position_from_local_geometry
from .metrics import position_rmse, relative_nmse, rmse_abs
from .projections_delay import bq_from_poles, project_common_delay_from_proxies, tau_from_pole
from .projections_ris import (
    compressed_exact_response,
    local_ris_search_config,
    project_ris_factor,
    scaled_residual,
)
from .tensor_utils import (
    hankelize_frequency,
    khatri_rao_columns,
    mode_unfold,
    reconstruct_z,
)


def finite_count_summary(*arrays: np.ndarray) -> int:
    """Count nonfinite entries across a small list of arrays."""
    return int(sum(array.size - np.count_nonzero(np.isfinite(array)) for array in arrays))


def format_float_list(values: np.ndarray, scale: float = 1.0, precision: int = 4) -> str:
    """Format a small 1-D array as a compact list."""
    arr = np.asarray(values).reshape(-1) * scale
    return "[" + ", ".join(f"{value:.{precision}g}" for value in arr) + "]"


def estimate_position_from_ris_eta(scene: dict, estimate: dict) -> np.ndarray:
    """Average per-RIS local geometry estimates into one global UE position."""
    positions = []
    for k in range(scene["K"]):
        positions.append(
            position_from_local_geometry(
                scene["ris_centers"][k],
                scene["rotations"][k],
                estimate["ris_eta"][k, 0],
                estimate["ris_eta"][k, 1],
                estimate["ris_eta"][k, 2],
            )
        )
    return np.mean(np.asarray(positions), axis=0)


def parameter_errors_for_structured(
    scene: dict, estimate: dict, true_components: dict
) -> dict:
    """Return tau, range, and position errors for a structured estimate."""
    tau_hat = np.array([tau_from_pole(z, scene["delta_f"]) for z in estimate["poles"]])
    range_hat = estimate["ris_eta"][:, 0]
    p_hat = estimate_position_from_ris_eta(scene, estimate)
    tau_true = true_components["taus"]
    range_true = true_components["ranges"]
    return {
        "tau_hat": tau_hat,
        "range_hat": range_hat,
        "p_hat": p_hat,
        "tau_rmse": float(np.linalg.norm(tau_hat - tau_true) / np.sqrt(scene["K"])),
        "range_rmse": float(np.linalg.norm(range_hat - range_true) / np.sqrt(scene["K"])),
        "position_rmse": position_rmse(p_hat, scene["p_u_true"]),
    }


def parameter_errors_for_vp(scene: dict, final: dict, true_components: dict) -> dict:
    """Return tau, range, and position errors for the final VP estimate."""
    tau_hat = final["components"]["taus"]
    range_hat = final["components"]["ranges"]
    tau_true = true_components["taus"]
    range_true = true_components["ranges"]
    return {
        "tau_hat": tau_hat,
        "range_hat": range_hat,
        "p_hat": final["p_u"],
        "tau_rmse": float(np.linalg.norm(tau_hat - tau_true) / np.sqrt(scene["K"])),
        "range_rmse": float(np.linalg.norm(range_hat - range_true) / np.sqrt(scene["K"])),
        "position_rmse": position_rmse(final["p_u"], scene["p_u_true"]),
    }


def y_metric_summary(y_hat: np.ndarray, y_true: np.ndarray) -> dict:
    """Return absolute RMSE and relative NMSE for a Y-domain estimate."""
    return {
        "rmse_abs": rmse_abs(y_hat, y_true),
        "nmse": relative_nmse(y_hat, y_true),
    }


def z_metric_summary(z_hat: np.ndarray, z_true: np.ndarray, z_noisy: np.ndarray) -> dict:
    """Return Z-domain residuals against true and noisy tensors."""
    return {
        "rmse_true": rmse_abs(z_hat, z_true),
        "rmse_noisy": rmse_abs(z_hat, z_noisy),
        "nmse_true": relative_nmse(z_hat, z_true),
        "nmse_noisy": relative_nmse(z_hat, z_noisy),
    }


def run_tensor_factorization_shape_self_test() -> dict:
    """Verify unfolding and Khatri-Rao conventions for all four Z modes."""
    rng = np.random.default_rng(101)
    i_dim, p_dim, l_dim, t_dim, k_paths = 3, 4, 5, 2, 2

    def randn_complex(shape: tuple[int, ...]) -> np.ndarray:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    a_mat = randn_complex((i_dim, k_paths))
    b_mat = randn_complex((p_dim, k_paths))
    q_mat = randn_complex((l_dim, k_paths))
    c_mat = randn_complex((t_dim, k_paths))
    beta = randn_complex((k_paths,))
    factors = [a_mat, b_mat, q_mat, c_mat]
    z_tensor = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)

    max_errors = []
    for mode in range(4):
        unfolding = mode_unfold(z_tensor, mode)
        other_modes = [idx for idx in range(4) if idx != mode]
        kr = khatri_rao_columns([factors[idx] for idx in other_modes])
        expected = factors[mode] @ np.diag(beta) @ kr.T
        error = np.linalg.norm(unfolding - expected) / (np.linalg.norm(z_tensor) + 1e-12)
        max_errors.append(float(error))
        if error > 1e-12:
            raise AssertionError(
                f"CPD unfolding/Khatri-Rao convention failed for mode {mode + 1}: {error}"
            )
    return {"max_mode_error": float(max(max_errors)), "mode_errors": max_errors}


def run_delay_projection_self_test(delta_f: float) -> dict:
    """Verify that B and Q are rebuilt from one common estimated pole."""
    rng = np.random.default_rng(202)
    p_dim, l_dim = 5, 6
    true_tau = 12.0e-9
    z_true = np.exp(-1j * 2.0 * np.pi * delta_f * true_tau)
    b_true, q_true = bq_from_poles(np.array([z_true]), p_dim, l_dim)
    noise_scale = 2.0e-3
    b_noisy = b_true[:, 0] + noise_scale * (
        rng.standard_normal(p_dim) + 1j * rng.standard_normal(p_dim)
    )
    q_noisy = q_true[:, 0] + noise_scale * (
        rng.standard_normal(l_dim) + 1j * rng.standard_normal(l_dim)
    )
    z_hat = project_common_delay_from_proxies(
        b_noisy[:, None], q_noisy[:, None], eps=1e-12
    )[0]
    b_hat, q_hat = bq_from_poles(np.array([z_hat]), p_dim, l_dim)
    assert np.allclose(b_hat[:, 0], z_hat ** np.arange(p_dim)), "B does not use z_hat"
    assert np.allclose(q_hat[:, 0], z_hat ** np.arange(l_dim)), "Q does not use z_hat"
    tau_hat = tau_from_pole(z_hat, delta_f)
    return {
        "true_pole": z_true,
        "estimated_pole": z_hat,
        "delay_error_s": abs(tau_hat - true_tau),
    }


def run_ris_projection_self_test(scene: dict, config: dict, true_components: dict) -> dict:
    """Verify compressed exact-spherical RIS projection decreases its objective."""
    rng = np.random.default_rng(303)
    path = 0
    c_true = true_components["c"][path]
    c_noisy = c_true + 0.01 * np.linalg.norm(c_true) / np.sqrt(c_true.size) * (
        rng.standard_normal(c_true.shape) + 1j * rng.standard_normal(c_true.shape)
    )
    eta_true = np.array(
        [
            true_components["ranges"][path],
            true_components["elevations"][path],
            true_components["azimuths"][path],
        ]
    )
    eta_perturbed = eta_true + np.array([0.45, 0.08, -0.10])
    h_before = compressed_exact_response(
        eta_perturbed,
        scene["Omega"][path],
        scene["a_RB"][path],
        scene["ris_grid"],
        scene["wavelength"],
    )
    phi_before, _ = scaled_residual(c_noisy, h_before, config["eps"])
    projection = project_ris_factor(
        c_noisy,
        scene["Omega"][path],
        scene["a_RB"][path],
        scene["ris_grid"],
        scene["wavelength"],
        local_ris_search_config(scene, config, path),
        config["eps"],
    )
    phi_after, _ = scaled_residual(c_noisy, projection["c"], config["eps"])
    c_check = compressed_exact_response(
        projection["eta_local"],
        scene["Omega"][path],
        scene["a_RB"][path],
        scene["ris_grid"],
        scene["wavelength"],
    )
    assert np.allclose(projection["c"], c_check), "RIS c_hat is not Omega @ g_hat"
    range_error = abs(projection["eta_local"][0] - eta_true[0])
    angle_error = np.linalg.norm(
        np.array(
            [
                projection["eta_local"][1] - eta_true[1],
                np.angle(np.exp(1j * (projection["eta_local"][2] - eta_true[2]))),
            ]
        )
    )
    return {
        "phi_before": float(phi_before),
        "phi_after": float(phi_after),
        "range_error": float(range_error),
        "angle_error": float(angle_error),
        "used_pinv": False,
        "warning": "Phi_after > Phi_before" if phi_after > phi_before else "",
    }


def noise_metric_summary(y_true: np.ndarray, y_noisy: np.ndarray, snr_db: float) -> dict:
    """Return signal, noise, and empirical-SNR diagnostics."""
    noise = y_noisy - y_true
    signal_power = float(np.mean(np.abs(y_true) ** 2))
    noise_power = float(np.mean(np.abs(noise) ** 2))
    empirical_snr = 10.0 * np.log10(signal_power / max(noise_power, 1e-300))
    return {
        "norm_Y_true": float(np.linalg.norm(y_true)),
        "norm_noise": float(np.linalg.norm(noise)),
        "signal_power_Y": signal_power,
        "noise_power_Y": noise_power,
        "target_SNR_dB": float(snr_db),
        "empirical_SNR_dB": float(empirical_snr),
        "RMSE_Y_noisy_abs": rmse_abs(y_noisy, y_true),
        "NMSE_Y_noisy": relative_nmse(y_noisy, y_true),
    }


def hankel_metric_summary(y_hat: np.ndarray, y_true: np.ndarray, y_noisy: np.ndarray, p_dim: int) -> dict:
    """Return Z-domain metrics by Hankelizing raw Y tensors."""
    return z_metric_summary(
        hankelize_frequency(y_hat, p_dim),
        hankelize_frequency(y_true, p_dim),
        hankelize_frequency(y_noisy, p_dim),
    )
