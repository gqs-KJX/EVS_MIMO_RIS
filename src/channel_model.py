"""Channel generation for the mixed near/far-field RIS-EVS-OFDM model."""

from __future__ import annotations

import numpy as np

from .geometry import (
    far_field_ris_response,
    local_geometry_from_position,
    make_ris_grid,
    maxwell_matrix,
    near_field_spherical_response,
    polarization_vector,
    ula_steering,
)
from .utils import check_finite, complex_awgn


def generate_scene(config: dict, rng: np.random.Generator) -> dict:
    """Generate fixed infrastructure and random training/polarization values."""
    k_paths = config["K"]
    mx, my = config["ris_shape"]
    m_r = mx * my
    wavelength = config["wavelength"]
    ris_grid = make_ris_grid(mx, my, wavelength / 2.0, wavelength / 2.0)
    rotations = np.repeat(np.eye(3)[None, :, :], k_paths, axis=0)

    omega = np.empty((k_paths, config["T"], m_r), dtype=complex)
    for k in range(k_paths):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(config["T"], m_r))
        omega[k] = np.exp(1j * phases) / np.sqrt(m_r)

    p_b = np.asarray(config["p_B"], dtype=float)
    ris_centers = np.asarray(config["ris_centers"], dtype=float)
    a_rb = np.empty((k_paths, m_r), dtype=complex)
    v_b = np.empty((k_paths, config["M_A"]), dtype=complex)
    theta = np.empty((k_paths, 6, 2), dtype=complex)
    d_rb = np.empty(k_paths, dtype=float)

    for k in range(k_paths):
        d_rb[k] = np.linalg.norm(ris_centers[k] - p_b)
        a_rb[k] = far_field_ris_response(
            ris_centers[k], p_b, rotations[k], ris_grid, wavelength
        )
        arrival_direction = (ris_centers[k] - p_b) / d_rb[k]
        propagation_direction = (p_b - ris_centers[k]) / d_rb[k]
        v_b[k] = ula_steering(config["M_A"], wavelength / 2.0, wavelength, arrival_direction)
        theta[k] = maxwell_matrix(propagation_direction)

    gamma_true = rng.uniform(0.35, 1.10, size=k_paths)
    eta_true = rng.uniform(-np.pi, np.pi, size=k_paths)
    beta_true = (
        rng.standard_normal(k_paths) + 1j * rng.standard_normal(k_paths)
    ) / np.sqrt(2.0 * k_paths)

    scene = {
        "K": k_paths,
        "M_A": config["M_A"],
        "I": 6 * config["M_A"],
        "M_Rx": mx,
        "M_Ry": my,
        "M_R": m_r,
        "N": config["N"],
        "P": config["P"],
        "L": config["N"] - config["P"] + 1,
        "T": config["T"],
        "p_B": p_b,
        "p_u_true": np.asarray(config["p_u_true"], dtype=float),
        "ris_centers": ris_centers,
        "rotations": rotations,
        "ris_grid": ris_grid,
        "Omega": omega,
        "a_RB": a_rb,
        "v_B": v_b,
        "Theta": theta,
        "d_RB": d_rb,
        "gamma_true": gamma_true,
        "eta_true": eta_true,
        "beta_true": beta_true,
        "delta_t_true": float(config["delta_t_true"]),
        "delta_f": float(config["delta_f"]),
        "wavelength": wavelength,
        "c0": float(config["c0"]),
    }
    return scene


def channel_components(
    scene: dict,
    p_u: np.ndarray,
    delta_t: float,
    gamma: np.ndarray,
    eta: np.ndarray,
) -> dict:
    """Compute physical factors for all paths.

    Shapes:
      a_EVS: K x I
      d: K x N
      c: K x T
      g: K x M_R
    """
    k_paths = scene["K"]
    a_evs = np.empty((k_paths, scene["I"]), dtype=complex)
    d_delay = np.empty((k_paths, scene["N"]), dtype=complex)
    c_train = np.empty((k_paths, scene["T"]), dtype=complex)
    g_elem = np.empty((k_paths, scene["M_R"]), dtype=complex)
    a_ur = np.empty((k_paths, scene["M_R"]), dtype=complex)
    ranges = np.empty(k_paths, dtype=float)
    elevations = np.empty(k_paths, dtype=float)
    azimuths = np.empty(k_paths, dtype=float)
    taus = np.empty(k_paths, dtype=float)
    poles = np.empty(k_paths, dtype=complex)

    for k in range(k_paths):
        range_m, elev, az, _ = local_geometry_from_position(
            p_u, scene["ris_centers"][k], scene["rotations"][k]
        )
        a_ur[k] = near_field_spherical_response(
            range_m, elev, az, scene["ris_grid"], scene["wavelength"]
        )
        g_elem[k] = scene["a_RB"][k] * a_ur[k]
        c_train[k] = scene["Omega"][k] @ g_elem[k]

        assert g_elem[k].shape == (scene["M_R"],), "len(g_k) must equal M_R"
        assert scene["Omega"][k].shape == (
            scene["T"],
            scene["M_R"],
        ), "Omega_k must have shape T x M_R"
        assert c_train[k].shape == (scene["T"],), "c_k must have shape (T,)"

        pol = scene["Theta"][k] @ polarization_vector(gamma[k], eta[k])
        a_evs[k] = np.kron(scene["v_B"][k], pol)

        tau = (range_m + scene["d_RB"][k]) / scene["c0"] + delta_t
        pole = np.exp(-1j * 2.0 * np.pi * scene["delta_f"] * tau)
        d_delay[k] = pole ** np.arange(scene["N"])

        ranges[k] = range_m
        elevations[k] = elev
        azimuths[k] = az
        taus[k] = tau
        poles[k] = pole

    components = {
        "a_EVS": a_evs,
        "d": d_delay,
        "c": c_train,
        "g": g_elem,
        "a_UR_NF": a_ur,
        "ranges": ranges,
        "elevations": elevations,
        "azimuths": azimuths,
        "taus": taus,
        "poles": poles,
    }
    for name, value in components.items():
        if isinstance(value, np.ndarray):
            check_finite(name, value)
    return components


def synthesize_raw_tensor(components: dict, beta: np.ndarray) -> np.ndarray:
    """Build Y in the raw OFDM domain, shape I x N x T."""
    a_evs = components["a_EVS"]
    d_delay = components["d"]
    c_train = components["c"]
    k_paths, i_dim = a_evs.shape
    assert beta.shape == (k_paths,), "beta must have shape (K,)"
    y = np.zeros((i_dim, d_delay.shape[1], c_train.shape[1]), dtype=complex)
    for k in range(k_paths):
        y += (
            beta[k]
            * a_evs[k, :, None, None]
            * d_delay[k, None, :, None]
            * c_train[k, None, None, :]
        )
    check_finite("Y", y)
    return y


def add_awgn(y_true: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Add AWGN to a raw tensor at the requested SNR in dB."""
    signal_power = float(np.mean(np.abs(y_true) ** 2))
    noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
    y_noisy = y_true + complex_awgn(y_true.shape, noise_variance, rng)
    return y_noisy, noise_variance
