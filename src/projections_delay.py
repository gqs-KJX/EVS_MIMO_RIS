"""Delay-pole estimation and common-pole delay projections."""

from __future__ import annotations

import numpy as np

from .tensor_utils import dehankelize_frequency


def pole_from_tau(tau: float, delta_f: float) -> complex:
    """Return z = exp(-j 2 pi Delta_f tau)."""
    return np.exp(-1j * 2.0 * np.pi * delta_f * tau)


def tau_from_pole(z: complex, delta_f: float) -> float:
    """Map a unit-circle delay pole to the unambiguous delay interval."""
    angle = np.angle(z)
    return float((-angle % (2.0 * np.pi)) / (2.0 * np.pi * delta_f))


def delay_matrix_from_poles(poles: np.ndarray, length: int) -> np.ndarray:
    """Build a Vandermonde delay matrix with shared pole columns."""
    powers = np.arange(length)
    return poles[None, :] ** powers[:, None]


def bq_from_poles(poles: np.ndarray, p_dim: int, l_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Build B and Q that share the same delay pole per path."""
    return delay_matrix_from_poles(poles, p_dim), delay_matrix_from_poles(poles, l_dim)


def estimate_poles_esprit_from_hankel(z_tensor: np.ndarray, k_paths: int) -> np.ndarray:
    """Estimate delay poles by ESPRIT from the Hankelized tensor."""
    assert z_tensor.ndim == 4, "Z must have shape I x P x L x T"
    _, p_dim, l_dim, _ = z_tensor.shape
    n_dim = p_dim + l_dim - 1
    y_like = dehankelize_frequency(z_tensor, n_dim)
    # Frequency x snapshots.
    y_freq = np.transpose(y_like, (1, 0, 2)).reshape(n_dim, -1)
    u, _, _ = np.linalg.svd(y_freq, full_matrices=False)
    signal_subspace = u[:, :k_paths]
    psi = np.linalg.lstsq(signal_subspace[:-1, :], signal_subspace[1:, :], rcond=None)[0]
    poles = np.linalg.eigvals(psi)
    magnitudes = np.maximum(np.abs(poles), 1e-12)
    poles = poles / magnitudes
    return poles


def estimate_common_pole_from_factors(
    b_col: np.ndarray, q_col: np.ndarray, eps: float = 1e-12
) -> complex:
    """Estimate one common unit-circle pole from B and Q proxy columns."""
    numer = 0.0j
    denom = 0.0
    for vec in (b_col, q_col):
        if vec.size >= 2:
            numer += np.vdot(vec[:-1], vec[1:])
            denom += float(np.vdot(vec[:-1], vec[:-1]).real)
    if denom <= eps or abs(numer) <= eps:
        return 1.0 + 0.0j
    pole = numer / denom
    return pole / abs(pole)

