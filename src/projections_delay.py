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


def _hankel_1d(vector: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Build a 1-D Hankel matrix from a mother delay vector."""
    assert rows + cols - 1 <= vector.size, "Hankel window is longer than vector"
    hankel = np.empty((rows, cols), dtype=vector.dtype)
    for row in range(rows):
        for col in range(cols):
            hankel[row, col] = vector[row + col]
    return hankel


def _dehankel_1d(hankel: np.ndarray, length: int) -> np.ndarray:
    """Average Hankel anti-diagonals back to a vector."""
    vector = np.zeros(length, dtype=hankel.dtype)
    counts = np.zeros(length, dtype=float)
    rows, cols = hankel.shape
    for row in range(rows):
        for col in range(cols):
            vector[row + col] += hankel[row, col]
            counts[row + col] += 1.0
    return vector / np.maximum(counts, 1.0)


def _project_delay_vector_hankel_rank_one(delay_proxy: np.ndarray, eps: float) -> np.ndarray:
    """Project one mother delay vector through Hankel rank-one lifting."""
    length = delay_proxy.size
    rows = max(2, length // 2)
    cols = length - rows + 1
    lifted = _hankel_1d(delay_proxy, rows, cols)
    u_vec, s_val, vh = np.linalg.svd(lifted, full_matrices=False)
    lifted_rank_one = s_val[0] * np.outer(u_vec[:, 0], vh[0, :])
    projected = _dehankel_1d(lifted_rank_one, length)
    if abs(projected[0]) > eps:
        projected = projected / projected[0]
    return projected


def _best_pole_from_delay_vector(delay_vector: np.ndarray, eps: float) -> complex:
    """Estimate a unit-circle pole from a projected mother delay vector."""
    if delay_vector.size < 2:
        return 1.0 + 0.0j
    numerator = np.vdot(delay_vector[:-1], delay_vector[1:])
    denominator = np.vdot(delay_vector[:-1], delay_vector[:-1]).real
    if denominator <= eps or abs(numerator) <= eps:
        return 1.0 + 0.0j
    pole = numerator / denominator
    return pole / abs(pole)


def project_common_delay_from_proxies(
    b_proxy: np.ndarray, q_proxy: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Project B/Q proxies onto a shared mother-delay Hankel rank-one manifold.

    Inputs have shapes P x K and L x K. The output has shape (K,) and each
    column's B and Q factors must be rebuilt from the same returned pole.
    """
    assert b_proxy.ndim == 2 and q_proxy.ndim == 2, "B and Q proxies must be matrices"
    assert b_proxy.shape[1] == q_proxy.shape[1], "B and Q must have the same K"
    p_dim, k_paths = b_proxy.shape
    l_dim = q_proxy.shape[0]
    r_dim = max(p_dim, l_dim)
    poles = np.empty(k_paths, dtype=complex)

    for k in range(k_paths):
        mother = np.zeros(r_dim, dtype=complex)
        counts = np.zeros(r_dim, dtype=float)
        mother[:p_dim] += b_proxy[:, k]
        counts[:p_dim] += 1.0
        mother[:l_dim] += q_proxy[:, k]
        counts[:l_dim] += 1.0
        mother = mother / np.maximum(counts, 1.0)
        projected = _project_delay_vector_hankel_rank_one(mother, eps)
        poles[k] = _best_pole_from_delay_vector(projected, eps)
    return poles


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
