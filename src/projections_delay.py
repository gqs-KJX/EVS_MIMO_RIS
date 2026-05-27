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


def _hankel_rank_one_dehankel(delay_proxy: np.ndarray) -> np.ndarray:
    """Apply H, frontal rank-one projection, and H-dagger anti-diagonal averaging."""
    length = delay_proxy.size
    rows = max(2, length // 2)
    cols = length - rows + 1
    lifted = _hankel_1d(delay_proxy, rows, cols)
    u_vec, s_val, vh = np.linalg.svd(lifted, full_matrices=False)
    lifted_rank_one = s_val[0] * np.outer(u_vec[:, 0], vh[0, :])
    return _dehankel_1d(lifted_rank_one, length)


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


def project_delay_mother_matrix(
    delay_mother: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Apply final unit-circle pole correction to a mother delay factor."""
    assert delay_mother.ndim == 2, "delay mother factor must be a matrix"
    length, k_paths = delay_mother.shape
    poles = np.empty(k_paths, dtype=complex)
    projected = np.empty_like(delay_mother, dtype=complex)
    for k in range(k_paths):
        hankel_projected = _project_delay_vector_hankel_rank_one(delay_mother[:, k], eps)
        poles[k] = _best_pole_from_delay_vector(hankel_projected, eps)
        projected[:, k] = poles[k] ** np.arange(length)
    return projected, poles


def project_delay_mother_hankel_rank_one(delay_mother: np.ndarray) -> np.ndarray:
    """Project each mother-delay column by F, Pi_r1, and F-dagger."""
    assert delay_mother.ndim == 2, "delay mother factor must be a matrix"
    projected = np.empty_like(delay_mother, dtype=complex)
    for k in range(delay_mother.shape[1]):
        projected[:, k] = _hankel_rank_one_dehankel(delay_mother[:, k])
    return projected


def structured_delay_mother_pgd(
    z_tensor: np.ndarray,
    beta: np.ndarray,
    a_mat: np.ndarray,
    c_mat: np.ndarray,
    poles_old: np.ndarray,
    p_dim: int,
    l_dim: int,
    r_dim: int,
    lambda_d: float,
    num_steps: int,
    step_scale: float,
    damping: float,
    eps: float = 1e-12,
) -> dict:
    """Projected-gradient update for the shared mother Vandermonde delay factor.

    This implements the structured LS surrogate

        ||J_P D E_B - Z_(2)||_F^2
      + ||J_L D E_Q - Z_(3)||_F^2
      + lambda_d ||D - D_old||_F^2,

    with fixed E_B/E_Q from the current iterate, followed by the Hankel-domain
    rank-one projection of each mother-delay column.
    """
    assert z_tensor.ndim == 4, "Z must have shape I x P x L x T"
    assert z_tensor.shape[1] == p_dim and z_tensor.shape[2] == l_dim
    k_paths = beta.size
    assert poles_old.shape == (k_paths,)

    d_old = delay_matrix_from_poles(poles_old, r_dim)
    b_ref = d_old[:p_dim]
    q_ref = d_old[:l_dim]

    design_b = np.empty((a_mat.shape[0] * l_dim * c_mat.shape[0], k_paths), dtype=complex)
    for k in range(k_paths):
        design_b[:, k] = (
            beta[k]
            * a_mat[:, k, None, None]
            * q_ref[None, :, k, None]
            * c_mat[None, None, :, k]
        ).reshape(-1)
    target_b = np.moveaxis(z_tensor, 1, 0).reshape(p_dim, -1).T

    design_q = np.empty((a_mat.shape[0] * p_dim * c_mat.shape[0], k_paths), dtype=complex)
    for k in range(k_paths):
        design_q[:, k] = (
            beta[k]
            * a_mat[:, k, None, None]
            * b_ref[None, :, k, None]
            * c_mat[None, None, :, k]
        ).reshape(-1)
    target_q = np.moveaxis(z_tensor, 2, 0).reshape(l_dim, -1).T

    lambda_d = max(float(lambda_d), 0.0)
    num_steps = max(int(num_steps), 1)
    step_scale = max(float(step_scale), eps)
    damping = float(np.clip(damping, eps, 1.0))

    def objective(delay_mother: np.ndarray) -> float:
        residual_b = design_b @ delay_mother[:p_dim].T - target_b
        residual_q = design_q @ delay_mother[:l_dim].T - target_q
        regularizer = lambda_d * np.linalg.norm(delay_mother - d_old) ** 2
        return float(
            np.linalg.norm(residual_b) ** 2
            + np.linalg.norm(residual_q) ** 2
            + regularizer
        )

    spectral_b = np.linalg.norm(design_b, ord=2) ** 2
    spectral_q = np.linalg.norm(design_q, ord=2) ** 2
    step = step_scale / (spectral_b + spectral_q + lambda_d + eps)

    delay_current = d_old.copy()
    best_delay = delay_current.copy()
    best_objective = objective(best_delay)
    initial_objective = best_objective

    for _ in range(num_steps):
        gradient = lambda_d * (delay_current - d_old)

        residual_b = design_b @ delay_current[:p_dim].T - target_b
        gradient[:p_dim] += (design_b.conj().T @ residual_b).T

        residual_q = design_q @ delay_current[:l_dim].T - target_q
        gradient[:l_dim] += (design_q.conj().T @ residual_q).T

        delay_trial = delay_current - step * gradient
        delay_projected = project_delay_mother_hankel_rank_one(delay_trial)
        delay_current = (1.0 - damping) * delay_current + damping * delay_projected
        value = objective(delay_current)
        if value <= best_objective:
            best_delay = delay_current.copy()
            best_objective = value

    final_delay, final_poles = project_delay_mother_matrix(best_delay, eps)
    final_objective = objective(final_delay)

    return {
        "delay_mother": final_delay,
        "poles": final_poles,
        "initial_objective": float(initial_objective),
        "projected_objective": float(best_objective),
        "final_objective": float(final_objective),
        "step": float(step),
        "damping": float(damping),
        "num_steps": int(num_steps),
    }


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
