"""Projection onto the EVS Maxwell-Kronecker structure."""

from __future__ import annotations

import numpy as np

from .geometry import polarization_vector


def project_evs_factor(
    a_tilde: np.ndarray,
    v_b: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-10,
) -> dict:
    """Project one EVS factor onto a_EVS = v_B kron Theta e(gamma, eta)."""
    assert a_tilde.ndim == 1, "a_tilde must be a vector"
    m_a = v_b.size
    assert a_tilde.size == 6 * m_a, "EVS factor length must be 6 * M_A"
    assert theta.shape == (6, 2), "theta must be 6 x 2"

    a_matrix = a_tilde.reshape(m_a, 6).T
    p_proxy = a_matrix @ np.conj(v_b) / (np.vdot(v_b, v_b).real + eps)
    normal = theta.conj().T @ theta + eps * np.eye(2)
    e_hat = np.linalg.solve(normal, theta.conj().T @ p_proxy)

    if abs(e_hat[1]) > eps:
        e_hat = e_hat * np.exp(-1j * np.angle(e_hat[1]))
    elif abs(e_hat[0]) > eps:
        e_hat = e_hat * np.exp(-1j * np.angle(e_hat[0]))

    e_norm = np.linalg.norm(e_hat)
    if e_norm <= eps:
        gamma = np.pi / 4.0
        eta = 0.0
    else:
        e_hat = e_hat / e_norm
        gamma = float(np.arctan2(abs(e_hat[0]), abs(e_hat[1])))
        eta = float(np.angle(e_hat[0]))

    p_projected = theta @ polarization_vector(gamma, eta)
    a_model = np.kron(v_b, p_projected)
    a_norm = np.linalg.norm(a_model)
    if a_norm > eps:
        a_model = a_model / a_norm
    scale = np.vdot(a_model, a_tilde) / (np.vdot(a_model, a_model) + eps)
    residual = np.linalg.norm(a_tilde - scale * a_model) / (
        np.linalg.norm(a_tilde) + eps
    )

    return {
        "a": a_model,
        "gamma": gamma,
        "eta": eta,
        "scale": scale,
        "residual": float(residual),
    }
