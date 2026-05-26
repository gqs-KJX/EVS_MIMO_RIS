"""Tensor utilities for raw and Hankelized OFDM observations."""

from __future__ import annotations

import numpy as np


def hankelize_frequency(y: np.ndarray, p_dim: int) -> np.ndarray:
    """Construct Z[i, p, l, t] = Y[i, p + l, t].

    Input Y has shape I x N x T. Output Z has shape I x P x L x T,
    where L = N - P + 1.
    """
    assert y.ndim == 3, "Y must have shape I x N x T"
    i_dim, n_dim, t_dim = y.shape
    assert 1 <= p_dim <= n_dim, "P must satisfy 1 <= P <= N"
    l_dim = n_dim - p_dim + 1
    z = np.empty((i_dim, p_dim, l_dim, t_dim), dtype=y.dtype)
    for p in range(p_dim):
        for ell in range(l_dim):
            z[:, p, ell, :] = y[:, p + ell, :]
    return z


def dehankelize_frequency(z: np.ndarray, n_dim: int) -> np.ndarray:
    """Average anti-diagonals of Z back into an I x N x T raw-like tensor."""
    assert z.ndim == 4, "Z must have shape I x P x L x T"
    i_dim, p_dim, l_dim, t_dim = z.shape
    assert n_dim == p_dim + l_dim - 1, "N must equal P + L - 1"
    y = np.zeros((i_dim, n_dim, t_dim), dtype=z.dtype)
    counts = np.zeros(n_dim, dtype=float)
    for p in range(p_dim):
        for ell in range(l_dim):
            y[:, p + ell, :] += z[:, p, ell, :]
            counts[p + ell] += 1.0
    return y / counts[None, :, None]


def mode_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Unfold a tensor by moving one mode to rows, using NumPy C order."""
    assert 0 <= mode < tensor.ndim, "mode index out of range"
    moved = np.moveaxis(tensor, mode, 0)
    return moved.reshape(tensor.shape[mode], -1)


def khatri_rao_columns(factors: list[np.ndarray]) -> np.ndarray:
    """Columnwise Kronecker product in the same order as explicit tensor sums."""
    assert factors, "at least one factor matrix is required"
    k_paths = factors[0].shape[1]
    for factor in factors:
        assert factor.ndim == 2, "all factors must be matrices"
        assert factor.shape[1] == k_paths, "all factors must have K columns"

    result = factors[0]
    for factor in factors[1:]:
        columns = [
            np.outer(result[:, k], factor[:, k]).reshape(-1) for k in range(k_paths)
        ]
        result = np.column_stack(columns)
    return result


def z_design_column(
    a_col: np.ndarray,
    b_col: np.ndarray,
    q_col: np.ndarray,
    c_col: np.ndarray,
) -> np.ndarray:
    """Vectorize one rank-one term in I x P x L x T order."""
    tensor = (
        a_col[:, None, None, None]
        * b_col[None, :, None, None]
        * q_col[None, None, :, None]
        * c_col[None, None, None, :]
    )
    return tensor.reshape(-1)


def reconstruct_z(
    beta: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    c_mat: np.ndarray,
) -> np.ndarray:
    """Reconstruct a Hankelized tensor from CP factors."""
    i_dim, k_paths = a_mat.shape
    p_dim = b_mat.shape[0]
    l_dim = q_mat.shape[0]
    t_dim = c_mat.shape[0]
    z_hat = np.zeros((i_dim, p_dim, l_dim, t_dim), dtype=complex)
    for k in range(k_paths):
        z_hat += beta[k] * (
            a_mat[:, k, None, None, None]
            * b_mat[None, :, k, None, None]
            * q_mat[None, None, :, k, None]
            * c_mat[None, None, None, :, k]
        )
    return z_hat
