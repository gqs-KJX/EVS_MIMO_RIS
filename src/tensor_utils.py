"""Tensor utilities for raw and Hankelized OFDM observations."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np


def _dense_khatri_rao_forced() -> bool:
    """Return True when the legacy dense Khatri-Rao path is requested.

    Set ``EVS_VP_DENSE_LSTSQ=1`` to reproduce the pre-factorization numerics
    exactly; the factorized path below is algebraically identical and agrees to
    ~1e-14 relative, but the switch keeps the old code reachable for audits.
    """
    return os.environ.get("EVS_VP_DENSE_LSTSQ", "").strip() not in ("", "0", "false", "False")


def khatri_rao_gram(factors: Sequence[np.ndarray]) -> np.ndarray:
    """Gram matrix ``D^H D`` of a design whose columns are rank-one Kronecker terms.

    ``factors[m]`` has shape ``(dim_m, K)``; column ``k`` of the implied design
    ``D`` is ``factors[0][:, k] (x) ... (x) factors[M-1][:, k]``.  Because the
    Kronecker product factorizes the inner product,
    ``(D^H D)[k, l] = prod_m (factors[m][:, k]^H factors[m][:, l])``, so the
    ``(prod_m dim_m) x K`` design never has to be materialized.
    """
    gram = np.ones((factors[0].shape[1], factors[0].shape[1]), dtype=complex)
    for factor in factors:
        matrix = np.asarray(factor, dtype=complex)
        gram = gram * (matrix.conj().T @ matrix)
    return gram


def khatri_rao_correlate(factors: Sequence[np.ndarray], tensor: np.ndarray) -> np.ndarray:
    """Correlation ``D^H vec(tensor)`` computed by successive mode contractions.

    ``tensor`` has shape ``(dim_0, ..., dim_{M-1})`` matching ``factors``; the
    contraction costs ``O(K * prod_m dim_m)`` instead of building ``D``.
    """
    work = np.asarray(tensor, dtype=complex)
    # Contract mode 0 against every column at once, keeping the path index last.
    work = np.tensordot(np.asarray(factors[0], dtype=complex).conj(), work, axes=([0], [0]))
    work = np.moveaxis(work, 0, -1)
    for factor in factors[1:]:
        matrix = np.asarray(factor, dtype=complex).conj()
        work = np.einsum("i...k,ik->...k", work, matrix, optimize=True)
    return np.asarray(work, dtype=complex).reshape(-1)


def khatri_rao_synthesize(
    factors: Sequence[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    """Reconstruct ``sum_k weights[k] * (x)_m factors[m][:, k]`` as a dense tensor."""
    matrices = [np.asarray(factor, dtype=complex) for factor in factors]
    scaled = matrices[0] * np.asarray(weights, dtype=complex)[None, :]
    subscripts = ",".join(
        f"{chr(ord('a') + index)}z" for index in range(len(matrices))
    )
    output = "".join(chr(ord("a") + index) for index in range(len(matrices)))
    return np.einsum(
        f"{subscripts}->{output}", scaled, *matrices[1:], optimize=True
    )


def solve_khatri_rao_lstsq(
    factors: Sequence[np.ndarray],
    tensor: np.ndarray,
    reg: float = 0.0,
    cond_limit: float = 1.0e12,
) -> np.ndarray:
    """Ridge least squares against a Khatri-Rao design without materializing it.

    Solves ``min_b ||D b - vec(tensor)||^2 + reg ||b||^2`` via the normal
    equations ``(D^H D + reg I) b = D^H vec(tensor)``.  For ``reg > 0`` the
    augmented design used by :func:`src.utils.solve_lstsq` always has full column
    rank, so this is the same unique minimizer, not an approximation.  The
    ``K x K`` system is solved by Cholesky and falls back to a least-squares
    solve of the Gram when it is ill conditioned.
    """
    matrices = [np.asarray(factor, dtype=complex) for factor in factors]
    gram = khatri_rao_gram(matrices)
    rhs = khatri_rao_correlate(matrices, tensor)
    if reg > 0.0:
        gram = gram + float(reg) * np.eye(gram.shape[0], dtype=complex)
    if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(rhs)):
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]
    # Hermitian by construction; symmetrize against round-off before Cholesky.
    gram = 0.5 * (gram + gram.conj().T)
    try:
        if np.linalg.cond(gram) > float(cond_limit):
            raise np.linalg.LinAlgError("ill-conditioned Khatri-Rao Gram")
        factorization = np.linalg.cholesky(gram)
        temporary = np.linalg.solve(factorization, rhs)
        return np.linalg.solve(factorization.conj().T, temporary)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def blocked_squared_error(left: np.ndarray, right: np.ndarray, block: int = 1 << 20) -> float:
    """Return ``||left - right||_F^2`` without materializing the difference.

    ``np.linalg.norm(left - right)`` allocates a full-size temporary, which for
    the Hankel tensor of the paper configuration is 403 MB per call.  The
    blocked form keeps the same explicit difference -- no Pythagorean identity,
    so no cancellation is introduced -- inside a 16 MB working buffer.  Block
    partial sums change the summation order relative to a single reduction, and
    agree with it to ~1e-14 relative.
    """
    left_flat = np.asarray(left).reshape(-1)
    right_flat = np.asarray(right).reshape(-1)
    assert left_flat.size == right_flat.size, "inputs must have the same size"
    total = 0.0
    for start in range(0, left_flat.size, block):
        stop = start + block
        difference = left_flat[start:stop] - right_flat[start:stop]
        total += float(np.vdot(difference, difference).real)
    return total


def hankelize_frequency(y: np.ndarray, p_dim: int) -> np.ndarray:
    """Construct Z[i, p, l, t] = Y[i, p + l, t].

    Input Y has shape I x N x T. Output Z has shape I x P x L x T,
    where L = N - P + 1.
    """
    assert y.ndim == 3, "Y must have shape I x N x T"
    i_dim, n_dim, t_dim = y.shape
    assert 1 <= p_dim <= n_dim, "P must satisfy 1 <= P <= N"
    # sliding_window_view over the frequency axis yields W[i, ell, t, p] = Y[i, p + ell, t]
    # as a zero-copy view; transpose to I x P x L x T and copy once (writable, C-contiguous).
    windows = np.lib.stride_tricks.sliding_window_view(y, p_dim, axis=1)
    return np.transpose(windows, (0, 3, 1, 2)).copy()


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
    if _dense_khatri_rao_forced():
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
    return khatri_rao_synthesize((a_mat, b_mat, q_mat, c_mat), beta)
