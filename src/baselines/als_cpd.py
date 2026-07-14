"""Standalone complex CP-ALS tensor baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .common import (
    BaselineResult,
    geometric_support_to_position_ls,
    position_grid_from_config,
    simple_atom_normalize,
    training_response_from_position,
)


def _unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Tensor-Toolbox-compatible mode unfolding for a 3-way tensor."""
    if mode == 0:
        return np.asarray(tensor).reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2], order="F")
    if mode == 1:
        return np.transpose(tensor, (1, 0, 2)).reshape(tensor.shape[1], tensor.shape[0] * tensor.shape[2], order="F")
    if mode == 2:
        return np.transpose(tensor, (2, 0, 1)).reshape(tensor.shape[2], tensor.shape[0] * tensor.shape[1], order="F")
    raise ValueError("mode must be 0, 1, or 2")


def _khatri_rao(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[1] != right.shape[1]:
        raise ValueError("Khatri-Rao inputs must have the same column count")
    return np.column_stack([np.kron(left[:, k], right[:, k]) for k in range(left.shape[1])])


def reconstruct_cp_tensor(factors: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    a, b, c = factors
    shape = (a.shape[0], b.shape[0], c.shape[0])
    tensor = np.zeros(shape, dtype=complex)
    for k in range(weights.size):
        tensor += weights[k] * a[:, k, None, None] * b[None, :, k, None] * c[None, None, :, k]
    return tensor


def _svd_init(tensor: np.ndarray, rank: int) -> list[np.ndarray]:
    factors = []
    rng = np.random.default_rng(10)
    for mode in range(3):
        unfold = _unfold(tensor, mode)
        try:
            u, _, _ = np.linalg.svd(unfold, full_matrices=False)
            factor = u[:, :rank]
        except np.linalg.LinAlgError:
            factor = (
                rng.standard_normal((tensor.shape[mode], rank))
                + 1j * rng.standard_normal((tensor.shape[mode], rank))
            )
        if factor.shape[1] < rank:
            pad = (
                rng.standard_normal((tensor.shape[mode], rank - factor.shape[1]))
                + 1j * rng.standard_normal((tensor.shape[mode], rank - factor.shape[1]))
            )
            factor = np.column_stack([factor, pad])
        factors.append(factor.astype(complex, copy=False))
    return factors


def _normalize_columns(factor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.asarray(factor, dtype=complex).copy()
    lambdas = np.ones(normalized.shape[1], dtype=complex)
    for col in range(normalized.shape[1]):
        norm = np.linalg.norm(normalized[:, col])
        if np.isfinite(norm) and norm > 0.0:
            normalized[:, col] /= norm
            lambdas[col] = norm
    return normalized, lambdas


def matlab_compatible_update_factor(
    tensor: np.ndarray,
    pi: np.ndarray,
    mode: int,
    sigma2: float,
) -> np.ndarray:
    """Direct translation of Complex_ALS.m factor update convention."""
    z_fold = _unfold(np.asarray(tensor, dtype=complex), int(mode))
    pi = np.asarray(pi, dtype=complex)
    gram = pi.conj().T @ pi + float(sigma2) * np.eye(pi.shape[1], dtype=complex)
    new_factor_conj = (np.conj(z_fold) @ pi) @ np.linalg.pinv(gram)
    result = np.conj(new_factor_conj)
    del z_fold, gram, new_factor_conj
    return result


def _mttkrp_update_factor(
    tensor: np.ndarray,
    factors: list[np.ndarray],
    mode: int,
    sigma2: float,
) -> np.ndarray:
    """Apply the same regularized ALS update without explicit Khatri-Rao."""
    x_conj = np.conj(np.asarray(tensor, dtype=complex))
    a_factor, b_factor, c_factor = factors
    if mode == 0:
        mttkrp = np.einsum(
            "int,nr,tr->ir", x_conj, b_factor, c_factor, optimize=True
        )
        gram = (c_factor.conj().T @ c_factor) * (
            b_factor.conj().T @ b_factor
        )
    elif mode == 1:
        mttkrp = np.einsum(
            "int,ir,tr->nr", x_conj, a_factor, c_factor, optimize=True
        )
        gram = (c_factor.conj().T @ c_factor) * (
            a_factor.conj().T @ a_factor
        )
    elif mode == 2:
        mttkrp = np.einsum(
            "int,ir,nr->tr", x_conj, a_factor, b_factor, optimize=True
        )
        gram = (b_factor.conj().T @ b_factor) * (
            a_factor.conj().T @ a_factor
        )
    else:
        raise ValueError("mode must be 0, 1, or 2")
    gram = gram + float(sigma2) * np.eye(gram.shape[0], dtype=complex)
    try:
        updated_conj = np.linalg.solve(gram.T, mttkrp.T).T
    except np.linalg.LinAlgError:
        updated_conj = mttkrp @ np.linalg.pinv(gram)
    return np.conj(updated_conj)


def _cp_cross_inner(
    left_factors: list[np.ndarray],
    left_weights: np.ndarray,
    right_factors: list[np.ndarray],
    right_weights: np.ndarray,
) -> complex:
    cross_gram = np.ones(
        (left_weights.size, right_weights.size), dtype=complex
    )
    for left, right in zip(left_factors, right_factors):
        cross_gram *= left.conj().T @ right
    return complex(
        np.einsum(
            "r,rs,s->",
            np.asarray(left_weights, dtype=complex).conj(),
            cross_gram,
            np.asarray(right_weights, dtype=complex),
            optimize=True,
        )
    )


def _cp_norm_sq(factors: list[np.ndarray], weights: np.ndarray) -> float:
    return float(
        max(_cp_cross_inner(factors, weights, factors, weights).real, 0.0)
    )


def _cp_difference_norm_sq(
    previous_factors: list[np.ndarray],
    previous_weights: np.ndarray,
    current_factors: list[np.ndarray],
    current_weights: np.ndarray,
) -> float:
    """Evaluate the small CP difference Gram in extended precision."""
    factors = [
        np.column_stack([previous, current]).astype(np.clongdouble)
        for previous, current in zip(previous_factors, current_factors)
    ]
    weights = np.r_[
        -np.asarray(previous_weights, dtype=np.clongdouble),
        np.asarray(current_weights, dtype=np.clongdouble),
    ]
    rank = weights.size
    gram_product = np.ones((rank, rank), dtype=np.clongdouble)
    for factor in factors:
        gram = np.empty((rank, rank), dtype=np.clongdouble)
        for row in range(rank):
            for column in range(rank):
                gram[row, column] = np.sum(
                    factor[:, row].conj() * factor[:, column],
                    dtype=np.clongdouble,
                )
        gram_product *= gram
    value = np.sum(
        weights.conj()[:, None] * gram_product * weights[None, :],
        dtype=np.clongdouble,
    ).real
    return float(max(value, np.longdouble(0.0)))


def _cp_data_inner(
    tensor: np.ndarray, factors: list[np.ndarray], weights: np.ndarray
) -> complex:
    a_factor, b_factor, c_factor = factors
    contracted = np.einsum(
        "int,nr,tr->ir",
        np.asarray(tensor, dtype=complex),
        b_factor.conj(),
        c_factor.conj(),
        optimize=True,
    )
    atom_inner_data = np.einsum(
        "ir,ir->r", a_factor.conj(), contracted, optimize=True
    )
    return complex(np.dot(np.asarray(weights, dtype=complex), atom_inner_data.conj()))


def _cp_residual_norm_sq(
    tensor: np.ndarray,
    tensor_norm_sq: float,
    factors: list[np.ndarray],
    weights: np.ndarray,
) -> float:
    value = (
        float(tensor_norm_sq)
        + _cp_norm_sq(factors, weights)
        - 2.0 * _cp_data_inner(tensor, factors, weights).real
    )
    return float(max(value, 0.0))


def complex_cp_als(
    tensor: np.ndarray,
    rank: int,
    *,
    max_iter: int = 3000,
    tol: float = 1.0e-8,
    sigma2: float | None = None,
    reg: float | None = None,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """Complex CP-ALS following the supplied Complex_ALS.m update logic."""
    x = np.asarray(tensor, dtype=complex)
    rank = int(rank)
    sigma2_value = float(sigma2 if sigma2 is not None else reg if reg is not None else 1.0e-8)
    a_factor, b_factor, c_factor = _svd_init(x, rank)
    weights = np.ones(rank, dtype=complex)
    factors = [a_factor, b_factor, c_factor]
    tensor_norm = float(np.linalg.norm(x))
    tensor_norm_sq = tensor_norm**2
    previous_factors = list(factors)
    previous_weights = weights.copy()
    converged = False
    residual = float("nan")
    for iteration in range(1, int(max_iter) + 1):
        a_factor, b_factor, c_factor = factors
        a_factor = _mttkrp_update_factor(x, factors, 0, sigma2_value)
        a_factor, _ = _normalize_columns(a_factor)
        factors_after_a = [a_factor, b_factor, c_factor]
        b_factor = _mttkrp_update_factor(
            x, factors_after_a, 1, sigma2_value
        )
        b_factor, _ = _normalize_columns(b_factor)
        factors_after_b = [a_factor, b_factor, c_factor]
        c_factor = _mttkrp_update_factor(
            x, factors_after_b, 2, sigma2_value
        )
        c_factor, weights = _normalize_columns(c_factor)
        factors = [a_factor, b_factor, c_factor]
        current_norm_sq = _cp_norm_sq(factors, weights)
        residual = float(
            np.sqrt(
                _cp_residual_norm_sq(
                    x, tensor_norm_sq, factors, weights
                )
            )
            / (tensor_norm + 1.0e-12)
        )
        difference_norm_sq = _cp_difference_norm_sq(
            previous_factors,
            previous_weights,
            factors,
            weights,
        )
        rel_change = float(
            np.sqrt(difference_norm_sq)
            / (np.sqrt(current_norm_sq) + 1.0e-12)
        )
        if rel_change < float(tol):
            converged = True
            break
        previous_factors = factors
        previous_weights = weights.copy()
    factors[2] = factors[2] @ np.diag(weights)
    weights = np.ones(rank, dtype=complex)
    residual = float(
        np.sqrt(
            _cp_residual_norm_sq(x, tensor_norm_sq, factors, weights)
        )
        / (tensor_norm + 1.0e-12)
    )
    diagnostics = {
        "als_matlab_compatible": True,
        "als_residual": residual,
        "als_iterations": iteration,
        "als_converged": bool(converged),
        "factor_shapes": [tuple(factor.shape) for factor in factors],
        "tensor_shape": tuple(x.shape),
        "rank": rank,
        "sigma2": sigma2_value,
        "als_update_kernel": "mttkrp_small_gram_solve",
        "als_residual_kernel": "cp_gram_identity",
    }
    return factors, weights, diagnostics


def _tau_from_delay_factor(delay_factor: np.ndarray, delta_f: float) -> float:
    values = np.asarray(delay_factor, dtype=complex).reshape(-1)
    if values.size < 2:
        return 0.0
    pole = np.vdot(values[:-1], values[1:]) / (np.vdot(values[:-1], values[:-1]) + 1.0e-12)
    tau = -float(np.angle(pole)) / (2.0 * np.pi * float(delta_f))
    period = 1.0 / float(delta_f)
    return float(np.mod(tau, period))


def _match_training_factor_to_geometry(
    training_factor: np.ndarray,
    scene: dict,
    config: dict,
    tau: float,
) -> dict[str, Any]:
    baseline_cfg = dict(config.get("baselines", {}).get("als_cpd", {}))
    grid_shape = tuple(int(v) for v in baseline_cfg.get("position_grid_shape", (5, 5, 3)))
    candidates = position_grid_from_config(config, grid_shape)
    target = simple_atom_normalize(training_factor)
    best_score = -np.inf
    best_support: dict[str, Any] = {"panel": 0, "tau": float(tau)}
    for panel in range(int(scene["K"])):
        for position in candidates:
            atom = simple_atom_normalize(training_response_from_position(scene, panel, position))
            score = abs(np.vdot(atom, target))
            if score > best_score:
                range_m, elev, az, _ = scene_geometry(scene, panel, position)
                best_score = float(score)
                best_support = {
                    "panel": int(panel),
                    "position": position,
                    "tau": float(tau),
                    "range": float(range_m),
                    "elevation": float(elev),
                    "azimuth": float(az),
                    "score": float(score),
                }
    return best_support


def scene_geometry(scene: dict, panel: int, position: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    from ..geometry import local_geometry_from_position

    return local_geometry_from_position(
        np.asarray(position, dtype=float),
        np.asarray(scene["ris_centers"][panel], dtype=float),
        np.asarray(scene["rotations"][panel], dtype=float),
    )


def _components_from_supports(scene: dict, supports: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    ranges = []
    taus = []
    for support in supports:
        if "range" in support:
            ranges.append(float(support["range"]))
        if "tau" in support:
            taus.append(float(support["tau"]))
    return {
        "ranges": np.asarray(ranges, dtype=float),
        "taus": np.asarray(taus, dtype=float),
    }


def run_als_cpd_baseline(data: dict, config: dict) -> BaselineResult:
    """Run standalone ALS-CPD on the raw observation tensor."""
    start = time.perf_counter()
    scene = data["scene"]
    y_noisy = np.asarray(data["Y_noisy"], dtype=complex)
    baseline_cfg = dict(config.get("baselines", {}).get("als_cpd", {}))
    rank = int(baseline_cfg.get("rank", scene["K"]))
    factors, weights, diagnostics = complex_cp_als(
        y_noisy,
        rank,
        max_iter=int(baseline_cfg.get("max_iter", 3000)),
        tol=float(baseline_cfg.get("tol", 1.0e-8)),
        sigma2=float(baseline_cfg.get("sigma2", baseline_cfg.get("reg", 1.0e-8))),
    )
    y_hat = reconstruct_cp_tensor(factors, weights)
    supports: list[dict[str, Any]] = []
    for comp in range(rank):
        tau = _tau_from_delay_factor(factors[1][:, comp], scene["delta_f"])
        support = _match_training_factor_to_geometry(factors[2][:, comp], scene, config, tau)
        support["component"] = int(comp)
        supports.append(support)
    p_hat, delta_t, geom_diag = geometric_support_to_position_ls(scene, supports, config)
    diagnostics.update(geom_diag)
    diagnostics.update(
        {
            "dictionary_mode": "als_cpd_tensor",
            "grid_size": int(np.prod(config.get("baselines", {}).get("als_cpd", {}).get("position_grid_shape", (5, 5, 3))) * scene["K"]),
        }
    )
    runtime_s = time.perf_counter() - start
    raw_objective = float(np.linalg.norm((y_hat - y_noisy).reshape(-1)) ** 2 / y_noisy.size)
    return BaselineResult(
        name="als_cpd",
        p_u=p_hat,
        delta_t=delta_t,
        Y_hat=y_hat,
        raw_objective_final=raw_objective,
        components=_components_from_supports(scene, supports),
        selected_support=supports,
        runtime_s=runtime_s,
        diagnostics=diagnostics,
    )
