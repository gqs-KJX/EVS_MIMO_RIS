"""VBI/SBL adaptation of Li et al. (IEEE TWC 2024) for the EVS-RIS tensor.

The cited method is a mean-field variational Bayesian (equivalently sparse
Bayesian learning, SBL) estimator: the RIS-path angle is a sparse support over
an angular dictionary with a Gamma-Gaussian ARD prior, the delays are free
complex-Gaussian phase vectors, and the user location is recovered from the
converged angle support and delays by a geometric constraint.

The original paper is SISO with one RIS and a far/near-field *angle* grid.  This
repository observes an EVS x subcarrier x training tensor with K near-field RIS
panels and a 2-D Jones gain per path, under a common clock offset.  Following
the per-link block structure of the paper (direct link + reflected link), this
adaptation runs one VBI/SBL block per physical RIS panel:

1. project the residual onto the panel's known EVS/Jones subspace ``B_k``;
2. run mean-field VBI over a per-panel near-field UE-position dictionary
   (ARD-driven sparse spatial support), a free-Gaussian delay-phase factor, and
   a 2-D Jones gain, with closed-form updates iterated to convergence;
3. extract the panel's UE position from the variational support posterior and
   its posterior delay factor;

then the K panels are fused into a common position and clock by a weighted
geometric solve, and the channel is reconstructed with the shared forward model.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..geometry import local_geometry_from_position
from ..utils import scipy_is_available
from .common import (
    BaselineResult,
    baseline_refinement_tier,
    build_jones_basis_evs_atoms,
    delay_grid_from_scene,
    delay_response,
    expand_jones_group,
    supports_from_position_clock,
    vectorize_raw_observation,
)
from .factorized_scoring import factorized_fit_supports
from .nf_ris_groupomp_localgrid_wls import _nf_position_grid, _nf_training_matrix


def _panel_evs_basis(scene: dict, config: dict, panel: int) -> np.ndarray:
    atoms, _ = build_jones_basis_evs_atoms(scene, config, panel_index=int(panel))
    columns = [np.asarray(atom, dtype=complex).reshape(-1) for atom in atoms[:2]]
    return np.column_stack(columns)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _rank1_init(reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-1 initialization of the (Jones x delay x training) reduced tensor."""
    jones_dim, n_sub, n_train = reduced.shape
    unfold_t = reduced.reshape(jones_dim * n_sub, n_train)
    u, singular_values, vh = np.linalg.svd(unfold_t, full_matrices=False)
    plane = (u[:, 0]).reshape(jones_dim, n_sub)
    up, plane_singular_values, vhp = np.linalg.svd(
        plane, full_matrices=False
    )
    jones = _unit(up[:, 0])
    delay = _unit(vhp[0].conj())
    amplitude = float(singular_values[0] * plane_singular_values[0])
    training = amplitude * _unit(vh[0].conj())
    return jones, delay, training


def _extract_delay(
    scene: dict, delay_factor: np.ndarray, taus: np.ndarray
) -> float:
    target = _unit(delay_factor)
    scores = np.asarray(
        [abs(np.vdot(_unit(delay_response(scene, float(tau))), target)) ** 2 for tau in taus],
        dtype=float,
    )
    best = int(np.argmax(scores))
    if 0 < best < len(taus) - 1:
        y0, y1, y2 = scores[best - 1], scores[best], scores[best + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1.0e-18:
            shift = 0.5 * (y0 - y2) / denom
            return float(taus[best] + shift * (taus[1] - taus[0]))
    return float(taus[best])


def _digamma(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    try:
        from scipy.special import digamma  # type: ignore[import-not-found]

        return np.asarray(digamma(array), dtype=float)
    except ImportError:
        safe = np.maximum(array, 1.0e-12)
        return np.log(safe) - 0.5 / safe


def _gammaln(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    try:
        from scipy.special import gammaln  # type: ignore[import-not-found]

        return np.asarray(gammaln(array), dtype=float)
    except ImportError:
        import math

        return np.vectorize(math.lgamma, otypes=[float])(array)


def _hermitian_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert one positive Hermitian matrix without discarding covariance."""
    hermitian = 0.5 * (np.asarray(matrix, dtype=complex) + np.asarray(matrix, dtype=complex).conj().T)
    try:
        chol = np.linalg.cholesky(hermitian)
        identity = np.eye(hermitian.shape[0], dtype=complex)
        inverse = np.linalg.solve(chol.conj().T, np.linalg.solve(chol, identity))
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(hermitian, rcond=1.0e-12)
    return 0.5 * (inverse + inverse.conj().T)


def _complex_gaussian_entropy(covariance: np.ndarray) -> float:
    covariance = np.asarray(covariance, dtype=complex)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign.real <= 0.0 or not np.isfinite(logdet):
        return float("-inf")
    dimension = covariance.shape[0]
    return float(dimension * (1.0 + np.log(np.pi)) + logdet)


def _gamma_elbo_terms(
    shape: np.ndarray,
    rate: np.ndarray,
    prior_shape: float,
    prior_rate: float,
) -> float:
    shape = np.asarray(shape, dtype=float)
    rate = np.asarray(rate, dtype=float)
    expected = shape / rate
    expected_log = _digamma(shape) - np.log(rate)
    expected_log_prior = (
        prior_shape * np.log(prior_rate)
        - _gammaln(prior_shape)
        + (prior_shape - 1.0) * expected_log
        - prior_rate * expected
    )
    entropy = (
        shape
        - np.log(rate)
        + _gammaln(shape)
        + (1.0 - shape) * _digamma(shape)
    )
    return float(np.sum(expected_log_prior + entropy))


def _panel_elbo(
    reduced: np.ndarray,
    jones_mean: np.ndarray,
    jones_cov: np.ndarray,
    delay_mean: np.ndarray,
    delay_cov: np.ndarray,
    spatial_mean: np.ndarray,
    spatial_second_energy: float,
    coeff_mean: np.ndarray,
    coeff_cov: np.ndarray,
    indicator_prob: np.ndarray,
    ard_shape: np.ndarray,
    ard_rate: np.ndarray,
    noise_shape: float,
    noise_rate: float,
    *,
    factor_prior_precision: float,
    spike_multiplier: float,
    activity_prior: float,
    ard_prior_shape: float,
    ard_prior_rate: float,
    noise_prior_shape: float,
    noise_prior_rate: float,
) -> float:
    """Mean-field ELBO for the adapted rank-one panel model."""
    e_jones = float(np.vdot(jones_mean, jones_mean).real + np.trace(jones_cov).real)
    e_delay = float(np.vdot(delay_mean, delay_mean).real + np.trace(delay_cov).real)
    model_mean = (
        jones_mean[:, None, None]
        * delay_mean[None, :, None]
        * spatial_mean[None, None, :]
    )
    expected_sse = max(
        float(
            np.vdot(reduced, reduced).real
            - 2.0 * np.vdot(model_mean, reduced).real
            + e_jones * e_delay * spatial_second_energy
        ),
        0.0,
    )
    expected_noise = float(noise_shape / noise_rate)
    expected_log_noise = float(_digamma(noise_shape) - np.log(noise_rate))
    sample_count = int(np.size(reduced))
    likelihood = sample_count * (expected_log_noise - np.log(np.pi)) - expected_noise * expected_sse

    factor_prior = (
        jones_mean.size * np.log(factor_prior_precision / np.pi)
        - factor_prior_precision * e_jones
        + delay_mean.size * np.log(factor_prior_precision / np.pi)
        - factor_prior_precision * e_delay
    )
    factor_entropy = _complex_gaussian_entropy(jones_cov) + _complex_gaussian_entropy(delay_cov)

    coeff_energy = np.abs(coeff_mean) ** 2 + np.clip(
        np.real(np.diag(coeff_cov)), 0.0, None
    )
    expected_ard = ard_shape / ard_rate
    expected_log_ard = _digamma(ard_shape) - np.log(ard_rate)
    multipliers = np.array([spike_multiplier, 1.0], dtype=float)
    component_log_prob = np.log(
        np.array([1.0 - activity_prior, activity_prior], dtype=float)
    )
    coeff_prior = np.sum(
        indicator_prob
        * (
            expected_log_ard[:, None]
            + np.log(multipliers)[None, :]
            - np.log(np.pi)
            - expected_ard[:, None]
            * multipliers[None, :]
            * coeff_energy[:, None]
        )
    )
    indicator_terms = np.sum(
        indicator_prob
        * (
            component_log_prob[None, :]
            - np.log(np.maximum(indicator_prob, 1.0e-300))
        )
    )
    coeff_entropy = _complex_gaussian_entropy(coeff_cov)
    ard_terms = _gamma_elbo_terms(
        ard_shape, ard_rate, ard_prior_shape, ard_prior_rate
    )
    noise_terms = _gamma_elbo_terms(
        np.asarray([noise_shape]),
        np.asarray([noise_rate]),
        noise_prior_shape,
        noise_prior_rate,
    )
    return float(
        likelihood
        + factor_prior
        + factor_entropy
        + coeff_prior
        + indicator_terms
        + coeff_entropy
        + ard_terms
        + noise_terms
    )


def _categorical_panel_elbo(
    reduced: np.ndarray,
    jones_mean: np.ndarray,
    jones_cov: np.ndarray,
    delay_mean: np.ndarray,
    delay_cov: np.ndarray,
    spatial_mean: np.ndarray,
    spatial_second_energy: float,
    support_probability: np.ndarray,
    conditional_coeff_mean: np.ndarray,
    conditional_coeff_variance: np.ndarray,
    ard_shape: float,
    ard_rate: float,
    noise_shape: float,
    noise_rate: float,
    *,
    factor_prior_precision: float,
    ard_prior_shape: float,
    ard_prior_rate: float,
    noise_prior_shape: float,
    noise_prior_rate: float,
) -> float:
    """ELBO for a one-active spatial-support variational mixture."""
    e_jones = float(
        np.vdot(jones_mean, jones_mean).real + np.trace(jones_cov).real
    )
    e_delay = float(
        np.vdot(delay_mean, delay_mean).real + np.trace(delay_cov).real
    )
    model_mean = (
        jones_mean[:, None, None]
        * delay_mean[None, :, None]
        * spatial_mean[None, None, :]
    )
    expected_sse = max(
        float(
            np.vdot(reduced, reduced).real
            - 2.0 * np.vdot(model_mean, reduced).real
            + e_jones * e_delay * spatial_second_energy
        ),
        0.0,
    )
    expected_noise = float(noise_shape / noise_rate)
    expected_log_noise = float(_digamma(noise_shape) - np.log(noise_rate))
    likelihood = int(reduced.size) * (
        expected_log_noise - np.log(np.pi)
    ) - expected_noise * expected_sse

    factor_prior = (
        jones_mean.size * np.log(factor_prior_precision / np.pi)
        - factor_prior_precision * e_jones
        + delay_mean.size * np.log(factor_prior_precision / np.pi)
        - factor_prior_precision * e_delay
    )
    factor_entropy = _complex_gaussian_entropy(
        jones_cov
    ) + _complex_gaussian_entropy(delay_cov)

    probability = np.maximum(
        np.asarray(support_probability, dtype=float), 1.0e-300
    )
    coefficient_second = np.abs(conditional_coeff_mean) ** 2 + np.maximum(
        np.asarray(conditional_coeff_variance, dtype=float), 1.0e-300
    )
    expected_coefficient_energy = float(
        np.dot(probability, coefficient_second)
    )
    expected_ard = float(ard_shape / ard_rate)
    expected_log_ard = float(_digamma(ard_shape) - np.log(ard_rate))
    coefficient_prior = (
        expected_log_ard
        - np.log(np.pi)
        - expected_ard * expected_coefficient_energy
    )
    coefficient_entropy = float(
        np.dot(
            probability,
            1.0
            + np.log(np.pi)
            + np.log(np.maximum(conditional_coeff_variance, 1.0e-300)),
        )
    )
    support_terms = float(
        -np.log(max(probability.size, 1))
        - np.dot(probability, np.log(probability))
    )
    ard_terms = _gamma_elbo_terms(
        np.asarray([ard_shape]),
        np.asarray([ard_rate]),
        ard_prior_shape,
        ard_prior_rate,
    )
    noise_terms = _gamma_elbo_terms(
        np.asarray([noise_shape]),
        np.asarray([noise_rate]),
        noise_prior_shape,
        noise_prior_rate,
    )
    return float(
        likelihood
        + factor_prior
        + factor_entropy
        + coefficient_prior
        + coefficient_entropy
        + support_terms
        + ard_terms
        + noise_terms
    )


def _panel_vbi_sbl(
    residual_tensor: np.ndarray,
    scene: dict,
    config: dict,
    panel: int,
    positions: np.ndarray,
    training_dict: np.ndarray,
    taus: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    cfg: dict[str, Any] | None = None,
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One mean-field VBI/SBL block for a single RIS panel."""
    cfg = {} if cfg is None else dict(cfg)
    basis = _panel_evs_basis(scene, config, panel)
    basis_q, _ = np.linalg.qr(basis, mode="reduced")
    reduced = np.einsum("ij,int->jnt", basis_q.conj(), residual_tensor)
    jones_dim, n_sub, n_train = reduced.shape
    n_grid = training_dict.shape[1]

    jones_init, delay_init, training_init = _rank1_init(reduced)
    signal_power = max(float(np.mean(np.abs(reduced) ** 2)), 1.0e-30)
    factor_prior_precision = max(
        float(cfg.get("vbi_factor_prior_precision", 1.0e-6)), 1.0e-12
    )
    ard_prior_shape = max(float(cfg.get("vbi_ard_shape", 1.0e-6)), 1.0e-12)
    ard_prior_rate = max(float(cfg.get("vbi_ard_rate", 1.0e-6)), 1.0e-12)
    noise_prior_shape = max(float(cfg.get("vbi_noise_shape", 1.0e-6)), 1.0e-12)
    noise_prior_rate = max(
        float(cfg.get("vbi_noise_rate", noise_prior_shape * signal_power)),
        1.0e-30,
    )
    min_iter = max(2, int(cfg.get("vbi_min_iter", 2)))
    dictionary_norm_sq = np.maximum(
        np.sum(np.abs(training_dict) ** 2, axis=0).real, 1.0e-30
    )

    if initial_state is None:
        jones_mean = jones_init
        delay_mean = delay_init
        jones_cov = np.eye(jones_dim, dtype=complex) * 1.0e-8
        delay_cov = np.eye(n_sub, dtype=complex) * 1.0e-8
        support_probability = np.full(n_grid, 1.0 / max(n_grid, 1))
        conditional_coeff_mean = np.zeros(n_grid, dtype=complex)
        conditional_coeff_variance = np.full(
            n_grid,
            max(float(np.vdot(training_init, training_init).real), 1.0e-12),
        )
        coeff_mean = np.zeros(n_grid, dtype=complex)
        initial_spatial = training_init
        ard_shape_scalar = float(ard_prior_shape + 1.0)
        ard_rate_scalar = float(
            ard_prior_rate
            + max(float(np.vdot(training_init, training_init).real), 1.0e-12)
        )
        initial_model = (
            jones_mean[:, None, None]
            * delay_mean[None, :, None]
            * initial_spatial[None, None, :]
        )
        noise_shape = float(noise_prior_shape + reduced.size)
        noise_rate = float(
            noise_prior_rate
            + np.vdot(reduced - initial_model, reduced - initial_model).real
        )
    else:
        jones_mean = np.asarray(initial_state["jones_mean"], dtype=complex).copy()
        delay_mean = np.asarray(initial_state["delay_mean"], dtype=complex).copy()
        jones_cov = np.asarray(initial_state["jones_cov"], dtype=complex).copy()
        delay_cov = np.asarray(initial_state["delay_cov"], dtype=complex).copy()
        support_probability = np.asarray(
            initial_state["support_probability_vector"], dtype=float
        ).copy()
        conditional_coeff_mean = np.asarray(
            initial_state["conditional_coeff_mean"], dtype=complex
        ).copy()
        conditional_coeff_variance = np.asarray(
            initial_state["conditional_coeff_variance"], dtype=float
        ).copy()
        coeff_mean = np.asarray(initial_state["coeff_mean"], dtype=complex).copy()
        ard_shape_scalar = float(initial_state["ard_shape_scalar"])
        ard_rate_scalar = float(initial_state["ard_rate_scalar"])
        noise_shape = float(initial_state["noise_shape"])
        noise_rate = float(initial_state["noise_rate"])

    previous_elbo: float | None = None
    elbo_trace: list[float] = []
    posterior_change_trace: list[float] = []
    converged = False
    numerical_rollbacks = 0
    spatial_mean = (
        np.asarray(initial_state["spatial_mean"], dtype=complex).copy()
        if initial_state is not None
        else training_init.copy()
    )
    spatial_second_energy = max(
        float(
            initial_state.get(
                "spatial_second_energy",
                np.vdot(spatial_mean, spatial_mean).real,
            )
        )
        if initial_state is not None
        else float(np.vdot(spatial_mean, spatial_mean).real),
        1.0e-30,
    )
    for iteration in range(1, max(1, int(max_iter)) + 1):
        old_jones = jones_mean.copy()
        old_jones_cov = jones_cov.copy()
        old_delay = delay_mean.copy()
        old_delay_cov = delay_cov.copy()
        old_coeff = coeff_mean.copy()
        old_support_probability = support_probability.copy()
        old_conditional_coeff_mean = conditional_coeff_mean.copy()
        old_conditional_coeff_variance = conditional_coeff_variance.copy()
        old_spatial_mean = spatial_mean.copy()
        old_spatial_second_energy = float(spatial_second_energy)
        old_ard_shape_scalar = float(ard_shape_scalar)
        old_ard_rate_scalar = float(ard_rate_scalar)
        old_noise_shape = float(noise_shape)
        old_noise_rate = float(noise_rate)
        old_coeff_second_diag = (
            coeff_second_diag.copy() if iteration > 1 else None
        )
        expected_noise_precision = float(noise_shape / noise_rate)
        expected_jones_energy = float(
            np.vdot(jones_mean, jones_mean).real + np.trace(jones_cov).real
        )
        expected_delay_energy = float(
            np.vdot(delay_mean, delay_mean).real + np.trace(delay_cov).real
        )

        matched_spatial = np.einsum(
            "j,n,jnt->t", jones_mean.conj(), delay_mean.conj(), reduced
        )
        correlations = training_dict.conj().T @ matched_spatial
        projection_energy = np.abs(correlations) ** 2 / dictionary_norm_sq
        spatial_noise_energy = max(
            float(
                np.vdot(matched_spatial, matched_spatial).real
                - np.max(projection_energy)
            ),
            1.0e-12
            * max(float(np.vdot(matched_spatial, matched_spatial).real), 1.0),
        )
        spatial_precision = float(
            max(n_train - 1, 1) / spatial_noise_energy
        )
        expected_ard = float(ard_shape_scalar / ard_rate_scalar)
        conditional_coeff_variance = 1.0 / np.maximum(
            spatial_precision
            * expected_jones_energy
            * expected_delay_energy
            * dictionary_norm_sq
            + expected_ard,
            1.0e-30,
        )
        conditional_coeff_mean = (
            conditional_coeff_variance
            * spatial_precision
            * correlations
        )
        log_support = (
            np.log(np.maximum(conditional_coeff_variance, 1.0e-300))
            + float(_digamma(ard_shape_scalar) - np.log(ard_rate_scalar))
            + np.abs(conditional_coeff_mean) ** 2
            / np.maximum(conditional_coeff_variance, 1.0e-300)
        )
        log_support -= float(np.max(log_support))
        support_probability = np.exp(np.maximum(log_support, -745.0))
        support_probability /= max(float(np.sum(support_probability)), 1.0e-300)
        coeff_mean = support_probability * conditional_coeff_mean
        coeff_second_diag = support_probability * (
            np.abs(conditional_coeff_mean) ** 2
            + conditional_coeff_variance
        )
        spatial_mean = training_dict @ coeff_mean
        spatial_second_energy = max(
            float(np.dot(coeff_second_diag, dictionary_norm_sq)),
            1.0e-30,
        )

        delay_rhs = np.einsum(
            "j,t,jnt->n", jones_mean.conj(), spatial_mean.conj(), reduced
        )
        delay_variance = 1.0 / max(
            expected_noise_precision
            * expected_jones_energy
            * spatial_second_energy
            + factor_prior_precision,
            1.0e-30,
        )
        delay_cov = np.eye(n_sub, dtype=complex) * delay_variance
        delay_mean = delay_variance * expected_noise_precision * delay_rhs
        # The rank-one likelihood has a free scale gauge.  Without fixing it,
        # the free delay/Jones factors can grow while the sparse coefficient
        # shrinks, making q(g) effectively uniform after the first update.
        # Normalize each free factor and absorb its amplitude into q(g); this
        # leaves both the posterior mean reconstruction and its second moment
        # unchanged.
        delay_scale = float(np.linalg.norm(delay_mean))
        if np.isfinite(delay_scale) and delay_scale > 1.0e-12:
            delay_mean /= delay_scale
            delay_cov /= delay_scale**2
            conditional_coeff_mean *= delay_scale
            conditional_coeff_variance *= delay_scale**2
            coeff_mean *= delay_scale
            coeff_second_diag *= delay_scale**2
            spatial_mean *= delay_scale
            spatial_second_energy *= delay_scale**2
        else:
            delay_mean = old_delay
            delay_cov = old_delay_cov
        expected_delay_energy = float(
            np.vdot(delay_mean, delay_mean).real + np.trace(delay_cov).real
        )

        jones_rhs = np.einsum(
            "n,t,jnt->j", delay_mean.conj(), spatial_mean.conj(), reduced
        )
        jones_variance = 1.0 / max(
            expected_noise_precision
            * expected_delay_energy
            * spatial_second_energy
            + factor_prior_precision,
            1.0e-30,
        )
        jones_cov = np.eye(jones_dim, dtype=complex) * jones_variance
        jones_mean = jones_variance * expected_noise_precision * jones_rhs
        jones_scale = float(np.linalg.norm(jones_mean))
        if np.isfinite(jones_scale) and jones_scale > 1.0e-12:
            jones_mean /= jones_scale
            jones_cov /= jones_scale**2
            conditional_coeff_mean *= jones_scale
            conditional_coeff_variance *= jones_scale**2
            coeff_mean *= jones_scale
            coeff_second_diag *= jones_scale**2
            spatial_mean *= jones_scale
            spatial_second_energy *= jones_scale**2
        else:
            jones_mean = old_jones
            jones_cov = old_jones_cov
        expected_jones_energy = float(
            np.vdot(jones_mean, jones_mean).real + np.trace(jones_cov).real
        )

        ard_shape_scalar = float(ard_prior_shape + 1.0)
        ard_rate_scalar = float(
            ard_prior_rate + np.sum(coeff_second_diag)
        )

        model_mean = (
            jones_mean[:, None, None]
            * delay_mean[None, :, None]
            * spatial_mean[None, None, :]
        )
        expected_sse = max(
            float(
                np.vdot(reduced, reduced).real
                - 2.0 * np.vdot(model_mean, reduced).real
                + expected_jones_energy
                * expected_delay_energy
                * spatial_second_energy
            ),
            0.0,
        )
        noise_shape = float(noise_prior_shape + reduced.size)
        noise_rate = float(noise_prior_rate + expected_sse)

        elbo = _categorical_panel_elbo(
            reduced,
            jones_mean,
            jones_cov,
            delay_mean,
            delay_cov,
            spatial_mean,
            spatial_second_energy,
            support_probability,
            conditional_coeff_mean,
            conditional_coeff_variance,
            ard_shape_scalar,
            ard_rate_scalar,
            noise_shape,
            noise_rate,
            factor_prior_precision=factor_prior_precision,
            ard_prior_shape=ard_prior_shape,
            ard_prior_rate=ard_prior_rate,
            noise_prior_shape=noise_prior_shape,
            noise_prior_rate=noise_prior_rate,
        )
        catastrophic_elbo_drop = bool(
            previous_elbo is not None
            and (
                not np.isfinite(elbo)
                or elbo
                < previous_elbo
                - 1.0e6 * max(abs(previous_elbo), 1.0)
            )
        )
        if catastrophic_elbo_drop:
            jones_mean = old_jones
            jones_cov = old_jones_cov
            delay_mean = old_delay
            delay_cov = old_delay_cov
            coeff_mean = old_coeff
            support_probability = old_support_probability
            conditional_coeff_mean = old_conditional_coeff_mean
            conditional_coeff_variance = old_conditional_coeff_variance
            spatial_mean = old_spatial_mean
            spatial_second_energy = old_spatial_second_energy
            ard_shape_scalar = old_ard_shape_scalar
            ard_rate_scalar = old_ard_rate_scalar
            noise_shape = old_noise_shape
            noise_rate = old_noise_rate
            if old_coeff_second_diag is not None:
                coeff_second_diag = old_coeff_second_diag
            numerical_rollbacks += 1
            posterior_change_trace.append(0.0)
            break
        elbo_trace.append(elbo)
        posterior_change = max(
            float(np.linalg.norm(jones_mean - old_jones) / (np.linalg.norm(old_jones) + 1.0e-12)),
            float(np.linalg.norm(delay_mean - old_delay) / (np.linalg.norm(old_delay) + 1.0e-12)),
            float(np.linalg.norm(coeff_mean - old_coeff) / (np.linalg.norm(old_coeff) + 1.0e-12)),
            float(
                np.linalg.norm(support_probability - old_support_probability)
                / (np.linalg.norm(old_support_probability) + 1.0e-12)
            ),
        )
        posterior_change_trace.append(posterior_change)
        if previous_elbo is not None and iteration >= min_iter:
            relative_elbo_change = abs(elbo - previous_elbo) / max(
                abs(previous_elbo), 1.0
            )
            if relative_elbo_change <= float(tol) and posterior_change <= np.sqrt(float(tol)):
                converged = True
                break
        previous_elbo = elbo

    active_probability = support_probability
    posterior_energy = support_probability * (
        np.abs(conditional_coeff_mean) ** 2
        + conditional_coeff_variance
    )
    maximum_activity = float(np.max(active_probability))
    activity_ties = np.flatnonzero(
        active_probability >= maximum_activity - 1.0e-12
    )
    argmax = int(
        activity_ties[int(np.argmax(posterior_energy[activity_ties]))]
    )
    coefficient_second_matrix = np.diag(posterior_energy.astype(complex))
    coeff_cov = coefficient_second_matrix - np.outer(
        coeff_mean, coeff_mean.conj()
    )
    coeff_cov = 0.5 * (coeff_cov + coeff_cov.conj().T)
    indicator_prob = np.column_stack(
        [1.0 - support_probability, support_probability]
    )
    ard_shape = np.full(n_grid, ard_shape_scalar, dtype=float)
    ard_rate = np.full(n_grid, ard_rate_scalar, dtype=float)
    grid = np.asarray(positions, dtype=float)
    position = grid[argmax]
    # A single near-field panel constrains angle well but range weakly, so only
    # the direction (not the ambiguous range) is used in the geometric fusion.
    rotation = np.asarray(scene["rotations"][int(panel)], dtype=float)
    center = np.asarray(scene["ris_centers"][int(panel)], dtype=float)
    direction_local = rotation @ (position - center)
    direction_local = direction_local / (float(np.linalg.norm(direction_local)) + 1.0e-15)
    tau = _extract_delay(scene, delay_mean, taus)
    return {
        "panel": int(panel),
        "position": np.asarray(position, dtype=float),
        "direction_local": direction_local,
        "tau": float(tau),
        "jones": jones_mean,
        "jones_mean": jones_mean,
        "jones_cov": jones_cov,
        "spatial": spatial_mean,
        "spatial_mean": spatial_mean,
        "spatial_second_energy": spatial_second_energy,
        "delay": delay_mean,
        "delay_mean": delay_mean,
        "delay_cov": delay_cov,
        "coeff_mean": coeff_mean,
        "coeff_cov": coeff_cov,
        "support_probability_vector": support_probability,
        "conditional_coeff_mean": conditional_coeff_mean,
        "conditional_coeff_variance": conditional_coeff_variance,
        "indicator_prob": indicator_prob,
        "ard_shape": ard_shape,
        "ard_rate": ard_rate,
        "ard_shape_scalar": ard_shape_scalar,
        "ard_rate_scalar": ard_rate_scalar,
        "noise_shape": noise_shape,
        "noise_rate": noise_rate,
        "support_index": argmax,
        "support_probability": float(active_probability[argmax]),
        "support_score": float(active_probability[argmax]),
        "support_posterior_energy": float(posterior_energy[argmax]),
        "evs_basis": basis_q,
        "confidence": float(active_probability[argmax]),
        "iterations_run": int(iteration),
        "converged": bool(converged),
        "numerical_rollbacks": int(numerical_rollbacks),
        "elbo_trace": elbo_trace,
        "posterior_change_trace": posterior_change_trace,
    }


def _reconstruct_panel(panel_state: dict[str, Any], scene: dict) -> np.ndarray:
    basis = panel_state["evs_basis"]
    jones = panel_state["jones_mean"]
    delay = panel_state["delay_mean"]
    spatial = panel_state["spatial_mean"]
    evs = basis @ jones
    return (
        evs[:, None, None] * delay[None, :, None] * spatial[None, None, :]
    )


def _raw_objective(
    scene: dict, config: dict, y_vec: np.ndarray, p: np.ndarray, delta_t: float
) -> float:
    groups = supports_from_position_clock(scene, np.asarray(p, dtype=float), float(delta_t), model_variant="near_field")
    expanded = [item for group in groups for item in expand_jones_group(group, 2)]
    _, _, residual = factorized_fit_supports(scene, config, expanded, y_vec, ridge=1.0e-10)
    return float(np.linalg.norm(residual) ** 2)


def _fuse_position_clock_geometric(
    scene: dict,
    config: dict,
    panels: list[dict[str, Any]],
) -> tuple[np.ndarray, float]:
    """Closed-form weighted-LS fusion of the converged per-panel VBI outputs.

    This is the localization step of the reference transported to the RIS-only
    geometry.  Each panel supplies a local direction ``u_k`` (read off the ARD
    spatial support) and a delay ``tau_k = (r_k + d_RB_k) / c0 + Delta_t``.
    Eliminating the unknown ranges ``r_k`` leaves a system that is *linear* in
    the four unknowns ``(p_u, c0 * Delta_t)``,

        p_u + (c0 Delta_t) R_k^T u_k = center_k + (c0 tau_k - d_RB_k) R_k^T u_k,

    i.e. three equations per panel, solved here in weighted least squares with
    the panel posterior confidences as weights.  Eqs. (80)-(85) of Li et al.
    solve the same system after cancelling the clock against the *direct AP-UE*
    link; this scene has no direct link, so the clock is retained as an explicit
    fourth unknown instead of being differenced away.

    No evaluation of the exact forward model enters here.  That is the whole
    point of the ``as_published`` tier: the reference contains no continuous
    likelihood refinement over ``(p_u, Delta_t)``, so neither does this path.
    """
    bounds_p = np.asarray(config.get("ue_bounds"), dtype=float)
    bounds_dt = np.asarray(config.get("delta_t_bounds"), dtype=float)
    c0 = float(scene["c0"])

    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    weights: list[float] = []
    for entry in panels:
        panel = int(entry["panel"])
        rotation = np.asarray(scene["rotations"][panel], dtype=float)
        center = np.asarray(scene["ris_centers"][panel], dtype=float)
        direction = rotation.T @ np.asarray(entry["direction_local"], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            continue
        direction = direction / norm
        offset = c0 * float(entry["tau"]) - float(scene["d_RB"][panel])
        block = np.zeros((3, 4), dtype=float)
        block[:, :3] = np.eye(3)
        block[:, 3] = direction
        rows.append(block)
        rhs.append(center + offset * direction)
        weights.append(float(max(entry.get("confidence", 0.0), 0.0)))

    if not rows:
        median_pos = np.clip(
            np.median(np.asarray([p["position"] for p in panels], dtype=float), axis=0),
            bounds_p[:, 0], bounds_p[:, 1],
        )
        return median_pos.astype(float), float(np.mean(bounds_dt))

    weight_vec = np.asarray(weights, dtype=float)
    total = float(np.sum(weight_vec))
    scales = np.sqrt(weight_vec / total) if total > 0.0 else np.ones(len(rows))
    design = np.vstack([scale * block for scale, block in zip(scales, rows)])
    target = np.concatenate([scale * value for scale, value in zip(scales, rhs)])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)

    position = np.clip(solution[:3], bounds_p[:, 0], bounds_p[:, 1])
    delta_t = float(np.clip(solution[3] / c0, bounds_dt[0], bounds_dt[1]))
    return position.astype(float), delta_t


def _fuse_position_clock(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    panels: list[dict[str, Any]],
) -> tuple[np.ndarray, float]:
    """Joint MAP/ML refinement of (position, clock) over the exact model.

    A single near-field panel is range-ambiguous and only the strongest panel is
    reliable after the per-panel peel, so the fused estimate is obtained by a
    likelihood refinement seeded from the strongest panel's direction combined
    with a clock scan; the objective itself fits all K panels jointly.

    This continuous exact-model polish has no counterpart in Li et al., which
    localizes in closed form from the converged support (see
    :func:`_fuse_position_clock_geometric`).  It is therefore gated on the
    declared ``baselines.refinement_tier``: it runs under
    ``"refinement_matched"``, where every method including the proposed one is
    granted the same final polish, and is skipped under ``"as_published"``.
    """
    bounds_p = np.asarray(config.get("ue_bounds"), dtype=float)
    bounds_dt = np.asarray(config.get("delta_t_bounds"), dtype=float)
    c0 = float(scene["c0"])
    strong = max(panels, key=lambda entry: float(entry.get("confidence", 0.0)))
    center = np.asarray(scene["ris_centers"][int(strong["panel"])], dtype=float)
    rotation = np.asarray(scene["rotations"][int(strong["panel"])], dtype=float)
    d_rb = float(scene["d_RB"][int(strong["panel"])])
    direction = np.asarray(strong["direction_local"], dtype=float)

    seeds: list[tuple[np.ndarray, float]] = []
    for delta_t in np.linspace(bounds_dt[0], bounds_dt[1], 13):
        rho = c0 * (float(strong["tau"]) - float(delta_t)) - d_rb
        if rho <= 0.0:
            continue
        position = center + rotation.T @ (rho * direction)
        seeds.append((np.clip(position, bounds_p[:, 0], bounds_p[:, 1]), float(delta_t)))
    median_pos = np.clip(
        np.median(np.asarray([p["position"] for p in panels], dtype=float), axis=0),
        bounds_p[:, 0], bounds_p[:, 1],
    )
    seeds.append((median_pos, float(np.mean(bounds_dt))))
    if not seeds:
        return median_pos.astype(float), float(np.mean(bounds_dt))

    best_seed = min(seeds, key=lambda s: _raw_objective(scene, config, y_vec, s[0], s[1]))
    p0, dt0 = best_seed
    lower = np.r_[bounds_p[:, 0], bounds_dt[0] * c0]
    upper = np.r_[bounds_p[:, 1], bounds_dt[1] * c0]

    def objective(state: np.ndarray) -> float:
        clipped = np.clip(np.asarray(state, dtype=float), lower, upper)
        return _raw_objective(scene, config, y_vec, clipped[:3], clipped[3] / c0)

    x0 = np.clip(np.r_[p0, dt0 * c0], lower, upper)
    if scipy_is_available():
        try:
            from scipy.optimize import minimize  # type: ignore[import-not-found]

            result = minimize(
                objective, x0, method="Nelder-Mead",
                options={"maxiter": int(config.get("baselines", {}).get("vbi_refine_maxiter", 200)), "xatol": 1.0e-4, "fatol": 1.0e-12},
            )
            final = np.clip(np.asarray(result.x, dtype=float), lower, upper)
            if objective(final) <= objective(x0):
                return final[:3].astype(float), float(final[3] / c0)
        except Exception:  # noqa: BLE001 - keep the best seed.
            pass
    return p0.astype(float), dt0


def _model_components(scene: dict, p_u: np.ndarray, delta_t: float) -> dict[str, np.ndarray]:
    ranges, taus = [], []
    for panel in range(int(scene["K"])):
        range_m, _, _, _ = local_geometry_from_position(
            np.asarray(p_u, dtype=float),
            np.asarray(scene["ris_centers"][panel], dtype=float),
            np.asarray(scene["rotations"][panel], dtype=float),
        )
        ranges.append(float(range_m))
        taus.append((float(range_m) + float(scene["d_RB"][panel])) / float(scene["c0"]) + float(delta_t))
    return {"ranges": np.asarray(ranges), "taus": np.asarray(taus)}


def run_ris_vbi_sbl_baseline(data: dict, config: dict) -> BaselineResult:
    start = time.perf_counter()
    scene = data["scene"]
    cfg = dict(config.get("baselines", {}).get("ris_vbi_sbl", {}))
    cfg.setdefault("nf_grid_x", 9)
    cfg.setdefault("nf_grid_y", 9)
    cfg.setdefault("nf_grid_z", 7)
    y_vec = vectorize_raw_observation(data["Y_noisy"])
    shape = (int(scene["I"]), int(scene["N"]), int(scene["T"]))
    observation_tensor = np.asarray(data["Y_noisy"], dtype=complex).reshape(shape)

    positions = _nf_position_grid(config, cfg)
    training_dicts = {
        panel: _nf_training_matrix(scene, panel, positions)
        for panel in range(int(scene["K"]))
    }
    taus = delay_grid_from_scene(scene, config, int(cfg.get("delay_grid_size", 121)))
    max_iter = int(cfg.get("vbi_max_iter", 40))
    tol = float(cfg.get("vbi_tol", 1.0e-6))

    # Coordinate the per-panel variational blocks on one shared residual.  A
    # single strongest-first peel is order dependent and does not revisit an
    # early panel after the other panel posteriors change.
    def _panel_energy(panel: int) -> float:
        basis_q, _ = np.linalg.qr(
            _panel_evs_basis(scene, config, panel), mode="reduced"
        )
        reduced = np.einsum("ij,int->jnt", basis_q.conj(), observation_tensor)
        return float(np.linalg.norm(reduced))

    panel_order = sorted(range(int(scene["K"])), key=_panel_energy, reverse=True)
    panel_cycles = max(1, int(cfg.get("vbi_panel_cycles", 2)))
    panel_cycle_tol = float(cfg.get("vbi_panel_cycle_tol", tol))
    states: dict[int, dict[str, Any]] = {}
    panel_trace: list[dict[str, Any]] = []
    cycle_objectives: list[float] = []

    def reconstruction(current: dict[int, dict[str, Any]]) -> np.ndarray:
        total = np.zeros(shape, dtype=complex)
        for child in current.values():
            total += _reconstruct_panel(child, scene)
        return total

    for cycle in range(panel_cycles):
        for panel in panel_order:
            other = {
                key: value for key, value in states.items() if int(key) != int(panel)
            }
            complete_data = observation_tensor - reconstruction(other)
            before_objective = float(
                np.linalg.norm(observation_tensor - reconstruction(states)) ** 2
            )
            warm_state = states.get(panel)
            warm_state_used = bool(
                warm_state is not None
                and float(warm_state.get("confidence", 0.0))
                > 1.01 / max(len(positions), 1)
            )
            candidate = _panel_vbi_sbl(
                complete_data,
                scene,
                config,
                panel,
                positions,
                training_dicts[panel],
                taus,
                max_iter=max_iter,
                tol=tol,
                cfg=cfg,
                initial_state=warm_state if warm_state_used else None,
            )
            candidate_states = dict(states)
            candidate_states[panel] = candidate
            after_objective = float(
                np.linalg.norm(
                    observation_tensor - reconstruction(candidate_states)
                )
                ** 2
            )
            accepted = bool(
                np.isfinite(after_objective)
                and (
                    panel not in states
                    or after_objective <= before_objective * (1.0 + 1.0e-10)
                    + 1.0e-12
                )
            )
            if accepted:
                states = candidate_states
                state = candidate
            else:
                state = states[panel]
            panel_trace.append(
                {
                    "cycle": int(cycle),
                    "panel": int(panel),
                    "position": np.asarray(state["position"], dtype=float).tolist(),
                    "tau": float(state["tau"]),
                    "confidence": float(state["confidence"]),
                    "support_index": int(state["support_index"]),
                    "warm_state_used": warm_state_used,
                    "iterations_run": int(candidate["iterations_run"]),
                    "converged": bool(candidate["converged"]),
                    "numerical_rollbacks": int(
                        candidate.get("numerical_rollbacks", 0)
                    ),
                    "elbo_initial": float(candidate["elbo_trace"][0]),
                    "elbo_final": float(candidate["elbo_trace"][-1]),
                    "global_raw_sse_before": before_objective,
                    "global_raw_sse_after": after_objective,
                    "accepted": accepted,
                }
            )
        cycle_objective = float(
            np.linalg.norm(observation_tensor - reconstruction(states)) ** 2
        )
        cycle_objectives.append(cycle_objective)
        if len(cycle_objectives) >= 2:
            relative_change = abs(cycle_objectives[-2] - cycle_objectives[-1]) / max(
                cycle_objectives[-2], 1.0e-30
            )
            if relative_change <= panel_cycle_tol:
                break

    panels = [states[panel] for panel in panel_order if panel in states]

    bounds_p = np.asarray(config.get("ue_bounds"), dtype=float)
    bounds_dt = np.asarray(config.get("delta_t_bounds"), dtype=float)
    tier = baseline_refinement_tier(config)
    if tier == "as_published":
        p_hat, dt_hat = _fuse_position_clock_geometric(scene, config, panels)
    else:
        p_hat, dt_hat = _fuse_position_clock(scene, config, y_vec, panels)
    p_hat = np.clip(p_hat, bounds_p[:, 0], bounds_p[:, 1])
    dt_hat = float(np.clip(dt_hat, bounds_dt[0], bounds_dt[1]))

    groups = supports_from_position_clock(scene, p_hat, dt_hat, model_variant="near_field")
    expanded = [item for group in groups for item in expand_jones_group(group, 2)]
    coeffs, y_hat, residual = factorized_fit_supports(scene, config, expanded, y_vec, ridge=1.0e-10)
    raw_objective = float(np.linalg.norm(residual) ** 2 / max(y_vec.size, 1))

    diagnostics = {
        "dictionary_mode": "ris_vbi_sbl_near_field_per_panel",
        "model_variant": "variational_bayesian_sbl_adaptation",
        "reference_algorithm": "VBI/SBL joint localization + channel reconstruction (Li et al., TWC 2024)",
        "adaptation_note": (
            "per_panel_mean_field_vbi_with_one_active_categorical_spatial_"
            "posterior_ard_amplitude_prior_and_scale_gauge_fixed_free_gaussian_"
            "delay_and_jones_gain;_geometric_common_clock_fusion"
        ),
        "refinement_tier": tier,
        "refinement_tier_sensitive": True,
        "fusion_rule": (
            "weighted_linear_ls_over_panel_directions_and_delays"
            if tier == "as_published"
            else "exact_model_seed_scan_plus_nelder_mead_over_position_and_clock"
        ),
        "exact_model_refinement_used": bool(tier != "as_published"),
        "clock_output_semantics": "native_joint_common_clock",
        "vbi_max_iter": max_iter,
        "vbi_tol": tol,
        "vbi_panel_cycles_requested": panel_cycles,
        "vbi_panel_cycles_run": len(cycle_objectives),
        "vbi_panel_cycle_objectives": cycle_objectives,
        "vbi_iterations_total": int(
            sum(int(row["iterations_run"]) for row in panel_trace)
        ),
        "vbi_numerical_rollbacks": int(
            sum(int(row["numerical_rollbacks"]) for row in panel_trace)
        ),
        "vbi_all_updates_used_posterior_spatial_state": True,
        "vbi_support_source": "one_active_categorical_variational_posterior",
        "grid_size": int(len(positions) + len(taus)),
        "nf_position_grid_size": int(len(positions)),
        "delay_grid_size": int(len(taus)),
        "panel_trace": panel_trace,
        "support_size": len(panels),
        "selected_panels": [int(p["panel"]) for p in panels],
        "selected_panel_count": len({int(p["panel"]) for p in panels}),
        "unique_panel_constraint": True,
        "coeff_norm": float(np.linalg.norm(coeffs)),
        "residual_norm": float(np.linalg.norm(residual)),
        "raw_objective_final": raw_objective,
        "backend": "cpu",
        "gpu_used": False,
        "gpu_device": "",
        "backend_warning": "",
        "warning": "",
    }
    return BaselineResult(
        name="ris_vbi_sbl",
        p_u=p_hat,
        delta_t=dt_hat,
        Y_hat=y_hat.reshape(shape),
        raw_objective_final=raw_objective,
        components=_model_components(scene, p_hat, dt_hat),
        selected_support=[{"panel": int(p["panel"]), "position": np.asarray(p["position"]).tolist(), "tau": float(p["tau"])} for p in panels],
        runtime_s=time.perf_counter() - start,
        diagnostics=diagnostics,
    )
