"""The Stage-II EVS subspace compression must be an identity, not a truncation.

Every admissible EVS factor is ``kron(v_B[k], Theta[k] p(gamma_k, eta_k))`` and
``v_B``/``Theta`` depend only on the known RIS->BS geometry, so the whole
variable-projection dictionary lies in a fixed ``r = 2K`` subspace of the ``I``
dimensional EVS mode whatever the nonlinear parameters are.  Projecting the
observation there therefore

  * splits the raw residual exactly into a compressed part plus a constant,
  * leaves the profiled path gains unchanged,
  * leaves the projected-Jacobian EFIM unchanged,

and only removes rows that cannot carry signal.  These tests pin all three so
the compression can never silently become an approximation.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.channel_model import (
    add_awgn,
    channel_components,
    generate_scene,
    synthesize_raw_tensor,
)
from src.config import default_config
from src.estimators import (
    _bounds_global,
    _global_factors_from_x,
    compress_raw_evs_observation,
    global_vp_evs_basis,
    global_vp_residual,
    initialize_from_hankel,
    refine_global_raw,
)
from src.global_vp import data_only_efim_diagnostic
from src.tensor_utils import (
    blocked_squared_error,
    hankelize_frequency,
    khatri_rao_synthesize,
)


def _small_config() -> dict:
    config = default_config()
    config["K"] = 2
    config["M_A"] = 4
    config["ris_shape"] = (8, 8)
    config["N"] = 15
    config["P"] = 8
    config["T"] = 24
    config["print_progress"] = False
    config["ris_centers"] = config["ris_centers"][:2]
    config["ris_rotations"] = config["ris_rotations"][:2]
    return config


@pytest.fixture(scope="module")
def scenario() -> dict:
    config = _small_config()
    rng = np.random.default_rng(20260728)
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(components, scene["beta_true"])
    noise_rng = np.random.default_rng(7)
    y_noisy, noise_variance = add_awgn(
        y_true, -5.0, noise_rng, active_mask=scene.get("evs_observation_mask")
    )
    return {
        "config": config,
        "scene": scene,
        "y_noisy": y_noisy,
        "noise_variance": noise_variance,
    }


def test_union_basis_is_a_strict_isometric_compression(scenario):
    scene, config = scenario["scene"], scenario["config"]
    basis = global_vp_evs_basis(scene, config)
    assert basis is not None
    assert basis.shape == (scene["I"], 2 * scene["K"])
    identity = basis.conj().T @ basis
    assert np.allclose(identity, np.eye(basis.shape[1]), atol=1e-12)


def test_residual_splits_exactly_for_arbitrary_parameters_and_gains(scenario):
    scene, config = scenario["scene"], scenario["config"]
    y_noisy = scenario["y_noisy"]
    basis = global_vp_evs_basis(scene, config)
    compressed, orthogonal = compress_raw_evs_observation(y_noisy, basis)

    # The explicit orthogonal energy and the Pythagorean one must agree here:
    # the discarded mass dominates, so this split is well conditioned.
    pythagorean = float(
        np.linalg.norm(y_noisy) ** 2 - np.linalg.norm(compressed) ** 2
    )
    assert abs(orthogonal - pythagorean) <= 1e-10 * abs(pythagorean)

    lower, upper = _bounds_global(scene, config)
    rng = np.random.default_rng(11)
    for _ in range(6):
        x = lower + rng.random(lower.size) * (upper - lower)
        factors, _ = _global_factors_from_x(scene, x)
        gains = rng.standard_normal(scene["K"]) + 1j * rng.standard_normal(scene["K"])

        raw = khatri_rao_synthesize(factors, gains) - y_noisy
        compressed_factors = (basis.conj().T @ factors[0], factors[1], factors[2])
        small = khatri_rao_synthesize(compressed_factors, gains) - compressed

        raw_energy = float(np.vdot(raw, raw).real)
        split_energy = float(np.vdot(small, small).real) + orthogonal
        assert abs(split_energy - raw_energy) <= 1e-11 * raw_energy


def test_profiled_gains_and_objective_are_unchanged(scenario):
    scene, config = scenario["scene"], scenario["config"]
    y_noisy = scenario["y_noisy"]
    basis = global_vp_evs_basis(scene, config)
    compressed, orthogonal = compress_raw_evs_observation(y_noisy, basis)
    lower, upper = _bounds_global(scene, config)
    rng = np.random.default_rng(23)
    for _ in range(6):
        x = lower + rng.random(lower.size) * (upper - lower)
        raw_residual, raw_beta, _ = global_vp_residual(scene, x, y_noisy)
        small_residual, small_beta, _ = global_vp_residual(
            scene, x, compressed, evs_basis=basis
        )
        assert small_residual.size * scene["I"] == raw_residual.size * basis.shape[1]
        assert np.allclose(raw_beta, small_beta, rtol=1e-10, atol=1e-12)
        raw_objective = float(np.vdot(raw_residual, raw_residual).real)
        small_objective = float(np.vdot(small_residual, small_residual).real)
        assert abs(small_objective + orthogonal - raw_objective) <= 1e-11 * raw_objective


def test_refine_global_raw_agrees_with_the_uncompressed_solver(scenario):
    scene, config = scenario["scene"], scenario["config"]
    y_noisy = scenario["y_noisy"]
    estimate = initialize_from_hankel(
        hankelize_frequency(y_noisy, scene["P"]), scene, config
    )
    on = copy.deepcopy(config)
    on["global_vp_raw_evs_compression"] = True
    off = copy.deepcopy(config)
    off["global_vp_raw_evs_compression"] = False

    result_on = refine_global_raw(y_noisy, scene, on, copy.deepcopy(estimate))
    result_off = refine_global_raw(y_noisy, scene, off, copy.deepcopy(estimate))

    assert result_on["global_vp_raw_evs_compression"] is True
    assert result_off["global_vp_raw_evs_compression"] is False
    # The compression adds one constant residual entry with an identically zero
    # Jacobian row, so the optimizer must take the same path.
    assert result_on["residual_eval_count"] == result_off["residual_eval_count"]
    assert result_on["optimizer"]["message"] == result_off["optimizer"]["message"]
    assert result_on["raw_objective_final"] == pytest.approx(
        result_off["raw_objective_final"], rel=1e-10
    )
    # The objective is the exact invariant.  The recovered parameters agree only
    # to the resolution of the solver's own 2-point finite-difference Jacobian:
    # its relative step is ~1.5e-8 of a 2.4 m position box, i.e. ~4e-8 m, and a
    # ~1e-15 relative difference in the objective moves the converged point by a
    # small multiple of that.  That floor is four orders below the estimator's
    # own position accuracy, so it is pinned loosely on purpose.
    assert np.linalg.norm(
        np.asarray(result_on["p_u"]) - np.asarray(result_off["p_u"])
    ) < 1e-5
    assert abs(result_on["delta_t"] - result_off["delta_t"]) < 1e-15


def test_data_only_efim_is_unchanged_by_the_compression(scenario):
    scene, config = scenario["scene"], scenario["config"]
    y_noisy = scenario["y_noisy"]
    estimate = initialize_from_hankel(
        hankelize_frequency(y_noisy, scene["P"]), scene, config
    )
    p_u = np.asarray(scene["p_u_true"], dtype=float) + 2.0e-3
    delta_t = float(scene["delta_t_true"]) + 1.0e-11

    results = {}
    for label, enabled in (("on", True), ("off", False)):
        local = copy.deepcopy(config)
        local["global_vp"] = dict(local["global_vp"])
        local["global_vp"]["efim_evs_compression"] = enabled
        results[label] = data_only_efim_diagnostic(
            y_noisy, p_u, delta_t, estimate, scene, local,
            sigma2=scenario["noise_variance"],
        )

    reference = results["off"]["data_only_scaled_efim"]
    scale = max(np.abs(reference).max(), 1e-300)
    assert np.abs(results["on"]["data_only_scaled_efim"] - reference).max() <= 1e-10 * scale
    assert results["on"]["data_only_scaled_efim_condition_number"] == pytest.approx(
        results["off"]["data_only_scaled_efim_condition_number"], rel=1e-9
    )
    assert results["on"]["data_only_rank_gram"] == results["off"]["data_only_rank_gram"]


def test_efim_noise_estimate_matches_without_a_supplied_sigma2(scenario):
    scene, config = scenario["scene"], scenario["config"]
    y_noisy = scenario["y_noisy"]
    estimate = initialize_from_hankel(
        hankelize_frequency(y_noisy, scene["P"]), scene, config
    )
    values = []
    for enabled in (True, False):
        local = copy.deepcopy(config)
        local["global_vp"] = dict(local["global_vp"])
        local["global_vp"]["efim_evs_compression"] = enabled
        values.append(
            data_only_efim_diagnostic(
                y_noisy, scene["p_u_true"], scene["delta_t_true"],
                estimate, scene, local, sigma2=None,
            )["efim_sigma2"]
        )
    assert values[0] == pytest.approx(values[1], rel=1e-10)


def test_blocked_squared_error_matches_the_materialized_difference():
    rng = np.random.default_rng(5)
    left = rng.standard_normal((7, 5, 11)) + 1j * rng.standard_normal((7, 5, 11))
    right = rng.standard_normal((7, 5, 11)) + 1j * rng.standard_normal((7, 5, 11))
    reference = float(np.linalg.norm(left - right) ** 2)
    for block in (1, 13, 4096):
        assert blocked_squared_error(left, right, block=block) == pytest.approx(
            reference, rel=1e-12
        )
