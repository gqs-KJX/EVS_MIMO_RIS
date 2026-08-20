"""Fidelity of the external baselines to the algorithms they cite.

Two reference behaviours are pinned here because they are what carries the
citation, and a silent regression in either would misattribute a published
method:

* Lin et al. (IEEE TWC 2021) solve the CPD *algebraically*, by GEVD
  simultaneous diagonalization -- their Algorithm 1 "avoids iterative runs and
  random initialization" and "guarantees to return the exact solution in the
  noiseless case".  CP-ALS appears in that paper only as a competing method.
* Yan et al. Remark 2 excludes a path whose implied clock offset departs from
  the consensus one from the weighted least squares of (50).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.als_cpd import (
    algebraic_cp_gevd,
    complex_cp_als,
    mdl_rank_estimate,
    reconstruct_cp_tensor,
)
from src.baselines.nf_ris_groupomp_localgrid_wls import (
    _near_field_support,
    _wls_position_clock,
)
from src.channel_model import generate_scene
from src.config import default_config


def _cp_tensor(shape, rank, seed):
    rng = np.random.default_rng(seed)
    factors = [
        rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank))
        for dim in shape
    ]
    tensor = np.zeros(shape, dtype=complex)
    for column in range(rank):
        tensor += (
            factors[0][:, column, None, None]
            * factors[1][None, :, column, None]
            * factors[2][None, None, :, column]
        )
    return tensor, factors


@pytest.mark.parametrize(
    "shape,rank", [((6, 5, 4), 3), ((12, 9, 7), 4), ((8, 8, 8), 5), ((20, 16, 10), 3)]
)
def test_algebraic_cpd_is_exact_and_noniterative(shape, rank):
    """Lin et al. Algorithm 1: exact in the noiseless case, with no iteration."""
    tensor, _ = _cp_tensor(shape, rank, seed=11)
    factors, weights, diagnostics = algebraic_cp_gevd(tensor, rank)
    error = np.linalg.norm(
        reconstruct_cp_tensor(factors, weights) - tensor
    ) / np.linalg.norm(tensor)
    assert error < 1.0e-10
    assert diagnostics["cpd_iterations"] == 0
    assert diagnostics["cpd_random_initialization"] is False
    assert diagnostics["cpd_solver"] == "lin_gevd"


def test_algebraic_start_reaches_the_als_solution_in_fewer_iterations():
    """The algebraic solution is a strictly better ALS start, not a worse one."""
    clean, _ = _cp_tensor((12, 9, 7), 3, seed=5)
    rng = np.random.default_rng(6)
    noise = rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
    noisy = clean + noise * np.linalg.norm(clean) / np.linalg.norm(noise) * 10 ** (-20 / 20)

    _, _, cold = complex_cp_als(noisy, 3, max_iter=500, tol=1.0e-10)
    algebraic, _, _ = algebraic_cp_gevd(noisy, 3)
    factors, weights, warm = complex_cp_als(
        noisy, 3, max_iter=500, tol=1.0e-10, init=algebraic
    )

    assert warm["als_initialization"] == "external"
    assert cold["als_initialization"] == "svd"
    assert warm["als_iterations"] <= cold["als_iterations"]
    assert warm["als_residual"] <= cold["als_residual"] * 1.05
    assert np.isfinite(reconstruct_cp_tensor(factors, weights)).all()


def test_mdl_recovers_the_component_count():
    """Lin et al. (12)-(13); kept for diagnostics, not used to set the rank."""
    for rank in (2, 3, 4):
        clean, _ = _cp_tensor((30, 1, 8), rank, seed=7)
        matrix = clean[:, 0, :]
        rng = np.random.default_rng(8)
        noise = rng.normal(size=matrix.shape) + 1j * rng.normal(size=matrix.shape)
        matrix = matrix + noise * np.linalg.norm(matrix) / np.linalg.norm(noise) * 10 ** (
            -25 / 20
        )
        assert mdl_rank_estimate(matrix) == rank


# ------------------------------------------------------- Yan et al. Remark 2 --


@pytest.fixture(scope="module")
def gate_scene():
    config = default_config()
    config["K"] = 3
    config["seed"] = 20260728
    config["diagnostic_fast_problem_size"] = True
    scene = generate_scene(config, np.random.default_rng(int(config["seed"])))
    return config, scene


def _consistent_supports(scene, clock_offsets_ns=None):
    """Panel supports that are exact for the true geometry, optionally detuned."""
    position = np.asarray(scene["p_u_true"], dtype=float)
    supports = []
    for panel in range(int(scene["K"])):
        rotation = np.asarray(scene["rotations"][panel], dtype=float)
        center = np.asarray(scene["ris_centers"][panel], dtype=float)
        local = rotation @ (position - center)
        range_m = float(np.linalg.norm(local))
        direction = local / range_m
        tau = (range_m + float(scene["d_RB"][panel])) / float(scene["c0"]) + float(
            scene["delta_t_true"]
        )
        if clock_offsets_ns is not None:
            tau += float(clock_offsets_ns[panel]) * 1.0e-9
        supports.append(
            _near_field_support(
                scene,
                panel,
                float(direction[0]),
                float(direction[1]),
                range_m,
                tau,
                sign=float(np.sign(direction[2]) or 1.0),
            )
        )
    return supports


def _solve(scene, config, supports, *, gate):
    coefficients = np.ones(2 * int(scene["K"]), dtype=complex)
    return _wls_position_clock(
        scene,
        config,
        supports,
        coefficients,
        1.0e-3,
        {"remark2_gate_enabled": gate, "wls_enabled": True, "wls_max_nfev": 200},
    )


def test_gate_is_silent_on_a_consistent_panel_set(gate_scene):
    config, scene = gate_scene
    position, clock, diagnostics = _solve(
        scene, config, _consistent_supports(scene), gate=True
    )
    assert diagnostics["remark2_gate_enabled"] is True
    assert diagnostics["remark2_gate_discarded"] == []
    assert np.linalg.norm(position - np.asarray(scene["p_u_true"], float)) < 1.0e-6
    assert abs(clock - float(scene["delta_t_true"])) < 1.0e-15


def test_gate_discards_the_inconsistent_path_and_restores_the_fix(gate_scene):
    """Remark 2: excluding the corrupted path recovers the exact solution."""
    config, scene = gate_scene
    truth = np.asarray(scene["p_u_true"], dtype=float)
    supports = _consistent_supports(scene, clock_offsets_ns=[0.0, 3.0, 0.0])

    _, _, without = _solve(scene, config, supports, gate=False)
    gated_position, gated_clock, with_gate = _solve(scene, config, supports, gate=True)
    ungated_position, _, _ = _solve(scene, config, supports, gate=False)

    assert without.get("remark2_gate_enabled") is False
    assert with_gate["remark2_gate_discarded"] == [1]
    assert np.linalg.norm(gated_position - truth) < 1.0e-6
    assert abs(gated_clock - float(scene["delta_t_true"])) < 1.0e-15
    # The gate has to actually buy something, not merely fire.
    assert np.linalg.norm(gated_position - truth) < np.linalg.norm(
        ungated_position - truth
    )


def test_gate_never_drops_below_the_identifiable_panel_count(gate_scene):
    """(50) needs two panels; a set with no majority must not be gated apart."""
    config, scene = gate_scene
    supports = _consistent_supports(scene, clock_offsets_ns=[-4.0, 0.0, 4.0])
    _, _, diagnostics = _solve(scene, config, supports, gate=True)
    kept = int(scene["K"]) - diagnostics["remark2_gate_num_discarded"]
    assert kept >= 2
