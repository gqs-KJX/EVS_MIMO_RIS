"""Analytic gradient of the exact spherical projection objective.

The exact refinement in ``_project_ris_factor_legacy`` minimises the profiled
residual with L-BFGS-B.  It used to supply no ``jac``, so SciPy spent four
response evaluations per gradient on a 3-D parameter.  ``search_config
["exact_refine_analytic_gradient"]`` (default on) supplies the closed-form
gradient instead.

The optimiser's trajectory now depends on this derivation being right, so the
gradient is pinned against central differences here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.channel_model import generate_scene
from src.config import default_config
from src.geometry import local_geometry_from_position
from src.projections_ris import (
    compressed_exact_response,
    exact_projection_objective_and_gradient,
    local_ris_search_config,
    project_ris_factor,
    scaled_residual,
)

EPS = 1.0e-10


def _panel(panel: int = 0):
    config = default_config()
    config["K"] = 3
    config["seed"] = 20260728
    rng = np.random.default_rng(int(config["seed"]))
    scene = generate_scene(config, rng)
    range_m, elevation, azimuth, _ = local_geometry_from_position(
        np.asarray(scene["p_u_true"], dtype=float),
        np.asarray(scene["ris_centers"][panel], dtype=float),
        np.asarray(scene["rotations"][panel], dtype=float),
    )
    eta = np.array([range_m, elevation, azimuth], dtype=float)
    return scene, config, eta, panel


def _observation(scene, eta, panel, noise=0.05):
    clean = compressed_exact_response(
        eta, scene["Omega"][panel], scene["a_RB"][panel],
        scene["ris_grid"], scene["wavelength"],
    )
    rng = np.random.default_rng(4242)
    return clean + noise * (
        rng.standard_normal(clean.size) + 1j * rng.standard_normal(clean.size)
    )


def _reference_objective(eta, c_tilde, scene, panel, c_norm_sq):
    h_model = compressed_exact_response(
        eta, scene["Omega"][panel], scene["a_RB"][panel],
        scene["ris_grid"], scene["wavelength"],
    )
    value, _ = scaled_residual(c_tilde, h_model, EPS)
    return value / c_norm_sq


@pytest.mark.parametrize("panel", [0, 1, 2])
def test_analytic_objective_matches_the_profiled_residual(panel):
    scene, _, eta, panel = _panel(panel)
    c_tilde = _observation(scene, eta, panel)
    c_norm_sq = float(np.linalg.norm(c_tilde) ** 2) + EPS

    rng = np.random.default_rng(7)
    for _ in range(4):
        probe = eta + rng.standard_normal(3) * np.array([0.05, 0.01, 0.01])
        analytic, _ = exact_projection_objective_and_gradient(
            probe, c_tilde, scene["Omega"][panel], scene["a_RB"][panel],
            scene["ris_grid"], scene["wavelength"], EPS, c_norm_sq,
        )
        reference = _reference_objective(probe, c_tilde, scene, panel, c_norm_sq)
        assert analytic == pytest.approx(reference, rel=1.0e-10, abs=1.0e-14)


@pytest.mark.parametrize("panel", [0, 1, 2])
def test_analytic_gradient_matches_central_differences(panel):
    scene, _, eta, panel = _panel(panel)
    c_tilde = _observation(scene, eta, panel)
    c_norm_sq = float(np.linalg.norm(c_tilde) ** 2) + EPS

    rng = np.random.default_rng(11)
    for _ in range(4):
        probe = eta + rng.standard_normal(3) * np.array([0.05, 0.01, 0.01])
        _, gradient = exact_projection_objective_and_gradient(
            probe, c_tilde, scene["Omega"][panel], scene["a_RB"][panel],
            scene["ris_grid"], scene["wavelength"], EPS, c_norm_sq,
        )
        numerical = np.empty(3)
        for axis in range(3):
            step = 1.0e-6 * max(abs(probe[axis]), 1.0e-3)
            forward, backward = probe.copy(), probe.copy()
            forward[axis] += step
            backward[axis] -= step
            numerical[axis] = (
                _reference_objective(forward, c_tilde, scene, panel, c_norm_sq)
                - _reference_objective(backward, c_tilde, scene, panel, c_norm_sq)
            ) / (2.0 * step)
        scale = max(float(np.abs(numerical).max()), 1.0e-30)
        assert float(np.abs(gradient - numerical).max()) / scale < 1.0e-6


def test_both_gradient_modes_reach_the_same_optimum():
    """The switch must be a cost change, not an answer change."""
    scene, config, eta, panel = _panel(0)
    c_tilde = _observation(scene, eta, panel)
    search = local_ris_search_config(scene, config, panel)
    search["projection_mode"] = "exact"
    search["_coarse_refine_starts"] = [eta + np.array([0.05, 0.005, 0.005])]

    analytic = project_ris_factor(
        c_tilde, scene["Omega"][panel], scene["a_RB"][panel], scene["ris_grid"],
        scene["wavelength"], dict(search, exact_refine_analytic_gradient=True), EPS,
    )
    numerical = project_ris_factor(
        c_tilde, scene["Omega"][panel], scene["a_RB"][panel], scene["ris_grid"],
        scene["wavelength"], dict(search, exact_refine_analytic_gradient=False), EPS,
    )
    # Well inside one mainlobe / one range step of the acquisition itself.
    delta = np.abs(
        np.asarray(analytic["eta_local"]) - np.asarray(numerical["eta_local"])
    )
    assert delta[0] < 1.0e-3
    assert delta[1] < 1.0e-6
    assert delta[2] < 1.0e-6


def _response_and_jacobian_form(eta, c_tilde, scene, panel, c_norm_sq):
    """The pre-adjoint evaluation: build h and its T x 3 Jacobian explicitly.

    This is the form the objective was written in before the four ``Omega``
    products per evaluation were folded down to two by pushing the four scalars
    that are actually used through the adjoint.  It is kept here purely as the
    reference the fast form has to reproduce.
    """
    from src.projections_ris import exact_spherical_response_and_jacobian

    h_vec, jac = exact_spherical_response_and_jacobian(
        eta, scene["Omega"][panel], scene["a_RB"][panel],
        scene["ris_grid"], scene["wavelength"],
    )
    u_value = np.vdot(h_vec, c_tilde)
    v_value = float(np.vdot(h_vec, h_vec).real)
    denominator = v_value + EPS
    g_value = (v_value + 2.0 * EPS) / (denominator * denominator)
    g_prime = -(v_value + 3.0 * EPS) / (denominator**3)
    u_abs_sq = float((u_value * np.conj(u_value)).real)
    objective = float(np.vdot(c_tilde, c_tilde).real) - u_abs_sq * g_value
    du = jac.conj().T @ c_tilde
    dv = 2.0 * np.real(jac.conj().T @ h_vec)
    du_abs_sq = 2.0 * np.real(np.conj(u_value) * du)
    gradient = -(du_abs_sq * g_value + u_abs_sq * g_prime * dv)
    return float(objective / c_norm_sq), np.asarray(gradient / c_norm_sq, dtype=float)


@pytest.mark.parametrize("panel", [0, 1, 2])
def test_adjoint_form_reproduces_the_response_and_jacobian_form(panel):
    from src.projections_ris import omega_adjoint

    scene, _, eta, panel = _panel(panel)
    c_tilde = _observation(scene, eta, panel)
    c_norm_sq = float(np.linalg.norm(c_tilde) ** 2) + EPS
    adjoint_c = omega_adjoint(scene["Omega"][panel], c_tilde)
    assert np.abs(
        adjoint_c - scene["Omega"][panel].conj().T @ c_tilde
    ).max() <= 1e-12 * np.abs(adjoint_c).max()

    rng = np.random.default_rng(909 + panel)
    for _ in range(5):
        probe = eta + rng.normal(scale=[0.05, 0.01, 0.01])
        fast, fast_grad = exact_projection_objective_and_gradient(
            probe, c_tilde, scene["Omega"][panel], scene["a_RB"][panel],
            scene["ris_grid"], scene["wavelength"], EPS, c_norm_sq,
            omega_adjoint_c=adjoint_c,
        )
        reference, reference_grad = _response_and_jacobian_form(
            probe, c_tilde, scene, panel, c_norm_sq
        )
        assert fast == pytest.approx(reference, rel=1e-11, abs=1e-15)
        scale = max(np.abs(reference_grad).max(), 1e-300)
        assert np.abs(fast_grad - reference_grad).max() <= 1e-9 * scale

        # Omitting the precomputed adjoint must give the same numbers.
        without, without_grad = exact_projection_objective_and_gradient(
            probe, c_tilde, scene["Omega"][panel], scene["a_RB"][panel],
            scene["ris_grid"], scene["wavelength"], EPS, c_norm_sq,
        )
        assert without == pytest.approx(fast, rel=1e-13, abs=1e-16)
        assert np.abs(without_grad - fast_grad).max() <= 1e-11 * scale
