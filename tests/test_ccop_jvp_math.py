import copy

import numpy as np
import pytest

from src.ccop_jvp import (
    CommonClockJonesProfiler,
    apply_common_clock_unitary,
    refine_ccop_jvp,
    refine_four_dimensional_jvp_experimental,
)
from src.channel_model import (
    add_awgn,
    channel_components,
    generate_scene,
    synthesize_raw_tensor,
)
from src.ccop_stage1_initializer import refresh_ccop_stage1_jones_anchor
from src.config import default_config
from src.global_vp import (
    _build_global_dictionary,
    build_jones_vp_dictionary,
    extract_stage1_jones_directions,
)


def _problem(mode: str, *, seed: int = 7301, clock_high_s: float = 12.0e-9):
    config = default_config()
    config.update(
        {
            "seed": int(seed),
            "K": 3,
            "M_A": 2,
            "ris_shape": (4, 4),
            "N": 9,
            "P": 5,
            "T": 10 + int(seed % 3),
            "SNR_dB": 15.0,
            "p_u_true": np.array([1.28, 0.31, 0.79]),
            "delta_t_true": 4.0e-9,
            "ris_centers": np.array(
                [
                    [4.20, -2.20, 1.05],
                    [5.10, 2.10, 1.15],
                    [4.80, 0.00, 1.25],
                ]
            ),
            "ue_bounds": np.array(
                [[0.8, 1.8], [-0.2, 0.8], [0.5, 1.2]], dtype=float
            ),
            "delta_t_bounds": np.array([0.0, float(clock_high_s)]),
        }
    )
    config["global_vp"] = dict(config["global_vp"])
    config["global_vp"].update(
        {
            "mode": mode,
            "vp_dictionary_mode": "matrix_free",
            "use_weight": False,
            "use_delay_prior": False,
            "jones_diagonal_loading": 0.0,
            "beta_reg": 0.0,
            "jones_lambda0": 0.7,
            "use_multistart": False,
            "max_iter": 4,
        }
    )
    config["ccop_jvp"] = {
        "clock_fft_size": 2048,
        "clock_abs_tol_objective": 1.0e-12,
        "clock_rel_tol": 1.0e-12,
        "clock_max_intervals": 20000,
        "outer_max_iter": 3,
        "outer_ftol": 1.0e-12,
        "outer_gtol": 1.0e-8,
    }
    rng = np.random.default_rng(config["seed"])
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(components, scene["beta_true"])
    y_noisy, _ = add_awgn(
        y_true,
        config["SNR_dB"],
        rng,
        active_mask=scene.get("evs_observation_mask"),
    )
    init = {
        "A": components["a_EVS"].T.copy(),
        "B": np.empty((scene["P"], scene["K"]), dtype=complex),
        "Q": np.empty((scene["L"], scene["K"]), dtype=complex),
        "C": components["c"].T.copy(),
        "poles": components["poles"].copy(),
        "ris_eta": np.column_stack(
            [
                components["ranges"],
                components["elevations"],
                components["azimuths"],
            ]
        ),
        "gamma": scene["gamma_true"].copy(),
        "eta_pol": scene["eta_true"].copy(),
        "assignment": list(range(scene["K"])),
        "panel_to_column_assignment": list(range(scene["K"])),
        "columns_are_panel_ordered": True,
        "p_u": scene["p_u_true"].copy(),
        "delta_t": float(scene["delta_t_true"]),
    }
    return config, scene, components, y_noisy, init


def _dictionary(position, clock_s, init, scene, config):
    if config["global_vp"]["mode"] == "fixed_pol":
        return _build_global_dictionary(
            np.r_[position, float(clock_s)], init, scene, config
        )[0]
    return build_jones_vp_dictionary(position, clock_s, scene, config)


def test_fixed_position_free_jones_refresh_replaces_only_anchor_and_clock():
    config, scene, _, y_noisy, init = _problem("jones_regularized", seed=7304)
    corrupted = copy.deepcopy(init)
    corrupted["A"] = np.roll(np.asarray(corrupted["A"]), 1, axis=1)
    position_before = np.asarray(corrupted["p_u"], dtype=float).copy()
    poles_before = np.asarray(corrupted["poles"], dtype=complex).copy()
    c_before = np.asarray(corrupted["C"], dtype=complex).copy()

    refreshed = refresh_ccop_stage1_jones_anchor(
        y_noisy, corrupted, scene, config
    )
    coefficients = np.asarray(
        refreshed["stage1_jones_anchor_refresh_coefficients"], dtype=complex
    ).reshape(scene["K"], 2)
    expected_directions = coefficients / np.linalg.norm(
        coefficients, axis=1, keepdims=True
    )

    np.testing.assert_allclose(refreshed["p_u"], position_before, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(refreshed["poles"], poles_before, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(refreshed["C"], c_before, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        extract_stage1_jones_directions(refreshed, scene),
        expected_directions,
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    assert refreshed["stage1_jones_anchor_refresh_clock_certified"]


@pytest.mark.parametrize("mode", ["fixed_pol", "adaptive_jones"])
@pytest.mark.parametrize(
    "seed,offset,clock_s",
    [
        (7301, np.array([0.0, 0.0, 0.0]), 1.0e-9),
        (7302, np.array([0.04, -0.03, 0.02]), 5.5e-9),
        (7303, np.array([-0.05, 0.02, -0.015]), 10.0e-9),
    ],
)
def test_common_clock_unitary_factorization(mode, seed, offset, clock_s):
    config, scene, _, _, init = _problem(mode, seed=seed)
    position = scene["p_u_true"] + offset
    a_zero = _dictionary(position, 0.0, init, scene, config)
    a_clock = _dictionary(position, clock_s, init, scene, config)
    orbit = apply_common_clock_unitary(a_zero, scene, clock_s)
    relative = np.linalg.norm(a_clock - orbit) / np.linalg.norm(a_clock)
    assert relative < 1.0e-11
    recovered = apply_common_clock_unitary(orbit, scene, clock_s, adjoint=True)
    assert np.linalg.norm(recovered - a_zero) / np.linalg.norm(a_zero) < 1.0e-13


@pytest.mark.parametrize("mode", ["fixed_pol", "adaptive_jones"])
@pytest.mark.parametrize("clock_s", [0.0, 2.3e-9, 9.7e-9])
def test_gram_is_clock_invariant(mode, clock_s):
    config, scene, _, _, init = _problem(mode, seed=7311)
    position = scene["p_u_true"] + np.array([0.03, -0.02, 0.01])
    a_zero = _dictionary(position, 0.0, init, scene, config)
    a_clock = _dictionary(position, clock_s, init, scene, config)
    gram_zero = a_zero.conj().T @ a_zero
    gram_clock = a_clock.conj().T @ a_clock
    relative = np.linalg.norm(gram_clock - gram_zero) / np.linalg.norm(gram_zero)
    assert relative < 1.0e-11


@pytest.mark.parametrize("mode", ["jones_regularized", "jones_free"])
def test_explicit_and_profiled_jones_elimination_are_equivalent(mode):
    config, scene, _, y_noisy, init = _problem(mode, seed=7321)
    position = scene["p_u_true"] + np.array([0.025, -0.018, 0.012])
    clock_s = 6.2e-9
    profiler = CommonClockJonesProfiler(y_noisy, init, scene, config)
    point = profiler.evaluate_clock(position, clock_s)
    orbit = profiler._position_orbit(position)
    dictionary = build_jones_vp_dictionary(position, clock_s, scene, config)
    gram = dictionary.conj().T @ dictionary
    rhs = dictionary.conj().T @ y_noisy.reshape(-1)
    normal = gram + orbit["regularizer"]
    x_explicit = np.linalg.solve(normal, rhs)
    residual = y_noisy.reshape(-1) - dictionary @ x_explicit
    raw_residual = float(np.vdot(residual, residual).real)
    regularized = raw_residual + float(
        np.vdot(x_explicit, orbit["regularizer"] @ x_explicit).real
    )
    profile_formula = float(
        np.vdot(y_noisy.reshape(-1), y_noisy.reshape(-1)).real
        - np.vdot(rhs, np.linalg.solve(normal, rhs)).real
    )
    assert np.linalg.norm(point["x_hat"] - x_explicit) / np.linalg.norm(x_explicit) < 1.0e-9
    assert abs(point["raw_residual_unscaled"] - raw_residual) / max(raw_residual, 1.0) < 1.0e-9
    assert abs(regularized - profile_formula) / max(abs(regularized), 1.0) < 1.0e-9
    assert abs(point["total_objective"] - regularized / y_noisy.size) < 1.0e-11


def test_clock_trigonometric_fft_derivatives_and_certificate_match_references():
    from scipy.optimize import brentq

    config, scene, _, y_noisy, init = _problem("jones_regularized", seed=7331)
    profiler = CommonClockJonesProfiler(y_noisy, init, scene, config)
    position = scene["p_u_true"] + np.array([0.035, -0.025, 0.018])
    orbit = profiler._position_orbit(position)
    clock_s = 5.1e-9
    point = profiler.evaluate_clock(position, clock_s, orbit=orbit)
    direct_score = float(
        np.vdot(point["b"], np.linalg.solve(orbit["normal"], point["b"])).real
    )
    assert abs(point["score"] - direct_score) / max(abs(direct_score), 1.0) < 1.0e-11

    theta = point["theta"]
    step = 1.0e-6
    f_minus = profiler._score_and_derivatives(theta - step, orbit["trig_coeff"])[0]
    f_plus = profiler._score_and_derivatives(theta + step, orbit["trig_coeff"])[0]
    first_fd = (f_plus - f_minus) / (2.0 * step)
    second_fd = (f_plus - 2.0 * point["score"] + f_minus) / (step**2)
    assert abs(first_fd - point["score_first_theta"]) / max(abs(first_fd), 1.0) < 2.0e-6
    assert abs(second_fd - point["score_second_theta"]) / max(abs(second_fd), 1.0) < 2.0e-4

    fft_size = 512
    fft_scores = np.sum(
        np.abs(np.fft.ifft(orbit["whitened_u"], n=fft_size, axis=1) * fft_size) ** 2,
        axis=0,
    ).real
    for index in (0, 1, 7, 31, 127, 255):
        theta_grid = 2.0 * np.pi * index / fft_size
        trig_score = profiler._score_and_derivatives(
            theta_grid, orbit["trig_coeff"]
        )[0]
        assert abs(fft_scores[index] - trig_score) / max(abs(trig_score), 1.0) < 1.0e-10

    theta_grid = np.linspace(profiler.theta_bounds[0], profiler.theta_bounds[1], 20001)
    values = np.asarray(
        [profiler._score_and_derivatives(value, orbit["trig_coeff"])[0] for value in theta_grid]
    )
    derivatives = np.asarray(
        [profiler._score_and_derivatives(value, orbit["trig_coeff"])[1] for value in theta_grid]
    )
    stationary = [float(theta_grid[0]), float(theta_grid[-1])]
    for left, right, f_left, f_right in zip(
        theta_grid[:-1], theta_grid[1:], derivatives[:-1], derivatives[1:]
    ):
        if f_left == 0.0:
            stationary.append(float(left))
        elif f_left * f_right < 0.0:
            stationary.append(
                float(
                    brentq(
                        lambda value: profiler._score_and_derivatives(
                            value, orbit["trig_coeff"]
                        )[1],
                        float(left),
                        float(right),
                    )
                )
            )
    reference_score = max(
        profiler._score_and_derivatives(value, orbit["trig_coeff"])[0]
        for value in stationary
    )
    profile = profiler.profile_clock(position)
    assert profile["clock_certified"]
    assert profile["score"] + 1.0e-9 >= float(np.max(values))
    assert abs(profile["score"] - reference_score) / max(abs(reference_score), 1.0) < 1.0e-10
    assert profile["clock_certificate_gap_score"] <= profile["clock_certificate_tolerance_score"]


@pytest.mark.parametrize("mode", ["fixed_pol", "jones_regularized"])
def test_profiled_position_gradient_matches_high_accuracy_finite_difference(mode):
    config, scene, _, y_noisy, init = _problem(mode, seed=7341)
    profiler = CommonClockJonesProfiler(y_noisy, init, scene, config)
    position = scene["p_u_true"] + np.array([0.02, -0.015, 0.011])
    profile = profiler.profile_clock(position)
    assert profile["gradient_reliable"]
    analytic = np.asarray(profile["gradient_p"], dtype=float)
    finite = np.empty(3, dtype=float)
    step = 1.0e-5
    for dim in range(3):
        direction = np.zeros(3, dtype=float)
        direction[dim] = step
        plus = profiler.profile_clock(position + direction)
        minus = profiler.profile_clock(position - direction)
        finite[dim] = (plus["total_objective"] - minus["total_objective"]) / (2.0 * step)
    relative = np.linalg.norm(analytic - finite) / max(
        np.linalg.norm(analytic), np.linalg.norm(finite), 1.0e-12
    )
    assert relative < 1.0e-5


def test_near_clock_branch_switch_is_flagged_and_uses_outer_safeguard():
    config, scene, _, y_noisy, init = _problem("jones_regularized", seed=7351)
    probe = CommonClockJonesProfiler(y_noisy, init, scene, config)
    initial = probe.profile_clock(scene["p_u_true"])
    assert np.isfinite(initial["clock_fft_peak_gap_objective"])
    config = copy.deepcopy(config)
    config["ccop_jvp"]["clock_branch_switch_abs_gap_objective"] = float(
        initial["clock_fft_peak_gap_objective"] * 1.01 + 1.0e-15
    )
    config["ccop_jvp"]["outer_safeguard_max_iter"] = 2
    result = refine_ccop_jvp(y_noisy, init, scene, config, incumbent=None)
    assert result["clock_branch_ambiguous_seen"]
    assert result["outer_branch_safeguard_used"]
    assert "Powell" in result["optimizer"]["method"]


def test_clock_distance_reparameterization_preserves_exact_objective():
    config, scene, _, y_noisy, init = _problem("jones_regularized", seed=7361)
    profiler = CommonClockJonesProfiler(y_noisy, init, scene, config)
    result = refine_four_dimensional_jvp_experimental(
        y_noisy,
        init,
        scene,
        config,
        clock_coordinate="distance_m",
        max_iter=0,
        max_evaluations=1,
    )
    reference = profiler.evaluate_clock(result["p_u"], result["delta_t"])
    assert result["clock_coordinate"] == "distance_m"
    assert np.isclose(
        result["optimized_coordinate"][3],
        scene["c0"] * result["delta_t"],
        rtol=0.0,
        atol=1.0e-14,
    )
    assert abs(result["total_objective_final"] - reference["total_objective"]) < 1.0e-12


def test_incumbent_ccop_selector_is_monotonic_for_exact_same_objective():
    config, scene, _, y_noisy, init = _problem("jones_regularized", seed=7371)
    profiler = CommonClockJonesProfiler(y_noisy, init, scene, config)
    incumbent_point = profiler.evaluate_clock(
        scene["p_u_true"] + np.array([0.04, -0.03, 0.02]), 7.0e-9
    )
    incumbent = {
        "p_u": incumbent_point["p_u"],
        "delta_t": incumbent_point["delta_t"],
        "total_objective_final": incumbent_point["total_objective"],
    }
    result = refine_ccop_jvp(y_noisy, init, scene, config, incumbent=incumbent)
    assert result["incumbent_non_degradation"]
    assert result["total_objective_final"] <= incumbent_point["total_objective"] + 1.0e-12
    assert result["incumbent_profile_relative_error_if_same_clock"] < 1.0e-12
