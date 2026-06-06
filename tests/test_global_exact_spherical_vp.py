import copy

import numpy as np
import pytest

from src.channel_model import channel_components, generate_scene, synthesize_raw_tensor
from src.config import default_config
from src.estimators import run_proposed_estimator
from src.global_vp import (
    _build_global_dictionary,
    _get_panel_ordered_stage1_factors,
    _objective_weight_from_config,
    _solve_beta_vp,
    _vp_objective_and_grad,
    global_exact_spherical_vp_refinement,
)
from src.metrics import position_rmse, relative_nmse
from src.utils import scipy_is_available


def _small_config(k_paths: int = 2, beta_reg: float = 0.0) -> dict:
    config = default_config()
    config.update(
        {
            "seed": 90210,
            "K": k_paths,
            "M_A": 2,
            "ris_shape": (4, 4),
            "N": 7,
            "P": 4,
            "T": 10,
            "delta_t_true": 2.0e-9,
            "p_u_true": np.array([1.35, 0.25, 0.82]),
            "ris_centers": np.array(
                [
                    [4.20, -2.20, 1.05],
                    [5.10, 2.10, 1.15],
                    [4.80, 0.00, 1.25],
                ]
            )[:k_paths],
            "ue_bounds": np.array(
                [
                    [0.50, 2.20],
                    [-0.80, 0.90],
                    [0.45, 1.25],
                ]
            ),
            "delta_t_bounds": np.array([0.0, 6.0e-9]),
            "stage2_mode": "none",
            "final_refinement_method": "global_exact_spherical_vp",
        }
    )
    config["global_vp"] = dict(config["global_vp"])
    config["global_vp"].update(
        {
            "max_iter": 60,
            "beta_reg": beta_reg,
            "use_multistart": False,
            "overwrite_factor_keys": False,
        }
    )
    return config


def _scene_truth_and_init(beta_reg: float = 0.0) -> tuple[dict, dict, dict, np.ndarray, dict]:
    config = _small_config(beta_reg=beta_reg)
    rng = np.random.default_rng(config["seed"])
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_raw = synthesize_raw_tensor(components, scene["beta_true"])
    init_estimate = _init_from_truth(scene, components)
    return config, scene, components, y_raw, init_estimate


def _init_from_truth(scene: dict, components: dict) -> dict:
    return {
        "A": components["a_EVS"].T.copy(),
        "D": components["d"].T.copy(),
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
        "Z_hat": np.zeros((scene["I"], scene["P"], scene["L"], scene["T"]), dtype=complex),
        "beta_z": np.ones(scene["K"], dtype=complex),
    }


def _finite_difference_gradient(
    xi: np.ndarray,
    y_vec: np.ndarray,
    init_estimate: dict,
    scene: dict,
    config: dict,
) -> np.ndarray:
    eps = np.array([1.0e-5, 1.0e-5, 1.0e-5, 1.0e-12])
    fd_grad = np.empty(4, dtype=float)
    for dim, step in enumerate(eps):
        delta = np.zeros(4, dtype=float)
        delta[dim] = step
        f_plus, _ = _vp_objective_and_grad(
            xi + delta, y_vec, init_estimate, scene, config
        )
        f_minus, _ = _vp_objective_and_grad(
            xi - delta, y_vec, init_estimate, scene, config
        )
        fd_grad[dim] = (f_plus - f_minus) / (2.0 * step)
    return fd_grad


def test_true_dictionary_reconstruction_with_polarization_basis():
    config, scene, components, y_raw, init_estimate = _scene_truth_and_init()
    xi = np.r_[scene["p_u_true"], scene["delta_t_true"]]

    for evs_mode, expected_atoms in (
        ("legacy_or_full_polarization", scene["K"]),
        ("linear_polarization_basis", 2 * scene["K"]),
    ):
        mode_config = copy.deepcopy(config)
        mode_config["global_vp"]["evs_mode"] = evs_mode
        phi, aux = _build_global_dictionary(xi, init_estimate, scene, mode_config)
        beta = _solve_beta_vp(phi, y_raw.reshape(-1), None, 0.0, 1.0 / y_raw.size)
        y_hat = (phi @ beta).reshape(y_raw.shape)

        assert phi.shape == (scene["I"] * scene["N"] * scene["T"], expected_atoms)
        assert aux["D"].shape == (scene["N"], scene["K"])
        assert aux["C"].shape == (scene["T"], scene["K"])
        assert aux["tau"].shape == (scene["K"],)
        np.testing.assert_allclose(aux["tau"], components["taus"], rtol=1.0e-12, atol=1.0e-14)
        assert relative_nmse(y_hat, y_raw) < 1.0e-24


def test_analytic_gradient_matches_finite_difference():
    config, scene, _, y_raw, init_estimate = _scene_truth_and_init()
    config["global_vp"].update(
        {
            "solver": "lbfgsb_reduced",
            "evs_mode": "linear_polarization_basis",
            "use_delay_prior": True,
        }
    )
    xi = np.r_[scene["p_u_true"] + np.array([0.06, -0.04, 0.03]), scene["delta_t_true"] + 1.5e-10]
    y_vec = y_raw.reshape(-1)

    _, analytic = _vp_objective_and_grad(xi, y_vec, init_estimate, scene, config)
    fd_grad = _finite_difference_gradient(xi, y_vec, init_estimate, scene, config)

    scale = max(1.0, np.linalg.norm(analytic), np.linalg.norm(fd_grad))
    assert np.linalg.norm(analytic - fd_grad) / scale < 1.0e-4


def test_noiseless_recovery_from_perturbed_initialization():
    config, scene, _, y_raw, init_estimate = _scene_truth_and_init()
    perturbed = copy.deepcopy(init_estimate)
    initial_p = scene["p_u_true"] + np.array([0.10, -0.08, 0.04])
    initial_dt = scene["delta_t_true"] + 2.0e-10
    perturbed["p_u"] = initial_p
    perturbed["delta_t"] = initial_dt

    refined = global_exact_spherical_vp_refinement(y_raw, perturbed, scene, config)

    initial_error = np.linalg.norm(initial_p - scene["p_u_true"])
    final_error = np.linalg.norm(refined["p_u"] - scene["p_u_true"])
    assert refined["raw_residual_final"] <= refined["raw_residual_initial"] + 1.0e-12
    assert refined["raw_residual_final"] < 1.0e-10
    assert final_error < initial_error


def test_regularized_beta_objective_and_gradient_consistency():
    config, scene, _, y_raw, init_estimate = _scene_truth_and_init(beta_reg=1.0e-3)
    config["global_vp"].update(
        {
            "solver": "lbfgsb_reduced",
            "evs_mode": "linear_polarization_basis",
            "use_delay_prior": True,
        }
    )
    xi = np.r_[scene["p_u_true"] + np.array([0.05, 0.03, -0.02]), scene["delta_t_true"] - 1.0e-10]
    y_vec = y_raw.reshape(-1)

    objective, analytic = _vp_objective_and_grad(xi, y_vec, init_estimate, scene, config)
    phi, _ = _build_global_dictionary(xi, init_estimate, scene, config)
    weight = _objective_weight_from_config(config, y_vec.size)
    objective_scale = 1.0 / y_vec.size
    beta = _solve_beta_vp(
        phi, y_vec, weight, config["global_vp"]["beta_reg"], objective_scale
    )
    residual = y_vec - phi @ beta
    expected = float(objective_scale * np.real(np.vdot(residual, residual)))
    expected += float(config["global_vp"]["beta_reg"] * np.real(np.vdot(beta, beta)))
    _, aux = _build_global_dictionary(xi, init_estimate, scene, config)
    tau_err = aux["tau"] - np.array(init_estimate["poles"], dtype=complex)
    tau_stage1 = np.array(
        [
            ((-np.angle(pole)) % (2.0 * np.pi)) / (2.0 * np.pi * scene["delta_f"])
            for pole in init_estimate["poles"]
        ]
    )
    tau_err = aux["tau"] - tau_stage1
    if config["global_vp"]["use_delay_prior"]:
        expected += float(
            config["global_vp"]["delay_prior_weight"]
            * np.sum((tau_err / config["global_vp"]["delay_prior_sigma_s"]) ** 2)
        )
    np.testing.assert_allclose(objective, expected, rtol=1.0e-12, atol=1.0e-12)

    fd_grad = _finite_difference_gradient(xi, y_vec, init_estimate, scene, config)
    scale = max(1.0, np.linalg.norm(analytic), np.linalg.norm(fd_grad))
    assert np.linalg.norm(analytic - fd_grad) / scale < 1.0e-4


def test_nontrivial_assignment_restores_physical_order():
    config = _small_config(k_paths=3)
    rng = np.random.default_rng(config["seed"])
    scene = generate_scene(config, rng)
    i_dim, t_dim, k_paths = scene["I"], scene["T"], scene["K"]
    a_phys = (
        np.arange(i_dim * k_paths).reshape(i_dim, k_paths)
        + 1j * np.arange(i_dim * k_paths).reshape(i_dim, k_paths)
    )
    c_phys = np.arange(t_dim * k_paths).reshape(t_dim, k_paths).astype(complex)
    poles_phys = np.exp(-1j * np.array([0.1, 0.2, 0.3]))
    ris_eta_phys = np.array([[3.0, 0.1, -0.2], [4.0, 0.0, 0.3], [5.0, -0.1, 0.7]])
    column_to_panel = [2, 0, 1]
    panel_to_column = [1, 2, 0]
    raw_order = column_to_panel
    init_estimate = {
        "A": a_phys[:, raw_order],
        "C": c_phys[:, raw_order],
        "poles": poles_phys[raw_order],
        "ris_eta": ris_eta_phys[raw_order],
        "assignment": column_to_panel,
        "panel_to_column_assignment": panel_to_column,
        "columns_are_panel_ordered": False,
    }

    ordered = _get_panel_ordered_stage1_factors(init_estimate, scene)

    np.testing.assert_allclose(ordered["A_phys"], a_phys)
    np.testing.assert_allclose(ordered["C_phys"], c_phys)
    np.testing.assert_allclose(ordered["poles_phys"], poles_phys)
    np.testing.assert_allclose(ordered["ris_eta_phys"], ris_eta_phys)
    assert ordered["global_vp_used_panel_to_column"] is True
    assert ordered["global_vp_panel_to_column"] == panel_to_column


def test_low_snr_guarded_basis_prior_reduces_stage1_delay_drift():
    config, scene, components, y_raw, init_estimate = _scene_truth_and_init()
    rng = np.random.default_rng(1234)
    signal_power = np.mean(np.abs(y_raw) ** 2)
    noise_scale = np.sqrt(signal_power / 2.0)
    noise = noise_scale * (
        rng.standard_normal(y_raw.shape) + 1j * rng.standard_normal(y_raw.shape)
    )
    y_noisy = y_raw + noise
    fixed_config = copy.deepcopy(config)
    fixed_config["global_vp"].update(
        {
            "solver": "lbfgsb_reduced",
            "evs_mode": "fixed_stage1_A",
            "use_delay_prior": False,
            "use_trust_region": False,
            "max_iter": 25,
        }
    )
    guarded_config = copy.deepcopy(config)
    guarded_config["global_vp"].update(
        {
            "solver": "lbfgsb_reduced",
            "evs_mode": "linear_polarization_basis",
            "use_delay_prior": True,
            "delay_prior_weight": 100.0,
            "use_trust_region": True,
            "position_trust_radius_m": 0.2,
            "clock_trust_radius_s": 2.0e-10,
            "max_iter": 25,
        }
    )

    fixed = global_exact_spherical_vp_refinement(
        y_noisy, copy.deepcopy(init_estimate), scene, fixed_config
    )
    guarded = global_exact_spherical_vp_refinement(
        y_noisy, copy.deepcopy(init_estimate), scene, guarded_config
    )
    tau_stage1 = np.asarray(init_estimate["poles"], dtype=complex)
    tau_stage1 = np.array(
        [
            ((-np.angle(pole)) % (2.0 * np.pi)) / (2.0 * np.pi * scene["delta_f"])
            for pole in tau_stage1
        ]
    )
    fixed_tau_drift = np.linalg.norm(fixed["tau_after_global_vp"] - tau_stage1)
    guarded_tau_drift = np.linalg.norm(guarded["tau_after_global_vp"] - tau_stage1)
    fixed_range_drift = np.linalg.norm(fixed["components"]["ranges"] - components["ranges"])
    guarded_range_drift = np.linalg.norm(guarded["components"]["ranges"] - components["ranges"])

    assert guarded["raw_residual_final"] <= 1.15 * fixed["raw_residual_final"]
    assert guarded_tau_drift <= fixed_tau_drift + 1.0e-15
    assert guarded_range_drift <= fixed_range_drift + 1.0e-12


def test_default_least_squares_reproduces_old_direct_stage1_vp_performance():
    if not scipy_is_available():
        pytest.skip("scipy.optimize.least_squares is required for this regression")
    from src.main_single_proposed import _run_single_pipeline

    config = default_config()
    config["stage2_mode"] = "none"
    config["stage1_ris_geometry_mode"] = "exact_projection"
    config["ris_search"]["num_exact_refine_starts"] = 3
    config["final_refinement_method"] = "global_exact_spherical_vp"
    config["global_vp"]["solver"] = "least_squares"
    results = _run_single_pipeline(config, use_structured=True)

    final = results["final"]
    pos_error = position_rmse(final["p_u"], results["scene"]["p_u_true"])
    y_nmse = relative_nmse(final["Y_hat"], results["Y_true"])

    assert final["optimizer"]["method"] == "scipy.optimize.least_squares"
    assert final["global_vp_solver"] == "least_squares"
    assert final["stage2_mode"] == "none"
    assert pos_error < 2.0e-4
    assert y_nmse < 2.0e-5


def test_pipeline_default_skips_legacy_structured_refinement(monkeypatch):
    import src.estimators as estimators

    calls = {"structured": 0, "global_vp": 0}
    init_estimate = {
        "A": np.ones((1, 1), dtype=complex),
        "poles": np.ones(1, dtype=complex),
        "ris_eta": np.zeros((1, 3)),
        "C": np.ones((1, 1), dtype=complex),
        "Z_hat": np.zeros((1, 1, 1, 1), dtype=complex),
    }

    def fake_initialize(z_tensor, scene, config):
        return copy.deepcopy(init_estimate)

    def fake_structured(*args, **kwargs):
        calls["structured"] += 1
        raise AssertionError("legacy structured_refinement should not run by default")

    def fake_global_vp(y_raw, estimate, scene, config):
        calls["global_vp"] += 1
        final = copy.deepcopy(estimate)
        final.update(
            {
                "p_u": np.zeros(3),
                "delta_t": 0.0,
                "components": {"taus": np.zeros(1), "ranges": np.zeros(1)},
                "Y_hat": y_raw.copy(),
                "final_refinement_method": "global_exact_spherical_vp",
            }
        )
        return final

    monkeypatch.setattr(estimators, "initialize_from_hankel", fake_initialize)
    monkeypatch.setattr(estimators, "structured_refinement", fake_structured)
    monkeypatch.setattr(estimators, "global_exact_spherical_vp_refinement", fake_global_vp)

    config = default_config()
    out = run_proposed_estimator(
        np.zeros((1, 1, 1), dtype=complex),
        np.zeros((1, 1, 1, 1), dtype=complex),
        {"K": 1},
        config,
    )

    assert calls["structured"] == 0
    assert calls["global_vp"] == 1
    assert out["final"]["stage2_mode"] == "none"
    assert out["final"]["final_refinement_method"] == "global_exact_spherical_vp"


def test_legacy_structured_refinement_mode_still_runs(monkeypatch):
    import src.estimators as estimators

    calls = {"structured": 0}
    init_estimate = {
        "A": np.ones((1, 1), dtype=complex),
        "poles": np.ones(1, dtype=complex),
        "ris_eta": np.zeros((1, 3)),
        "C": np.ones((1, 1), dtype=complex),
        "Z_hat": np.zeros((1, 1, 1, 1), dtype=complex),
    }

    def fake_initialize(z_tensor, scene, config):
        return copy.deepcopy(init_estimate)

    def fake_structured(z_tensor, scene, config, estimate):
        calls["structured"] += 1
        estimate = copy.deepcopy(estimate)
        estimate["legacy_stage2_ran"] = True
        return estimate, {"updates": [{"legacy": True}], "ris_projection_total_s": 0.0}

    monkeypatch.setattr(estimators, "initialize_from_hankel", fake_initialize)
    monkeypatch.setattr(estimators, "structured_refinement", fake_structured)

    config = default_config()
    config["stage2_mode"] = "full_legacy"
    config["final_refinement_method"] = "none"
    out = run_proposed_estimator(
        np.zeros((1, 1, 1), dtype=complex),
        np.zeros((1, 1, 1, 1), dtype=complex),
        {"K": 1},
        config,
    )

    assert calls["structured"] == 1
    assert out["stage2"]["legacy_stage2_ran"] is True
    assert out["final"]["stage2_mode"] == "full_legacy"
    assert out["final"]["final_refinement_method"] == "none"
