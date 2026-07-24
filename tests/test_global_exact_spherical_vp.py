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
from src.estimators import run_proposed_estimator
from src.global_vp import (
    _as_path_vector,
    _build_global_dictionary,
    _get_panel_ordered_stage1_factors,
    _jones_regularizer,
    _objective_weight_from_config,
    _solve_beta_vp,
    _vp_objective_parts,
    _vp_objective_and_grad,
    build_jones_vp_dictionary,
    global_exact_spherical_vp_refinement,
)
from src.metrics import position_rmse, relative_nmse
from src.utils import scipy_is_available
from src.main_single_proposed import _fmt_vector, run_single_proposed_diagnostic


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
            "mode": "jones_regularized",
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

    for mode, expected_atoms in (
        ("fixed_pol", scene["K"]),
        ("jones_free", 2 * scene["K"]),
    ):
        mode_config = copy.deepcopy(config)
        mode_config["global_vp"]["mode"] = mode
        phi, aux = _build_global_dictionary(xi, init_estimate, scene, mode_config)
        beta = _solve_beta_vp(phi, y_raw.reshape(-1), None, 0.0, 1.0 / y_raw.size)
        y_hat = (phi @ beta).reshape(y_raw.shape)

        assert phi.shape == (scene["I"] * scene["N"] * scene["T"], expected_atoms)
        assert aux["D"].shape == (scene["N"], scene["K"])
        assert aux["C"].shape == (scene["T"], scene["K"])
        assert aux["tau"].shape == (scene["K"],)
        np.testing.assert_allclose(aux["tau"], components["taus"], rtol=1.0e-12, atol=1.0e-14)
        assert relative_nmse(y_hat, y_raw) < 1.0e-24


def test_jones_dictionary_reproduces_fixed_pol_synthesis():
    config, scene, components, y_raw, _ = _scene_truth_and_init()
    psi = build_jones_vp_dictionary(
        scene["p_u_true"], scene["delta_t_true"], scene, config
    )
    x = np.empty(2 * scene["K"], dtype=complex)
    for k in range(scene["K"]):
        e = np.array(
            [
                np.sin(scene["gamma_true"][k]) * np.exp(1j * scene["eta_true"][k]),
                np.cos(scene["gamma_true"][k]),
            ],
            dtype=complex,
        )
        x[2 * k : 2 * k + 2] = scene["beta_true"][k] * e
    y_hat = (psi @ x).reshape(y_raw.shape)
    np.testing.assert_allclose(y_hat, y_raw, rtol=1.0e-12, atol=1.0e-12)


def test_analytic_gradient_matches_finite_difference():
    config, scene, _, y_raw, init_estimate = _scene_truth_and_init()
    config["global_vp"].update(
        {
            "mode": "fixed_pol",
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


def test_fixed_pol_mode_matches_legacy_fixed_pol_objective():
    config, scene, _, y_raw, init_estimate = _scene_truth_and_init()
    xi = np.r_[scene["p_u_true"], scene["delta_t_true"]]
    fixed_config = copy.deepcopy(config)
    fixed_config["global_vp"]["mode"] = "fixed_pol"
    phi, _ = _build_global_dictionary(xi, init_estimate, scene, fixed_config)
    beta = _solve_beta_vp(phi, y_raw.reshape(-1), None, 0.0, 1.0 / y_raw.size)
    residual = y_raw.reshape(-1) - phi @ beta
    fixed_objective = float(np.vdot(residual, residual).real / y_raw.size)

    refined = global_exact_spherical_vp_refinement(y_raw, init_estimate, scene, fixed_config)
    assert refined["vp_mode"] == "fixed_pol"
    assert refined["linear_nuisance_dim"] == scene["K"]
    np.testing.assert_allclose(refined["raw_objective_initial"], fixed_objective, atol=1.0e-20)


def test_jones_modes_report_four_nonlinear_and_2k_linear_dimensions():
    config, scene, _, y_raw, init_estimate = _scene_truth_and_init()
    for mode in ("jones_regularized", "jones_free"):
        mode_config = copy.deepcopy(config)
        mode_config["global_vp"]["mode"] = mode
        mode_config["global_vp"]["max_iter"] = 1
        refined = global_exact_spherical_vp_refinement(
            y_raw, init_estimate, scene, mode_config
        )
        assert refined["vp_mode"] == mode
        assert refined["nonlinear_dim"] == 4
        assert refined["linear_nuisance_dim"] == 2 * scene["K"]
        assert refined["x_hat"].shape == (2 * scene["K"],)


@pytest.mark.parametrize(
    "value",
    [
        0.5,
        np.array(0.5),
        np.array([0.5]),
        np.array([[0.5]]),
        [0.5, 0.6],
        np.array([[0.5, 0.6]]),
    ],
)
def test_as_path_vector_accepts_scalar_like_and_per_path_shapes(value):
    expected = (
        np.array([0.5, 0.5])
        if np.asarray(value).size == 1
        else np.array([0.5, 0.6])
    )
    np.testing.assert_allclose(_as_path_vector(value, 2, name="test_value"), expected)


@pytest.mark.parametrize("value", [None, []])
def test_as_path_vector_uses_default_for_none_or_empty(value):
    np.testing.assert_allclose(
        _as_path_vector(value, 2, name="test_value", default=0.7),
        [0.7, 0.7],
    )


def test_as_path_vector_rejects_wrong_size():
    with pytest.raises(ValueError, match="test_value.*exactly 2 path values"):
        _as_path_vector([0.5, 0.6, 0.7], 2, name="test_value")


def test_adaptive_jones_regularizer_accepts_k1_lambda_matrix():
    scene = {"K": 1}
    config = default_config()
    config["global_vp"] = dict(config["global_vp"])
    config["global_vp"]["mode"] = "adaptive_jones"
    init_estimate = {"jones_lambda_per_path": np.array([[0.5]])}

    _, _, lambda_path, _, _ = _jones_regularizer(init_estimate, scene, config)

    assert lambda_path.shape == (1,)
    np.testing.assert_allclose(lambda_path, [0.5])


def test_k1_lambda_jones_formats_as_one_entry_list():
    assert _fmt_vector(np.array([[0.5]])) == "[0.5000]"


def test_large_jones_lambda_degenerates_to_fixed_pol_objective():
    config, scene, _, y_raw, init_estimate = _scene_truth_and_init()
    xi = np.r_[scene["p_u_true"], scene["delta_t_true"]]
    fixed_config = copy.deepcopy(config)
    fixed_config["global_vp"]["mode"] = "fixed_pol"
    fixed_parts = _vp_objective_parts(xi, y_raw.reshape(-1), init_estimate, scene, fixed_config)

    jones_config = copy.deepcopy(config)
    jones_config["global_vp"].update(
        {
            "mode": "jones_regularized",
            "jones_lambda_max": 1.0e8,
            "jones_diagonal_loading": 0.0,
        }
    )
    init_strong = copy.deepcopy(init_estimate)
    init_strong["jones_lambda_per_path"] = np.full(scene["K"], 1.0e4)
    jones_parts = _vp_objective_parts(
        xi, y_raw.reshape(-1), init_strong, scene, jones_config
    )
    np.testing.assert_allclose(
        jones_parts["raw_objective"], fixed_parts["raw_objective"], rtol=1.0e-8, atol=1.0e-10
    )


def test_adaptive_jones_leakage_guard_selects_fixed_pol(monkeypatch):
    import src.global_vp as global_vp

    scene = {"K": 1, "I": 1, "N": 1, "T": 4}
    y_raw = np.ones((1, 1, 4), dtype=complex)
    config = default_config()
    config["global_vp"] = dict(config["global_vp"])
    config["global_vp"].update(
        {
            "mode": "adaptive_jones",
            "jones_leakage_threshold": 0.25,
            "enable_z_rescue_multistart": False,
        }
    )

    def fake_fixed(*args, **kwargs):
        return {
            "p_u": np.zeros(3),
            "delta_t": 0.0,
            "raw_objective_final": 1.0,
            "raw_objective": 1.0,
            "Y_hat": y_raw.copy(),
            "beta_raw": np.ones(1, dtype=complex),
            "linear_nuisance_dim": 1,
            "nonlinear_dim": 6,
        }

    def fake_jones(*args, **kwargs):
        return {
            "p_u": np.zeros(3),
            "delta_t": 0.0,
            "raw_objective_final": 0.9999,
            "raw_objective": 0.9999,
            "Y_hat": y_raw.copy(),
            "trace_H": 2.0,
            "jones_leakage_per_path": np.array([0.9]),
            "linear_nuisance_dim": 2,
            "nonlinear_dim": 4,
        }

    monkeypatch.setattr(global_vp, "_global_exact_spherical_vp_refinement_least_squares", fake_fixed)
    monkeypatch.setattr(global_vp, "_global_exact_spherical_vp_refinement_lbfgsb_reduced", fake_jones)
    monkeypatch.setattr(global_vp, "_adaptive_jones_lambdas", lambda *args, **kwargs: (np.array([1.0e-6]), np.array([1.0e8])))

    result = global_vp.global_exact_spherical_vp_refinement(y_raw, {}, scene, config)
    assert result["selected_vp_family_branch"] == "fixed_pol_anchor"
    assert result["jones_leakage_guard_triggered"] is True
    assert result["lambda_jones_per_path"][0] >= 1.0e8


def test_adaptive_jones_selects_jones_when_score_is_better(monkeypatch):
    import src.global_vp as global_vp

    scene = {"K": 1, "I": 1, "N": 1, "T": 8}
    y_raw = np.ones((1, 1, 8), dtype=complex)
    config = default_config()
    config["global_vp"] = dict(config["global_vp"])
    config["global_vp"]["mode"] = "adaptive_jones"
    config["global_vp"]["enable_z_rescue_multistart"] = False

    def fake_fixed(*args, **kwargs):
        return {
            "p_u": np.zeros(3),
            "delta_t": 0.0,
            "raw_objective_final": 10.0,
            "raw_objective": 10.0,
            "Y_hat": y_raw.copy(),
            "beta_raw": np.ones(1, dtype=complex),
            "linear_nuisance_dim": 1,
            "nonlinear_dim": 6,
        }

    def fake_jones(*args, **kwargs):
        return {
            "p_u": np.zeros(3),
            "delta_t": 0.0,
            "raw_objective_final": 1.0,
            "raw_objective": 1.0,
            "Y_hat": y_raw.copy(),
            "trace_H": 1.2,
            "jones_leakage_per_path": np.array([0.01]),
            "linear_nuisance_dim": 2,
            "nonlinear_dim": 4,
            "x_hat": np.ones(2, dtype=complex),
        }

    monkeypatch.setattr(global_vp, "_global_exact_spherical_vp_refinement_least_squares", fake_fixed)
    monkeypatch.setattr(global_vp, "_global_exact_spherical_vp_refinement_lbfgsb_reduced", fake_jones)
    monkeypatch.setattr(global_vp, "_adaptive_jones_lambdas", lambda *args, **kwargs: (np.array([10.0]), np.array([0.1])))

    result = global_vp.global_exact_spherical_vp_refinement(y_raw, {}, scene, config)
    assert result["selected_vp_family_branch"] == "adaptive_jones"
    assert result["jones_score"] < result["fixed_pol_score"]


def test_adaptive_jones_skips_when_fixed_anchor_reaches_known_noise_floor(monkeypatch):
    import src.global_vp as global_vp

    scene = {"K": 1, "I": 1, "N": 1, "T": 8}
    y_raw = np.ones((1, 1, 8), dtype=complex)
    config = default_config()
    config["noise_variance"] = 1.0
    config["global_vp"] = dict(config["global_vp"])
    config["global_vp"].update(
        {
            "mode": "adaptive_jones",
            "adaptive_jones_trigger_mode": "noise_floor",
            "adaptive_jones_noise_floor_factor": 1.02,
            "enable_z_rescue_multistart": False,
        }
    )

    def fake_fixed(*args, **kwargs):
        return {
            "p_u": np.zeros(3),
            "delta_t": 0.0,
            "raw_objective_final": 1.01,
            "raw_objective": 1.01,
            "Y_hat": y_raw.copy(),
            "beta_raw": np.ones(1, dtype=complex),
            "linear_nuisance_dim": 1,
            "nonlinear_dim": 6,
            "global_vp_success": True,
        }

    def fail_jones(*args, **kwargs):
        raise AssertionError("Jones branch should be skipped at the noise floor")

    monkeypatch.setattr(
        global_vp, "_global_exact_spherical_vp_refinement_least_squares", fake_fixed
    )
    monkeypatch.setattr(
        global_vp, "_global_exact_spherical_vp_refinement_lbfgsb_reduced", fail_jones
    )
    monkeypatch.setattr(
        global_vp,
        "_adaptive_jones_lambdas",
        lambda *args, **kwargs: (np.array([1.0]), np.array([1.0])),
    )

    result = global_vp.global_exact_spherical_vp_refinement(y_raw, {}, scene, config)

    assert result["selected_vp_family_branch"] == "fixed_pol_anchor"
    assert result["adaptive_jones_triggered"] is False
    assert result["adaptive_jones_trigger_reason"] == "fixed_at_noise_floor"
    assert result["global_vp_jones_runtime_s"] == 0.0


def test_main_single_smoke_run_executes():
    config = default_config()
    config.update(
        {
            "diagnostic_mode": "smoke",
            "print_progress": False,
            "run_full_legacy_comparison": False,
        }
    )
    config["global_vp"] = dict(config["global_vp"])
    config["global_vp"].update({"mode": "jones_free", "max_iter": 1})
    result = run_single_proposed_diagnostic(config, allow_stage2=False)
    assert result["final"]["Y_hat"].shape == result["Y_noisy"].shape
    assert result["final"]["nonlinear_dim"] == 4


def test_regularized_beta_objective_and_gradient_consistency():
    config, scene, _, y_raw, init_estimate = _scene_truth_and_init(beta_reg=1.0e-3)
    config["global_vp"].update(
        {
            "mode": "fixed_pol",
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


def test_fixed_pol_vp_recovers_mm_accuracy_for_multi_ris_geometry():
    if not scipy_is_available():
        pytest.skip("scipy.optimize.least_squares is required for this regression")
    config, scene, _, y_raw, init_estimate = _scene_truth_and_init()
    assert scene["K"] == 2
    config["global_vp"]["solver"] = "least_squares"
    config["global_vp"]["mode"] = "fixed_pol"
    init_estimate = copy.deepcopy(init_estimate)
    init_estimate["p_u"] = scene["p_u_true"] + np.array([0.01, -0.008, 0.004])
    init_estimate["delta_t"] = scene["delta_t_true"] + 2.0e-11
    final = global_exact_spherical_vp_refinement(
        y_raw, init_estimate, scene, config
    )
    pos_error = position_rmse(final["p_u"], scene["p_u_true"])
    y_nmse = relative_nmse(final["Y_hat"], y_raw)

    assert final["optimizer"]["method"] == "scipy.optimize.least_squares"
    assert final["global_vp_solver"] == "least_squares"
    assert pos_error < 2.0e-3
    assert y_nmse < 5.0e-4


def test_single_ris_has_snr_independent_position_floor():
    if not scipy_is_available():
        pytest.skip("scipy.optimize.least_squares is required for this regression")
    config = _small_config(k_paths=1)
    config["global_vp"]["mode"] = "fixed_pol"
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
    init_estimate = _init_from_truth(scene, components)
    init_estimate["p_u"] = scene["p_u_true"] + np.array([0.08, -0.06, 0.03])
    init_estimate["delta_t"] = scene["delta_t_true"] + 2.0e-10

    errors = []
    for snr_db in (0.0, 30.0):
        y_noisy, _ = add_awgn(
            y_true, snr_db, np.random.default_rng(777)
        )
        final = global_exact_spherical_vp_refinement(
            y_noisy, copy.deepcopy(init_estimate), scene, config
        )
        errors.append(position_rmse(final["p_u"], scene["p_u_true"]))

    assert errors[0] > 0.1
    assert errors[1] > 0.1
    # The oriented-v2 panel improves conditioning slightly, but one RIS still
    # retains a large, SNR-independent localization floor.
    assert errors[1] >= 0.75 * errors[0]


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
