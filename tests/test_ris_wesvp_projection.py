import numpy as np

from src.channel_model import generate_scene
from src.config import default_config
from src.estimators import structured_refinement
from src.geometry import make_ris_grid
from src.projections_delay import bq_from_poles
from src.projections_ris import (
    _element_domain_proxy,
    _wesvp_objective_and_grad,
    compressed_exact_response,
    project_ris_factor,
)
from src.tensor_utils import reconstruct_z


def _complex_normal(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _small_search_config(mode="wesvp_qd"):
    return {
        "range_bounds": (2.0, 4.0),
        "elev_bounds": (-0.25, 0.25),
        "az_bounds": (-0.5, 0.5),
        "num_range": 5,
        "num_elev": 5,
        "num_az": 5,
        "stage2_num_range": 3,
        "stage2_num_elev": 3,
        "stage2_num_az": 3,
        "stage2_range_span": 0.35,
        "stage2_angle_span": 0.15,
        "num_lift_candidates": 2,
        "num_lift_steps": 1,
        "lambda_phys": 1.0e-2,
        "ris_pgd_step_scale": 0.5,
        "num_exact_refine_starts": 3,
        "projection_mode": mode,
        "use_qd_init": True,
        "qd_proxy_reg": 1.0e-6,
        "qd_proxy_max_rel_residual": 0.8,
        "qd_num_range": 21,
        "wesvp_max_iter": 60,
        "wesvp_ftol": 1.0e-12,
        "wesvp_gtol": 1.0e-8,
        "use_fresnel_warm_start": True,
    }


def test_wesvp_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(201)
    wavelength = 0.05
    ris_grid = make_ris_grid(3, 4, wavelength / 2.0, wavelength / 2.0)
    m_dim = ris_grid.shape[0]
    t_dim = 9
    omega = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(t_dim, m_dim))) / np.sqrt(m_dim)
    a_rb = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=m_dim))
    c_tilde = _complex_normal(rng, t_dim)
    weight = rng.uniform(0.5, 1.5, size=t_dim)
    eta = np.array([3.1, 0.08, -0.12])
    lower = np.array([2.0, -0.3, -0.6])
    upper = np.array([4.0, 0.3, 0.6])

    _, grad = _wesvp_objective_and_grad(
        eta, c_tilde, omega, a_rb, ris_grid, wavelength, lower, upper, weight
    )
    steps = np.array([1.0e-5, 1.0e-6, 1.0e-6])
    grad_fd = np.empty(3, dtype=float)
    for dim, step in enumerate(steps):
        offset = np.zeros(3)
        offset[dim] = step
        plus, _ = _wesvp_objective_and_grad(
            eta + offset, c_tilde, omega, a_rb, ris_grid, wavelength, lower, upper, weight
        )
        minus, _ = _wesvp_objective_and_grad(
            eta - offset, c_tilde, omega, a_rb, ris_grid, wavelength, lower, upper, weight
        )
        grad_fd[dim] = (plus - minus) / (2.0 * step)

    rel_error = np.linalg.norm(grad - grad_fd) / max(np.linalg.norm(grad_fd), 1.0e-12)
    assert rel_error < 1.0e-4


def test_wesvp_qd_recovers_noiseless_uncompressed_exact_response():
    wavelength = 0.05
    ris_grid = make_ris_grid(5, 5, wavelength / 2.0, wavelength / 2.0)
    m_dim = ris_grid.shape[0]
    omega = np.eye(m_dim, dtype=complex)
    a_rb = np.ones(m_dim, dtype=complex)
    true_eta = np.array([3.0, 0.08, 0.14])
    alpha = 1.2 - 0.7j
    c_tilde = alpha * compressed_exact_response(true_eta, omega, a_rb, ris_grid, wavelength)
    search = _small_search_config("wesvp_qd")
    search["elev_bounds"] = (0.0, 0.25)

    projection = project_ris_factor(
        c_tilde,
        omega,
        a_rb,
        ris_grid,
        wavelength,
        search,
        current_eta=true_eta + np.array([0.08, -0.03, 0.04]),
    )

    assert projection["relative_residual"] < 1.0e-6
    assert abs(projection["eta_local"][0] - true_eta[0]) < 2.0e-2
    assert np.linalg.norm(projection["eta_local"][1:] - true_eta[1:]) < 2.0e-2
    assert projection["selected_model"] == "wesvp_qd"


def test_wesvp_qd_does_not_catastrophically_degrade_compressed_noisy_projection():
    rng = np.random.default_rng(202)
    wavelength = 0.05
    ris_grid = make_ris_grid(5, 5, wavelength / 2.0, wavelength / 2.0)
    m_dim = ris_grid.shape[0]
    t_dim = 10
    omega = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=(t_dim, m_dim))) / np.sqrt(m_dim)
    a_rb = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=m_dim))
    true_eta = np.array([3.0, -0.05, 0.18])
    clean = (0.8 + 0.4j) * compressed_exact_response(true_eta, omega, a_rb, ris_grid, wavelength)
    noise = _complex_normal(rng, t_dim)
    noise *= 0.08 * np.linalg.norm(clean) / np.linalg.norm(noise)
    c_tilde = clean + noise

    paper = project_ris_factor(
        c_tilde,
        omega,
        a_rb,
        ris_grid,
        wavelength,
        _small_search_config("paper"),
        current_eta=true_eta + np.array([0.05, 0.02, -0.03]),
    )
    wesvp = project_ris_factor(
        c_tilde,
        omega,
        a_rb,
        ris_grid,
        wavelength,
        _small_search_config("wesvp_qd"),
        current_eta=true_eta + np.array([0.05, 0.02, -0.03]),
    )

    assert wesvp["relative_residual"] <= 1.2 * paper["relative_residual"] + 1.0e-10
    assert wesvp["selected_model"] == "wesvp_qd"


def test_qd_poor_proxy_is_skipped_but_exact_vp_returns_valid_projection():
    wavelength = 0.05
    ris_grid = make_ris_grid(5, 5, wavelength / 2.0, wavelength / 2.0)
    m_dim = ris_grid.shape[0]
    t_dim = 4
    omega = np.zeros((t_dim, m_dim), dtype=complex)
    a_rb = np.ones(m_dim, dtype=complex)
    c_tilde = np.ones(t_dim, dtype=complex)
    search = _small_search_config("wesvp_qd")
    search["qd_proxy_max_rel_residual"] = 0.1

    _, proxy_rel = _element_domain_proxy(c_tilde, omega, a_rb)
    projection = project_ris_factor(c_tilde, omega, a_rb, ris_grid, wavelength, search)

    assert proxy_rel > search["qd_proxy_max_rel_residual"]
    assert projection["qd_used_as_start"] is False
    assert np.isfinite(projection["relative_residual"])
    assert "c" in projection and projection["c"].shape == c_tilde.shape


def test_wesvp_projection_keys_and_structured_refinement_compatibility():
    rng = np.random.default_rng(203)
    config = default_config()
    config.update(
        {
            "K": 1,
            "M_A": 1,
            "ris_shape": (3, 3),
            "N": 5,
            "P": 3,
            "T": 8,
            "ris_centers": config["ris_centers"][:1].copy(),
            "num_structured_iters": 1,
            "stage2_enable_evs": False,
            "stage2_enable_delay": False,
            "stage2_enable_ris": True,
            "stage2_guarded": False,
        }
    )
    config["ris_search"].update(_small_search_config("wesvp_qd"))
    scene = generate_scene(config, rng)
    poles = np.exp(1j * np.array([0.22]))
    b_mat, q_mat = bq_from_poles(poles, scene["P"], scene["L"])
    a_mat = _complex_normal(rng, (scene["I"], scene["K"]))
    true_eta = np.array([3.0, 0.0, 0.0])
    c_col = compressed_exact_response(
        true_eta,
        scene["Omega"][0],
        scene["a_RB"][0],
        scene["ris_grid"],
        scene["wavelength"],
    )
    c_mat = c_col[:, None] / np.linalg.norm(c_col)
    beta = np.array([1.0 + 0.2j])
    z_tensor = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)
    estimate = {
        "poles": poles.copy(),
        "A": a_mat.copy(),
        "B": b_mat.copy(),
        "Q": q_mat.copy(),
        "C": c_mat.copy(),
        "beta_z": beta.copy(),
        "gamma": np.zeros(1),
        "eta_pol": np.zeros(1),
        "ris_eta": true_eta[None, :] + np.array([[0.04, 0.02, -0.02]]),
        "Z_hat": z_tensor.copy(),
    }

    projection = project_ris_factor(
        c_mat[:, 0],
        scene["Omega"][0],
        scene["a_RB"][0],
        scene["ris_grid"],
        scene["wavelength"],
        config["ris_search"],
        current_eta=estimate["ris_eta"][0],
    )
    expected_keys = {
        "c",
        "eta_local",
        "alpha",
        "relative_residual",
        "selected_model",
        "candidates",
        "coarse_eta_local",
        "coarse_relative_residual",
        "exact_relative_residual",
        "optimizer_message",
    }
    assert expected_keys.issubset(projection.keys())

    refined, diagnostics = structured_refinement(z_tensor, scene, config, estimate)
    assert refined["C"].shape == c_mat.shape
    assert diagnostics["updates"]
