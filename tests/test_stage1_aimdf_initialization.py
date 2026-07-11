import itertools

import numpy as np
import pytest

from src.channel_model import channel_components, generate_scene, synthesize_raw_tensor
from src.config import apply_stage1_init_preset, default_config
from src.estimators import (
    _coupled_hankel_factor_initialization,
    _fit_z_model,
    _rank_one_snapshot_initialization,
    initialize_from_hankel,
)
from src.projections_delay import (
    bq_from_poles,
    estimate_poles_aimdf_asym_tls_from_hankel,
    estimate_poles_aimdf_tls_from_hankel,
    estimate_poles_aimdf_tls_from_hankel_with_diagnostics,
    estimate_poles_esprit_from_hankel,
)
from src.tensor_utils import hankelize_frequency, reconstruct_z


def _complex_normal(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _best_phase_error(estimated: np.ndarray, true: np.ndarray) -> float:
    best = np.inf
    for perm in itertools.permutations(range(true.size)):
        aligned = estimated[list(perm)]
        error = np.max(np.abs(np.angle(aligned / true)))
        best = min(best, float(error))
    return best


def _normalized_snapshot_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vec = left.reshape(-1)
    right_vec = right.reshape(-1)
    left_vec = left_vec / np.linalg.norm(left_vec)
    right_vec = right_vec / np.linalg.norm(right_vec)
    return float(abs(np.vdot(left_vec, right_vec)))


def test_aimdf_tls_recovers_noiseless_delay_poles_up_to_permutation():
    rng = np.random.default_rng(101)
    i_dim, p_dim, l_dim, t_dim, k_paths = 5, 5, 5, 7, 3
    poles = np.exp(1j * np.array([-0.91, 0.27, 1.13]))
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    beta = _complex_normal(rng, (k_paths,))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    z_tensor = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)

    estimated = estimate_poles_aimdf_tls_from_hankel(
        z_tensor, k_paths, forward_backward=True, tls=True
    )

    assert estimated.shape == (k_paths,)
    assert _best_phase_error(estimated, poles) < 1.0e-8


def test_stage1_asym_tls_noiseless_poles():
    rng = np.random.default_rng(108)
    i_dim, p_dim, l_dim, t_dim, k_paths = 5, 5, 6, 7, 3
    poles = np.exp(1j * np.array([-0.82, 0.19, 1.04]))
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    beta = _complex_normal(rng, (k_paths,))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    z_tensor = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)

    estimated, diagnostics = estimate_poles_aimdf_asym_tls_from_hankel(
        z_tensor, k_paths, forward_backward=True, tls=True
    )

    assert estimated.shape == (k_paths,)
    assert _best_phase_error(estimated, poles) < 1.0e-8
    assert diagnostics["delay_method"] == "aimdf_asym_tls"
    assert diagnostics["Y_asym_shape"] == (l_dim, i_dim * t_dim * p_dim)


def test_stage1_fullfreq_tls_still_available():
    rng = np.random.default_rng(109)
    config = default_config()
    config.update(
        {
            "K": 3,
            "M_A": 1,
            "ris_shape": (4, 4),
            "N": 9,
            "P": 5,
            "T": 12,
            "ris_centers": config["ris_centers"][:3].copy(),
            "stage1_delay_method": "aimdf_fullfreq_tls",
        }
    )
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_tensor = synthesize_raw_tensor(components, scene["beta_true"])
    z_tensor = hankelize_frequency(y_tensor, scene["P"])
    estimate = initialize_from_hankel(z_tensor, scene, config)

    assert estimate["stage1_delay_method"] == "aimdf_fullfreq_tls"
    assert estimate["stage1_delay_singular_values"].size >= scene["K"]


def test_covariance_eigh_delay_subspace_matches_svd():
    rng = np.random.default_rng(117)
    i_dim, p_dim, l_dim, t_dim, k_paths = 5, 5, 5, 7, 3
    poles = np.exp(1j * np.array([-0.71, 0.23, 1.08]))
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    beta = _complex_normal(rng, (k_paths,))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    z_tensor = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)

    svd_poles, svd_diag = estimate_poles_aimdf_tls_from_hankel_with_diagnostics(
        z_tensor, k_paths, subspace_solver="svd"
    )
    eigh_poles, eigh_diag = estimate_poles_aimdf_tls_from_hankel_with_diagnostics(
        z_tensor, k_paths, subspace_solver="covariance_eigh"
    )

    assert _best_phase_error(eigh_poles, svd_poles) < 1.0e-10
    np.testing.assert_allclose(
        eigh_diag["singular_values"][:k_paths],
        svd_diag["singular_values"][:k_paths],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_default_config_contains_explicit_stage1_options():
    config = default_config()

    assert config["stage1_init_mode"] == "paper_stable"
    assert config["stage1_delay_method"] == "aimdf_fullfreq_tls"
    assert config["stage1_delay_subspace_solver"] == "covariance_eigh"
    assert config["stage1_forward_backward"] is True
    assert config["stage1_tls"] is True
    assert config["stage1_factor_init"] == "hankel_coupled_ls"
    assert config["stage1_factor_reg_mode"] == "relative"
    assert config["stage1_factor_reg_rel"] == 1.0e-6
    assert config["stage1_ris_geometry_mode"] == "coarse_to_exact_assignment"
    assert config["stage1_assignment_num_exact_permutations"] == 2
    assert config["ris_search"]["num_range"] == 15
    assert config["ris_search"]["num_elev"] == 9
    assert config["ris_search"]["num_az"] == 25
    assert config["ris_search"]["num_exact_refine_starts"] == 6
    assert config["ris_search"]["num_lift_candidates"] == 4
    assert config["ris_search"]["num_lift_steps"] == 4


def test_coupled_hankel_factor_initialization_recovers_rank_one_snapshots():
    rng = np.random.default_rng(102)
    i_dim, p_dim, l_dim, t_dim, k_paths = 6, 5, 4, 8, 3
    poles = np.exp(1j * np.array([-0.74, 0.18, 0.96]))
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    beta = _complex_normal(rng, (k_paths,))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    z_tensor = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)

    a_proxy, c_proxy, diagnostics = _coupled_hankel_factor_initialization(
        z_tensor,
        poles,
        reg=1.0e-14,
        config={"stage1_factor_reg_mode": "absolute", "stage1_factor_reg": 1.0e-14},
        return_diagnostics=True,
    )

    assert a_proxy.shape == (i_dim, k_paths)
    assert c_proxy.shape == (t_dim, k_paths)
    similarities = np.empty((k_paths, k_paths), dtype=float)
    for est_col in range(k_paths):
        estimated_snapshot = a_proxy[:, est_col, None] * c_proxy[None, :, est_col]
        for true_col in range(k_paths):
            true_snapshot = (
                beta[true_col] * a_mat[:, true_col, None] * c_mat[None, :, true_col]
            )
            similarities[est_col, true_col] = _normalized_snapshot_similarity(
                estimated_snapshot, true_snapshot
            )
    best = 0.0
    for perm in itertools.permutations(range(k_paths)):
        best = max(best, min(similarities[col, perm[col]] for col in range(k_paths)))
    assert best > 1.0 - 1.0e-8
    assert diagnostics["stage1_max_rank1_ratio"] < 1.0e-8


def test_stage1_relative_regularization():
    rng = np.random.default_rng(110)
    z_tensor = _complex_normal(rng, (4, 3, 3, 5))
    poles = np.exp(1j * np.array([0.1, 0.7]))
    _, _, diagnostics = _coupled_hankel_factor_initialization(
        z_tensor,
        poles,
        config={
            "stage1_factor_reg_mode": "relative",
            "stage1_factor_reg_rel": 1.0e-3,
            "stage1_factor_reg_floor": 2.0e-12,
            "eps": 1.0e-12,
        },
        return_diagnostics=True,
    )

    b_mat, q_mat = bq_from_poles(poles, 3, 3)
    d_delay = np.column_stack(
        [(b_mat[:, k, None] * q_mat[None, :, k]).reshape(-1) for k in range(2)]
    )
    gram0 = d_delay.T @ d_delay.conj()
    expected = 1.0e-3 * np.trace(gram0).real / 2 + 2.0e-12
    assert diagnostics["stage1_factor_reg_abs"] == pytest.approx(expected)
    assert np.isfinite(diagnostics["stage1_factor_gram_condition_number"])


def test_invalid_stage1_factor_init_raises_value_error():
    rng = np.random.default_rng(103)
    i_dim, p_dim, l_dim, t_dim, k_paths = 4, 4, 4, 5, 2
    z_tensor = _complex_normal(rng, (i_dim, p_dim, l_dim, t_dim))
    scene = {"I": i_dim, "P": p_dim, "L": l_dim, "T": t_dim, "K": k_paths}
    config = default_config()
    config.update(
        {
            "stage1_delay_method": "aimdf_tls",
            "stage1_factor_init": "not_a_stage1_method",
        }
    )

    with pytest.raises(ValueError, match="unknown stage1_factor_init"):
        initialize_from_hankel(z_tensor, scene, config)


def test_invalid_stage1_delay_method_raises_value_error():
    rng = np.random.default_rng(104)
    i_dim, p_dim, l_dim, t_dim, k_paths = 4, 4, 4, 5, 2
    z_tensor = _complex_normal(rng, (i_dim, p_dim, l_dim, t_dim))
    scene = {"I": i_dim, "P": p_dim, "L": l_dim, "T": t_dim, "K": k_paths}
    config = default_config()
    config["stage1_delay_method"] = "not_a_delay_method"

    with pytest.raises(ValueError, match="unknown stage1_delay_method"):
        initialize_from_hankel(z_tensor, scene, config)


def test_aimdf_tls_falls_back_to_ls_when_tls_is_not_identifiable():
    rng = np.random.default_rng(105)
    i_dim, p_dim, l_dim, t_dim, k_paths = 5, 3, 3, 6, 3
    poles = np.exp(1j * np.array([-0.73, 0.11, 0.84]))
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    beta = _complex_normal(rng, (k_paths,))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    z_tensor = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)

    estimated_tls_requested = estimate_poles_aimdf_tls_from_hankel(
        z_tensor, k_paths, forward_backward=True, tls=True
    )
    estimated_ls = estimate_poles_aimdf_tls_from_hankel(
        z_tensor, k_paths, forward_backward=True, tls=False
    )

    assert p_dim + l_dim - 2 < 2 * k_paths
    assert np.all(np.isfinite(estimated_tls_requested))
    np.testing.assert_allclose(np.abs(estimated_tls_requested), 1.0, atol=1.0e-12)
    assert _best_phase_error(estimated_tls_requested, estimated_ls) < 1.0e-10


def test_stage1_delay_factor_combinations_are_stable_for_noisy_close_delays():
    rng = np.random.default_rng(107)
    i_dim, p_dim, l_dim, t_dim, k_paths = 8, 7, 7, 10, 3
    phases = np.array([-0.18, -0.14, 0.31])
    poles = np.exp(1j * phases)
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    beta = np.array([1.0 + 0.2j, 0.9 - 0.3j, 1.1 + 0.1j])
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    z_clean = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)
    noise = _complex_normal(rng, z_clean.shape)
    noise *= 0.03 * np.linalg.norm(z_clean) / np.linalg.norm(noise)
    z_noisy = z_clean + noise

    pole_estimates = {
        "esprit_ls": estimate_poles_esprit_from_hankel(z_noisy, k_paths),
        "aimdf_tls": estimate_poles_aimdf_tls_from_hankel(
            z_noisy, k_paths, forward_backward=True, tls=True
        ),
    }
    residuals = {}
    pole_errors = {}
    for delay_method, poles_hat in pole_estimates.items():
        pole_errors[delay_method] = _best_phase_error(poles_hat, poles)
        for factor_init in ("raw_snapshot", "hankel_coupled_ls"):
            if factor_init == "raw_snapshot":
                a_proxy, c_proxy = _rank_one_snapshot_initialization(z_noisy, poles_hat)
            else:
                a_proxy, c_proxy = _coupled_hankel_factor_initialization(
                    z_noisy,
                    poles_hat,
                    reg=1.0e-10,
                    config={"stage1_factor_reg_mode": "absolute", "stage1_factor_reg": 1.0e-10},
                )
            b_hat, q_hat = bq_from_poles(poles_hat, p_dim, l_dim)
            _, _, sse = _fit_z_model(z_noisy, a_proxy, b_hat, q_hat, c_proxy)
            residuals[(delay_method, factor_init)] = sse / (
                np.linalg.norm(z_noisy) ** 2 + 1.0e-12
            )

    default_key = ("aimdf_tls", "hankel_coupled_ls")
    baseline_key = ("esprit_ls", "raw_snapshot")

    assert set(residuals) == {
        ("esprit_ls", "raw_snapshot"),
        ("aimdf_tls", "raw_snapshot"),
        ("esprit_ls", "hankel_coupled_ls"),
        default_key,
    }
    assert pole_errors["aimdf_tls"] <= pole_errors["esprit_ls"] + 1.0e-12
    assert residuals[default_key] <= residuals[baseline_key] + 1.0e-12
    assert residuals[default_key] < 0.05


def test_initialize_from_hankel_returns_expected_keys_and_rebuilds_bq_from_poles():
    rng = np.random.default_rng(106)
    config = default_config()
    config.update(
        {
            "K": 2,
            "M_A": 1,
            "ris_shape": (4, 4),
            "N": 9,
            "P": 5,
            "T": 12,
            "ris_centers": config["ris_centers"][:2].copy(),
            "stage1_delay_method": "aimdf_fullfreq_tls",
            "stage1_forward_backward": True,
            "stage1_tls": True,
            "stage1_factor_init": "hankel_coupled_ls",
            "stage1_factor_reg": 1.0e-12,
        }
    )
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_tensor = synthesize_raw_tensor(components, scene["beta_true"])
    z_tensor = hankelize_frequency(y_tensor, scene["P"])

    estimate = initialize_from_hankel(z_tensor, scene, config)

    expected = {
        "poles",
        "A",
        "B",
        "Q",
        "C",
        "beta_z",
        "gamma",
        "eta_pol",
        "ris_eta",
        "assignment",
        "initial_z_residual",
        "Z_hat",
        "stage1_delay_method",
        "stage1_factor_init",
        "stage1_forward_backward",
        "stage1_tls",
        "stage1_delay_singular_values",
        "stage1_pole_magnitudes_before_unit_circle",
        "stage1_min_delay_pole_phase_sep",
        "stage1_factor_reg_abs",
        "stage1_factor_gram_condition_number",
        "stage1_snapshot_singular_values",
        "stage1_rank1_ratios",
        "stage1_max_rank1_ratio",
        "stage1_assignment_margin",
        "stage1_ris_residuals",
        "stage1_max_ris_residual",
        "stage1_ris_residual_type",
        "stage1_local_geometry_valid",
        "stage1_delay_valid",
        "stage1_boundary_hit",
        "stage1_assignment_confident",
        "stage1_time_delay_estimation",
        "stage1_time_vandermonde_reconstruction",
        "stage1_time_coupled_ls",
        "stage1_time_rank1_svd_split",
        "stage1_time_assignment_total",
        "stage1_time_assignment_evs",
        "stage1_time_assignment_ris",
        "stage1_time_ris_codebook_build",
        "stage1_time_ris_projection_refine",
        "stage1_time_reliability_diagnostics",
        "stage1_time_other",
        "assignment_costs_col_by_panel",
        "best_assignment_score",
        "second_assignment_score",
        "assignment_margin",
        "selected_clock_offsets",
        "selected_clock_mean",
        "selected_clock_std",
        "all_assignment_scores",
    }
    assert expected.issubset(estimate.keys())
    b_expected, q_expected = bq_from_poles(estimate["poles"], scene["P"], scene["L"])
    np.testing.assert_allclose(estimate["B"], b_expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(estimate["Q"], q_expected, atol=0.0, rtol=0.0)
    expected_residual = np.linalg.norm(estimate["Z_hat"] - z_tensor) ** 2 / (
        np.linalg.norm(z_tensor) ** 2 + config["eps"]
    )
    assert estimate["initial_z_residual"] == pytest.approx(expected_residual)
    assert estimate["stage1_delay_method"] == "aimdf_fullfreq_tls"
    assert estimate["stage1_factor_init"] == "hankel_coupled_ls"
    assert estimate["stage1_forward_backward"] is True
    assert estimate["stage1_tls"] is True
    assert estimate["columns_are_panel_ordered"] is True
    assert estimate["panel_to_column_assignment"][0] in range(scene["K"])
    assert len(estimate["stage1_rank1_ratios"]) == scene["K"]
    assert estimate["stage1_max_rank1_ratio"] == pytest.approx(
        float(np.max(estimate["stage1_rank1_ratios"]))
    )
    assert len(estimate["stage1_ris_residuals"]) == scene["K"]
    assert np.isfinite(estimate["stage1_max_ris_residual"])
    for key in (
        "stage1_local_geometry_valid",
        "stage1_delay_valid",
        "stage1_boundary_hit",
        "stage1_assignment_confident",
    ):
        assert estimate[key].shape == (scene["K"],)
        assert estimate[key].dtype == bool
    for key in (
        "stage1_time_delay_estimation",
        "stage1_time_coupled_ls",
        "stage1_time_assignment_total",
    ):
        assert np.isfinite(estimate[key])
        assert estimate[key] >= 0.0


def test_stage1_rank1_diagnostics_exist():
    rng = np.random.default_rng(111)
    z_tensor = _complex_normal(rng, (4, 4, 4, 5))
    poles = np.exp(1j * np.array([0.2, 0.9]))
    _, _, diagnostics = _coupled_hankel_factor_initialization(
        z_tensor,
        poles,
        config={"stage1_factor_reg_mode": "relative", "eps": 1.0e-12},
        return_diagnostics=True,
    )

    assert len(diagnostics["stage1_snapshot_singular_values"]) == 2
    assert diagnostics["stage1_rank1_ratios"].shape == (2,)
    assert np.isfinite(diagnostics["stage1_max_rank1_ratio"])


def test_stage1_physical_panel_ordering():
    rng = np.random.default_rng(112)
    config = default_config()
    config.update(
        {
            "K": 2,
            "M_A": 1,
            "ris_shape": (4, 4),
            "N": 9,
            "P": 5,
            "T": 12,
            "ris_centers": config["ris_centers"][:2].copy(),
        }
    )
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    z_tensor = hankelize_frequency(synthesize_raw_tensor(components, scene["beta_true"]), scene["P"])

    estimate = initialize_from_hankel(z_tensor, scene, config)

    assert estimate["columns_are_panel_ordered"] is True
    for panel, column in enumerate(estimate["panel_to_column_assignment"]):
        assert estimate["column_to_panel_assignment"][column] == panel
    assert estimate["A"].shape[1] == scene["K"]


def test_stage1_paper_balanced_uses_coarse_to_exact_assignment():
    config = default_config()
    apply_stage1_init_preset(config, "paper_balanced")
    assert config["stage1_ris_geometry_mode"] == "coarse_to_exact_assignment"
    assert config["ris_search"]["num_exact_refine_starts"] == 3
    assert config["ris_search"]["num_lift_candidates"] == 3
    assert config["ris_search"]["num_lift_steps"] == 3

    apply_stage1_init_preset(config, "paper_balanced_light")
    assert config["stage1_ris_geometry_mode"] == "coarse_correlation"


def test_stage1_paper_balanced_final_vp_not_worse_than_current_by_more_than_tolerance():
    rng = np.random.default_rng(114)
    config = default_config()
    config.update(
        {
            "K": 2,
            "M_A": 1,
            "ris_shape": (4, 4),
            "N": 9,
            "P": 5,
            "T": 12,
            "ris_centers": config["ris_centers"][:2].copy(),
        }
    )
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    z_tensor = hankelize_frequency(synthesize_raw_tensor(components, scene["beta_true"]), scene["P"])
    paper = initialize_from_hankel(z_tensor, scene, config)
    exact_config = dict(config)
    exact_config["stage1_ris_geometry_mode"] = "exact_projection"
    exact_config["ris_search"] = dict(config["ris_search"])
    exact_config["ris_search"]["num_exact_refine_starts"] = 1
    exact = initialize_from_hankel(z_tensor, scene, exact_config)

    assert paper["initial_z_residual"] <= exact["initial_z_residual"] + 5.0e-1


def _small_stage1_assignment_case(seed: int = 115):
    rng = np.random.default_rng(seed)
    config = default_config()
    config.update(
        {
            "K": 2,
            "M_A": 1,
            "ris_shape": (4, 4),
            "N": 9,
            "P": 5,
            "T": 12,
            "ris_centers": config["ris_centers"][:2].copy(),
        }
    )
    apply_stage1_init_preset(config, "smoke")
    config["ris_search"]["num_exact_refine_starts"] = 2
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_tensor = synthesize_raw_tensor(components, scene["beta_true"])
    return config, scene, hankelize_frequency(y_tensor, scene["P"])


def test_stage1_coarse_to_exact_shortlist_refines_only_selected_pairs():
    config, scene, z_tensor = _small_stage1_assignment_case()
    config["stage1_ris_geometry_mode"] = "coarse_to_exact_assignment"
    config["stage1_assignment_num_exact_permutations"] = 1

    estimate = initialize_from_hankel(z_tensor, scene, config)

    assert estimate["stage1_assignment_num_coarse_pairs"] == 4
    assert estimate["stage1_assignment_num_exact_pairs"] == 2
    assert len(estimate["stage1_shortlisted_assignments"]) == 1
    assert estimate["assignment"] == list(estimate["stage1_shortlisted_assignments"][0])
    refined_mask = estimate["stage1_exact_refined_mask_col_by_panel"]
    assert np.count_nonzero(refined_mask) == 2
    assert all(refined_mask[col, panel] for col, panel in enumerate(estimate["assignment"]))
    assert np.all(np.isfinite(estimate["assignment_costs_col_by_panel"]))


def test_stage1_coarse_to_exact_matches_full_exact_when_all_permutations_shortlisted():
    config, scene, z_tensor = _small_stage1_assignment_case(seed=116)
    exact_config = dict(config)
    exact_config["ris_search"] = dict(config["ris_search"])
    exact_config["stage1_ris_geometry_mode"] = "legacy_fast_projection"
    hybrid_config = dict(config)
    hybrid_config["ris_search"] = dict(config["ris_search"])
    hybrid_config["stage1_ris_geometry_mode"] = "coarse_to_exact_assignment"
    hybrid_config["stage1_assignment_num_exact_permutations"] = 2

    exact = initialize_from_hankel(z_tensor, scene, exact_config)
    hybrid = initialize_from_hankel(z_tensor, scene, hybrid_config)

    assert hybrid["stage1_assignment_num_exact_pairs"] == 4
    assert hybrid["assignment"] == exact["assignment"]
    np.testing.assert_allclose(hybrid["ris_eta"], exact["ris_eta"], atol=1.0e-9)
    np.testing.assert_allclose(
        hybrid["assignment_costs_col_by_panel"],
        exact["assignment_costs_col_by_panel"],
        atol=1.0e-10,
    )
