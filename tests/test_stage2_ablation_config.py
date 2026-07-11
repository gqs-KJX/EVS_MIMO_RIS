import numpy as np

from src.config import apply_stage1_init_preset, default_config
from src.estimators import _accept_strict_sse, structured_refinement
from src.main_single_proposed import (
    _ngc_certificate,
    _print_stage_two_update_diagnostics,
    _weak_reasonable_stage1_config,
    enumerate_top_assignment_hypotheses,
    select_proposed_branch,
    stage2_severe_unreliable,
)
from src.projections_delay import bq_from_poles, pole_from_tau


def test_default_config_matches_single_diagnostic_defaults():
    config = default_config()
    assert config["fc"] == 60.0e9
    assert config["delta_f"] == 5.0e6
    assert config["N"] == 63
    assert config["P"] == 32
    assert config["N"] - config["P"] + 1 == 32
    assert config["K"] == 3
    assert config["M_A"] == 16
    assert config["ris_shape"] == (64, 64)
    assert config["T"] == 256
    assert config["SNR_dB"] == -10.0
    assert config["trials"] == 1
    assert config["num_structured_iters"] == 1
    assert config["stage2_ris_rescue_max_iters"] == 1
    assert config["stage2_ris_rescue_impl"] == "local_ris_projection"
    assert config["stage2_ris_rescue_use_damping"] is False
    assert tuple(config["stage2_ris_rescue_damping_grid"]) == (0.0, 1.0)
    # The clock-annihilated pseudorange block is z-blind at K = 3 and is kept
    # only as an ablation knob, so it must stay disabled by default.
    assert config["stage2_pllg_pseudorange_block_weight"] == 0.0
    assert config["stage2_clock_estimator"] == "decoupled_robust"
    assert config["stage2_clock_sigma_range_m"] == 0.12
    assert config["stage2_clock_outlier_kappa"] == 3.0
    assert config["jnpp_use_confidence_weights"] is True
    assert config["jnpp_rank_weight_rho"] == 2.0
    assert config["jnpp_min_weight"] == 0.05
    assert config["jnpp_use_leave_one_out"] is True
    assert config["jnpp_max_candidates"] == 2
    assert config["jnpp_num_starts"] == 4
    assert config["jnpp_start_perturb_m"] == 0.25
    assert config["jnpp_use_coarse_grid"] is False
    assert config["jnpp_position_box_m"] == 1.5
    assert config["jnpp_check_gradient"] is False
    assert config["jnpp_clock_postcheck_ns"] == 0.5
    assert config["jnpp_clock_tie_rel_tol"] == 1.0e-3
    assert config["jnpp_assignment_aware"] is False
    assert config["jnpp_assignment_margin_threshold"] == 0.2
    assert config["jnpp_top_assignments"] == 1
    assert config["stage2_precise_ablation"] is False
    assert config["direct_vp_first"] is True
    assert config["direct_vp_max_good_nfev"] == 12
    assert config["direct_vp_noise_floor_factor"] == 1.5
    assert config["direct_vp_min_rel_residual_decrease"] == 1.0e-4
    assert config["proposed_stage2_policy"] == "ngc_certified_ris_only"
    assert config["rescue_accept_min_rel_improvement"] == 0.0
    assert config["rescue_accept_min_abs_improvement"] == 1.0e-8
    assert config["ngc_lambda_ris"] == 1.0
    assert config["ngc_clock_green_quantile"] == 0.99
    assert config["ngc_clock_red_quantile"] == 0.999
    assert config["ngc_clock_sigma_floor_ns"] == 0.5
    assert config["ngc_ris_green_threshold"] == 0.3
    assert config["ngc_ris_red_threshold"] == 0.7
    assert config["mhr_assignment_margin_threshold"] == 0.3
    assert config["mhr_rank1_ratio_threshold"] == 0.9
    assert config["mhr_z_residual_threshold"] == 0.98
    assert config["mhr_top_assignments"] == 6
    assert tuple(config["mhr_ris_grid"]) == (7, 5, 9)
    assert config["mhr_top_ris_candidates_per_path"] == 3
    assert config["mhr_max_global_hypotheses"] == 8
    assert config["mhr_short_vp_max_nfev"] == 5
    assert config["mhr_num_full_vp_candidates"] == 1
    assert config["enable_global_vp"] is True
    assert config["stage2_mode"] == "none"
    assert config["diagnostic_mode"] == "performance"
    assert config["diagnostic_fast_problem_size"] is False
    assert config["diagnostic_fast_stage1_search"] is False
    assert config["final_refinement_method"] == "global_exact_spherical_vp"
    assert config["vp_max_nfev"] == 10
    assert config["vp_max_iter"] == 10
    assert config["delta_t_true"] == 5.0e-9
    assert config["delta_t_bounds"][1] == 10.0e-9
    assert config["stage1_init_mode"] == "paper_stable"
    assert config["stage1_delay_method"] == "aimdf_fullfreq_tls"
    assert config["stage1_forward_backward"] is True
    assert config["stage1_tls"] is True
    assert config["stage1_factor_init"] == "hankel_coupled_ls"
    assert config["stage1_factor_reg"] == 1.0e-10
    assert config["stage1_factor_reg_mode"] == "relative"
    assert config["stage1_ris_geometry_mode"] == "coarse_to_exact_assignment"
    assert config["ris_search"]["projection_mode"] == "wesvp_ms"
    assert config["ris_search"]["use_qd_init"] is False
    assert config["ris_search"]["qd_proxy_reg"] == 1.0e-6
    assert config["ris_search"]["qd_proxy_max_rel_residual"] == 0.5
    assert config["ris_search"]["qd_num_range"] == 41
    assert config["ris_search"]["wesvp_max_iter"] == 100
    assert config["ris_search"]["wesvp_ftol"] == 1.0e-12
    assert config["ris_search"]["wesvp_gtol"] == 1.0e-8
    assert config["ris_search"]["use_fresnel_warm_start"] is True
    assert config["ris_search"]["stage2_warm_start_mode"] == "coarse_exact_multistart"
    assert config["ris_search"]["stage2_warm_start_shortlist_size"] == 4
    assert config["global_vp"]["adaptive_jones_trigger_mode"] == "noise_floor"
    assert config["global_vp"]["adaptive_jones_max_iter"] == 20
    assert config["global_vp"]["z_rescue_strategy"] == "probe_then_refine"
    assert config["global_vp"]["z_rescue_refine_vp_mode"] == "fixed_pol"
    assert config["ris_centers"].shape == (3, 3)
    assert config["stage2_enable_evs"] is False
    assert config["stage2_enable_delay"] is False
    assert config["stage2_enable_ris"] is True
    assert config["stage2_guarded"] is True
    assert config["global_vp"]["solver"] == "least_squares"
    assert config["global_vp"]["mode"] == "adaptive_jones"
    assert config["global_vp"]["max_iter"] == 80
    assert config["global_vp"]["ftol"] == 1.0e-12
    assert config["global_vp"]["gtol"] == 1.0e-8
    assert config["global_vp"]["beta_reg"] == 0.0
    assert config["global_vp"]["evs_mode"] == "legacy_or_full_polarization"
    assert config["global_vp"]["jones_regularization_scaling"] == "gram"
    assert config["global_vp"]["jones_lambda0"] == 1.0
    assert config["global_vp"]["jones_lambda_min"] == 1.0e-4
    assert config["global_vp"]["jones_lambda_max"] == 1.0e8
    assert config["global_vp"]["jones_snr_eps"] == 1.0e-12
    assert config["global_vp"]["run_fixed_pol_anchor"] is True
    assert config["global_vp"]["jones_leakage_threshold"] == 0.25
    assert config["global_vp"]["jones_min_rel_improvement"] == 1.0e-3
    assert config["global_vp"]["jones_tau"] == 0.25
    assert config["global_vp"]["jones_tau_min"] == 1.0e-3
    assert config["global_vp"]["jones_tau_max"] == 10.0
    assert config["global_vp"]["jones_diagonal_loading"] == 1.0e-10
    assert config["global_vp"]["gof_pfa"] == 0.05
    assert config["global_vp"]["efim_lambda_min_threshold"] == 1.0e-8
    assert config["global_vp"]["efim_cond_threshold"] == 1.0e12
    assert config["global_vp"]["use_data_only_efim_gate"] is True
    assert config["global_vp"]["use_delay_prior"] is False
    assert config["global_vp"]["delay_prior_weight"] == 1.0
    assert config["global_vp"]["delay_prior_sigma_s"] == 2.0e-11
    assert config["global_vp"]["use_weight"] is False
    assert config["global_vp"]["use_multistart"] is False
    assert config["global_vp"]["num_perturb_starts"] == 0
    assert config["global_vp"]["position_perturb_std_m"] == 0.05
    assert config["global_vp"]["clock_perturb_std_s"] == 1.0e-10
    assert config["global_vp"]["use_trust_region"] is False
    assert config["global_vp"]["position_trust_radius_m"] == 0.3
    assert config["global_vp"]["clock_trust_radius_s"] == 3.0e-10
    assert config["global_vp"]["objective_rollback_tolerance"] == 1.0e-12
    assert config["global_vp"]["overwrite_factor_keys"] is False
    assert config["global_vp"]["finite_difference_check"] is False
    assert config["global_vp"]["use_analytic_jacobian"] is True
    assert config["global_vp"]["matrix_free_beta"] is False
    assert config["global_vp"]["vp_dictionary_mode"] == "matrix_free"
    assert config["global_vp"]["vp_debug_compare_explicit"] is False
    assert config["global_vp"]["vp_debug_compare_max_evals"] == 3
    assert config["stage2_strict_accept_rel"] == 1.0e-6
    assert config["ris_min_relative_improvement"] == 5.0e-3
    assert tuple(config["stage2_damping_grid"]) == (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    assert config["stage2_ris_weight_mode"] == "residual_diag"
    assert config["stage2_ris_weight_floor_rel"] == 5.0e-2
    assert config["stage2_ris_weight_clip"] == (0.25, 4.0)
    assert config["stage2_ris_weight_normalize"] is True


def test_structured_refinement_all_modules_disabled_keeps_factors_unchanged():
    rng = np.random.default_rng(7)
    i_dim, p_dim, l_dim, t_dim, k_paths = 5, 3, 4, 6, 2
    z_tensor = rng.normal(size=(i_dim, p_dim, l_dim, t_dim)) + 1j * rng.normal(
        size=(i_dim, p_dim, l_dim, t_dim)
    )
    a_mat = rng.normal(size=(i_dim, k_paths)) + 1j * rng.normal(size=(i_dim, k_paths))
    c_mat = rng.normal(size=(t_dim, k_paths)) + 1j * rng.normal(size=(t_dim, k_paths))
    poles = np.exp(1j * np.array([0.23, -0.71]))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    estimate = {
        "A": a_mat.copy(),
        "B": b_mat.copy(),
        "Q": q_mat.copy(),
        "C": c_mat.copy(),
        "poles": poles.copy(),
        "beta_z": np.ones(k_paths, dtype=complex),
        "gamma": np.zeros(k_paths),
        "eta_pol": np.zeros(k_paths),
        "ris_eta": np.zeros((k_paths, 3)),
    }
    config = default_config()
    config.update(
        {
            "num_structured_iters": 0,
            "stage2_enable_evs": False,
            "stage2_enable_delay": False,
            "stage2_enable_ris": False,
        }
    )
    scene = {"P": p_dim, "L": l_dim}

    refined, _ = structured_refinement(z_tensor, scene, config, estimate)

    np.testing.assert_allclose(refined["A"], a_mat)
    np.testing.assert_allclose(refined["C"], c_mat)
    np.testing.assert_allclose(refined["poles"], poles)
    assert refined["Z_hat"].shape == z_tensor.shape


def test_mhrr_severe_unreliable_uses_requested_thresholds():
    config = default_config()
    estimate = {
        "stage1_assignment_margin": 0.29,
        "stage1_max_rank1_ratio": 0.1,
        "initial_z_residual": 0.1,
    }
    reliability = {"assignment_margin": 0.29}

    severe = stage2_severe_unreliable(estimate, reliability, config)

    assert severe["severe_unreliable"] is True
    assert severe["margin_bad"] is True
    assert severe["rank1_bad"] is False
    assert severe["z_residual_bad"] is False


def test_mhrr_assignment_hypotheses_sort_by_cost_and_clock():
    config = default_config()
    config["mhr_top_assignments"] = 2
    estimate = {
        "assignment_costs_col_by_panel": np.array(
            [
                [0.0, 5.0, 5.0],
                [5.0, 0.0, 5.0],
                [5.0, 5.0, 0.0],
            ]
        ),
        "poles": np.ones(3, dtype=complex),
        "ris_eta": np.zeros((3, 3), dtype=float),
    }

    hypotheses = enumerate_top_assignment_hypotheses(estimate, config)

    assert len(hypotheses) == 2
    assert hypotheses[0]["assignment"] == (0, 1, 2)
    assert hypotheses[0]["assignment_score"] == 0.0
    assert np.isfinite(hypotheses[0]["clock_std"])


def test_mhrr_acceptance_and_rollback_use_raw_objective():
    config = default_config()
    reliability = {"decision": "ris_only_stage2_then_vp"}
    direct = {
        "final": {"raw_objective_final": 1.0},
        "Y_true": np.ones(1),
    }
    rescue_good = {
        "branch_name": "multi_hypothesis_ris_reacquisition_then_vp",
        "structured_diag": {"mhr_accepted": False},
        "final": {"raw_objective_final": 0.99},
        "Y_true": np.ones(1),
    }
    selected, no_gain = select_proposed_branch(direct, rescue_good, reliability, config)
    assert no_gain is False
    assert selected["selected_branch"] == "multi_hypothesis_ris_reacquisition_then_vp"
    assert rescue_good["structured_diag"]["mhr_accepted"] is True

    rescue_equal = {
        "branch_name": "multi_hypothesis_ris_reacquisition_then_vp",
        "structured_diag": {"mhr_accepted": False},
        "final": {"raw_objective_final": 1.0},
        "Y_true": np.ones(1),
    }
    selected, no_gain = select_proposed_branch(direct, rescue_equal, reliability, config)
    assert no_gain is True
    assert selected["selected_branch"] == "direct_vp_rollback"
    assert rescue_equal["structured_diag"]["mhr_accepted"] is False


def test_ngc_certificate_uses_efim_tau_crb_and_marks_bad_clock_red():
    config = default_config()
    scene = {
        "K": 3,
        "T": 4,
        "delta_f": 5.0e6,
        "c0": 299_792_458.0,
        "ris_centers": np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.2, 0.0]],
            dtype=float,
        ),
        "d_RB": np.zeros(3, dtype=float),
    }
    stage1_estimate = {
        "poles": np.array(
            [pole_from_tau(tau, scene["delta_f"]) for tau in (0.0, 5.0e-9, 10.0e-9)]
        ),
        "ris_eta": np.zeros((3, 3), dtype=float),
        "columns_are_panel_ordered": True,
    }
    branch = {
        "final": {
            "p_u": np.zeros(3, dtype=float),
            "delta_t": 0.0,
        },
        "direct_vp_quality": {
            "data_only_efim": np.diag([1.0e20, 1.0e20, 1.0e20, 1.0e24]),
        },
    }

    cert = _ngc_certificate("direct", branch, stage1_estimate, scene, config)

    assert cert["ngc_direct_clock_sigma_source"] == "data_only_efim_tau_crb"
    # The delay uncertainty is now returned explicitly instead of being cached
    # back onto Stage-I, so certification no longer depends on call order.
    assert "sigma_tau_k" not in stage1_estimate
    assert cert["ngc_direct_cert_status"] == "red"
    assert cert["ngc_direct_clock_score_norm"] >= cert["ngc_threshold_clock_red"]


def test_ngc_certificate_marks_single_path_clock_not_applicable():
    config = default_config()
    scene = {
        "K": 1,
        "T": 4,
        "delta_f": 5.0e6,
        "c0": 299_792_458.0,
        "ris_centers": np.array([[1.0, 0.0, 0.0]], dtype=float),
        "d_RB": np.zeros(1, dtype=float),
    }
    stage1_estimate = {
        "poles": np.array([pole_from_tau(0.0, scene["delta_f"])]),
        "ris_eta": np.zeros((1, 3), dtype=float),
        "columns_are_panel_ordered": True,
    }
    branch = {"final": {"p_u": np.zeros(3, dtype=float), "delta_t": 0.0}}

    cert = _ngc_certificate("direct", branch, stage1_estimate, scene, config)

    assert cert["ngc_direct_clock_dof"] == 0
    assert cert["ngc_direct_cert_status"] == "not_applicable"
    assert cert["ngc_direct_cert_reason"] == "clock_not_applicable_k_lt_2"


def test_main_single_weak_reasonable_stage1_config():
    config = default_config()
    weak_config = _weak_reasonable_stage1_config(config)
    weak_search = weak_config["ris_search"]
    original_search = config["ris_search"]

    assert weak_search["num_range"] == original_search["num_range"]
    assert weak_search["num_elev"] == original_search["num_elev"]
    assert weak_search["num_az"] == original_search["num_az"]
    assert weak_search["num_exact_refine_starts"] == original_search["num_exact_refine_starts"]
    assert weak_search["num_lift_candidates"] == original_search["num_lift_candidates"]
    assert weak_search["num_lift_steps"] == original_search["num_lift_steps"]
    assert original_search["num_range"] == 15
    assert original_search["num_elev"] == 9
    assert original_search["num_az"] == 25
    assert original_search["num_exact_refine_starts"] == 6
    assert original_search["num_lift_candidates"] == 4
    assert original_search["num_lift_steps"] == 4


def test_stage1_light_and_heavy_presets_are_explicit():
    config = default_config()
    apply_stage1_init_preset(config, "paper_balanced")
    assert config["stage1_init_mode"] == "paper_balanced"
    assert config["stage1_ris_geometry_mode"] == "coarse_to_exact_assignment"
    assert config["ris_search"]["num_range"] == 9
    assert config["ris_search"]["num_elev"] == 5
    assert config["ris_search"]["num_az"] == 13
    assert config["ris_search"]["num_exact_refine_starts"] == 3
    assert config["ris_search"]["num_lift_candidates"] == 3
    assert config["ris_search"]["num_lift_steps"] == 3

    apply_stage1_init_preset(config, "paper_balanced_light")
    assert config["stage1_init_mode"] == "paper_balanced_light"
    assert config["stage1_ris_geometry_mode"] == "coarse_correlation"
    assert config["ris_search"]["num_range"] == 9
    assert config["ris_search"]["num_elev"] == 5
    assert config["ris_search"]["num_az"] == 13
    assert config["ris_search"]["num_exact_refine_starts"] == 1
    assert config["ris_search"]["num_lift_candidates"] == 1
    assert config["ris_search"]["num_lift_steps"] == 1

    apply_stage1_init_preset(config, "normal_heavy")
    assert config["stage1_init_mode"] == "normal_heavy"
    assert config["stage1_ris_geometry_mode"] == "coarse_to_exact_assignment"
    assert config["ris_search"]["num_range"] == 15
    assert config["ris_search"]["num_elev"] == 9
    assert config["ris_search"]["num_az"] == 25
    assert config["ris_search"]["num_exact_refine_starts"] == 6
    assert config["ris_search"]["num_lift_candidates"] == 4
    assert config["ris_search"]["num_lift_steps"] == 4


def test_accept_strict_sse_requires_strict_relative_decrease():
    assert _accept_strict_sse(9.0, 10.0, 0.0, 1.0e-3)
    assert not _accept_strict_sse(10.0, 10.0, 0.0, 1.0e-3)
    assert not _accept_strict_sse(10.1, 10.0, 0.0, 1.0e-3)


def test_stage_two_diagnostics_prints_skipped_submodules(capsys):
    results = {
        "structured_diag": {
            "updates": [
                {
                    "delta_A": 1.0e-3,
                    "delta_B": 0.0,
                    "delta_Q": 0.0,
                    "delta_C": 0.0,
                    "delta_beta": 1.0e-4,
                    "nonfinite_A": 0,
                    "nonfinite_B": 0,
                    "nonfinite_Q": 0,
                    "nonfinite_C": 0,
                    "nonfinite_beta": 0,
                    "evs_projection_details": [
                        {"accepted": True, "skipped": False},
                        {"accepted": False, "skipped": False},
                    ],
                    "delay_projection_details": {
                        "accepted": False,
                        "skipped": True,
                        "guarded": True,
                        "global_sse_before": 12.5,
                        "global_sse_after": 12.5,
                    },
                    "mode4_assignment_order": [0, 1],
                    "ris_projection_details": [
                        {
                            "path": 0,
                            "accepted": False,
                            "skipped": True,
                            "selected_eta": np.array([0.0, 0.0, 0.0]),
                        },
                        {
                            "path": 1,
                            "accepted": False,
                            "skipped": True,
                            "selected_eta": np.array([1.0, 0.1, -0.1]),
                        },
                    ],
                }
            ]
        }
    }

    _print_stage_two_update_diagnostics(results)

    output = capsys.readouterr().out
    assert "delay structured LS: skipped=True" in output
    assert "RIS projection accepted = ['skipped', 'skipped']" in output
    assert "RIS path 0: skipped=True" in output
