import numpy as np

from src.config import default_config
from src.estimators import _accept_strict_sse, structured_refinement
from src.main_single_proposed import (
    _print_stage_two_update_diagnostics,
    _weak_reasonable_stage1_config,
)
from src.projections_delay import bq_from_poles


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
    assert config["SNR_dB"] == 0.0
    assert config["trials"] == 1
    assert config["num_structured_iters"] == 2
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
    assert config["stage1_delay_method"] == "aimdf_tls"
    assert config["stage1_forward_backward"] is True
    assert config["stage1_tls"] is True
    assert config["stage1_factor_init"] == "hankel_coupled_ls"
    assert config["stage1_factor_reg"] == 1.0e-10
    assert config["ris_search"]["projection_mode"] == "wesvp_ms"
    assert config["ris_search"]["use_qd_init"] is False
    assert config["ris_search"]["qd_proxy_reg"] == 1.0e-6
    assert config["ris_search"]["qd_proxy_max_rel_residual"] == 0.5
    assert config["ris_search"]["qd_num_range"] == 41
    assert config["ris_search"]["wesvp_max_iter"] == 100
    assert config["ris_search"]["wesvp_ftol"] == 1.0e-12
    assert config["ris_search"]["wesvp_gtol"] == 1.0e-8
    assert config["ris_search"]["use_fresnel_warm_start"] is True
    assert config["ris_centers"].shape == (3, 3)
    assert config["stage2_enable_evs"] is False
    assert config["stage2_enable_delay"] is False
    assert config["stage2_enable_ris"] is True
    assert config["stage2_guarded"] is True
    assert config["global_vp"]["solver"] == "least_squares"
    assert config["global_vp"]["max_iter"] == 80
    assert config["global_vp"]["ftol"] == 1.0e-12
    assert config["global_vp"]["gtol"] == 1.0e-8
    assert config["global_vp"]["beta_reg"] == 0.0
    assert config["global_vp"]["evs_mode"] == "legacy_or_full_polarization"
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
