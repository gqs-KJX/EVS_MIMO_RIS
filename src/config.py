"""Default parameters for the single proposed-method demo."""

from __future__ import annotations

import math

import numpy as np


def apply_stage1_init_preset(config: dict, mode: str | None = None) -> dict:
    """Apply one of the named Stage-I initializer presets in-place."""
    preset = str(mode if mode is not None else config.get("stage1_init_mode", "paper_stable"))
    if preset not in {
        "smoke",
        "paper_balanced",
        "paper_balanced_light",
        "paper_light_ablation",
        "paper_stable",
        "robust_low_snr",
        "normal_heavy",
    }:
        raise ValueError(f"unknown stage1_init_mode {preset!r}")
    if preset == "paper_light_ablation":
        preset = "paper_balanced_light"
    config["stage1_init_mode"] = preset
    ris_search = dict(config["ris_search"])
    k_paths = int(config.get("K", 3))

    common = {
        "stage1_factor_init": "hankel_coupled_ls",
        "stage1_factor_reg_mode": "relative",
        "stage1_factor_reg_rel": 1.0e-6,
        "stage1_factor_reg_floor": 1.0e-12,
    }
    config.update(common)

    if preset == "smoke":
        config.update(
            {
                "stage1_delay_method": "aimdf_asym_tls",
                "stage1_snapshot_sketch_dim": max(4 * k_paths, 16),
                "stage1_forward_backward": True,
                "stage1_tls": True,
                "stage1_ris_geometry_mode": "coarse_correlation",
                "stage1_ris_residual_type": "coarse_proxy",
            }
        )
        ris_search.update(
            {
                "num_range": 5,
                "num_elev": 3,
                "num_az": 7,
                "num_exact_refine_starts": 0,
                "num_lift_candidates": 1,
                "num_lift_steps": 1,
            }
        )
    elif preset == "paper_balanced_light":
        config.update(
            {
                "stage1_delay_method": "aimdf_fullfreq_tls",
                "stage1_snapshot_sketch_dim": None,
                "stage1_forward_backward": True,
                "stage1_tls": True,
                "stage1_ris_geometry_mode": "coarse_correlation",
                "stage1_ris_residual_type": "coarse_proxy",
            }
        )
        ris_search.update(
            {
                "num_range": 9,
                "num_elev": 5,
                "num_az": 13,
                "num_exact_refine_starts": 1,
                "num_lift_candidates": 1,
                "num_lift_steps": 1,
            }
        )
    elif preset == "paper_balanced":
        config.update(
            {
                "stage1_delay_method": "aimdf_fullfreq_tls",
                "stage1_snapshot_sketch_dim": None,
                "stage1_forward_backward": True,
                "stage1_tls": True,
                "stage1_ris_geometry_mode": "legacy_fast_projection",
                "stage1_ris_residual_type": "legacy_fast_projection",
            }
        )
        ris_search.update(
            {
                "num_range": 9,
                "num_elev": 5,
                "num_az": 13,
                "num_exact_refine_starts": 3,
                "num_lift_candidates": 3,
                "num_lift_steps": 3,
            }
        )
    elif preset in ("paper_stable", "normal_heavy"):
        config.update(
            {
                "stage1_delay_method": "aimdf_fullfreq_tls",
                "stage1_snapshot_sketch_dim": None,
                "stage1_forward_backward": True,
                "stage1_tls": True,
                "stage1_ris_geometry_mode": "legacy_fast_projection",
                "stage1_ris_residual_type": "legacy_fast_projection",
            }
        )
        ris_search.update(
            {
                "num_range": 15,
                "num_elev": 9,
                "num_az": 25,
                "num_exact_refine_starts": 6,
                "num_lift_candidates": 4,
                "num_lift_steps": 4,
            }
        )
    else:
        config.update(
            {
                "stage1_delay_method": "aimdf_asym_tls",
                "stage1_snapshot_sketch_dim": max(8 * k_paths, 32),
                "stage1_forward_backward": True,
                "stage1_tls": True,
                "stage1_ris_geometry_mode": "legacy_fast_projection",
                "stage1_ris_residual_type": "legacy_fast_projection",
            }
        )
        ris_search.update(
            {
                "num_range": 15,
                "num_elev": 9,
                "num_az": 25,
                "num_exact_refine_starts": 6,
                "num_lift_candidates": 4,
                "num_lift_steps": 4,
            }
        )
    config["ris_search"] = ris_search
    return config


def default_config() -> dict:
    """Return the default single-diagnostic XL-RIS configuration."""
    c0 = 299_792_458.0
    fc = 60.0e9
    wavelength = c0 / fc
    k_default = 1

    config = {
        "seed": 821394406,
        "trials": 1,
        "SNR_dB": -10.0,
        "enable_global_vp": True,
        "stage2_mode": "none",
        "diagnostic_mode": "performance",
        "run_full_legacy_comparison": False,
        "verbose_stage2": False,
        "verbose_timing": False,
        "print_progress": True,
        "diagnostic_fast_problem_size": False,
        "diagnostic_fast_stage1_search": False,
        "stage2_adaptive": True,
        "stage2_rescue_type": "ris_only",
        "direct_vp_first": True,
        "direct_vp_max_good_nfev": 12,
        "direct_vp_noise_floor_factor": 1.5,
        "direct_vp_min_rel_residual_decrease": 1.0e-4,
        "proposed_stage2_policy": "ngc_certified_ris_only",
        "rescue_accept_min_rel_improvement": 0.0,
        "rescue_accept_min_abs_improvement": 1.0e-8,
        "ngc_lambda_ris": 1.0,
        "ngc_clock_green_quantile": 0.99,
        "ngc_clock_red_quantile": 0.999,
        "ngc_clock_sigma_floor_ns": 0.5,
        "ngc_ris_green_threshold": 0.3,
        "ngc_ris_red_threshold": 0.7,
        "mhr_assignment_margin_threshold": 0.3,
        "mhr_rank1_ratio_threshold": 0.9,
        "mhr_z_residual_threshold": 0.98,
        "mhr_top_assignments": 6,
        "mhr_clock_weight": 0.5,
        "mhr_clock_scale_s": 1.0e-9,
        "mhr_ris_grid": (7, 5, 9),
        "mhr_range_span": 1.5,
        "mhr_angle_span": 0.35,
        "mhr_use_current_eta_as_center": True,
        "mhr_allow_global_if_rank1_bad": True,
        "mhr_top_ris_candidates_per_path": 3,
        "mhr_max_global_hypotheses": 8,
        "mhr_short_vp_max_nfev": 5,
        "mhr_num_full_vp_candidates": 1,
        "reliability_assignment_good": 1.0,
        "reliability_assignment_low": 0.3,
        "reliability_clock_good_ns": 0.1,
        "reliability_clock_bad_ns": 0.5,
        "reliability_ris_good": 0.3,
        "reliability_ris_bad": 0.7,
        "final_refinement_method": "global_exact_spherical_vp",
        "K": 3,
        "M_A": 16,
        "ris_shape": (64, 64),
        "N": 63,
        "P": 32,
        "T": 256,
        "c0": c0,
        "fc": fc,
        "wavelength": wavelength,
        "delta_f": 5.0e6,
        "delta_t_true": 5.0e-9,
        "p_B": np.array([0.0, 0.0, 1.0]),
        "p_u_true": np.array([1.25, 0.55, 0.75]),
        "ris_centers": np.array(
            [
                [4.20, -2.20, 1.05],
                [5.10, 2.10, 1.15],
                [4.80, 0.00, 1.25],
            ]
        ),
        "ue_bounds": np.array(
            [
                [0.30, 2.70],
                [-1.40, 1.50],
                [0.35, 1.45],
            ]
        ),
        "delta_t_bounds": np.array([0.0, 10.0e-9]),
        "stage1_init_mode": "paper_stable",
        "stage1_delay_method": "aimdf_fullfreq_tls",
        "stage1_forward_backward": True,
        "stage1_tls": True,
        "stage1_snapshot_sketch_dim": None,
        "stage1_factor_init": "hankel_coupled_ls",
        "stage1_factor_reg": 1.0e-10,
        "stage1_factor_reg_mode": "relative",
        "stage1_factor_reg_rel": 1.0e-6,
        "stage1_factor_reg_floor": 1.0e-12,
        "stage1_ris_geometry_mode": "legacy_fast_projection",
        "stage1_ris_residual_type": "legacy_fast_projection",
        "ris_search": {
            "range_bounds": (2.5, 6.5),
            "elev_bounds": (-0.45, 0.25),
            "az_bounds": (-np.pi, np.pi),
            "num_range": 9,
            "num_elev": 5,
            "num_az": 13,
            "stage2_num_range": 3,
            "stage2_num_elev": 3,
            "stage2_num_az": 5,
            "stage2_range_span": 0.45,
            "stage2_angle_span": 0.12,
            "num_lift_candidates": 4,
            "num_lift_steps": 4,
            "lambda_phys": 1.0e-2,
            "ris_pgd_step_scale": 0.5,
            "num_exact_refine_starts": 3,
            "projection_mode": "wesvp_ms",
            "use_qd_init": False,
            "qd_proxy_reg": 1.0e-6,
            "qd_proxy_max_rel_residual": 0.5,
            "qd_num_range": 41,
            "wesvp_max_iter": 100,
            "wesvp_ftol": 1.0e-12,
            "wesvp_gtol": 1.0e-8,
            "use_fresnel_warm_start": True,
        },
        "num_structured_iters": 1,
        "stage2_ris_rescue_max_iters": 1,
        "stage2_rescue_impl": "legacy_multistart",
        "stage2_pllg_reweight_steps": 1,
        "stage2_pllg_cond_max": 1.0e12,
        "stage2_pllg_local_weight_mode": "auto",
        # Clock-annihilated pseudorange rows are rank-deficient at K = 3 and
        # carry ~no information about z (measured z-GDOP 35.1, range-only CRB
        # sigma_z = 4.2 m).  Enabling them degraded the seed monotonically in
        # the block-weight sweep, so they are off by default and retained only
        # as an ablation knob.
        "stage2_pllg_pseudorange_block_weight": 0.0,
        "stage2_delay_sigma_floor_ns": 0.5,
        "stage2_ris_normalization_scale": 1.0e-4,
        "stage2_lambda_ris_normalized": 1.0,
        # Exported Stage-II clock: "decoupled_robust" uses the position-free
        # m_k - r_k replicas with a median/MAD screen; "weighted_mean" restores
        # the previous non-robust closed-form clock of Phi_S2.
        "stage2_clock_estimator": "decoupled_robust",
        "stage2_clock_sigma_range_m": 0.12,
        "stage2_clock_outlier_kappa": 3.0,
        "stage2_pllg_max_projection_distance_m": 0.05,
        "stage2_pllg_legacy_fallback": True,
        "stage2_force_run_for_diagnostics": False,
        "stage2_selector_guard": True,
        "stage2_selector_raw_degradation_abs_tol": 1.0e-8,
        "stage2_selector_raw_degradation_rel_tol": 1.0e-4,
        "stage2_selector_boundary_override_min_rel_improvement": 1.0e-3,
        "stage2_ris_rescue_impl": "local_ris_projection",
        "stage2_ris_rescue_use_damping": False,
        "stage2_ris_rescue_damping_grid": (0.0, 1.0),
        "jnpp_weight_mode": "exponential",
        "jnpp_use_confidence_weights": True,
        "jnpp_rank_weight_rho": 2.0,
        "jnpp_min_weight": 0.05,
        "jnpp_inverse_rank_eps": 1.0e-2,
        "jnpp_exp_rank_rho": 2.0,
        "jnpp_exp_min_weight": 0.05,
        "jnpp_normalize_weights": True,
        "jnpp_use_leave_one_out": True,
        "jnpp_max_candidates": 1 + k_default,
        "jnpp_num_starts": 4,
        "jnpp_start_perturb_m": 0.25,
        "jnpp_enable_z_starts": True,
        "jnpp_z_start_grid_size": 7,
        "jnpp_z_start_margin_m": 0.02,
        "jnpp_use_coarse_grid": False,
        "jnpp_position_box_m": 1.5,
        "jnpp_check_gradient": False,
        "jnpp_clock_postcheck_ns": 0.5,
        "jnpp_clock_tie_rel_tol": 1.0e-3,
        "jnpp_assignment_aware": False,
        "jnpp_assignment_margin_threshold": 0.2,
        "jnpp_top_assignments": min(3, int(math.factorial(k_default))),
        "stage2_precise_ablation": False,
        "stage2_enable_evs": False,
        "stage2_enable_delay": False,
        "stage2_enable_ris": True,
        "stage2_guarded": True,
        "stage2_strict_accept_rel": 1.0e-6,
        "ris_min_relative_improvement": 5.0e-3,
        "stage2_damping_grid": (0.0, 0.125, 0.25, 0.5, 0.75, 1.0),
        "stage2_global_safeguard": True,
        "stage2_tol": 1.0e-5,
        "stage2_ris_weight_mode": "residual_diag",
        "stage2_ris_weight_floor_rel": 5.0e-2,
        "stage2_ris_weight_clip": (0.25, 4.0),
        "stage2_ris_weight_normalize": True,
        "delay_lambda": 1.0e-2,
        "delay_num_pgd_steps": 10,
        "delay_step_scale": 0.8,
        "delay_damping": 0.8,
        "delay_geometry_rho": 0.2,
        "assignment_clock_weight": 0.5,
        "assignment_clock_scale_s": 1.0e-9,
        "delay_refine_phase_span": 0.35,
        "delay_refine_phase_grid": 9,
        "vp_max_nfev": 10,
        "vp_max_iter": 10,
        "global_vp": {
            "backend": "cpu",
            "gpu_device": 0,
            "gpu_dtype": "complex128",
            "gpu_keep_arrays_on_device": True,
            "gpu_transfer_policy": "scalar_grad_only",
            "validate_gpu_against_cpu": False,
            "solver": "least_squares",
            "mode": "adaptive_jones",
            "max_iter": 80,
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
            "beta_reg": 0.0,
            "evs_mode": "legacy_or_full_polarization",
            "jones_regularization_scaling": "gram",
            "jones_lambda0": 1.0,
            "jones_lambda_min": 1.0e-4,
            "jones_lambda_max": 1.0e8,
            "jones_snr_eps": 1.0e-12,
            "run_fixed_pol_anchor": True,
            "jones_leakage_threshold": 0.25,
            "jones_min_rel_improvement": 1.0e-3,
            "jones_tau": 0.25,
            "jones_tau_min": 1.0e-3,
            "jones_tau_max": 10.0,
            "jones_diagonal_loading": 1.0e-10,
            "gof_pfa": 0.05,
            "efim_lambda_min_threshold": 1.0e-8,
            "efim_cond_threshold": 1.0e12,
            "use_data_only_efim_gate": True,
            "use_delay_prior": False,
            "delay_prior_weight": 1.0,
            "delay_prior_sigma_s": 2.0e-11,
            "use_weight": False,
            "use_multistart": False,
            "num_perturb_starts": 0,
            "position_perturb_std_m": 0.05,
            "clock_perturb_std_s": 1.0e-10,
            "use_trust_region": False,
            "position_trust_radius_m": 0.3,
            "clock_trust_radius_s": 3.0e-10,
            "objective_rollback_tolerance": 1.0e-12,
            "overwrite_factor_keys": False,
            "finite_difference_check": False,
            "use_analytic_jacobian": True,
            "matrix_free_beta": False,
            "vp_dictionary_mode": "matrix_free",
            "vp_debug_compare_explicit": False,
            "vp_debug_compare_max_evals": 3,
            "enable_z_rescue_multistart": True,
            "z_rescue_num_starts": 7,
            "z_rescue_trigger": "boundary_or_unreliable",
            "z_rescue_keep_xy": True,
            "z_rescue_margin_m": 0.02,
            "boundary_tol_m": 0.02,
            "boundary_accept_rel_tol": 1.0e-3,
        },
        "crb": {
            "enable_unscaled_efim_cache": False,
        },
        "baselines": {
            "backend_config": {
                "backend": "cpu",
                "gpu_device": 0,
                "gpu_batch_size": None,
                "cpu_batch_size": None,
                "cache_enabled": True,
                "cache_memory_budget_gb": None,
                "gpu_memory_fraction": None,
                "dtype": "complex128",
            },
            "ff_omp": {
                "batch_size": 256,
                "max_batch_memory_mb": 256.0,
                "offgrid_refinement": True,
            },
            "ris_momp": {
                "batch_size": 256,
                "max_batch_memory_mb": 256.0,
                "offgrid_refinement": True,
                "local_refinement": True,
                "refinement_levels": 2,
                "refinement_shrink": 0.5,
            },
            "nf_mmpsr": {
                "batch_size": 64,
                "max_batch_memory_mb": 256.0,
                "offgrid_refinement": True,
                "top_candidates": 8,
                "local_refinement": True,
                "refinement_levels": 3,
                "refinement_shrink": 0.5,
                "local_position_grid_shape": (3, 3, 3),
                "local_clock_grid_size": 5,
            },
            "nf_ris_groupomp_localgrid_wls": {
                "direction_grid_size": 9,
                "delay_grid_size": 9,
                "batch_size": 64,
                "max_batch_memory_mb": 64.0,
                "use_subris": True,
                "subris_shape": (2, 2),
                "l1_tau_lambda": 0.0,
                "local_refinement": True,
                "coarse_to_nf_refinement_levels": 2,
                "refinement_shrink": 0.5,
                "local_grid_enabled": True,
                "local_grid_iterations": 5,
                "local_grid_refinement_levels": 2,
                "local_grid_position_radius_m": "auto",
                "local_grid_clock_radius_ns": 2.0,
                "local_grid_shrink": 0.5,
                "wls_enabled": True,
                "wls_lambda_ray": 1.0,
                "wls_lambda_tau": 1.0,
            },
        },
        "eps": 1.0e-10,
    }
    return apply_stage1_init_preset(config, config["stage1_init_mode"])
