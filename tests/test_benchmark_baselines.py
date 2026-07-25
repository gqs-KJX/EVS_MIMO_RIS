import numpy as np

from src.baselines import (
    als_cpd,
    common,
    nf_ris_groupomp_localgrid_wls,
    ris_vbi_sbl,
)
from src.config import default_config
from src.main_single_proposed import _make_data
from src.experiments import run_benchmark_comparison as bench
from src.experiments import run_robustness_and_scaling_figures as robust
from src.experiments.run_paper_ablation_figures import (
    _constrained_jones_design,
    _peb_from_efim,
    peb_cache_key,
    position_peb_from_global_efim,
)


def _tiny_config():
    config = default_config()
    config.update(
        {
            "seed": 123,
            "K": 1,
            "M_A": 1,
            "ris_shape": (2, 2),
            "N": 5,
            "P": 3,
            "T": 4,
            "SNR_dB": 60.0,
            "receiver_mode": "full_6d",
            "print_progress": False,
            "p_u_true": np.array([1.2, 0.4, 0.8]),
            "ris_centers": np.array([[4.2, -2.2, 1.05]]),
            "ue_bounds": np.array([[1.0, 1.4], [0.2, 0.6], [0.6, 1.0]]),
            "delta_t_true": 5.0e-9,
            "delta_t_bounds": np.array([4.0e-9, 6.0e-9]),
        }
    )
    config["baselines"] = {
        "als_cpd": {"max_iter": 5, "position_grid_shape": (3, 3, 3)},
        "ris_vbi_sbl": {
            "nf_grid_x": 3,
            "nf_grid_y": 3,
            "nf_grid_z": 3,
            "delay_grid_size": 5,
            "vbi_max_iter": 5,
            "vbi_refine_maxiter": 20,
        },
        "nf_ris_groupomp_localgrid_wls": {
            "direction_grid_size": 3,
            "range_grid_size": 3,
            "delay_grid_size": 3,
            "max_groups": 1,
            "cpd_max_iter": 10,
            "sage_enabled": True,
            "sage_iterations": 1,
            "sage_maxiter": 3,
            "wls_enabled": True,
            "wls_max_nfev": 10,
        },
    }
    return config


def test_import_all_benchmark_baseline_modules():
    assert common.BaselineResult
    assert als_cpd.run_als_cpd_baseline
    assert ris_vbi_sbl.run_ris_vbi_sbl_baseline
    assert nf_ris_groupomp_localgrid_wls.run_nf_ris_groupomp_localgrid_wls_baseline


def test_benchmark_default_baselines_include_targeted_nf_and_constrained_peb():
    baselines = bench.parse_baselines(bench.DEFAULT_BASELINES)
    if "peb" in baselines and "constrained_jones_peb" not in baselines:
        baselines.append("constrained_jones_peb")
    assert baselines == [
        "als_cpd",
        "scaled_4d",
        "nf_ris_groupomp_localgrid_wls",
        "ris_vbi_sbl",
        "mksc_ccop",
        "peb",
        "constrained_jones_peb",
    ]


def test_proposed_trace_diagnostics_are_csv_visible():
    config = _tiny_config()
    data = _make_data(config)
    trace = common.proposed_trace_diagnostics(
        {
            "final": {
                "selected_branch": "ris_only_stage2_then_vp",
                "reliability": {
                    "proposed_stage2_policy": "ngc_certified_ris_only"
                },
            },
            "ngc_policy_active": True,
            "ngc_rescue_requested": True,
            "ngc_selected_by": "ngc_certified_candidate",
            "ngc_final_unreliable": False,
        }
    )
    result = common.BaselineResult(
        name="proposed",
        p_u=np.asarray(data["scene"]["p_u_true"], dtype=float),
        delta_t=float(data["scene"]["delta_t_true"]),
        Y_hat=np.asarray(data["Y_true"]),
        raw_objective_final=0.0,
        diagnostics={"dictionary_mode": "proposed_ngc_adaptive_jones_vp", **trace},
    )
    row = common.make_baseline_row(result, data, config, baseline="proposed")
    fields = (
        "selected_branch",
        "proposed_stage2_policy",
        "ngc_policy_active",
        "ngc_rescue_requested",
        "rescue_requested",
        "ngc_selected_by",
        "ngc_final_unreliable",
    )
    for field in fields:
        assert field in row
        assert field in bench.FIELDNAMES
        assert field in robust.FIELDNAMES
    assert row["selected_branch"] == "ris_only_stage2_then_vp"
    assert row["proposed_stage2_policy"] == "ngc_certified_ris_only"
    assert row["ngc_rescue_requested"] is True


def test_external_clock_uses_raw_panel_delays_and_frozen_median():
    p_hat = np.array([1.0, 0.5, 0.8])
    centers = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    d_rb = np.array([2.0, 2.5, 3.0])
    c0 = 299_792_458.0
    panel_clock_ns = np.array([4.0, 5.0, 9.0])
    supports = []
    for panel, clock_ns in enumerate(panel_clock_ns):
        geometric_s = (
            np.linalg.norm(p_hat - centers[panel]) + d_rb[panel]
        ) / c0
        supports.append(
            {
                "panel": panel,
                "tau": geometric_s + clock_ns * 1.0e-9,
            }
        )
    data = {
        "scene": {
            "K": 3,
            "p_u_true": p_hat.copy(),
            "delta_t_true": 5.0e-9,
            "ris_centers": centers,
            "d_RB": d_rb,
            "c0": c0,
        },
        "true_components": {},
        "Y_true": np.ones((1, 1, 1), dtype=complex),
        "Y_noisy": np.ones((1, 1, 1), dtype=complex),
        "noise_variance": 0.0,
    }
    config = {
        "seed": 1,
        "SNR_dB": 0.0,
        "K": 3,
        "benchmark_clock_catastrophic_threshold_ns": 1.0,
    }
    result = common.BaselineResult(
        name="als_cpd",
        p_u=p_hat,
        delta_t=6.5e-9,
        Y_hat=data["Y_true"].copy(),
        raw_objective_final=0.0,
        selected_support=supports,
    )
    row = common.make_baseline_row(result, data, config)

    assert np.isclose(row["clock_estimate_ns"], 5.0)
    assert np.isclose(row["clock_native_estimate_ns"], 6.5)
    assert np.isclose(row["clock_error_ns"], 0.0, atol=1.0e-12)
    assert np.isclose(row["clock_panel_mad_ns"], 1.0)
    assert row["clock_num_panels"] == 3
    assert row["clock_complete_panel_set"] is True
    assert row["clock_invalid"] is False
    assert row["clock_catastrophic"] is False
    assert "no_clipping" in row["clock_extraction_rule"]
    for field in (
        "clock_estimate_ns",
        "clock_native_estimate_ns",
        "clock_error_ns",
        "clock_invalid",
        "clock_catastrophic",
        "clock_extraction_rule",
    ):
        assert field in bench.FIELDNAMES


def test_native_joint_clock_semantics_uses_result_delta_t():
    config = _tiny_config()
    data = _make_data(config)
    result = common.BaselineResult(
        name="ris_vbi_sbl",
        p_u=np.asarray(data["scene"]["p_u_true"], dtype=float),
        delta_t=5.25e-9,
        Y_hat=np.asarray(data["Y_true"]),
        raw_objective_final=0.0,
        selected_support=[
            {"panel": 0, "tau": 100.0e-9},
        ],
        diagnostics={
            "clock_output_semantics": "native_joint_common_clock",
        },
    )

    row = common.make_baseline_row(result, data, config)

    assert np.isclose(row["clock_estimate_ns"], 5.25)
    assert np.isclose(
        row["clock_estimate_ns"], row["clock_native_estimate_ns"]
    )
    assert np.isclose(row["clock_error_ns"], 0.25)
    assert row["clock_extraction_rule"] == "native_common_clock_parameter"
    assert row["clock_delay_source"] == "baseline_native_delta_t"
    assert row["clock_num_panels"] == 1
    assert np.isclose(row["clock_panel_mad_ns"], 0.0)
    assert row["clock_panel_estimates_ns"] != "[]"


def test_linear_ls_fit_recovers_known_complex_coefficients():
    rng = np.random.default_rng(1)
    Phi = rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
    coeff_true = np.array([1.2 - 0.3j, -0.7 + 0.5j])
    y = Phi @ coeff_true
    coeff, y_hat, residual = common.linear_ls_fit(Phi, y)
    assert np.allclose(coeff, coeff_true)
    assert np.linalg.norm(residual) < 1.0e-10
    assert np.allclose(y_hat, y)


def test_group_projection_score_uses_subspace_energy():
    group = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=complex)
    residual = np.array([0.0, 1.0, 0.0], dtype=complex)
    score = common.group_projection_score(group, residual)
    single_column_score = abs(np.vdot(common.simple_atom_normalize(group[:, 0]), residual)) ** 2
    assert score > single_column_score
    assert np.isclose(score, 1.0)


def test_complex_cp_als_reconstructs_tiny_rank_two_tensor():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    b = rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2))
    c = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    weights = np.array([1.0 + 0.2j, -0.6 + 0.4j])
    tensor = als_cpd.reconstruct_cp_tensor([a, b, c], weights)
    factors, est_weights, diagnostics = als_cpd.complex_cp_als(
        tensor,
        2,
        max_iter=300,
        tol=1.0e-9,
        reg=1.0e-10,
    )
    recon = als_cpd.reconstruct_cp_tensor(factors, est_weights)
    rel_residual = np.linalg.norm(tensor - recon) / np.linalg.norm(tensor)
    assert rel_residual < 1.0e-3
    assert diagnostics["rank"] == 2
    assert diagnostics["als_matlab_compatible"] is True


def test_als_update_factor_matches_complex_als_translation():
    rng = np.random.default_rng(3)
    tensor = rng.normal(size=(3, 4, 2)) + 1j * rng.normal(size=(3, 4, 2))
    b = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    c = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    pi = als_cpd._khatri_rao(c, b)
    sigma2 = 1.0e-4
    actual = als_cpd.matlab_compatible_update_factor(tensor, pi, 0, sigma2)
    z_fold = als_cpd._unfold(tensor, 0)
    expected = np.conj(
        (np.conj(z_fold) @ pi)
        @ np.linalg.pinv(pi.conj().T @ pi + sigma2 * np.eye(2, dtype=complex))
    )
    assert np.allclose(actual, expected)


def test_mttkrp_updates_match_explicit_khatri_rao_updates():
    rng = np.random.default_rng(31)
    tensor = rng.normal(size=(4, 5, 3)) + 1j * rng.normal(size=(4, 5, 3))
    factors = [
        rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)),
        rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2)),
        rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2)),
    ]
    sigma2 = 1.0e-6
    khatri_rao_pairs = (
        (factors[2], factors[1]),
        (factors[2], factors[0]),
        (factors[1], factors[0]),
    )
    for mode, (left, right) in enumerate(khatri_rao_pairs):
        explicit = als_cpd.matlab_compatible_update_factor(
            tensor, als_cpd._khatri_rao(left, right), mode, sigma2
        )
        mttkrp = als_cpd._mttkrp_update_factor(
            tensor, factors, mode, sigma2
        )
        assert np.allclose(mttkrp, explicit, rtol=1.0e-11, atol=1.0e-12)


def test_als_geometry_mapping_uses_common_position_and_unique_panels():
    config = _tiny_config()
    config["K"] = 2
    config["ris_centers"] = np.array(
        [[4.2, -2.2, 1.05], [4.3, 2.1, 1.15]], dtype=float
    )
    config["baselines"]["als_cpd"].update(
        {
            "position_grid_shape": (3, 3, 3),
            "geometry_offgrid_refinement": False,
        }
    )
    scene = _make_data(config)["scene"]
    truth = np.asarray(config["p_u_true"], dtype=float)
    training_factor = np.column_stack(
        [
            common.training_response_from_position(scene, 1, truth),
            common.training_response_from_position(scene, 0, truth),
        ]
    )
    taus = np.asarray(
        [
            (
                np.linalg.norm(truth - scene["ris_centers"][panel])
                + scene["d_RB"][panel]
            )
            / scene["c0"]
            + 5.0e-9
            for panel in (1, 0)
        ],
        dtype=float,
    )
    supports, diagnostics = als_cpd._joint_match_training_factors_to_geometry(
        training_factor,
        taus,
        scene,
        config,
    )

    assert [support["panel"] for support in supports] == [1, 0]
    assert all(np.array_equal(support["position"], truth) for support in supports)
    assert diagnostics["als_geometry_unique_panel_count"] == 2
    assert diagnostics["als_geometry_mapping"].startswith("joint_common_position")
    assert diagnostics["als_geometry_refined_clock_std_ns"] < 1.0e-6


def test_ris_vbi_sbl_runs_and_reports_bayesian_structure():
    config = _tiny_config()
    data = _make_data(config)
    data["Y_noisy"] = data["Y_true"].copy()
    result = ris_vbi_sbl.run_ris_vbi_sbl_baseline(data, config)
    assert result.diagnostics["model_variant"] == "variational_bayesian_sbl_adaptation"
    assert result.diagnostics["dictionary_mode"] == "ris_vbi_sbl_near_field_per_panel"
    assert (
        result.diagnostics["clock_output_semantics"]
        == "native_joint_common_clock"
    )
    assert result.diagnostics["unique_panel_constraint"] is True
    assert np.all(np.isfinite(result.p_u))
    assert np.isfinite(result.delta_t)
    assert result.Y_hat.shape == data["Y_noisy"].shape
    row = common.make_baseline_row(result, data, config)
    assert np.isclose(
        row["clock_estimate_ns"], row["clock_native_estimate_ns"]
    )


def test_nf_ris_adaptation_runs_cpd_sage_and_fim_weighted_wls():
    config = _tiny_config()
    data = _make_data(config)
    data["Y_noisy"] = data["Y_true"].copy()
    result = nf_ris_groupomp_localgrid_wls.run_nf_ris_groupomp_localgrid_wls_baseline(
        data, config
    )
    assert result.diagnostics["cpd_omp_adapted_used"] is True
    assert result.diagnostics["cpd_rank1_sequential"] is True
    assert result.diagnostics["sage_enabled"] is True
    assert result.diagnostics["sage_iterations"] == 1
    assert (
        result.diagnostics["sage_final_objective"]
        <= result.diagnostics["sage_initial_objective"] * (1.0 + 1.0e-12)
    )
    assert result.diagnostics["wls_weight_model"] == "local_channel_efim_after_jones_projection"
    assert np.all(np.isfinite(result.p_u))
    assert np.isfinite(result.delta_t)
    assert result.Y_hat.shape == data["Y_noisy"].shape


def test_baseline_wrappers_do_not_call_proposed_vp(monkeypatch):
    import src.global_vp as global_vp

    def forbidden(*args, **kwargs):
        raise AssertionError("proposed VP must not be called by standalone baselines")

    monkeypatch.setattr(global_vp, "global_exact_spherical_vp_refinement", forbidden)
    config = _tiny_config()
    data = _make_data(config)
    data["Y_noisy"] = data["Y_true"].copy()

    results = [
        als_cpd.run_als_cpd_baseline(data, config),
        ris_vbi_sbl.run_ris_vbi_sbl_baseline(data, config),
        nf_ris_groupomp_localgrid_wls.run_nf_ris_groupomp_localgrid_wls_baseline(
            data, config
        ),
    ]
    for result in results:
        row = common.make_baseline_row(result, data, config)
        assert row["clock_invalid"] is False
        assert np.isfinite(row["clock_estimate_ns"])
        assert np.isfinite(row["clock_error_ns"])


def test_benchmark_plot_metric_mapping_uses_peb_for_peb():
    assert bench.get_plot_metric("peb", "rmse") == "peb_position_m"
    assert bench.get_plot_metric("peb", "nmse") is None
    assert bench.get_plot_metric("proposed", "rmse") == "position_rmse_m"


def test_benchmark_summary_distinguishes_mean_error_rmse_and_peb_rms():
    estimator_rows = [
        {
            "baseline": "proposed",
            "snr_db": -10.0,
            "failed": False,
            "position_error_m": error,
            "position_rmse_m": 999.0,
            "runtime_s": 1.0,
        }
        for error in (3.0, 4.0)
    ]
    peb_rows = [
        {
            "baseline": "peb",
            "snr_db": -10.0,
            "failed": False,
            "peb_position_m": value,
            "runtime_s": 1.0,
        }
        for value in (1.0, 3.0)
    ]
    summary = bench.summarize_rows(
        estimator_rows + peb_rows,
        outlier_threshold_m=3.5,
    )
    estimator = next(row for row in summary if row["baseline"] == "proposed")
    peb = next(row for row in summary if row["baseline"] == "peb")
    assert estimator["position_mean_error_m"] == 3.5
    assert estimator["position_rmse_m"] == np.sqrt(12.5)
    assert estimator["position_conditional_rmse_m"] == 3.0
    assert estimator["position_error_m_mean"] == 3.5
    assert estimator["position_rmse_m_mean"] == 3.5
    assert peb["peb_position_m_mean"] == 2.0
    assert peb["peb_position_m_rms"] == np.sqrt(5.0)


def test_benchmark_summary_reports_clock_tail_and_invalid_rates():
    rows = [
        {
            "baseline": "als_cpd",
            "snr_db": -10.0,
            "failed": False,
            "clock_error_ns": error,
            "clock_invalid": False,
            "clock_catastrophic": error > 1.0,
            "runtime_s": 1.0,
        }
        for error in (3.0, 4.0)
    ]
    rows.append(
        {
            "baseline": "als_cpd",
            "snr_db": -10.0,
            "failed": True,
            "clock_error_ns": float("nan"),
            "clock_invalid": True,
            "clock_catastrophic": False,
            "runtime_s": float("nan"),
        }
    )
    summary = bench.summarize_rows(rows)[0]

    assert np.isclose(summary["clock_rmse_ns"], np.sqrt(12.5))
    assert np.isclose(summary["clock_median_abs_error_ns"], 3.5)
    assert np.isclose(summary["clock_p95_abs_error_ns"], 3.95)
    assert np.isclose(summary["clock_invalid_rate"], 1.0 / 3.0)
    assert np.isclose(summary["clock_catastrophic_rate"], 1.0)
    assert np.isclose(summary["clock_catastrophic_or_invalid_rate"], 1.0)


def test_constrained_jones_design_reduces_complex_nuisance_dimension():
    rng = np.random.default_rng(123)
    k_paths = 3
    phi = rng.normal(size=(20, 2 * k_paths)) + 1j * rng.normal(
        size=(20, 2 * k_paths)
    )
    derivatives = [
        rng.normal(size=phi.shape) + 1j * rng.normal(size=phi.shape)
        for _ in range(4)
    ]
    x_true = rng.normal(size=2 * k_paths) + 1j * rng.normal(size=2 * k_paths)
    design, d_model, beta = _constrained_jones_design(
        phi, derivatives, x_true, k_paths
    )
    assert phi.shape[1] == 2 * k_paths
    assert design.shape == (phi.shape[0], k_paths)
    assert d_model.shape == (phi.shape[0], 4)
    assert beta.shape == (k_paths,)


def test_proposed_benchmark_uses_existing_data(monkeypatch):
    config = _tiny_config()
    data = _make_data(config)

    def fake_run(config_arg, allow_stage2=True, data_override=None):
        assert data_override is data
        return {
            "final": {
                "Y_hat": data["Y_true"],
                "p_u": data["scene"]["p_u_true"],
                "raw_objective_final": 0.0,
                "components": data["true_components"],
            }
        }

    monkeypatch.setattr(bench, "run_single_proposed_diagnostic", fake_run)
    row = bench._proposed_row(data, config, 0, "proposed")
    assert row["y_noisy_hash"] == common.y_noisy_hash(data)


def test_same_data_hash_validator_detects_mismatch():
    rows = [
        {"trial_id": 0, "snr_db": -20.0, "K": 1, "failed": False, "y_noisy_hash": "a"},
        {"trial_id": 0, "snr_db": -20.0, "K": 1, "failed": False, "y_noisy_hash": "b"},
    ]
    try:
        bench.validate_same_data_hashes(rows)
    except RuntimeError as exc:
        assert "same-data hash mismatch" in str(exc)
    else:
        raise AssertionError("expected same-data hash mismatch")


def test_position_only_support_does_not_use_true_clock():
    config = _tiny_config()
    config["delta_t_true"] = 2.0e-9
    config["delta_t_bounds"] = np.array([0.0, 10.0e-9])
    data = _make_data(config)
    p_candidate = np.array([1.1, 0.3, 0.7])
    p_hat, delta_t, diagnostics = common.geometric_support_to_position_ls(
        data["scene"],
        [{"panel": 0, "position": p_candidate}],
        config,
    )
    assert np.array_equal(p_hat, p_candidate)
    assert delta_t == np.mean(common.clock_grid_from_config(config, 3))
    assert delta_t != config["delta_t_true"]
    assert diagnostics["geometry_solver"] == "direct_position_candidate"


def test_peb_diagnostics_and_regularization_independence():
    config = _tiny_config()
    data = _make_data(config)
    base = _peb_from_efim(data, config)
    changed = {**config, "global_vp": {**config.get("global_vp", {}), "mode": "adaptive_jones", "jones_lambda0": 1.0e6}}
    changed["rho"] = 1.0e9
    other = _peb_from_efim(data, changed)
    assert base["peb_is_data_only"] is True
    assert base["peb_uses_regularization"] is False
    assert base["nuisance_model"] == "jones_linear"
    assert base["clock_eliminated"] is True
    assert np.allclose(base["peb_position_m"], other["peb_position_m"], equal_nan=True)


def test_position_peb_explicitly_schur_eliminates_clock():
    efim = np.array(
        [
            [5.0, 0.2, 0.0, 1.5],
            [0.2, 4.0, 0.1, -0.7],
            [0.0, 0.1, 3.0, 0.4],
            [1.5, -0.7, 0.4, 2.0],
        ]
    )
    peb = position_peb_from_global_efim(
        efim,
        ["p_x_m", "p_y_m", "p_z_m", "c_delta_t_m"],
    )
    j_pp = efim[:3, :3]
    j_pc = efim[:3, 3:4]
    j_cc = efim[3:4, 3:4]
    expected = np.sqrt(
        np.trace(np.linalg.pinv(j_pp - j_pc @ np.linalg.pinv(j_cc) @ j_pc.T))
    )
    naive = np.sqrt(np.trace(np.linalg.pinv(j_pp)))
    assert np.isclose(peb, expected)
    assert not np.isclose(peb, naive)


def test_benchmark_peb_row_records_clock_and_reference_metadata():
    config = _tiny_config()
    data = _make_data(config)
    row = bench._peb_row(data, config, trial_id=0)
    assert row["clock_eliminated"] is True
    assert row["peb_reference_type"] == "matched_model"
    assert row["peb_reference_data_hash"] == common.data_hash(data)
    assert row["peb_position_m"] == row["peb_position_m"] or np.isnan(
        row["peb_position_m"]
    )


def test_peb_cache_key_changes_with_receiver_mode_or_k():
    config = _tiny_config()
    key_base = peb_cache_key(config)
    mode_config = {**config, "receiver_mode": "scalar"}
    k_config = {**config, "K": 2, "ris_centers": np.vstack([config["ris_centers"], [5.0, 2.0, 1.0]])}
    assert peb_cache_key(mode_config) != key_base
    assert peb_cache_key(k_config) != key_base
