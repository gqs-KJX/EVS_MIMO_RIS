import argparse

import numpy as np
import pytest

from src.config import default_config
from src.experiments import run_paper_ablation_figures as figures


def test_import_run_paper_ablation_figures():
    assert figures.FIELDNAMES


def test_snr_grid_parser_default_grid():
    assert figures.parse_snr_grid("-30,-25,-20,-15,-10,-5,0,5,10") == [
        -30.0,
        -25.0,
        -20.0,
        -15.0,
        -10.0,
        -5.0,
        0.0,
        5.0,
        10.0,
    ]


def test_fig1_variant_specs_are_exact():
    specs = figures._variant_specs("fig1")
    assert list(specs) == [
        "stage1_only",
        "fixed_pol_vp",
        "free_jones_vp",
        "regularized_jones_vp",
        "adaptive_jones_vp_proposed",
    ]
    assert specs["adaptive_jones_vp_proposed"]["proposed_stage2_policy"] == "ngc_certified_ris_only"
    assert specs["adaptive_jones_vp_proposed"]["stage2_adaptive"] is True
    assert specs["adaptive_jones_vp_proposed"]["stage2_rescue_type"] == "ris_only"
    assert specs["adaptive_jones_vp_proposed"]["_allow_stage2"] is True
    assert specs["adaptive_jones_vp_proposed"]["rescue_accept_min_rel_improvement"] == 0.0
    assert specs["adaptive_jones_vp_proposed"]["rescue_accept_min_abs_improvement"] == 1.0e-8
    diagnostics = figures._diagnostic_variant_specs("fig1")
    assert "adaptive_jones_vp_proposed_old_gated" in diagnostics
    assert "adaptive_jones_vp_proposed_ngc" not in diagnostics
    assert "adaptive_jones_vp_proposed_force_lower_raw" in diagnostics
    assert (
        diagnostics["adaptive_jones_vp_proposed_force_lower_raw"][
            "proposed_stage2_policy"
        ]
        == "force_ris_only"
    )
    assert (
        diagnostics["adaptive_jones_vp_proposed_force_lower_raw"][
            "rescue_accept_min_rel_improvement"
        ]
        == 0.0
    )
    assert (
        diagnostics["adaptive_jones_vp_proposed_old_gated"]["proposed_stage2_policy"]
        == "reliability_gated_ris_only"
    )
    assert (
        diagnostics["adaptive_jones_vp_proposed_old_gated"][
            "rescue_accept_min_rel_improvement"
        ]
        == 1.0e-3
    )


def test_ngc_diagnostic_variant_requires_include_flag():
    with pytest.raises(ValueError, match="include-diagnostic-variants"):
        figures._variants_for_figure(
            "fig1",
            figures.parse_variant_filter("adaptive_jones_vp_proposed_force_lower_raw"),
        )
    variants = figures._variants_for_figure(
        "fig1",
        figures.parse_variant_filter("adaptive_jones_vp_proposed_force_lower_raw"),
        include_diagnostic_variants=True,
    )
    assert list(variants) == ["adaptive_jones_vp_proposed_force_lower_raw"]


def test_variant_filter_defaults_to_none():
    args = figures.parse_args([])
    assert args.variant_filter is None
    assert args.variant_filter_values is None


def test_vp_dictionary_mode_cli_override_without_backend():
    args = figures.parse_args(
        ["--vp-dictionary-mode", "matrix_free", "--vp-debug-compare-explicit"]
    )
    overrides = figures.global_vp_cli_overrides(args)
    assert overrides["vp_dictionary_mode"] == "matrix_free"
    assert overrides["vp_debug_compare_explicit"] is True
    assert "backend" not in overrides


def test_variant_filter_keeps_named_variant_and_peb_alias():
    variants = figures._variants_for_figure(
        "fig1",
        figures.parse_variant_filter("free_jones_vp,peb_only"),
    )
    assert list(variants) == ["free_jones_vp", "PEB", "constrained_jones_peb"]


def test_variant_filter_does_not_apply_outside_fig1_fig2():
    variants = figures._variants_for_figure(
        "fig4",
        figures.parse_variant_filter("free_jones_vp"),
    )
    assert list(variants) == [
        "scalar_peb",
        "dual_pol_peb",
        "full_6d_evs_peb",
        "full_6d_constrained_jones_peb",
    ]


def test_fig3_variant_specs_include_receiver_modes():
    specs = figures._variant_specs("fig3")
    assert "scalar_receiver" in specs
    assert "dual_pol_receiver" in specs
    assert "full_6d_evs" in specs


def _tiny_nested_receiver_config():
    config = default_config()
    config.update(
        {
            "seed": 321,
            "K": 1,
            "M_A": 1,
            "ris_shape": (2, 2),
            "N": 5,
            "P": 3,
            "T": 4,
            "SNR_dB": 0.0,
            "receiver_mode": "full_6d",
            "ris_centers": np.array([[4.2, -2.2, 1.05]]),
            "print_progress": False,
        }
    )
    return config


def test_nested_receiver_modes_share_reference_noise_and_base_hash():
    config = _tiny_nested_receiver_config()
    base = figures._make_data(config)
    nested = [
        figures.make_nested_receiver_mode_data(base, mode, config)
        for mode in ("scalar", "dual_pol", "full_6d")
    ]
    assert {data["noise_variance"] for data in nested} == {
        base["noise_variance"]
    }
    assert {data["reference_sigma2"] for data in nested} == {
        base["noise_variance"]
    }
    assert len({data["nested_base_y_noisy_hash"] for data in nested}) == 1
    assert all(
        data["nested_receiver_noise_convention"]
        == figures.NESTED_RECEIVER_NOISE_CONVENTION
        for data in nested
    )
    scalar_mask = nested[0]["scene"]["evs_observation_mask"]
    dual_mask = nested[1]["scene"]["evs_observation_mask"]
    full_mask = nested[2]["scene"]["evs_observation_mask"]
    assert np.all(scalar_mask <= dual_mask)
    assert np.all(dual_mask <= full_mask)
    assert np.array_equal(
        nested[0]["Y_noisy"][scalar_mask],
        base["Y_noisy"][scalar_mask],
    )
    assert np.array_equal(
        nested[1]["Y_noisy"][dual_mask],
        base["Y_noisy"][dual_mask],
    )


def test_nested_receiver_peb_rows_share_reference_sigma2(monkeypatch):
    config = _tiny_nested_receiver_config()
    base = figures._make_data(config)
    monkeypatch.setattr(
        figures,
        "_peb_from_efim",
        lambda data, cfg: {
            "peb_position_m": 1.0,
            "peb_scalar_m": 1.0 if cfg["receiver_mode"] == "scalar" else np.nan,
            "peb_dual_m": 1.0 if cfg["receiver_mode"] == "dual_pol" else np.nan,
            "peb_evs_m": 1.0 if cfg["receiver_mode"] == "full_6d" else np.nan,
            "warning": "",
        },
    )
    figures._PEB_CACHE.clear()
    rows = []
    for mode in ("scalar", "dual_pol", "full_6d"):
        mode_config = {**config, "receiver_mode": mode}
        data = figures.make_nested_receiver_mode_data(base, mode, mode_config)
        result = figures._peb_metrics_result_for_config(
            mode_config, None, data
        )
        rows.append(figures.extract_metrics(result, 0.1))
    assert {row["reference_sigma2"] for row in rows} == {
        base["noise_variance"]
    }
    assert {row["nested_base_y_noisy_hash"] for row in rows} == {
        nested_hash := figures._hash_array(base["Y_noisy"])
    }
    assert nested_hash


def test_fig5_variant_specs_include_gate_variants():
    specs = figures._variant_specs("fig5")
    assert list(specs) == [
        "direct_vp",
        "old_gated",
        "adaptive_jones_vp_proposed",
        "force_rescue",
        "oracle_init_vp",
    ]
    assert specs["direct_vp"]["_allow_stage2"] is False
    assert (
        specs["old_gated"]["proposed_stage2_policy"]
        == "reliability_gated_ris_only"
    )
    assert (
        specs["adaptive_jones_vp_proposed"]["proposed_stage2_policy"]
        == "ngc_certified_ris_only"
    )
    assert specs["force_rescue"]["proposed_stage2_policy"] == "force_ris_only"
    assert specs["oracle_init_vp"]["_runner"] == "oracle_init_vp"


def test_figure6_k_grid():
    assert figures.FIGURE6_K_GRID == [1, 2, 3, 4]


def test_fig6_vp_family_proposed_uses_ngc():
    specs = figures._variant_specs("fig6")
    adaptive_no_rescue = specs["adaptive_jones_no_rescue"]
    assert adaptive_no_rescue["global_vp"]["mode"] == "adaptive_jones"
    assert adaptive_no_rescue["_allow_stage2"] is False
    proposed = specs["adaptive_jones_vp_proposed"]
    assert proposed["global_vp"]["mode"] == "adaptive_jones"
    assert proposed["stage2_adaptive"] is True
    assert proposed["stage2_rescue_type"] == "ris_only"
    assert proposed["proposed_stage2_policy"] == "ngc_certified_ris_only"
    assert proposed["_allow_stage2"] is True
    assert proposed["rescue_accept_min_rel_improvement"] == 0.0
    assert proposed["rescue_accept_min_abs_improvement"] == 1.0e-8


def test_default_paper_k_is_three():
    args = figures.parse_args([])
    assert args.paper_k == 3


def test_fig1_to_fig5_config_generation_applies_paper_k():
    for figure in ["fig1", "fig2", "fig3", "fig4", "fig5"]:
        config = figures._config_for_point(
            figure=figure,
            variant_updates={},
            seed=1,
            snr_db=-10.0,
            x_value=-10.0,
            paper_k=3,
        )
        assert config["K"] == 3


def test_fig6_uses_k_grid_and_ignores_paper_k():
    args = figures.parse_args([])
    assert args.k_grid_values == [1, 2, 3, 4]
    assert figures._figure_x_grid("fig6", [-30.0], args.k_grid_values) == (
        "K",
        [1.0, 2.0, 3.0, 4.0],
    )
    config = figures._config_for_point(
        figure="fig6",
        variant_updates={},
        seed=1,
        snr_db=-30.0,
        x_value=4.0,
        paper_k=3,
    )
    assert config["K"] == 4
    assert config["ris_centers"].shape[0] >= 4


def test_task_rows_expose_requested_and_effective_k(tmp_path):
    args = figures.parse_args(["--out-dir", str(tmp_path)])
    variants = {"fixed_pol_vp": figures._variant_specs("fig1")["fixed_pol_vp"]}
    fig1_tasks = figures._tasks_for_figure(
        figure=figures.FIG1_FIG2_SHARED_FIGURE,
        grouped_group=None,
        x_name="snr_db",
        x_values=[0.0],
        variants=variants,
        trial_seeds=[123],
        args=args,
    )
    assert fig1_tasks[0]["effective_K"] == fig1_tasks[0]["paper_k"] == 3

    fig6_tasks = figures._tasks_for_figure(
        figure="fig6",
        grouped_group=None,
        x_name="K",
        x_values=[4.0],
        variants={"fixed_pol_vp": figures._variant_specs("fig6")["fixed_pol_vp"]},
        trial_seeds=[123],
        args=args,
    )
    assert fig6_tasks[0]["effective_K"] == int(fig6_tasks[0]["x_value"]) == 4


def test_grouped_fig1_task_records_filtered_variants(tmp_path):
    args = figures.parse_args(
        [
            "--figures",
            "fig1",
            "--variant-filter",
            "free_jones_vp,data_only_peb",
            "--out-dir",
            str(tmp_path),
        ]
    )
    variants = figures._variants_for_figure(
        figures.FIG1_FIG2_SHARED_FIGURE,
        args.variant_filter_values,
    )
    tasks = figures._tasks_for_figure(
        figure=figures.FIG1_FIG2_SHARED_FIGURE,
        grouped_group="fig1_fig2",
        x_name="snr_db",
        x_values=[0.0],
        variants=variants,
        trial_seeds=[123],
        args=args,
    )
    assert len(tasks) == 1
    assert tasks[0]["selected_variants"] == ["free_jones_vp", "PEB"]


def test_peb_variants_map_to_peb_position():
    row = {"failed": False, "peb_position_m": "1.0"}
    assert figures.get_plot_metric(row, "fig1", "PEB") == "peb_position_m"
    assert figures.get_plot_metric(row, "fig3", "full_6d_evs_peb") == "peb_position_m"
    assert figures.get_plot_metric(row, "fig6", "proposed_peb") == "peb_position_m"


def test_fig4_peb_variants_map_to_specific_fields():
    assert (
        figures.get_plot_metric(
            [{"failed": False, "peb_scalar_m": "1.0", "peb_position_m": "2.0"}],
            "fig4",
            "scalar_peb",
        )
        == "peb_scalar_m"
    )
    assert (
        figures.get_plot_metric(
            [{"failed": False, "peb_dual_m": "1.0", "peb_position_m": "2.0"}],
            "fig4",
            "dual_pol_peb",
        )
        == "peb_dual_m"
    )
    assert (
        figures.get_plot_metric(
            [{"failed": False, "peb_evs_m": "1.0", "peb_position_m": "2.0"}],
            "fig4",
            "full_6d_evs_peb",
        )
        == "peb_evs_m"
    )


def test_summary_uses_plot_metric_name_for_peb_and_outlier():
    rows = [
        {
            "figure": "fig1",
            "variant": "PEB",
            "x_value": "-30",
            "failed": "False",
            "outlier_flag": "False",
            "peb_position_m": "0.5",
        },
        {
            "figure": "fig5",
            "variant": "direct_vp",
            "x_value": "-30",
            "failed": "False",
            "outlier_flag": "True",
        },
    ]
    fig1 = figures.summarize_rows(rows[:1], "fig1")
    fig5 = figures.summarize_rows(rows[1:], "fig5")
    assert fig1[0]["plot_metric_name"] == "peb_position_m"
    assert fig1[0]["plot_y_mean"] == 0.5
    assert fig5[0]["plot_metric_name"] == "outlier_flag_mean"
    assert fig5[0]["plot_y_mean"] == 1.0


def test_summary_carries_unique_k_metadata():
    rows = [
        {
            "figure": "fig1",
            "variant": "fixed_pol_vp",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "position_rmse_m": "1.0",
            "K": "3",
            "paper_k": "3",
            "effective_K": "3",
        }
    ]
    summary = figures.summarize_rows(rows, "fig1")
    assert summary[0]["K"] == 3
    assert summary[0]["paper_k"] == 3
    assert summary[0]["effective_K"] == 3


def test_summary_carries_ngc_rescue_run_rate():
    rows = [
        {
            "figure": "fig1",
            "variant": "adaptive_jones_vp_proposed",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "position_rmse_m": "1.0",
            "ngc_policy_active": "True",
            "ngc_rescue_requested": "True",
        },
        {
            "figure": "fig1",
            "variant": "adaptive_jones_vp_proposed",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "position_rmse_m": "2.0",
            "ngc_policy_active": "True",
            "ngc_rescue_requested": "False",
        },
        {
            "figure": "fig1",
            "variant": "fixed_pol_vp",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "position_rmse_m": "3.0",
            "ngc_policy_active": "False",
            "ngc_rescue_requested": "False",
        },
    ]
    summary = figures.summarize_rows(rows, "fig1")
    by_variant = {row["variant"]: row for row in summary}
    assert by_variant["adaptive_jones_vp_proposed"]["rescue_run_rate"] == 0.5
    assert np.isnan(by_variant["fixed_pol_vp"]["rescue_run_rate"])


def test_summary_carries_fig5_rescue_trigger_rate():
    rows = [
        {
            "figure": "fig5",
            "variant": "adaptive_jones_vp_proposed",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "ngc_policy_active": "True",
            "ngc_rescue_requested": "True",
        },
        {
            "figure": "fig5",
            "variant": "adaptive_jones_vp_proposed",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "ngc_policy_active": "True",
            "ngc_rescue_requested": "False",
        },
        {
            "figure": "fig5",
            "variant": "old_gated",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "proposed_stage2_policy": "reliability_gated_ris_only",
            "rescue_candidate_available": "True",
        },
        {
            "figure": "fig5",
            "variant": "old_gated",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "proposed_stage2_policy": "reliability_gated_ris_only",
            "rescue_candidate_available": "False",
        },
        {
            "figure": "fig5",
            "variant": "force_rescue",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "proposed_stage2_policy": "force_ris_only",
        },
        {
            "figure": "fig5",
            "variant": "direct_vp",
            "x_value": "0",
            "failed": "False",
            "outlier_flag": "False",
            "selected_branch": "direct_vp",
        },
    ]
    summary = figures.summarize_rows(rows, "fig5")
    by_variant = {row["variant"]: row for row in summary}
    assert by_variant["adaptive_jones_vp_proposed"]["rescue_trigger_rate"] == 0.5
    assert by_variant["old_gated"]["rescue_trigger_rate"] == 0.5
    assert by_variant["force_rescue"]["rescue_trigger_rate"] == 1.0
    assert by_variant["direct_vp"]["rescue_trigger_rate"] == 0.0


def test_cache_reuse_refuses_ten_trial_csv_when_requesting_fifty():
    args = argparse.Namespace(
        n_trials=50,
        paper_k=3,
        k_grid_values=[1, 2, 3, 4],
        seed=20260526,
    )
    rows = []
    for trial_id in range(10):
        for variant in figures._expected_variant_names("fig1"):
            rows.append(
                {
                    "figure": "fig1",
                    "variant": variant,
                    "trial_id": str(trial_id),
                    "x_name": "snr_db",
                    "x_value": "-30.0",
                    "K": "3",
                    "paper_k": "3",
                    "effective_K": "3",
                    "failed": "False",
                }
            )
    assert not figures._csv_matches_request(rows, "fig1", args, [-30.0])


def test_stale_csv_without_k_columns_is_rejected(tmp_path):
    args = figures.parse_args(
        [
            "--figures",
            "fig1",
            "--n-trials",
            "1",
            "--snr-grid",
            "0",
            "--out-dir",
            str(tmp_path),
            "--reuse-existing",
        ]
    )
    trial_csv = tmp_path / "stale.csv"
    figures._write_csv(
        trial_csv,
        [
            {
                "figure": "fig1",
                "variant": "fixed_pol_vp",
                "x_name": "snr_db",
                "x_value": 0,
            }
        ],
        ["figure", "variant", "x_name", "x_value"],
    )
    can_reuse, _ = figures._can_reuse_csv(
        trial_csv,
        "fig1",
        args,
        [0.0],
        ["fig1"],
        existing_metadata={},
    )
    assert not can_reuse


def test_csv_fieldnames_include_required_diagnostics():
    assert "selected_vp_family_branch" in figures.FIELDNAMES
    assert "lambda_jones_per_path" in figures.FIELDNAMES
    assert "data_only_scaled_efim_condition_number" in figures.FIELDNAMES
    assert "peb_position_m" in figures.FIELDNAMES
    assert "ngc_policy_active" in figures.FIELDNAMES
    assert "ngc_selected_by" in figures.FIELDNAMES
    assert "ngc_threshold_clock_red" in figures.FIELDNAMES
    assert {"K", "paper_k", "effective_K", "num_ris_paths", "receiver_mode", "config_seed"} <= set(
        figures.FIELDNAMES
    )


def test_jobs_cli_default_and_override():
    assert figures.parse_args([]).jobs == 10
    assert figures.parse_args(["--jobs", "1"]).jobs == 1
    assert figures.parse_args(["--max-workers", "2"]).max_workers == 2
