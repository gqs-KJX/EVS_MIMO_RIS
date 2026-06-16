import argparse

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
    assert list(figures._variant_specs("fig1")) == [
        "stage1_only",
        "fixed_pol_vp",
        "free_jones_vp",
        "regularized_jones_vp",
        "adaptive_jones_vp_proposed",
    ]


def test_fig3_variant_specs_include_receiver_modes():
    specs = figures._variant_specs("fig3")
    assert "scalar_receiver" in specs
    assert "dual_pol_receiver" in specs
    assert "full_6d_evs" in specs


def test_fig5_variant_specs_include_gate_variants():
    specs = figures._variant_specs("fig5")
    assert "direct_vp" in specs
    assert "jnpp_always" in specs
    assert "reliability_gated_proposed" in specs
    assert "oracle_init_vp" in specs


def test_figure6_k_grid():
    assert figures.FIGURE6_K_GRID == [1, 2, 3, 4]


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
                    "failed": "False",
                }
            )
    assert not figures._csv_matches_request(rows, "fig1", args, [-30.0])


def test_csv_fieldnames_include_required_diagnostics():
    assert "selected_vp_family_branch" in figures.FIELDNAMES
    assert "lambda_jones_per_path" in figures.FIELDNAMES
    assert "data_only_scaled_efim_condition_number" in figures.FIELDNAMES
    assert "peb_position_m" in figures.FIELDNAMES


def test_jobs_cli_default_and_override():
    assert figures.parse_args([]).jobs == 10
    assert figures.parse_args(["--jobs", "1"]).jobs == 1
    assert figures.parse_args(["--max-workers", "2"]).max_workers == 2
