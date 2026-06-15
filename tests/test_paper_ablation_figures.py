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


def test_csv_fieldnames_include_required_diagnostics():
    assert "selected_vp_family_branch" in figures.FIELDNAMES
    assert "lambda_jones_per_path" in figures.FIELDNAMES
    assert "data_only_scaled_efim_condition_number" in figures.FIELDNAMES
    assert "peb_position_m" in figures.FIELDNAMES


def test_jobs_cli_default_and_override():
    assert figures.parse_args([]).jobs == 10
    assert figures.parse_args(["--jobs", "1"]).jobs == 1
