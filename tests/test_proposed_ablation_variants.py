from oldcode.legacy_stage2 import run_proposed_ablation as ablation


def test_import_run_proposed_ablation():
    assert ablation.FIELDNAMES


def test_vp_family_specs_include_proposed_adaptive_jones():
    specs = ablation._variant_specs("vp_family")
    assert "adaptive_jones_vp_proposed" in specs
    proposed = specs["adaptive_jones_vp_proposed"]
    assert proposed["global_vp"]["mode"] == "adaptive_jones"
    assert proposed["stage2_adaptive"] is False
    assert proposed["proposed_stage2_policy"] == "reliability_gated"
    assert proposed["_allow_stage2"] is False


def test_stage2_gate_specs_include_direct_and_proposed_gated():
    specs = ablation._variant_specs("stage2_gate")
    assert "direct_vp_only" in specs
    assert "adaptive_jones_vp_proposed" in specs
    assert "adaptive_jones_vp_proposed_force_lower_raw" in specs
    assert "ris_jnpp_always_then_vp" not in specs
    assert "reliability_gated_ris_jnpp_then_vp_proposed" not in specs
    assert "adaptive_jones_vp_proposed_old_gated" in specs
    assert specs["direct_vp_only"]["proposed_stage2_policy"] == "reliability_gated"
    assert (
        specs["adaptive_jones_vp_proposed"]["proposed_stage2_policy"]
        == "ngc_certified_ris_only"
    )
    assert (
        specs["adaptive_jones_vp_proposed_force_lower_raw"][
            "proposed_stage2_policy"
        ]
        == "force_ris_only"
    )
    assert (
        specs["adaptive_jones_vp_proposed_old_gated"]["proposed_stage2_policy"]
        == "reliability_gated_ris_only"
    )
    assert (
        specs["adaptive_jones_vp_proposed_old_gated"][
            "rescue_accept_min_rel_improvement"
        ]
        == 1.0e-3
    )


def test_ngc_rescue_run_rate_counts_active_ngc_rows_only():
    rows = [
        {"ngc_policy_active": True, "ngc_rescue_requested": True},
        {"ngc_policy_active": True, "ngc_rescue_requested": False},
        {"ngc_policy_active": False, "ngc_rescue_requested": True},
    ]
    assert ablation._ngc_rescue_run_rate(rows) == 0.5
    assert ablation._ngc_rescue_run_rate(rows[2:]) != ablation._ngc_rescue_run_rate(rows[2:])


def test_jones_lambda_specs_include_adaptive_and_free():
    specs = ablation._variant_specs("jones_lambda")
    assert "adaptive_jones" in specs
    assert "free_jones" in specs


def test_csv_fieldnames_include_required_diagnostics():
    assert "selected_vp_family_branch" in ablation.FIELDNAMES
    assert "lambda_jones_per_path" in ablation.FIELDNAMES
    assert "data_only_scaled_efim_condition_number" in ablation.FIELDNAMES
