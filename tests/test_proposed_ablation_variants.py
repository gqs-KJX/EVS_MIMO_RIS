from src.experiments import run_proposed_ablation as ablation


def test_import_run_proposed_ablation():
    assert ablation.FIELDNAMES


def test_vp_family_specs_include_proposed_adaptive_jones():
    specs = ablation._variant_specs("vp_family")
    assert "adaptive_jones_vp_proposed" in specs


def test_stage2_gate_specs_include_direct_and_proposed_gated():
    specs = ablation._variant_specs("stage2_gate")
    assert "direct_vp_only" in specs
    assert "reliability_gated_ris_jnpp_then_vp_proposed" in specs


def test_jones_lambda_specs_include_adaptive_and_free():
    specs = ablation._variant_specs("jones_lambda")
    assert "adaptive_jones" in specs
    assert "free_jones" in specs


def test_csv_fieldnames_include_required_diagnostics():
    assert "selected_vp_family_branch" in ablation.FIELDNAMES
    assert "lambda_jones_per_path" in ablation.FIELDNAMES
    assert "data_only_scaled_efim_condition_number" in ablation.FIELDNAMES
