import numpy as np

from src.baselines import als_cpd, common, far_field_omp, near_field_mmpsr, ris_momp
from src.config import default_config
from src.main_single_proposed import _make_data
from src.experiments import run_benchmark_comparison as bench
from src.experiments.run_paper_ablation_figures import _peb_from_efim, peb_cache_key


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
        "ff_omp": {"angle_grid_size": 3, "delay_grid_size": 3, "max_atoms": 1},
        "ris_momp": {"direction_grid_size": 3, "delay_grid_size": 3, "max_atoms": 1},
        "nf_mmpsr": {"grid_shape": (3, 3, 3), "clock_grid_size": 3},
    }
    return config


def test_import_all_benchmark_baseline_modules():
    assert common.BaselineResult
    assert als_cpd.run_als_cpd_baseline
    assert far_field_omp.run_far_field_omp_baseline
    assert ris_momp.run_ris_momp_baseline
    assert near_field_mmpsr.run_near_field_mmpsr_baseline


def test_linear_ls_fit_recovers_known_complex_coefficients():
    rng = np.random.default_rng(1)
    Phi = rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
    coeff_true = np.array([1.2 - 0.3j, -0.7 + 0.5j])
    y = Phi @ coeff_true
    coeff, y_hat, residual = common.linear_ls_fit(Phi, y)
    assert np.allclose(coeff, coeff_true)
    assert np.linalg.norm(residual) < 1.0e-10
    assert np.allclose(y_hat, y)


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


def test_ff_omp_selects_known_atom_in_tiny_dictionary():
    Phi = np.eye(4, dtype=complex)
    y = 2.0 * Phi[:, 2]
    selected = far_field_omp.omp_select_from_dictionary(Phi, y, max_atoms=1)
    assert selected == [2]


def test_ff_omp_recovers_known_far_field_support():
    config = _tiny_config()
    data = _make_data(config)
    scene = data["scene"]
    support = list(far_field_omp._far_field_supports(scene, config))[0]
    atom = common.simple_atom_normalize(common.raw_atom_from_support(scene, config, support))
    data["Y_noisy"] = atom.reshape(scene["I"], scene["N"], scene["T"])
    data["Y_true"] = data["Y_noisy"].copy()
    result = far_field_omp.run_far_field_omp_baseline(data, config)
    assert result.selected_support[0]["direction_index"] == support["direction_index"]
    assert result.selected_support[0]["tau_index"] == support["tau_index"]
    assert result.diagnostics["dictionary_mode"] == "far_field_angular_delay_omp"


def test_ris_momp_recovers_known_multidimensional_support():
    config = _tiny_config()
    data = _make_data(config)
    scene = data["scene"]
    support = list(ris_momp._ris_momp_supports(scene, config))[0]
    atom = common.simple_atom_normalize(common.raw_atom_from_support(scene, config, support))
    data["Y_noisy"] = atom.reshape(scene["I"], scene["N"], scene["T"])
    data["Y_true"] = data["Y_noisy"].copy()
    result = ris_momp.run_ris_momp_baseline(data, config)
    assert result.selected_support[0]["direction_index"] == support["direction_index"]
    assert result.selected_support[0]["tau_index"] == support["tau_index"]
    assert result.diagnostics["dictionary_mode"] == "batched_momp_equivalent"


def test_nf_mmpsr_grid_search_selects_known_grid_point():
    config = _tiny_config()
    data = _make_data(config)
    data["Y_noisy"] = data["Y_true"].copy()
    result = near_field_mmpsr.run_near_field_mmpsr_baseline(data, config)
    assert np.allclose(result.p_u, config["p_u_true"])
    assert abs(result.delta_t - config["delta_t_true"]) < 1.0e-15
    assert result.diagnostics["dictionary_mode"] == "near_field_spherical_grid_mmpsr"


def test_baseline_wrappers_do_not_call_proposed_vp(monkeypatch):
    import src.global_vp as global_vp

    def forbidden(*args, **kwargs):
        raise AssertionError("proposed VP must not be called by standalone baselines")

    monkeypatch.setattr(global_vp, "global_exact_spherical_vp_refinement", forbidden)
    config = _tiny_config()
    data = _make_data(config)
    data["Y_noisy"] = data["Y_true"].copy()

    als_cpd.run_als_cpd_baseline(data, config)
    far_field_omp.run_far_field_omp_baseline(data, config)
    ris_momp.run_ris_momp_baseline(data, config)
    near_field_mmpsr.run_near_field_mmpsr_baseline(data, config)


def test_benchmark_plot_metric_mapping_uses_peb_for_peb():
    assert bench.get_plot_metric("peb", "rmse") == "peb_position_m"
    assert bench.get_plot_metric("peb", "nmse") is None
    assert bench.get_plot_metric("proposed", "rmse") == "position_rmse_m"


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


def test_peb_cache_key_changes_with_receiver_mode_or_k():
    config = _tiny_config()
    key_base = peb_cache_key(config)
    mode_config = {**config, "receiver_mode": "scalar"}
    k_config = {**config, "K": 2, "ris_centers": np.vstack([config["ris_centers"], [5.0, 2.0, 1.0]])}
    assert peb_cache_key(mode_config) != key_base
    assert peb_cache_key(k_config) != key_base
