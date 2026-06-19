import copy

import numpy as np
import pytest

from src.baselines.common import BaselineResult, y_noisy_hash
from src.config import default_config
from src.experiments import run_robustness_and_scaling_figures as figures
from src.main_single_proposed import _make_data


def _tiny_config(k_paths: int = 1) -> dict:
    config = default_config()
    config.update(
        {
            "seed": 1234,
            "K": k_paths,
            "M_A": 1,
            "ris_shape": (2, 2),
            "N": 5,
            "P": 3,
            "T": 4,
            "SNR_dB": 20.0,
            "receiver_mode": "full_6d",
            "print_progress": False,
            "p_u_true": np.array([1.2, 0.4, 0.8]),
            "ris_centers": np.array(
                [
                    [4.2, -2.2, 1.05],
                    [5.1, 2.1, 1.15],
                    [4.8, 0.0, 1.25],
                ]
            ),
            "ue_bounds": np.array([[0.8, 1.6], [0.0, 0.8], [0.5, 1.1]]),
            "delta_t_bounds": np.array([4.0e-9, 6.0e-9]),
        }
    )
    return config


def test_calibration_phase_error_changes_generation_response_only():
    data = _make_data(_tiny_config())
    nominal = data["scene"]["a_RB"].copy()
    perturbed = figures.inject_ris_bs_calibration_phase_error(
        data, 10.0, np.random.default_rng(9)
    )
    assert not np.allclose(perturbed["scene"]["a_RB"], nominal)
    assert np.array_equal(data["scene"]["a_RB"], nominal)
    assert data["scene"]["K"] == perturbed["scene"]["K"]
    assert np.array_equal(data["scene"]["Omega"], perturbed["scene"]["Omega"])


@pytest.mark.parametrize("figure", ["fig8", "fig9", "fig10a"])
def test_one_trial_all_baselines_share_noisy_hash(figure, monkeypatch):
    config = _tiny_config()
    data = _make_data(config)

    monkeypatch.setattr(
        figures,
        "_prepare_task_data",
        lambda task: (data, data, config),
    )

    def fake_proposed(data_arg, config_arg, task):
        return {
            "baseline": "proposed",
            "failed": False,
            "error": "",
            "position_rmse_m": 0.1,
            "y_nmse": 0.2,
            "raw_objective_final": 0.3,
            "runtime_s": 0.01,
            "warning": "",
        }

    def fake_baseline(data_arg, config_arg):
        return BaselineResult(
            name="fake",
            p_u=data_arg["scene"]["p_u_true"],
            delta_t=data_arg["scene"]["delta_t_true"],
            Y_hat=data_arg["Y_true"],
            raw_objective_final=0.0,
        )

    monkeypatch.setattr(figures, "_proposed_result_row", fake_proposed)
    monkeypatch.setitem(figures.BASELINE_RUNNERS, "ff_omp", fake_baseline)
    monkeypatch.setitem(figures.BASELINE_RUNNERS, "ris_momp", fake_baseline)
    monkeypatch.setitem(figures.BASELINE_RUNNERS, "nf_mmpsr", fake_baseline)
    task = {
        "figure": figure,
        "trial_id": 0,
        "seed": config["seed"],
        "snr_db": config["SNR_dB"],
        "true_K": 1,
        "assumed_K": 1,
        "calibration_std_deg": 0.0,
        "baselines": ["proposed", "ff_omp", "ris_momp", "nf_mmpsr"],
        "blas_threads": 1,
        "profile_memory": False,
        "trim_memory": False,
    }
    rows = figures.run_shared_trial(task)
    assert {row["y_noisy_hash"] for row in rows} == {y_noisy_hash(data)}


def test_k_mismatch_changes_estimator_order_not_true_data_order():
    true_config = _tiny_config(k_paths=1)
    estimator_data, true_data = figures.make_k_mismatch_data(
        true_config, assumed_k=2, max_assumed_k=2
    )
    configured = figures._configure_assumed_k(true_config, 2)
    assert true_data["scene"]["K"] == 1
    assert estimator_data["scene"]["K"] == 2
    assert configured["K"] == 2
    assert configured["baselines"]["ff_omp"]["max_atoms"] == 2
    assert configured["baselines"]["ris_momp"]["max_atoms"] == 2
    assert np.array_equal(estimator_data["Y_noisy"], true_data["Y_noisy"])


def test_figure8_peb_policy_is_explicit():
    requested = figures.parse_baselines(figures.DEFAULT_BASELINES)
    without = figures.baselines_for_figure("fig8", requested)
    with_oracle = figures.baselines_for_figure(
        "fig8", requested, include_calibration_oracle_peb=True
    )
    assert "peb" not in without
    assert "oracle_calibrated_peb" not in without
    assert "oracle_calibrated_peb" in with_oracle


def test_figure9_peb_policy_is_explicit():
    requested = figures.parse_baselines(figures.DEFAULT_BASELINES)
    without = figures.baselines_for_figure("fig9", requested)
    with_reference = figures.baselines_for_figure(
        "fig9", requested, include_trueK_peb_reference=True
    )
    assert "peb" not in without
    assert "trueK_peb_reference" not in without
    assert "trueK_peb_reference" in with_reference


@pytest.mark.parametrize("figure", ["fig10a", "fig10b", "fig10c"])
def test_figure10_includes_matched_model_peb(figure):
    requested = figures.parse_baselines(figures.DEFAULT_BASELINES)
    assert "peb" in figures.baselines_for_figure(figure, requested)


def test_rue_sweep_stays_on_fixed_centroid_ray():
    config = _tiny_config(k_paths=3)
    centers = config["ris_centers"][:3]
    centroid = np.mean(centers, axis=0)
    original_direction = config["p_u_true"] - centroid
    original_direction /= np.linalg.norm(original_direction)
    for radius in (0.5, 1.0, 3.0):
        position = figures.ue_position_on_centroid_ray(config, radius)
        displacement = position - centroid
        assert np.isclose(np.linalg.norm(displacement), radius)
        assert np.allclose(displacement / np.linalg.norm(displacement), original_direction)


def test_calibration_data_keeps_nominal_estimator_scene(monkeypatch):
    config = _tiny_config()
    original = copy.deepcopy(config)
    data, oracle = figures.make_calibration_mismatch_data(config, 5.0)
    assert np.array_equal(config["ris_centers"], original["ris_centers"])
    assert not np.allclose(data["scene"]["a_RB"], oracle["scene"]["a_RB"])
    assert np.array_equal(data["Y_noisy"], oracle["Y_noisy"])
