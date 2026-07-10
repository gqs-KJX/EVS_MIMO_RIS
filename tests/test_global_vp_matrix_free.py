import copy

import numpy as np
import pytest

from src.config import default_config
from src.global_vp import (
    _build_vp_matrix_free_cache,
    _compare_matrix_free_to_explicit,
    _get_panel_ordered_stage1_factors,
    _initial_xi_from_stage1_with_diagnostics,
    _vp_objective_parts_matrix_free,
)
from src.main_single_proposed import _make_data, run_stage1_only


def _tiny_config(k_paths: int, receiver_mode: str = "full_6d") -> dict:
    config = default_config()
    config.update(
        {
            "seed": 4321 + k_paths,
            "K": k_paths,
            "M_A": 1,
            "ris_shape": (2, 2),
            "N": 5,
            "P": 3,
            "T": 4,
            "SNR_dB": 30.0,
            "receiver_mode": receiver_mode,
            "print_progress": False,
            "p_u_true": np.array([1.2, 0.4, 0.8]),
            "ris_centers": np.array(
                [
                    [4.2, -2.2, 1.05],
                    [5.1, 2.1, 1.15],
                    [4.8, 0.0, 1.25],
                ]
            )[:k_paths],
            "ue_bounds": np.array([[0.9, 1.5], [0.1, 0.7], [0.55, 1.05]]),
            "delta_t_true": 5.0e-9,
            "delta_t_bounds": np.array([4.0e-9, 6.0e-9]),
        }
    )
    config["global_vp"] = dict(config.get("global_vp", {}))
    config["global_vp"].update(
        {
            "mode": "adaptive_jones",
            "vp_dictionary_mode": "matrix_free",
            "vp_debug_compare_explicit": True,
        }
    )
    return config


def _stage1_bundle(config: dict):
    data = _make_data(config)
    stage1 = run_stage1_only(copy.deepcopy(data), copy.deepcopy(config))["estimate"]
    stage1["jones_lambda_per_path"] = np.ones(int(config["K"]))
    factors = _get_panel_ordered_stage1_factors(stage1, data["scene"])
    xi0, _ = _initial_xi_from_stage1_with_diagnostics(
        stage1, data["scene"], config, factors
    )
    return data, stage1, xi0


@pytest.mark.parametrize("k_paths", [1, 3])
def test_matrix_free_stats_match_explicit_for_k_values(k_paths):
    config = _tiny_config(k_paths)
    data, stage1, xi0 = _stage1_bundle(config)
    y_vec = data["Y_noisy"].reshape(-1)
    cache = _build_vp_matrix_free_cache(y_vec, stage1, data["scene"], config)
    offsets = [
        np.zeros(4),
        np.array([0.01, -0.015, 0.005, 0.1e-9]),
        np.array([-0.02, 0.01, -0.006, -0.1e-9]),
    ]
    for offset in offsets:
        xi = xi0 + offset
        xi[:3] = np.clip(xi[:3], config["ue_bounds"][:, 0], config["ue_bounds"][:, 1])
        xi[3] = np.clip(xi[3], config["delta_t_bounds"][0], config["delta_t_bounds"][1])
        parts, stats = _vp_objective_parts_matrix_free(
            xi, y_vec, stage1, data["scene"], config, cache=cache
        )
        diagnostics = _compare_matrix_free_to_explicit(
            xi, y_vec, stage1, data["scene"], config, parts, stats
        )
        assert diagnostics["rel_G_diff"] <= 1.0e-9
        assert diagnostics["rel_b_diff"] <= 1.0e-9
        assert diagnostics["rel_x_hat_diff"] <= 1.0e-8
        assert diagnostics["rel_regularized_objective_diff"] <= 1.0e-8
        assert diagnostics["rel_gradient_diff"] <= 1.0e-8


@pytest.mark.parametrize("receiver_mode", ["scalar", "dual_pol", "full_6d"])
def test_matrix_free_stats_match_explicit_for_receiver_modes(receiver_mode):
    config = _tiny_config(1, receiver_mode=receiver_mode)
    data, stage1, xi0 = _stage1_bundle(config)
    y_vec = data["Y_noisy"].reshape(-1)
    cache = _build_vp_matrix_free_cache(y_vec, stage1, data["scene"], config)
    parts, stats = _vp_objective_parts_matrix_free(
        xi0, y_vec, stage1, data["scene"], config, cache=cache
    )
    diagnostics = _compare_matrix_free_to_explicit(
        xi0, y_vec, stage1, data["scene"], config, parts, stats
    )
    assert diagnostics["rel_G_diff"] <= 1.0e-9
    assert diagnostics["rel_b_diff"] <= 1.0e-9
    assert diagnostics["rel_x_hat_diff"] <= 1.0e-8
    assert diagnostics["rel_regularized_objective_diff"] <= 1.0e-8
    assert diagnostics["rel_gradient_diff"] <= 1.0e-8
