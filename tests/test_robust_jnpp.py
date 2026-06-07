import numpy as np

from src.channel_model import channel_components, generate_scene
from src.config import default_config
from src.robust_jnpp import robust_jnpp_basin_recovery


def test_robust_jnpp_recovers_position_and_updates_vp_initialization():
    config = default_config()
    config.update(
        {
            "K": 3,
            "T": 24,
            "M_A": 2,
            "ris_shape": (6, 6),
            "N": 9,
            "P": 5,
            "jnpp_num_starts": 4,
            "jnpp_use_leave_one_out": True,
            "jnpp_max_candidates": 4,
            "jnpp_check_gradient": True,
        }
    )
    rng = np.random.default_rng(123)
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    c_panel = components["c"].T.copy()
    stage1 = {
        "C": c_panel * np.array([1.2 - 0.2j, 0.8 + 0.1j, -1.0j])[None, :],
        "A": components["a_EVS"].T.copy(),
        "B": np.ones((scene["P"], scene["K"]), dtype=complex),
        "Q": np.ones((scene["L"], scene["K"]), dtype=complex),
        "poles": components["poles"].copy(),
        "beta_z": np.ones(scene["K"], dtype=complex),
        "gamma": scene["gamma_true"].copy(),
        "eta_pol": scene["eta_true"].copy(),
        "ris_eta": np.column_stack(
            [
                components["ranges"] + np.array([0.2, -0.1, 0.15]),
                components["elevations"],
                components["azimuths"],
            ]
        ),
        "assignment": np.arange(scene["K"]),
        "columns_are_panel_ordered": True,
        "stage1_rank1_ratios": np.array([0.0, 0.5, 1.0]),
    }

    estimate, diag = robust_jnpp_basin_recovery(stage1, scene, config)

    assert diag["stage2_rescue_mode"] == "robust_jnpp"
    assert diag["jnpp_num_candidates"] == 4
    assert diag["jnpp_gradient_mode"] == "analytic"
    assert diag["jnpp_gradient_check_rel_error"] < 1.0e-3
    np.testing.assert_allclose(
        diag["jnpp_weights"],
        np.maximum(np.exp(-2.0 * stage1["stage1_rank1_ratios"]), 0.05),
    )
    assert np.linalg.norm(estimate["p_u"] - scene["p_u_true"]) < 0.15
    assert np.isfinite(estimate["delta_t"])
    assert estimate["C"].shape == stage1["C"].shape
    assert estimate["ris_eta"].shape == (scene["K"], 3)
    assert estimate["columns_are_panel_ordered"] is True
