import copy

import numpy as np

from src.channel_model import channel_components, generate_scene, synthesize_raw_tensor
from src.config import default_config
from src.experiments.run_paper_ablation_figures import position_peb_from_global_efim
from src.global_vp import (
    _UNSCALED_EFIM_CACHE,
    data_only_efim_diagnostic,
    efim_unscaled_cache_key,
)


def _problem():
    config = default_config()
    config.update(
        {
            "seed": 99,
            "K": 1,
            "M_A": 1,
            "ris_shape": (2, 2),
            "N": 5,
            "P": 3,
            "T": 4,
            "receiver_mode": "full_6d",
            "ris_centers": np.array([[4.2, -2.2, 1.05]]),
        }
    )
    rng = np.random.default_rng(config["seed"])
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y = synthesize_raw_tensor(components, scene["beta_true"])
    init = {
        "A": components["a_EVS"].T.copy(),
        "poles": components["poles"].copy(),
        "ris_eta": np.column_stack(
            [components["ranges"], components["elevations"], components["azimuths"]]
        ),
        "gamma": scene["gamma_true"].copy(),
        "eta_pol": scene["eta_true"].copy(),
        "assignment": [0],
        "panel_to_column_assignment": [0],
        "columns_are_panel_ordered": True,
    }
    return config, scene, components, y, init


def test_unscaled_efim_cache_matches_uncached_peb():
    config, scene, _, y, init = _problem()
    sigma2 = 0.02
    uncached = data_only_efim_diagnostic(
        y, scene["p_u_true"], scene["delta_t_true"], init, scene, config, sigma2
    )
    cached_config = copy.deepcopy(config)
    cached_config["crb"]["enable_unscaled_efim_cache"] = True
    _UNSCALED_EFIM_CACHE.clear()
    first = data_only_efim_diagnostic(
        y,
        scene["p_u_true"],
        scene["delta_t_true"],
        init,
        scene,
        cached_config,
        sigma2,
    )
    second = data_only_efim_diagnostic(
        y,
        scene["p_u_true"],
        scene["delta_t_true"],
        init,
        scene,
        cached_config,
        sigma2 * 2.0,
    )
    np.testing.assert_allclose(first["data_only_scaled_efim"], uncached["data_only_scaled_efim"])
    np.testing.assert_allclose(
        second["data_only_efim"],
        0.5 * first["data_only_efim"],
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert first["efim_unscaled_cache_hit"] is False
    assert second["efim_unscaled_cache_hit"] is True


def test_unscaled_cache_key_changes_for_required_fields():
    config, scene, _, _, _ = _problem()
    p = scene["p_u_true"]
    dt = scene["delta_t_true"]
    base = efim_unscaled_cache_key(scene, config, p, dt)
    mode_scene = dict(scene)
    mode_scene["receiver_mode"] = "scalar"
    assert efim_unscaled_cache_key(mode_scene, config, p, dt) != base
    k_scene = dict(scene)
    k_scene["K"] = 2
    assert efim_unscaled_cache_key(k_scene, config, p, dt) != base
    changed_scene = dict(scene)
    changed_scene["scene_hash"] = "different"
    assert efim_unscaled_cache_key(changed_scene, config, p, dt) != base
    assert efim_unscaled_cache_key(scene, config, p + [0.01, 0, 0], dt) != base
    assert efim_unscaled_cache_key(scene, config, p, dt + 1.0e-10) != base
    assert (
        efim_unscaled_cache_key(
            scene, config, p, dt, parameter_scaling="different_scaling"
        )
        != base
    )


def test_schur_complement_peb_scaling():
    j_unscaled = np.array(
        [
            [5.0, 0.2, 0.0, 1.5],
            [0.2, 4.0, 0.1, -0.7],
            [0.0, 0.1, 3.0, 0.4],
            [1.5, -0.7, 0.4, 2.0],
        ]
    )
    sigma2 = 0.25
    alpha = 2.0 / sigma2
    order = ["p_x_m", "p_y_m", "p_z_m", "c_delta_t_m"]
    peb_unscaled = position_peb_from_global_efim(j_unscaled, order)
    peb_scaled = position_peb_from_global_efim(alpha * j_unscaled, order)
    np.testing.assert_allclose(
        peb_scaled, peb_unscaled / np.sqrt(alpha), rtol=1.0e-13, atol=1.0e-13
    )
