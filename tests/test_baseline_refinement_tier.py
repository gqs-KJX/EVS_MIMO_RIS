"""Declared two-tier baseline comparison policy.

``baselines.refinement_tier`` records, in the result schema, whether every
method was granted the same continuous exact-model polish over
``(p_u, Delta_t)`` ("refinement_matched") or each baseline was stopped where its
own reference stops ("as_published").  Only ``ris_vbi_sbl`` changes behaviour
between the tiers, because only its reference lacks such a refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.common import REFINEMENT_TIERS, baseline_refinement_tier
from src.baselines.ris_vbi_sbl import (
    _fuse_position_clock_geometric,
    run_ris_vbi_sbl_baseline,
)
from src.channel_model import (
    add_awgn,
    channel_components,
    generate_scene,
    synthesize_raw_tensor,
)
from src.config import default_config


def _small_config(snr_db: float = 10.0) -> dict:
    config = default_config()
    config["K"] = 3
    config["SNR_dB"] = snr_db
    config["seed"] = 20260728
    config["diagnostic_fast_problem_size"] = True
    baselines = dict(config.get("baselines", {}))
    vbi = dict(baselines.get("ris_vbi_sbl", {}))
    vbi.update({"nf_grid_x": 3, "nf_grid_y": 3, "nf_grid_z": 3,
                "delay_grid_size": 21, "vbi_max_iter": 4, "vbi_refine_maxiter": 5})
    baselines["ris_vbi_sbl"] = vbi
    config["baselines"] = baselines
    return config


def _make_data(config: dict) -> dict:
    rng = np.random.default_rng(int(config["seed"]))
    scene = generate_scene(config, rng)
    components = channel_components(
        scene, scene["p_u_true"], scene["delta_t_true"],
        scene["gamma_true"], scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(components, scene["beta_true"])
    y_noisy, noise_variance = add_awgn(
        y_true, float(config["SNR_dB"]), rng,
        active_mask=scene.get("evs_observation_mask"),
    )
    return {"scene": scene, "Y_true": y_true, "Y_noisy": y_noisy,
            "noise_variance": noise_variance}


def test_default_tier_is_as_published():
    """The paper's headline benchmark is the as-published protocol, so that is
    the default; ``refinement_matched`` is the explicitly requested tier."""
    assert baseline_refinement_tier({}) == "as_published"
    assert baseline_refinement_tier(default_config()) == "as_published"
    assert set(REFINEMENT_TIERS) == {"refinement_matched", "as_published"}


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError, match="refinement_tier"):
        baseline_refinement_tier({"baselines": {"refinement_tier": "best_effort"}})


def test_geometric_fusion_is_exact_on_noiseless_panel_outputs():
    """The linear system is an identity when the panel outputs are exact."""
    config = _small_config()
    data = _make_data(config)
    scene = data["scene"]
    p_true = np.asarray(scene["p_u_true"], dtype=float)
    delta_t_true = float(scene["delta_t_true"])

    panels = []
    for panel in range(int(scene["K"])):
        rotation = np.asarray(scene["rotations"][panel], dtype=float)
        center = np.asarray(scene["ris_centers"][panel], dtype=float)
        q_local = rotation @ (p_true - center)
        range_m = float(np.linalg.norm(q_local))
        panels.append({
            "panel": panel,
            "direction_local": q_local / range_m,
            "tau": (range_m + float(scene["d_RB"][panel])) / float(scene["c0"]) + delta_t_true,
            "position": p_true.copy(),
            "confidence": 1.0,
        })

    p_hat, dt_hat = _fuse_position_clock_geometric(scene, config, panels)
    assert np.allclose(p_hat, p_true, atol=1e-9)
    assert dt_hat == pytest.approx(delta_t_true, abs=1e-18)


def test_as_published_tier_skips_the_exact_model_refinement(monkeypatch):
    """No exact-model objective evaluation may occur in the as_published tier."""
    import src.baselines.ris_vbi_sbl as module

    calls = {"n": 0}
    original = module._raw_objective

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_raw_objective", counting)

    config = _small_config()
    config["baselines"]["refinement_tier"] = "as_published"
    data = _make_data(config)
    result = run_ris_vbi_sbl_baseline(data, config)

    assert calls["n"] == 0
    assert result.diagnostics["refinement_tier"] == "as_published"
    assert result.diagnostics["exact_model_refinement_used"] is False
    assert result.diagnostics["fusion_rule"].startswith("weighted_linear_ls")


def test_refinement_matched_tier_preserves_the_existing_behaviour(monkeypatch):
    import src.baselines.ris_vbi_sbl as module

    calls = {"n": 0}
    original = module._raw_objective

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_raw_objective", counting)

    config = _small_config()
    config["baselines"]["refinement_tier"] = "refinement_matched"
    data = _make_data(config)
    result = run_ris_vbi_sbl_baseline(data, config)

    assert calls["n"] > 0
    assert result.diagnostics["refinement_tier"] == "refinement_matched"
    assert result.diagnostics["exact_model_refinement_used"] is True
    assert result.diagnostics["fusion_rule"].startswith("exact_model_seed_scan")


def test_tier_is_recorded_and_the_two_tiers_differ():
    config_matched = _small_config()
    config_matched["baselines"]["refinement_tier"] = "refinement_matched"
    data = _make_data(config_matched)

    matched = run_ris_vbi_sbl_baseline(data, config_matched)

    config_published = _small_config()
    config_published["baselines"]["refinement_tier"] = "as_published"
    published = run_ris_vbi_sbl_baseline(data, config_published)

    # Same per-panel VBI/SBL core, different localization step.
    assert matched.diagnostics["selected_panels"] == published.diagnostics["selected_panels"]
    assert not np.allclose(matched.p_u, published.p_u)
