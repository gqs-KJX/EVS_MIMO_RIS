"""Acquisition-stage ablation switch for the fixed-count coarse RIS codebook.

``ris_search.coarse_codebook_mode`` selects the candidate set scored by the
exact spherical residual in Stage-I RIS factor projection:

``"union"``          legacy fixed-count (range, elevation, azimuth) dictionary
                     together with the Nyquist beam-space candidates (default,
                     published behaviour);
``"beamspace_only"`` beam-space candidates alone, which removes an
                     ``O(G * M_R)`` spherical-response build.

The switch must never be able to empty the candidate set, and it must not
change the selection when the legacy dictionary contributes no winner.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.channel_model import generate_scene
from src.config import default_config
from src.estimators import (
    _coarse_ris_factor_projection,
    _coarse_ris_factor_projections_batched,
    _legacy_coarse_codebook_enabled,
)
from src.projections_ris import compressed_exact_response, local_ris_search_config


def _scene_and_search(panel: int = 0):
    config = default_config()
    config["K"] = 3
    config["seed"] = 20260728
    rng = np.random.default_rng(int(config["seed"]))
    scene = generate_scene(config, rng)
    return scene, config, local_ris_search_config(scene, config, panel)


def _panel_proxy(scene: dict, panel: int, etas: list[np.ndarray]) -> np.ndarray:
    columns = [
        compressed_exact_response(
            eta, scene["Omega"][panel], scene["a_RB"][panel],
            scene["ris_grid"], scene["wavelength"],
        )
        for eta in etas
    ]
    return np.column_stack(columns)


def _true_local_eta(scene: dict, panel: int) -> np.ndarray:
    from src.geometry import local_geometry_from_position

    range_m, elev, az, _ = local_geometry_from_position(
        np.asarray(scene["p_u_true"], dtype=float),
        np.asarray(scene["ris_centers"][panel], dtype=float),
        np.asarray(scene["rotations"][panel], dtype=float),
    )
    return np.array([range_m, elev, az], dtype=float)


def test_default_mode_keeps_the_legacy_codebook():
    """The library default stays "union"; only the benchmark path drops it.

    Dropping the dictionary is validated for the reference configuration
    (64x64 RIS, six exact refine starts), not for every array, so the ablation
    is applied in ``run_benchmark_comparison.make_config`` rather than here.
    See ``test_benchmark_config_drops_the_legacy_codebook`` and
    ``test_beamspace_only_is_not_safe_on_a_small_array``.
    """
    assert default_config()["ris_search"]["coarse_codebook_mode"] == "union"
    assert _legacy_coarse_codebook_enabled({"coarse_codebook_mode": "union"})
    assert not _legacy_coarse_codebook_enabled(
        {"coarse_codebook_mode": "beamspace_only"}
    )


def test_benchmark_config_drops_the_legacy_codebook():
    """The paper's configuration is the one the ablation evidence covers."""
    from src.experiments.run_benchmark_comparison import make_config

    config = make_config(seed=1, snr_db=-10.0, paper_k=3, grid_profile="medium")
    assert config["ris_search"]["coarse_codebook_mode"] == "beamspace_only"


def test_beamspace_only_is_ignored_when_beamspace_is_disabled():
    """Otherwise the candidate set would be empty."""
    assert _legacy_coarse_codebook_enabled(
        {"coarse_codebook_mode": "beamspace_only", "use_beamspace_acquisition": False}
    )


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="coarse_codebook_mode"):
        _legacy_coarse_codebook_enabled({"coarse_codebook_mode": "nyquist"})


def test_beamspace_only_is_not_safe_on_a_small_array():
    """Pins the scope condition behind the benchmark-only ablation.

    On a 4x4 RIS the beam-space direction-cosine grid is coarser in angle than
    the legacy dictionary, so the exact refinement started from beam-space
    candidates alone converges away from the full-grid reference, while the
    union start set reproduces it exactly.  Raising the beam-space candidate
    budget does not recover it, so this is a resolution limit of the beam-space
    grid on a small aperture rather than start starvation.

    This is why ``coarse_codebook_mode`` is left at ``"union"`` in
    ``default_config()`` and switched only in the benchmark configuration.
    """
    from src.channel_model import channel_components, synthesize_raw_tensor
    from src.config import apply_stage1_init_preset
    from src.estimators import initialize_from_hankel
    from src.tensor_utils import hankelize_frequency

    config = default_config()
    config.update(
        {
            "K": 2, "M_A": 1, "ris_shape": (4, 4), "N": 9, "P": 5, "T": 12,
            "ris_centers": config["ris_centers"][:2].copy(),
        }
    )
    apply_stage1_init_preset(config, "smoke")
    config["ris_search"]["num_exact_refine_starts"] = 2
    scene = generate_scene(config, np.random.default_rng(116))
    components = channel_components(
        scene, scene["p_u_true"], scene["delta_t_true"],
        scene["gamma_true"], scene["eta_true"],
    )
    z_tensor = hankelize_frequency(
        synthesize_raw_tensor(components, scene["beta_true"]), scene["P"]
    )

    reference_config = dict(config, ris_search=dict(config["ris_search"]))
    reference_config["stage1_ris_geometry_mode"] = "legacy_fast_projection"
    reference = initialize_from_hankel(z_tensor, scene, reference_config)

    def hybrid_eta(mode: str, num_beamspace: int = 4) -> np.ndarray:
        local = dict(config, ris_search=dict(config["ris_search"]))
        local["stage1_ris_geometry_mode"] = "coarse_to_exact_assignment"
        local["stage1_assignment_num_exact_permutations"] = 2
        local["ris_search"]["coarse_codebook_mode"] = mode
        local["ris_search"]["beamspace_num_candidates"] = num_beamspace
        return np.asarray(
            initialize_from_hankel(z_tensor, scene, local)["ris_eta"], dtype=float
        )

    reference_eta = np.asarray(reference["ris_eta"], dtype=float)
    np.testing.assert_allclose(hybrid_eta("union"), reference_eta, atol=1.0e-9)

    # Beam-space alone misses it, and more candidates do not help.
    for num_beamspace in (4, 8, 16):
        gap = float(
            np.abs(hybrid_eta("beamspace_only", num_beamspace) - reference_eta).max()
        )
        assert gap > 1.0e-3


@pytest.mark.parametrize("mode", ["union", "beamspace_only"])
def test_batched_projection_recovers_the_true_geometry(mode):
    scene, config, search = _scene_and_search(panel=0)
    search["coarse_codebook_mode"] = mode
    eta_true = _true_local_eta(scene, 0)
    proxy = _panel_proxy(scene, 0, [eta_true])

    projections = _coarse_ris_factor_projections_batched(
        proxy, scene["Omega"][0], scene["a_RB"][0], scene["ris_grid"],
        scene["wavelength"], search, 1.0e-12, {},
    )
    assert len(projections) == 1
    entry = projections[0]
    # Acquisition is deliberately coarse in range: the beam-space range grid has
    # ``beamspace_num_range`` points across the induced bounds, and the residual
    # at the coarse stage is dominated by that step.  What must hold here is
    # that the *angles* land inside one panel mainlobe, since local refinement
    # cannot recover from a start on the wrong lobe.
    lower, upper = search["range_bounds"]
    range_step = (float(upper) - float(lower)) / max(
        int(search.get("beamspace_num_range", 9)) - 1, 1
    )
    assert abs(entry["eta_local"][0] - eta_true[0]) <= range_step
    mainlobe = scene["wavelength"] / float(
        np.ptp(np.asarray(scene["ris_grid"], dtype=float)[:, 0])
    )
    assert abs(entry["eta_local"][1] - eta_true[1]) < mainlobe
    assert abs(entry["eta_local"][2] - eta_true[2]) < mainlobe


def test_beamspace_only_drops_the_codebook_candidates_but_not_the_winner():
    scene, config, search = _scene_and_search(panel=0)
    eta_true = _true_local_eta(scene, 0)
    proxy = _panel_proxy(scene, 0, [eta_true])

    union_search = dict(search, coarse_codebook_mode="union")
    beam_search = dict(search, coarse_codebook_mode="beamspace_only")
    union = _coarse_ris_factor_projections_batched(
        proxy, scene["Omega"][0], scene["a_RB"][0], scene["ris_grid"],
        scene["wavelength"], union_search, 1.0e-12, {},
    )[0]
    beam = _coarse_ris_factor_projections_batched(
        proxy, scene["Omega"][0], scene["a_RB"][0], scene["ris_grid"],
        scene["wavelength"], beam_search, 1.0e-12, {},
    )[0]

    assert union["coarse_num_candidates_codebook"] > 0
    assert beam["coarse_num_candidates_codebook"] == 0
    assert beam["coarse_num_candidates_beamspace"] > 0
    assert union["coarse_num_candidates_beamspace"] == beam["coarse_num_candidates_beamspace"]
    # The union is a superset scored by the same objective, so it can only tie
    # or beat the subset; here the beam-space block already supplies the winner.
    assert union["data_residual"] <= beam["data_residual"] * (1.0 + 1.0e-12)
    assert union["coarse_candidate_source"] == "beamspace"
    assert beam["coarse_candidate_source"] == "beamspace"
    assert np.allclose(union["eta_local"], beam["eta_local"])


def test_union_selection_is_a_minimum_over_the_superset():
    """Whatever the winner's provenance, union must not be worse than either arm."""
    scene, config, search = _scene_and_search(panel=1)
    eta_true = _true_local_eta(scene, 1)
    proxy = _panel_proxy(scene, 1, [eta_true])

    union = _coarse_ris_factor_projections_batched(
        proxy, scene["Omega"][1], scene["a_RB"][1], scene["ris_grid"],
        scene["wavelength"], dict(search, coarse_codebook_mode="union"),
        1.0e-12, {},
    )[0]
    codebook_only = _coarse_ris_factor_projections_batched(
        proxy, scene["Omega"][1], scene["a_RB"][1], scene["ris_grid"],
        scene["wavelength"],
        dict(search, coarse_codebook_mode="union", use_beamspace_acquisition=False),
        1.0e-12, {},
    )[0]
    beam_only = _coarse_ris_factor_projections_batched(
        proxy, scene["Omega"][1], scene["a_RB"][1], scene["ris_grid"],
        scene["wavelength"], dict(search, coarse_codebook_mode="beamspace_only"),
        1.0e-12, {},
    )[0]

    tolerance = 1.0 + 1.0e-9
    assert union["data_residual"] <= codebook_only["data_residual"] * tolerance
    assert union["data_residual"] <= beam_only["data_residual"] * tolerance


def test_unbatched_path_mirrors_the_batched_selection():
    scene, config, search = _scene_and_search(panel=2)
    eta_true = _true_local_eta(scene, 2)
    proxy = _panel_proxy(scene, 2, [eta_true])

    for mode in ("union", "beamspace_only"):
        local = dict(search, coarse_codebook_mode=mode)
        batched = _coarse_ris_factor_projections_batched(
            proxy, scene["Omega"][2], scene["a_RB"][2], scene["ris_grid"],
            scene["wavelength"], local, 1.0e-12, {},
        )[0]
        single = _coarse_ris_factor_projection(
            proxy[:, 0], scene["Omega"][2], scene["a_RB"][2], scene["ris_grid"],
            scene["wavelength"], local, 1.0e-12, {},
        )
        assert np.allclose(single["eta_local"], batched["eta_local"])
        assert single["data_residual"] == pytest.approx(
            batched["data_residual"], rel=1e-9
        )
