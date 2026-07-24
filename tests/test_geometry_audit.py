import copy

import numpy as np

from src.channel_model import generate_scene
from src.experiments.audit_bs_geometry import (
    aligned_matrix_mismatch,
    aperture_diameter,
    audit_mksc_union,
    exact_and_plane_ris_bs_matrices,
    fraunhofer_distance,
)
from src.experiments.final_mksc_ccop_common import make_paper_config
from src.projections_delay import pole_from_tau, tau_from_pole


def _scene(bs_position):
    config = make_paper_config(20260727, -10.0)
    config = copy.deepcopy(config)
    config["p_B"] = np.asarray(bs_position, dtype=float)
    return generate_scene(config, np.random.default_rng(20260727))


def test_reference_aperture_and_delay_roundtrip():
    scene = _scene([0.0, 0.0, 1.0])
    diameter = aperture_diameter(scene["ris_grid"])
    fraunhofer = fraunhofer_distance(scene["ris_grid"], scene["wavelength"])
    assert np.isclose(diameter, 0.2225845440004024)
    assert np.isclose(fraunhofer, 19.8312710967)
    tau = 137.5e-9
    recovered = tau_from_pole(pole_from_tau(tau, scene["delta_f"]), scene["delta_f"])
    assert abs(recovered - tau) < 1.0e-20


def test_spherical_plane_residual_decreases_with_range():
    residuals = []
    for bs_position in ([0.0, 0.0, 1.0], [-15.0, 0.0, 1.0], [-35.0, 0.0, 1.0]):
        scene = _scene(bs_position)
        exact, _, plane = exact_and_plane_ris_bs_matrices(scene, 0)
        residuals.append(
            aligned_matrix_mismatch(exact, plane)[
                "normalized_frobenius_residual"
            ]
        )
    assert residuals[0] > residuals[1] > residuals[2]


def test_formal_mksc_union_is_96_to_6_at_old_geometry():
    scene = _scene([0.0, 0.0, 1.0])
    audit = audit_mksc_union(scene, rtol=1.0e-12, absolute_tolerance=1.0e-10)
    assert audit["compression_statement"] == "96->6"
    assert audit["relative_rank"] == 6
    assert audit["formal_basis_rank"] == 6
    assert audit["expected_rank_reached"]
