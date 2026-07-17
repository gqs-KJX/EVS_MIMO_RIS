import numpy as np

from src.config import default_config
from src.main_single_proposed import solve_stage2_legacy_multistart
from src.projections_delay import pole_from_tau
from src.stage2_rescue import Stage2CommonState, decoupled_clock_estimate


C0 = 299_792_458.0
DELTA_F = 5.0e6
RIS_CENTERS = np.array(
    [[4.2, -2.2, 1.05], [5.1, 2.1, 1.15], [4.8, 0.0, 1.25]], dtype=float
)
D_RB = np.array([4.7416, 5.5175, 4.8065], dtype=float)


def _scene() -> dict:
    return {
        "K": 3,
        "c0": C0,
        "delta_f": DELTA_F,
        "ris_centers": RIS_CENTERS,
        "d_RB": D_RB,
    }


def _state(position, delta_t, *, range_bias=None, delay_bias_m=None, sigma_tau_s=5.2e-10):
    """Build a Stage2CommonState consistent with a UE at ``position``."""
    scene = _scene()
    k_paths = scene["K"]
    rho = np.linalg.norm(position[None, :] - RIS_CENTERS, axis=1)

    range_bias = np.zeros(k_paths) if range_bias is None else np.asarray(range_bias, float)
    delay_bias_m = (
        np.zeros(k_paths) if delay_bias_m is None else np.asarray(delay_bias_m, float)
    )

    # RIS near-field absolute range, and the clock-biased OFDM pseudorange.
    ris_range = rho + range_bias
    pseudorange = rho + C0 * delta_t + delay_bias_m
    tau_hat = (pseudorange + D_RB) / C0

    ris_eta = np.zeros((k_paths, 3), dtype=float)
    ris_eta[:, 0] = ris_range

    records = [
        {
            "panel_index": k,
            "valid": True,
            "reject_reason": "",
            "position": np.asarray(position, dtype=float).copy(),
        }
        for k in range(k_paths)
    ]
    return Stage2CommonState(
        stage1_estimate={},
        refined_estimate={
            "ris_eta": ris_eta,
            "poles": np.array([pole_from_tau(t, DELTA_F) for t in tau_hat]),
        },
        rescue_config={},
        tau_hat_s=tau_hat,
        sigma_tau_s=np.full(k_paths, sigma_tau_s),
        sigma_tau_sq_s2=np.full(k_paths, sigma_tau_s**2),
        sigma_tau_source="test",
        sigma_tau_used_floor=False,
        local_fix_records=records,
        common_refinement_success=True,
        common_refinement_runtime_s=0.0,
    ), scene


def test_decoupled_clock_is_exact_and_position_free():
    """m_k - r_k = s exactly, so the clock is recovered without any position."""
    config = default_config()
    delta_t = 5.0e-9
    state, scene = _state(np.array([1.25, 0.55, 0.75]), delta_t)

    result = decoupled_clock_estimate(state, scene, config)

    assert result["available"] is True
    assert result["num_inliers"] == 3
    assert np.isclose(result["clock_s"], delta_t, rtol=0.0, atol=1e-16)


def test_decoupled_clock_is_invariant_to_ue_position():
    """The estimate must not change when the UE moves, given consistent inputs."""
    config = default_config()
    delta_t = 5.0e-9
    first, scene = _state(np.array([1.25, 0.55, 0.75]), delta_t)
    second, _ = _state(np.array([2.10, -0.90, 1.30]), delta_t)

    a = decoupled_clock_estimate(first, scene, config)
    b = decoupled_clock_estimate(second, scene, config)

    assert a["available"] and b["available"]
    assert np.isclose(a["clock_s"], b["clock_s"], atol=1e-15)


def test_decoupled_clock_rejects_a_single_corrupted_delay_at_k3():
    """One gross delay outlier is screened out even with only three panels.

    Clock annihilation cannot do this: it consumes one of the K equations and
    leaves K - 1 = 2 coupled rows, so no path can be dropped.
    """
    config = default_config()
    delta_t = 5.0e-9
    outlier = np.array([0.0, 40.0, 0.0])  # 40 m gross error on path 1
    state, scene = _state(np.array([1.25, 0.55, 0.75]), delta_t, delay_bias_m=outlier)

    result = decoupled_clock_estimate(state, scene, config)

    assert result["available"] is True
    assert result["num_inliers"] == 2
    assert bool(result["inlier_mask"][1]) is False
    assert abs(C0 * (result["clock_s"] - delta_t)) < 1e-6


def test_decoupled_clock_reports_unavailable_on_bad_inputs():
    config = default_config()
    state, scene = _state(np.array([1.25, 0.55, 0.75]), 5.0e-9)
    for record in state.local_fix_records:
        record["valid"] = False

    result = decoupled_clock_estimate(state, scene, config)

    assert result["available"] is False
    assert result["reason"] == "no_valid_local_fixes"
    assert not np.isfinite(result["clock_s"])


def test_decoupled_clock_rejects_out_of_bounds_clock():
    config = default_config()
    config["delta_t_bounds"] = (-1.0e-9, 1.0e-9)
    state, scene = _state(np.array([1.25, 0.55, 0.75]), 5.0e-8)

    result = decoupled_clock_estimate(state, scene, config)

    assert result["available"] is False
    assert result["reason"] == "clock_out_of_bounds"


def test_legacy_position_seed_averages_only_screened_valid_fixes():
    config = default_config()
    position = np.array([1.25, 0.55, 0.75])
    state, scene = _state(position, 5.0e-9)
    state.local_fix_records[0]["position"] = np.array([1.05, 0.35, 0.65])
    state.local_fix_records[1]["position"] = np.array([20.0, -30.0, 40.0])
    state.local_fix_records[1]["valid"] = False
    state.local_fix_records[1]["reject_reason"] = "local_geometry_out_of_domain"
    state.local_fix_records[2]["position"] = np.array([1.45, 0.75, 0.85])

    result = solve_stage2_legacy_multistart(state, scene, config)

    np.testing.assert_allclose(
        result["diagnostics"]["seed_position"], position, rtol=0.0, atol=1.0e-15
    )


def test_legacy_position_seed_can_be_reported_without_position_polish():
    config = default_config()
    config["stage2_position_polish_enabled"] = False
    position = np.array([1.25, 0.55, 0.75])
    state, scene = _state(position, 5.0e-9)
    state.local_fix_records[0]["position"] = position + np.array([-0.2, 0.0, 0.0])
    state.local_fix_records[2]["position"] = position + np.array([0.2, 0.0, 0.0])

    result = solve_stage2_legacy_multistart(state, scene, config)

    np.testing.assert_allclose(result["position"], position, rtol=0.0, atol=1.0e-15)
    assert result["diagnostics"]["position_polish_enabled"] is False
    assert result["diagnostics"]["polish_skipped_reason"] == "disabled"
    assert result["diagnostics"]["polish_accepted"] is False


def test_legacy_position_seed_is_unavailable_without_valid_fixes():
    config = default_config()
    state, scene = _state(np.array([1.25, 0.55, 0.75]), 5.0e-9)
    for record in state.local_fix_records:
        record["valid"] = False
        record["reject_reason"] = "geometry_validity_false"

    result = solve_stage2_legacy_multistart(state, scene, config)

    assert result["rescue_available"] is False
    assert result["failure_reason"] == "no_valid_local_fixes"
    assert result["diagnostics"]["stage2_failure_reason"] == "no_valid_local_fixes"
