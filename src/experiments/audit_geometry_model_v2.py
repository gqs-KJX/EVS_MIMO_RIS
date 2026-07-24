"""Deterministic audit for the frozen oriented geometry/model v2.

This entry point runs no estimator and no Monte Carlo trial.  It checks the
calibrated RIS frames, UE-box-induced local search domains, near/far-field
distance assumptions, and the center-frequency spatial-narrowband validity
gate against an exact per-subcarrier two-hop element path difference.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from ..channel_model import generate_scene
from ..config import default_config
from ..geometry import (
    solve_ue_box_bs_maximin_rotation,
    ue_box_corners,
    validate_ris_rotations,
)
from ..projections_ris import local_ris_search_config


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _fraunhofer_distance(scene: dict) -> float:
    grid = np.asarray(scene["ris_grid"], dtype=float)
    diameter = float(np.linalg.norm(np.max(grid, axis=0) - np.min(grid, axis=0)))
    return float(2.0 * diameter**2 / float(scene["wavelength"]))


def _global_element_positions(scene: dict, panel: int) -> np.ndarray:
    return (
        np.asarray(scene["ris_centers"][panel], dtype=float)
        + np.asarray(scene["ris_grid"], dtype=float)
        @ np.asarray(scene["rotations"][panel], dtype=float)
    )


def _two_hop_path_differences(
    scene: dict,
    panel: int,
    position: np.ndarray,
) -> np.ndarray:
    center = np.asarray(scene["ris_centers"][panel], dtype=float)
    elements = _global_element_positions(scene, panel)
    p_u = np.asarray(position, dtype=float).reshape(3)
    p_b = np.asarray(scene["p_B"], dtype=float)
    ue_offsets = np.linalg.norm(p_u - elements, axis=1) - np.linalg.norm(
        p_u - center
    )
    bs_offsets = np.linalg.norm(p_b - elements, axis=1) - np.linalg.norm(
        p_b - center
    )
    return ue_offsets + bs_offsets


def _aligned_frequency_residual(
    path_differences_m: np.ndarray,
    frequency_offsets_hz: np.ndarray,
    c0: float,
) -> float:
    """Return exact-frequency/SNB mismatch after the best common complex scale."""
    delta = np.asarray(path_differences_m, dtype=float).reshape(-1)
    offsets = np.asarray(frequency_offsets_hz, dtype=float).reshape(-1)
    phase = np.exp(
        -1j * 2.0 * np.pi * offsets[:, None] * delta[None, :] / float(c0)
    )
    best_scale = np.mean(phase)
    return float(np.sqrt(max(0.0, 1.0 - abs(best_scale) ** 2)))


def audit_orientations_and_search(scene: dict, config: dict) -> dict:
    rotations = validate_ris_rotations(
        scene["rotations"], expected_count=int(scene["K"])
    )
    corners = ue_box_corners(config["ue_bounds"])
    panels = []
    for panel in range(int(scene["K"])):
        center = np.asarray(scene["ris_centers"][panel], dtype=float)
        normal = rotations[panel, 2]
        bs_direction = np.asarray(scene["p_B"], dtype=float) - center
        bs_direction /= np.linalg.norm(bs_direction)
        ue_directions = corners - center
        ue_directions /= np.linalg.norm(ue_directions, axis=1, keepdims=True)
        search = local_ris_search_config(scene, config, panel)
        solved_rotation, solved_diagnostics = solve_ue_box_bs_maximin_rotation(
            center,
            scene["p_B"],
            config["ue_bounds"],
            initial_normal=normal,
        )
        panels.append(
            {
                "panel": panel,
                "rotation_orthogonality_error": float(
                    np.max(
                        np.abs(rotations[panel] @ rotations[panel].T - np.eye(3))
                    )
                ),
                "rotation_determinant": float(np.linalg.det(rotations[panel])),
                "normal_global": normal,
                "bs_normal_cosine": float(normal @ bs_direction),
                "ue_corner_min_normal_cosine": float(
                    np.min(ue_directions @ normal)
                ),
                "worst_link_normal_cosine": float(
                    min(normal @ bs_direction, np.min(ue_directions @ normal))
                ),
                "worst_link_angle_deg": float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                min(
                                    normal @ bs_direction,
                                    np.min(ue_directions @ normal),
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
                "frozen_vs_resolved_rotation_max_error": float(
                    np.max(np.abs(rotations[panel] - solved_rotation))
                ),
                "resolved_maximin": solved_diagnostics,
                "range_bounds_exact_m": search["range_bounds_exact"],
                "range_bounds_guarded_m": search["range_bounds"],
                "elevation_bounds_exact_deg": np.degrees(
                    search["elev_bounds_exact"]
                ),
                "elevation_bounds_guarded_deg": np.degrees(
                    search["elev_bounds"]
                ),
                "azimuth_bounds_exact_deg": np.degrees(
                    search["az_bounds_exact"]
                ),
                "azimuth_bounds_guarded_deg": np.degrees(search["az_bounds"]),
                "azimuth_full_circle": bool(search["azimuth_full_circle"]),
            }
        )
    return {
        "all_rotations_valid": True,
        "all_frozen_rotations_match_resolved_maximin": bool(
            all(
                item["frozen_vs_resolved_rotation_max_error"] <= 1.0e-10
                for item in panels
            )
        ),
        "all_bs_and_ue_corners_in_reflection_halfspace": bool(
            all(
                item["bs_normal_cosine"] > 0.0
                and item["ue_corner_min_normal_cosine"] > 0.0
                for item in panels
            )
        ),
        "panels": panels,
    }


def audit_near_far_field(scene: dict, config: dict) -> dict:
    fraunhofer = _fraunhofer_distance(scene)
    corners = ue_box_corners(config["ue_bounds"])
    ue_distances = np.linalg.norm(
        corners[:, None, :]
        - np.asarray(scene["ris_centers"], dtype=float)[None, :, :],
        axis=2,
    )
    ris_bs = np.asarray(scene["d_RB"], dtype=float)
    return {
        "ris_fraunhofer_distance_m": fraunhofer,
        "maximum_ue_ris_distance_m": float(np.max(ue_distances)),
        "minimum_ris_bs_distance_m": float(np.min(ris_bs)),
        "all_ue_box_corners_near_field": bool(np.all(ue_distances < fraunhofer)),
        "all_ris_bs_links_far_field": bool(np.all(ris_bs > fraunhofer)),
        "maximum_ue_ris_fraunhofer_ratio": float(
            np.max(ue_distances / fraunhofer)
        ),
        "minimum_ris_bs_fraunhofer_ratio": float(
            np.min(ris_bs / fraunhofer)
        ),
    }


def audit_spatial_narrowband(
    scene: dict,
    config: dict,
    *,
    optimizer_maxiter: int = 60,
) -> dict:
    audit_config = dict(config["spatial_narrowband_audit"])
    ue_bounds = np.asarray(config["ue_bounds"], dtype=float)
    scipy_bounds = [tuple(item) for item in ue_bounds]
    q_index = np.asarray(scene["subcarrier_indices"], dtype=float)
    frequency_offsets = q_index * float(scene["delta_f"])
    c0 = float(scene["c0"])
    corner_positions = ue_box_corners(ue_bounds)
    panels = []

    for panel in range(int(scene["K"])):
        def max_abs_path_difference(position: np.ndarray) -> float:
            return float(
                np.max(
                    np.abs(
                        _two_hop_path_differences(scene, panel, position)
                    )
                )
            )

        def aligned_residual(position: np.ndarray) -> float:
            return _aligned_frequency_residual(
                _two_hop_path_differences(scene, panel, position),
                frequency_offsets,
                c0,
            )

        corner_delta = np.asarray(
            [max_abs_path_difference(position) for position in corner_positions]
        )
        corner_residual = np.asarray(
            [aligned_residual(position) for position in corner_positions]
        )
        delta_result = differential_evolution(
            lambda position: -max_abs_path_difference(position),
            scipy_bounds,
            seed=20260724 + panel,
            maxiter=int(optimizer_maxiter),
            popsize=10,
            tol=1.0e-10,
            polish=True,
            workers=1,
            updating="immediate",
        )
        residual_result = differential_evolution(
            lambda position: -aligned_residual(position),
            scipy_bounds,
            seed=20260734 + panel,
            maxiter=int(optimizer_maxiter),
            popsize=10,
            tol=1.0e-10,
            polish=True,
            workers=1,
            updating="immediate",
        )
        delta_max = max(float(np.max(corner_delta)), -float(delta_result.fun))
        residual_max = max(
            float(np.max(corner_residual)), -float(residual_result.fun)
        )
        panels.append(
            {
                "panel": panel,
                "delta_path_max_m": delta_max,
                "delta_path_argmax_position_m": np.asarray(
                    delta_result.x, dtype=float
                ),
                "aligned_residual_max": residual_max,
                "aligned_residual_argmax_position_m": np.asarray(
                    residual_result.x, dtype=float
                ),
                "default_ue_aligned_residual": aligned_residual(
                    scene["p_u_true"]
                ),
            }
        )

    delta_path_max = float(max(item["delta_path_max_m"] for item in panels))
    aligned_residual_max = float(
        max(item["aligned_residual_max"] for item in panels)
    )
    frequency_offset_max = float(np.max(np.abs(frequency_offsets)))
    phase_residual_max = float(
        2.0 * np.pi * frequency_offset_max * delta_path_max / c0
    )
    phase_threshold = float(audit_config["max_phase_residual_rad"])
    preferred_threshold = float(audit_config["preferred_aligned_residual"])
    hard_threshold = float(audit_config["hard_aligned_residual"])
    return {
        "subcarrier_index_mode": scene["subcarrier_index_mode"],
        "subcarrier_indices": q_index,
        "occupied_bandwidth_hz": float(
            (np.max(q_index) - np.min(q_index)) * scene["delta_f"]
        ),
        "maximum_frequency_offset_hz": frequency_offset_max,
        "delta_path_max_m": delta_path_max,
        "phase_residual_max_rad": phase_residual_max,
        "phase_residual_max_deg": float(np.degrees(phase_residual_max)),
        "aligned_residual_max": aligned_residual_max,
        "phase_threshold_rad": phase_threshold,
        "preferred_aligned_residual_threshold": preferred_threshold,
        "hard_aligned_residual_threshold": hard_threshold,
        "phase_gate_pass": bool(phase_residual_max <= phase_threshold),
        "preferred_aligned_residual_pass": bool(
            aligned_residual_max <= preferred_threshold
        ),
        "hard_aligned_residual_pass": bool(
            aligned_residual_max <= hard_threshold
        ),
        "hard_gate_pass": bool(
            phase_residual_max <= phase_threshold
            and aligned_residual_max <= hard_threshold
        ),
        "panels": panels,
        "optimizer": {
            "method": "deterministic seeded differential_evolution",
            "maxiter": int(optimizer_maxiter),
            "box_corners_always_included": True,
        },
    }


def run_audit(config: dict, *, optimizer_maxiter: int = 60) -> dict:
    scene = generate_scene(config, np.random.default_rng(int(config["seed"])))
    orientation = audit_orientations_and_search(scene, config)
    propagation = audit_near_far_field(scene, config)
    spatial = audit_spatial_narrowband(
        scene, config, optimizer_maxiter=optimizer_maxiter
    )
    passed = bool(
        orientation["all_rotations_valid"]
        and orientation["all_frozen_rotations_match_resolved_maximin"]
        and orientation["all_bs_and_ue_corners_in_reflection_halfspace"]
        and propagation["all_ue_box_corners_near_field"]
        and propagation["all_ris_bs_links_far_field"]
        and spatial["hard_gate_pass"]
    )
    return {
        "schema": "evs_mimo_geometry_model_v2_audit_v1",
        "geometry_version": scene["geometry_version"],
        "pass": passed,
        "orientation_and_search": orientation,
        "near_far_field": propagation,
        "spatial_narrowband": spatial,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimizer-maxiter",
        type=int,
        default=60,
        help="deterministic continuous UE-box audit iterations",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="optional JSON output path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.optimizer_maxiter < 1:
        raise ValueError("--optimizer-maxiter must be positive")
    result = run_audit(
        default_config(), optimizer_maxiter=int(args.optimizer_maxiter)
    )
    payload = json.dumps(_json_safe(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
