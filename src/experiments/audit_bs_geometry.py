"""Deterministic BS-geometry audit for the mixed near/far-field EVS-RIS model.

This entry point does not run an estimator or a noisy Monte Carlo experiment.
It rebuilds the repository's calibrated scene for each candidate BS position
and audits propagation consistency, delay ambiguity, Maxwell--Jones
separability, MKSC compression, and the matched free-Jones geometry EFIM.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pathlib
import shlex
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..channel_model import (
    channel_components,
    evs_component_selection,
    generate_scene,
    synthesize_raw_tensor,
)
from ..ccop_stage1_initializer import known_evs_union_basis
from ..global_vp import data_only_efim_diagnostic
from ..projections_delay import pole_from_tau, tau_from_pole
from ..validation_artifacts import (
    canonical_hash,
    git_value,
    validation_environment,
)
from .final_mksc_ccop_common import make_paper_config
from .run_paper_ablation_figures import (
    _truth_init_estimate,
    position_peb_from_global_efim,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = pathlib.Path("results/geometry_audit")
DEFAULT_SCENE_SEED = 20260727
DEFAULT_FULL_GRID_SHAPE = (5, 5, 2)
DEFAULT_QUICK_GRID_SHAPE = (2, 2, 1)
MASK_MODES = ("scalar", "dual_pol", "full_6d")


def _float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return values


def _int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _shape3(value: str) -> tuple[int, int, int]:
    parsed = _int_list(value)
    if len(parsed) != 3 or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("shape must contain three positive integers")
    return tuple(parsed)  # type: ignore[return-value]


def _position3(value: str) -> np.ndarray:
    parsed = _float_list(value)
    if len(parsed) != 3 or not np.all(np.isfinite(parsed)):
        raise argparse.ArgumentTypeError("position must be finite x,y,z")
    return np.asarray(parsed, dtype=float)


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
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0.0 else "-Infinity"
        return number
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        return {"real": float(number.real), "imag": float(number.imag)}
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _deep_update(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_audit_config(args: argparse.Namespace) -> dict:
    """Resolve the frozen paper/default configuration without modifying it."""
    if args.preset == "paper":
        config = make_paper_config(args.scene_seeds[0], args.snr_db)
    else:
        from ..config import default_config

        config = default_config()
        config["seed"] = int(args.scene_seeds[0])
        config["SNR_dB"] = float(args.snr_db)
    if args.config is not None:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        if "config" in payload and isinstance(payload["config"], dict):
            payload = payload["config"]
        config = _deep_update(config, payload)
    config["K"] = int(config.get("K", 3))
    if config["K"] != 3:
        raise ValueError("the reference geometry audit currently requires K=3")
    config.setdefault("crb", {})
    config["crb"]["enable_unscaled_efim_cache"] = False
    config["SNR_dB"] = float(args.snr_db)
    return config


def aperture_diameter(points: np.ndarray) -> float:
    """Return the maximum corner-to-corner span of a rectangular array grid."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 1:
        raise ValueError("array points must have shape (M, 3)")
    return float(np.linalg.norm(np.max(array, axis=0) - np.min(array, axis=0)))


def fraunhofer_distance(points: np.ndarray, wavelength: float) -> float:
    diameter = aperture_diameter(points)
    return float(2.0 * diameter**2 / float(wavelength))


def bs_sensor_offsets(scene: dict) -> np.ndarray:
    """Return the repository's fixed global-x, half-wavelength BS ULA offsets."""
    count = int(scene["M_A"])
    offsets = np.zeros((count, 3), dtype=float)
    offsets[:, 0] = (
        np.arange(count, dtype=float) - (count - 1) / 2.0
    ) * float(scene["wavelength"]) / 2.0
    return offsets


def exact_and_plane_ris_bs_matrices(
    scene: dict,
    panel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact spherical and repository plane-wave RIS--BS phase matrices.

    The exact sign is selected so that its first-order expansion matches
    ``outer(v_B[k], a_RB[k])`` under the repository's steering conventions.
    The amplitude-aware matrix includes relative center-distance spreading
    ``d_RB / d_element_sensor``; the production model is phase-only.
    """
    panel = int(panel)
    rotation = np.asarray(scene["rotations"][panel], dtype=float)
    ris_global = (
        np.asarray(scene["ris_centers"][panel], dtype=float)
        + np.asarray(scene["ris_grid"], dtype=float) @ rotation
    )
    sensors_global = np.asarray(scene["p_B"], dtype=float) + bs_sensor_offsets(scene)
    distances = np.linalg.norm(
        sensors_global[:, None, :] - ris_global[None, :, :],
        axis=2,
    )
    center_distance = float(scene["d_RB"][panel])
    wavenumber = 2.0 * np.pi / float(scene["wavelength"])
    exact_phase = np.exp(1j * wavenumber * (distances - center_distance))
    exact_amplitude = (center_distance / distances) * exact_phase
    plane = np.outer(
        np.asarray(scene["v_B"][panel], dtype=complex),
        np.asarray(scene["a_RB"][panel], dtype=complex),
    )
    return exact_phase, exact_amplitude, plane


def aligned_matrix_mismatch(exact: np.ndarray, approximate: np.ndarray) -> dict:
    """Compare two complex matrices after the best common complex scale."""
    exact_array = np.asarray(exact, dtype=complex)
    approximate_array = np.asarray(approximate, dtype=complex)
    denominator = float(np.vdot(approximate_array, approximate_array).real)
    if denominator <= 0.0:
        raise ValueError("approximate matrix has zero energy")
    scale = np.vdot(approximate_array, exact_array) / denominator
    aligned = scale * approximate_array
    residual = exact_array - aligned
    normalized_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(exact_array), np.finfo(float).tiny)
    )
    correlation = float(
        abs(np.vdot(exact_array, approximate_array))
        / max(
            np.linalg.norm(exact_array) * np.linalg.norm(approximate_array),
            np.finfo(float).tiny,
        )
    )
    valid = (np.abs(exact_array) > 0.0) & (np.abs(aligned) > 0.0)
    phase_residual = np.angle(exact_array[valid] / aligned[valid])
    return {
        "best_scale_real": float(scale.real),
        "best_scale_imag": float(scale.imag),
        "normalized_frobenius_residual": normalized_residual,
        "normalized_correlation": correlation,
        "phase_residual_max_rad": float(np.max(np.abs(phase_residual))),
        "phase_residual_rms_rad": float(np.sqrt(np.mean(phase_residual**2))),
        "phase_residual_max_deg": float(np.degrees(np.max(np.abs(phase_residual)))),
        "phase_residual_rms_deg": float(
            np.degrees(np.sqrt(np.mean(phase_residual**2)))
        ),
    }


def position_grid(
    config: dict,
    shape: tuple[int, int, int],
    margin_m: float,
) -> list[np.ndarray]:
    """Reproduce the paper's Cartesian position-grid convention."""
    bounds = np.asarray(config["ue_bounds"], dtype=float)
    axes = []
    for dim, count in enumerate(shape):
        low = float(bounds[dim, 0] + margin_m)
        high = float(bounds[dim, 1] - margin_m)
        if high < low:
            raise ValueError("UE grid margin leaves an empty interval")
        axes.append(np.linspace(low, high, int(count)))
    return [
        np.array([x, y, z], dtype=float)
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    ]


def audit_distances(scene: dict, ue_positions: Sequence[np.ndarray]) -> dict:
    ff_distance = fraunhofer_distance(scene["ris_grid"], scene["wavelength"])
    ris_centers = np.asarray(scene["ris_centers"], dtype=float)
    ue_array = np.asarray(ue_positions, dtype=float)
    ue_ris = np.linalg.norm(
        ue_array[:, None, :] - ris_centers[None, :, :],
        axis=2,
    )
    default_ue_ris = np.linalg.norm(
        np.asarray(scene["p_u_true"], dtype=float)[None, :] - ris_centers,
        axis=1,
    )
    d_rb = np.asarray(scene["d_RB"], dtype=float)
    return {
        "wavelength_m": float(scene["wavelength"]),
        "ris_aperture_diameter_m": aperture_diameter(scene["ris_grid"]),
        "ris_fraunhofer_distance_m": ff_distance,
        "ris_bs_distances_m": d_rb,
        "ris_bs_fraunhofer_ratios": d_rb / ff_distance,
        "ris_bs_min_fraunhofer_ratio": float(np.min(d_rb / ff_distance)),
        "ue_ris_grid_distances_m": ue_ris,
        "ue_ris_grid_ratio_min": float(np.min(ue_ris / ff_distance)),
        "ue_ris_grid_ratio_median": float(np.median(ue_ris / ff_distance)),
        "ue_ris_grid_ratio_max": float(np.max(ue_ris / ff_distance)),
        "default_ue_ris_distances_m": default_ue_ris,
        "default_ue_ris_ratio_max": float(np.max(default_ue_ris / ff_distance)),
        "grid_all_near_field": bool(np.all(ue_ris < ff_distance)),
        "default_ue_all_near_field": bool(np.all(default_ue_ris < ff_distance)),
    }


def audit_ris_bs_mismatch(scene: dict) -> dict:
    panels = []
    for panel in range(int(scene["K"])):
        exact_phase, exact_amplitude, plane = exact_and_plane_ris_bs_matrices(
            scene, panel
        )
        panels.append(
            {
                "panel": panel,
                "phase_only": aligned_matrix_mismatch(exact_phase, plane),
                "amplitude_aware": aligned_matrix_mismatch(exact_amplitude, plane),
            }
        )
    return {
        "panels": panels,
        "phase_residual_max_rad": float(
            max(item["phase_only"]["phase_residual_max_rad"] for item in panels)
        ),
        "phase_residual_rms_max_rad": float(
            max(item["phase_only"]["phase_residual_rms_rad"] for item in panels)
        ),
        "normalized_residual_max": float(
            max(
                item["phase_only"]["normalized_frobenius_residual"]
                for item in panels
            )
        ),
        "normalized_correlation_min": float(
            min(item["phase_only"]["normalized_correlation"] for item in panels)
        ),
        "amplitude_aware_normalized_residual_max": float(
            max(
                item["amplitude_aware"]["normalized_frobenius_residual"]
                for item in panels
            )
        ),
    }


def audit_delays(
    scene: dict,
    config: dict,
    ue_positions: Sequence[np.ndarray],
) -> dict:
    centers = np.asarray(scene["ris_centers"], dtype=float)
    d_rb = np.asarray(scene["d_RB"], dtype=float)
    c0 = float(scene["c0"])
    clock_true = float(config["delta_t_true"])
    clock_bounds = np.asarray(config["delta_t_bounds"], dtype=float)
    ambiguity = 1.0 / float(scene["delta_f"])
    point_records = []
    separations = []
    absolute_taus = []
    wrap_errors = []
    for position in ue_positions:
        d_ur = np.linalg.norm(np.asarray(position)[None, :] - centers, axis=1)
        geom_tau = (d_ur + d_rb) / c0
        total_tau = geom_tau + clock_true
        pairwise = [
            abs(float(total_tau[left] - total_tau[right]))
            for left in range(total_tau.size)
            for right in range(left + 1, total_tau.size)
        ]
        min_separation = min(pairwise) if pairwise else float("inf")
        recovered = np.asarray(
            [
                tau_from_pole(pole_from_tau(tau, scene["delta_f"]), scene["delta_f"])
                for tau in total_tau
            ],
            dtype=float,
        )
        wrap_error = np.abs(recovered - total_tau)
        point_records.append(
            {
                "position_m": np.asarray(position, dtype=float),
                "ue_ris_distances_m": d_ur,
                "geometric_delays_s": geom_tau,
                "total_delays_s": total_tau,
                "total_delays_ns": total_tau * 1.0e9,
                "minimum_pairwise_separation_s": min_separation,
                "minimum_pairwise_separation_ns": min_separation * 1.0e9,
                "pole_roundtrip_max_error_s": float(np.max(wrap_error)),
            }
        )
        separations.append(min_separation)
        absolute_taus.extend(total_tau.tolist())
        wrap_errors.extend(wrap_error.tolist())
    absolute = np.asarray(absolute_taus, dtype=float)
    separation_array = np.asarray(separations, dtype=float)
    centers_to_grid = np.asarray(
        [
            np.linalg.norm(np.asarray(position)[None, :] - centers, axis=1)
            for position in ue_positions
        ]
    )
    min_possible = (
        centers_to_grid + d_rb[None, :]
    ) / c0 + float(clock_bounds[0])
    max_possible = (
        centers_to_grid + d_rb[None, :]
    ) / c0 + float(clock_bounds[1])
    interval_width = float(clock_bounds[1] - clock_bounds[0])
    return {
        "points": point_records,
        "ambiguity_period_s": ambiguity,
        "ambiguity_period_ns": ambiguity * 1.0e9,
        "clock_bounds_s": clock_bounds,
        "clock_interval_width_s": interval_width,
        "clock_interval_shorter_than_ambiguity": bool(interval_width < ambiguity),
        "delay_min_s": float(np.min(absolute)),
        "delay_median_s": float(np.median(absolute)),
        "delay_max_s": float(np.max(absolute)),
        "delay_min_ns": float(np.min(absolute) * 1.0e9),
        "delay_max_ns": float(np.max(absolute) * 1.0e9),
        "minimum_separation_s": float(np.min(separation_array)),
        "minimum_separation_ns": float(np.min(separation_array) * 1.0e9),
        "median_minimum_separation_ns": float(
            np.median(separation_array) * 1.0e9
        ),
        "maximum_minimum_separation_ns": float(
            np.max(separation_array) * 1.0e9
        ),
        "all_delays_unambiguous_at_true_clock": bool(
            np.all((absolute >= 0.0) & (absolute < ambiguity))
        ),
        "all_delays_unambiguous_over_clock_bounds": bool(
            np.all((min_possible >= 0.0) & (max_possible < ambiguity))
        ),
        "pole_roundtrip_max_error_s": float(np.max(wrap_errors)),
        "pole_conversion": "tau_from_pole maps to [0, 1/delta_f)",
        "nominal_dft_resolution_s": float(
            1.0 / (float(scene["N"]) * float(scene["delta_f"]))
        ),
    }


def _subspace_block(scene: dict, panel: int, mode: str) -> np.ndarray:
    theta = np.asarray(scene["Theta"][panel], dtype=complex)
    v_b = np.asarray(scene["v_B"][panel], dtype=complex)
    mask = np.tile(evs_component_selection(mode), int(scene["M_A"]))
    block = np.column_stack(
        [np.kron(v_b, theta[:, component]) for component in range(2)]
    )
    return mask[:, None] * block


def _orthonormal_basis(matrix: np.ndarray, rtol: float) -> tuple[np.ndarray, np.ndarray, int]:
    left, singular, _ = np.linalg.svd(np.asarray(matrix, dtype=complex), full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=complex), singular, 0
    rank = int(np.count_nonzero(singular > float(rtol) * singular[0]))
    return left[:, :rank], singular, rank


def audit_maxwell_jones(scene: dict, rtol: float) -> dict:
    modes = {}
    for mode in MASK_MODES:
        blocks = []
        panel_records = []
        for panel in range(int(scene["K"])):
            block = _subspace_block(scene, panel, mode)
            basis, singular, rank = _orthonormal_basis(block, rtol)
            nonzero = singular[:rank]
            panel_records.append(
                {
                    "panel": panel,
                    "rank": rank,
                    "singular_values": singular,
                    "nonzero_condition": (
                        float(nonzero[0] / nonzero[-1])
                        if nonzero.size and nonzero[-1] > 0.0
                        else float("inf")
                    ),
                    "row_rank_condition": (
                        "expected_rank_1_scalar"
                        if mode == "scalar" and rank == 1
                        else "full_jones_row_rank"
                        if rank == 2
                        else "rank_deficient"
                    ),
                }
            )
            blocks.append((block, basis))
        pairs = []
        for left in range(int(scene["K"])):
            for right in range(left + 1, int(scene["K"])):
                q_left = blocks[left][1]
                q_right = blocks[right][1]
                coherence_singular = np.linalg.svd(
                    q_left.conj().T @ q_right, compute_uv=False
                )
                angles = np.arccos(np.clip(coherence_singular, 0.0, 1.0))
                projector_residual = float(
                    np.linalg.norm(
                        q_right - q_left @ (q_left.conj().T @ q_right)
                    )
                    / max(np.sqrt(q_right.shape[1]), np.finfo(float).tiny)
                )
                pairs.append(
                    {
                        "pair": [left, right],
                        "principal_angles_rad": angles,
                        "principal_angles_deg": np.degrees(angles),
                        "minimum_principal_angle_deg": float(
                            np.degrees(np.min(angles))
                        ),
                        "sin_minimum_principal_angle": float(np.sin(np.min(angles))),
                        "projector_normalized_separation_residual": projector_residual,
                        "subspace_coherence": float(
                            np.max(coherence_singular)
                            if coherence_singular.size
                            else 0.0
                        ),
                    }
                )
        modes[mode] = {"panels": panel_records, "pairs": pairs}
    full_pairs = modes["full_6d"]["pairs"]
    return {
        "modes": modes,
        "full_pair_minimum_angles_deg": [
            item["minimum_principal_angle_deg"] for item in full_pairs
        ],
        "full_minimum_principal_angle_deg": float(
            min(item["minimum_principal_angle_deg"] for item in full_pairs)
        ),
        "full_maximum_subspace_coherence": float(
            max(item["subspace_coherence"] for item in full_pairs)
        ),
        "dual_all_rank_two": bool(
            all(item["rank"] == 2 for item in modes["dual_pol"]["panels"])
        ),
        "full_all_rank_two": bool(
            all(item["rank"] == 2 for item in modes["full_6d"]["panels"])
        ),
    }


def audit_mksc_union(scene: dict, rtol: float, absolute_tolerance: float) -> dict:
    blocks = [
        _subspace_block(scene, panel, "full_6d")
        for panel in range(int(scene["K"]))
    ]
    union = np.column_stack(blocks)
    singular = np.linalg.svd(union, compute_uv=False)
    relative_rank = int(np.count_nonzero(singular > rtol * singular[0]))
    absolute_rank = int(np.count_nonzero(singular > absolute_tolerance))
    basis, formal = known_evs_union_basis(scene, relative_tolerance=rtol)
    block_coherence = []
    for left in range(int(scene["K"])):
        q_left, _, _ = _orthonormal_basis(blocks[left], rtol)
        for right in range(left + 1, int(scene["K"])):
            q_right, _, _ = _orthonormal_basis(blocks[right], rtol)
            block_coherence.append(
                {
                    "pair": [left, right],
                    "coherence": float(
                        np.max(
                            np.linalg.svd(
                                q_left.conj().T @ q_right, compute_uv=False
                            )
                        )
                    ),
                }
            )
    expected_rank = 2 * int(scene["K"])
    retained = singular[:relative_rank]
    return {
        "original_evs_dimension": int(scene["I"]),
        "nominal_columns": int(union.shape[1]),
        "expected_rank": expected_rank,
        "singular_values": singular,
        "relative_tolerance": float(rtol),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_rank": relative_rank,
        "absolute_rank": absolute_rank,
        "formal_basis_rank": int(formal["stage1_evs_union_rank"]),
        "formal_basis_shape": list(basis.shape),
        "sigma_last_over_first": float(
            singular[expected_rank - 1] / singular[0]
            if singular.size >= expected_rank and singular[0] > 0.0
            else 0.0
        ),
        "nonzero_condition": float(
            retained[0] / retained[-1]
            if retained.size and retained[-1] > 0.0
            else float("inf")
        ),
        "block_subspace_coherence": block_coherence,
        "compression_statement": (
            f"{int(scene['I'])}->{int(formal['stage1_evs_union_rank'])}"
        ),
        "expected_rank_reached": bool(
            relative_rank == expected_rank
            and int(formal["stage1_evs_union_rank"]) == expected_rank
        ),
    }


def _signal_noise_variance(
    y_true: np.ndarray,
    scene: dict,
    snr_db: float,
) -> float:
    active = np.asarray(scene["evs_observation_mask"], dtype=bool)
    signal_power = float(np.mean(np.abs(np.asarray(y_true)[active]) ** 2))
    return float(signal_power / (10.0 ** (float(snr_db) / 10.0)))


def _rank_condition(matrix: np.ndarray, rtol: float) -> tuple[int, float]:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    if singular.size == 0:
        return 0, float("inf")
    threshold = rtol * max(float(singular[0]), 1.0)
    rank = int(np.count_nonzero(singular > threshold))
    positive = singular[singular > threshold]
    condition = (
        float(singular[0] / positive[-1]) if positive.size else float("inf")
    )
    return rank, condition


def audit_efim_point(
    scene: dict,
    config: dict,
    position: np.ndarray,
    rtol: float,
) -> dict:
    components = channel_components(
        scene,
        np.asarray(position, dtype=float),
        float(config["delta_t_true"]),
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(components, scene["beta_true"])
    sigma2 = _signal_noise_variance(y_true, scene, config["SNR_dB"])
    init = _truth_init_estimate(scene, components)
    diagnostic = data_only_efim_diagnostic(
        y_true,
        position,
        config["delta_t_true"],
        init,
        scene,
        config,
        sigma2=sigma2,
    )
    efim = np.asarray(diagnostic["data_only_scaled_efim"], dtype=float)
    efim = 0.5 * (efim + efim.T)
    rank, condition = _rank_condition(efim, rtol)
    eigenvalues = np.linalg.eigvalsh(efim)
    peb, peb_diagnostic = position_peb_from_global_efim(
        efim,
        diagnostic["data_only_scaled_efim_parameter_order"],
        already_clock_eliminated=False,
        condition_threshold=float(
            config.get("global_vp", {}).get("efim_cond_threshold", 1.0e12)
        ),
        return_diagnostics=True,
    )
    if rank == 4 and np.isfinite(condition):
        covariance = np.linalg.pinv(efim, rcond=rtol)
        axis_crb = np.sqrt(np.maximum(np.diag(covariance)[:3], 0.0))
        ceb_s = float(
            np.sqrt(max(float(covariance[3, 3]), 0.0)) / float(scene["c0"])
        )
    else:
        axis_crb = np.full(3, np.nan)
        ceb_s = float("nan")
    return {
        "position_m": np.asarray(position, dtype=float),
        "rank": rank,
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "condition_number": condition,
        "position_condition_number_after_clock_schur": float(
            peb_diagnostic["efim_condition_number"]
        ),
        "peb_m": float(peb),
        "ceb_s": ceb_s,
        "ceb_ns": ceb_s * 1.0e9,
        "axis_crb_m": axis_crb,
        "sigma2": sigma2,
        "efim_cache_hit": bool(diagnostic["efim_unscaled_cache_hit"]),
        "efim_cache_key": str(diagnostic["efim_unscaled_cache_key"]),
        "warning": str(peb_diagnostic["warning"]),
    }


def _percentiles(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "maximum": float("nan"),
        }
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "maximum": float(np.max(finite)),
    }


def audit_efim_grid(
    config: dict,
    bs_position: np.ndarray,
    ue_positions: Sequence[np.ndarray],
    scene_seeds: Sequence[int],
    rtol: float,
) -> dict:
    records = []
    default_position = np.asarray(config["p_u_true"], dtype=float)
    evaluation_positions = [default_position] + [
        np.asarray(item, dtype=float)
        for item in ue_positions
        if not np.allclose(item, default_position, rtol=0.0, atol=1.0e-14)
    ]
    for seed in scene_seeds:
        local_config = copy.deepcopy(config)
        local_config["p_B"] = np.asarray(bs_position, dtype=float)
        local_config["seed"] = int(seed)
        scene = generate_scene(local_config, np.random.default_rng(int(seed)))
        seen_keys = set()
        for position in evaluation_positions:
            record = audit_efim_point(scene, local_config, position, rtol)
            record["scene_seed"] = int(seed)
            record["is_default_ue"] = bool(
                np.allclose(position, default_position, rtol=0.0, atol=1.0e-14)
            )
            if record["efim_cache_key"] in seen_keys:
                raise RuntimeError("EFIM cache key repeated across distinct UE points")
            seen_keys.add(record["efim_cache_key"])
            records.append(record)
    peb_stats = _percentiles(item["peb_m"] for item in records)
    ceb_stats = _percentiles(item["ceb_ns"] for item in records)
    condition_stats = _percentiles(item["condition_number"] for item in records)
    min_eigen_stats = _percentiles(item["minimum_eigenvalue"] for item in records)
    axis_stats = {
        axis: _percentiles(item["axis_crb_m"][index] for item in records)
        for index, axis in enumerate(("x", "y", "z"))
    }
    finite_peb = [item for item in records if np.isfinite(item["peb_m"])]
    worst = (
        max(finite_peb, key=lambda item: item["peb_m"]) if finite_peb else records[0]
    )
    default_records = [item for item in records if item["is_default_ue"]]
    default_peb = float(np.median([item["peb_m"] for item in default_records]))
    grid_peb = np.asarray(
        [item["peb_m"] for item in records if not item["is_default_ue"]],
        dtype=float,
    )
    grid_peb = grid_peb[np.isfinite(grid_peb)]
    default_percentile = (
        float(100.0 * np.mean(grid_peb <= default_peb))
        if grid_peb.size and np.isfinite(default_peb)
        else float("nan")
    )
    return {
        "scene_seeds": list(scene_seeds),
        "jones_gain_realization_policy": (
            "fixed paper-consistent scene seed(s); identical Omega/Jones/gain "
            "realizations are reused across BS candidates"
        ),
        "records": records,
        "rank_deficiency_rate": float(
            np.mean([item["rank"] < 4 for item in records])
        ),
        "minimum_rank": int(min(item["rank"] for item in records)),
        "minimum_eigenvalue_stats": min_eigen_stats,
        "condition_number_stats": condition_stats,
        "peb_m_stats": peb_stats,
        "ceb_ns_stats": ceb_stats,
        "axis_crb_m_stats": axis_stats,
        "worst_case_position_m": worst["position_m"],
        "worst_case_scene_seed": int(worst["scene_seed"]),
        "default_ue_peb_m": default_peb,
        "default_ue_peb_percentile": default_percentile,
        "any_cache_hit": bool(any(item["efim_cache_hit"] for item in records)),
    }


def default_thresholds(args: argparse.Namespace) -> dict:
    return {
        "minimum_ris_bs_fraunhofer_ratio": float(args.min_farfield_ratio),
        "preferred_ris_bs_fraunhofer_ratio": float(args.preferred_farfield_ratio),
        "maximum_ue_ris_fraunhofer_ratio": float(args.max_nearfield_ratio),
        "maximum_phase_residual_rad": float(args.max_phase_residual_rad),
        "maximum_phase_rms_rad": float(args.max_phase_rms_rad),
        "maximum_normalized_channel_residual": float(
            args.max_channel_residual
        ),
        "minimum_delay_separation_ns": float(args.min_delay_separation_ns),
        "minimum_maxwell_angle_deg": float(args.min_maxwell_angle_deg),
        "minimum_union_sigma_ratio": float(args.min_union_sigma_ratio),
        "maximum_efim_condition": float(args.max_efim_condition),
        "maximum_worst_peb_m": float(args.max_worst_peb_m),
        "preferred_channel_residual": float(args.preferred_channel_residual),
        "preferred_efim_condition": float(args.preferred_efim_condition),
    }


def rejection_reasons(candidate: dict, thresholds: dict) -> list[str]:
    distance = candidate["distance"]
    mismatch = candidate["ris_bs_mismatch"]
    delays = candidate["delay"]
    maxwell = candidate["maxwell_jones"]
    union = candidate["mksc_union"]
    efim = candidate.get("efim")
    reasons = []
    if distance["ris_bs_min_fraunhofer_ratio"] < thresholds[
        "minimum_ris_bs_fraunhofer_ratio"
    ]:
        reasons.append("ris_bs_not_far_field")
    if distance["ue_ris_grid_ratio_max"] >= thresholds[
        "maximum_ue_ris_fraunhofer_ratio"
    ]:
        reasons.append("ue_grid_not_near_field")
    if mismatch["phase_residual_max_rad"] > thresholds[
        "maximum_phase_residual_rad"
    ]:
        reasons.append("plane_wave_phase_max_mismatch")
    if mismatch["phase_residual_rms_max_rad"] > thresholds[
        "maximum_phase_rms_rad"
    ]:
        reasons.append("plane_wave_phase_rms_mismatch")
    if mismatch["normalized_residual_max"] > thresholds[
        "maximum_normalized_channel_residual"
    ]:
        reasons.append("plane_wave_channel_residual")
    if not delays["all_delays_unambiguous_over_clock_bounds"]:
        reasons.append("delay_ambiguity_over_clock_bounds")
    if not delays["clock_interval_shorter_than_ambiguity"]:
        reasons.append("clock_interval_spans_ambiguity_period")
    if delays["minimum_separation_ns"] < thresholds[
        "minimum_delay_separation_ns"
    ]:
        reasons.append("delay_separation_too_small")
    if maxwell["full_minimum_principal_angle_deg"] < thresholds[
        "minimum_maxwell_angle_deg"
    ]:
        reasons.append("maxwell_subspaces_nearly_overlap")
    if not maxwell["dual_all_rank_two"] or not maxwell["full_all_rank_two"]:
        reasons.append("reduced_or_full_jones_rank_deficient")
    if not union["expected_rank_reached"]:
        reasons.append("mksc_union_rank_not_2K")
    if union["sigma_last_over_first"] < thresholds["minimum_union_sigma_ratio"]:
        reasons.append("mksc_union_ill_conditioned")
    if efim is None:
        reasons.append("efim_not_computed")
    else:
        if efim["rank_deficiency_rate"] > 0.0:
            reasons.append("efim_rank_deficient_on_grid")
        if (
            not np.isfinite(efim["condition_number_stats"]["maximum"])
            or efim["condition_number_stats"]["maximum"]
            > thresholds["maximum_efim_condition"]
        ):
            reasons.append("efim_condition_too_large")
        if (
            not np.isfinite(efim["peb_m_stats"]["maximum"])
            or efim["peb_m_stats"]["maximum"] > thresholds["maximum_worst_peb_m"]
        ):
            reasons.append("worst_case_peb_too_large")
    return reasons


def preferred_pass(candidate: dict, thresholds: dict) -> bool:
    efim = candidate["efim"]
    return bool(
        candidate["pass"]
        and candidate["distance"]["ris_bs_min_fraunhofer_ratio"]
        >= thresholds["preferred_ris_bs_fraunhofer_ratio"]
        and candidate["ris_bs_mismatch"]["normalized_residual_max"]
        <= thresholds["preferred_channel_residual"]
        and efim["condition_number_stats"]["p95"]
        <= thresholds["preferred_efim_condition"]
    )


def audit_candidate(
    config: dict,
    bs_position: np.ndarray,
    ue_positions: Sequence[np.ndarray],
    scene_seeds: Sequence[int],
    thresholds: dict,
    *,
    subspace_rtol: float,
    absolute_rank_tolerance: float,
    compute_efim: bool = True,
) -> dict:
    local_config = copy.deepcopy(config)
    local_config["p_B"] = np.asarray(bs_position, dtype=float)
    scene = generate_scene(
        local_config, np.random.default_rng(int(scene_seeds[0]))
    )
    result = {
        "bs_position_m": np.asarray(bs_position, dtype=float),
        "distance": audit_distances(scene, ue_positions),
        "ris_bs_mismatch": audit_ris_bs_mismatch(scene),
        "delay": audit_delays(scene, local_config, ue_positions),
        "maxwell_jones": audit_maxwell_jones(scene, subspace_rtol),
        "mksc_union": audit_mksc_union(
            scene, subspace_rtol, absolute_rank_tolerance
        ),
        "efim": None,
    }
    if compute_efim:
        result["efim"] = audit_efim_grid(
            local_config,
            np.asarray(bs_position, dtype=float),
            ue_positions,
            scene_seeds,
            subspace_rtol,
        )
    result["rejection_reasons"] = rejection_reasons(result, thresholds)
    result["pass"] = not result["rejection_reasons"]
    result["preferred_pass"] = (
        preferred_pass(result, thresholds) if compute_efim else False
    )
    return result


def _unit_from_azimuth_elevation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=float,
    )


def generate_bs_candidates(
    config: dict,
    radial_ratios: Sequence[float],
    azimuth_offsets_deg: Sequence[float],
    elevations_deg: Sequence[float],
    coordinate_snap_m: float,
) -> list[np.ndarray]:
    """Generate a distance--azimuth--elevation grid around the RIS centroid."""
    reference_scene = generate_scene(
        config, np.random.default_rng(int(config["seed"]))
    )
    centroid = np.mean(np.asarray(config["ris_centers"], dtype=float), axis=0)
    old_direction = np.asarray(config["p_B"], dtype=float) - centroid
    base_azimuth = float(np.degrees(np.arctan2(old_direction[1], old_direction[0])))
    ff_distance = fraunhofer_distance(
        reference_scene["ris_grid"], reference_scene["wavelength"]
    )
    candidates = []
    for ratio in radial_ratios:
        for offset in azimuth_offsets_deg:
            for elevation in elevations_deg:
                direction = _unit_from_azimuth_elevation(
                    base_azimuth + float(offset), float(elevation)
                )
                position = centroid + float(ratio) * ff_distance * direction
                if coordinate_snap_m > 0.0:
                    position = (
                        np.round(position / coordinate_snap_m) * coordinate_snap_m
                    )
                candidates.append(position)
    return _deduplicate_positions(candidates)


def _deduplicate_positions(positions: Sequence[np.ndarray]) -> list[np.ndarray]:
    unique = []
    keys = set()
    for position in positions:
        array = np.asarray(position, dtype=float).reshape(3)
        key = tuple(np.round(array, decimals=9))
        if key not in keys:
            unique.append(array)
            keys.add(key)
    return unique


def _quick_subsample(positions: Sequence[np.ndarray], maximum: int) -> list[np.ndarray]:
    if len(positions) <= maximum:
        return list(positions)
    indices = np.linspace(0, len(positions) - 1, maximum)
    rounded = sorted(set(int(round(item)) for item in indices))
    return [np.asarray(positions[index], dtype=float) for index in rounded]


def _cheap_physical_pass(candidate: dict, thresholds: dict) -> bool:
    return not any(
        reason
        for reason in rejection_reasons(
            {**candidate, "efim": {"rank_deficiency_rate": 0.0,
                                   "condition_number_stats": {"maximum": 0.0},
                                   "peb_m_stats": {"maximum": 0.0}}},
            thresholds,
        )
        if reason != "efim_not_computed"
    )


def pareto_front(candidates: Sequence[dict]) -> list[dict]:
    """Return hard-pass candidates not dominated on transparent audit axes."""
    passed = [item for item in candidates if item["pass"]]
    if len(passed) < 2:
        return passed

    def objectives(item: dict) -> np.ndarray:
        return np.array(
            [
                -item["distance"]["ris_bs_min_fraunhofer_ratio"],
                item["ris_bs_mismatch"]["normalized_residual_max"],
                -item["maxwell_jones"]["full_minimum_principal_angle_deg"],
                item["efim"]["condition_number_stats"]["p95"],
                item["efim"]["peb_m_stats"]["p95"],
            ],
            dtype=float,
        )

    front = []
    for index, candidate in enumerate(passed):
        value = objectives(candidate)
        dominated = False
        for other_index, other in enumerate(passed):
            if index == other_index:
                continue
            other_value = objectives(other)
            if np.all(other_value <= value) and np.any(other_value < value):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _coordinate_complexity(position: np.ndarray) -> float:
    array = np.asarray(position, dtype=float)
    return float(np.sum(np.abs(array * 2.0 - np.round(array * 2.0))))


def label_candidates(
    candidates: Sequence[dict],
    old_position: np.ndarray,
) -> None:
    passed = [item for item in candidates if item["pass"]]
    if not passed:
        return
    criteria = {
        "strongest physical far-field margin": lambda item: -item["distance"][
            "ris_bs_min_fraunhofer_ratio"
        ],
        "best angular diversity": lambda item: -item["maxwell_jones"][
            "full_minimum_principal_angle_deg"
        ],
        "best EFIM conditioning": lambda item: item["efim"][
            "condition_number_stats"
        ]["p95"],
        "closest to current layout": lambda item: float(
            np.linalg.norm(item["bs_position_m"] - old_position)
        ),
        "simplest coordinates": lambda item: _coordinate_complexity(
            item["bs_position_m"]
        ),
    }
    for item in candidates:
        item["labels"] = []
    for label, key in criteria.items():
        best = min(passed, key=key)
        best["labels"].append(label)
    medians = {
        "ratio": float(
            np.median(
                [
                    item["distance"]["ris_bs_min_fraunhofer_ratio"]
                    for item in passed
                ]
            )
        ),
        "delay": float(
            np.median(
                [item["delay"]["minimum_separation_ns"] for item in passed]
            )
        ),
        "angle": float(
            np.median(
                [
                    item["maxwell_jones"]["full_minimum_principal_angle_deg"]
                    for item in passed
                ]
            )
        ),
        "condition": float(
            np.median(
                [item["efim"]["condition_number_stats"]["p95"] for item in passed]
            )
        ),
        "peb": float(
            np.median([item["efim"]["peb_m_stats"]["p95"] for item in passed])
        ),
    }
    scales = {
        key: max(abs(value), np.finfo(float).tiny)
        for key, value in medians.items()
    }

    def representative_distance(item: dict) -> float:
        values = {
            "ratio": item["distance"]["ris_bs_min_fraunhofer_ratio"],
            "delay": item["delay"]["minimum_separation_ns"],
            "angle": item["maxwell_jones"]["full_minimum_principal_angle_deg"],
            "condition": item["efim"]["condition_number_stats"]["p95"],
            "peb": item["efim"]["peb_m_stats"]["p95"],
        }
        return float(
            np.linalg.norm(
                [
                    (values[key] - medians[key]) / scales[key]
                    for key in medians
                ]
            )
        )

    representative = min(passed, key=representative_distance)
    representative["labels"].append("most representative")
    for item in passed:
        item["representative_distance_to_pass_medians"] = representative_distance(
            item
        )


def recommendation_order(candidates: Sequence[dict], limit: int = 5) -> list[dict]:
    passed = [item for item in candidates if item["pass"]]
    return sorted(
        passed,
        key=lambda item: (
            not item["preferred_pass"],
            item.get("representative_distance_to_pass_medians", float("inf")),
            _coordinate_complexity(item["bs_position_m"]),
            abs(item["distance"]["ris_bs_min_fraunhofer_ratio"] - 1.5),
        ),
    )[:limit]


def candidate_csv_row(candidate: dict) -> dict:
    distances = candidate["distance"]
    mismatch = candidate["ris_bs_mismatch"]
    delay = candidate["delay"]
    maxwell = candidate["maxwell_jones"]
    union = candidate["mksc_union"]
    efim = candidate["efim"]
    position = candidate["bs_position_m"]
    row = {
        "bs_x_m": position[0],
        "bs_y_m": position[1],
        "bs_z_m": position[2],
        "ris_bs_d0_m": distances["ris_bs_distances_m"][0],
        "ris_bs_d1_m": distances["ris_bs_distances_m"][1],
        "ris_bs_d2_m": distances["ris_bs_distances_m"][2],
        "min_fraunhofer_ratio": distances["ris_bs_min_fraunhofer_ratio"],
        "ue_ris_grid_ratio_max": distances["ue_ris_grid_ratio_max"],
        "phase_residual_max_rad": mismatch["phase_residual_max_rad"],
        "phase_residual_rms_max_rad": mismatch["phase_residual_rms_max_rad"],
        "normalized_channel_residual_max": mismatch["normalized_residual_max"],
        "normalized_channel_correlation_min": mismatch[
            "normalized_correlation_min"
        ],
        "amplitude_aware_residual_max": mismatch[
            "amplitude_aware_normalized_residual_max"
        ],
        "delay_min_ns": delay["delay_min_ns"],
        "delay_max_ns": delay["delay_max_ns"],
        "minimum_delay_separation_ns": delay["minimum_separation_ns"],
        "delay_unambiguous": delay["all_delays_unambiguous_over_clock_bounds"],
        "maxwell_pair01_min_angle_deg": maxwell[
            "full_pair_minimum_angles_deg"
        ][0],
        "maxwell_pair02_min_angle_deg": maxwell[
            "full_pair_minimum_angles_deg"
        ][1],
        "maxwell_pair12_min_angle_deg": maxwell[
            "full_pair_minimum_angles_deg"
        ][2],
        "maxwell_min_angle_deg": maxwell["full_minimum_principal_angle_deg"],
        "mksc_rank_relative": union["relative_rank"],
        "mksc_rank_absolute": union["absolute_rank"],
        "mksc_sigma_last_over_first": union["sigma_last_over_first"],
        "mksc_nonzero_condition": union["nonzero_condition"],
        "mksc_singular_values": json.dumps(
            _json_safe(union["singular_values"]), separators=(",", ":")
        ),
        "efim_min_rank": efim["minimum_rank"],
        "efim_rank_deficiency_rate": efim["rank_deficiency_rate"],
        "efim_condition_median": efim["condition_number_stats"]["median"],
        "efim_condition_p95": efim["condition_number_stats"]["p95"],
        "efim_condition_max": efim["condition_number_stats"]["maximum"],
        "peb_median_m": efim["peb_m_stats"]["median"],
        "peb_p95_m": efim["peb_m_stats"]["p95"],
        "peb_max_m": efim["peb_m_stats"]["maximum"],
        "ceb_median_ns": efim["ceb_ns_stats"]["median"],
        "ceb_p95_ns": efim["ceb_ns_stats"]["p95"],
        "axis_x_crb_p95_m": efim["axis_crb_m_stats"]["x"]["p95"],
        "axis_y_crb_p95_m": efim["axis_crb_m_stats"]["y"]["p95"],
        "axis_z_crb_p95_m": efim["axis_crb_m_stats"]["z"]["p95"],
        "worst_grid_point_m": json.dumps(
            _json_safe(efim["worst_case_position_m"]), separators=(",", ":")
        ),
        "default_ue_peb_percentile": efim["default_ue_peb_percentile"],
        "pass": candidate["pass"],
        "preferred_pass": candidate["preferred_pass"],
        "labels": ";".join(candidate.get("labels", [])),
        "rejection_reasons": ";".join(candidate["rejection_reasons"]),
    }
    return row


def _write_csv(
    path: pathlib.Path,
    candidates: Sequence[dict],
    *,
    template: dict | None = None,
) -> None:
    rows = [candidate_csv_row(item) for item in candidates]
    if rows:
        fieldnames = list(rows[0])
    elif template is not None:
        fieldnames = list(candidate_csv_row(template))
    else:
        raise ValueError("an empty candidate CSV requires a row template")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_markdown(candidate: dict) -> str:
    position = candidate["bs_position_m"]
    labels = ", ".join(candidate.get("labels", [])) or "tradeoff candidate"
    return (
        f"- `[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m`: "
        f"{labels}; min far-field ratio "
        f"{candidate['distance']['ris_bs_min_fraunhofer_ratio']:.3f}, "
        f"phase-only residual "
        f"{candidate['ris_bs_mismatch']['normalized_residual_max']:.4f}, "
        f"min delay separation {candidate['delay']['minimum_separation_ns']:.3f} ns, "
        f"min Maxwell angle "
        f"{candidate['maxwell_jones']['full_minimum_principal_angle_deg']:.2f} deg, "
        f"EFIM p95 cond {candidate['efim']['condition_number_stats']['p95']:.3g}, "
        f"PEB p95 {candidate['efim']['peb_m_stats']['p95'] * 1e3:.3f} mm."
    )


def write_report(
    path: pathlib.Path,
    *,
    old_candidate: dict,
    candidates: Sequence[dict],
    pareto: Sequence[dict],
    recommendations: Sequence[dict],
    args: argparse.Namespace,
    thresholds: dict,
) -> None:
    passed = [item for item in candidates if item["pass"]]
    rejection_counts: dict[str, int] = {}
    for candidate in candidates:
        for reason in candidate["rejection_reasons"]:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    old_distance = old_candidate["distance"]
    lines = [
        "# BS Geometry Audit",
        "",
        "## Current frozen geometry",
        "",
        (
            "The frozen BS position is "
            f"`{_json_safe(old_candidate['bs_position_m'])} m`. Its RIS--BS "
            f"distances are `{_json_safe(old_distance['ris_bs_distances_m'])} m`, "
            "while the 64x64 half-wavelength RIS Fraunhofer distance is "
            f"`{old_distance['ris_fraunhofer_distance_m']:.6f} m`. The minimum "
            f"distance ratio is only `{old_distance['ris_bs_min_fraunhofer_ratio']:.3f}`, "
            "so the repository's matched plane-wave generator/estimator is not a "
            "physical far-field realization of the stated scene."
        ),
        "",
        "## Audit scope and hard conditions",
        "",
        (
            "No estimator, baseline, noisy realization, or Monte Carlo trial was "
            "run. Candidate selection uses only deterministic physical/model, "
            "delay, subspace, and matched free-Jones EFIM diagnostics."
        ),
        "",
        "```json",
        json.dumps(_json_safe(thresholds), indent=2, sort_keys=True),
        "```",
        "",
        "## Search",
        "",
        (
            f"Mode: `{'full' if args.full else 'quick'}`. Candidate radial ratios: "
            f"`{args.radial_ratios}`; azimuth offsets: "
            f"`{args.azimuth_offsets_deg} deg`; elevations: "
            f"`{args.elevations_deg} deg`; coordinate snap: "
            f"`{args.coordinate_snap_m} m`. Audited {len(candidates)} total "
            f"candidates including the old BS, with {len(passed)} hard passes."
        ),
        "",
        "Rejection counts: "
        + (json.dumps(rejection_counts, sort_keys=True) if rejection_counts else "none"),
        "",
        "## Pareto candidates",
        "",
    ]
    lines.extend(_candidate_markdown(item) for item in pareto)
    lines.extend(["", "## Recommended shortlist (not frozen)", ""])
    if recommendations:
        lines.extend(_candidate_markdown(item) for item in recommendations)
    else:
        lines.append(
            "No candidate passed every hard condition. Expand the radial/angle "
            "search before considering a geometry freeze."
        )
    lines.extend(
        [
            "",
            "## Freeze decision",
            "",
            (
                "The quick audit is a screening result only. Run `--full` over the "
                "paper 5x5x2 UE grid before freezing any new BS position."
                if not args.full
                else (
                    "The full deterministic grid audit found viable candidates; "
                    "the final choice remains a manual geometry-freeze decision."
                    if recommendations
                    else "The search should be expanded before any geometry freeze."
                )
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(
    output_dir: pathlib.Path,
    config: dict,
    candidates: Sequence[dict],
    pareto: Sequence[dict],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    positions = np.asarray([item["bs_position_m"] for item in candidates])
    passed = np.asarray([item["pass"] for item in candidates], dtype=bool)
    ris = np.asarray(config["ris_centers"], dtype=float)
    bounds = np.asarray(config["ue_bounds"], dtype=float)
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*ris.T, marker="s", s=80, label="RIS centers")
    axis.scatter(
        positions[~passed, 0],
        positions[~passed, 1],
        positions[~passed, 2],
        c="tab:red",
        marker="x",
        label="fail",
    )
    axis.scatter(
        positions[passed, 0],
        positions[passed, 1],
        positions[passed, 2],
        c="tab:green",
        marker="o",
        label="pass",
    )
    axis.scatter(
        [np.mean(bounds[0])],
        [np.mean(bounds[1])],
        [np.mean(bounds[2])],
        c="tab:blue",
        marker="^",
        label="UE-box center",
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend()
    figure.tight_layout()
    name = "candidate_layout_3d.png"
    figure.savefig(output_dir / name, dpi=180)
    plt.close(figure)
    files.append(name)

    figure, axis = plt.subplots(figsize=(7, 5))
    for item in candidates:
        color = "tab:green" if item["pass"] else "tab:red"
        axis.scatter(
            item["distance"]["ris_bs_min_fraunhofer_ratio"],
            item["ris_bs_mismatch"]["normalized_residual_max"],
            c=color,
        )
    for item in pareto:
        position = item["bs_position_m"]
        axis.annotate(
            f"({position[0]:.1f},{position[1]:.1f},{position[2]:.1f})",
            (
                item["distance"]["ris_bs_min_fraunhofer_ratio"],
                item["ris_bs_mismatch"]["normalized_residual_max"],
            ),
            fontsize=7,
        )
    axis.set_xlabel("minimum RIS-BS Fraunhofer ratio")
    axis.set_ylabel("maximum aligned channel residual")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    name = "fraunhofer_ratio_vs_model_residual.png"
    figure.savefig(output_dir / name, dpi=180)
    plt.close(figure)
    files.append(name)

    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        [
            item["maxwell_jones"]["full_minimum_principal_angle_deg"]
            for item in candidates
        ],
        [item["delay"]["minimum_separation_ns"] for item in candidates],
        c=[
            np.log10(max(item["efim"]["condition_number_stats"]["p95"], 1.0))
            for item in candidates
        ],
        cmap="viridis",
    )
    axis.set_xlabel("minimum Maxwell principal angle [deg]")
    axis.set_ylabel("minimum delay separation [ns]")
    figure.colorbar(scatter, ax=axis, label="log10(EFIM p95 condition)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    name = "principal_angle_vs_delay_separation.png"
    figure.savefig(output_dir / name, dpi=180)
    plt.close(figure)
    files.append(name)
    return files


def _resolve_output_dir(args: argparse.Namespace) -> pathlib.Path:
    if args.out_dir is not None:
        output = args.out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        mode = "full" if args.full else "quick"
        output = DEFAULT_OUTPUT_ROOT / f"{mode}_{stamp}"
    output = output if output.is_absolute() else REPO_ROOT / output
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty result directory: {output}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically audit BS candidate geometry without running "
            "estimators, baselines, or noisy Monte Carlo."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="small screening grid")
    mode.add_argument("--full", action="store_true", help="paper 5x5x2 UE grid")
    parser.add_argument(
        "--preset", choices=("paper", "default"), default="paper"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="optional JSON configuration merged over the selected preset",
    )
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument(
        "--scene-seeds",
        type=_int_list,
        default=[DEFAULT_SCENE_SEED],
        help="fixed representative Omega/Jones/gain scene seeds",
    )
    parser.add_argument(
        "--radial-ratios",
        type=_float_list,
        default=[1.0, 1.25, 1.5, 1.75, 2.0],
        help="RIS-centroid distance in multiples of the RIS Fraunhofer distance",
    )
    parser.add_argument(
        "--azimuth-offsets-deg",
        type=_float_list,
        default=[-25.0, 0.0, 25.0],
        help="offsets from the current centroid-to-BS horizontal direction",
    )
    parser.add_argument(
        "--elevations-deg",
        type=_float_list,
        default=[0.0, 2.0],
        help="absolute global elevation angles",
    )
    parser.add_argument(
        "--coordinate-snap-m",
        type=float,
        default=0.5,
        help="snap generated coordinates to this spacing; zero disables",
    )
    parser.add_argument(
        "--candidate",
        type=_position3,
        action="append",
        default=[],
        help="add an explicit BS candidate x,y,z (repeatable)",
    )
    parser.add_argument(
        "--explicit-only",
        action="store_true",
        help="audit only --candidate positions plus the mandatory old BS",
    )
    parser.add_argument(
        "--quick-max-candidates",
        type=int,
        default=5,
        help="maximum generated candidates in quick mode, excluding old BS",
    )
    parser.add_argument(
        "--ue-grid-shape",
        type=_shape3,
        help="override quick/full UE grid shape",
    )
    parser.add_argument("--ue-grid-margin-m", type=float, default=0.1)
    parser.add_argument("--subspace-rtol", type=float, default=1.0e-12)
    parser.add_argument("--absolute-rank-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--min-farfield-ratio", type=float, default=1.0)
    parser.add_argument("--preferred-farfield-ratio", type=float, default=1.25)
    parser.add_argument("--max-nearfield-ratio", type=float, default=1.0)
    parser.add_argument("--max-phase-residual-rad", type=float, default=0.8)
    parser.add_argument("--max-phase-rms-rad", type=float, default=0.35)
    parser.add_argument("--max-channel-residual", type=float, default=0.15)
    parser.add_argument("--preferred-channel-residual", type=float, default=0.10)
    parser.add_argument("--min-delay-separation-ns", type=float, default=0.05)
    parser.add_argument("--min-maxwell-angle-deg", type=float, default=5.0)
    parser.add_argument("--min-union-sigma-ratio", type=float, default=1.0e-6)
    parser.add_argument("--max-efim-condition", type=float, default=1.0e6)
    parser.add_argument("--preferred-efim-condition", type=float, default=1.0e4)
    parser.add_argument("--max-worst-peb-m", type=float, default=0.1)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="new/empty output directory (default: timestamped geometry_audit dir)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="skip optional diagnostic plots"
    )
    return parser


def run(args: argparse.Namespace) -> pathlib.Path:
    if not args.quick and not args.full:
        args.quick = True
    if args.quick_max_candidates < 1:
        raise ValueError("--quick-max-candidates must be positive")
    config = load_audit_config(args)
    thresholds = default_thresholds(args)
    shape = (
        args.ue_grid_shape
        if args.ue_grid_shape is not None
        else DEFAULT_FULL_GRID_SHAPE
        if args.full
        else DEFAULT_QUICK_GRID_SHAPE
    )
    ue_positions = position_grid(config, shape, args.ue_grid_margin_m)
    generated = []
    if not args.explicit_only:
        generated = generate_bs_candidates(
            config,
            args.radial_ratios,
            args.azimuth_offsets_deg,
            args.elevations_deg,
            args.coordinate_snap_m,
        )
        if args.quick:
            generated = _quick_subsample(generated, args.quick_max_candidates)
    old_position = np.asarray(config["p_B"], dtype=float)
    positions = _deduplicate_positions(
        [old_position, *generated, *args.candidate]
    )
    output_dir = _resolve_output_dir(args)
    print(
        f"Auditing {len(positions)} BS positions over {len(ue_positions)} UE grid "
        f"points and {len(args.scene_seeds)} fixed scene seed(s).",
        flush=True,
    )
    candidates = []
    for index, position in enumerate(positions):
        print(
            f"[{index + 1}/{len(positions)}] BS={position.tolist()}",
            flush=True,
        )
        candidate = audit_candidate(
            config,
            position,
            ue_positions,
            args.scene_seeds,
            thresholds,
            subspace_rtol=args.subspace_rtol,
            absolute_rank_tolerance=args.absolute_rank_tolerance,
        )
        candidate["candidate_index"] = index
        candidate["is_old_bs"] = bool(
            np.allclose(position, old_position, rtol=0.0, atol=1.0e-12)
        )
        candidates.append(candidate)
    label_candidates(candidates, old_position)
    pareto = pareto_front(candidates)
    recommendations = recommendation_order(candidates)
    old_candidate = next(item for item in candidates if item["is_old_bs"])
    _write_csv(output_dir / "all_candidates.csv", candidates)
    _write_csv(
        output_dir / "pareto_candidates.csv",
        pareto,
        template=old_candidate,
    )
    command = " ".join(shlex.quote(item) for item in sys.argv)
    environment = validation_environment(command, repo_root=REPO_ROOT)
    environment["geometry_audit_code_hash"] = canonical_hash(
        {
            "script": pathlib.Path(__file__).read_bytes().hex(),
            "channel_model_commit": git_value(
                ["hash-object", "src/channel_model.py"], repo_root=REPO_ROOT
            ),
            "geometry_commit": git_value(
                ["hash-object", "src/geometry.py"], repo_root=REPO_ROOT
            ),
            "global_vp_commit": git_value(
                ["hash-object", "src/global_vp.py"], repo_root=REPO_ROOT
            ),
        }
    )
    plot_files = (
        []
        if args.no_plots
        else write_plots(output_dir, config, candidates, pareto)
    )
    manifest = {
        "schema": "evs_mimo_ris_geometry_audit_v1",
        "input_config": config,
        "input_config_hash": canonical_hash(config),
        "thresholds": thresholds,
        "candidate_generation": {
            "reference": "RIS centroid",
            "radial_ratios": args.radial_ratios,
            "azimuth_offsets_deg": args.azimuth_offsets_deg,
            "elevations_deg": args.elevations_deg,
            "coordinate_snap_m": args.coordinate_snap_m,
            "explicit_candidates_m": args.candidate,
            "quick_subsample_limit": (
                args.quick_max_candidates if args.quick else None
            ),
        },
        "ue_grid": {
            "shape": shape,
            "margin_m": args.ue_grid_margin_m,
            "positions_m": ue_positions,
            "paper_full_grid": list(DEFAULT_FULL_GRID_SHAPE),
        },
        "selection_policy": {
            "method": "hard constraints then transparent Pareto front",
            "forbidden_metrics": [
                "proposed RMSE",
                "baseline RMSE",
                "ground-truth estimation error",
            ],
            "recommendation_order": (
                "preferred pass, distance to pass-set medians, simple half-metre "
                "coordinates, distance ratio near 1.5"
            ),
        },
        "old_bs_audit": old_candidate,
        "all_candidates": candidates,
        "pareto_candidates": pareto,
        "recommended_candidates": recommendations,
        "plot_files": plot_files,
        "environment": environment,
        "command": command,
    }
    (output_dir / "geometry_audit.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / "geometry_audit.md",
        old_candidate=old_candidate,
        candidates=candidates,
        pareto=pareto,
        recommendations=recommendations,
        args=args,
        thresholds=thresholds,
    )
    print(f"Wrote geometry audit to {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
