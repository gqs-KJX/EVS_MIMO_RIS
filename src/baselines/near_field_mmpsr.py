"""Near-field spherical-domain sparse recovery baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .common import (
    BaselineResult,
    clock_grid_from_config,
    linear_ls_fit,
    position_grid_from_config,
    raw_atom_from_support,
    simple_atom_normalize,
    vectorize_raw_observation,
)
from ..geometry import local_geometry_from_position
from ..experiments.resource_control import trim_memory


def score_candidate_block(Psi: np.ndarray, y: np.ndarray, eps: float = 1.0e-12) -> tuple[float, np.ndarray, np.ndarray]:
    """Return CC projection score, coefficients, and fitted vector."""
    coeffs, y_hat, _ = linear_ls_fit(Psi, y, ridge=1.0e-10)
    score = float(np.linalg.norm(y_hat) ** 2 / (np.linalg.norm(y) ** 2 + float(eps)))
    return score, coeffs, y_hat


def _supports_for_candidate(scene: dict, p_u: np.ndarray, delta_t: float) -> list[dict[str, Any]]:
    supports: list[dict[str, Any]] = []
    for panel in range(int(scene["K"])):
        range_m, elev, az, _ = local_geometry_from_position(
            np.asarray(p_u, dtype=float),
            np.asarray(scene["ris_centers"][panel], dtype=float),
            np.asarray(scene["rotations"][panel], dtype=float),
        )
        tau = (range_m + scene["d_RB"][panel]) / scene["c0"] + float(delta_t)
        for pol_index in range(2):
            supports.append(
                {
                    "panel": int(panel),
                    "position": np.asarray(p_u, dtype=float),
                    "tau": float(tau),
                    "pol_index": int(pol_index),
                    "range": float(range_m),
                    "elevation": float(elev),
                    "azimuth": float(az),
                    "near_field": True,
                }
            )
    return supports


def _candidate_design(scene: dict, config: dict, supports: list[dict[str, Any]]) -> np.ndarray:
    return np.column_stack(
        [simple_atom_normalize(raw_atom_from_support(scene, config, support)) for support in supports]
    )


def _components_for_position(scene: dict, p_u: np.ndarray, delta_t: float) -> dict[str, np.ndarray]:
    ranges = []
    taus = []
    for panel in range(int(scene["K"])):
        range_m, _, _, _ = local_geometry_from_position(
            np.asarray(p_u, dtype=float),
            np.asarray(scene["ris_centers"][panel], dtype=float),
            np.asarray(scene["rotations"][panel], dtype=float),
        )
        ranges.append(float(range_m))
        taus.append(float((range_m + scene["d_RB"][panel]) / scene["c0"] + delta_t))
    return {"ranges": np.asarray(ranges, dtype=float), "taus": np.asarray(taus, dtype=float)}


def run_near_field_mmpsr_baseline(data: dict, config: dict) -> BaselineResult:
    start = time.perf_counter()
    scene = data["scene"]
    y_vec = vectorize_raw_observation(data["Y_noisy"])
    cfg = dict(config.get("baselines", {}).get("nf_mmpsr", {}))
    grid_shape = tuple(int(v) for v in cfg.get("grid_shape", (11, 11, 5)))
    clock_grid_size = int(cfg.get("clock_grid_size", 11))
    positions = position_grid_from_config(config, grid_shape)
    clocks = clock_grid_from_config(config, clock_grid_size)
    candidates = [
        (int(pos_idx), int(clock_idx), position, float(delta_t))
        for pos_idx, position in enumerate(positions)
        for clock_idx, delta_t in enumerate(clocks)
    ]
    max_batch_memory_mb = float(cfg.get("max_batch_memory_mb", 256.0))
    requested_batch_size = int(cfg.get("batch_size", 64))
    columns_per_candidate = max(1, 2 * int(scene["K"]))
    bytes_per_candidate = max(
        int(y_vec.size * columns_per_candidate * np.dtype(complex).itemsize),
        1,
    )
    memory_batch_size = max(
        1,
        int(max_batch_memory_mb * 1024.0**2 // bytes_per_candidate),
    )
    batch_size = max(1, min(requested_batch_size, memory_batch_size))
    num_batches = 0
    best_score = -np.inf
    best_index = (-1, -1)
    best_position = positions[0]
    best_delta_t = float(clocks[0])
    best_coeffs = np.zeros(0, dtype=complex)
    best_y_hat = np.zeros_like(y_vec)
    best_supports: list[dict[str, Any]] = []
    for batch_start in range(0, len(candidates), batch_size):
        batch_candidates = candidates[batch_start : batch_start + batch_size]
        batch_designs = []
        for pos_idx, clock_idx, position, delta_t in batch_candidates:
            supports = _supports_for_candidate(scene, position, delta_t)
            batch_designs.append(
                (
                    pos_idx,
                    clock_idx,
                    position,
                    delta_t,
                    supports,
                    _candidate_design(scene, config, supports),
                )
            )
        num_batches += 1
        for pos_idx, clock_idx, position, delta_t, supports, Psi in batch_designs:
            score, coeffs, y_hat = score_candidate_block(Psi, y_vec)
            if score > best_score:
                best_score = float(score)
                best_index = (int(pos_idx), int(clock_idx))
                best_position = np.asarray(position, dtype=float)
                best_delta_t = float(delta_t)
                best_coeffs = coeffs
                best_y_hat = y_hat
                best_supports = supports
        del batch_designs, batch_candidates
        trim_memory()
    residual = y_vec - best_y_hat
    diagnostics = {
        "dictionary_mode": "near_field_spherical_grid_mmpsr",
        "grid_shape": list(grid_shape),
        "clock_grid_size": clock_grid_size,
        "grid_size": len(positions) * len(clocks),
        "best_score": float(best_score),
        "selected_grid_index": list(best_index),
        "selected_delta_t": float(best_delta_t),
        "coeff_norm": float(np.linalg.norm(best_coeffs)),
        "batch_size": batch_size,
        "max_batch_memory_mb": max_batch_memory_mb,
        "num_batches": num_batches,
    }
    return BaselineResult(
        name="nf_mmpsr",
        p_u=best_position,
        delta_t=best_delta_t,
        Y_hat=best_y_hat.reshape(scene["I"], scene["N"], scene["T"]),
        raw_objective_final=float(np.linalg.norm(residual) ** 2 / y_vec.size),
        components=_components_for_position(scene, best_position, best_delta_t),
        selected_support=best_supports,
        runtime_s=time.perf_counter() - start,
        diagnostics=diagnostics,
    )
