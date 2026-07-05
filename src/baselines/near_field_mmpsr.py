"""Near-field spherical-domain sparse recovery baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .backend import BackendConfig, choose_batch_size, get_backend
from .cache import BASELINE_CACHE, baseline_cache_key, cache_diagnostics_delta
from .common import (
    BaselineResult,
    clock_grid_from_config,
    expand_jones_group,
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


def _score_position_clock(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    p_u: np.ndarray,
    delta_t: float,
) -> dict[str, Any]:
    supports = _supports_for_candidate(scene, np.asarray(p_u, dtype=float), float(delta_t))
    Psi = _candidate_design(scene, config, supports)
    score, coeffs, y_hat = score_candidate_block(Psi, y_vec)
    residual = y_vec - y_hat
    return {
        "score": float(score),
        "coeffs": coeffs,
        "y_hat": y_hat,
        "residual": residual,
        "supports": supports,
        "position": np.asarray(p_u, dtype=float),
        "delta_t": float(delta_t),
    }


def _push_top_candidate(
    top: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    limit: int,
) -> None:
    top.append(candidate)
    top.sort(key=lambda item: float(item["score"]), reverse=True)
    del top[max(1, int(limit)) :]


def _axis_spacing_from_grid(values: list[np.ndarray], axis: int, fallback: float) -> float:
    coords = sorted({round(float(value[axis]), 12) for value in values})
    if len(coords) < 2:
        return fallback
    diffs = np.diff(np.asarray(coords, dtype=float))
    diffs = diffs[np.abs(diffs) > 0.0]
    return float(np.min(np.abs(diffs))) if diffs.size else fallback


def _refine_candidate_local(
    scene: dict,
    config: dict,
    y_vec: np.ndarray,
    candidate: dict[str, Any],
    *,
    positions: list[np.ndarray],
    clocks: np.ndarray,
    levels: int,
    shrink: float,
    local_position_grid_shape: tuple[int, int, int],
    local_clock_grid_size: int,
) -> tuple[dict[str, Any], int]:
    bounds_p = np.asarray(
        config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]),
        dtype=float,
    )
    bounds_dt = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    span = bounds_p[:, 1] - bounds_p[:, 0]
    hp = np.array(
        [
            _axis_spacing_from_grid(positions, axis, float(span[axis] / 4.0))
            for axis in range(3)
        ],
        dtype=float,
    )
    if len(clocks) > 1:
        clock_diffs = np.diff(np.asarray(clocks, dtype=float))
        clock_diffs = clock_diffs[np.abs(clock_diffs) > 0.0]
        ht = float(np.min(np.abs(clock_diffs))) if clock_diffs.size else 1.0e-9
    else:
        ht = 1.0e-9

    best = dict(candidate)
    evals = 0
    shape = tuple(max(1, int(v)) for v in local_position_grid_shape)
    clock_size = max(1, int(local_clock_grid_size))
    for level in range(max(0, int(levels))):
        scale = float(shrink) ** level
        axes = [
            np.linspace(-hp[axis] * scale, hp[axis] * scale, shape[axis])
            for axis in range(3)
        ]
        clock_offsets = np.linspace(-ht * scale, ht * scale, clock_size)
        center_p = np.asarray(best["position"], dtype=float)
        center_dt = float(best["delta_t"])
        for dx in axes[0]:
            for dy in axes[1]:
                for dz in axes[2]:
                    p = np.clip(center_p + np.array([dx, dy, dz]), bounds_p[:, 0], bounds_p[:, 1])
                    for dt_offset in clock_offsets:
                        dt = float(np.clip(center_dt + dt_offset, bounds_dt[0], bounds_dt[1]))
                        trial = _score_position_clock(scene, config, y_vec, p, dt)
                        trial["selected_grid_index"] = candidate.get("selected_grid_index", [-1, -1])
                        evals += 1
                        if float(trial["score"]) > float(best["score"]):
                            best = trial
    return best, evals


def run_near_field_mmpsr_baseline(data: dict, config: dict) -> BaselineResult:
    start = time.perf_counter()
    scene = data["scene"]
    y_vec = vectorize_raw_observation(data["Y_noisy"])
    cfg = dict(config.get("baselines", {}).get("nf_mmpsr", {}))
    backend_cfg = BackendConfig.from_value(
        config.get("baselines", {}).get("backend_config")
    )
    backend = get_backend(backend_cfg)
    BASELINE_CACHE.configure(
        enabled=backend_cfg.cache_enabled,
        memory_budget_gb=backend_cfg.cache_memory_budget_gb,
    )
    cache_before = BASELINE_CACHE.snapshot()
    grid_shape = tuple(int(v) for v in cfg.get("grid_shape", (11, 11, 5)))
    clock_grid_size = int(cfg.get("clock_grid_size", 11))
    top_candidate_count = max(1, int(cfg.get("top_candidates", 8)))
    cache_key = baseline_cache_key(
        "nf_mmpsr",
        scene,
        config,
        grid_sizes=(grid_shape, clock_grid_size),
    )
    positions, clocks = BASELINE_CACHE.get_or_create(
        cache_key,
        lambda: (
            position_grid_from_config(config, grid_shape),
            clock_grid_from_config(config, clock_grid_size),
        ),
    )
    candidates = [
        (int(pos_idx), int(clock_idx), position, float(delta_t))
        for pos_idx, position in enumerate(positions)
        for clock_idx, delta_t in enumerate(clocks)
    ]
    max_batch_memory_mb = float(cfg.get("max_batch_memory_mb", 256.0))
    requested_batch_size = int(cfg.get("batch_size", 64))
    columns_per_candidate = max(1, 2 * int(scene["K"]))
    memory_budget_bytes = max_batch_memory_mb * 1024.0**2
    if backend.name == "cupy" and backend_cfg.gpu_memory_fraction is not None:
        memory_budget_bytes = min(
            memory_budget_bytes,
            float(backend.memory_info()["free_bytes"]) * backend_cfg.gpu_memory_fraction,
        )
    memory_batch_size = choose_batch_size(
        len(candidates),
        y_vec.size * columns_per_candidate,
        memory_budget_bytes,
        np.complex128,
    )
    requested_batch_size = (
        backend_cfg.gpu_batch_size
        if backend.name == "cupy" and backend_cfg.gpu_batch_size is not None
        else backend_cfg.cpu_batch_size
        if backend.name == "cpu" and backend_cfg.cpu_batch_size is not None
        else requested_batch_size
    )
    batch_size = max(1, min(int(requested_batch_size), memory_batch_size))
    num_batches = 0
    top_candidates: list[dict[str, Any]] = []
    scoring_start = time.perf_counter()
    y_device = (
        backend.asarray(y_vec, dtype=backend.xp.complex128)
        if backend.name == "cupy"
        else None
    )
    y_energy = float(np.linalg.norm(y_vec) ** 2 + 1.0e-12)
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
        if backend.name == "cupy":
            designs_device = backend.asarray(
                np.stack([entry[-1] for entry in batch_designs], axis=0),
                dtype=backend.xp.complex128,
            )
            gram = backend.xp.matmul(
                designs_device.conj().transpose(0, 2, 1), designs_device
            )
            rhs = backend.xp.matmul(
                designs_device.conj().transpose(0, 2, 1),
                backend.xp.broadcast_to(y_device, (len(batch_designs), y_vec.size))[
                    :, :, None
                ],
            )[:, :, 0]
            eye = backend.xp.eye(columns_per_candidate, dtype=backend.xp.complex128)
            try:
                coeffs_device = backend.solve(
                    gram + 1.0e-10 * eye[None, :, :], rhs
                )
            except Exception:
                coeffs_device = backend.xp.matmul(
                    backend.xp.linalg.pinv(gram + 1.0e-10 * eye[None, :, :]),
                    rhs[:, :, None],
                )[:, :, 0]
            fitted_device = backend.xp.matmul(
                designs_device, coeffs_device[:, :, None]
            )[:, :, 0]
            scores_device = (
                backend.xp.sum(backend.xp.abs(fitted_device) ** 2, axis=1)
                / y_energy
            )
            local_best = int(backend.to_host(backend.argmax(scores_device)))
            scores_host = np.asarray(backend.to_host(scores_device), dtype=float)
            local_order = np.argsort(scores_host)[::-1][:top_candidate_count]
            _ = local_best
            for local_idx in local_order:
                pos_idx, clock_idx, position, delta_t, supports, Psi = batch_designs[int(local_idx)]
                score, coeffs, y_hat = score_candidate_block(Psi, y_vec)
                _push_top_candidate(
                    top_candidates,
                    {
                        "score": float(score),
                        "coeffs": coeffs,
                        "y_hat": y_hat,
                        "residual": y_vec - y_hat,
                        "selected_grid_index": [int(pos_idx), int(clock_idx)],
                        "position": np.asarray(position, dtype=float),
                        "delta_t": float(delta_t),
                        "supports": supports,
                    },
                    limit=top_candidate_count,
                )
            del designs_device, gram, rhs, coeffs_device, fitted_device, scores_device
        else:
            for pos_idx, clock_idx, position, delta_t, supports, Psi in batch_designs:
                score, coeffs, y_hat = score_candidate_block(Psi, y_vec)
                _push_top_candidate(
                    top_candidates,
                    {
                        "score": float(score),
                        "coeffs": coeffs,
                        "y_hat": y_hat,
                        "residual": y_vec - y_hat,
                        "selected_grid_index": [int(pos_idx), int(clock_idx)],
                        "position": np.asarray(position, dtype=float),
                        "delta_t": float(delta_t),
                        "supports": supports,
                    },
                    limit=top_candidate_count,
                )
        del batch_designs, batch_candidates
        if bool(config.get("baselines", {}).get("trim_memory", True)):
            trim_memory()
    backend.synchronize()
    if not top_candidates:
        fallback = _score_position_clock(scene, config, y_vec, positions[0], float(clocks[0]))
        fallback["selected_grid_index"] = [0, 0]
        top_candidates = [fallback]
    coarse_best = top_candidates[0]
    coarse_position = np.asarray(coarse_best["position"], dtype=float).copy()
    coarse_delta_t = float(coarse_best["delta_t"])
    coarse_best_score = float(coarse_best["score"])
    coarse_residual_norm = float(np.linalg.norm(coarse_best["residual"]))
    local_refinement = bool(cfg.get("local_refinement", cfg.get("offgrid_refinement", True)))
    refinement_levels = int(cfg.get("refinement_levels", 3))
    refinement_evals = 0
    best = coarse_best
    if local_refinement:
        refined_candidates = []
        for candidate in top_candidates:
            refined, evals = _refine_candidate_local(
                scene,
                config,
                y_vec,
                candidate,
                positions=positions,
                clocks=clocks,
                levels=refinement_levels,
                shrink=float(cfg.get("refinement_shrink", 0.5)),
                local_position_grid_shape=tuple(
                    int(v) for v in cfg.get("local_position_grid_shape", (3, 3, 3))
                ),
                local_clock_grid_size=int(cfg.get("local_clock_grid_size", 5)),
            )
            refined_candidates.append(refined)
            refinement_evals += evals
        refined_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        best = refined_candidates[0]
        refine_diag = {
            "offgrid_refinement": True,
            "refinement_objective": "cc_projection_local_grid",
            "refinement_solver": "deterministic_local_grid",
            "refinement_success": True,
            "refinement_initial_residual_norm": coarse_residual_norm,
            "refinement_residual_norm": float(np.linalg.norm(best["residual"])),
            "refinement_cost": float(np.linalg.norm(best["residual"]) ** 2),
            "refinement_initial_position": coarse_position.copy(),
            "refinement_initial_clock": coarse_delta_t,
            "refined_position": np.asarray(best["position"], dtype=float).copy(),
            "refined_clock": float(best["delta_t"]),
            "warning": "",
        }
    else:
        refine_diag = {
            "offgrid_refinement": False,
            "refinement_objective": "",
            "refined_position": np.asarray(best["position"], dtype=float).copy(),
            "refined_clock": float(best["delta_t"]),
        }
    best_position = np.asarray(best["position"], dtype=float)
    best_delta_t = float(best["delta_t"])
    best_coeffs = np.asarray(best["coeffs"], dtype=complex)
    best_y_hat = np.asarray(best["y_hat"], dtype=complex)
    best_supports = list(best["supports"])
    residual = y_vec - best_y_hat
    best_index = tuple(int(v) for v in best.get("selected_grid_index", [-1, -1]))
    diagnostics = {
        "dictionary_mode": "near_field_spherical_grid_mmpsr_refined",
        "model_variant": "near_field_mmpsr",
        "group_omp": False,
        "offgrid_refinement": bool(refine_diag.get("offgrid_refinement", False)),
        "refinement_objective": refine_diag.get("refinement_objective", ""),
        "grid_shape": list(grid_shape),
        "clock_grid_size": clock_grid_size,
        "grid_size": len(positions) * len(clocks),
        "best_score": float(best["score"]),
        "coarse_score": coarse_best_score,
        "coarse_grid_position": coarse_position.copy(),
        "coarse_clock": coarse_delta_t,
        "refined_position": np.asarray(best_position, dtype=float).copy(),
        "refined_clock": float(best_delta_t),
        "coarse_residual_norm": coarse_residual_norm,
        "refined_residual_norm": float(np.linalg.norm(residual)),
        "selected_grid_index": list(best_index),
        "selected_delta_t": float(best_delta_t),
        "coeff_norm": float(np.linalg.norm(best_coeffs)),
        "expanded_supports": [
            support for group in best_supports for support in expand_jones_group(group, 2)
        ],
        "batch_size": batch_size,
        "max_batch_memory_mb": max_batch_memory_mb,
        "num_batches": num_batches,
        "backend": backend.name,
        "gpu_used": backend.name == "cupy",
        "gpu_num_batches": num_batches if backend.name == "cupy" else 0,
        "gpu_batch_size": batch_size if backend.name == "cupy" else "",
        "gpu_device": backend.device if backend.name == "cupy" else "",
        "scoring_time_s": time.perf_counter() - scoring_start,
        "backend_warning": backend.warning,
        "nf_mmpsr_cc_metric": "normalized_projection",
        "nf_mmpsr_top_candidates": top_candidate_count,
        "nf_mmpsr_local_refinement_used": bool(local_refinement),
        "nf_mmpsr_refinement_levels": refinement_levels if local_refinement else 0,
        "nf_mmpsr_refinement_num_evals": refinement_evals,
        "nf_mmpsr_coarse_best_score": coarse_best_score,
        "nf_mmpsr_refined_best_score": float(best["score"]),
        "nf_mmpsr_coarse_position": coarse_position.copy(),
        "nf_mmpsr_refined_position": np.asarray(best_position, dtype=float).copy(),
        "nf_mmpsr_coarse_delta_t": coarse_delta_t,
        "nf_mmpsr_refined_delta_t": float(best_delta_t),
    }
    diagnostics.update(refine_diag)
    diagnostics.update(cache_diagnostics_delta(cache_before, BASELINE_CACHE.snapshot()))
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
