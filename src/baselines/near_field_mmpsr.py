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
    fit_position_clock_data_domain,
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
    best_score = -np.inf
    best_selection_score = -np.inf
    best_index = (-1, -1)
    best_position = positions[0]
    best_delta_t = float(clocks[0])
    best_coeffs = np.zeros(0, dtype=complex)
    best_y_hat = np.zeros_like(y_vec)
    best_supports: list[dict[str, Any]] = []
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
            local_score = float(backend.to_host(scores_device[local_best]))
            if local_score > best_selection_score:
                pos_idx, clock_idx, position, delta_t, supports, Psi = batch_designs[
                    local_best
                ]
                score, coeffs, y_hat = score_candidate_block(Psi, y_vec)
                best_selection_score = local_score
                best_score = float(score)
                best_index = (int(pos_idx), int(clock_idx))
                best_position = np.asarray(position, dtype=float)
                best_delta_t = float(delta_t)
                best_coeffs = coeffs
                best_y_hat = y_hat
                best_supports = supports
            del designs_device, gram, rhs, coeffs_device, fitted_device, scores_device
        else:
            for pos_idx, clock_idx, position, delta_t, supports, Psi in batch_designs:
                score, coeffs, y_hat = score_candidate_block(Psi, y_vec)
                if score > best_score:
                    best_selection_score = float(score)
                    best_score = float(score)
                    best_index = (int(pos_idx), int(clock_idx))
                    best_position = np.asarray(position, dtype=float)
                    best_delta_t = float(delta_t)
                    best_coeffs = coeffs
                    best_y_hat = y_hat
                    best_supports = supports
        del batch_designs, batch_candidates
        if bool(config.get("baselines", {}).get("trim_memory", True)):
            trim_memory()
    backend.synchronize()
    residual = y_vec - best_y_hat
    coarse_position = best_position.copy()
    coarse_delta_t = float(best_delta_t)
    coarse_residual_norm = float(np.linalg.norm(residual))
    if bool(cfg.get("offgrid_refinement", True)):
        (
            refined_position,
            refined_delta_t,
            refined_coeffs,
            refined_y_hat,
            refined_groups,
            refine_diag,
        ) = fit_position_clock_data_domain(
            scene,
            config,
            y_vec,
            best_position,
            best_delta_t,
            panels=list(range(int(scene["K"]))),
            model_variant="near_field",
            enabled=True,
            max_nfev=int(cfg.get("refinement_max_nfev", 60)),
        )
        best_position = refined_position
        best_delta_t = refined_delta_t
        best_coeffs = refined_coeffs
        best_y_hat = refined_y_hat
        best_supports = refined_groups
        residual = y_vec - best_y_hat
    else:
        refine_diag = {
            "offgrid_refinement": False,
            "refinement_objective": "",
            "refined_position": best_position.copy(),
            "refined_clock": float(best_delta_t),
        }
    diagnostics = {
        "dictionary_mode": "near_field_spherical_grid_mmpsr",
        "model_variant": "near_field_mmpsr",
        "group_omp": False,
        "offgrid_refinement": bool(refine_diag.get("offgrid_refinement", False)),
        "refinement_objective": refine_diag.get("refinement_objective", ""),
        "grid_shape": list(grid_shape),
        "clock_grid_size": clock_grid_size,
        "grid_size": len(positions) * len(clocks),
        "best_score": float(best_score),
        "coarse_score": float(best_score),
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
