"""Far-field angular-delay OMP baseline."""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from .common import (
    BaselineResult,
    delay_grid_from_scene,
    direction_grid,
    geometric_support_to_position_ls,
    raw_atom_from_support,
    reconstruct_from_supports,
    simple_atom_normalize,
    vectorize_raw_observation,
)
from ..experiments.resource_control import trim_memory


def omp_select_from_dictionary(Phi: np.ndarray, y: np.ndarray, max_atoms: int) -> list[int]:
    """Tiny dense OMP helper used by unit tests."""
    Phi = np.asarray(Phi, dtype=complex)
    y = np.asarray(y, dtype=complex).reshape(-1)
    atoms = np.column_stack([simple_atom_normalize(Phi[:, k]) for k in range(Phi.shape[1])])
    residual = y.copy()
    selected: list[int] = []
    for _ in range(int(max_atoms)):
        scores = np.abs(atoms.conj().T @ residual)
        for idx in selected:
            scores[idx] = -np.inf
        best = int(np.argmax(scores))
        selected.append(best)
        coeffs, *_ = np.linalg.lstsq(atoms[:, selected], y, rcond=None)
        residual = y - atoms[:, selected] @ coeffs
    return selected


def _support_key(support: dict[str, Any]) -> tuple[Any, ...]:
    direction = tuple(np.round(np.asarray(support.get("direction", [0, 0, 0]), dtype=float), 8))
    return (
        int(support.get("panel", 0)),
        direction,
        round(float(support.get("tau", 0.0)), 15),
        int(support.get("pol_index", 0)),
    )


def _far_field_supports(scene: dict, config: dict) -> Iterable[dict[str, Any]]:
    cfg = dict(config.get("baselines", {}).get("ff_omp", {}))
    directions = direction_grid(int(cfg.get("angle_grid_size", 31)))
    taus = delay_grid_from_scene(scene, config, int(cfg.get("delay_grid_size", 41)))
    use_jones = bool(cfg.get("use_jones_basis", True))
    pol_indices = range(2 if use_jones else 1)
    for panel in range(int(scene["K"])):
        for direction_idx, direction in enumerate(directions):
            for tau_idx, tau in enumerate(taus):
                for pol_index in pol_indices:
                    yield {
                        "panel": int(panel),
                        "direction": direction,
                        "tau": float(tau),
                        "pol_index": int(pol_index),
                        "direction_index": int(direction_idx),
                        "tau_index": int(tau_idx),
                        "near_field": False,
                    }


def _omp_over_supports(
    scene: dict,
    config: dict,
    supports_iter: Iterable[dict[str, Any]],
    y_vec: np.ndarray,
    *,
    max_atoms: int,
    batch_size: int,
    max_batch_memory_mb: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    residual = y_vec.copy()
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()
    y_hat = np.zeros_like(y_vec)
    grid_size = 0
    best_score_last = float("nan")
    all_supports = list(supports_iter)
    bytes_per_atom = max(int(y_vec.size * np.dtype(complex).itemsize), 1)
    memory_batch_size = max(
        1,
        int(float(max_batch_memory_mb) * 1024.0**2 // bytes_per_atom),
    )
    effective_batch_size = max(1, min(int(batch_size), memory_batch_size))
    num_batches = 0
    for _ in range(int(max_atoms)):
        best_score = -np.inf
        best_support = None
        for batch_start in range(0, len(all_supports), effective_batch_size):
            batch_supports = all_supports[
                batch_start : batch_start + effective_batch_size
            ]
            atoms = np.column_stack(
                [
                    simple_atom_normalize(
                        raw_atom_from_support(scene, config, support)
                    )
                    for support in batch_supports
                ]
            )
            scores = np.abs(atoms.conj().T @ residual)
            grid_size += len(batch_supports)
            num_batches += 1
            for local_idx, support in enumerate(batch_supports):
                if _support_key(support) in selected_keys:
                    continue
                score = float(scores[local_idx])
                if score > best_score:
                    best_score = score
                    best_support = support
            del atoms, scores, batch_supports
            trim_memory()
        if best_support is None:
            break
        selected.append(dict(best_support))
        selected_keys.add(_support_key(best_support))
        _, y_hat_tensor, residual_vec, _ = reconstruct_from_supports(
            scene,
            config,
            selected,
            y_vec,
        )
        y_hat = y_hat_tensor.reshape(-1)
        residual = residual_vec
        best_score_last = best_score
    diagnostics = {
        "grid_size": len(all_supports),
        "scored_atoms": grid_size,
        "last_best_score": best_score_last,
        "residual_norm": float(np.linalg.norm(residual)),
        "batch_size": effective_batch_size,
        "max_batch_memory_mb": float(max_batch_memory_mb),
        "num_batches": num_batches,
    }
    return selected, y_hat.reshape(scene["I"], scene["N"], scene["T"]), residual, diagnostics


def _components_from_supports(scene: dict, supports: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    ranges = []
    taus = []
    for support in supports:
        if "direction" in support:
            panel = int(support.get("panel", 0))
            ranges.append(float("nan"))
        if "tau" in support:
            taus.append(float(support["tau"]))
    return {"ranges": np.asarray(ranges, dtype=float), "taus": np.asarray(taus, dtype=float)}


def run_far_field_omp_baseline(data: dict, config: dict) -> BaselineResult:
    start = time.perf_counter()
    scene = data["scene"]
    y_vec = vectorize_raw_observation(data["Y_noisy"])
    cfg = dict(config.get("baselines", {}).get("ff_omp", {}))
    selected, y_hat, residual, diagnostics = _omp_over_supports(
        scene,
        config,
        _far_field_supports(scene, config),
        y_vec,
        max_atoms=int(cfg.get("max_atoms", scene["K"])),
        batch_size=int(cfg.get("batch_size", 256)),
        max_batch_memory_mb=float(cfg.get("max_batch_memory_mb", 256.0)),
    )
    p_hat, delta_t, geom_diag = geometric_support_to_position_ls(scene, selected, config)
    diagnostics.update(geom_diag)
    diagnostics["dictionary_mode"] = "far_field_angular_delay_omp"
    diagnostics["angle_grid_type"] = "direction_cosine_cos_uniform"
    diagnostics["angle_grid_size"] = int(cfg.get("angle_grid_size", 31))
    diagnostics["delay_grid_size"] = int(cfg.get("delay_grid_size", 41))
    diagnostics["selected_support"] = selected
    raw_objective = float(np.linalg.norm(residual) ** 2 / y_vec.size)
    return BaselineResult(
        name="ff_omp",
        p_u=p_hat,
        delta_t=delta_t,
        Y_hat=y_hat,
        raw_objective_final=raw_objective,
        components=_components_from_supports(scene, selected),
        selected_support=selected,
        runtime_s=time.perf_counter() - start,
        diagnostics=diagnostics,
    )
