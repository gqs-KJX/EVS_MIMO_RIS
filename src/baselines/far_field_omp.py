"""Far-field angular-delay OMP baseline."""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from .backend import BackendConfig, choose_batch_size, get_backend
from .cache import BASELINE_CACHE, baseline_cache_key, cache_diagnostics_delta
from .common import (
    BaselineResult,
    delay_grid_from_scene,
    direction_grid,
    expand_jones_group,
    fit_position_clock_data_domain,
    geometric_support_to_position_ls,
    group_omp_select,
    raw_atom_from_support,
    reconstruct_from_supports,
    simple_atom_normalize,
    vectorize_raw_observation,
)


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
    )


def _far_field_supports(scene: dict, config: dict) -> Iterable[dict[str, Any]]:
    cfg = dict(config.get("baselines", {}).get("ff_omp", {}))
    directions = direction_grid(int(cfg.get("angle_grid_size", 31)))
    taus = delay_grid_from_scene(scene, config, int(cfg.get("delay_grid_size", 41)))
    for panel in range(int(scene["K"])):
        for direction_idx, direction in enumerate(directions):
            for tau_idx, tau in enumerate(taus):
                yield {
                    "panel": int(panel),
                    "direction": direction,
                    "tau": float(tau),
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
    backend_config: BackendConfig | dict[str, Any] | None = None,
    trim_memory_enabled: bool = True,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    _ = trim_memory_enabled  # Memory trimming is intentionally method-scoped.
    backend_cfg = BackendConfig.from_value(backend_config)
    backend = get_backend(backend_cfg)
    residual = y_vec.copy()
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()
    y_hat = np.zeros_like(y_vec)
    grid_size = 0
    best_score_last = float("nan")
    all_supports = list(supports_iter)
    memory_budget_bytes = float(max_batch_memory_mb) * 1024.0**2
    if backend.name == "cupy" and backend_cfg.gpu_memory_fraction is not None:
        memory_budget_bytes = min(
            memory_budget_bytes,
            float(backend.memory_info()["free_bytes"]) * backend_cfg.gpu_memory_fraction,
        )
    memory_batch_size = choose_batch_size(
        len(all_supports), y_vec.size, memory_budget_bytes, np.complex128
    )
    requested_batch_size = (
        backend_cfg.gpu_batch_size
        if backend.name == "cupy" and backend_cfg.gpu_batch_size is not None
        else backend_cfg.cpu_batch_size
        if backend.name == "cpu" and backend_cfg.cpu_batch_size is not None
        else batch_size
    )
    effective_batch_size = max(1, min(int(requested_batch_size), memory_batch_size))
    num_batches = 0
    scoring_start = time.perf_counter()
    for _ in range(int(max_atoms)):
        best_score = -np.inf
        best_support = None
        residual_device = (
            backend.asarray(residual, dtype=backend.xp.complex128)
            if backend.name == "cupy"
            else None
        )
        for batch_start in range(0, len(all_supports), effective_batch_size):
            batch_supports = all_supports[
                batch_start : batch_start + effective_batch_size
            ]
            if backend.name == "cupy":
                atoms_cpu = np.column_stack(
                    [raw_atom_from_support(scene, config, support) for support in batch_supports]
                )
                atoms_device = backend.asarray(atoms_cpu, dtype=backend.xp.complex128)
                norms = backend.xp.linalg.norm(atoms_device, axis=0)
                atoms_device = atoms_device / backend.xp.where(norms > 0.0, norms, 1.0)
                scores_device = backend.abs(atoms_device.conj().T @ residual_device)
                for local_idx, support in enumerate(batch_supports):
                    if _support_key(support) in selected_keys:
                        scores_device[local_idx] = -backend.xp.inf
                local_best = int(backend.to_host(backend.argmax(scores_device)))
                local_score = float(backend.to_host(scores_device[local_best]))
                scores = None
            else:
                atoms_cpu = np.column_stack(
                    [
                        simple_atom_normalize(
                            raw_atom_from_support(scene, config, support)
                        )
                        for support in batch_supports
                    ]
                )
                scores = np.abs(atoms_cpu.conj().T @ residual)
            grid_size += len(batch_supports)
            num_batches += 1
            if backend.name == "cupy":
                if local_score > best_score:
                    best_score = local_score
                    best_support = batch_supports[local_best]
            else:
                for local_idx, support in enumerate(batch_supports):
                    if _support_key(support) in selected_keys:
                        continue
                    score = float(scores[local_idx])
                    if score > best_score:
                        best_score = score
                        best_support = support
            if backend.name == "cupy":
                del atoms_device, scores_device
            del atoms_cpu, scores, batch_supports
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
    backend.synchronize()
    diagnostics = {
        "grid_size": len(all_supports),
        "scored_atoms": grid_size,
        "last_best_score": best_score_last,
        "residual_norm": float(np.linalg.norm(residual)),
        "batch_size": effective_batch_size,
        "max_batch_memory_mb": float(max_batch_memory_mb),
        "num_batches": num_batches,
        "backend": backend.name,
        "gpu_used": backend.name == "cupy",
        "gpu_num_batches": num_batches if backend.name == "cupy" else 0,
        "gpu_batch_size": effective_batch_size if backend.name == "cupy" else "",
        "gpu_device": backend.device if backend.name == "cupy" else "",
        "scoring_time_s": time.perf_counter() - scoring_start,
        "backend_warning": backend.warning,
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
    backend_cfg = BackendConfig.from_value(
        config.get("baselines", {}).get("backend_config")
    )
    BASELINE_CACHE.configure(
        enabled=backend_cfg.cache_enabled,
        memory_budget_gb=backend_cfg.cache_memory_budget_gb,
    )
    cache_before = BASELINE_CACHE.snapshot()
    cache_key = baseline_cache_key(
        "ff_omp",
        scene,
        config,
        grid_sizes=(
            int(cfg.get("angle_grid_size", 31)),
            int(cfg.get("delay_grid_size", 41)),
            "jones_group_omp",
        ),
    )
    groups = BASELINE_CACHE.get_or_create(
        cache_key,
        lambda: list(_far_field_supports(scene, config)),
    )
    max_groups = int(cfg.get("max_groups", cfg.get("max_atoms", scene["K"])))
    selected, expanded_supports, coeffs, y_hat_vec, diagnostics = group_omp_select(
        scene,
        config,
        groups,
        y_vec,
        max_groups=max_groups,
        batch_size=int(cfg.get("batch_size", 256)),
        trim_memory_enabled=bool(
            config.get("baselines", {}).get("trim_memory", True)
        ),
        backend_config=backend_cfg,
        static_cache_key=cache_key,
    )
    residual = y_vec - y_hat_vec
    p_hat, delta_t, geom_diag = geometric_support_to_position_ls(scene, selected, config)
    panels = [int(support.get("panel", 0)) for support in selected]
    if bool(cfg.get("offgrid_refinement", True)) and selected:
        p_hat, delta_t, coeffs, y_hat_vec, refined_groups, refine_diag = fit_position_clock_data_domain(
            scene,
            config,
            y_vec,
            p_hat,
            delta_t,
            panels=panels,
            model_variant="far_field",
            enabled=True,
            max_nfev=int(cfg.get("refinement_max_nfev", 60)),
        )
        selected = [
            {**selected[idx], **refined_groups[idx]}
            for idx in range(min(len(selected), len(refined_groups)))
        ]
        expanded_supports = [
            support for group in selected for support in expand_jones_group(group, 2)
        ]
        residual = y_vec - y_hat_vec
        diagnostics.update(refine_diag)
    else:
        diagnostics["offgrid_refinement"] = False
        diagnostics["refinement_objective"] = ""
    diagnostics.update(geom_diag)
    diagnostics["dictionary_mode"] = "far_field_angular_delay_omp"
    diagnostics["model_variant"] = "far_field_omp"
    diagnostics["group_omp"] = True
    diagnostics["expanded_supports"] = expanded_supports
    diagnostics["coeff_norm"] = float(np.linalg.norm(coeffs))
    diagnostics["angle_grid_type"] = "direction_cosine_cos_uniform"
    diagnostics["angle_grid_size"] = int(cfg.get("angle_grid_size", 31))
    diagnostics["delay_grid_size"] = int(cfg.get("delay_grid_size", 41))
    diagnostics["max_groups"] = max_groups
    diagnostics["selected_support"] = selected
    diagnostics.update(cache_diagnostics_delta(cache_before, BASELINE_CACHE.snapshot()))
    raw_objective = float(np.linalg.norm(residual) ** 2 / y_vec.size)
    return BaselineResult(
        name="ff_omp",
        p_u=p_hat,
        delta_t=delta_t,
        Y_hat=y_hat_vec.reshape(scene["I"], scene["N"], scene["T"]),
        raw_objective_final=raw_objective,
        components=_components_from_supports(scene, selected),
        selected_support=selected,
        runtime_s=time.perf_counter() - start,
        diagnostics=diagnostics,
    )
