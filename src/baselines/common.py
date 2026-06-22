"""Common utilities for standalone benchmark baselines.

The helpers in this module deliberately avoid the proposed JNPP gate and
continuous raw-domain VP refinement.  Baselines may use discrete dictionaries,
linear least-squares over selected atoms, and neutral geometry least-squares
post-processing.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..channel_model import channel_components, synthesize_raw_tensor
from ..geometry import (
    elev_az_from_unit_vector,
    far_field_ris_response,
    local_geometry_from_position,
    maxwell_matrix,
    near_field_spherical_response,
    unit_vector_from_elev_az,
)
from ..metrics import position_rmse, relative_nmse
from ..utils import scipy_is_available


@dataclass
class BaselineResult:
    name: str
    p_u: np.ndarray | None
    delta_t: float | None
    Y_hat: np.ndarray | None
    raw_objective_final: float
    components: dict[str, Any] = field(default_factory=dict)
    selected_support: list[dict[str, Any]] = field(default_factory=list)
    runtime_s: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def vectorize_raw_observation(Y: np.ndarray) -> np.ndarray:
    """Vectorize raw-domain observations in the repository's VP ordering."""
    return np.asarray(Y, dtype=complex).reshape(-1)


def hash_array(value: Any) -> str:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(float), decimals=12)
    payload = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha256(payload).hexdigest()[:16]


def y_noisy_hash(data: dict) -> str:
    return hash_array(data["Y_noisy"])


def data_hash(data: dict) -> str:
    scene = data.get("scene", {})
    payload = {
        "Y_noisy": y_noisy_hash(data),
        "K": int(scene.get("K", 0)),
        "receiver_mode": str(scene.get("receiver_mode", "")),
        "noise_variance": _finite_float(data.get("noise_variance")),
        "p_B": hash_array(scene.get("p_B", [])),
        "ris_centers": hash_array(scene.get("ris_centers", [])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return default
    return value_float if np.isfinite(value_float) else default


def _rmse_array(estimate: Any, truth: Any) -> float:
    if estimate is None or truth is None:
        return float("nan")
    estimate_arr = np.asarray(estimate, dtype=float).reshape(-1)
    truth_arr = np.asarray(truth, dtype=float).reshape(-1)
    if estimate_arr.size == 0 or estimate_arr.size != truth_arr.size:
        return float("nan")
    return float(np.linalg.norm(estimate_arr - truth_arr) / np.sqrt(estimate_arr.size))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def make_baseline_row(
    result: BaselineResult,
    data: dict,
    config: dict,
    *,
    baseline: str | None = None,
    trial_id: int = 0,
    seed: int | None = None,
    snr_db: float | None = None,
    failed: bool = False,
    error: str = "",
    warning: str = "",
) -> dict[str, Any]:
    """Convert a baseline result to one benchmark CSV row."""
    scene = data.get("scene", {})
    true_components = data.get("true_components", {})
    y_true = data.get("Y_true")
    y_hat = result.Y_hat
    p_true = scene.get("p_u_true")
    p_hat = result.p_u
    if y_hat is not None and y_true is not None and y_hat.shape == y_true.shape:
        y_nmse = relative_nmse(y_hat, y_true)
        raw_objective = float(
            np.linalg.norm(vectorize_raw_observation(y_hat - data["Y_noisy"])) ** 2
            / max(np.size(data["Y_noisy"]), 1)
        )
    else:
        y_nmse = float("nan")
        raw_objective = _finite_float(result.raw_objective_final)
    if np.isfinite(_finite_float(result.raw_objective_final)):
        raw_objective = _finite_float(result.raw_objective_final)
    selected_support = result.selected_support or []
    diagnostics = result.diagnostics or {}
    row_warning = warning or str(diagnostics.get("warning", ""))
    return {
        "baseline": baseline or result.name,
        "trial_id": int(trial_id),
        "seed": int(seed if seed is not None else config.get("seed", 0)),
        "snr_db": float(snr_db if snr_db is not None else config.get("SNR_dB", float("nan"))),
        "K": int(scene.get("K", config.get("K", 0))),
        "data_hash": data_hash(data),
        "y_noisy_hash": y_noisy_hash(data),
        "failed": bool(failed),
        "error": str(error),
        "runtime_s": float(result.runtime_s),
        "position_rmse_m": (
            position_rmse(np.asarray(p_hat, dtype=float), np.asarray(p_true, dtype=float))
            if p_hat is not None and p_true is not None
            else float("nan")
        ),
        "y_nmse": y_nmse,
        "range_rmse_m": _rmse_array(result.components.get("ranges"), true_components.get("ranges")),
        "tau_rmse_s": _rmse_array(result.components.get("taus"), true_components.get("taus")),
        "raw_objective_final": raw_objective,
        "support_size": len(selected_support),
        "grid_size": diagnostics.get("grid_size", ""),
        "dictionary_mode": diagnostics.get("dictionary_mode", ""),
        "selected_support": json.dumps(_jsonable(selected_support), separators=(",", ":")),
        "peb_position_m": _finite_float(diagnostics.get("peb_position_m")),
        "peb_is_data_only": diagnostics.get("peb_is_data_only", ""),
        "peb_uses_regularization": diagnostics.get("peb_uses_regularization", ""),
        "nuisance_model": diagnostics.get("nuisance_model", ""),
        "clock_eliminated": diagnostics.get("clock_eliminated", ""),
        "efim_condition_number": diagnostics.get("efim_condition_number", ""),
        "batch_size": diagnostics.get("batch_size", ""),
        "max_batch_memory_mb": diagnostics.get("max_batch_memory_mb", ""),
        "num_batches": diagnostics.get("num_batches", ""),
        "baseline_backend": diagnostics.get("backend", "cpu"),
        "gpu_used": diagnostics.get("gpu_used", False),
        "gpu_device": diagnostics.get("gpu_device", ""),
        "gpu_num_batches": diagnostics.get("gpu_num_batches", 0),
        "gpu_batch_size": diagnostics.get("gpu_batch_size", ""),
        "cache_enabled": diagnostics.get("cache_enabled", False),
        "cache_hits": diagnostics.get("cache_hits", 0),
        "cache_misses": diagnostics.get("cache_misses", 0),
        "cache_estimated_bytes": diagnostics.get("cache_estimated_bytes", 0),
        "scoring_time_s": diagnostics.get("scoring_time_s", ""),
        "backend_warning": diagnostics.get("backend_warning", ""),
        "warning": row_warning,
    }


def linear_ls_fit(Phi: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust complex linear least-squares fit."""
    Phi = np.asarray(Phi, dtype=complex)
    y = np.asarray(y, dtype=complex).reshape(-1)
    if Phi.ndim == 1:
        Phi = Phi[:, None]
    if Phi.shape[0] != y.size:
        raise ValueError("Phi row count must match y length")
    if Phi.shape[1] == 0:
        y_hat = np.zeros_like(y)
        return np.zeros(0, dtype=complex), y_hat, y - y_hat
    ridge_value = float(ridge)
    if ridge_value > 0.0:
        gram = Phi.conj().T @ Phi
        rhs = Phi.conj().T @ y
        coeffs = np.linalg.solve(
            gram + ridge_value * np.eye(gram.shape[0], dtype=complex),
            rhs,
        )
    else:
        coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    y_hat = Phi @ coeffs
    return coeffs, y_hat, y - y_hat


def simple_atom_normalize(atom: np.ndarray) -> np.ndarray:
    atom = np.asarray(atom, dtype=complex).reshape(-1)
    norm = np.linalg.norm(atom)
    if not np.isfinite(norm) or norm <= 0.0:
        return atom
    return atom / norm


def build_jones_basis_evs_atoms(scene: dict, config: dict, path_index: int | None = None, panel_index: int | None = None) -> tuple[list[np.ndarray], list[str]]:
    """Return EVS basis responses for Jones basis vectors e1/e2."""
    _ = config
    k = int(panel_index if panel_index is not None else path_index if path_index is not None else 0)
    warnings: list[str] = []
    try:
        theta = np.asarray(scene["Theta"][k], dtype=complex)
        v_b = np.asarray(scene["v_B"][k], dtype=complex)
        mask = np.asarray(scene.get("evs_observation_mask", np.ones(6 * v_b.size)), dtype=float)
        atoms = [np.kron(v_b, theta[:, j]) * mask for j in range(2)]
    except Exception as exc:  # noqa: BLE001 - fallback for synthetic mocks.
        i_dim = int(scene.get("I", 1))
        atoms = [np.ones(i_dim, dtype=complex)]
        warnings.append(f"evs_jones_basis_fallback: {type(exc).__name__}: {exc}")
    return atoms, warnings


def delay_response(scene: dict, tau: float) -> np.ndarray:
    n_idx = np.arange(int(scene["N"]), dtype=float)
    pole = np.exp(-1j * 2.0 * np.pi * float(scene["delta_f"]) * float(tau))
    return pole ** n_idx


def training_response_from_position(scene: dict, panel: int, p_u: np.ndarray, *, near_field: bool = True) -> np.ndarray:
    panel = int(panel)
    if near_field:
        range_m, elevation, azimuth, _ = local_geometry_from_position(
            np.asarray(p_u, dtype=float),
            np.asarray(scene["ris_centers"][panel], dtype=float),
            np.asarray(scene["rotations"][panel], dtype=float),
        )
        a_ur = near_field_spherical_response(
            range_m,
            elevation,
            azimuth,
            np.asarray(scene["ris_grid"], dtype=float),
            float(scene["wavelength"]),
        )
    else:
        a_ur = far_field_ris_response(
            np.asarray(scene["ris_centers"][panel], dtype=float),
            np.asarray(p_u, dtype=float),
            np.asarray(scene["rotations"][panel], dtype=float),
            np.asarray(scene["ris_grid"], dtype=float),
            float(scene["wavelength"]),
        )
    g_elem = np.asarray(scene["a_RB"][panel], dtype=complex) * a_ur
    return np.asarray(scene["Omega"][panel], dtype=complex) @ g_elem


def training_response_from_direction(scene: dict, panel: int, direction_local: np.ndarray) -> np.ndarray:
    panel = int(panel)
    direction = np.asarray(direction_local, dtype=float).reshape(3)
    direction /= np.linalg.norm(direction) + 1.0e-15
    wavenumber = 2.0 * np.pi / float(scene["wavelength"])
    a_ur = np.exp(-1j * wavenumber * (np.asarray(scene["ris_grid"], dtype=float) @ direction))
    g_elem = np.asarray(scene["a_RB"][panel], dtype=complex) * a_ur
    return np.asarray(scene["Omega"][panel], dtype=complex) @ g_elem


def raw_atom_from_factors(evs: np.ndarray, delay: np.ndarray, training: np.ndarray) -> np.ndarray:
    return (
        np.asarray(evs, dtype=complex)[:, None, None]
        * np.asarray(delay, dtype=complex)[None, :, None]
        * np.asarray(training, dtype=complex)[None, None, :]
    ).reshape(-1)


def raw_atom_from_support(scene: dict, config: dict, support: dict[str, Any]) -> np.ndarray:
    panel = int(support.get("panel", 0))
    pol_index = int(support.get("pol_index", 0))
    evs_atoms, _ = build_jones_basis_evs_atoms(scene, config, panel_index=panel)
    evs = evs_atoms[min(pol_index, len(evs_atoms) - 1)]
    tau = float(support.get("tau", 0.0))
    delay = delay_response(scene, tau)
    if "position" in support:
        training = training_response_from_position(
            scene,
            panel,
            np.asarray(support["position"], dtype=float),
            near_field=bool(support.get("near_field", True)),
        )
    elif "direction" in support:
        training = training_response_from_direction(
            scene,
            panel,
            np.asarray(support["direction"], dtype=float),
        )
    else:
        training = np.ones(int(scene["T"]), dtype=complex)
    return raw_atom_from_factors(evs, delay, training)


def supports_to_design(scene: dict, config: dict, supports: list[dict[str, Any]]) -> np.ndarray:
    atoms = [simple_atom_normalize(raw_atom_from_support(scene, config, support)) for support in supports]
    if not atoms:
        return np.empty((int(scene["I"]) * int(scene["N"]) * int(scene["T"]), 0), dtype=complex)
    return np.column_stack(atoms)


def reconstruct_from_supports(
    scene: dict,
    config: dict,
    supports: list[dict[str, Any]],
    y_vec: np.ndarray,
    *,
    ridge: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Phi = supports_to_design(scene, config, supports)
    coeffs, y_hat_vec, residual = linear_ls_fit(Phi, y_vec, ridge=ridge)
    return coeffs, y_hat_vec.reshape(scene["I"], scene["N"], scene["T"]), residual, Phi


def position_grid_from_config(config: dict, shape: tuple[int, int, int]) -> list[np.ndarray]:
    bounds = np.asarray(config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]), dtype=float)
    axes = [np.linspace(bounds[idx, 0], bounds[idx, 1], int(shape[idx])) for idx in range(3)]
    return [np.array([x, y, z], dtype=float) for x in axes[0] for y in axes[1] for z in axes[2]]


def clock_grid_from_config(config: dict, size: int) -> np.ndarray:
    bounds = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    return np.linspace(float(bounds[0]), float(bounds[1]), int(size))


def direction_grid(angle_grid_size: int) -> list[np.ndarray]:
    size = max(int(angle_grid_size), 3)
    axis_count = max(3, int(np.ceil(np.sqrt(size))))
    ux_axis = np.linspace(-0.85, 0.85, axis_count)
    uy_axis = np.linspace(-0.85, 0.85, axis_count)
    directions = []
    for ux in ux_axis:
        for uy in uy_axis:
            rem = 1.0 - ux * ux - uy * uy
            if rem <= 0.0:
                continue
            directions.append(np.array([ux, uy, np.sqrt(rem)], dtype=float))
            if len(directions) >= size:
                return directions
    return directions


def delay_grid_from_scene(scene: dict, config: dict, size: int) -> np.ndarray:
    bounds = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    ue_bounds = np.asarray(config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]), dtype=float)
    corners = np.array(
        [[x, y, z] for x in ue_bounds[0] for y in ue_bounds[1] for z in ue_bounds[2]],
        dtype=float,
    )
    taus = []
    for panel in range(int(scene["K"])):
        ranges = np.linalg.norm(corners - np.asarray(scene["ris_centers"][panel]), axis=1)
        taus.extend(((ranges + scene["d_RB"][panel]) / scene["c0"] + bounds[0]).tolist())
        taus.extend(((ranges + scene["d_RB"][panel]) / scene["c0"] + bounds[1]).tolist())
    return np.linspace(float(np.min(taus)), float(np.max(taus)), int(size))


def geometric_support_to_position_ls(scene: dict, supports: list[dict[str, Any]], config: dict) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Estimate UE position/clock from neutral support geometry."""
    direct_positions = [
        np.asarray(support["position"], dtype=float).reshape(3)
        for support in supports
        if "position" in support
    ]
    taus = [float(support["tau"]) for support in supports if "tau" in support]
    if direct_positions:
        p_hat = np.median(np.asarray(direct_positions, dtype=float), axis=0)
        if taus:
            dt_values = []
            for support in supports:
                if "tau" not in support:
                    continue
                panel = int(support.get("panel", 0))
                dist = np.linalg.norm(p_hat - scene["ris_centers"][panel])
                dt_values.append(float(support["tau"]) - (dist + scene["d_RB"][panel]) / scene["c0"])
            delta_t = (
                float(np.median(dt_values))
                if dt_values
                else float(np.mean(clock_grid_from_config(config, 3)))
            )
        else:
            delta_t = float(np.mean(clock_grid_from_config(config, 3)))
        return p_hat, delta_t, {"geometry_solver": "direct_position_candidate"}

    bounds_p = np.asarray(config.get("ue_bounds", [[0.3, 2.7], [-1.4, 1.5], [0.35, 1.45]]), dtype=float)
    bounds_dt = np.asarray(config.get("delta_t_bounds", [0.0, 10.0e-9]), dtype=float)
    p0 = np.mean(bounds_p, axis=1)
    dt0 = float(np.mean(bounds_dt))
    if supports:
        panel_points = []
        for support in supports:
            if "direction" in support:
                panel = int(support.get("panel", 0))
                direction = np.asarray(support["direction"], dtype=float).reshape(3)
                direction /= np.linalg.norm(direction) + 1.0e-15
                panel_points.append(scene["ris_centers"][panel] + 4.0 * scene["rotations"][panel].T @ direction)
        if panel_points:
            p0 = np.mean(np.asarray(panel_points), axis=0)
            p0 = np.clip(p0, bounds_p[:, 0], bounds_p[:, 1])

    def residual_fn(chi: np.ndarray) -> np.ndarray:
        p = chi[:3]
        dt = float(chi[3])
        residuals: list[float] = []
        for support in supports:
            panel = int(support.get("panel", 0))
            q_local = scene["rotations"][panel] @ (p - scene["ris_centers"][panel])
            rng = np.linalg.norm(q_local) + 1.0e-15
            direction = q_local / rng
            if "direction" in support:
                direction_hat = np.asarray(support["direction"], dtype=float).reshape(3)
                direction_hat /= np.linalg.norm(direction_hat) + 1.0e-15
                residuals.extend((direction - direction_hat).tolist())
            if "tau" in support:
                tau_model = (rng + scene["d_RB"][panel]) / scene["c0"] + dt
                residuals.append((tau_model - float(support["tau"])) / 1.0e-9)
        if not residuals:
            residuals.extend((p - p0).tolist())
        return np.asarray(residuals, dtype=float)

    lower = np.r_[bounds_p[:, 0], bounds_dt[0]]
    upper = np.r_[bounds_p[:, 1], bounds_dt[1]]
    x0 = np.clip(np.r_[p0, dt0], lower, upper)
    if scipy_is_available():
        try:
            from scipy.optimize import least_squares  # type: ignore[import-not-found]

            result = least_squares(
                residual_fn,
                x0,
                bounds=(lower, upper),
                max_nfev=int(config.get("baselines", {}).get("geometry_max_nfev", 100)),
            )
            chi = result.x
            diagnostics = {
                "geometry_solver": "scipy.optimize.least_squares",
                "geometry_cost": float(result.cost),
                "geometry_success": bool(result.success),
            }
            return chi[:3].astype(float), float(chi[3]), diagnostics
        except Exception as exc:  # noqa: BLE001 - fall back to bounded center.
            return p0.astype(float), dt0, {"geometry_solver": "fallback_center", "warning": str(exc)}
    return p0.astype(float), dt0, {"geometry_solver": "fallback_center"}


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.runtime_s = time.perf_counter() - self.start
        return False


def synthesize_from_position_jones(
    scene: dict,
    p_u: np.ndarray,
    delta_t: float,
    coeffs: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Synthesize a raw tensor from per-panel Jones coefficients."""
    k_paths = int(scene["K"])
    gamma = np.full(k_paths, np.pi / 4.0)
    eta = np.zeros(k_paths)
    comps = channel_components(scene, np.asarray(p_u, dtype=float), float(delta_t), gamma, eta)
    beta = np.zeros(k_paths, dtype=complex)
    coeffs = np.asarray(coeffs, dtype=complex).reshape(-1)
    for k in range(k_paths):
        start = 2 * k
        if start < coeffs.size:
            beta[k] = coeffs[start]
    return synthesize_raw_tensor(comps, beta), comps
