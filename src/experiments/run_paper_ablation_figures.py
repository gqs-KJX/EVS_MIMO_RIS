"""Generate paper ablation CSVs and PDF figures for the revised estimator."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import pathlib
import platform
import subprocess
import sys
import time
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.channel_model import channel_components
    from src.config import default_config
    from src.diagnostics import estimate_position_from_ris_eta
    from src.estimators import (
        reconstruct_raw_tensor_from_structured_estimate,
        estimate_position_from_local_ris,
        global_exact_spherical_vp_refinement,
    )
    from src.global_vp import data_only_efim_diagnostic
    from src.main_single_proposed import (
        _apply_main_single_defaults,
        _make_data,
        run_from_existing_stage1,
        run_single_proposed_diagnostic,
        run_stage1_only,
    )
    from src.metrics import position_rmse, relative_nmse
    from src.projections_delay import tau_from_pole
    from src.utils import scipy_is_available
else:
    from ..channel_model import channel_components
    from ..config import default_config
    from ..diagnostics import estimate_position_from_ris_eta
    from ..estimators import (
        reconstruct_raw_tensor_from_structured_estimate,
        estimate_position_from_local_ris,
        global_exact_spherical_vp_refinement,
    )
    from ..global_vp import data_only_efim_diagnostic
    from ..main_single_proposed import (
        _apply_main_single_defaults,
        _make_data,
        run_from_existing_stage1,
        run_single_proposed_diagnostic,
        run_stage1_only,
    )
    from ..metrics import position_rmse, relative_nmse
    from ..projections_delay import tau_from_pole
    from ..utils import scipy_is_available


DEFAULT_SNR_GRID = "-30,-25,-20,-15,-10,-5,0,5,10"
DEFAULT_PAPER_K = 3
DEFAULT_BLAS_THREADS = 1
LARGE_ARRAY_ELEMENT_THRESHOLD = 1_000_000
RECEIVER_NOISE_CONVENTION = (
    "AWGN is added only on active EVS observation components with variance set "
    "by active-component signal power."
)
RECEIVER_MODE_CONVENTION = (
    "receiver_mode selects the EVS component observation mask before noise "
    "generation and estimator evaluation."
)
FIGURE6_K_GRID = [1, 2, 3, 4]
FIGURE_ORDER = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]
FIG1_FIG2_SHARED_FIGURE = "fig1_fig2"
FIG1_FIG2_SHARED_TRIAL_CSV = "fig1_fig2_vp_family_trials.csv"
FIG1_FIG2_SHARED_SUMMARY_CSV = "fig1_fig2_vp_family_summary.csv"
FIGURE_PDFS = {
    "fig1": "fig1_vp_family_rmse_vs_snr.pdf",
    "fig2": "fig2_vp_family_nmse_vs_snr.pdf",
    "fig3": "fig3_evs_sensing_rmse_vs_snr.pdf",
    "fig4": "fig4_evs_peb_vs_snr.pdf",
    "fig5": "fig5_stage2_gate_outlier_vs_snr.pdf",
    "fig6": "fig6_rmse_vs_K_snr0.pdf",
}
FIGURE_METRICS = {
    "fig1": "position_rmse_m",
    "fig2": "y_nmse",
    "fig3": "position_rmse_m",
    "fig4": "peb_position_m",
    "fig5": "outlier_flag",
    "fig6": "position_rmse_m",
}
RAW_SUMMARY_METRICS = [
    "position_rmse_m",
    "y_nmse",
    "range_rmse_m",
    "tau_rmse_s",
    "raw_objective_final",
    "outlier_flag",
    "peb_position_m",
    "peb_scalar_m",
    "peb_dual_m",
    "peb_evs_m",
]

FIELDNAMES = [
    "figure",
    "variant",
    "trial_id",
    "seed",
    "snr_db",
    "x_name",
    "x_value",
    "K",
    "receiver_mode",
    "failed",
    "error",
    "runtime_s",
    "position_rmse_m",
    "y_nmse",
    "range_rmse_m",
    "tau_rmse_s",
    "raw_objective_final",
    "outlier_flag",
    "selected_branch",
    "final_refinement_method",
    "global_vp_mode",
    "selected_vp_family_branch",
    "linear_nuisance_dim",
    "nonlinear_dim",
    "fixed_pol_score",
    "jones_score",
    "lambda_jones_per_path",
    "snr_eff_per_path",
    "jones_leakage_per_path",
    "reliability_decision",
    "trigger_reasons",
    "gof_stat",
    "gof_dof",
    "gof_pass",
    "data_only_scaled_efim_lambda_min",
    "data_only_scaled_efim_condition_number",
    "stage1_assignment_margin",
    "stage1_selected_clock_std_ns",
    "delta_t_k_ns",
    "stage1_runtime_s",
    "global_vp_runtime_s",
    "total_runtime_s",
    "peb_position_m",
    "peb_scalar_m",
    "peb_dual_m",
    "peb_evs_m",
    "rss_mb_before",
    "rss_mb_after",
    "rss_mb_delta",
    "warning",
]

_PEB_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_WORKER_BASE_CONFIG: dict[str, Any] | None = None
_WORKER_OUT_DIR: pathlib.Path | None = None
_WORKER_BLAS_THREADS: int = DEFAULT_BLAS_THREADS


def _is_fig1_fig2(figure: str) -> bool:
    return figure in {"fig1", "fig2", FIG1_FIG2_SHARED_FIGURE}


def parse_snr_grid(value: str | Iterable[float] = DEFAULT_SNR_GRID) -> list[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def parse_k_grid(value: str | Iterable[int] = "1,2,3,4") -> list[int]:
    if isinstance(value, str):
        grid = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        grid = [int(item) for item in value]
    if not grid:
        raise ValueError("--k-grid must contain at least one positive integer")
    if any(k <= 0 for k in grid):
        raise ValueError("--k-grid entries must be positive")
    return grid


def _apply_blas_thread_env(blas_threads: int) -> None:
    value = str(int(blas_threads))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, value)


def parse_figures(value: str) -> list[str]:
    if value == "all":
        return list(FIGURE_ORDER)
    figures = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in figures if item not in FIGURE_ORDER]
    if unknown:
        raise ValueError(f"unknown figure ids: {unknown}")
    requested = set(figures)
    return [figure for figure in FIGURE_ORDER if figure in requested]


def apply_nested_update(config: dict, update_dict: dict) -> dict:
    result = copy.deepcopy(config)
    for key, value in update_dict.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = apply_nested_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_number_of_ris_paths(config: dict, k_paths: int) -> dict:
    """Set the physical path/RIS count used by scene generation and estimators."""
    k_paths = int(k_paths)
    if k_paths <= 0:
        raise ValueError("K must be positive")
    config["K"] = k_paths
    ris_centers = np.asarray(config.get("ris_centers"), dtype=float)
    if ris_centers.ndim != 2 or ris_centers.shape[1] != 3:
        raise ValueError("config['ris_centers'] must have shape (num_ris, 3)")
    if ris_centers.shape[0] < k_paths:
        extra = []
        base_z = float(np.mean(ris_centers[:, 2])) if ris_centers.size else 1.2
        while ris_centers.shape[0] + len(extra) < k_paths:
            idx = ris_centers.shape[0] + len(extra)
            side = -1.0 if idx % 2 else 1.0
            y_offset = side * (2.8 + 0.35 * idx)
            x_offset = 4.6 + 0.35 * idx
            z_offset = base_z + 0.05 * ((idx % 3) - 1)
            extra.append([x_offset, y_offset, z_offset])
        ris_centers = np.vstack([ris_centers, np.asarray(extra, dtype=float)])
    config["ris_centers"] = ris_centers
    config["jnpp_max_candidates"] = max(int(config.get("jnpp_max_candidates", 1)), 1 + k_paths)
    config["jnpp_top_assignments"] = min(3, math.factorial(k_paths))
    return config


def make_base_config(seed: int, snr_db: float, overrides: dict | None = None) -> dict:
    config = default_config()
    config["seed"] = int(seed)
    config["SNR_dB"] = float(snr_db)
    config["print_progress"] = False
    config["verbose_stage2"] = False
    config["run_full_legacy_comparison"] = False
    if overrides:
        config = apply_nested_update(config, overrides)
    set_number_of_ris_paths(config, int(config.get("K", 1)))
    return config


def _variant_specs(figure: str) -> dict[str, dict[str, Any]]:
    if _is_fig1_fig2(figure):
        return {
            "stage1_only": {
                "enable_global_vp": False,
                "stage2_adaptive": False,
                "_runner": "stage1_only",
                "_allow_stage2": False,
            },
            "fixed_pol_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "fixed_pol"},
                "_allow_stage2": False,
            },
            "free_jones_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_free"},
                "_allow_stage2": False,
            },
            "regularized_jones_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_regularized"},
                "_allow_stage2": False,
            },
            "adaptive_jones_vp_proposed": {
                "enable_global_vp": True,
                "global_vp": {"mode": "adaptive_jones"},
                "_allow_stage2": False,
            },
        }
    if figure == "fig3":
        return {
            "scalar_receiver": {"receiver_mode": "scalar", "global_vp": {"mode": "adaptive_jones"}},
            "dual_pol_receiver": {"receiver_mode": "dual_pol", "global_vp": {"mode": "adaptive_jones"}},
            "full_6d_evs": {"receiver_mode": "full_6d", "global_vp": {"mode": "adaptive_jones"}},
        }
    if figure == "fig4":
        return {
            "scalar_peb": {"receiver_mode": "scalar", "_runner": "peb_only"},
            "dual_pol_peb": {"receiver_mode": "dual_pol", "_runner": "peb_only"},
            "full_6d_evs_peb": {"receiver_mode": "full_6d", "_runner": "peb_only"},
        }
    if figure == "fig5":
        return {
            "direct_vp": {
                "global_vp": {"mode": "adaptive_jones"},
                "_allow_stage2": False,
            },
            "jnpp_always": {
                "global_vp": {"mode": "adaptive_jones"},
                "proposed_stage2_policy": "force_ris_only",
                "_allow_stage2": True,
            },
            "reliability_gated_proposed": {
                "global_vp": {"mode": "adaptive_jones"},
                "proposed_stage2_policy": "reliability_gated_ris_only",
                "_allow_stage2": True,
            },
            "oracle_init_vp": {
                "global_vp": {"mode": "adaptive_jones"},
                "_runner": "oracle_init_vp",
                "_allow_stage2": False,
            },
        }
    if figure == "fig6":
        return {
            "fixed_pol_vp": {"global_vp": {"mode": "fixed_pol"}, "_allow_stage2": False},
            "free_jones_vp": {"global_vp": {"mode": "jones_free"}, "_allow_stage2": False},
            "adaptive_jones_vp_proposed": {
                "global_vp": {"mode": "adaptive_jones"},
                "_allow_stage2": False,
            },
            "proposed_peb": {"global_vp": {"mode": "adaptive_jones"}, "_runner": "peb_only"},
        }
    raise ValueError(f"unknown figure {figure!r}")


def _extra_peb_specs(figure: str) -> dict[str, dict[str, Any]]:
    if _is_fig1_fig2(figure):
        return {"PEB": {"receiver_mode": "full_6d", "_runner": "peb_only"}}
    if figure == "fig3":
        return {
            "full_6d_evs_peb": {
                "receiver_mode": "full_6d",
                "_runner": "peb_only",
            }
        }
    return {}


def _trial_seed(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _path_get(container: Any, path: tuple[Any, ...], default: Any) -> Any:
    current = container
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        elif isinstance(current, (list, tuple)) and isinstance(key, int):
            if key >= len(current):
                return default
            current = current[key]
        else:
            return default
    return current


def get_nested(result: dict, possible_paths: Iterable[str | tuple[Any, ...]], default: Any = np.nan) -> Any:
    for path in possible_paths:
        parts = tuple(path.split(".")) if isinstance(path, str) else tuple(path)
        value = _path_get(result, parts, default)
        if value is not default:
            return value
    return default


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


def _vector_string(value: Any) -> str:
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    if np.iscomplexobj(arr):
        payload = [[float(np.real(item)), float(np.imag(item))] for item in arr.reshape(-1)]
    else:
        payload = [float(item) for item in np.asarray(arr, dtype=float).reshape(-1)]
    return json.dumps(payload, separators=(",", ":"))


def _list_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    return str(value)


def _empty_row() -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    numeric_fields = [
        "runtime_s",
        "position_rmse_m",
        "y_nmse",
        "range_rmse_m",
        "tau_rmse_s",
        "raw_objective_final",
        "gof_stat",
        "data_only_scaled_efim_lambda_min",
        "data_only_scaled_efim_condition_number",
        "stage1_assignment_margin",
        "stage1_selected_clock_std_ns",
        "stage1_runtime_s",
        "global_vp_runtime_s",
        "total_runtime_s",
        "peb_position_m",
        "peb_scalar_m",
        "peb_dual_m",
        "peb_evs_m",
        "rss_mb_before",
        "rss_mb_after",
        "rss_mb_delta",
    ]
    for field in numeric_fields:
        row[field] = float("nan")
    return row


def _rss_mb() -> float:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return float("nan")
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0**2))
    except Exception:  # noqa: BLE001 - memory profiling is best-effort only.
        return float("nan")


def compact_experiment_result(
    result: Any,
    keep_large_arrays: bool = False,
    *,
    large_array_threshold: int = LARGE_ARRAY_ELEMENT_THRESHOLD,
) -> Any:
    """Drop large tensors from a diagnostic result after metrics are extracted."""
    if keep_large_arrays:
        return result
    large_names = {
        "Y_true",
        "Y_noisy",
        "Y_hat",
        "Z_true",
        "Z_noisy",
        "Z_hat",
        "tensor",
        "hankel",
        "raw_tensor",
        "raw_tensors",
    }

    def _compact(value: Any, key: str | None = None) -> Any:
        if isinstance(value, np.ndarray):
            key_lower = "" if key is None else key.lower()
            if key in large_names or key_lower in large_names or value.size > large_array_threshold:
                return None
            return value
        if isinstance(value, dict):
            for child_key in list(value.keys()):
                value[child_key] = _compact(value[child_key], str(child_key))
            return value
        if isinstance(value, list):
            for idx, item in enumerate(value):
                value[idx] = _compact(item)
            return value
        if isinstance(value, tuple):
            return tuple(_compact(item) for item in value)
        return value

    return _compact(result)


def _truth_init_estimate(scene: dict, true_components: dict) -> dict:
    k_paths = int(scene["K"])
    return {
        "A": true_components["a_EVS"].T.copy(),
        "D": true_components["d"].T.copy(),
        "C": true_components["c"].T.copy(),
        "poles": true_components["poles"].copy(),
        "ris_eta": np.column_stack(
            [
                true_components["ranges"],
                true_components["elevations"],
                true_components["azimuths"],
            ]
        ),
        "gamma": scene["gamma_true"].copy(),
        "eta_pol": scene["eta_true"].copy(),
        "assignment": list(range(k_paths)),
        "panel_to_column_assignment": list(range(k_paths)),
        "columns_are_panel_ordered": True,
        "Z_hat": np.zeros((scene["I"], scene["P"], scene["L"], scene["T"]), dtype=complex),
        "beta_z": np.ones(k_paths, dtype=complex),
    }


def _hash_array(value: Any) -> str:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(float), decimals=12)
    payload = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha256(payload).hexdigest()[:16]


def peb_cache_key(config: dict) -> tuple[Any, ...]:
    """Return a deterministic key for data-only EFIM PEB computations."""
    return (
        float(config.get("SNR_dB")),
        int(config.get("K")),
        str(config.get("receiver_mode", config.get("evs_selection", "full_6d"))),
        int(config.get("seed")),
        float(config.get("fc")),
        tuple(int(item) for item in config.get("ris_shape", ())),
        int(config.get("M_A")),
        int(config.get("N")),
        int(config.get("P")),
        int(config.get("T")),
        _hash_array(config.get("p_B")),
        _hash_array(config.get("p_u_true")),
        _hash_array(config.get("ris_centers")),
        float(config.get("delta_t_true")),
        _hash_config_for_peb(config),
    )


def _hash_config_for_peb(config: dict) -> str:
    payload = {
        "global_vp": config.get("global_vp", {}),
        "eps": config.get("eps", None),
        "delta_f": config.get("delta_f", None),
        "wavelength": config.get("wavelength", None),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _json_safe_float(value: Any) -> float | None:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if np.isfinite(value_float) else None


def _restore_cached_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def _peb_cache_path(out_dir: pathlib.Path | None) -> pathlib.Path | None:
    if out_dir is None:
        return None
    return pathlib.Path(out_dir) / ".cache" / "peb_cache.sqlite"


def _init_peb_cache(db_path: pathlib.Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS peb_cache (
                cache_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _peb_cache_key_string(config: dict) -> str:
    return json.dumps(peb_cache_key(config), sort_keys=True, default=str, separators=(",", ":"))


def _read_persistent_peb_cache(config: dict, out_dir: pathlib.Path | None) -> dict[str, Any] | None:
    db_path = _peb_cache_path(out_dir)
    if db_path is None or not db_path.exists():
        return None
    cache_key = _peb_cache_key_string(config)
    try:
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            row = conn.execute(
                "SELECT value_json FROM peb_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    payload = json.loads(row[0])
    return {
        "peb_position_m": _restore_cached_float(payload.get("peb_position_m")),
        "peb_scalar_m": _restore_cached_float(payload.get("peb_scalar_m")),
        "peb_dual_m": _restore_cached_float(payload.get("peb_dual_m")),
        "peb_evs_m": _restore_cached_float(payload.get("peb_evs_m")),
        "warning": payload.get("warning", ""),
    }


def _write_persistent_peb_cache(
    config: dict,
    out_dir: pathlib.Path | None,
    value: dict[str, Any],
) -> None:
    db_path = _peb_cache_path(out_dir)
    if db_path is None:
        return
    _init_peb_cache(db_path)
    payload = {
        "peb_position_m": _json_safe_float(value.get("peb_position_m")),
        "peb_scalar_m": _json_safe_float(value.get("peb_scalar_m")),
        "peb_dual_m": _json_safe_float(value.get("peb_dual_m")),
        "peb_evs_m": _json_safe_float(value.get("peb_evs_m")),
        "warning": str(value.get("warning", "")),
    }
    with sqlite3.connect(db_path, timeout=30.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT OR REPLACE INTO peb_cache(cache_key, value_json) VALUES (?, ?)",
            (_peb_cache_key_string(config), json.dumps(payload, sort_keys=True)),
        )
        conn.commit()


def _peb_from_efim(data: dict, config: dict) -> dict[str, float]:
    scene = data["scene"]
    init = _truth_init_estimate(scene, data["true_components"])
    warning = ""
    try:
        diag = data_only_efim_diagnostic(
            data["Y_true"],
            scene["p_u_true"],
            scene["delta_t_true"],
            init,
            scene,
            config,
            sigma2=data.get("noise_variance"),
        )
        efim = np.asarray(diag["data_only_scaled_efim"], dtype=float)
        pos_efim = efim[:3, :3]
        cov = np.linalg.pinv((pos_efim + pos_efim.T) * 0.5)
        peb = float(np.sqrt(max(np.trace(cov), 0.0)))
        if not np.isfinite(peb):
            warning = "data_only_efim_peb_nonfinite"
    except (KeyError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        peb = float("nan")
        warning = f"data_only_efim_peb_failed: {type(exc).__name__}: {exc}"
    mode = str(config.get("receiver_mode", "full_6d"))
    return {
        "peb_position_m": peb,
        "peb_scalar_m": peb if mode == "scalar" else float("nan"),
        "peb_dual_m": peb if mode == "dual_pol" else float("nan"),
        "peb_evs_m": peb if mode == "full_6d" else float("nan"),
        "warning": warning,
    }


def extract_metrics(result: dict, outlier_threshold_m: float) -> dict[str, Any]:
    final = result.get("final", {})
    scene = result.get("scene", {})
    true_components = result.get("true_components", {})
    timing = result.get("timing", {})
    reliability = result.get("reliability", final.get("reliability", {}))

    y_nmse = float("nan")
    if final.get("Y_hat") is not None and result.get("Y_true") is not None:
        y_nmse = float(relative_nmse(final["Y_hat"], result["Y_true"]))

    pos_rmse = float("nan")
    if final.get("p_u") is not None and scene.get("p_u_true") is not None:
        pos_rmse = float(position_rmse(np.asarray(final["p_u"]), np.asarray(scene["p_u_true"])))

    components = final.get("components", {})
    metrics = {
        "K": scene.get("K", ""),
        "receiver_mode": scene.get("receiver_mode", ""),
        "warning": result.get("warning", ""),
        "position_rmse_m": pos_rmse,
        "y_nmse": y_nmse,
        "range_rmse_m": _rmse_array(components.get("ranges"), true_components.get("ranges")),
        "tau_rmse_s": _rmse_array(components.get("taus"), true_components.get("taus")),
        "raw_objective_final": _finite_float(
            get_nested(result, ["final.raw_objective_final", "final.raw_objective"], np.nan)
        ),
        "outlier_flag": bool(np.isfinite(pos_rmse) and pos_rmse > outlier_threshold_m),
        "selected_branch": get_nested(result, ["selected_branch", "final.selected_branch"], ""),
        "final_refinement_method": get_nested(result, ["final.final_refinement_method"], ""),
        "global_vp_mode": get_nested(result, ["final.global_vp_mode", "final.vp_mode"], ""),
        "selected_vp_family_branch": get_nested(result, ["final.selected_vp_family_branch"], ""),
        "linear_nuisance_dim": get_nested(result, ["final.linear_nuisance_dim"], ""),
        "nonlinear_dim": get_nested(result, ["final.nonlinear_dim"], ""),
        "fixed_pol_score": _finite_float(get_nested(result, ["final.fixed_pol_score"], np.nan)),
        "jones_score": _finite_float(get_nested(result, ["final.jones_score"], np.nan)),
        "lambda_jones_per_path": _vector_string(
            get_nested(result, ["final.lambda_jones_per_path"], None)
        ),
        "snr_eff_per_path": _vector_string(get_nested(result, ["final.snr_eff_per_path"], None)),
        "jones_leakage_per_path": _vector_string(
            get_nested(result, ["final.jones_leakage_per_path"], None)
        ),
        "reliability_decision": reliability.get("decision", ""),
        "trigger_reasons": _list_string(reliability.get("trigger_reasons", [])),
        "gof_stat": _finite_float(reliability.get("gof_stat")),
        "gof_dof": reliability.get("gof_dof", ""),
        "gof_pass": reliability.get("gof_pass", ""),
        "data_only_scaled_efim_lambda_min": _finite_float(
            reliability.get("data_only_scaled_efim_lambda_min")
        ),
        "data_only_scaled_efim_condition_number": _finite_float(
            reliability.get("data_only_scaled_efim_condition_number")
        ),
        "stage1_assignment_margin": _finite_float(reliability.get("assignment_margin")),
        "stage1_selected_clock_std_ns": _finite_float(reliability.get("sigma_delta_t_ns")),
        "delta_t_k_ns": _vector_string(reliability.get("delta_t_k_ns")),
        "stage1_runtime_s": _finite_float(timing.get("stage1")),
        "global_vp_runtime_s": _finite_float(timing.get("vp")),
        "total_runtime_s": _finite_float(timing.get("total", timing.get("diagnostic_total"))),
    }
    for key in ("peb_position_m", "peb_scalar_m", "peb_dual_m", "peb_evs_m"):
        metrics[key] = _finite_float(result.get(key))
    return metrics


def _stage1_only_result(config: dict) -> dict:
    data = _make_data(config)
    stage1 = run_stage1_only(data, config)
    estimate = stage1["estimate"]
    y_hat = reconstruct_raw_tensor_from_structured_estimate(estimate, data["scene"])
    try:
        p_hat = estimate_position_from_local_ris(data["scene"], estimate, config)
    except (KeyError, ValueError, np.linalg.LinAlgError):
        p_hat = estimate_position_from_ris_eta(data["scene"], estimate)
    tau_hat = np.array([tau_from_pole(pole, data["scene"]["delta_f"]) for pole in estimate["poles"]])
    ranges = np.asarray(estimate["ris_eta"], dtype=float)[:, 0]
    raw_residual = y_hat - data["Y_noisy"]
    final = {
        "Y_hat": y_hat,
        "p_u": p_hat,
        "components": {"taus": tau_hat, "ranges": ranges},
        "raw_objective_final": float(np.vdot(raw_residual.reshape(-1), raw_residual.reshape(-1)).real / data["Y_noisy"].size),
        "final_refinement_method": "stage1_only",
        "vp_mode": "none",
        "global_vp_mode": "none",
        "linear_nuisance_dim": 0,
        "nonlinear_dim": 0,
    }
    return {
        **data,
        "estimate_initial": estimate,
        "estimate_used": estimate,
        "final": final,
        "timing": {
            **data.get("timing", {}),
            **stage1["timing"],
            "stage2": 0.0,
            "vp": 0.0,
            "total": float(sum(data.get("timing", {}).values()) + stage1["timing"].get("stage1", 0.0)),
        },
        "reliability": {},
        "selected_branch": "stage1_only",
    }


def _oracle_result(config: dict) -> dict:
    data = _make_data(config)
    init = _truth_init_estimate(data["scene"], data["true_components"])
    vp_start = time.perf_counter()
    final = global_exact_spherical_vp_refinement(data["Y_noisy"], init, data["scene"], config)
    vp_s = time.perf_counter() - vp_start
    final["selected_branch"] = "oracle_init_vp"
    final["final_refinement_method"] = "global_exact_spherical_vp"
    return {
        **data,
        "estimate_initial": init,
        "estimate_used": init,
        "final": final,
        "timing": {**data.get("timing", {}), "stage1": 0.0, "stage2": 0.0, "vp": vp_s, "total": vp_s},
        "reliability": {"decision": "oracle_init_vp", "trigger_reasons": ["oracle_truth_initialization"]},
        "selected_branch": "oracle_init_vp",
    }


def _stage1_only_result_from_shared(data: dict, stage1: dict, config: dict) -> dict:
    estimate = copy.deepcopy(stage1["estimate"])
    y_hat = reconstruct_raw_tensor_from_structured_estimate(estimate, data["scene"])
    try:
        p_hat = estimate_position_from_local_ris(data["scene"], estimate, config)
    except (KeyError, ValueError, np.linalg.LinAlgError):
        p_hat = estimate_position_from_ris_eta(data["scene"], estimate)
    tau_hat = np.array([tau_from_pole(pole, data["scene"]["delta_f"]) for pole in estimate["poles"]])
    ranges = np.asarray(estimate["ris_eta"], dtype=float)[:, 0]
    raw_residual = y_hat - data["Y_noisy"]
    final = {
        "Y_hat": y_hat,
        "p_u": p_hat,
        "components": {"taus": tau_hat, "ranges": ranges},
        "raw_objective_final": float(
            np.vdot(raw_residual.reshape(-1), raw_residual.reshape(-1)).real
            / data["Y_noisy"].size
        ),
        "final_refinement_method": "stage1_only",
        "vp_mode": "none",
        "global_vp_mode": "none",
        "linear_nuisance_dim": 0,
        "nonlinear_dim": 0,
    }
    return {
        **data,
        "estimate_initial": estimate,
        "estimate_used": estimate,
        "final": final,
        "timing": {
            **data.get("timing", {}),
            **stage1["timing"],
            "stage2": 0.0,
            "vp": 0.0,
            "total": float(
                sum(data.get("timing", {}).values()) + stage1["timing"].get("stage1", 0.0)
            ),
        },
        "reliability": {},
        "selected_branch": "stage1_only",
    }


def _oracle_result_from_shared(data: dict, config: dict) -> dict:
    init = _truth_init_estimate(data["scene"], data["true_components"])
    vp_start = time.perf_counter()
    final = global_exact_spherical_vp_refinement(data["Y_noisy"], init, data["scene"], config)
    vp_s = time.perf_counter() - vp_start
    final["selected_branch"] = "oracle_init_vp"
    final["final_refinement_method"] = "global_exact_spherical_vp"
    return {
        **data,
        "estimate_initial": init,
        "estimate_used": init,
        "final": final,
        "timing": {**data.get("timing", {}), "stage1": 0.0, "stage2": 0.0, "vp": vp_s, "total": vp_s},
        "reliability": {"decision": "oracle_init_vp", "trigger_reasons": ["oracle_truth_initialization"]},
        "selected_branch": "oracle_init_vp",
    }


def run_stage1_shared_trial(data: dict, config: dict) -> tuple[dict, dict, dict]:
    stage1 = run_stage1_only(data, config)
    reliability_base: dict[str, Any] = {}
    return stage1["estimate"], stage1["timing"], reliability_base


def _result_to_row(
    result: dict,
    *,
    figure: str,
    variant: str,
    trial_id: int,
    trial_seed: int,
    snr_db: float,
    x_name: str,
    x_value: float,
    outlier_threshold_m: float,
    runtime_s: float,
    store_large_arrays: bool,
    rss_before: float,
    profile_memory: bool,
    compact_result: bool = True,
) -> dict[str, Any]:
    row = _empty_row()
    row.update(
        {
            "figure": figure,
            "variant": variant,
            "trial_id": int(trial_id),
            "seed": int(trial_seed),
            "snr_db": float(snr_db),
            "x_name": x_name,
            "x_value": float(x_value),
            "failed": False,
            "error": "",
        }
    )
    row.update(extract_metrics(result, outlier_threshold_m))
    if compact_result:
        compact_experiment_result(result, keep_large_arrays=store_large_arrays)
    rss_after = _rss_mb() if profile_memory else float("nan")
    row["rss_mb_before"] = rss_before
    row["rss_mb_after"] = rss_after
    row["rss_mb_delta"] = (
        rss_after - rss_before
        if np.isfinite(rss_before) and np.isfinite(rss_after)
        else float("nan")
    )
    row["runtime_s"] = float(runtime_s)
    return row


def _failure_row_from_payload(
    *,
    figure: str,
    variant: str,
    trial_id: int,
    trial_seed: int,
    snr_db: float,
    x_name: str,
    x_value: float,
    k_paths: int,
    receiver_mode: str,
    exc: BaseException,
) -> dict[str, Any]:
    row = _empty_row()
    row.update(
        {
            "figure": figure,
            "variant": variant,
            "trial_id": int(trial_id),
            "seed": int(trial_seed),
            "snr_db": float(snr_db),
            "x_name": x_name,
            "x_value": float(x_value),
            "K": int(k_paths),
            "receiver_mode": receiver_mode,
            "failed": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    return row


def run_final_vp_from_shared_stage1(
    data: dict,
    stage1: dict,
    config: dict,
    variant_spec: dict[str, Any],
    allow_stage2: bool,
) -> dict:
    _ = variant_spec
    return run_from_existing_stage1(
        data,
        {"estimate": copy.deepcopy(stage1["estimate"]), "timing": dict(stage1["timing"])},
        config,
        allow_stage2=allow_stage2,
    )


def _peb_result(config: dict, out_dir: pathlib.Path | None = None) -> dict:
    key = peb_cache_key(config)
    cached = _PEB_CACHE.get(key)
    if cached is None:
        cached = _read_persistent_peb_cache(config, out_dir)
        if cached is not None:
            _PEB_CACHE[key] = copy.deepcopy(cached)
    if cached is not None:
        return {
            "scene": {
                "K": int(config.get("K", "")),
                "receiver_mode": str(
                    config.get("receiver_mode", config.get("evs_selection", "full_6d"))
                ),
            },
            **copy.deepcopy(cached),
            "final": {},
            "timing": {},
        }
    data = _make_data(config)
    peb_metrics = _peb_from_efim(data, config)
    _PEB_CACHE[key] = copy.deepcopy(peb_metrics)
    _write_persistent_peb_cache(config, out_dir, peb_metrics)
    return {**data, **peb_metrics, "final": {}, "timing": data.get("timing", {})}


def run_one_trial(
    config: dict,
    trial_seed: int,
    variant_name: str,
    figure_name: str,
    *,
    trial_id: int,
    x_name: str,
    x_value: float,
    outlier_threshold_m: float,
    verbose: bool = False,
    runner: str = "proposed",
    allow_stage2: bool = True,
    store_large_arrays: bool = False,
    profile_memory: bool = False,
    blas_threads: int = DEFAULT_BLAS_THREADS,
    out_dir: pathlib.Path | None = None,
) -> tuple[dict[str, Any], str]:
    _apply_blas_thread_env(blas_threads)
    config = copy.deepcopy(config)
    config["seed"] = int(trial_seed)
    config["experiment"] = dict(config.get("experiment", {}))
    config["experiment"]["store_large_arrays"] = bool(store_large_arrays)
    log_buffer = io.StringIO()
    start = time.perf_counter()
    rss_before = _rss_mb() if profile_memory else float("nan")
    row = _empty_row()
    row.update(
        {
            "figure": figure_name,
            "variant": variant_name,
            "trial_id": trial_id,
            "seed": int(trial_seed),
            "snr_db": float(config["SNR_dB"]),
            "x_name": x_name,
            "x_value": float(x_value),
            "failed": False,
            "error": "",
        }
    )
    try:
        stream = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(log_buffer)
        with stream:
            if runner == "stage1_only":
                result = _stage1_only_result(config)
            elif runner == "oracle_init_vp":
                result = _oracle_result(config)
            elif runner == "peb_only":
                result = _peb_result(config, out_dir)
            else:
                result = run_single_proposed_diagnostic(config, allow_stage2=allow_stage2)
        row.update(extract_metrics(result, outlier_threshold_m))
        compact_experiment_result(result, keep_large_arrays=store_large_arrays)
        del result
    except Exception as exc:  # noqa: BLE001 - failed trials must be logged as rows.
        row["failed"] = True
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["K"] = config.get("K", "")
        row["receiver_mode"] = config.get("receiver_mode", "full_6d")
        log_buffer.write(f"\nERROR {type(exc).__name__}: {exc}\n")
    gc.collect()
    rss_after = _rss_mb() if profile_memory else float("nan")
    row["rss_mb_before"] = rss_before
    row["rss_mb_after"] = rss_after
    row["rss_mb_delta"] = (
        rss_after - rss_before
        if np.isfinite(rss_before) and np.isfinite(rss_after)
        else float("nan")
    )
    row["runtime_s"] = float(time.perf_counter() - start)
    return row, log_buffer.getvalue()


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_atomic_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with StreamingCsvWriter(path, fieldnames) as writer:
        for row in rows:
            writer.writerow(row)


class StreamingCsvWriter:
    """Write trial rows to a temporary CSV and atomically publish on completion."""

    def __init__(self, final_path: pathlib.Path, fieldnames: list[str]):
        self.final_path = pathlib.Path(final_path)
        self.tmp_path = self.final_path.with_name(f"{self.final_path.name}.tmp")
        self.fieldnames = fieldnames
        self.handle: Any | None = None
        self.writer: csv.DictWriter | None = None

    def __enter__(self) -> "StreamingCsvWriter":
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.tmp_path.open("w", newline="")
        self.writer = csv.DictWriter(
            self.handle, fieldnames=self.fieldnames, extrasaction="ignore"
        )
        self.writer.writeheader()
        self.handle.flush()
        return self

    def writerow(self, row: dict[str, Any]) -> None:
        if self.writer is None or self.handle is None:
            raise RuntimeError("StreamingCsvWriter is not open")
        self.writer.writerow(row)
        self.handle.flush()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.handle is not None:
            self.handle.close()
        if exc_type is None:
            os.replace(self.tmp_path, self.final_path)
        return False


def _read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_metric_values(group: list[dict[str, Any]], metric: str) -> np.ndarray:
    values = np.asarray(
        [
            _to_float(row.get(metric))
            for row in group
            if str(row.get("failed")) != "True"
        ],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _metric_available(group: list[dict[str, Any]], metric: str) -> bool:
    return bool(_finite_metric_values(group, metric).size)


def get_plot_metric(row_or_group: dict[str, Any] | list[dict[str, Any]], figure: str, variant: str) -> str | None:
    """Return the source metric column to summarize and plot for a variant."""
    group = row_or_group if isinstance(row_or_group, list) else [row_or_group]
    if figure == "fig1":
        return "peb_position_m" if variant == "PEB" else "position_rmse_m"
    if figure == "fig2":
        return None if "peb" in variant.lower() else "y_nmse"
    if figure == "fig3":
        return "peb_position_m" if variant == "full_6d_evs_peb" else "position_rmse_m"
    if figure == "fig4":
        preferred = {
            "scalar_peb": "peb_scalar_m",
            "dual_pol_peb": "peb_dual_m",
            "full_6d_evs_peb": "peb_evs_m",
        }.get(variant, "peb_position_m")
        return preferred if _metric_available(group, preferred) else "peb_position_m"
    if figure == "fig5":
        return "outlier_flag"
    if figure == "fig6":
        return "peb_position_m" if variant == "proposed_peb" else "position_rmse_m"
    raise ValueError(f"unknown figure {figure!r}")


def _plot_metric_name(metric: str) -> str:
    return "outlier_flag_mean" if metric == "outlier_flag" else metric


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p10": float(np.percentile(values, 10.0)),
            "p90": float(np.percentile(values, 90.0)),
        }
    return {name: float("nan") for name in ("mean", "median", "std", "p10", "p90")}


def summarize_rows(rows: list[dict[str, Any]], figure: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["variant"]), _to_float(row["x_value"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (variant, x_value), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        metric = get_plot_metric(group, figure, variant)
        if metric is None:
            continue
        values = _finite_metric_values(group, metric)
        failed_count = sum(str(row.get("failed")) == "True" for row in group)
        outliers = np.asarray([str(row.get("outlier_flag")) == "True" for row in group], dtype=float)
        stats = _summary_stats(values)
        row_summary = {
            "figure": figure,
            "variant": variant,
            "x_value": x_value,
            "metric": metric,
            "plot_metric_name": _plot_metric_name(metric),
            "plot_y_mean": stats["mean"],
            "plot_y_median": stats["median"],
            "plot_y_std": stats["std"],
            "plot_y_p10": stats["p10"],
            "plot_y_p90": stats["p90"],
            **stats,
            "success_rate": float((len(group) - failed_count) / max(len(group), 1)),
            "outlier_rate": float(np.mean(outliers)) if outliers.size else float("nan"),
            "n": len(group),
        }
        for raw_metric in RAW_SUMMARY_METRICS:
            raw_stats = _summary_stats(_finite_metric_values(group, raw_metric))
            for name, value in raw_stats.items():
                row_summary[f"{raw_metric}_{name}"] = value
        summary.append(row_summary)
    return summary


def summarize_fig1_fig2_shared_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shared_summary: list[dict[str, Any]] = []
    for plot_name in ("fig1", "fig2"):
        for row in summarize_rows(rows, plot_name):
            shared_summary.append({"plot_name": plot_name, **row})
    return shared_summary


def _plot_figure(figure: str, summary_rows: list[dict[str, Any]], out_dir: pathlib.Path) -> None:
    mpl_config = out_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib.pyplot as plt

    metric = FIGURE_METRICS[figure]
    ylabel = {
        "position_rmse_m": "Position RMSE (m)",
        "y_nmse": "Channel NMSE",
        "peb_position_m": "PEB (m)",
        "outlier_flag": "Outlier probability",
    }[metric]
    xlabel = "K" if figure == "fig6" else "SNR (dB)"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    markers = ["o", "s", "^", "D", "v", "P"]
    variants = list(dict.fromkeys(row["variant"] for row in summary_rows))
    for idx, variant in enumerate(variants):
        rows = [row for row in summary_rows if row["variant"] == variant]
        xs = np.asarray([_to_float(row["x_value"]) for row in rows], dtype=float)
        ys = np.asarray([_to_float(row["plot_y_mean"]) for row in rows], dtype=float)
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker=markers[idx % len(markers)], linewidth=1.5, label=variant)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if figure == "fig5":
        ax.set_ylim(-0.02, 1.02)
    else:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / FIGURE_PDFS[figure])
    plt.close(fig)


def _cache_signature(args: argparse.Namespace, snr_grid: list[float], figures: list[str]) -> dict[str, Any]:
    return {
        "n_trials": int(args.n_trials),
        "snr_grid": [float(value) for value in snr_grid],
        "paper_k": int(args.paper_k),
        "k_grid": [int(value) for value in args.k_grid_values],
        "figures": list(figures),
        "seed": int(args.seed),
        "variant_list": {figure: _expected_variant_names(figure) for figure in figures},
        "git_commit": _git_commit_hash(),
        "receiver_noise_convention": RECEIVER_NOISE_CONVENTION,
        "receiver_mode_convention": RECEIVER_MODE_CONVENTION,
    }


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _metadata(args: argparse.Namespace, snr_grid: list[float], figures: list[str]) -> dict[str, Any]:
    commit = _git_commit_hash()
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "git_commit": commit,
        "timestamp": timestamp,
        "timestamp_utc": timestamp,
        "command_line": " ".join(sys.argv),
        "n_trials": int(args.n_trials),
        "jobs": int(args.jobs),
        "max_workers": int(args.jobs),
        "maxtasksperchild": int(args.maxtasksperchild),
        "task_grouping": str(args.task_grouping),
        "streaming_csv": bool(args.streaming_csv),
        "store_large_arrays": bool(args.store_large_arrays),
        "blas_threads": int(args.blas_threads),
        "profile_memory": bool(args.profile_memory),
        "snr_grid": snr_grid,
        "paper_k": int(args.paper_k),
        "k_grid": [int(value) for value in args.k_grid_values],
        "figures": figures,
        "seed": int(args.seed),
        "receiver_noise_convention": RECEIVER_NOISE_CONVENTION,
        "receiver_mode_convention": RECEIVER_MODE_CONVENTION,
        "config_overrides": {
            "seed": int(args.seed),
            "outlier_threshold_m": float(args.outlier_threshold_m),
            "paper_k_for_fig1_to_fig5": int(args.paper_k),
            "k_grid_for_fig6": [int(value) for value in args.k_grid_values],
        },
        "shared_cache_signatures": {
            FIG1_FIG2_SHARED_FIGURE: _cache_signature(
                args, snr_grid, [FIG1_FIG2_SHARED_FIGURE]
            )
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy_available": bool(scipy_is_available()),
    }


def _read_metadata(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _metadata_matches_request(
    metadata: dict[str, Any] | None,
    args: argparse.Namespace,
    snr_grid: list[float],
    figures: list[str],
) -> bool:
    if not metadata:
        return False
    expected = _cache_signature(args, snr_grid, figures)
    if figures == [FIG1_FIG2_SHARED_FIGURE]:
        shared = metadata.get("shared_cache_signatures", {})
        if shared.get(FIG1_FIG2_SHARED_FIGURE) == expected:
            return True
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            return False
    return True


def _expected_variant_names(figure: str) -> list[str]:
    variants = {
        **_variant_specs(FIG1_FIG2_SHARED_FIGURE if _is_fig1_fig2(figure) else figure),
        **_extra_peb_specs(figure),
    }
    return list(variants)


def _csv_matches_request(
    rows: list[dict[str, Any]],
    figure: str,
    args: argparse.Namespace,
    snr_grid: list[float],
) -> bool:
    if not rows:
        return False
    canonical_figure = FIG1_FIG2_SHARED_FIGURE if _is_fig1_fig2(figure) else figure
    x_name, x_values = _figure_x_grid(canonical_figure, snr_grid, args.k_grid_values)
    expected_variants = set(_expected_variant_names(figure))
    expected_x_values = {float(value) for value in x_values}
    groups: dict[tuple[str, float], int] = {}
    for row in rows:
        row_figure = row.get("figure")
        if _is_fig1_fig2(figure):
            if row_figure not in {"fig1", "fig2", FIG1_FIG2_SHARED_FIGURE}:
                return False
        elif row_figure != figure:
            return False
        variant = str(row.get("variant"))
        x_value = _to_float(row.get("x_value"))
        if variant not in expected_variants or x_value not in expected_x_values:
            return False
        row_k_value = _to_float(row.get("K"))
        row_k = int(row_k_value) if np.isfinite(row_k_value) else None
        expected_k = int(x_value) if canonical_figure == "fig6" else int(args.paper_k)
        if row_k != expected_k:
            return False
        if row.get("x_name") != x_name:
            return False
        groups[(variant, x_value)] = groups.get((variant, x_value), 0) + 1
    expected_groups = {
        (variant, x_value)
        for variant in expected_variants
        for x_value in expected_x_values
    }
    return set(groups) == expected_groups and all(
        count == int(args.n_trials) for count in groups.values()
    )


def _can_reuse_csv(
    trial_csv: pathlib.Path,
    figure: str,
    args: argparse.Namespace,
    snr_grid: list[float],
    figures: list[str],
    existing_metadata: dict[str, Any] | None,
) -> tuple[bool, list[dict[str, Any]]]:
    if not trial_csv.exists() or args.force_rerun:
        return False, []
    rows = _read_csv(trial_csv)
    if args.reuse_existing:
        return True, rows
    metadata_ok = _metadata_matches_request(existing_metadata, args, snr_grid, figures)
    csv_ok = _csv_matches_request(rows, figure, args, snr_grid)
    return bool(metadata_ok and csv_ok), rows


def _figure_x_grid(figure: str, snr_grid: list[float], k_grid: list[int]) -> tuple[str, list[float]]:
    if figure == "fig6":
        return "K", [float(k) for k in k_grid]
    return "snr_db", snr_grid


def _config_for_point(
    *,
    figure: str,
    variant_updates: dict[str, Any],
    seed: int,
    snr_db: float,
    x_value: float,
    paper_k: int,
) -> dict:
    overrides = copy.deepcopy(variant_updates)
    if figure == "fig6":
        overrides["K"] = int(x_value)
        snr_db = 0.0
    else:
        overrides["K"] = int(paper_k)
    return make_base_config(seed, snr_db, overrides)


def _failure_row_from_task(task: dict[str, Any], exc: BaseException) -> tuple[list[dict[str, Any]], str]:
    row = _empty_row()
    row.update(
        {
            "figure": task["figure"],
            "variant": task["variant"],
            "trial_id": int(task["trial_id"]),
            "seed": int(task["trial_seed"]),
            "snr_db": float(task["snr_db"]),
            "x_name": task["x_name"],
            "x_value": float(task["x_value"]),
            "failed": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    return [row], f"\nERROR {type(exc).__name__}: {exc}\n"


def _run_trial_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    try:
        k_paths = int(task["x_value"]) if task["figure"] == "fig6" else int(task["paper_k"])
        config = _config_from_base(
            figure=task["figure"],
            seed=task["trial_seed"],
            snr_db=task["snr_db"],
            k_paths=k_paths,
            updates=task["updates"],
        )
        row, log_text = run_one_trial(
            config,
            task["trial_seed"],
            task["variant"],
            task["figure"],
            trial_id=task["trial_id"],
            x_name=task["x_name"],
            x_value=task["x_value"],
            outlier_threshold_m=task["outlier_threshold_m"],
            verbose=task["verbose"],
            runner=task["runner"],
            allow_stage2=task["allow_stage2"],
            store_large_arrays=task["store_large_arrays"],
            profile_memory=task["profile_memory"],
            blas_threads=task["blas_threads"],
            out_dir=pathlib.Path(task["out_dir"]) if task.get("out_dir") else None,
        )
        return [row], log_text
    except Exception as exc:  # noqa: BLE001 - preserve failed-trial logging.
        return _failure_row_from_task(task, exc)


def _trial_tasks(
    *,
    figure: str,
    x_name: str,
    x_values: list[float],
    variants: dict[str, dict[str, Any]],
    trial_seeds: list[int],
    outlier_threshold_m: float,
    verbose: bool,
    paper_k: int,
    store_large_arrays: bool,
    profile_memory: bool,
    blas_threads: int,
    out_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    tasks = []
    for x_value in x_values:
        snr_db = float(x_value) if x_name == "snr_db" else 0.0
        for variant, updates in variants.items():
            for trial_id, trial_seed in enumerate(trial_seeds):
                tasks.append(
                    {
                        "figure": figure,
                        "variant": variant,
                        "updates": updates,
                        "trial_id": trial_id,
                        "trial_seed": int(trial_seed),
                        "snr_db": snr_db,
                        "x_name": x_name,
                        "x_value": float(x_value),
                        "paper_k": int(paper_k),
                        "outlier_threshold_m": float(outlier_threshold_m),
                        "verbose": bool(verbose),
                        "runner": str(updates.get("_runner", "proposed")),
                        "allow_stage2": bool(updates.get("_allow_stage2", True)),
                        "store_large_arrays": bool(store_large_arrays),
                        "profile_memory": bool(profile_memory),
                        "blas_threads": int(blas_threads),
                        "out_dir": str(out_dir),
                    }
                )
    return tasks


def _grouped_tasks(
    *,
    figure: str,
    group: str,
    x_name: str,
    x_values: list[float],
    trial_seeds: list[int],
    outlier_threshold_m: float,
    paper_k: int,
    store_large_arrays: bool,
    profile_memory: bool,
    blas_threads: int,
    out_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    tasks = []
    for x_value in x_values:
        snr_db = float(x_value) if x_name == "snr_db" else 0.0
        k_paths = int(x_value) if figure == "fig6" else int(paper_k)
        for trial_id, trial_seed in enumerate(trial_seeds):
            tasks.append(
                {
                    "task_kind": "grouped",
                    "figure": figure,
                    "group": group,
                    "trial_id": int(trial_id),
                    "trial_seed": int(trial_seed),
                    "snr_db": float(snr_db),
                    "x_name": x_name,
                    "x_value": float(x_value),
                    "K": int(k_paths),
                    "outlier_threshold_m": float(outlier_threshold_m),
                    "store_large_arrays": bool(store_large_arrays),
                    "profile_memory": bool(profile_memory),
                    "blas_threads": int(blas_threads),
                    "out_dir": str(out_dir),
                }
            )
    return tasks


def _tasks_for_figure(
    *,
    figure: str,
    x_name: str,
    x_values: list[float],
    variants: dict[str, dict[str, Any]],
    trial_seeds: list[int],
    args: argparse.Namespace,
    grouped_group: str | None = None,
) -> list[dict[str, Any]]:
    if args.task_grouping == "grouped" and grouped_group is not None:
        return _grouped_tasks(
            figure=figure,
            group=grouped_group,
            x_name=x_name,
            x_values=x_values,
            trial_seeds=trial_seeds,
            outlier_threshold_m=float(args.outlier_threshold_m),
            paper_k=int(args.paper_k),
            store_large_arrays=bool(args.store_large_arrays),
            profile_memory=bool(args.profile_memory),
            blas_threads=int(args.blas_threads),
            out_dir=pathlib.Path(args.out_dir),
        )
    tasks = _trial_tasks(
        figure=figure,
        x_name=x_name,
        x_values=x_values,
        variants=variants,
        trial_seeds=trial_seeds,
        outlier_threshold_m=float(args.outlier_threshold_m),
        verbose=bool(args.verbose),
        paper_k=int(args.paper_k),
        store_large_arrays=bool(args.store_large_arrays),
        profile_memory=bool(args.profile_memory),
        blas_threads=int(args.blas_threads),
        out_dir=pathlib.Path(args.out_dir),
    )
    if args.task_grouping == "grouped" and grouped_group is None and figure == "fig3":
        for task in tasks:
            task["task_grouping_warning"] = (
                "Fig.3 receiver modes change observation masks/noise; using variant tasks."
            )
    return tasks


def _init_worker(base_config: dict, out_dir: str, blas_threads: int) -> None:
    global _WORKER_BASE_CONFIG, _WORKER_OUT_DIR, _WORKER_BLAS_THREADS
    _WORKER_BASE_CONFIG = base_config
    _WORKER_OUT_DIR = pathlib.Path(out_dir)
    _WORKER_BLAS_THREADS = int(blas_threads)
    _apply_blas_thread_env(_WORKER_BLAS_THREADS)
    cache_path = _peb_cache_path(_WORKER_OUT_DIR)
    if cache_path is not None:
        _init_peb_cache(cache_path)


def _base_worker_config() -> dict:
    if _WORKER_BASE_CONFIG is not None:
        return copy.deepcopy(_WORKER_BASE_CONFIG)
    return default_config()


def _config_from_base(
    *,
    figure: str,
    seed: int,
    snr_db: float,
    k_paths: int,
    updates: dict[str, Any] | None = None,
) -> dict:
    config = copy.deepcopy(_base_worker_config())
    config["seed"] = int(seed)
    config["SNR_dB"] = float(0.0 if figure == "fig6" else snr_db)
    config["print_progress"] = False
    config["verbose_stage2"] = False
    config["run_full_legacy_comparison"] = False
    config = apply_nested_update(config, updates or {})
    set_number_of_ris_paths(config, int(k_paths))
    return _apply_main_single_defaults(config)


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            clean[key] = _vector_string(value) if value.size <= 16 else ""
        elif isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def _peb_metrics_result_for_config(
    config: dict,
    out_dir: pathlib.Path | None,
    data: dict | None = None,
) -> dict:
    key = peb_cache_key(config)
    cached = _PEB_CACHE.get(key)
    if cached is None:
        cached = _read_persistent_peb_cache(config, out_dir)
        if cached is not None:
            _PEB_CACHE[key] = copy.deepcopy(cached)
    if cached is None:
        if data is None:
            result = _peb_result(config, out_dir)
            return result
        cached = _peb_from_efim(data, config)
        _PEB_CACHE[key] = copy.deepcopy(cached)
        _write_persistent_peb_cache(config, out_dir, cached)
    return {
        "scene": {
            "K": int(config.get("K", "")),
            "receiver_mode": str(config.get("receiver_mode", config.get("evs_selection", "full_6d"))),
        },
        **copy.deepcopy(cached),
        "final": {},
        "timing": {},
    }


def _row_for_result_or_failure(
    *,
    result_factory,
    figure: str,
    variant: str,
    trial_id: int,
    trial_seed: int,
    snr_db: float,
    x_name: str,
    x_value: float,
    k_paths: int,
    receiver_mode: str,
    outlier_threshold_m: float,
    store_large_arrays: bool,
    profile_memory: bool,
) -> tuple[dict[str, Any], str]:
    start = time.perf_counter()
    rss_before = _rss_mb() if profile_memory else float("nan")
    try:
        result = result_factory()
        row = _result_to_row(
            result,
            figure=figure,
            variant=variant,
            trial_id=trial_id,
            trial_seed=trial_seed,
            snr_db=snr_db,
            x_name=x_name,
            x_value=x_value,
            outlier_threshold_m=outlier_threshold_m,
            runtime_s=time.perf_counter() - start,
            store_large_arrays=store_large_arrays,
            rss_before=rss_before,
            profile_memory=profile_memory,
            compact_result=False,
        )
        del result
        return _sanitize_row(row), ""
    except Exception as exc:  # noqa: BLE001 - failed variants must be logged as rows.
        row = _failure_row_from_payload(
            figure=figure,
            variant=variant,
            trial_id=trial_id,
            trial_seed=trial_seed,
            snr_db=snr_db,
            x_name=x_name,
            x_value=x_value,
            k_paths=k_paths,
            receiver_mode=receiver_mode,
            exc=exc,
        )
        return _sanitize_row(row), f"\nERROR {variant} {type(exc).__name__}: {exc}\n"


def _run_grouped_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    _apply_blas_thread_env(int(task.get("blas_threads", _WORKER_BLAS_THREADS)))
    figure = str(task["figure"])
    group = str(task["group"])
    trial_id = int(task["trial_id"])
    trial_seed = int(task["trial_seed"])
    snr_db = float(task["snr_db"])
    x_name = str(task["x_name"])
    x_value = float(task["x_value"])
    k_paths = int(task["K"])
    out_dir = pathlib.Path(task["out_dir"]) if task.get("out_dir") else _WORKER_OUT_DIR
    outlier_threshold_m = float(task["outlier_threshold_m"])
    store_large_arrays = bool(task["store_large_arrays"])
    profile_memory = bool(task["profile_memory"])
    rows: list[dict[str, Any]] = []
    logs: list[str] = []
    data = None
    stage1 = None
    try:
        base_config = _config_from_base(
            figure=figure,
            seed=trial_seed,
            snr_db=snr_db,
            k_paths=k_paths,
        )
        receiver_mode = str(base_config.get("receiver_mode", base_config.get("evs_selection", "full_6d")))
        if group in {"fig1_fig2", "fig5", "fig6"}:
            data = _make_data(base_config)
            if group != "fig4":
                stage1 = run_stage1_only(data, base_config)
        if group == "fig1_fig2":
            variants = _variant_specs(FIG1_FIG2_SHARED_FIGURE)
            for variant, updates in variants.items():
                config = apply_nested_update(copy.deepcopy(base_config), updates)
                allow_stage2 = bool(updates.get("_allow_stage2", True))
                runner = str(updates.get("_runner", "proposed"))
                if runner == "stage1_only":
                    factory = lambda data=data, stage1=stage1, config=config: _stage1_only_result_from_shared(data, stage1, config)
                else:
                    factory = lambda data=data, stage1=stage1, config=config, updates=updates, allow_stage2=allow_stage2: run_final_vp_from_shared_stage1(
                        data, stage1, config, updates, allow_stage2
                    )
                row, log = _row_for_result_or_failure(
                    result_factory=factory,
                    figure=figure,
                    variant=variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=snr_db,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    receiver_mode=receiver_mode,
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                )
                rows.append(row)
                if log:
                    logs.append(log)
            peb_config = apply_nested_update(copy.deepcopy(base_config), _extra_peb_specs(FIG1_FIG2_SHARED_FIGURE)["PEB"])
            row, log = _row_for_result_or_failure(
                result_factory=lambda config=peb_config, data=data: _peb_metrics_result_for_config(config, out_dir, data),
                figure=figure,
                variant="PEB",
                trial_id=trial_id,
                trial_seed=trial_seed,
                snr_db=snr_db,
                x_name=x_name,
                x_value=x_value,
                k_paths=k_paths,
                receiver_mode="full_6d",
                outlier_threshold_m=outlier_threshold_m,
                store_large_arrays=store_large_arrays,
                profile_memory=profile_memory,
            )
            rows.append(row)
            if log:
                logs.append(log)
        elif group == "fig5":
            for variant, updates in _variant_specs("fig5").items():
                config = apply_nested_update(copy.deepcopy(base_config), updates)
                allow_stage2 = bool(updates.get("_allow_stage2", True))
                runner = str(updates.get("_runner", "proposed"))
                if runner == "oracle_init_vp":
                    factory = lambda data=data, config=config: _oracle_result_from_shared(data, config)
                else:
                    factory = lambda data=data, stage1=stage1, config=config, updates=updates, allow_stage2=allow_stage2: run_final_vp_from_shared_stage1(
                        data, stage1, config, updates, allow_stage2
                    )
                row, log = _row_for_result_or_failure(
                    result_factory=factory,
                    figure=figure,
                    variant=variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=snr_db,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    receiver_mode=receiver_mode,
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                )
                rows.append(row)
                if log:
                    logs.append(log)
        elif group == "fig6":
            for variant, updates in _variant_specs("fig6").items():
                if str(updates.get("_runner", "proposed")) == "peb_only":
                    continue
                config = apply_nested_update(copy.deepcopy(base_config), updates)
                row, log = _row_for_result_or_failure(
                    result_factory=lambda data=data, stage1=stage1, config=config, updates=updates: run_final_vp_from_shared_stage1(
                        data, stage1, config, updates, bool(updates.get("_allow_stage2", True))
                    ),
                    figure=figure,
                    variant=variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=0.0,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    receiver_mode=receiver_mode,
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                )
                rows.append(row)
                if log:
                    logs.append(log)
            peb_config = apply_nested_update(copy.deepcopy(base_config), _variant_specs("fig6")["proposed_peb"])
            row, log = _row_for_result_or_failure(
                result_factory=lambda config=peb_config, data=data: _peb_metrics_result_for_config(config, out_dir, data),
                figure=figure,
                variant="proposed_peb",
                trial_id=trial_id,
                trial_seed=trial_seed,
                snr_db=0.0,
                x_name=x_name,
                x_value=x_value,
                k_paths=k_paths,
                receiver_mode=receiver_mode,
                outlier_threshold_m=outlier_threshold_m,
                store_large_arrays=store_large_arrays,
                profile_memory=profile_memory,
            )
            rows.append(row)
            if log:
                logs.append(log)
        elif group == "fig4":
            for variant, updates in _variant_specs("fig4").items():
                config = _config_from_base(
                    figure=figure,
                    seed=trial_seed,
                    snr_db=snr_db,
                    k_paths=k_paths,
                    updates=updates,
                )
                row, log = _row_for_result_or_failure(
                    result_factory=lambda config=config: _peb_result(config, out_dir),
                    figure=figure,
                    variant=variant,
                    trial_id=trial_id,
                    trial_seed=trial_seed,
                    snr_db=snr_db,
                    x_name=x_name,
                    x_value=x_value,
                    k_paths=k_paths,
                    receiver_mode=str(config.get("receiver_mode", "full_6d")),
                    outlier_threshold_m=outlier_threshold_m,
                    store_large_arrays=store_large_arrays,
                    profile_memory=profile_memory,
                )
                rows.append(row)
                if log:
                    logs.append(log)
        else:
            raise ValueError(f"unsupported grouped task group {group!r}")
    finally:
        del data, stage1
        gc.collect()
    return rows, "".join(logs)


def _iter_task_results(
    tasks: list[dict[str, Any]],
    *,
    jobs: int,
    maxtasksperchild: int,
    base_config: dict,
    out_dir: pathlib.Path,
    blas_threads: int,
) -> Iterable[tuple[list[dict[str, Any]], str]]:
    if jobs == 1:
        _init_worker(base_config, str(out_dir), blas_threads)
        for task in tasks:
            if task.get("task_kind") == "grouped":
                yield _run_grouped_task(task)
            else:
                yield _run_trial_task(task)
        return
    processes = min(int(jobs), max(len(tasks), 1))
    worker = _run_grouped_task if tasks and tasks[0].get("task_kind") == "grouped" else _run_trial_task
    with mp.Pool(
        processes=processes,
        maxtasksperchild=int(maxtasksperchild),
        initializer=_init_worker,
        initargs=(base_config, str(out_dir), int(blas_threads)),
    ) as pool:
        for result in pool.imap_unordered(worker, tasks, chunksize=1):
            yield result


def _write_trial_results(
    trial_csv: pathlib.Path,
    tasks: list[dict[str, Any]],
    log_path: pathlib.Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    jobs = min(int(args.jobs), max(len(tasks), 1))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_config = default_config()
    out_dir = pathlib.Path(args.out_dir)
    writer_context = (
        StreamingCsvWriter(trial_csv, FIELDNAMES)
        if args.streaming_csv
        else contextlib.nullcontext()
    )
    buffered_rows: list[dict[str, Any]] = []
    with log_path.open("w") as log_handle, writer_context as csv_writer:
        log_handle.write(f"parallel_jobs={jobs}\n")
        log_handle.write(f"task_grouping={args.task_grouping}\n")
        log_handle.write(f"streaming_csv={bool(args.streaming_csv)}\n")
        log_handle.write(f"store_large_arrays={bool(args.store_large_arrays)}\n")
        if tasks and tasks[0].get("task_grouping_warning"):
            log_handle.write(f"WARNING: {tasks[0]['task_grouping_warning']}\n")
        for row_batch, log_text in _iter_task_results(
            tasks,
            jobs=jobs,
            maxtasksperchild=int(args.maxtasksperchild),
            base_config=base_config,
            out_dir=out_dir,
            blas_threads=int(args.blas_threads),
        ):
            for row in row_batch:
                if args.streaming_csv:
                    csv_writer.writerow(row)
                else:
                    buffered_rows.append(row)
                log_handle.write(
                    f"figure={row['figure']} variant={row['variant']} trial={row['trial_id']} "
                    f"seed={row['seed']} x={row['x_value']} failed={row['failed']}\n"
                )
                log_handle.flush()
            if log_text:
                log_handle.write(log_text)
                if not log_text.endswith("\n"):
                    log_handle.write("\n")
                log_handle.flush()
    if not args.streaming_csv:
        with StreamingCsvWriter(trial_csv, FIELDNAMES) as writer:
            for row in buffered_rows:
                writer.writerow(row)
        return buffered_rows
    return _read_csv(trial_csv)


def _fig1_fig2_shared_trial_csv(out_dir: pathlib.Path) -> pathlib.Path:
    return out_dir / FIG1_FIG2_SHARED_TRIAL_CSV


def _fig1_fig2_shared_summary_csv(out_dir: pathlib.Path) -> pathlib.Path:
    return out_dir / FIG1_FIG2_SHARED_SUMMARY_CSV


def _write_fig1_fig2_derived_outputs(out_dir: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fig1_rows = [{**row, "figure": "fig1"} for row in rows]
    fig2_rows = [{**row, "figure": "fig2"} for row in rows]
    _write_rows_atomic_csv(out_dir / "fig1_trials.csv", fig1_rows, FIELDNAMES)
    _write_rows_atomic_csv(out_dir / "fig2_trials.csv", fig2_rows, FIELDNAMES)
    fig1_summary = summarize_rows(rows, "fig1")
    fig2_summary = summarize_rows(rows, "fig2")
    _write_csv(
        out_dir / "fig1_summary.csv",
        fig1_summary,
        list(fig1_summary[0].keys()) if fig1_summary else [],
    )
    _write_csv(
        out_dir / "fig2_summary.csv",
        fig2_summary,
        list(fig2_summary[0].keys()) if fig2_summary else [],
    )


def _run_fig1_fig2_shared_trials(
    *,
    args: argparse.Namespace,
    snr_grid: list[float],
    trial_seeds: list[int],
    existing_metadata: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    out_dir = pathlib.Path(args.out_dir)
    shared_trial_csv = _fig1_fig2_shared_trial_csv(out_dir)
    shared_summary_csv = _fig1_fig2_shared_summary_csv(out_dir)
    log_path = out_dir / "fig1_fig2_vp_family_raw.log"
    can_reuse, rows = _can_reuse_csv(
        shared_trial_csv,
        FIG1_FIG2_SHARED_FIGURE,
        args,
        snr_grid,
        [FIG1_FIG2_SHARED_FIGURE],
        existing_metadata,
    )
    if not can_reuse:
        x_name, x_values = _figure_x_grid(FIG1_FIG2_SHARED_FIGURE, snr_grid, args.k_grid_values)
        variants = {
            **_variant_specs(FIG1_FIG2_SHARED_FIGURE),
            **_extra_peb_specs(FIG1_FIG2_SHARED_FIGURE),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks = _tasks_for_figure(
            figure=FIG1_FIG2_SHARED_FIGURE,
            grouped_group="fig1_fig2",
            x_name=x_name,
            x_values=x_values,
            variants=variants,
            trial_seeds=trial_seeds,
            args=args,
        )
        rows = _write_trial_results(shared_trial_csv, tasks, log_path, args)
    shared_summary = summarize_fig1_fig2_shared_rows(rows)
    _write_csv(
        shared_summary_csv,
        shared_summary,
        list(shared_summary[0].keys()) if shared_summary else [],
    )
    _write_fig1_fig2_derived_outputs(out_dir, rows)
    return rows


def _run_figure(
    figure: str,
    *,
    args: argparse.Namespace,
    snr_grid: list[float],
    trial_seeds: list[int],
    figures: list[str],
    existing_metadata: dict[str, Any] | None,
    completed_figures: set[str],
) -> list[dict[str, Any]]:
    if _is_fig1_fig2(figure):
        return _run_fig1_fig2_shared_trials(
            args=args,
            snr_grid=snr_grid,
            trial_seeds=trial_seeds,
            existing_metadata=existing_metadata,
        )
    out_dir = pathlib.Path(args.out_dir)
    trial_csv = out_dir / f"{figure}_trials.csv"
    summary_csv = out_dir / f"{figure}_summary.csv"
    log_path = out_dir / f"{figure}_raw.log"
    can_reuse, cached_rows = _can_reuse_csv(
        trial_csv, figure, args, snr_grid, figures, existing_metadata
    )
    if can_reuse:
        rows = cached_rows
        summary = summarize_rows(rows, figure)
        _write_csv(summary_csv, summary, list(summary[0].keys()) if summary else [])
        return rows

    x_name, x_values = _figure_x_grid(figure, snr_grid, args.k_grid_values)
    variants = {
        **_variant_specs("fig1" if figure == "fig2" else figure),
        **_extra_peb_specs(figure),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped_group = {
        "fig4": "fig4",
        "fig5": "fig5",
        "fig6": "fig6",
    }.get(figure)
    if figure == "fig3":
        grouped_group = None
    tasks = _tasks_for_figure(
        figure=figure,
        grouped_group=grouped_group,
        x_name=x_name,
        x_values=x_values,
        variants=variants,
        trial_seeds=trial_seeds,
        args=args,
    )
    rows = _write_trial_results(trial_csv, tasks, log_path, args)
    summary = summarize_rows(rows, figure)
    _write_csv(summary_csv, summary, list(summary[0].keys()) if summary else [])
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper ablation figures.")
    parser.add_argument("--figures", default="all")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--snr-grid", default=DEFAULT_SNR_GRID)
    parser.add_argument("--paper-k", type=int, default=DEFAULT_PAPER_K)
    parser.add_argument("--k-grid", default="1,2,3,4")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results/ablation_paper"))
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--maxtasksperchild", type=int, default=1)
    parser.add_argument(
        "--task-grouping",
        choices=("grouped", "variant"),
        default="grouped",
    )
    parser.add_argument(
        "--streaming-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--store-large-arrays",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--blas-threads", type=int, default=DEFAULT_BLAS_THREADS)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs is None and args.max_workers is None:
        args.jobs = 10
        args.max_workers = 10
    elif args.jobs is None:
        args.jobs = args.max_workers
    elif args.max_workers is None:
        args.max_workers = args.jobs
    elif args.max_workers != args.jobs:
        raise ValueError("--jobs and --max-workers were both provided with different values")
    args.k_grid_values = parse_k_grid(args.k_grid)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.maxtasksperchild <= 0:
        raise ValueError("--maxtasksperchild must be positive")
    if args.blas_threads <= 0:
        raise ValueError("--blas-threads must be positive")
    if args.paper_k <= 0:
        raise ValueError("--paper-k must be positive")
    _apply_blas_thread_env(args.blas_threads)
    figures = parse_figures(args.figures)
    snr_grid = parse_snr_grid(args.snr_grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "experiment_metadata.json"
    existing_metadata = _read_metadata(metadata_path)
    seed_sequence = np.random.SeedSequence(args.seed)
    trial_seeds = [_trial_seed(child) for child in seed_sequence.spawn(args.n_trials)]

    completed_figures: set[str] = set()
    fig1_fig2_shared_rows: list[dict[str, Any]] | None = None
    for figure in figures:
        if _is_fig1_fig2(figure):
            if fig1_fig2_shared_rows is None:
                fig1_fig2_shared_rows = _run_fig1_fig2_shared_trials(
                    args=args,
                    snr_grid=snr_grid,
                    trial_seeds=trial_seeds,
                    existing_metadata=existing_metadata,
                )
            rows = fig1_fig2_shared_rows
        else:
            rows = _run_figure(
                figure,
                args=args,
                snr_grid=snr_grid,
                trial_seeds=trial_seeds,
                figures=figures,
                existing_metadata=existing_metadata,
                completed_figures=completed_figures,
            )
        summary = summarize_rows(rows, figure)
        if not args.no_plots:
            _plot_figure(figure, summary, args.out_dir)
        completed_figures.add(figure)
    with metadata_path.open("w") as handle:
        json.dump(_metadata(args, snr_grid, figures), handle, indent=2)
    print(f"Wrote paper ablation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
