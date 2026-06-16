"""Generate paper ablation CSVs and PDF figures for the revised estimator."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
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
        _make_data,
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
        _make_data,
        run_single_proposed_diagnostic,
        run_stage1_only,
    )
    from ..metrics import position_rmse, relative_nmse
    from ..projections_delay import tau_from_pole
    from ..utils import scipy_is_available


DEFAULT_SNR_GRID = "-30,-25,-20,-15,-10,-5,0,5,10"
DEFAULT_PAPER_K = 3
FIGURE6_K_GRID = [1, 2, 3, 4]
FIGURE_ORDER = ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]
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
    "warning",
]


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


def parse_figures(value: str) -> list[str]:
    if value == "all":
        return list(FIGURE_ORDER)
    figures = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in figures if item not in FIGURE_ORDER]
    if unknown:
        raise ValueError(f"unknown figure ids: {unknown}")
    return figures


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
    if figure in {"fig1", "fig2"}:
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
    if figure == "fig1":
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
    ]
    for field in numeric_fields:
        row[field] = float("nan")
    return row


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


def _peb_result(config: dict) -> dict:
    data = _make_data(config)
    return {**data, **_peb_from_efim(data, config), "final": {}, "timing": data.get("timing", {})}


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
) -> tuple[dict[str, Any], str]:
    config = copy.deepcopy(config)
    config["seed"] = int(trial_seed)
    log_buffer = io.StringIO()
    start = time.perf_counter()
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
                result = _peb_result(config)
            else:
                result = run_single_proposed_diagnostic(config, allow_stage2=allow_stage2)
        row.update(extract_metrics(result, outlier_threshold_m))
    except Exception as exc:  # noqa: BLE001 - failed trials must be logged as rows.
        row["failed"] = True
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["K"] = config.get("K", "")
        row["receiver_mode"] = config.get("receiver_mode", "full_6d")
        log_buffer.write(f"\nERROR {type(exc).__name__}: {exc}\n")
    row["runtime_s"] = float(time.perf_counter() - start)
    return row, log_buffer.getvalue()


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
    }


def _metadata(args: argparse.Namespace, snr_grid: list[float], figures: list[str]) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "git_commit": commit,
        "timestamp": timestamp,
        "timestamp_utc": timestamp,
        "command_line": " ".join(sys.argv),
        "n_trials": int(args.n_trials),
        "jobs": int(args.jobs),
        "snr_grid": snr_grid,
        "paper_k": int(args.paper_k),
        "k_grid": [int(value) for value in args.k_grid_values],
        "figures": figures,
        "seed": int(args.seed),
        "receiver_noise_convention": "AWGN is added only on active EVS observation components with variance set by active-component signal power.",
        "config_overrides": {
            "seed": int(args.seed),
            "outlier_threshold_m": float(args.outlier_threshold_m),
            "paper_k_for_fig1_to_fig5": int(args.paper_k),
            "k_grid_for_fig6": [int(value) for value in args.k_grid_values],
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
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            return False
    return True


def _expected_variant_names(figure: str) -> list[str]:
    variants = {
        **_variant_specs("fig1" if figure == "fig2" else figure),
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
    x_name, x_values = _figure_x_grid(figure, snr_grid, args.k_grid_values)
    expected_variants = set(_expected_variant_names(figure))
    expected_x_values = {float(value) for value in x_values}
    groups: dict[tuple[str, float], int] = {}
    for row in rows:
        if row.get("figure") != figure:
            return False
        variant = str(row.get("variant"))
        x_value = _to_float(row.get("x_value"))
        if variant not in expected_variants or x_value not in expected_x_values:
            return False
        row_k = int(_to_float(row.get("K"))) if np.isfinite(_to_float(row.get("K"))) else None
        expected_k = int(x_value) if figure == "fig6" else int(args.paper_k)
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


def _failure_row_from_task(task: dict[str, Any], exc: BaseException) -> tuple[dict[str, Any], str]:
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
    return row, f"\nERROR {type(exc).__name__}: {exc}\n"


def _run_trial_task(task: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        config = _config_for_point(
            figure=task["figure"],
            variant_updates=task["updates"],
            seed=task["trial_seed"],
            snr_db=task["snr_db"],
            x_value=task["x_value"],
            paper_k=task["paper_k"],
        )
        return run_one_trial(
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
        )
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
                    }
                )
    return tasks


def _run_figure(
    figure: str,
    *,
    args: argparse.Namespace,
    snr_grid: list[float],
    trial_seeds: list[int],
    figures: list[str],
    existing_metadata: dict[str, Any] | None,
) -> list[dict[str, Any]]:
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
    if figure == "fig2" and not args.force_rerun:
        fig1_csv = out_dir / "fig1_trials.csv"
        if fig1_csv.exists():
            fig1_reusable, fig1_rows = _can_reuse_csv(
                fig1_csv, "fig1", args, snr_grid, figures, existing_metadata
            )
            if fig1_reusable or args.reuse_existing:
                allowed = set(_variant_specs("fig2"))
                rows = [
                    {**row, "figure": "fig2"}
                    for row in fig1_rows
                    if row.get("variant") in allowed
                ]
                _write_csv(trial_csv, rows, FIELDNAMES)
                summary = summarize_rows(rows, figure)
                _write_csv(summary_csv, summary, list(summary[0].keys()) if summary else [])
                with log_path.open("w") as log_handle:
                    log_handle.write("Reused fig1_trials.csv for Figure 2 NMSE curves.\n")
                return rows

    x_name, x_values = _figure_x_grid(figure, snr_grid, args.k_grid_values)
    variants = {
        **_variant_specs("fig1" if figure == "fig2" else figure),
        **_extra_peb_specs(figure),
    }
    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = _trial_tasks(
        figure=figure,
        x_name=x_name,
        x_values=x_values,
        variants=variants,
        trial_seeds=trial_seeds,
        outlier_threshold_m=float(args.outlier_threshold_m),
        verbose=bool(args.verbose),
        paper_k=int(args.paper_k),
    )
    processes = min(int(args.jobs), max(len(tasks), 1))
    with log_path.open("w") as log_handle:
        log_handle.write(f"parallel_jobs={processes}\n")
        with mp.Pool(processes=processes) as pool:
            for row, log_text in pool.map(_run_trial_task, tasks):
                rows.append(row)
                log_handle.write(
                    f"figure={figure} variant={row['variant']} trial={row['trial_id']} "
                    f"seed={row['seed']} x={row['x_value']} failed={row['failed']}\n"
                )
                if log_text:
                    log_handle.write(log_text)
                    if not log_text.endswith("\n"):
                        log_handle.write("\n")
    _write_csv(trial_csv, rows, FIELDNAMES)
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
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args(argv)
    args.k_grid_values = parse_k_grid(args.k_grid)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.paper_k <= 0:
        raise ValueError("--paper-k must be positive")
    figures = parse_figures(args.figures)
    snr_grid = parse_snr_grid(args.snr_grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "experiment_metadata.json"
    existing_metadata = _read_metadata(metadata_path)
    seed_sequence = np.random.SeedSequence(args.seed)
    trial_seeds = [_trial_seed(child) for child in seed_sequence.spawn(args.n_trials)]

    for figure in figures:
        rows = _run_figure(
            figure,
            args=args,
            snr_grid=snr_grid,
            trial_seeds=trial_seeds,
            figures=figures,
            existing_metadata=existing_metadata,
        )
        summary = summarize_rows(rows, figure)
        if not args.no_plots:
            _plot_figure(figure, summary, args.out_dir)
    with metadata_path.open("w") as handle:
        json.dump(_metadata(args, snr_grid, figures), handle, indent=2)
    print(f"Wrote paper ablation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
