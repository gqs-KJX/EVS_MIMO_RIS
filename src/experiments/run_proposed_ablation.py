"""Formal ablations for the revised proposed EVS-RIS-OFDM pipeline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import sys
import time
import traceback
from collections.abc import Iterable
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.config import default_config
    from src.main_single_proposed import run_single_proposed_diagnostic
    from src.metrics import position_rmse, relative_nmse
else:
    from ..config import default_config
    from ..main_single_proposed import run_single_proposed_diagnostic
    from ..metrics import position_rmse, relative_nmse


ABLATION_GROUPS = ("vp_family", "stage2_gate", "jones_lambda")

FIELDNAMES = [
    "trial_id",
    "seed",
    "ablation",
    "variant",
    "snr_db",
    "failed",
    "error",
    "runtime_s",
    "y_nmse",
    "position_rmse_m",
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
    "global_vp_runtime_s",
    "total_runtime_s",
    "reliability_decision",
    "legacy_stage1_decision",
    "bad_score",
    "trigger_reasons",
    "gof_stat",
    "gof_dof",
    "gof_pass",
    "data_only_efim_lambda_min",
    "data_only_efim_condition_number",
    "data_only_scaled_efim_lambda_min",
    "data_only_scaled_efim_condition_number",
    "fixed_pol_score",
    "jones_score",
    "lambda_jones_per_path",
    "snr_eff_per_path",
    "jones_leakage_per_path",
    "jones_rho_summary",
    "stage1_assignment_margin",
    "stage1_selected_clock_std_ns",
    "delta_t_k_ns",
    "stage1_runtime_s",
    "proposed_stage2_policy",
    "ngc_policy_active",
    "ngc_lambda_ris",
    "ngc_direct_clock_score_norm",
    "ngc_direct_clock_std_ns",
    "ngc_direct_ris_score_norm",
    "ngc_direct_ris_available",
    "ngc_direct_total_score",
    "ngc_direct_cert_status",
    "ngc_rescue_requested",
    "ngc_rescue_request_reason",
    "ngc_rescue_clock_score_norm",
    "ngc_rescue_ris_score_norm",
    "ngc_rescue_total_score",
    "ngc_rescue_cert_status",
    "ngc_selected_by",
    "ngc_final_unreliable",
    "ngc_threshold_clock_green",
    "ngc_threshold_clock_red",
]


def _deep_update(base: dict, updates: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _proposed_ngc_spec(*, allow_stage2: bool = True) -> dict[str, Any]:
    return {
        "enable_global_vp": True,
        "global_vp": {"mode": "adaptive_jones"},
        "stage2_adaptive": True,
        "stage2_rescue_type": "ris_only",
        "proposed_stage2_policy": "ngc_certified_ris_only",
        "rescue_accept_min_rel_improvement": 0.0,
        "rescue_accept_min_abs_improvement": 1.0e-8,
        "_allow_stage2": bool(allow_stage2),
    }


def _proposed_force_lower_raw_spec() -> dict[str, Any]:
    spec = _proposed_ngc_spec(allow_stage2=True)
    spec["proposed_stage2_policy"] = "force_ris_only"
    return spec


def _proposed_old_gated_spec() -> dict[str, Any]:
    spec = _proposed_ngc_spec(allow_stage2=True)
    spec["proposed_stage2_policy"] = "reliability_gated_ris_only"
    spec["rescue_accept_min_rel_improvement"] = 1.0e-3
    return spec


def _variant_specs(ablation: str) -> dict[str, dict[str, Any]]:
    """Return config updates keyed by formal ablation variant name."""
    if ablation == "vp_family":
        return {
            "stage1_only_no_vp": {
                "enable_global_vp": False,
                "stage2_adaptive": False,
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "fixed_pol_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "fixed_pol"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "jones_free_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_free"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "jones_regularized_vp": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_regularized"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "adaptive_jones_vp_proposed": {
                "enable_global_vp": True,
                "global_vp": {"mode": "adaptive_jones"},
                "stage2_adaptive": False,
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
        }
    if ablation == "stage2_gate":
        return {
            "direct_vp_only": {
                "enable_global_vp": True,
                "global_vp": {"mode": "adaptive_jones"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "adaptive_jones_vp_proposed_force_lower_raw": {
                **_proposed_force_lower_raw_spec(),
            },
            "adaptive_jones_vp_proposed_old_gated": {
                **_proposed_old_gated_spec(),
            },
            "adaptive_jones_vp_proposed": {
                **_proposed_ngc_spec(allow_stage2=True),
            },
        }
    if ablation == "jones_lambda":
        fixed_lambda = {
            "jones_regularization_scaling": "gram",
        }

        def lambda_cfg(value: float) -> dict[str, Any]:
            return {
                "enable_global_vp": True,
                "global_vp": {
                    "mode": "jones_regularized",
                    **fixed_lambda,
                    "jones_lambda0": value,
                    "jones_lambda_min": value,
                    "jones_lambda_max": value,
                },
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            }

        return {
            "fixed_pol_limit": {
                "enable_global_vp": True,
                "global_vp": {"mode": "fixed_pol"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "lambda_1e4": lambda_cfg(1.0e4),
            "lambda_1e2": lambda_cfg(1.0e2),
            "lambda_1": lambda_cfg(1.0),
            "lambda_1e_minus_2": lambda_cfg(1.0e-2),
            "free_jones": {
                "enable_global_vp": True,
                "global_vp": {"mode": "jones_free"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
            "adaptive_jones": {
                "enable_global_vp": True,
                "global_vp": {"mode": "adaptive_jones"},
                "proposed_stage2_policy": "reliability_gated",
                "_allow_stage2": False,
            },
        }
    raise ValueError(f"unknown ablation group {ablation!r}")


def _ablation_groups(name: str) -> list[str]:
    if name == "all":
        return list(ABLATION_GROUPS)
    if name not in ABLATION_GROUPS:
        raise ValueError(f"unknown ablation {name!r}")
    return [name]


def _parse_snr_grid(args: argparse.Namespace) -> list[float]:
    if args.snr_grid is None or str(args.snr_grid).strip() == "":
        return [float(args.snr_db)]
    return [float(item.strip()) for item in str(args.snr_grid).split(",") if item.strip()]


def _policy_log_fragment(spec: dict[str, Any]) -> str:
    return (
        f"proposed_stage2_policy={spec.get('proposed_stage2_policy', '')} "
        f"ngc_lambda_ris={spec.get('ngc_lambda_ris', 1.0)} "
        f"ngc_clock_green_quantile={spec.get('ngc_clock_green_quantile', 0.99)} "
        f"ngc_clock_red_quantile={spec.get('ngc_clock_red_quantile', 0.999)} "
        "rescue_accept_min_rel_improvement="
        f"{spec.get('rescue_accept_min_rel_improvement', '')} "
        "rescue_accept_min_abs_improvement="
        f"{spec.get('rescue_accept_min_abs_improvement', '')}"
    )


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
    """Return the first available nested value from dotted or tuple paths."""
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
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return str(value)
    if arr.size == 0:
        return ""
    if np.iscomplexobj(arr):
        values = [[float(np.real(item)), float(np.imag(item))] for item in arr.reshape(-1)]
    else:
        values = [float(item) for item in np.asarray(arr, dtype=float).reshape(-1)]
    return json.dumps(values, separators=(",", ":"))


def _list_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    return str(value)


def _rho_summary(value: Any) -> str:
    if value is None:
        return ""
    arr = np.asarray(value, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return ""
    summary = {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }
    return json.dumps(summary, separators=(",", ":"))


def _empty_row() -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    for field in (
        "runtime_s",
        "y_nmse",
        "position_rmse_m",
        "range_rmse_m",
        "tau_rmse_s",
        "raw_objective_final",
        "global_vp_runtime_s",
        "total_runtime_s",
        "gof_stat",
        "data_only_efim_lambda_min",
        "data_only_efim_condition_number",
        "data_only_scaled_efim_lambda_min",
        "data_only_scaled_efim_condition_number",
        "fixed_pol_score",
        "jones_score",
        "stage1_assignment_margin",
        "stage1_selected_clock_std_ns",
        "stage1_runtime_s",
    ):
        row[field] = float("nan")
    return row


def _failure_row(
    *,
    trial_id: int,
    seed: int,
    ablation: str,
    variant: str,
    snr_db: float,
    runtime_s: float,
    error: BaseException,
) -> dict[str, Any]:
    row = _empty_row()
    row.update(
        {
            "trial_id": trial_id,
            "seed": seed,
            "ablation": ablation,
            "variant": variant,
            "snr_db": float(snr_db),
            "failed": True,
            "error": f"{type(error).__name__}: {error}".replace("\n", " | ")[:2000],
            "runtime_s": float(runtime_s),
        }
    )
    return row


def _extract_row(
    *,
    result: dict,
    trial_id: int,
    seed: int,
    ablation: str,
    variant: str,
    snr_db: float,
    runtime_s: float,
    outlier_threshold_m: float,
) -> dict[str, Any]:
    final = result.get("final", {})
    scene = result.get("scene", {})
    true_components = result.get("true_components", {})
    timing = result.get("timing", {})
    reliability = result.get("reliability", final.get("reliability", {}))

    y_hat = final.get("Y_hat")
    y_true = result.get("Y_true")
    y_nmse = float("nan")
    if y_hat is not None and y_true is not None:
        y_nmse = float(relative_nmse(y_hat, y_true))

    p_hat = final.get("p_u")
    p_true = scene.get("p_u_true")
    pos_rmse = float("nan")
    if p_hat is not None and p_true is not None:
        pos_rmse = float(position_rmse(np.asarray(p_hat, dtype=float), np.asarray(p_true, dtype=float)))

    components = final.get("components", {})
    range_rmse = _rmse_array(components.get("ranges"), true_components.get("ranges"))
    tau_rmse = _rmse_array(components.get("taus"), true_components.get("taus"))
    raw_objective = _finite_float(
        get_nested(result, ["final.raw_objective_final", "final.raw_objective"], np.nan)
    )

    lambda_path = get_nested(result, ["final.lambda_jones_per_path", "lambda_jones_per_path"], None)
    snr_eff = get_nested(result, ["final.snr_eff_per_path", "snr_eff_per_path"], None)
    leakage = get_nested(result, ["final.jones_leakage_per_path", "jones_leakage_per_path"], None)
    jones_rho = get_nested(result, ["final.jones_rho", "jones_rho"], None)

    row = _empty_row()
    row.update(
        {
            "trial_id": trial_id,
            "seed": seed,
            "ablation": ablation,
            "variant": variant,
            "snr_db": float(snr_db),
            "failed": False,
            "error": "",
            "runtime_s": float(runtime_s),
            "y_nmse": y_nmse,
            "position_rmse_m": pos_rmse,
            "range_rmse_m": range_rmse,
            "tau_rmse_s": tau_rmse,
            "raw_objective_final": raw_objective,
            "outlier_flag": bool(np.isfinite(pos_rmse) and pos_rmse > outlier_threshold_m),
            "selected_branch": get_nested(result, ["selected_branch", "final.selected_branch"], ""),
            "final_refinement_method": get_nested(result, ["final.final_refinement_method"], ""),
            "global_vp_mode": get_nested(result, ["final.global_vp_mode", "final.vp_mode"], ""),
            "selected_vp_family_branch": get_nested(result, ["final.selected_vp_family_branch"], ""),
            "linear_nuisance_dim": get_nested(result, ["final.linear_nuisance_dim"], ""),
            "nonlinear_dim": get_nested(result, ["final.nonlinear_dim"], ""),
            "global_vp_runtime_s": _finite_float(timing.get("vp")),
            "total_runtime_s": _finite_float(timing.get("total", timing.get("diagnostic_total"))),
            "reliability_decision": reliability.get("decision", ""),
            "legacy_stage1_decision": reliability.get("legacy_stage1_decision", ""),
            "bad_score": reliability.get("bad_score", ""),
            "trigger_reasons": _list_string(reliability.get("trigger_reasons", [])),
            "gof_stat": _finite_float(reliability.get("gof_stat")),
            "gof_dof": reliability.get("gof_dof", ""),
            "gof_pass": reliability.get("gof_pass", ""),
            "data_only_efim_lambda_min": _finite_float(reliability.get("data_only_efim_lambda_min")),
            "data_only_efim_condition_number": _finite_float(
                reliability.get("data_only_efim_condition_number")
            ),
            "data_only_scaled_efim_lambda_min": _finite_float(
                reliability.get("data_only_scaled_efim_lambda_min")
            ),
            "data_only_scaled_efim_condition_number": _finite_float(
                reliability.get("data_only_scaled_efim_condition_number")
            ),
            "fixed_pol_score": _finite_float(get_nested(result, ["final.fixed_pol_score"], np.nan)),
            "jones_score": _finite_float(get_nested(result, ["final.jones_score"], np.nan)),
            "lambda_jones_per_path": _vector_string(lambda_path),
            "snr_eff_per_path": _vector_string(snr_eff),
            "jones_leakage_per_path": _vector_string(leakage),
            "jones_rho_summary": _rho_summary(jones_rho),
            "stage1_assignment_margin": _finite_float(reliability.get("assignment_margin")),
            "stage1_selected_clock_std_ns": _finite_float(reliability.get("sigma_delta_t_ns")),
            "delta_t_k_ns": _vector_string(reliability.get("delta_t_k_ns")),
            "stage1_runtime_s": _finite_float(timing.get("stage1")),
            "proposed_stage2_policy": reliability.get(
                "proposed_stage2_policy",
                get_nested(result, ["stage1_config.proposed_stage2_policy"], ""),
            ),
            "ngc_policy_active": bool(result.get("ngc_policy_active", False)),
            "ngc_lambda_ris": _finite_float(result.get("ngc_lambda_ris")),
            "ngc_direct_clock_score_norm": _finite_float(
                result.get("ngc_direct_clock_score_norm")
            ),
            "ngc_direct_clock_std_ns": _finite_float(
                result.get("ngc_direct_clock_std_ns")
            ),
            "ngc_direct_ris_score_norm": _finite_float(
                result.get("ngc_direct_ris_score_norm")
            ),
            "ngc_direct_ris_available": bool(
                result.get("ngc_direct_ris_available", False)
            ),
            "ngc_direct_total_score": _finite_float(
                result.get("ngc_direct_total_score")
            ),
            "ngc_direct_cert_status": str(result.get("ngc_direct_cert_status", "")),
            "ngc_rescue_requested": bool(result.get("ngc_rescue_requested", False)),
            "ngc_rescue_request_reason": str(
                result.get("ngc_rescue_request_reason", "")
            ),
            "ngc_rescue_clock_score_norm": _finite_float(
                result.get("ngc_rescue_clock_score_norm")
            ),
            "ngc_rescue_ris_score_norm": _finite_float(
                result.get("ngc_rescue_ris_score_norm")
            ),
            "ngc_rescue_total_score": _finite_float(
                result.get("ngc_rescue_total_score")
            ),
            "ngc_rescue_cert_status": str(result.get("ngc_rescue_cert_status", "")),
            "ngc_selected_by": str(result.get("ngc_selected_by", "")),
            "ngc_final_unreliable": bool(result.get("ngc_final_unreliable", False)),
            "ngc_threshold_clock_green": _finite_float(
                result.get("ngc_threshold_clock_green")
            ),
            "ngc_threshold_clock_red": _finite_float(
                result.get("ngc_threshold_clock_red")
            ),
        }
    )
    return row


def _base_config(seed: int, snr_db: float, diagnostic_mode: str) -> dict:
    config = default_config()
    config["seed"] = int(seed)
    config["SNR_dB"] = float(snr_db)
    config["print_progress"] = False
    config["verbose_stage2"] = False
    config["run_full_legacy_comparison"] = False
    config["diagnostic_mode"] = "smoke" if diagnostic_mode == "fast" else "performance"
    if diagnostic_mode == "fast":
        config["diagnostic_fast_problem_size"] = True
        config["diagnostic_fast_stage1_search"] = True
        config["global_vp"] = dict(config["global_vp"])
        config["global_vp"]["max_iter"] = min(int(config["global_vp"].get("max_iter", 80)), 10)
    return config


def _run_variant(
    *,
    trial_id: int,
    seed: int,
    ablation: str,
    variant: str,
    updates: dict[str, Any],
    snr_db: float,
    diagnostic_mode: str,
    outlier_threshold_m: float,
) -> dict[str, Any]:
    config = _deep_update(_base_config(seed, snr_db, diagnostic_mode), updates)
    allow_stage2 = bool(updates.get("_allow_stage2", True))
    start = time.perf_counter()
    result = run_single_proposed_diagnostic(config, allow_stage2=allow_stage2)
    runtime_s = time.perf_counter() - start
    return _extract_row(
        result=result,
        trial_id=trial_id,
        seed=seed,
        ablation=ablation,
        variant=variant,
        snr_db=snr_db,
        runtime_s=runtime_s,
        outlier_threshold_m=outlier_threshold_m,
    )


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        pd.DataFrame(rows, columns=FIELDNAMES).to_csv(path, index=False)
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        if lowered == "":
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(numeric):
        return bool(numeric)
    return None


def _ngc_rescue_run_rate(rows: list[dict[str, Any]]) -> float:
    active_rows = [
        row for row in rows if _optional_bool(row.get("ngc_policy_active")) is True
    ]
    if not active_rows:
        return float("nan")
    requested = sum(
        _optional_bool(row.get("ngc_rescue_requested")) is True for row in active_rows
    )
    return float(requested / len(active_rows))


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nGrouped summary")
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        frame = pd.DataFrame(rows)
        frame = frame[~frame["failed"].astype(bool)]
        if frame.empty:
            print("No successful rows.")
            return
        ngc_active = frame["ngc_policy_active"].map(_optional_bool)
        ngc_requested = frame["ngc_rescue_requested"].map(_optional_bool)
        frame = frame.assign(
            rescue_run_rate=np.where(
                ngc_active == True,  # noqa: E712 - pandas elementwise comparison.
                ngc_requested.fillna(False).astype(float),
                np.nan,
            )
        )
        summary = frame.groupby(["ablation", "variant", "snr_db"], dropna=False).agg(
            median_position_rmse_m=("position_rmse_m", "median"),
            p90_position_rmse_m=("position_rmse_m", lambda x: float(np.percentile(x, 90.0))),
            outlier_rate=("outlier_flag", "mean"),
            rescue_run_rate=("rescue_run_rate", "mean"),
            median_y_nmse=("y_nmse", "median"),
            median_runtime_s=("runtime_s", "median"),
        )
        print(summary.to_string())
        return

    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("failed"):
            continue
        key = (str(row["ablation"]), str(row["variant"]), float(row["snr_db"]))
        groups.setdefault(key, []).append(row)
    if not groups:
        print("No successful rows.")
        return
    header = (
        "ablation, variant, snr_db, median_position_rmse_m, "
        "p90_position_rmse_m, outlier_rate, rescue_run_rate, "
        "median_y_nmse, median_runtime_s"
    )
    print(header)
    for key in sorted(groups):
        group = groups[key]
        pos = np.asarray([row["position_rmse_m"] for row in group], dtype=float)
        y_nmse = np.asarray([row["y_nmse"] for row in group], dtype=float)
        runtime = np.asarray([row["runtime_s"] for row in group], dtype=float)
        outliers = np.asarray([bool(row["outlier_flag"]) for row in group], dtype=float)
        rescue_run_rate = _ngc_rescue_run_rate(group)
        print(
            f"{key[0]}, {key[1]}, {key[2]:.6g}, {np.nanmedian(pos):.6e}, "
            f"{np.nanpercentile(pos, 90.0):.6e}, {np.nanmean(outliers):.6f}, "
            f"{rescue_run_rate:.6f}, "
            f"{np.nanmedian(y_nmse):.6e}, {np.nanmedian(runtime):.6e}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal ablations for Stage-I, RIS/JNPP gating, and adaptive Jones-VP."
    )
    parser.add_argument("--ablation", choices=(*ABLATION_GROUPS, "all"), default="all")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--snr-db", type=float, default=-20.0)
    parser.add_argument("--snr-grid", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/proposed_ablation.csv"))
    parser.add_argument("--diagnostic-mode", choices=("performance", "fast"), default="performance")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outlier-threshold-m", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    snr_values = _parse_snr_grid(args)
    ablations = _ablation_groups(args.ablation)
    trial_sequence = np.random.SeedSequence(args.seed)
    trial_seeds = [_trial_seed(child) for child in trial_sequence.spawn(args.n_trials)]

    rows: list[dict[str, Any]] = []
    for snr_db in snr_values:
        for trial_id, trial_seed in enumerate(trial_seeds):
            for ablation in ablations:
                for variant, updates in _variant_specs(ablation).items():
                    print(
                        f"Running {ablation}/{variant} trial={trial_id + 1}/{args.n_trials} "
                        f"seed={trial_seed} snr_db={snr_db} "
                        f"{_policy_log_fragment(updates)}",
                        flush=True,
                    )
                    start = time.perf_counter()
                    try:
                        rows.append(
                            _run_variant(
                                trial_id=trial_id,
                                seed=trial_seed,
                                ablation=ablation,
                                variant=variant,
                                updates=updates,
                                snr_db=snr_db,
                                diagnostic_mode=args.diagnostic_mode,
                                outlier_threshold_m=args.outlier_threshold_m,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - one failed variant should be recordable.
                        if not args.continue_on_error:
                            raise
                        rows.append(
                            _failure_row(
                                trial_id=trial_id,
                                seed=trial_seed,
                                ablation=ablation,
                                variant=variant,
                                snr_db=snr_db,
                                runtime_s=time.perf_counter() - start,
                                error=RuntimeError(traceback.format_exc(limit=8)),
                            )
                        )
    _write_csv(args.out, rows)
    print(f"\nWrote {args.out}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
