"""Pilot sweep for Robust JNPP confidence-weight strategies."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pathlib
import sys
import traceback
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from src.config import default_config
    from src.main_single_proposed import _final_raw_objective, run_single_proposed_diagnostic
    from src.metrics import position_rmse, relative_nmse
else:
    from .config import default_config
    from .main_single_proposed import _final_raw_objective, run_single_proposed_diagnostic
    from .metrics import position_rmse, relative_nmse


RAW_FIELDS = [
    "snr_db",
    "seed",
    "weight_mode",
    "selected_branch",
    "ue_position_rmse_m",
    "y_nmse",
    "raw_objective",
    "total_runtime_s",
    "error_message",
    "assignment_margin",
    "sigma_delta_t_ns",
    "rank1_ratio_max",
    "initial_z_residual",
    "direct_vp_good",
    "direct_vp_raw_objective",
    "direct_vp_nfev",
    "stage2_rescue_mode",
    "jnpp_weight_mode",
    "jnpp_weights",
    "jnpp_rank1_ratios",
    "jnpp_use_leave_one_out",
    "jnpp_leave_one_out_effective",
    "jnpp_num_candidates",
    "jnpp_num_subsets",
    "jnpp_best_objective",
    "jnpp_best_clock_std_ns",
    "jnpp_runtime_s",
    "direct_vp_selected",
    "jnpp_selected",
    "rollback_selected",
]

SUMMARY_FIELDS = [
    "snr_db",
    "weight_mode",
    "num_runs",
    "num_success",
    "median_rmse_m",
    "mean_rmse_m",
    "p90_rmse_m",
    "outlier_rate",
    "median_y_nmse",
    "median_runtime_s",
    "direct_vp_selected_rate",
    "jnpp_selected_rate",
    "rollback_selected_rate",
    "median_jnpp_runtime_s",
]


def _parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("list must contain at least one float")
    return values


def _parse_str_list(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("list must contain at least one item")
    allowed = {"equal", "exponential", "inverse_rank"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown weight modes: {unknown}")
    return values


def _nan() -> float:
    return float("nan")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return default
    return value_float if np.isfinite(value_float) else default


def _safe_int(value: Any) -> int | float:
    try:
        return int(value)
    except (TypeError, ValueError):
        return _nan()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(_jsonable(value))
    if isinstance(value, (list, tuple)):
        return json.dumps(_jsonable(value))
    if value is None:
        return ""
    return value


def _rescue_diagnostics(results: dict) -> dict:
    structured = results.get("structured_diag")
    if isinstance(structured, dict) and structured.get("stage2_rescue_mode"):
        return structured
    branches = results.get("branches", {})
    for name in ("ris_only_stage2_then_vp", "multi_hypothesis_ris_reacquisition_then_vp"):
        branch = branches.get(name)
        if isinstance(branch, dict):
            diag = branch.get("structured_diag", {})
            if isinstance(diag, dict) and diag.get("stage2_rescue_mode"):
                return diag
    return {}


def _direct_branch(results: dict) -> dict:
    branches = results.get("branches", {})
    branch = branches.get("direct_vp", {})
    return branch if isinstance(branch, dict) else {}


def _direct_vp_nfev(branch: dict) -> int | float:
    final = branch.get("final", {}) if isinstance(branch, dict) else {}
    optimizer = final.get("optimizer", {}) if isinstance(final, dict) else {}
    if isinstance(optimizer, dict) and "n_eval" in optimizer:
        return _safe_int(optimizer.get("n_eval"))
    if isinstance(final, dict) and "vp_nfev" in final:
        return _safe_int(final.get("vp_nfev"))
    if isinstance(final, dict) and "global_vp_num_iter" in final:
        return _safe_int(final.get("global_vp_num_iter"))
    return _nan()


def _extract_success_row(results: dict, snr_db: float, seed: int, weight_mode: str) -> dict:
    final = results.get("final", {})
    estimate_initial = results.get("estimate_initial", {})
    reliability = results.get("reliability", {})
    timing = results.get("timing", {})
    direct = _direct_branch(results)
    direct_quality = results.get("direct_vp_quality", {})
    rescue_diag = _rescue_diagnostics(results)
    selected_branch = str(results.get("selected_branch", ""))

    row = {field: _nan() for field in RAW_FIELDS}
    row.update(
        {
            "snr_db": float(snr_db),
            "seed": int(seed),
            "weight_mode": str(weight_mode),
            "selected_branch": selected_branch,
            "error_message": "",
            "direct_vp_selected": selected_branch == "direct_vp",
            "jnpp_selected": False,
            "rollback_selected": selected_branch == "direct_vp_rollback",
        }
    )

    if isinstance(final, dict) and "p_u" in final:
        row["ue_position_rmse_m"] = position_rmse(
            np.asarray(final["p_u"]), np.asarray(results["scene"]["p_u_true"])
        )
    if (
        isinstance(final, dict)
        and final.get("Y_hat") is not None
        and results.get("Y_true") is not None
    ):
        row["y_nmse"] = relative_nmse(final["Y_hat"], results["Y_true"])
    row["raw_objective"] = _final_raw_objective(final if isinstance(final, dict) else {})
    row["total_runtime_s"] = _safe_float(
        timing.get("diagnostic_total", timing.get("total")) if isinstance(timing, dict) else None
    )

    if isinstance(reliability, dict):
        row["assignment_margin"] = _safe_float(reliability.get("assignment_margin"))
        row["sigma_delta_t_ns"] = _safe_float(reliability.get("sigma_delta_t_ns"))
    if isinstance(estimate_initial, dict):
        row["rank1_ratio_max"] = _safe_float(
            estimate_initial.get("stage1_max_rank1_ratio", estimate_initial.get("rank1_ratio_max"))
        )
        row["initial_z_residual"] = _safe_float(estimate_initial.get("initial_z_residual"))

    if isinstance(direct_quality, dict):
        row["direct_vp_good"] = bool(direct_quality.get("good", False))
        row["direct_vp_raw_objective"] = _safe_float(direct_quality.get("raw_objective_final"))
    if math.isnan(_safe_float(row["direct_vp_raw_objective"])):
        direct_final = direct.get("final", {}) if isinstance(direct, dict) else {}
        row["direct_vp_raw_objective"] = _final_raw_objective(
            direct_final if isinstance(direct_final, dict) else {}
        )
    row["direct_vp_nfev"] = _direct_vp_nfev(direct)

    if isinstance(rescue_diag, dict) and rescue_diag:
        row["stage2_rescue_mode"] = str(rescue_diag.get("stage2_rescue_mode", ""))
        row["jnpp_weight_mode"] = str(rescue_diag.get("jnpp_weight_mode", ""))
        row["jnpp_weights"] = _csv_value(rescue_diag.get("jnpp_weights", ""))
        row["jnpp_rank1_ratios"] = _csv_value(
            rescue_diag.get("jnpp_rank1_ratios", rescue_diag.get("jnpp_rank1_ratios_used", ""))
        )
        row["jnpp_use_leave_one_out"] = bool(rescue_diag.get("jnpp_use_leave_one_out", False))
        row["jnpp_leave_one_out_effective"] = bool(
            rescue_diag.get("jnpp_leave_one_out_effective", False)
        )
        row["jnpp_num_candidates"] = _safe_int(rescue_diag.get("jnpp_num_candidates"))
        row["jnpp_num_subsets"] = _safe_int(rescue_diag.get("jnpp_num_subsets"))
        row["jnpp_best_objective"] = _safe_float(rescue_diag.get("jnpp_best_objective"))
        row["jnpp_best_clock_std_ns"] = _safe_float(rescue_diag.get("jnpp_best_clock_std_ns"))
        row["jnpp_runtime_s"] = _safe_float(
            rescue_diag.get("jnpp_runtime_total", rescue_diag.get("jnpp_runtime_s"))
        )
        row["jnpp_selected"] = (
            selected_branch == "ris_only_stage2_then_vp"
            and row["stage2_rescue_mode"] == "robust_jnpp"
        )
    else:
        row["stage2_rescue_mode"] = ""
        row["jnpp_weight_mode"] = ""
        row["jnpp_weights"] = ""
        row["jnpp_rank1_ratios"] = ""
    return row


def _failure_row(
    snr_db: float,
    seed: int,
    weight_mode: str,
    error: BaseException,
) -> dict:
    row = {field: _nan() for field in RAW_FIELDS}
    row.update(
        {
            "snr_db": float(snr_db),
            "seed": int(seed),
            "weight_mode": str(weight_mode),
            "selected_branch": "",
            "stage2_rescue_mode": "",
            "jnpp_weight_mode": "",
            "jnpp_weights": "",
            "jnpp_rank1_ratios": "",
            "direct_vp_selected": False,
            "jnpp_selected": False,
            "rollback_selected": False,
            "error_message": f"{type(error).__name__}: {error}",
        }
    )
    return row


def _nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else _nan()


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else _nan()


def _nanpercentile(values: list[float], percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else _nan()


def _summarize(rows: list[dict], snr_values: list[float], weight_modes: list[str], outlier_threshold: float) -> list[dict]:
    summary = []
    for snr_db in sorted(snr_values):
        for mode in weight_modes:
            group = [
                row
                for row in rows
                if float(row["snr_db"]) == float(snr_db) and str(row["weight_mode"]) == mode
            ]
            success = [row for row in group if not row.get("error_message")]
            rmse = [_safe_float(row.get("ue_position_rmse_m")) for row in success]
            y_nmse = [_safe_float(row.get("y_nmse")) for row in success]
            runtime = [_safe_float(row.get("total_runtime_s")) for row in success]
            jnpp_runtime = [_safe_float(row.get("jnpp_runtime_s")) for row in success]
            finite_rmse = np.asarray([value for value in rmse if np.isfinite(value)])
            direct_flags = [bool(row.get("direct_vp_selected", False)) for row in success]
            jnpp_flags = [bool(row.get("jnpp_selected", False)) for row in success]
            rollback_flags = [bool(row.get("rollback_selected", False)) for row in success]
            summary.append(
                {
                    "snr_db": float(snr_db),
                    "weight_mode": mode,
                    "num_runs": int(len(group)),
                    "num_success": int(finite_rmse.size),
                    "median_rmse_m": _nanmedian(rmse),
                    "mean_rmse_m": _nanmean(rmse),
                    "p90_rmse_m": _nanpercentile(rmse, 90.0),
                    "outlier_rate": float(np.mean(finite_rmse > outlier_threshold))
                    if finite_rmse.size
                    else _nan(),
                    "median_y_nmse": _nanmedian(y_nmse),
                    "median_runtime_s": _nanmedian(runtime),
                    "direct_vp_selected_rate": float(np.mean(direct_flags)) if direct_flags else _nan(),
                    "jnpp_selected_rate": float(np.mean(jnpp_flags)) if jnpp_flags else _nan(),
                    "rollback_selected_rate": float(np.mean(rollback_flags)) if rollback_flags else _nan(),
                    "median_jnpp_runtime_s": _nanmedian(jnpp_runtime),
                }
            )
    return summary


def _write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, _nan())) for field in fields})


def _fmt(value: Any) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return "NA"
    return f"{value_float:.6e}"


def _print_summary(summary: list[dict]) -> None:
    print(
        "\nsnr_db | weight_mode | num_success | median_rmse_m | mean_rmse_m | "
        "p90_rmse_m | outlier_rate | median_y_nmse | median_runtime_s | "
        "direct_vp_selected_rate | jnpp_selected_rate | rollback_selected_rate | "
        "median_jnpp_runtime_s"
    )
    for row in summary:
        print(
            f"{_fmt(row['snr_db'])} | {row['weight_mode']} | "
            f"{int(row['num_success'])} | {_fmt(row['median_rmse_m'])} | "
            f"{_fmt(row['mean_rmse_m'])} | {_fmt(row['p90_rmse_m'])} | "
            f"{_fmt(row['outlier_rate'])} | {_fmt(row['median_y_nmse'])} | "
            f"{_fmt(row['median_runtime_s'])} | {_fmt(row['direct_vp_selected_rate'])} | "
            f"{_fmt(row['jnpp_selected_rate'])} | {_fmt(row['rollback_selected_rate'])} | "
            f"{_fmt(row['median_jnpp_runtime_s'])}"
        )


def _rmse_close(a: float, b: float, eps: float = 1.0e-15) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) / max(abs(b), eps) < 0.05


def _outlier_close(a: float, b: float) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) < 0.05


def _comparable(a: dict, b: dict) -> bool:
    return (
        _rmse_close(a["median_rmse_m"], b["median_rmse_m"])
        and _rmse_close(a["p90_rmse_m"], b["p90_rmse_m"])
        and _outlier_close(a["outlier_rate"], b["outlier_rate"])
    )


def _better(a: dict, b: dict) -> bool:
    return (
        np.isfinite(a["median_rmse_m"])
        and np.isfinite(a["p90_rmse_m"])
        and np.isfinite(a["outlier_rate"])
        and (
            a["median_rmse_m"] < b["median_rmse_m"] * 0.95
            or a["p90_rmse_m"] < b["p90_rmse_m"] * 0.95
            or a["outlier_rate"] < b["outlier_rate"] - 0.05
        )
    )


def _print_recommendations(summary: list[dict]) -> None:
    by_snr: dict[float, dict[str, dict]] = {}
    for row in summary:
        by_snr.setdefault(float(row["snr_db"]), {})[str(row["weight_mode"])] = row
    for snr_db in sorted(by_snr):
        group = by_snr[snr_db]
        equal = group.get("equal")
        exponential = group.get("exponential")
        inverse_rank = group.get("inverse_rank")
        print(f"\nSNR {snr_db:g} dB recommendation:")
        if equal is None or exponential is None or inverse_rank is None:
            print("Missing one or more weight modes; inspect jnpp_weight_pilot_summary.csv.")
        elif _comparable(equal, exponential):
            print("Equal JNPP is comparable to exponential. Prefer equal weights for the final algorithm.")
        elif _better(exponential, equal) and _comparable(inverse_rank, exponential):
            print("Inverse-rank JNPP matches exponential and is more defensible than exponential. Prefer inverse-rank WLS.")
        elif _better(exponential, equal) and _better(exponential, inverse_rank):
            print("Exponential weighting is empirically best but theoretically weakest. Do not freeze it before additional justification or sensitivity analysis of rho/min_weight.")
        else:
            print("No clear winner; inspect jnpp_weight_pilot_summary.csv.")


def run_sweep(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    snr_values = _parse_float_list(args.snr_list)
    weight_modes = _parse_str_list(args.weight_modes)
    seeds = list(range(int(args.seed0), int(args.seed0) + int(args.num_seeds)))
    base_config = default_config()
    rows = []
    for snr_db in snr_values:
        for seed in seeds:
            for mode in weight_modes:
                config = copy.deepcopy(base_config)
                config["SNR_dB"] = float(snr_db)
                config["seed"] = int(seed)
                config["jnpp_weight_mode"] = str(mode)
                config["jnpp_inverse_rank_eps"] = float(args.inverse_rank_eps)
                config["jnpp_exp_rank_rho"] = float(args.exp_rank_rho)
                config["jnpp_exp_min_weight"] = float(args.exp_min_weight)
                config["jnpp_normalize_weights"] = True
                config["jnpp_use_leave_one_out"] = True
                config["print_progress"] = False
                try:
                    results = run_single_proposed_diagnostic(config)
                    row = _extract_success_row(results, snr_db, seed, mode)
                except Exception as exc:  # noqa: BLE001 - pilot must continue per run.
                    traceback.print_exc()
                    row = _failure_row(snr_db, seed, mode, exc)
                rows.append(row)
                suffix = f", error={row['error_message']}" if row.get("error_message") else ""
                print(
                    f"snr_db={snr_db:g}, seed={seed}, weight_mode={mode}, "
                    f"branch={row.get('selected_branch', '')}, "
                    f"rmse_m={_fmt(row.get('ue_position_rmse_m'))}{suffix}",
                    flush=True,
                )
    return rows, _summarize(rows, snr_values, weight_modes, float(args.outlier_threshold))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot sweep for Robust JNPP weight modes.")
    parser.add_argument("--snr-list", default="-20,-25")
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--seed0", type=int, default=2025)
    parser.add_argument("--weight-modes", default="equal,exponential,inverse_rank")
    parser.add_argument("--outdir", default="results/jnpp_weight_pilot")
    parser.add_argument("--outlier-threshold", type=float, default=0.1)
    parser.add_argument("--inverse-rank-eps", type=float, default=1.0e-2)
    parser.add_argument("--exp-rank-rho", type=float, default=2.0)
    parser.add_argument("--exp-min-weight", type=float, default=0.05)
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    snr_values = _parse_float_list(args.snr_list)
    weight_modes = _parse_str_list(args.weight_modes)
    seeds = list(range(int(args.seed0), int(args.seed0) + int(args.num_seeds)))

    rows, summary = run_sweep(args)
    _write_csv(outdir / "jnpp_weight_pilot_raw.csv", rows, RAW_FIELDS)
    _write_csv(outdir / "jnpp_weight_pilot_summary.csv", summary, SUMMARY_FIELDS)
    config_record = {
        "snr_list": snr_values,
        "num_seeds": int(args.num_seeds),
        "seed0": int(args.seed0),
        "seeds": seeds,
        "weight_modes": weight_modes,
        "outlier_threshold": float(args.outlier_threshold),
        "inverse_rank_eps": float(args.inverse_rank_eps),
        "exp_rank_rho": float(args.exp_rank_rho),
        "exp_min_weight": float(args.exp_min_weight),
        "base_config": _jsonable(default_config()),
    }
    with (outdir / "jnpp_weight_pilot_config.json").open("w") as handle:
        json.dump(config_record, handle, indent=2, sort_keys=True)

    _print_summary(summary)
    _print_recommendations(summary)
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
