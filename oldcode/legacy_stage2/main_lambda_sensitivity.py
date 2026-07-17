"""Sweep Stage-I coupled-LS relative regularization for proposed diagnostics."""

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
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.config import default_config
    from src.main_single_proposed import _final_raw_objective, run_single_proposed_diagnostic
    from src.metrics import position_rmse, relative_nmse
else:
    from src.config import default_config
    from src.main_single_proposed import _final_raw_objective, run_single_proposed_diagnostic
    from src.metrics import position_rmse, relative_nmse


RAW_FIELDS = [
    "snr_db",
    "seed",
    "lambda_rel",
    "lambda_floor",
    "selected_branch",
    "ue_position_rmse_m",
    "y_nmse",
    "raw_objective",
    "total_runtime_s",
    "assignment_margin",
    "sigma_delta_t_ns",
    "rank1_ratio_max",
    "initial_z_residual",
    "direct_vp_good",
    "direct_vp_raw_objective",
    "direct_vp_nfev",
    "stage2_rescue_mode",
    "jnpp_num_candidates",
    "jnpp_best_objective",
    "jnpp_best_clock_std_ns",
    "jnpp_runtime_s",
    "error_message",
]

SUMMARY_FIELDS = [
    "lambda_rel",
    "num_success",
    "median_rmse_m",
    "mean_rmse_m",
    "p90_rmse_m",
    "outlier_rate_0p1m",
    "median_y_nmse",
    "median_runtime_s",
    "direct_vp_rate",
    "jnpp_selected_rate",
]


def _parse_float_list(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("--lambda-rel-list must contain at least one value")
    return values


def parse_lambda_list(text: str) -> list[float]:
    """Public wrapper retained for lightweight tests and ad-hoc imports."""
    return _parse_float_list(text)


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


def _extract_success_row(
    results: dict,
    snr_db: float,
    seed: int,
    lambda_rel: float,
    lambda_floor: float,
) -> dict:
    final = results.get("final", {})
    estimate_initial = results.get("estimate_initial", {})
    reliability = results.get("reliability", {})
    timing = results.get("timing", {})
    direct = _direct_branch(results)
    direct_quality = results.get("direct_vp_quality", {})
    rescue_diag = _rescue_diagnostics(results)

    row = {field: _nan() for field in RAW_FIELDS}
    row.update(
        {
            "snr_db": float(snr_db),
            "seed": int(seed),
            "lambda_rel": float(lambda_rel),
            "lambda_floor": float(lambda_floor),
            "selected_branch": str(results.get("selected_branch", "")),
            "error_message": "",
        }
    )

    if isinstance(final, dict) and "position_rmse" in final:
        row["ue_position_rmse_m"] = _safe_float(final.get("position_rmse"))
    elif isinstance(final, dict) and "p_u" in final:
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
            estimate_initial.get(
                "stage1_max_rank1_ratio",
                estimate_initial.get("rank1_ratio_max"),
            )
        )
        row["initial_z_residual"] = _safe_float(estimate_initial.get("initial_z_residual"))
    if isinstance(reliability, dict):
        if not np.isfinite(_safe_float(row["rank1_ratio_max"])):
            row["rank1_ratio_max"] = _safe_float(reliability.get("rank1_ratio_max"))
        if not np.isfinite(_safe_float(row["initial_z_residual"])):
            row["initial_z_residual"] = _safe_float(
                reliability.get("initial_z_residual")
            )

    if isinstance(direct_quality, dict):
        row["direct_vp_good"] = bool(direct_quality.get("good", False))
        row["direct_vp_raw_objective"] = _safe_float(
            direct_quality.get("raw_objective_final")
        )
    if math.isnan(_safe_float(row["direct_vp_raw_objective"])):
        direct_final = direct.get("final", {}) if isinstance(direct, dict) else {}
        row["direct_vp_raw_objective"] = _final_raw_objective(
            direct_final if isinstance(direct_final, dict) else {}
        )
    row["direct_vp_nfev"] = _direct_vp_nfev(direct)

    if isinstance(rescue_diag, dict) and rescue_diag:
        row["stage2_rescue_mode"] = str(rescue_diag.get("stage2_rescue_mode", ""))
        row["jnpp_num_candidates"] = _safe_int(rescue_diag.get("jnpp_num_candidates"))
        row["jnpp_best_objective"] = _safe_float(rescue_diag.get("jnpp_best_objective"))
        row["jnpp_best_clock_std_ns"] = _safe_float(
            rescue_diag.get("jnpp_best_clock_std_ns")
        )
        row["jnpp_runtime_s"] = _safe_float(
            rescue_diag.get("jnpp_runtime_total", rescue_diag.get("jnpp_runtime_s"))
        )
    else:
        row["stage2_rescue_mode"] = ""
    return row


def extract_metrics_from_result(
    results: dict,
    lambda_rel: float,
    lambda_floor: float,
    snr_db: float,
    seed: int,
) -> dict:
    """Public wrapper for extracting one raw CSV row from an existing result dict."""
    return _extract_success_row(
        results,
        snr_db=float(snr_db),
        seed=int(seed),
        lambda_rel=float(lambda_rel),
        lambda_floor=float(lambda_floor),
    )


def _failure_row(
    snr_db: float,
    seed: int,
    lambda_rel: float,
    lambda_floor: float,
    error: BaseException,
) -> dict:
    row = {field: _nan() for field in RAW_FIELDS}
    row.update(
        {
            "snr_db": float(snr_db),
            "seed": int(seed),
            "lambda_rel": float(lambda_rel),
            "lambda_floor": float(lambda_floor),
            "selected_branch": "",
            "stage2_rescue_mode": "",
            "error_message": f"{type(error).__name__}: {error}",
        }
    )
    return row


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


def _summarize(rows: list[dict], lambda_values: list[float]) -> list[dict]:
    summary = []
    for lambda_rel in sorted(lambda_values):
        group = [row for row in rows if float(row["lambda_rel"]) == float(lambda_rel)]
        success = [row for row in group if not row.get("error_message")]
        rmse = [_safe_float(row.get("ue_position_rmse_m")) for row in success]
        y_nmse = [_safe_float(row.get("y_nmse")) for row in success]
        runtime = [_safe_float(row.get("total_runtime_s")) for row in success]
        finite_rmse = np.asarray([value for value in rmse if np.isfinite(value)])
        num_success = int(finite_rmse.size)
        outlier_rate = (
            float(np.mean(finite_rmse > 0.1)) if finite_rmse.size else _nan()
        )
        direct_selected = [
            str(row.get("selected_branch", "")).startswith("direct_vp") for row in success
        ]
        jnpp_selected = [
            str(row.get("selected_branch", "")) == "ris_only_stage2_then_vp"
            and str(row.get("stage2_rescue_mode", "")) == "robust_jnpp"
            for row in success
        ]
        summary.append(
            {
                "lambda_rel": float(lambda_rel),
                "num_success": num_success,
                "median_rmse_m": _nanmedian(rmse),
                "mean_rmse_m": _nanmean(rmse),
                "p90_rmse_m": _nanpercentile(rmse, 90.0),
                "outlier_rate_0p1m": outlier_rate,
                "median_y_nmse": _nanmedian(y_nmse),
                "median_runtime_s": _nanmedian(runtime),
                "direct_vp_rate": float(np.mean(direct_selected)) if direct_selected else _nan(),
                "jnpp_selected_rate": float(np.mean(jnpp_selected)) if jnpp_selected else _nan(),
            }
        )
    return summary


def _write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, _nan()) for field in fields})


def _fmt(value: Any) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return "NA"
    return f"{value_float:.6e}"


def _print_summary(summary: list[dict]) -> None:
    print("\nlambda_rel | num_success | median_rmse_m | mean_rmse_m | p90_rmse_m | "
          "outlier_rate_0p1m | median_y_nmse | median_runtime_s | direct_vp_rate | "
          "jnpp_selected_rate")
    for row in summary:
        print(
            f"{_fmt(row['lambda_rel'])} | "
            f"{int(row['num_success'])} | "
            f"{_fmt(row['median_rmse_m'])} | "
            f"{_fmt(row['mean_rmse_m'])} | "
            f"{_fmt(row['p90_rmse_m'])} | "
            f"{_fmt(row['outlier_rate_0p1m'])} | "
            f"{_fmt(row['median_y_nmse'])} | "
            f"{_fmt(row['median_runtime_s'])} | "
            f"{_fmt(row['direct_vp_rate'])} | "
            f"{_fmt(row['jnpp_selected_rate'])}"
        )


def _relative_close(a: float, b: float, tolerance: float = 0.10) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) / max(abs(b), 1.0e-15) < tolerance


def _print_recommendation(summary: list[dict]) -> None:
    by_lambda = {float(row["lambda_rel"]): row for row in summary}
    center = by_lambda.get(1.0e-6)
    left = by_lambda.get(1.0e-7)
    right = by_lambda.get(1.0e-5)
    stable = False
    if center is not None and left is not None and right is not None:
        stable = all(
            [
                _relative_close(center["median_rmse_m"], left["median_rmse_m"]),
                _relative_close(center["median_rmse_m"], right["median_rmse_m"]),
                _relative_close(center["p90_rmse_m"], left["p90_rmse_m"]),
                _relative_close(center["p90_rmse_m"], right["p90_rmse_m"]),
            ]
        )
    if stable:
        print("lambda_rel=1e-6 appears stable and can be frozen.")
    else:
        print("lambda_rel=1e-6 is not stable; inspect sensitivity_summary.csv.")


def run_sweep(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    lambda_values = _parse_float_list(args.lambda_rel_list)
    seeds = list(range(int(args.seed0), int(args.seed0) + int(args.num_seeds)))
    base_config = default_config()
    rows = []
    for lambda_rel in lambda_values:
        for seed in seeds:
            config = copy.deepcopy(base_config)
            config["SNR_dB"] = float(args.snr_db)
            config["seed"] = int(seed)
            config["stage1_factor_reg_mode"] = "relative"
            config["stage1_factor_reg_rel"] = float(lambda_rel)
            config["stage1_factor_reg_floor"] = float(args.lambda_floor)
            config["print_progress"] = False
            try:
                results = run_single_proposed_diagnostic(config)
                row = _extract_success_row(
                    results,
                    snr_db=float(args.snr_db),
                    seed=int(seed),
                    lambda_rel=float(lambda_rel),
                    lambda_floor=float(args.lambda_floor),
                )
            except Exception as exc:  # noqa: BLE001 - sweep must continue per seed.
                traceback.print_exc()
                row = _failure_row(
                    snr_db=float(args.snr_db),
                    seed=int(seed),
                    lambda_rel=float(lambda_rel),
                    lambda_floor=float(args.lambda_floor),
                    error=exc,
                )
            rows.append(row)
            rmse_text = _fmt(row.get("ue_position_rmse_m"))
            branch_text = row.get("selected_branch", "")
            error_text = row.get("error_message", "")
            suffix = f", error={error_text}" if error_text else ""
            print(
                f"lambda_rel={lambda_rel:.1e}, seed={seed}, "
                f"branch={branch_text}, rmse_m={rmse_text}{suffix}",
                flush=True,
            )
    return rows, _summarize(rows, lambda_values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Stage-I coupled LS regularization lambda_rel."
    )
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=2025)
    parser.add_argument("--lambda-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--lambda-rel-list",
        default="1e-4,3e-4,1e-3,3e-3,1e-2",
    )
    parser.add_argument("--outdir", default="results/lambda_sensitivity")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lambda_values = _parse_float_list(args.lambda_rel_list)
    seeds = list(range(int(args.seed0), int(args.seed0) + int(args.num_seeds)))

    rows, summary = run_sweep(args)
    _write_csv(outdir / "lambda_sensitivity_raw.csv", rows, RAW_FIELDS)
    _write_csv(outdir / "lambda_sensitivity_summary.csv", summary, SUMMARY_FIELDS)

    config_record = {
        "snr_db": float(args.snr_db),
        "num_seeds": int(args.num_seeds),
        "seed0": int(args.seed0),
        "seeds": seeds,
        "lambda_floor": float(args.lambda_floor),
        "lambda_rel_list": lambda_values,
        "regularization_config_keys": {
            "mode": "stage1_factor_reg_mode",
            "lambda_rel": "stage1_factor_reg_rel",
            "lambda_floor": "stage1_factor_reg_floor",
        },
        "base_config": _jsonable(default_config()),
    }
    with (outdir / "lambda_sensitivity_config.json").open("w") as handle:
        json.dump(config_record, handle, indent=2, sort_keys=True)

    _print_summary(summary)
    _print_recommendation(summary)
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
