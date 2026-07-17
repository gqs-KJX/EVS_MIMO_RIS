"""Publication-oriented paired analysis for R1--R5 validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

from .run_ccop_full_validation import ROUTES


def _read(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict, field: str) -> float:
    return float(row[field])


def _b(row: dict, field: str) -> bool:
    return str(row[field]).lower() in {"1", "true", "yes"}


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _exact_binomial_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [np.nan, np.nan]
    from scipy.stats import beta

    alpha = 0.05
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return [low, high]


def _route_record(rows: list[dict], route_id: str, threshold: float) -> dict:
    route = ROUTES[route_id]
    selected = [row for row in rows if row["route"] == route and not _b(row, "failed")]
    position = np.asarray([_f(row, "position_error_m") for row in selected])
    clock = np.asarray([_f(row, "clock_error_ns") for row in selected])
    channel = np.asarray([_f(row, "channel_y_nmse") for row in selected])
    correct = position <= threshold

    def stats(values: np.ndarray) -> dict:
        return {
            "rmse": float(np.sqrt(np.mean(values**2))),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
        }

    return {
        "route_id": route_id,
        "route": route,
        "n": len(selected),
        "position_error_m": stats(position),
        "clock_error_ns": stats(clock),
        "channel_y_nmse": stats(channel),
        "wrong_basin_rate": float(np.mean(~correct)),
        "wrong_basin_count": int(np.sum(~correct)),
        "wrong_basin_exact_binomial_ci95": _exact_binomial_interval(
            int(np.sum(~correct)), len(selected)
        ),
        "conditional_correct_basin": {
            "n": int(np.sum(correct)),
            "position_rmse_m": float(np.sqrt(np.mean(position[correct] ** 2))) if np.any(correct) else np.nan,
            "position_median_m": float(np.median(position[correct])) if np.any(correct) else np.nan,
            "clock_rmse_ns": float(np.sqrt(np.mean(clock[correct] ** 2))) if np.any(correct) else np.nan,
            "channel_nmse_median": float(np.median(channel[correct])) if np.any(correct) else np.nan,
        },
        "runtime": {
            "stage3_median_s": float(np.median([_f(row, "route_runtime_s") for row in selected])),
            "deployment_median_s": float(np.median([_f(row, "deployment_runtime_s") for row in selected])),
        },
    }


def _paired(
    rows: list[dict], alternative_id: str, threshold: float, bootstrap: int, seed: int
) -> tuple[dict, list[dict]]:
    baseline = {
        int(row["trial_id"]): row
        for row in rows
        if row["route"] == ROUTES["R1"] and not _b(row, "failed")
    }
    alternative = {
        int(row["trial_id"]): row
        for row in rows
        if row["route"] == ROUTES[alternative_id] and not _b(row, "failed")
    }
    ids = sorted(set(baseline) & set(alternative))
    fields = (
        "position_error_m",
        "clock_error_ns",
        "channel_y_nmse",
        "regularized_objective_final",
        "route_runtime_s",
        "deployment_runtime_s",
    )
    left = {field: np.asarray([_f(baseline[index], field) for index in ids]) for field in fields}
    right = {field: np.asarray([_f(alternative[index], field) for index in ids]) for field in fields}
    baseline_outlier = left["position_error_m"] > threshold
    alternative_outlier = right["position_error_m"] > threshold
    rescued = int(np.sum(baseline_outlier & ~alternative_outlier))
    introduced = int(np.sum(~baseline_outlier & alternative_outlier))
    from scipy.stats import binomtest

    discordant = rescued + introduced
    mcnemar_p = float(
        binomtest(rescued, discordant, 0.5, alternative="two-sided").pvalue
    ) if discordant else 1.0
    rng = np.random.default_rng(int(seed))
    bootstrap_rows = []
    samples = {
        "position_rmse_difference_m": np.empty(bootstrap),
        "position_median_difference_m": np.empty(bootstrap),
        "position_p95_difference_m": np.empty(bootstrap),
        "clock_rmse_difference_ns": np.empty(bootstrap),
        "channel_nmse_median_difference": np.empty(bootstrap),
        "stage3_runtime_median_difference_s": np.empty(bootstrap),
    }
    for draw in range(bootstrap):
        indices = rng.integers(0, len(ids), len(ids))
        samples["position_rmse_difference_m"][draw] = (
            np.sqrt(np.mean(right["position_error_m"][indices] ** 2))
            - np.sqrt(np.mean(left["position_error_m"][indices] ** 2))
        )
        samples["position_median_difference_m"][draw] = np.median(
            right["position_error_m"][indices]
        ) - np.median(left["position_error_m"][indices])
        samples["position_p95_difference_m"][draw] = np.percentile(
            right["position_error_m"][indices], 95
        ) - np.percentile(left["position_error_m"][indices], 95)
        samples["clock_rmse_difference_ns"][draw] = (
            np.sqrt(np.mean(right["clock_error_ns"][indices] ** 2))
            - np.sqrt(np.mean(left["clock_error_ns"][indices] ** 2))
        )
        samples["channel_nmse_median_difference"][draw] = np.median(
            right["channel_y_nmse"][indices]
        ) - np.median(left["channel_y_nmse"][indices])
        samples["stage3_runtime_median_difference_s"][draw] = np.median(
            right["route_runtime_s"][indices]
        ) - np.median(left["route_runtime_s"][indices])
    intervals = {name: _interval(values) for name, values in samples.items()}
    for name, values in samples.items():
        bootstrap_rows.append(
            {
                "alternative_id": alternative_id,
                "metric": name,
                "estimate_mean": float(np.mean(values)),
                "ci95_low": intervals[name][0],
                "ci95_high": intervals[name][1],
            }
        )
    common_correct = ~baseline_outlier & ~alternative_outlier
    record = {
        "baseline_id": "R1",
        "alternative_id": alternative_id,
        "n_paired": len(ids),
        "paired_win_rates": {
            "position": float(np.mean(right["position_error_m"] < left["position_error_m"])),
            "clock": float(np.mean(right["clock_error_ns"] < left["clock_error_ns"])),
            "channel": float(np.mean(right["channel_y_nmse"] < left["channel_y_nmse"])),
        },
        "objective_non_degradation_rate": float(
            np.mean(right["regularized_objective_final"] <= left["regularized_objective_final"] + 1.0e-12)
        ),
        "outlier_rescued": rescued,
        "outlier_introduced": introduced,
        "outlier_rescued_exact_binomial_ci95": _exact_binomial_interval(
            rescued, int(np.sum(baseline_outlier))
        ),
        "outlier_introduced_exact_binomial_ci95": _exact_binomial_interval(
            introduced, int(np.sum(~baseline_outlier))
        ),
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "common_correct_basin": {
            "n": int(np.sum(common_correct)),
            "alternative_position_rmse_m": float(np.sqrt(np.mean(right["position_error_m"][common_correct] ** 2))) if np.any(common_correct) else np.nan,
            "baseline_position_rmse_m": float(np.sqrt(np.mean(left["position_error_m"][common_correct] ** 2))) if np.any(common_correct) else np.nan,
        },
        "paired_bootstrap_95_ci": intervals,
    }
    return record, bootstrap_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=91027)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    rows = _read(args.input_dir / "route_trials.csv")
    config_path = args.input_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    split_role = str(config.get("arguments", {}).get("split_role", "unknown"))
    route_records = [_route_record(rows, route_id, args.outlier_threshold_m) for route_id in ROUTES]
    paired_records = []
    bootstrap_rows = []
    for offset, alternative_id in enumerate(("R2", "R3", "R4", "R5")):
        record, samples = _paired(
            rows,
            alternative_id,
            args.outlier_threshold_m,
            int(args.bootstrap),
            int(args.seed) + offset,
        )
        paired_records.append(record)
        bootstrap_rows.extend(samples)
    output = {
        "input_dir": str(args.input_dir),
        "split_role": split_role,
        "outlier_threshold_m": float(args.outlier_threshold_m),
        "route_summaries": route_records,
        "paired_against_R1": paired_records,
        "interpretation_gate": (
            "held-out evidence; do not retune from these results"
            if split_role == "heldout"
            else (
                "independent validation evidence; tuning is allowed only before held-out freeze"
                if split_role == "validation"
                else "development/pilot evidence only"
            )
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "stage3_analysis.json"
    bootstrap_path = args.out_dir / "paired_bootstrap.csv"
    if not args.force_rerun and (summary_path.exists() or bootstrap_path.exists()):
        raise FileExistsError("analysis outputs exist; use --force-rerun")
    summary_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    with bootstrap_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]))
        writer.writeheader()
        writer.writerows(bootstrap_rows)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
