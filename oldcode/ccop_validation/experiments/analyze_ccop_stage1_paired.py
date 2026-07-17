"""Pair a new CCOP Stage-I validation route with the recorded frozen R3 route."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np
from scipy.stats import beta, binomtest


OLD_ROUTE = "R3_independent_ccop_jvp"
OLD_ROUTE_ID = "R3"
NEW_ROUTE = "evs_subspace_joint_geometry_stage1_ccop"


def _rows(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    return float(row[key])


def _bool(row: dict, key: str) -> bool:
    return str(row[key]).strip().lower() in {"1", "true", "yes"}


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> list[float]:
    alpha = 1.0 - float(confidence)
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return [lower, upper]


def _bootstrap(old: dict[str, np.ndarray], new: dict[str, np.ndarray], count: int) -> dict:
    rng = np.random.default_rng(20260716)
    n = len(old["position"])
    samples = rng.integers(0, n, size=(int(count), n))

    old_pos = old["position"][samples]
    new_pos = new["position"][samples]
    old_channel = old["channel"][samples]
    new_channel = new["channel"][samples]
    old_runtime = old["runtime"][samples]
    new_runtime = new["runtime"][samples]
    statistics = {
        "position_rmse_delta_m": np.sqrt(np.mean(new_pos**2, axis=1))
        - np.sqrt(np.mean(old_pos**2, axis=1)),
        "position_p95_delta_m": np.quantile(new_pos, 0.95, axis=1)
        - np.quantile(old_pos, 0.95, axis=1),
        "channel_median_delta": np.median(new_channel, axis=1)
        - np.median(old_channel, axis=1),
        "channel_p95_delta": np.quantile(new_channel, 0.95, axis=1)
        - np.quantile(old_channel, 0.95, axis=1),
        "runtime_median_ratio": np.median(new_runtime, axis=1)
        / np.maximum(np.median(old_runtime, axis=1), np.finfo(float).tiny),
    }
    return {
        key: {
            "estimate": float(
                (
                    np.sqrt(np.mean(new["position"] ** 2))
                    - np.sqrt(np.mean(old["position"] ** 2))
                )
                if key == "position_rmse_delta_m"
                else (
                    np.quantile(new["position"], 0.95)
                    - np.quantile(old["position"], 0.95)
                )
                if key == "position_p95_delta_m"
                else np.median(new["channel"]) - np.median(old["channel"])
                if key == "channel_median_delta"
                else np.quantile(new["channel"], 0.95)
                - np.quantile(old["channel"], 0.95)
                if key == "channel_p95_delta"
                else np.median(new["runtime"]) / np.median(old["runtime"])
            ),
            "ci95": [float(v) for v in np.quantile(values, [0.025, 0.975])],
        }
        for key, values in statistics.items()
    }


def analyze(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    old_trials = {
        int(row["seed"]): row
        for row in _rows(args.old_trials)
        if row["route"] == args.old_route and not _bool(row, "failed")
    }
    old_diagnostics = {
        int(row["seed"]): row
        for row in _rows(args.old_diagnostics)
        if row["route_id"] == args.old_route_id
    }
    new_trials = {
        int(row["seed"]): row
        for row in _rows(args.new_trials)
        if row["route"] == NEW_ROUTE
    }
    common = sorted(set(old_trials) & set(old_diagnostics) & set(new_trials))
    if len(common) != len(new_trials) or len(common) != len(old_trials):
        raise RuntimeError(
            f"route seed mismatch: common={len(common)}, old={len(old_trials)}, new={len(new_trials)}"
        )

    paired_rows = []
    for seed in common:
        old = old_trials[seed]
        old_diag = old_diagnostics[seed]
        new = new_trials[seed]
        hash_match = str(old_diag["y_noisy_hash"]) == str(new["y_noisy_hash"])
        paired_rows.append(
            {
                "seed": seed,
                "y_noisy_hash_match": hash_match,
                "old_position_error_m": _float(old, "position_error_m"),
                "new_position_error_m": _float(new, "position_error_m"),
                "old_clock_error_ns": _float(old, "clock_error_ns"),
                "new_clock_error_ns": _float(new, "clock_error_ns"),
                "old_channel_y_nmse": _float(old, "channel_y_nmse"),
                "new_channel_y_nmse": _float(new, "channel_y_nmse"),
                "old_raw_objective": _float(old, "raw_objective_final"),
                "new_raw_objective": _float(new, "raw_objective_final"),
                "old_deployment_runtime_s": _float(old, "deployment_runtime_s"),
                "new_deployment_runtime_s": _float(new, "stage1_runtime_s")
                + _float(new, "ccop_runtime_s"),
                "old_outlier": _bool(old, "outlier_flag"),
                "new_outlier": _bool(new, "outlier"),
            }
        )
    if not all(row["y_noisy_hash_match"] for row in paired_rows):
        raise RuntimeError("Y_noisy hash mismatch between old and new routes")

    arrays = {}
    for label, prefix in (("old", "old_"), ("new", "new_")):
        arrays[label] = {
            "position": np.array([row[prefix + "position_error_m"] for row in paired_rows]),
            "clock": np.array([row[prefix + "clock_error_ns"] for row in paired_rows]),
            "channel": np.array([row[prefix + "channel_y_nmse"] for row in paired_rows]),
            "raw": np.array([row[prefix + "raw_objective"] for row in paired_rows]),
            "runtime": np.array([row[prefix + "deployment_runtime_s"] for row in paired_rows]),
            "outlier": np.array([row[prefix + "outlier"] for row in paired_rows], dtype=bool),
        }

    n = len(paired_rows)
    old_outliers = int(np.sum(arrays["old"]["outlier"]))
    new_outliers = int(np.sum(arrays["new"]["outlier"]))
    rescued = int(np.sum(arrays["old"]["outlier"] & ~arrays["new"]["outlier"]))
    introduced = int(np.sum(~arrays["old"]["outlier"] & arrays["new"]["outlier"]))
    discordant = rescued + introduced
    mcnemar_p = (
        float(binomtest(min(rescued, introduced), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    report = {
        "n": n,
        "hash_matches": int(sum(row["y_noisy_hash_match"] for row in paired_rows)),
        "old_route": args.old_route,
        "new_route": NEW_ROUTE,
        "old": {
            "position_error_m": _summary(arrays["old"]["position"]),
            "clock_error_ns": _summary(arrays["old"]["clock"]),
            "channel_y_nmse": _summary(arrays["old"]["channel"]),
            "deployment_runtime_s": _summary(arrays["old"]["runtime"]),
            "outliers": old_outliers,
            "outlier_rate": old_outliers / n,
            "outlier_rate_ci95_exact": _binomial_interval(old_outliers, n),
        },
        "new": {
            "position_error_m": _summary(arrays["new"]["position"]),
            "clock_error_ns": _summary(arrays["new"]["clock"]),
            "channel_y_nmse": _summary(arrays["new"]["channel"]),
            "deployment_runtime_s": _summary(arrays["new"]["runtime"]),
            "stage1_position_error_m": _summary(
                np.array([_float(new_trials[seed], "stage1_position_error_m") for seed in common])
            ),
            "stage1_delay_rmse_ns": _summary(
                np.array([_float(new_trials[seed], "stage1_delay_rmse_ns") for seed in common])
            ),
            "outliers": new_outliers,
            "outlier_rate": new_outliers / n,
            "outlier_rate_ci95_exact": _binomial_interval(new_outliers, n),
        },
        "paired": {
            "rescued_outliers": rescued,
            "introduced_outliers": introduced,
            "mcnemar_exact_two_sided_p": mcnemar_p,
            "position_win_rate": float(np.mean(arrays["new"]["position"] < arrays["old"]["position"])),
            "clock_win_rate": float(np.mean(arrays["new"]["clock"] < arrays["old"]["clock"])),
            "channel_win_rate": float(np.mean(arrays["new"]["channel"] < arrays["old"]["channel"])),
            "raw_objective_non_degradation_rate": float(
                np.mean(arrays["new"]["raw"] <= arrays["old"]["raw"] + 1.0e-12)
            ),
        },
        "bootstrap": _bootstrap(arrays["old"], arrays["new"], args.bootstrap),
    }
    for label in ("old", "new"):
        correct = ~arrays[label]["outlier"]
        report[label]["conditional_correct_basin"] = {
            "n": int(np.count_nonzero(correct)),
            "position_error_m": _summary(arrays[label]["position"][correct]),
            "clock_error_ns": _summary(arrays[label]["clock"][correct]),
            "channel_y_nmse": _summary(arrays[label]["channel"][correct]),
        }
    return report, paired_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-trials", type=pathlib.Path, required=True)
    parser.add_argument("--old-trials", type=pathlib.Path, required=True)
    parser.add_argument("--old-diagnostics", type=pathlib.Path, required=True)
    parser.add_argument("--old-route", default=OLD_ROUTE)
    parser.add_argument("--old-route-id", default=OLD_ROUTE_ID)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def _markdown(report: dict) -> str:
    old = report["old"]
    new = report["new"]
    paired = report["paired"]
    boot = report["bootstrap"]
    return "\n".join(
        [
            "# Paired Stage-I + CCOP validation summary",
            "",
            f"- Trials: {report['n']} (Y_noisy hash matches: {report['hash_matches']})",
            f"- Old/new outliers: {old['outliers']} / {new['outliers']}",
            f"- Rescued/introduced outliers: {paired['rescued_outliers']} / {paired['introduced_outliers']}",
            f"- McNemar exact two-sided p: {paired['mcnemar_exact_two_sided_p']:.6g}",
            f"- Position RMSE: {old['position_error_m']['rmse']:.6g} m -> {new['position_error_m']['rmse']:.6g} m",
            f"- Position p95: {old['position_error_m']['p95']:.6g} m -> {new['position_error_m']['p95']:.6g} m",
            f"- Clock RMSE: {old['clock_error_ns']['rmse']:.6g} ns -> {new['clock_error_ns']['rmse']:.6g} ns",
            f"- Channel NMSE median: {old['channel_y_nmse']['median']:.6g} -> {new['channel_y_nmse']['median']:.6g}",
            f"- Channel NMSE p95: {old['channel_y_nmse']['p95']:.6g} -> {new['channel_y_nmse']['p95']:.6g}",
            f"- Median deployment runtime ratio: {boot['runtime_median_ratio']['estimate']:.6g} "
            f"(bootstrap 95% CI {boot['runtime_median_ratio']['ci95']})",
            f"- New outlier-rate exact 95% CI: {new['outlier_rate_ci95_exact']}",
            "",
            f"The old route is `{report['old_route']}`. The new route uses the same trial seeds and byte-identical Y_noisy observations.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    report, paired_rows = analyze(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paired_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out_dir / "paired_analysis.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    if paired_rows:
        with (args.out_dir / "paired_with_frozen_r3.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
