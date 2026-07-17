"""Analyze two routes emitted by the paired CCOP Stage-I runner."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np
from scipy.stats import beta, binomtest


OLD_ROUTE = "frozen_stage1_4d_jones_vp"
NEW_ROUTE = "evs_subspace_joint_geometry_stage1_ccop"


def _rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _summary(values: np.ndarray) -> dict[str, float]:
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
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return [lower, upper]


def _bootstrap(
    old: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
    count: int,
) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(20260716)
    n = len(old["position"])
    names = (
        "position_rmse_delta_m",
        "position_p95_delta_m",
        "clock_rmse_delta_ns",
        "channel_median_delta",
        "channel_p95_delta",
        "runtime_median_ratio",
        "outlier_rate_delta",
    )
    samples = {name: np.empty(int(count), dtype=float) for name in names}
    batch_size = 500
    for begin in range(0, int(count), batch_size):
        end = min(begin + batch_size, int(count))
        indices = rng.integers(0, n, size=(end - begin, n))
        old_pos = old["position"][indices]
        new_pos = new["position"][indices]
        old_clock = old["clock"][indices]
        new_clock = new["clock"][indices]
        old_channel = old["channel"][indices]
        new_channel = new["channel"][indices]
        old_runtime = old["runtime"][indices]
        new_runtime = new["runtime"][indices]
        samples["position_rmse_delta_m"][begin:end] = (
            np.sqrt(np.mean(new_pos**2, axis=1))
            - np.sqrt(np.mean(old_pos**2, axis=1))
        )
        samples["position_p95_delta_m"][begin:end] = (
            np.quantile(new_pos, 0.95, axis=1)
            - np.quantile(old_pos, 0.95, axis=1)
        )
        samples["clock_rmse_delta_ns"][begin:end] = (
            np.sqrt(np.mean(new_clock**2, axis=1))
            - np.sqrt(np.mean(old_clock**2, axis=1))
        )
        samples["channel_median_delta"][begin:end] = (
            np.median(new_channel, axis=1) - np.median(old_channel, axis=1)
        )
        samples["channel_p95_delta"][begin:end] = (
            np.quantile(new_channel, 0.95, axis=1)
            - np.quantile(old_channel, 0.95, axis=1)
        )
        samples["runtime_median_ratio"][begin:end] = np.median(
            new_runtime, axis=1
        ) / np.maximum(np.median(old_runtime, axis=1), np.finfo(float).tiny)
        samples["outlier_rate_delta"][begin:end] = np.mean(
            new["outlier"][indices], axis=1
        ) - np.mean(old["outlier"][indices], axis=1)

    estimates = {
        "position_rmse_delta_m": np.sqrt(np.mean(new["position"] ** 2))
        - np.sqrt(np.mean(old["position"] ** 2)),
        "position_p95_delta_m": np.quantile(new["position"], 0.95)
        - np.quantile(old["position"], 0.95),
        "clock_rmse_delta_ns": np.sqrt(np.mean(new["clock"] ** 2))
        - np.sqrt(np.mean(old["clock"] ** 2)),
        "channel_median_delta": np.median(new["channel"])
        - np.median(old["channel"]),
        "channel_p95_delta": np.quantile(new["channel"], 0.95)
        - np.quantile(old["channel"], 0.95),
        "runtime_median_ratio": np.median(new["runtime"])
        / np.median(old["runtime"]),
        "outlier_rate_delta": np.mean(new["outlier"])
        - np.mean(old["outlier"]),
    }
    return {
        name: {
            "estimate": float(estimates[name]),
            "ci95": [float(value) for value in np.quantile(samples[name], [0.025, 0.975])],
        }
        for name in names
    }


def analyze(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = _rows(args.trials)
    by_route: dict[str, dict[int, dict[str, str]]] = {}
    for route in (args.old_route, args.new_route):
        selected = {int(row["seed"]): row for row in rows if row["route"] == route}
        if len(selected) * 2 != len([row for row in rows if row["route"] in (args.old_route, args.new_route)]):
            raise RuntimeError("duplicate, missing, or extra route rows in paired CSV")
        by_route[route] = selected
    old_rows = by_route[args.old_route]
    new_rows = by_route[args.new_route]
    seeds = sorted(set(old_rows) & set(new_rows))
    if len(seeds) != len(old_rows) or len(seeds) != len(new_rows):
        raise RuntimeError(
            f"route seed mismatch: common={len(seeds)}, old={len(old_rows)}, new={len(new_rows)}"
        )

    paired_rows: list[dict[str, object]] = []
    for seed in seeds:
        old = old_rows[seed]
        new = new_rows[seed]
        paired_rows.append(
            {
                "seed": seed,
                "y_noisy_hash_match": old["y_noisy_hash"] == new["y_noisy_hash"],
                "resolved_config_hash_match": old["resolved_config_hash"]
                == new["resolved_config_hash"],
                "old_position_error_m": float(old["position_error_m"]),
                "new_position_error_m": float(new["position_error_m"]),
                "old_clock_error_ns": float(old["clock_error_ns"]),
                "new_clock_error_ns": float(new["clock_error_ns"]),
                "old_channel_y_nmse": float(old["channel_y_nmse"]),
                "new_channel_y_nmse": float(new["channel_y_nmse"]),
                "old_raw_objective": float(old["raw_objective_final"]),
                "new_raw_objective": float(new["raw_objective_final"]),
                "old_stage1_runtime_s": float(old["stage1_runtime_s"]),
                "new_stage1_runtime_s": float(new["stage1_runtime_s"]),
                "old_stage3_runtime_s": float(old["ccop_runtime_s"]),
                "new_stage3_runtime_s": float(new["ccop_runtime_s"]),
                "old_deployment_runtime_s": float(old["stage1_runtime_s"])
                + float(old["ccop_runtime_s"]),
                "new_deployment_runtime_s": float(new["stage1_runtime_s"])
                + float(new["ccop_runtime_s"]),
                "old_outlier": _bool(old["outlier"]),
                "new_outlier": _bool(new["outlier"]),
                "new_clock_certified": _bool(new["clock_certified"]),
                "new_stage1_position_error_m": float(new["stage1_position_error_m"]),
                "new_stage1_delay_rmse_ns": float(new["stage1_delay_rmse_ns"]),
                "new_assignment_margin": float(new["stage1_assignment_margin"]),
                "new_max_rank1_ratio": float(new["stage1_max_rank1_ratio"]),
                "new_projection_runtime_s": float(new["stage1_projection_runtime_s"]),
                "new_anchor_refresh_runtime_s": float(
                    new["stage1_jones_anchor_refresh_runtime_s"]
                ),
            }
        )

    if not all(bool(row["y_noisy_hash_match"]) for row in paired_rows):
        raise RuntimeError("Y_noisy hash mismatch between paired routes")
    if not all(bool(row["resolved_config_hash_match"]) for row in paired_rows):
        raise RuntimeError("resolved config hash mismatch between paired routes")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for label in ("old", "new"):
        arrays[label] = {
            "position": np.array(
                [row[f"{label}_position_error_m"] for row in paired_rows], dtype=float
            ),
            "clock": np.array(
                [row[f"{label}_clock_error_ns"] for row in paired_rows], dtype=float
            ),
            "channel": np.array(
                [row[f"{label}_channel_y_nmse"] for row in paired_rows], dtype=float
            ),
            "raw": np.array(
                [row[f"{label}_raw_objective"] for row in paired_rows], dtype=float
            ),
            "stage1_runtime": np.array(
                [row[f"{label}_stage1_runtime_s"] for row in paired_rows], dtype=float
            ),
            "stage3_runtime": np.array(
                [row[f"{label}_stage3_runtime_s"] for row in paired_rows], dtype=float
            ),
            "runtime": np.array(
                [row[f"{label}_deployment_runtime_s"] for row in paired_rows], dtype=float
            ),
            "outlier": np.array(
                [row[f"{label}_outlier"] for row in paired_rows], dtype=bool
            ),
        }
    if not all(np.all(np.isfinite(value)) for data in arrays.values() for key, value in data.items() if key != "outlier"):
        raise RuntimeError("nonfinite metric in paired route output")

    n = len(paired_rows)
    old_outliers = int(np.sum(arrays["old"]["outlier"]))
    new_outliers = int(np.sum(arrays["new"]["outlier"]))
    rescued = int(np.sum(arrays["old"]["outlier"] & ~arrays["new"]["outlier"]))
    introduced = int(np.sum(~arrays["old"]["outlier"] & arrays["new"]["outlier"]))
    shared = int(np.sum(arrays["old"]["outlier"] & arrays["new"]["outlier"]))
    discordant = rescued + introduced
    mcnemar_p = (
        float(binomtest(min(rescued, introduced), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    both_correct = ~arrays["old"]["outlier"] & ~arrays["new"]["outlier"]

    report: dict[str, object] = {
        "n": n,
        "csv_rows": len(rows),
        "hash_matches": int(sum(bool(row["y_noisy_hash_match"]) for row in paired_rows)),
        "config_hash_matches": int(
            sum(bool(row["resolved_config_hash_match"]) for row in paired_rows)
        ),
        "old_route": args.old_route,
        "new_route": args.new_route,
        "old": {
            "position_error_m": _summary(arrays["old"]["position"]),
            "clock_error_ns": _summary(arrays["old"]["clock"]),
            "channel_y_nmse": _summary(arrays["old"]["channel"]),
            "stage1_runtime_s": _summary(arrays["old"]["stage1_runtime"]),
            "stage3_runtime_s": _summary(arrays["old"]["stage3_runtime"]),
            "deployment_runtime_s": _summary(arrays["old"]["runtime"]),
            "outliers": old_outliers,
            "outlier_rate": old_outliers / n,
            "outlier_rate_ci95_exact": _binomial_interval(old_outliers, n),
        },
        "new": {
            "position_error_m": _summary(arrays["new"]["position"]),
            "clock_error_ns": _summary(arrays["new"]["clock"]),
            "channel_y_nmse": _summary(arrays["new"]["channel"]),
            "stage1_runtime_s": _summary(arrays["new"]["stage1_runtime"]),
            "stage3_runtime_s": _summary(arrays["new"]["stage3_runtime"]),
            "deployment_runtime_s": _summary(arrays["new"]["runtime"]),
            "stage1_position_error_m": _summary(
                np.array([row["new_stage1_position_error_m"] for row in paired_rows])
            ),
            "stage1_delay_rmse_ns": _summary(
                np.array([row["new_stage1_delay_rmse_ns"] for row in paired_rows])
            ),
            "projection_runtime_s": _summary(
                np.array([row["new_projection_runtime_s"] for row in paired_rows])
            ),
            "anchor_refresh_runtime_s": _summary(
                np.array([row["new_anchor_refresh_runtime_s"] for row in paired_rows])
            ),
            "clock_certified": int(
                sum(bool(row["new_clock_certified"]) for row in paired_rows)
            ),
            "outliers": new_outliers,
            "outlier_rate": new_outliers / n,
            "outlier_rate_ci95_exact": _binomial_interval(new_outliers, n),
        },
        "paired": {
            "rescued_outliers": rescued,
            "introduced_outliers": introduced,
            "introduced_rate_ci95_exact": _binomial_interval(introduced, n),
            "shared_outliers": shared,
            "mcnemar_exact_two_sided_p": mcnemar_p,
            "position_win_rate": float(
                np.mean(arrays["new"]["position"] < arrays["old"]["position"])
            ),
            "clock_win_rate": float(
                np.mean(arrays["new"]["clock"] < arrays["old"]["clock"])
            ),
            "channel_win_rate": float(
                np.mean(arrays["new"]["channel"] < arrays["old"]["channel"])
            ),
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
        report[label]["conditional_both_correct"] = {
            "n": int(np.count_nonzero(both_correct)),
            "position_error_m": _summary(arrays[label]["position"][both_correct]),
            "clock_error_ns": _summary(arrays[label]["clock"][both_correct]),
            "channel_y_nmse": _summary(arrays[label]["channel"][both_correct]),
        }
    return report, paired_rows


def _markdown(report: dict[str, object]) -> str:
    old = report["old"]
    new = report["new"]
    paired = report["paired"]
    bootstrap = report["bootstrap"]
    return "\n".join(
        [
            "# Final held-out paired Stage-I/Stage-III analysis",
            "",
            f"- Complete trials / CSV rows: {report['n']} / {report['csv_rows']}",
            f"- Y/config hash matches: {report['hash_matches']} / {report['config_hash_matches']}",
            f"- Old/new outliers: {old['outliers']} / {new['outliers']}",
            f"- Rescued/introduced/shared: {paired['rescued_outliers']} / {paired['introduced_outliers']} / {paired['shared_outliers']}",
            f"- McNemar exact two-sided p: {paired['mcnemar_exact_two_sided_p']:.6g}",
            f"- Position RMSE: {old['position_error_m']['rmse']:.6g} m -> {new['position_error_m']['rmse']:.6g} m",
            f"- Position median/p95: {old['position_error_m']['median']:.6g}/{old['position_error_m']['p95']:.6g} m -> {new['position_error_m']['median']:.6g}/{new['position_error_m']['p95']:.6g} m",
            f"- Clock RMSE: {old['clock_error_ns']['rmse']:.6g} ns -> {new['clock_error_ns']['rmse']:.6g} ns",
            f"- Channel NMSE median/p95: {old['channel_y_nmse']['median']:.6g}/{old['channel_y_nmse']['p95']:.6g} -> {new['channel_y_nmse']['median']:.6g}/{new['channel_y_nmse']['p95']:.6g}",
            f"- Median deployment runtime ratio: {bootstrap['runtime_median_ratio']['estimate']:.6g} (bootstrap 95% CI {bootstrap['runtime_median_ratio']['ci95']})",
            f"- New outlier-rate exact 95% CI: {new['outlier_rate_ci95_exact']}",
            f"- New clock certificates: {new['clock_certified']} / {report['n']}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=pathlib.Path, required=True)
    parser.add_argument("--old-route", default=OLD_ROUTE)
    parser.add_argument("--new-route", default=NEW_ROUTE)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser.parse_args()


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
    with (args.out_dir / "paired_heldout.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    failures = [row for row in paired_rows if bool(row["new_outlier"])]
    with (args.out_dir / "new_failure_taxonomy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(failures)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
