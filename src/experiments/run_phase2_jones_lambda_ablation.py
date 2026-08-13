"""Paired Phase-II directional-Jones regularization ablation.

For every receiver/SNR/trial cell, all requested ``lambda_k`` values reuse the
same noisy observation and the same frozen Phase-I output.  Only the
dimensionless per-path coefficient in the Gram-scaled Phase-II penalty is
changed.  ``lambda_k=0`` is therefore the data-only free-Jones control, while
``lambda_k=1`` reproduces the fixed weight stated in the paper.

The default is deliberately a one-trial smoke run.  A paper campaign must set
``--n-trials`` explicitly and should retain both a difficult SNR and a high-SNR
point so that catastrophic failures and a possible regularization floor are
both visible.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterable

import numpy as np

from ..utils import scipy_is_available
from ..validation_artifacts import validation_environment
from .final_mksc_ccop_common import (
    TRIAL_FIELDS,
    Stage1Cache,
    _binomial_interval,
    make_paper_config,
    make_shared_data,
    run_paper_variant,
    save_resolved_config,
    save_run_manifest,
    write_csv,
)
from .resource_control import apply_thread_limits
from .run_paper_ablation_figures import make_nested_receiver_mode_data


DEFAULT_LAMBDAS = "0,0.1,1,10"
DEFAULT_SNRS = "-10,30"
DEFAULT_RECEIVER_MODES = "full_6d,scalar"
REFERENCE_LAMBDA = 1.0

LAMBDA_DIAGNOSTIC_FIELDS = (
    "phase2_lambda_k",
    "lambda_jones_per_path",
    "lambda_jones_min",
    "lambda_jones_max",
    "jones_regularizer_objective_final",
    "jones_direction_error_mean_deg",
    "jones_direction_error_max_deg",
)
LAMBDA_TRIAL_FIELDS = (
    tuple(dict.fromkeys(TRIAL_FIELDS)) + LAMBDA_DIAGNOSTIC_FIELDS
)

SUMMARY_FIELDS = (
    "receiver_mode",
    "snr_db",
    "phase2_lambda_k",
    "n",
    "n_failed",
    "n_success",
    "failure_rate",
    "position_rmse_m",
    "position_conditional_rmse_m",
    "position_median_m",
    "position_p90_m",
    "position_p95_m",
    "clock_rmse_ns",
    "clock_p95_ns",
    "channel_nmse_median",
    "channel_nmse_p95",
    "noisy_fit_nmse_median",
    "noisy_fit_nmse_p95",
    "raw_objective_median",
    "jones_regularizer_objective_median",
    "jones_direction_error_mean_median_deg",
    "jones_direction_error_mean_p95_deg",
    "catastrophic_rate",
    "catastrophic_ci_low",
    "catastrophic_ci_high",
    "catastrophic_ci_method",
    "runtime_median_s",
    "stage1_hash_count",
    "y_noisy_hash_count",
)

PAIRED_FIELDS = (
    "receiver_mode",
    "snr_db",
    "reference_lambda_k",
    "candidate_lambda_k",
    "n_pairs",
    "n_metric_pairs",
    "position_win_rate",
    "clock_win_rate",
    "channel_win_rate",
    "jones_direction_win_rate",
    "raw_objective_non_degradation_rate",
    "rescued_outliers",
    "introduced_outliers",
    "mcnemar_exact_p",
    "paired_position_rmse_difference_m",
    "paired_position_rmse_difference_ci_low_m",
    "paired_position_rmse_difference_ci_high_m",
    "paired_raw_objective_difference_median",
    "paired_noisy_fit_nmse_difference_median",
    "paired_jones_direction_difference_median_deg",
    "bootstrap_replicates",
)


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _float_grid(value: str) -> list[float]:
    return [float(item) for item in _csv_list(value)]


def _trial_seeds(root_seed: int, count: int) -> list[int]:
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(int(root_seed)).spawn(int(count))
    ]


def _lambda_config(config: dict, value: float, maximum: float) -> dict:
    """Return a config whose effective per-path weight is exactly ``value``."""
    resolved = copy.deepcopy(config)
    options = dict(resolved.get("global_vp", {}))
    options.update(
        {
            "mode": "jones_regularized",
            "jones_lambda0": float(value),
            "jones_lambda_min": 0.0,
            "jones_lambda_max": max(float(maximum), 1.0),
            # Disable the legacy tau-dependent rescaling.  Pinning tau to one
            # makes lambda_jones_per_path equal to the requested fixed weight,
            # even if a future Stage-I estimate records stage1_jones_tau.
            "jones_tau": 1.0,
            "jones_tau_min": 1.0,
            "jones_tau_max": 1.0,
            "jones_snr_eps": 0.0,
        }
    )
    resolved["global_vp"] = options
    return resolved


def _successful(row: dict[str, Any]) -> bool:
    return str(row.get("failed", "False")).lower() != "true"


def _finite(rows: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        if not _successful(row):
            continue
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def _run_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    apply_thread_limits(int(task["blas_threads"]))
    full_config = make_paper_config(
        int(task["seed"]),
        float(task["snr_db"]),
        diagnostic_mode=str(task["diagnostic_mode"]),
        overrides={"receiver_mode": "full_6d"},
    )
    full_data = make_shared_data(full_config)
    rows: list[dict[str, Any]] = []
    maximum_lambda = max(float(value) for value in task["lambda_grid"])

    for receiver_mode in task["receiver_modes"]:
        mode_config = make_paper_config(
            int(task["seed"]),
            float(task["snr_db"]),
            diagnostic_mode=str(task["diagnostic_mode"]),
            overrides={"receiver_mode": str(receiver_mode)},
        )
        data = make_nested_receiver_mode_data(
            full_data, str(receiver_mode), mode_config
        )
        mode_config["noise_variance"] = float(data["noise_variance"])
        cache = Stage1Cache(data, mode_config)

        mode_rows = []
        for lambda_k in task["lambda_grid"]:
            config = _lambda_config(mode_config, float(lambda_k), maximum_lambda)
            row = run_paper_variant(
                "proposed",
                data=data,
                config=config,
                cache=cache,
                suite="phase2_jones_lambda",
                x_name="phase2_lambda_k",
                x_value=float(lambda_k),
                trial_id=int(task["trial_id"]),
                outlier_threshold_m=float(task["outlier_threshold_m"]),
            )
            row["phase2_lambda_k"] = float(lambda_k)
            if _successful(row):
                effective_min = float(row.get("lambda_jones_min", np.nan))
                effective_max = float(row.get("lambda_jones_max", np.nan))
                if not (
                    np.isclose(
                        effective_min, float(lambda_k), rtol=0.0, atol=1.0e-14
                    )
                    and np.isclose(
                        effective_max, float(lambda_k), rtol=0.0, atol=1.0e-14
                    )
                ):
                    raise RuntimeError(
                        "effective Jones weights differ from the requested fixed "
                        f"lambda_k={lambda_k}: min={effective_min}, max={effective_max}"
                    )
            mode_rows.append(row)

        y_hashes = {str(row["y_noisy_hash"]) for row in mode_rows}
        if len(y_hashes) != 1:
            raise RuntimeError(
                f"lambda sweep did not share Y_noisy for receiver={receiver_mode}"
            )
        stage1_hashes = {
            str(row["stage1_output_hash"])
            for row in mode_rows
            if _successful(row) and str(row.get("stage1_output_hash", ""))
        }
        if len(stage1_hashes) > 1:
            raise RuntimeError(
                "lambda sweep changed the Phase-I output: "
                f"receiver={receiver_mode}, hashes={sorted(stage1_hashes)}"
            )
        rows.extend(mode_rows)
    return rows


def _summarize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["receiver_mode"]),
            float(row["snr_db"]),
            float(row["phase2_lambda_k"]),
        )
        grouped.setdefault(key, []).append(row)

    output = []
    for (receiver_mode, snr_db, lambda_k), selected in sorted(grouped.items()):
        success = [row for row in selected if _successful(row)]
        conditional = [
            row
            for row in success
            if str(row.get("outlier", "False")).lower() != "true"
        ]
        position = _finite(success, "position_error_m")
        conditional_position = _finite(conditional, "position_error_m")
        clock = _finite(success, "clock_error_ns")
        channel = _finite(success, "channel_nmse")
        noisy_fit = _finite(success, "noisy_fit_nmse")
        raw_objective = _finite(success, "raw_objective_final")
        regularizer = _finite(success, "jones_regularizer_objective_final")
        jones_error = _finite(success, "jones_direction_error_mean_deg")
        runtime = _finite(success, "deployment_runtime_s")
        catastrophes = sum(
            (not _successful(row))
            or str(row.get("outlier", "False")).lower() == "true"
            for row in selected
        )
        ci_low, ci_high, ci_method = _binomial_interval(
            catastrophes, len(selected)
        )
        output.append(
            {
                "receiver_mode": receiver_mode,
                "snr_db": snr_db,
                "phase2_lambda_k": lambda_k,
                "n": len(selected),
                "n_failed": len(selected) - len(success),
                "n_success": len(success),
                "failure_rate": (len(selected) - len(success)) / len(selected),
                "position_rmse_m": (
                    float(np.sqrt(np.mean(position**2)))
                    if position.size
                    else float("nan")
                ),
                "position_conditional_rmse_m": (
                    float(np.sqrt(np.mean(conditional_position**2)))
                    if conditional_position.size
                    else float("nan")
                ),
                "position_median_m": _percentile(position, 50),
                "position_p90_m": _percentile(position, 90),
                "position_p95_m": _percentile(position, 95),
                "clock_rmse_ns": (
                    float(np.sqrt(np.mean(clock**2)))
                    if clock.size
                    else float("nan")
                ),
                "clock_p95_ns": _percentile(clock, 95),
                "channel_nmse_median": _percentile(channel, 50),
                "channel_nmse_p95": _percentile(channel, 95),
                "noisy_fit_nmse_median": _percentile(noisy_fit, 50),
                "noisy_fit_nmse_p95": _percentile(noisy_fit, 95),
                "raw_objective_median": _percentile(raw_objective, 50),
                "jones_regularizer_objective_median": _percentile(
                    regularizer, 50
                ),
                "jones_direction_error_mean_median_deg": _percentile(
                    jones_error, 50
                ),
                "jones_direction_error_mean_p95_deg": _percentile(
                    jones_error, 95
                ),
                "catastrophic_rate": catastrophes / len(selected),
                "catastrophic_ci_low": ci_low,
                "catastrophic_ci_high": ci_high,
                "catastrophic_ci_method": ci_method,
                "runtime_median_s": _percentile(runtime, 50),
                "stage1_hash_count": len(
                    {
                        str(row.get("stage1_output_hash", ""))
                        for row in success
                        if str(row.get("stage1_output_hash", ""))
                    }
                ),
                "y_noisy_hash_count": len(
                    {str(row.get("y_noisy_hash", "")) for row in selected}
                ),
            }
        )
    return output


def _bootstrap_rmse_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if reference.size == 0 or replicates <= 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, reference.size, size=(int(replicates), reference.size))
    differences = np.sqrt(np.mean(candidate[indices] ** 2, axis=1)) - np.sqrt(
        np.mean(reference[indices] ** 2, axis=1)
    )
    return float(np.percentile(differences, 2.5)), float(
        np.percentile(differences, 97.5)
    )


def _paired_summary(
    rows: Iterable[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, float], dict[tuple[int, int], dict[float, dict[str, Any]]]
    ] = {}
    for row in rows:
        cell = (str(row["receiver_mode"]), float(row["snr_db"]))
        trial = (int(row["trial_id"]), int(row["seed"]))
        grouped.setdefault(cell, {}).setdefault(trial, {})[
            float(row["phase2_lambda_k"])
        ] = row

    rng = np.random.default_rng(int(bootstrap_seed))
    output = []
    for (receiver_mode, snr_db), trials in sorted(grouped.items()):
        candidates = sorted(
            {
                value
                for by_lambda in trials.values()
                for value in by_lambda
                if not np.isclose(value, REFERENCE_LAMBDA)
            }
        )
        for candidate_lambda in candidates:
            all_pairs = [
                (by_lambda[REFERENCE_LAMBDA], by_lambda[candidate_lambda])
                for by_lambda in trials.values()
                if REFERENCE_LAMBDA in by_lambda and candidate_lambda in by_lambda
            ]
            metric_pairs = [
                pair
                for pair in all_pairs
                if _successful(pair[0]) and _successful(pair[1])
            ]

            def paired_values(key: str) -> tuple[np.ndarray, np.ndarray]:
                reference_values = []
                candidate_values = []
                for reference, candidate in metric_pairs:
                    try:
                        left = float(reference[key])
                        right = float(candidate[key])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if np.isfinite(left) and np.isfinite(right):
                        reference_values.append(left)
                        candidate_values.append(right)
                return np.asarray(reference_values), np.asarray(candidate_values)

            ref_position, cand_position = paired_values("position_error_m")
            ref_clock, cand_clock = paired_values("clock_error_ns")
            ref_channel, cand_channel = paired_values("channel_nmse")
            ref_raw, cand_raw = paired_values("raw_objective_final")
            ref_fit, cand_fit = paired_values("noisy_fit_nmse")
            ref_jones, cand_jones = paired_values(
                "jones_direction_error_mean_deg"
            )
            reference_outlier = np.asarray(
                [
                    (not _successful(pair[0]))
                    or str(pair[0].get("outlier", "False")).lower() == "true"
                    for pair in all_pairs
                ],
                dtype=bool,
            )
            candidate_outlier = np.asarray(
                [
                    (not _successful(pair[1]))
                    or str(pair[1].get("outlier", "False")).lower() == "true"
                    for pair in all_pairs
                ],
                dtype=bool,
            )
            rescued = int(np.sum(reference_outlier & ~candidate_outlier))
            introduced = int(np.sum(~reference_outlier & candidate_outlier))
            discordant = rescued + introduced
            try:
                from scipy.stats import binomtest

                mcnemar_p = (
                    float(
                        binomtest(
                            min(rescued, introduced), discordant, 0.5
                        ).pvalue
                    )
                    if discordant
                    else 1.0
                )
            except ImportError:
                mcnemar_p = float("nan")

            rmse_difference = (
                float(
                    np.sqrt(np.mean(cand_position**2))
                    - np.sqrt(np.mean(ref_position**2))
                )
                if ref_position.size
                else float("nan")
            )
            rmse_low, rmse_high = _bootstrap_rmse_difference(
                ref_position,
                cand_position,
                replicates=int(bootstrap_replicates),
                rng=rng,
            )
            output.append(
                {
                    "receiver_mode": receiver_mode,
                    "snr_db": snr_db,
                    "reference_lambda_k": REFERENCE_LAMBDA,
                    "candidate_lambda_k": candidate_lambda,
                    "n_pairs": len(all_pairs),
                    "n_metric_pairs": len(metric_pairs),
                    "position_win_rate": (
                        float(np.mean(cand_position < ref_position))
                        if ref_position.size
                        else float("nan")
                    ),
                    "clock_win_rate": (
                        float(np.mean(cand_clock < ref_clock))
                        if ref_clock.size
                        else float("nan")
                    ),
                    "channel_win_rate": (
                        float(np.mean(cand_channel < ref_channel))
                        if ref_channel.size
                        else float("nan")
                    ),
                    "jones_direction_win_rate": (
                        float(np.mean(cand_jones < ref_jones))
                        if ref_jones.size
                        else float("nan")
                    ),
                    "raw_objective_non_degradation_rate": (
                        float(np.mean(cand_raw <= ref_raw + 1.0e-12))
                        if ref_raw.size
                        else float("nan")
                    ),
                    "rescued_outliers": rescued,
                    "introduced_outliers": introduced,
                    "mcnemar_exact_p": mcnemar_p,
                    "paired_position_rmse_difference_m": rmse_difference,
                    "paired_position_rmse_difference_ci_low_m": rmse_low,
                    "paired_position_rmse_difference_ci_high_m": rmse_high,
                    "paired_raw_objective_difference_median": (
                        float(np.median(cand_raw - ref_raw))
                        if ref_raw.size
                        else float("nan")
                    ),
                    "paired_noisy_fit_nmse_difference_median": (
                        float(np.median(cand_fit - ref_fit))
                        if ref_fit.size
                        else float("nan")
                    ),
                    "paired_jones_direction_difference_median_deg": (
                        float(np.median(cand_jones - ref_jones))
                        if ref_jones.size
                        else float("nan")
                    ),
                    "bootstrap_replicates": int(bootstrap_replicates),
                }
            )
    return output


def _plot_from_summary(summary_csv: pathlib.Path, out_dir: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {summary_csv}")

    groups: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["receiver_mode"]), float(row["snr_db"])), []
        ).append(row)

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    panels = (
        ("position_rmse_m", "Position RMSE (mm)", 1.0e3, False),
        ("noisy_fit_nmse_median", "Raw-domain fit NMSE (dB)", 1.0, True),
        (
            "jones_direction_error_mean_median_deg",
            "Jones direction error (deg)",
            1.0,
            False,
        ),
        ("catastrophic_rate", "Catastrophic rate (%)", 100.0, False),
    )
    for axis, (key, ylabel, scale, decibels) in zip(axes.flat, panels):
        for (receiver_mode, snr_db), selected in sorted(groups.items()):
            ordered = sorted(selected, key=lambda row: float(row["phase2_lambda_k"]))
            x = np.asarray([float(row["phase2_lambda_k"]) for row in ordered])
            y = np.asarray([float(row[key]) for row in ordered])
            if decibels:
                y = 10.0 * np.log10(np.maximum(y, 1.0e-300))
            else:
                y *= scale
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=1.4,
                label=f"{receiver_mode}, {snr_db:g} dB",
            )
        axis.axvline(REFERENCE_LAMBDA, color="0.55", linestyle="--", linewidth=1.0)
        axis.set_xscale("symlog", linthresh=0.05, linscale=0.6)
        axis.set_xlabel(r"Phase-II $\lambda_k$")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    figure.savefig(out_dir / "phase2_jones_lambda_ablation.pdf")
    figure.savefig(out_dir / "phase2_jones_lambda_ablation.png", dpi=220)
    plt.close(figure)


def _write_markdown(
    path: pathlib.Path,
    summary: Iterable[dict[str, Any]],
    paired: Iterable[dict[str, Any]],
) -> None:
    lines = [
        "# Phase-II Jones-penalty ablation",
        "",
        "All lambda values within a receiver/SNR/trial cell use the same noisy observation and the same Phase-I output. Lambda=0 is the free-Jones control; lambda=1 is the paper setting.",
        "",
        "| Receiver | SNR (dB) | lambda | N | Position RMSE (m) | Position p95 (m) | Clock RMSE (ns) | Channel NMSE median | Raw-fit NMSE median | Jones error median (deg) | Catastrophic rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {receiver_mode} | {snr_db:g} | {phase2_lambda_k:g} | {n} | "
            "{position_rmse_m:.6g} | {position_p95_m:.6g} | {clock_rmse_ns:.6g} | "
            "{channel_nmse_median:.6g} | {noisy_fit_nmse_median:.6g} | "
            "{jones_direction_error_mean_median_deg:.6g} | {catastrophic_rate:.3%} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Paired comparisons against lambda=1",
            "",
            "A negative RMSE difference favors the candidate lambda. Confidence intervals are paired bootstrap intervals; catastrophic outcomes are compared by exact McNemar tests.",
            "",
            "| Receiver | SNR (dB) | Candidate lambda | Pairs | Delta position RMSE (m) | 95% CI | Rescued | Introduced | McNemar p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            "| {receiver_mode} | {snr_db:g} | {candidate_lambda_k:g} | {n_pairs} | "
            "{paired_position_rmse_difference_m:.6g} | "
            "[{paired_position_rmse_difference_ci_low_m:.6g}, {paired_position_rmse_difference_ci_high_m:.6g}] | "
            "{rescued_outliers} | {introduced_outliers} | {mcnemar_exact_p:.6g} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-grid", default=DEFAULT_LAMBDAS)
    parser.add_argument("--snr-grid", default=DEFAULT_SNRS)
    parser.add_argument("--receiver-modes", default=DEFAULT_RECEIVER_MODES)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--diagnostic-mode", choices=("fast", "performance"), default="performance"
    )
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/phase2_jones_lambda_smoke"),
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    args.lambda_grid = _float_grid(args.lambda_grid)
    args.snr_grid = _float_grid(args.snr_grid)
    args.receiver_modes = _csv_list(args.receiver_modes)
    if args.n_trials < 1 or args.jobs < 1 or args.blas_threads < 1:
        parser.error("--n-trials, --jobs, and --blas-threads must be positive")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be nonnegative")
    if not args.lambda_grid or any(
        not np.isfinite(value) or value < 0.0 for value in args.lambda_grid
    ):
        parser.error("--lambda-grid must contain finite nonnegative values")
    if len(set(args.lambda_grid)) != len(args.lambda_grid):
        parser.error("--lambda-grid values must be unique")
    if REFERENCE_LAMBDA not in args.lambda_grid:
        parser.error("--lambda-grid must contain the paper reference lambda=1")
    if not args.snr_grid or any(not np.isfinite(value) for value in args.snr_grid):
        parser.error("--snr-grid must contain finite values")
    valid_modes = {"scalar", "dual_pol", "full_6d"}
    unknown_modes = set(args.receiver_modes) - valid_modes
    if unknown_modes:
        parser.error(f"unknown receiver modes: {sorted(unknown_modes)}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_thread_limits(int(args.blas_threads))
    out_dir = args.out_dir
    trials_csv = out_dir / "phase2_jones_lambda_trials.csv"
    summary_csv = out_dir / "phase2_jones_lambda_summary.csv"
    paired_csv = out_dir / "phase2_jones_lambda_paired.csv"
    if args.plot_only:
        _plot_from_summary(summary_csv, out_dir)
        return
    if trials_csv.exists() and not args.force_rerun:
        raise FileExistsError(
            f"{trials_csv} exists; choose a new directory or use --force-rerun"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [sys.executable, "-m", __spec__.name, *(argv or sys.argv[1:])]
    )
    print(f"Command: {command}", flush=True)
    environment = validation_environment(
        command, repo_root=pathlib.Path(__file__).resolve().parents[2]
    )
    environment["scipy_optimizer_available"] = bool(scipy_is_available())
    save_run_manifest(
        out_dir,
        command=command,
        arguments=vars(args),
        environment=environment,
    )
    seeds = _trial_seeds(args.seed, args.n_trials)
    base_config = make_paper_config(
        seeds[0],
        args.snr_grid[0],
        diagnostic_mode=args.diagnostic_mode,
        overrides={"receiver_mode": "full_6d"},
    )
    save_resolved_config(out_dir, base_config, "resolved_base_config.json")
    (out_dir / "phase2_lambda_overrides.json").write_text(
        json.dumps(
            {
                "lambda_grid": args.lambda_grid,
                "reference_lambda_k": REFERENCE_LAMBDA,
                "global_vp_overrides": {
                    "mode": "jones_regularized",
                    "jones_lambda_min": 0.0,
                    "jones_tau": 1.0,
                    "jones_tau_min": 1.0,
                    "jones_tau_max": 1.0,
                    "jones_snr_eps": 0.0,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    tasks = [
        {
            "trial_id": trial_id,
            "seed": seed,
            "snr_db": snr_db,
            "lambda_grid": args.lambda_grid,
            "receiver_modes": args.receiver_modes,
            "diagnostic_mode": args.diagnostic_mode,
            "outlier_threshold_m": args.outlier_threshold_m,
            "blas_threads": args.blas_threads,
        }
        for snr_db in args.snr_grid
        for trial_id, seed in enumerate(seeds)
    ]
    rows: list[dict[str, Any]] = []
    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            rows.extend(_run_task(task))
            print(
                f"[{index}/{len(tasks)}] SNR={task['snr_db']:g} dB seed={task['seed']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {executor.submit(_run_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                rows.extend(future.result())
                print(
                    f"[{index}/{len(tasks)}] SNR={task['snr_db']:g} dB seed={task['seed']}",
                    flush=True,
                )

    rows.sort(
        key=lambda row: (
            str(row["receiver_mode"]),
            float(row["snr_db"]),
            int(row["trial_id"]),
            float(row["phase2_lambda_k"]),
        )
    )
    write_csv(trials_csv, rows, LAMBDA_TRIAL_FIELDS)
    summary = _summarize(rows)
    paired = _paired_summary(
        rows,
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.seed) + 1,
    )
    write_csv(summary_csv, summary, SUMMARY_FIELDS)
    write_csv(paired_csv, paired, PAIRED_FIELDS)
    _write_markdown(out_dir / "summary.md", summary, paired)
    _plot_from_summary(summary_csv, out_dir)
    (out_dir / "completion.json").write_text(
        json.dumps(
            {
                "complete": True,
                "tasks": len(tasks),
                "rows": len(rows),
                "failed_rows": sum(not _successful(row) for row in rows),
                "same_phase1_and_noise_assertions": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
