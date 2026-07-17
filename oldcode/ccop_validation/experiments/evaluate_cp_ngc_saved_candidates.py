"""Offline CP-NGC diagnostic for saved CCOP paired-run candidates.

The paired runner saves final position candidates but not the Stage-I joint
delay/local-geometry vector required by CP-NGC.  This diagnostic regenerates
only the deterministic noisy data and Stage-I estimate for every saved seed,
verifies the raw-data hash, and evaluates the saved CCOP position.  It does not
rerun either Stage-III route.

Because the current Stage-I implementation does not return an analytic
covariance, this script uses an explicitly oracle, leave-one-out empirical
covariance computed from Stage-I errors relative to simulation truth.  The
result diagnoses separability and covariance calibration; it is not a
deployable CP-NGC covariance estimator.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..cp_ngc import (
    cp_ngc_clock_vector,
    cp_ngc_geometry,
    cp_ngc_stage1_vector,
    cp_ngc_statistic,
)
from src.main_single_proposed import _make_data, run_stage1_only
from src.experiments.resource_control import apply_thread_limits
from .run_ccop_paired_mc import ROUTE_CCOP, _build_config


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _wrap_azimuth_entries(vector: np.ndarray, k_paths: int) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    for path in range(int(k_paths)):
        index = int(k_paths) + 3 * path + 2
        result[index] = float(np.angle(np.exp(1j * result[index])))
    return result


def _paired_spec(config_record: dict) -> dict:
    arguments = dict(config_record["arguments"])
    return {
        "snr_db": float(arguments["snr_db"]),
        "diagnostic_mode": str(arguments["diagnostic_mode"]),
        "outlier_threshold_m": float(arguments["outlier_threshold_m"]),
        "jones_mode": str(arguments["jones_mode"]),
        "old_max_iter": int(arguments["old_max_iter"]),
        "ccop_outer_max_iter": int(arguments["ccop_outer_max_iter"]),
        "clock_fft_size": int(arguments["clock_fft_size"]),
        "clock_abs_tol": float(arguments["clock_abs_tol"]),
        "clock_rel_tol": float(arguments["clock_rel_tol"]),
        "clock_max_intervals": int(arguments["clock_max_intervals"]),
        "use_old_incumbent": bool(arguments["use_old_incumbent"]),
        "old_vp_backend": str(arguments["old_vp_backend"]),
        "gpu_device": int(arguments["gpu_device"]),
    }


def _load_candidates(paired_dir: pathlib.Path, max_trials: int | None) -> tuple[list[dict], dict]:
    config_record = json.loads((paired_dir / "config.json").read_text(encoding="utf-8"))
    with (paired_dir / "paired_trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["route"] == ROUTE_CCOP and not _as_bool(row["failed"])
        ]
    rows.sort(key=lambda row: int(row["trial_id"]))
    if max_trials is not None:
        rows = rows[: int(max_trials)]
    expected_seeds = [int(seed) for seed in config_record["trial_seeds"]]
    for row in rows:
        trial_id = int(row["trial_id"])
        if int(row["seed"]) != expected_seeds[trial_id]:
            raise ValueError(f"seed mismatch in saved trial {trial_id}")
    return rows, config_record


def _regenerate_stage1(task: dict) -> dict:
    apply_thread_limits(int(task["blas_threads"]))
    trial_id = int(task["trial_id"])
    seed = int(task["seed"])
    config = _build_config(dict(task["spec"]), seed)
    data_start = time.perf_counter()
    data = _make_data(config)
    data_runtime = time.perf_counter() - data_start
    actual_hash = hashlib.sha256(
        np.ascontiguousarray(data["Y_noisy"]).view(np.uint8)
    ).hexdigest()[:20]
    stage1_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        stage1_record = run_stage1_only(data, config)
    stage1_runtime = time.perf_counter() - stage1_start
    estimate = stage1_record["estimate"]
    scene = data["scene"]
    z_hat = cp_ngc_stage1_vector(estimate, scene)
    truth = cp_ngc_geometry(scene["p_u_true"], scene)
    clock = cp_ngc_clock_vector(scene)
    target = truth + clock * float(scene["delta_t_true"])
    ris_residuals = np.asarray(
        estimate.get("stage1_ris_residuals", np.array([], dtype=float)), dtype=float
    ).reshape(-1)
    finite_ris = ris_residuals[np.isfinite(ris_residuals)]
    return {
        "trial_id": trial_id,
        "seed": seed,
        "hash_expected": str(task["hash_expected"]),
        "hash_actual": actual_hash,
        "hash_match": bool(actual_hash == str(task["hash_expected"])),
        "z_hat": z_hat,
        "target_truth": target,
        "p_true": np.asarray(scene["p_u_true"], dtype=float),
        "delta_t_true": float(scene["delta_t_true"]),
        "K": int(scene["K"]),
        "scene": {
            "K": int(scene["K"]),
            "ris_centers": np.asarray(scene["ris_centers"], dtype=float),
            "rotations": np.asarray(scene["rotations"], dtype=float),
            "d_RB": np.asarray(scene["d_RB"], dtype=float),
            "c0": float(scene["c0"]),
            "delta_f": float(scene["delta_f"]),
        },
        "assignment_margin": float(estimate.get("assignment_margin", np.nan)),
        "selected_clock_std_s": float(estimate.get("selected_clock_std", np.nan)),
        "max_ris_residual": float(np.max(finite_ris)) if finite_ris.size else float("nan"),
        "data_runtime_s": float(data_runtime),
        "stage1_runtime_s": float(stage1_runtime),
    }


def _write_stage1_rows(path: pathlib.Path, rows: list[dict]) -> None:
    dimension = int(np.asarray(rows[0]["z_hat"]).size)
    fields = [
        "trial_id",
        "seed",
        "hash_expected",
        "hash_actual",
        "hash_match",
        "assignment_margin",
        "selected_clock_std_s",
        "max_ris_residual",
        "data_runtime_s",
        "stage1_runtime_s",
        *[f"z_{index}" for index in range(dimension)],
        *[f"target_{index}" for index in range(dimension)],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(rows, key=lambda row: int(row["trial_id"])):
            row = {field: result.get(field, "") for field in fields}
            row.update({f"z_{index}": value for index, value in enumerate(result["z_hat"])})
            row.update(
                {f"target_{index}": value for index, value in enumerate(result["target_truth"])}
            )
            writer.writerow(row)


def _empirical_covariance(errors: np.ndarray) -> np.ndarray:
    if errors.shape[0] <= errors.shape[1] + 1:
        raise ValueError("too few Stage-I realizations for a full empirical CP-NGC covariance")
    covariance = np.cov(np.asarray(errors, dtype=float), rowvar=False, ddof=1)
    covariance = 0.5 * (covariance + covariance.T)
    scales = np.sqrt(np.diag(covariance))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise np.linalg.LinAlgError("Stage-I empirical covariance has a non-positive scale")
    correlation = covariance / np.outer(scales, scales)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(0.5 * (correlation + correlation.T))))
    if min_eigenvalue <= 0.0:
        raise np.linalg.LinAlgError(
            f"Stage-I empirical correlation is not positive definite: min eig {min_eigenvalue:.3e}"
        )
    return covariance


def _exact_interval(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    from scipy.stats import beta

    if total <= 0:
        return [float("nan"), float("nan")]
    tail = (1.0 - float(confidence)) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return [lower, upper]


def _rate_summary(flags: np.ndarray) -> dict:
    values = np.asarray(flags, dtype=bool)
    count = int(np.sum(values))
    total = int(values.size)
    return {
        "count": count,
        "total": total,
        "rate": float(count / total) if total else float("nan"),
        "exact_95_ci": _exact_interval(count, total),
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = np.asarray(scores, dtype=float)[np.asarray(labels, dtype=bool)]
    negative = np.asarray(scores, dtype=float)[~np.asarray(labels, dtype=bool)]
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0)) / comparisons.size)


def _evaluate(
    candidates: list[dict],
    stage1_rows: list[dict],
    *,
    alphas: list[float],
) -> tuple[list[dict], dict, np.ndarray]:
    from scipy.stats import chi2, kstest, spearmanr

    stage1_lookup = {int(row["trial_id"]): row for row in stage1_rows}
    dimension = int(np.asarray(stage1_rows[0]["z_hat"]).size)
    k_paths = int(stage1_rows[0]["K"])
    errors = np.vstack(
        [
            _wrap_azimuth_entries(
                np.asarray(stage1_lookup[int(candidate["trial_id"])]["z_hat"])
                - np.asarray(stage1_lookup[int(candidate["trial_id"])]["target_truth"]),
                k_paths,
            )
            for candidate in candidates
        ]
    )
    covariance_all = _empirical_covariance(errors)
    result_rows = []
    for index, candidate in enumerate(candidates):
        stage1 = stage1_lookup[int(candidate["trial_id"])]
        errors_loo = np.delete(errors, index, axis=0)
        covariance_loo = _empirical_covariance(errors_loo)
        bias_loo = np.mean(errors_loo, axis=0)
        p_candidate = np.array(
            [candidate["p_hat_x"], candidate["p_hat_y"], candidate["p_hat_z"]], dtype=float
        )
        candidate_diagnostic = cp_ngc_statistic(
            stage1["z_hat"], p_candidate, covariance_loo, stage1["scene"]
        )
        truth_diagnostic = cp_ngc_statistic(
            stage1["z_hat"], stage1["p_true"], covariance_loo, stage1["scene"]
        )
        bias_corrected_z = np.asarray(stage1["z_hat"], dtype=float) - bias_loo
        corrected_candidate_diagnostic = cp_ngc_statistic(
            bias_corrected_z, p_candidate, covariance_loo, stage1["scene"]
        )
        corrected_truth_diagnostic = cp_ngc_statistic(
            bias_corrected_z, stage1["p_true"], covariance_loo, stage1["scene"]
        )
        dof = int(candidate_diagnostic["dof"])
        statistic = float(candidate_diagnostic["statistic"])
        truth_statistic = float(truth_diagnostic["statistic"])
        corrected_statistic = float(corrected_candidate_diagnostic["statistic"])
        corrected_truth_statistic = float(corrected_truth_diagnostic["statistic"])
        row = {
            "trial_id": int(candidate["trial_id"]),
            "seed": int(candidate["seed"]),
            "hash_match": bool(stage1["hash_match"]),
            "position_error_m": float(candidate["position_error_m"]),
            "outlier_flag": _as_bool(candidate["outlier_flag"]),
            "boundary_hit": _as_bool(candidate["boundary_hit"]),
            "cp_ngc_statistic": statistic,
            "cp_ngc_truth_statistic": truth_statistic,
            "cp_ngc_dof": dof,
            "cp_ngc_pvalue": float(chi2.sf(statistic, dof)),
            "cp_ngc_truth_pvalue": float(chi2.sf(truth_statistic, dof)),
            "oracle_bias_corrected_statistic": corrected_statistic,
            "oracle_bias_corrected_truth_statistic": corrected_truth_statistic,
            "oracle_bias_corrected_pvalue": float(chi2.sf(corrected_statistic, dof)),
            "oracle_bias_corrected_truth_pvalue": float(
                chi2.sf(corrected_truth_statistic, dof)
            ),
            "cp_ngc_delta_t_gls_ns": float(candidate_diagnostic["delta_t_gls"] * 1.0e9),
            "projected_geometry_rank": int(candidate_diagnostic["projected_geometry_rank"]),
            "cert_information_min_eigenvalue": float(
                np.min(candidate_diagnostic["cert_information_eigenvalues"])
            ),
            "assignment_margin": float(stage1["assignment_margin"]),
            "selected_clock_std_ns": float(stage1["selected_clock_std_s"] * 1.0e9),
            "max_ris_residual": float(stage1["max_ris_residual"]),
            "stage1_runtime_s": float(stage1["stage1_runtime_s"]),
        }
        for alpha in alphas:
            suffix = f"a{alpha:g}".replace(".", "p")
            threshold = float(chi2.ppf(1.0 - alpha, dof))
            row[f"threshold_{suffix}"] = threshold
            row[f"rejected_{suffix}"] = bool(statistic > threshold)
            row[f"truth_rejected_{suffix}"] = bool(truth_statistic > threshold)
            row[f"oracle_bias_corrected_rejected_{suffix}"] = bool(
                corrected_statistic > threshold
            )
            row[f"oracle_bias_corrected_truth_rejected_{suffix}"] = bool(
                corrected_truth_statistic > threshold
            )
        result_rows.append(row)

    labels = np.asarray([row["outlier_flag"] for row in result_rows], dtype=bool)
    boundary = np.asarray([row["boundary_hit"] for row in result_rows], dtype=bool)
    statistics = np.asarray([row["cp_ngc_statistic"] for row in result_rows], dtype=float)
    truth_statistics = np.asarray(
        [row["cp_ngc_truth_statistic"] for row in result_rows], dtype=float
    )
    corrected_statistics = np.asarray(
        [row["oracle_bias_corrected_statistic"] for row in result_rows], dtype=float
    )
    corrected_truth_statistics = np.asarray(
        [row["oracle_bias_corrected_truth_statistic"] for row in result_rows], dtype=float
    )
    position_errors = np.asarray([row["position_error_m"] for row in result_rows], dtype=float)
    dof = int(result_rows[0]["cp_ngc_dof"])
    summary: dict[str, Any] = {
        "method": "direct CP-NGC with oracle leave-one-out empirical Stage-I covariance",
        "n_trials": int(len(result_rows)),
        "n_inlier": int(np.sum(~labels)),
        "n_outlier": int(np.sum(labels)),
        "dof": dof,
        "all_hashes_match": bool(all(row["hash_match"] for row in result_rows)),
        "all_projected_geometry_rank_three": bool(
            all(row["projected_geometry_rank"] == 3 for row in result_rows)
        ),
        "score_auc": _auc(statistics, labels),
        "statistic_inlier_median": float(np.median(statistics[~labels])),
        "statistic_inlier_p95": float(np.quantile(statistics[~labels], 0.95)),
        "statistic_outlier_median": float(np.median(statistics[labels])),
        "statistic_outlier_min": float(np.min(statistics[labels])),
        "truth_statistic_mean": float(np.mean(truth_statistics)),
        "truth_statistic_p95": float(np.quantile(truth_statistics, 0.95)),
        "truth_chi_square_ks": {
            "statistic": float(kstest(truth_statistics, chi2(df=dof).cdf).statistic),
            "pvalue": float(kstest(truth_statistics, chi2(df=dof).cdf).pvalue),
        },
        "candidate_inlier_chi_square_ks": {
            "statistic": float(kstest(statistics[~labels], chi2(df=dof).cdf).statistic),
            "pvalue": float(kstest(statistics[~labels], chi2(df=dof).cdf).pvalue),
        },
        "spearman_statistic_position_error": {
            "rho": float(spearmanr(statistics, position_errors).statistic),
            "pvalue": float(spearmanr(statistics, position_errors).pvalue),
        },
        "thresholds": {},
        "oracle_bias_corrected": {
            "score_auc": _auc(corrected_statistics, labels),
            "statistic_inlier_median": float(np.median(corrected_statistics[~labels])),
            "statistic_inlier_p95": float(np.quantile(corrected_statistics[~labels], 0.95)),
            "statistic_outlier_median": float(np.median(corrected_statistics[labels])),
            "statistic_outlier_min": float(np.min(corrected_statistics[labels])),
            "truth_statistic_mean": float(np.mean(corrected_truth_statistics)),
            "truth_statistic_p95": float(np.quantile(corrected_truth_statistics, 0.95)),
            "truth_chi_square_ks": {
                "statistic": float(
                    kstest(corrected_truth_statistics, chi2(df=dof).cdf).statistic
                ),
                "pvalue": float(
                    kstest(corrected_truth_statistics, chi2(df=dof).cdf).pvalue
                ),
            },
            "thresholds": {},
        },
    }
    for alpha in alphas:
        suffix = f"a{alpha:g}".replace(".", "p")
        rejected = np.asarray([row[f"rejected_{suffix}"] for row in result_rows], dtype=bool)
        truth_rejected = np.asarray(
            [row[f"truth_rejected_{suffix}"] for row in result_rows], dtype=bool
        )
        summary["thresholds"][str(alpha)] = {
            "chi_square_threshold": float(chi2.ppf(1.0 - alpha, dof)),
            "outlier_detection": _rate_summary(rejected[labels]),
            "inlier_false_rejection": _rate_summary(rejected[~labels]),
            "truth_false_rejection": _rate_summary(truth_rejected),
            "boundary_outlier_detection": _rate_summary(rejected[labels & boundary]),
            "interior_outlier_detection": _rate_summary(rejected[labels & ~boundary]),
        }
        corrected_rejected = np.asarray(
            [
                row[f"oracle_bias_corrected_rejected_{suffix}"]
                for row in result_rows
            ],
            dtype=bool,
        )
        corrected_truth_rejected = np.asarray(
            [
                row[f"oracle_bias_corrected_truth_rejected_{suffix}"]
                for row in result_rows
            ],
            dtype=bool,
        )
        summary["oracle_bias_corrected"]["thresholds"][str(alpha)] = {
            "chi_square_threshold": float(chi2.ppf(1.0 - alpha, dof)),
            "outlier_detection": _rate_summary(corrected_rejected[labels]),
            "inlier_false_rejection": _rate_summary(corrected_rejected[~labels]),
            "truth_false_rejection": _rate_summary(corrected_truth_rejected),
            "boundary_outlier_detection": _rate_summary(
                corrected_rejected[labels & boundary]
            ),
            "interior_outlier_detection": _rate_summary(
                corrected_rejected[labels & ~boundary]
            ),
        }
    empirical_threshold = float(np.quantile(statistics[~labels], 0.95, method="higher"))
    empirical_rejected = statistics > empirical_threshold
    summary["oracle_empirical_5pct_threshold"] = {
        "threshold": empirical_threshold,
        "outlier_detection": _rate_summary(empirical_rejected[labels]),
        "inlier_false_rejection": _rate_summary(empirical_rejected[~labels]),
        "boundary_outlier_detection": _rate_summary(empirical_rejected[labels & boundary]),
        "interior_outlier_detection": _rate_summary(empirical_rejected[labels & ~boundary]),
    }
    corrected_empirical_threshold = float(
        np.quantile(corrected_statistics[~labels], 0.95, method="higher")
    )
    corrected_empirical_rejected = corrected_statistics > corrected_empirical_threshold
    summary["oracle_bias_corrected"]["oracle_empirical_5pct_threshold"] = {
        "threshold": corrected_empirical_threshold,
        "outlier_detection": _rate_summary(corrected_empirical_rejected[labels]),
        "inlier_false_rejection": _rate_summary(corrected_empirical_rejected[~labels]),
        "boundary_outlier_detection": _rate_summary(
            corrected_empirical_rejected[labels & boundary]
        ),
        "interior_outlier_detection": _rate_summary(
            corrected_empirical_rejected[labels & ~boundary]
        ),
    }
    scales = np.sqrt(np.diag(covariance_all))
    correlation = covariance_all / np.outer(scales, scales)
    mean_error = np.mean(errors, axis=0)
    summary["covariance_diagnostics"] = {
        "dimension": dimension,
        "standard_deviations": scales,
        "correlation_condition_number": float(np.linalg.cond(correlation)),
        "correlation_min_eigenvalue": float(np.min(np.linalg.eigvalsh(correlation))),
        "mean_stage1_error": mean_error,
        "absolute_bias_over_standard_deviation": np.abs(mean_error) / scales,
    }
    return result_rows, summary, covariance_all


def _write_result_rows(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(path: pathlib.Path, summary: dict) -> None:
    primary = summary["thresholds"]["0.05"]
    empirical = summary["oracle_empirical_5pct_threshold"]
    corrected = summary["oracle_bias_corrected"]["thresholds"]["0.05"]
    corrected_empirical = summary["oracle_bias_corrected"][
        "oracle_empirical_5pct_threshold"
    ]
    lines = [
        "# Offline CP-NGC on saved CCOP candidates",
        "",
        "This is an oracle diagnostic, not a deployable certification result. The current",
        "Stage-I code does not provide `C_z`, so each trial uses a leave-one-out empirical",
        "covariance estimated from the other simulation errors relative to ground truth.",
        (
            "The candidate and Stage-I statistic use the same noisy realization."
            if summary["all_hashes_match"]
            else "The regenerated same-seed Stage-I observations do not match the saved raw-data hashes."
        ),
        "",
        f"- Trials/inliers/outliers: {summary['n_trials']} / {summary['n_inlier']} / {summary['n_outlier']}",
        f"- Reconstructed raw-data hashes all match: {summary['all_hashes_match']}",
        f"- Projected geometry rank is three for all candidates: {summary['all_projected_geometry_rank_three']}",
        f"- CP-NGC score AUC: {summary['score_auc']:.6g}",
        "",
        "## Chi-square 95% gate",
        "",
        f"- Threshold (df={summary['dof']}): {primary['chi_square_threshold']:.6g}",
        f"- Outlier detection: {primary['outlier_detection']['count']}/{primary['outlier_detection']['total']} ({primary['outlier_detection']['rate']:.3%})",
        f"- Inlier false rejection: {primary['inlier_false_rejection']['count']}/{primary['inlier_false_rejection']['total']} ({primary['inlier_false_rejection']['rate']:.3%})",
        f"- Boundary outlier detection: {primary['boundary_outlier_detection']['count']}/{primary['boundary_outlier_detection']['total']}",
        f"- Interior outlier detection: {primary['interior_outlier_detection']['count']}/{primary['interior_outlier_detection']['total']}",
        f"- Truth-position rejection: {primary['truth_false_rejection']['count']}/{primary['truth_false_rejection']['total']} ({primary['truth_false_rejection']['rate']:.3%})",
        "",
        "## Oracle empirical 5% inlier gate",
        "",
        f"- Threshold: {empirical['threshold']:.6g}",
        f"- Outlier detection: {empirical['outlier_detection']['count']}/{empirical['outlier_detection']['total']} ({empirical['outlier_detection']['rate']:.3%})",
        f"- Inlier false rejection: {empirical['inlier_false_rejection']['count']}/{empirical['inlier_false_rejection']['total']} ({empirical['inlier_false_rejection']['rate']:.3%})",
        f"- Boundary/interior detection: {empirical['boundary_outlier_detection']['count']}/{empirical['boundary_outlier_detection']['total']} and {empirical['interior_outlier_detection']['count']}/{empirical['interior_outlier_detection']['total']}",
        "",
        "The empirical gate uses the known inlier labels and is reported only as a score-separation diagnostic.",
        "",
        "## Oracle leave-one-out Stage-I bias correction",
        "",
        "This subtracts the mean Stage-I error learned from the other truth-labelled trials;",
        "it is a calibration diagnosis and is not part of the current CP-NGC implementation.",
        "",
        f"- Chi-square outlier detection: {corrected['outlier_detection']['count']}/{corrected['outlier_detection']['total']} ({corrected['outlier_detection']['rate']:.3%})",
        f"- Chi-square inlier false rejection: {corrected['inlier_false_rejection']['count']}/{corrected['inlier_false_rejection']['total']} ({corrected['inlier_false_rejection']['rate']:.3%})",
        f"- Chi-square truth-position rejection: {corrected['truth_false_rejection']['count']}/{corrected['truth_false_rejection']['total']} ({corrected['truth_false_rejection']['rate']:.3%})",
        f"- Empirical-gate outlier detection: {corrected_empirical['outlier_detection']['count']}/{corrected_empirical['outlier_detection']['total']} ({corrected_empirical['outlier_detection']['rate']:.3%})",
        f"- Empirical-gate inlier false rejection: {corrected_empirical['inlier_false_rejection']['count']}/{corrected_empirical['inlier_false_rejection']['total']} ({corrected_empirical['inlier_false_rejection']['rate']:.3%})",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-dir", type=pathlib.Path, default=pathlib.Path("results/ccop_paired_100")
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--reconstruct-only", action="store_true")
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help=(
            "Continue as a same-seed sensitivity diagnostic when the original "
            "Y_noisy cannot be reproduced bitwise. The mismatch is recorded."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.jobs <= 0 or args.blas_threads <= 0:
        raise ValueError("--jobs and --blas-threads must be positive")
    candidates, config_record = _load_candidates(args.paired_dir, args.max_trials)
    if not candidates:
        raise ValueError("no successful saved CCOP candidates found")
    output_dir = args.output_dir or (args.paired_dir / "cp_ngc_offline")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / "stage1_vectors.csv"
    protected = [
        output_dir / "cp_ngc_trials.csv",
        output_dir / "summary.json",
        output_dir / "summary.md",
        output_dir / "covariance_z.npy",
    ]
    if not args.force and any(path.exists() for path in protected):
        raise FileExistsError(f"analysis outputs already exist under {output_dir}; use --force")

    spec = _paired_spec(config_record)
    tasks = [
        {
            "trial_id": int(row["trial_id"]),
            "seed": int(row["seed"]),
            "hash_expected": str(row["shared_data_hash"]),
            "spec": spec,
            "blas_threads": int(args.blas_threads),
        }
        for row in candidates
    ]
    start = time.perf_counter()
    stage1_rows = []
    if int(args.jobs) == 1:
        for number, task in enumerate(tasks, start=1):
            stage1_rows.append(_regenerate_stage1(task))
            print(f"Stage-I {number}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {executor.submit(_regenerate_stage1, task): task for task in tasks}
            for number, future in enumerate(as_completed(futures), start=1):
                stage1_rows.append(future.result())
                print(f"Stage-I {number}/{len(tasks)}", flush=True)
    stage1_rows.sort(key=lambda row: int(row["trial_id"]))
    _write_stage1_rows(stage1_path, stage1_rows)
    if not all(bool(row["hash_match"]) for row in stage1_rows):
        mismatches = [int(row["trial_id"]) for row in stage1_rows if not row["hash_match"]]
        if not bool(args.allow_hash_mismatch):
            raise RuntimeError(f"regenerated Y_noisy hash mismatch for trials {mismatches}")
        print(
            "WARNING: continuing with same-seed regenerated Stage-I despite "
            f"Y_noisy hash mismatches for {len(mismatches)} trials",
            flush=True,
        )
    if args.reconstruct_only:
        print(f"reconstructed {len(stage1_rows)} Stage-I vectors; evaluation skipped", flush=True)
        return

    result_rows, summary, covariance = _evaluate(
        candidates, stage1_rows, alphas=[0.05, 0.01]
    )
    summary["runtime_s"] = float(time.perf_counter() - start)
    summary["source_paired_dir"] = str(args.paired_dir)
    summary["caveats"] = [
        "C_z is an oracle leave-one-out empirical covariance based on simulation truth.",
        "No candidate-position covariance correction or held-out fold is available.",
        "Only the saved SNR/configuration is evaluated.",
    ]
    if bool(summary["all_hashes_match"]):
        summary["caveats"].insert(
            1, "The saved candidate and Stage-I z_hat use the same noisy realization."
        )
    if not bool(summary["all_hashes_match"]):
        summary["caveats"].insert(
            0,
            "Regenerated same-seed Y_noisy hashes do not match the saved run; results are sensitivity diagnostics, not exact paired certification.",
        )
    _write_result_rows(output_dir / "cp_ngc_trials.csv", result_rows)
    np.save(output_dir / "covariance_z.npy", covariance)
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_summary_markdown(output_dir / "summary.md", summary)
    primary = summary["thresholds"]["0.05"]
    print(
        "CP-NGC 95% gate: detected "
        f"{primary['outlier_detection']['count']}/{primary['outlier_detection']['total']} "
        "outliers; false-rejected "
        f"{primary['inlier_false_rejection']['count']}/{primary['inlier_false_rejection']['total']} inliers",
        flush=True,
    )


if __name__ == "__main__":
    main()
