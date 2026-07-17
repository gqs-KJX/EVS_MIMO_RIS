"""Fit CP-NGC empirical thresholds from an independent validation output.

The script never modifies estimator code and refuses held-out inputs.  Green
and red thresholds use only correct-basin validation statistics; wrong-basin
labels are used only to report detection/false-green performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any

import numpy as np

from ..cp_ngc_covariance import (
    apply_empirical_cp_ngc_calibration,
    fit_empirical_cp_ngc_calibration,
)


def _read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    from scipy.stats import beta

    alpha = 1.0 - float(confidence)
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["threshold", "tpr", "fpr"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only empirical CP-NGC calibration")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--route-id", default="R3")
    parser.add_argument("--correct-trigger-rate", type=float, default=0.10)
    parser.add_argument("--correct-red-rate", type=float, default=0.02)
    parser.add_argument("--minimum-stratum-size", type=int, default=20)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.input_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("split_role", "")) != "validation":
        raise ValueError("calibration input must have split_role='validation'")
    rows = [
        row
        for row in _read_csv(args.input_dir / "cp_ngc_diagnostics.csv")
        if row.get("route_id") == str(args.route_id) and not row.get("error")
    ]
    records = [
        {
            "statistic": float(row["statistic"]),
            "correct_basin": _bool(row["correct_basin_evaluation_label"]),
            "stratum": str(row["reliability_stratum"]),
            "covariance_reliable": _bool(row["covariance_hard_certificate_reliable"]),
        }
        for row in rows
        if np.isfinite(float(row["statistic"]))
    ]
    calibration = fit_empirical_cp_ngc_calibration(
        records,
        correct_trigger_rate=float(args.correct_trigger_rate),
        correct_red_rate=float(args.correct_red_rate),
        minimum_stratum_size=int(args.minimum_stratum_size),
    )
    statuses = [
        apply_empirical_cp_ngc_calibration(
            record["statistic"], record["stratum"], calibration
        )
        for record in records
    ]
    correct_indices = [index for index, record in enumerate(records) if record["correct_basin"]]
    wrong_indices = [index for index, record in enumerate(records) if not record["correct_basin"]]
    correct_triggers = sum(statuses[index] != "green" for index in correct_indices)
    false_green = sum(statuses[index] == "green" for index in wrong_indices)
    wrong_detected = sum(statuses[index] != "green" for index in wrong_indices)
    correct_trigger_ci = _binomial_interval(correct_triggers, len(correct_indices))
    false_green_ci = _binomial_interval(false_green, len(wrong_indices))
    wrong_detected_ci = _binomial_interval(wrong_detected, len(wrong_indices))
    covariance_reliable_rate = float(
        np.mean([record["covariance_reliable"] for record in records])
    ) if records else float("nan")
    acceptance_targets = {
        "wrong_basin_detection_at_least_0.90": bool(
            wrong_indices and wrong_detected / len(wrong_indices) >= 0.90
        ),
        "correct_trigger_at_most_0.10": bool(
            correct_indices and correct_triggers / len(correct_indices) <= 0.10
        ),
        "false_green_at_most_0.05": bool(
            wrong_indices and false_green / len(wrong_indices) <= 0.05
        ),
        "covariance_reliable_for_every_hard_decision": bool(
            records and all(record["covariance_reliable"] for record in records)
        ),
    }
    hard_gate_passed = bool(all(acceptance_targets.values()))
    summary = {
        "source_split": "validation",
        "route_id": str(args.route_id),
        "n": len(records),
        "n_correct": len(correct_indices),
        "n_wrong": len(wrong_indices),
        "correct_trigger_rate": correct_triggers / len(correct_indices) if correct_indices else np.nan,
        "correct_trigger_rate_exact_95_ci": correct_trigger_ci,
        "wrong_basin_detection_rate": wrong_detected / len(wrong_indices) if wrong_indices else np.nan,
        "wrong_basin_detection_exact_95_ci": wrong_detected_ci,
        "false_green_rate": false_green / len(wrong_indices) if wrong_indices else np.nan,
        "false_green_rate_exact_95_ci": false_green_ci,
        "covariance_reliable_rate": covariance_reliable_rate,
        "acceptance_targets": acceptance_targets,
        "hard_gate_passed": hard_gate_passed,
        "deployment_status": "hard_certificate" if hard_gate_passed else "soft_score_or_gray_trigger_only",
    }
    statistics = np.asarray([record["statistic"] for record in records], dtype=float)
    labels = np.asarray([not record["correct_basin"] for record in records], dtype=bool)
    thresholds = np.unique(np.r_[[-np.inf], statistics, [np.inf]])
    roc_rows = []
    for threshold in thresholds:
        predicted_wrong = statistics >= threshold
        tp = int(np.sum(predicted_wrong & labels))
        fp = int(np.sum(predicted_wrong & ~labels))
        fn = int(np.sum(~predicted_wrong & labels))
        roc_rows.append(
            {
                "threshold": float(threshold),
                "tpr": tp / max(tp + fn, 1),
                "fpr": fp / max(int(np.sum(~labels)), 1),
                "precision": tp / max(int(np.sum(predicted_wrong)), 1),
                "recall": tp / max(tp + fn, 1),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protected = [args.out_dir / name for name in ("calibration.json", "summary.json", "roc_pr.csv")]
    if not args.force_rerun and any(path.exists() for path in protected):
        raise FileExistsError(f"outputs already exist under {args.out_dir}; use --force-rerun")
    (args.out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(args.out_dir / "roc_pr.csv", roc_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
