"""Reproducible per-seed C1/C2/C4 CP-NGC covariance pilot."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import pathlib
import shlex
import sys
import time

import numpy as np

from src.ccop_jvp import refine_ccop_jvp
from ..cp_ngc import cp_ngc_stage1_vector, cp_ngc_statistic
from ..cp_ngc_covariance import (
    linearized_stage1_covariance,
    one_way_heldout_cp_ngc,
    parametric_bootstrap_stage1_covariance,
)
from src.main_single_proposed import _make_data, run_stage1_only
from src.validation_artifacts import (
    array_sha256,
    canonical_hash,
    deterministic_stage1_output,
    validation_environment,
)
from .run_ccop_paired_mc import _build_config, _jsonable


FIELDS = [
    "seed",
    "failed",
    "error",
    "position_error_m",
    "boundary_hit",
    "resolved_config_hash",
    "y_noisy_hash",
    "stage1_output_hash",
    "candidate_hash",
    "c1_statistic",
    "c1_rank",
    "c1_fisher_rank",
    "c1_covariance_reliable",
    "c1_runtime_s",
    "c2_statistic",
    "c2_rank",
    "c2_valid_replicates",
    "c2_failed_replicates",
    "c2_runtime_s",
    "c4_statistic",
    "c4_rank",
    "c4_candidate_position_error_m",
    "c4_position_covariance_valid",
    "c4_heldout_certificate_valid",
    "c4_runtime_s",
]


def _run(seed: int, spec: dict) -> dict:
    row = {field: "" for field in FIELDS}
    row.update({"seed": int(seed), "failed": True, "error": "not_run"})
    try:
        config = _build_config(spec, int(seed))
        data = _make_data(config)
        config["noise_variance"] = float(data["noise_variance"])
        with contextlib.redirect_stdout(io.StringIO()):
            stage1 = run_stage1_only(data, config)["estimate"]
            candidate = refine_ccop_jvp(
                data["Y_noisy"], stage1, data["scene"], config, incumbent=None
            )
        z_hat = cp_ngc_stage1_vector(stage1, data["scene"])
        candidate_hash = canonical_hash(
            {
                "p_u": candidate["p_u"],
                "delta_t": candidate["delta_t"],
                "x_hat": candidate["x_hat"],
                "objective": candidate["total_objective_final"],
            }
        )
        common = {
            "seed": int(seed),
            "failed": False,
            "error": "",
            "position_error_m": float(
                np.linalg.norm(
                    np.asarray(candidate["p_u"]) - np.asarray(data["scene"]["p_u_true"])
                )
            ),
            "boundary_hit": bool(
                np.any(
                    np.minimum(
                        np.asarray(candidate["p_u"]) - np.asarray(config["ue_bounds"])[:, 0],
                        np.asarray(config["ue_bounds"])[:, 1] - np.asarray(candidate["p_u"]),
                    )
                    <= float(config["global_vp"].get("boundary_tol_m", 0.02))
                )
            ),
            "resolved_config_hash": canonical_hash(config),
            "y_noisy_hash": array_sha256(data["Y_noisy"]),
            "stage1_output_hash": canonical_hash(deterministic_stage1_output(stage1)),
            "candidate_hash": candidate_hash,
        }
        c1_start = time.perf_counter()
        c1 = linearized_stage1_covariance(
            data["Y_noisy"], stage1, data["scene"], data["noise_variance"]
        )
        c1_stat = cp_ngc_statistic(
            z_hat, candidate["p_u"], c1["covariance_z"], data["scene"]
        )
        common.update(
            {
                "c1_statistic": float(c1_stat["statistic"]),
                "c1_rank": int(c1_stat["projected_geometry_rank"]),
                "c1_fisher_rank": int(c1["fisher_rank_before_floor"]),
                "c1_covariance_reliable": bool(
                    c1["covariance_reliable_for_hard_certificate"]
                ),
                "c1_runtime_s": float(time.perf_counter() - c1_start),
            }
        )
        details: dict = {
            "candidate": {
                "p_u": candidate["p_u"],
                "delta_t": candidate["delta_t"],
                "raw_objective_final": candidate["raw_objective_final"],
                "total_objective_final": candidate["total_objective_final"],
                "clock_certified": candidate["clock_certified"],
                "optimizer": candidate["optimizer"],
            },
            "c1": {
                key: value
                for key, value in c1.items()
                if key not in {"fisher_effective_scaled"}
            },
            "c1_statistic": c1_stat,
        }
        bootstrap_count = int(spec["bootstrap_count"])
        if bootstrap_count > 0:
            c2_start = time.perf_counter()
            c2 = parametric_bootstrap_stage1_covariance(
                stage1,
                data["scene"],
                config,
                data["noise_variance"],
                n_bootstrap=bootstrap_count,
                bootstrap_seed=int(seed) ^ 0xC20C2026,
            )
            c2_stat = cp_ngc_statistic(
                z_hat, candidate["p_u"], c2["covariance_z"], data["scene"]
            )
            common.update(
                {
                    "c2_statistic": float(c2_stat["statistic"]),
                    "c2_rank": int(c2_stat["projected_geometry_rank"]),
                    "c2_valid_replicates": int(c2["n_valid"]),
                    "c2_failed_replicates": int(c2["n_failed"]),
                    "c2_runtime_s": float(time.perf_counter() - c2_start),
                }
            )
            details.update({"c2": c2, "c2_statistic": c2_stat})
        if bool(spec["run_heldout"]):
            c4_start = time.perf_counter()
            c4 = one_way_heldout_cp_ngc(
                data,
                config,
                covariance_regularization={
                    "shrinkage": 0.02,
                    "eigenvalue_floor_relative": 1.0e-6,
                },
            )
            c4_position_error = float(
                np.linalg.norm(
                    np.asarray(c4["candidate_a"]["p_u"])
                    - np.asarray(data["scene"]["p_u_true"])
                )
            )
            common.update(
                {
                    "c4_statistic": float(c4["statistic"]["statistic"]),
                    "c4_rank": int(c4["statistic"]["projected_geometry_rank"]),
                    "c4_candidate_position_error_m": c4_position_error,
                    "c4_position_covariance_valid": bool(
                        c4["covariance_p_a"]["valid_local_minimum"]
                    ),
                    "c4_heldout_certificate_valid": bool(
                        c4["heldout_certificate_valid"]
                    ),
                    "c4_runtime_s": float(time.perf_counter() - c4_start),
                }
            )
            details["c4"] = {
                "statistic": c4["statistic"],
                "candidate_a": {
                    "p_u": c4["candidate_a"]["p_u"],
                    "delta_t": c4["candidate_a"]["delta_t"],
                    "raw_objective_final": c4["candidate_a"]["raw_objective_final"],
                    "total_objective_final": c4["candidate_a"]["total_objective_final"],
                },
                "covariance_p_a": c4["covariance_p_a"],
                "covariance_b": {
                    key: value
                    for key, value in c4["covariance_b"].items()
                    if key not in {"fisher_effective_scaled"}
                },
                "fold_a_indices": c4["fold_a_indices"],
                "fold_b_indices": c4["fold_b_indices"],
                "stage1_a_hash": canonical_hash(
                    deterministic_stage1_output(c4["stage1_a"])
                ),
                "stage1_b_hash": canonical_hash(
                    deterministic_stage1_output(c4["stage1_b"])
                ),
                "heldout_certificate_valid": c4["heldout_certificate_valid"],
                "direction": c4["direction"],
            }
        row.update(common)
        return {"row": row, "details": details}
    except Exception as error:  # noqa: BLE001 - pilot failure is a result.
        row["error"] = f"{type(error).__name__}: {error}"
        return {"row": row, "details": {}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--diagnostic-mode", choices=("fast", "performance"), default="performance")
    parser.add_argument("--bootstrap-count", type=int, default=0)
    parser.add_argument("--run-heldout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(value.strip()) for value in args.seed_list.split(",") if value.strip()]
    spec = {
        "snr_db": -10.0,
        "diagnostic_mode": str(args.diagnostic_mode),
        "outlier_threshold_m": 0.1,
        "jones_mode": "jones_regularized",
        "old_max_iter": 20,
        "ccop_outer_max_iter": 20,
        "clock_fft_size": 4096,
        "clock_abs_tol": 1.0e-12,
        "clock_rel_tol": 1.0e-10,
        "clock_max_intervals": 20000,
        "use_old_incumbent": False,
        "old_vp_backend": "cpu",
        "gpu_device": 0,
        "bootstrap_count": int(args.bootstrap_count),
        "run_heldout": bool(args.run_heldout),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "covariance_pilot.csv"
    config_path = args.out_dir / "config.json"
    if not args.force_rerun and (csv_path.exists() or config_path.exists()):
        raise FileExistsError("outputs exist; use --force-rerun")
    rows = []
    detail_dir = args.out_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        result = _run(seed, spec)
        rows.append(result["row"])
        (detail_dir / f"seed_{seed}.json").write_text(
            json.dumps(_jsonable(result["details"]), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"completed seed={seed} failed={result['row']['failed']}", flush=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    command = shlex.join(
        [sys.executable, "-m", "src.experiments.run_cp_ngc_covariance_pilot", *(argv or sys.argv[1:])]
    )
    config_path.write_text(
        json.dumps(
            _jsonable(
                {
                    "split_role": "regression_or_development_only",
                    "seeds": seeds,
                    "spec": spec,
                    "arguments": vars(args),
                    "environment": validation_environment(
                        command, repo_root=pathlib.Path(__file__).resolve().parents[3]
                    ),
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
