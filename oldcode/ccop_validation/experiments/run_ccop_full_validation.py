"""Gate-driven paired R1--R5 validation for experimental CCOP-JVP.

This runner is deliberately separate from the frozen paper entry points.  One
realization creates ``Y_noisy`` and Stage-I exactly once, then all five routes
receive deep copies of that same Stage-I output.  No CP-NGC threshold or rescue
selector is active here; the script isolates Stage-III optimization.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import pathlib
import resource
import shlex
import sys
import time
import traceback
from typing import Any

import numpy as np

from src.ccop_jvp import (
    refine_ccop_jvp,
    refine_four_dimensional_jvp_experimental,
)
from ..cp_ngc import cp_ngc_stage1_vector, cp_ngc_statistic
from ..cp_ngc_covariance import linearized_stage1_covariance, reliability_stratum
from src.global_vp import (
    _global_exact_spherical_vp_refinement_lbfgsb_reduced,
    build_jones_vp_dictionary,
)
from src.main_single_proposed import _make_data, run_stage1_only
from src.validation_artifacts import (
    array_sha256,
    canonical_hash,
    deterministic_stage1_output,
    validation_environment,
)
from src.experiments.resource_control import apply_thread_limits
from .ccop_validation_presets import preset as validation_preset
from .run_ccop_paired_mc import (
    TRIAL_FIELDS,
    _build_config,
    _jsonable,
    _route_row,
    _trial_seed,
    _write_csv,
)


ROUTES = {
    "R1": "R1_frozen_4d_jones_vp",
    "R2": "R2_clock_distance_scaled_4d",
    "R3": "R3_independent_ccop_jvp",
    "R4": "R4_raw_seconds_4d_matched_eval_budget",
    "R5": "R5_frozen_incumbent_ccop_polish",
}

DIAGNOSTIC_FIELDS = [
    "trial_id",
    "seed",
    "route_id",
    "route",
    "candidate_hash",
    "resolved_config_hash",
    "y_noisy_hash",
    "stage1_output_hash",
    "clock_coordinate",
    "clock_coordinate_scale_per_second",
    "evaluation_budget",
    "budget_source",
    "objective_evaluations",
    "position_orbit_evaluations",
    "clock_interval_evaluations",
    "selected_candidate",
    "peak_rss_mb_after_route",
]

CP_NGC_FIELDS = [
    "trial_id",
    "seed",
    "route_id",
    "route",
    "statistic",
    "dof",
    "delta_t_gls_s",
    "projected_geometry_rank",
    "minimum_certificate_eigenvalue",
    "full_3d_certificate",
    "uncertifiable_direction_count",
    "covariance_source",
    "covariance_fisher_rank",
    "covariance_full_rank",
    "covariance_hard_certificate_reliable",
    "raw_fit_residual_to_noise_ratio",
    "assignment_margin",
    "max_rank_one_ratio",
    "clock_dispersion_ns",
    "max_ris_residual",
    "boundary_hit",
    "stage1_valid",
    "reliability_stratum",
    "correct_basin_evaluation_label",
    "error",
]


def _candidate_hash(estimate: dict) -> str:
    return canonical_hash(
        {
            "p_u": np.asarray(estimate["p_u"], dtype=float),
            "delta_t": float(estimate["delta_t"]),
            "x_hat": np.asarray(estimate.get("x_hat", []), dtype=complex),
            "raw_objective": float(estimate.get("raw_objective_final", np.nan)),
            "total_objective": float(estimate.get("total_objective_final", np.nan)),
        }
    )


def _reconstruct(estimate: dict, data: dict, config: dict) -> np.ndarray:
    if estimate.get("Y_hat") is not None:
        return np.asarray(estimate["Y_hat"], dtype=complex)
    dictionary = build_jones_vp_dictionary(
        estimate["p_u"], estimate["delta_t"], data["scene"], config
    )
    return (dictionary @ np.asarray(estimate["x_hat"], dtype=complex)).reshape(
        data["Y_noisy"].shape
    )


def _failed_row(trial_id: int, seed: int, route: str, spec: dict, error: BaseException) -> dict:
    row = {field: "" for field in TRIAL_FIELDS}
    row.update(
        {
            "trial_id": int(trial_id),
            "seed": int(seed),
            "snr_db": float(spec["snr_db"]),
            "route": route,
            "failed": True,
            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=6)}",
            "diagnostic_mode": str(spec["diagnostic_mode"]),
        }
    )
    return row


def _run_route(
    route_id: str,
    *,
    data: dict,
    stage1_estimate: dict,
    config: dict,
    spec: dict,
    r1_estimate: dict | None,
    r1_evaluations: int,
    r3_position_evaluations: int,
) -> tuple[dict, float, float, int | None, str]:
    start = time.perf_counter()
    evaluation_budget: int | None = None
    budget_source = "route_default"
    if route_id == "R1":
        estimate = _global_exact_spherical_vp_refinement_lbfgsb_reduced(
            data["Y_noisy"], copy.deepcopy(stage1_estimate), data["scene"], config
        )
    elif route_id == "R2":
        estimate = refine_four_dimensional_jvp_experimental(
            data["Y_noisy"],
            copy.deepcopy(stage1_estimate),
            data["scene"],
            config,
            clock_coordinate="distance_m",
            max_iter=int(spec["old_max_iter"]),
        )
    elif route_id == "R3":
        estimate = refine_ccop_jvp(
            data["Y_noisy"],
            copy.deepcopy(stage1_estimate),
            data["scene"],
            config,
            incumbent=None,
        )
    elif route_id == "R4":
        evaluation_budget = max(
            5, int(r1_evaluations) + int(r3_position_evaluations)
        )
        budget_source = "R1 objective evaluations + R3 position-orbit evaluations"
        estimate = refine_four_dimensional_jvp_experimental(
            data["Y_noisy"],
            copy.deepcopy(stage1_estimate),
            data["scene"],
            config,
            clock_coordinate="seconds",
            max_iter=int(spec["old_max_iter"]) + int(r3_position_evaluations),
            max_evaluations=evaluation_budget,
        )
    elif route_id == "R5":
        if r1_estimate is None:
            raise RuntimeError("R5 requires a finite R1 incumbent")
        estimate = refine_ccop_jvp(
            data["Y_noisy"],
            copy.deepcopy(stage1_estimate),
            data["scene"],
            config,
            incumbent=r1_estimate,
        )
    else:  # pragma: no cover - protected by the constant route list.
        raise ValueError(f"unknown route {route_id}")
    refinement_runtime = time.perf_counter() - start
    reconstruction_start = time.perf_counter()
    y_hat = _reconstruct(estimate, data, config)
    reconstruction_runtime = time.perf_counter() - reconstruction_start
    estimate["_validation_y_hat"] = y_hat
    return estimate, refinement_runtime, reconstruction_runtime, evaluation_budget, budget_source


def _run_trial(trial_id: int, seed: int, spec: dict, environment: dict) -> dict:
    apply_thread_limits(int(spec["blas_threads"]))
    trial_start = time.perf_counter()
    config = _build_config(spec, seed)
    data_start = time.perf_counter()
    data = _make_data(config)
    data_runtime = time.perf_counter() - data_start
    config["noise_variance"] = float(data["noise_variance"])
    stage1_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        stage1_record = run_stage1_only(data, config)
    stage1_runtime = time.perf_counter() - stage1_start
    stage1_estimate = stage1_record["estimate"]
    resolved_config_hash = canonical_hash(config)
    y_hash = array_sha256(data["Y_noisy"])
    stage1_hash = canonical_hash(deterministic_stage1_output(stage1_estimate))
    shared_short_hash = y_hash[:20]

    covariance_error = ""
    covariance_c1 = None
    if str(spec["cp_ngc_covariance"]) == "c1":
        try:
            covariance_c1 = linearized_stage1_covariance(
                data["Y_noisy"],
                stage1_estimate,
                data["scene"],
                data["noise_variance"],
            )
        except Exception as error:  # noqa: BLE001 - diagnostic failure must not fail a route.
            covariance_error = f"{type(error).__name__}: {error}"
    else:
        covariance_error = "disabled_for_stage3_isolation"

    rows: list[dict] = []
    diagnostics: list[dict] = []
    cp_diagnostics: list[dict] = []
    estimates: dict[str, dict] = {}
    runtimes: dict[str, float] = {}
    r1_evaluations = 0
    r3_position_evaluations = 0
    for route_id, route_name in ROUTES.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                estimate, refinement_runtime, reconstruction_runtime, budget, budget_source = _run_route(
                    route_id,
                    data=data,
                    stage1_estimate=stage1_estimate,
                    config=config,
                    spec=spec,
                    r1_estimate=estimates.get("R1"),
                    r1_evaluations=r1_evaluations,
                    r3_position_evaluations=r3_position_evaluations,
                )
            estimates[route_id] = estimate
            runtimes[route_id] = refinement_runtime + reconstruction_runtime
            optimizer = dict(estimate.get("optimizer", {}))
            if route_id == "R1":
                r1_evaluations = int(optimizer.get("n_eval", 0))
            elif route_id == "R3":
                r3_position_evaluations = int(
                    estimate.get("ccop_position_evaluations", optimizer.get("n_eval", 0))
                )
            row = _route_row(
                trial_id=trial_id,
                seed=seed,
                route=route_name,
                estimate=estimate,
                y_hat=estimate.pop("_validation_y_hat"),
                data=data,
                config=config,
                spec=spec,
                data_hash=shared_short_hash,
                data_runtime_s=data_runtime,
                stage1_runtime_s=stage1_runtime,
                refinement_runtime_s=refinement_runtime,
                reconstruction_runtime_s=reconstruction_runtime,
                old_runtime_s=runtimes.get("R1", 0.0),
                paired_wall_runtime_s=0.0,
            )
            row["old_incumbent_used"] = bool(route_id == "R5")
            if route_id == "R5":
                row["incumbent_generation_runtime_s"] = float(runtimes["R1"])
                row["deployment_runtime_s"] = float(
                    data_runtime + stage1_runtime + runtimes["R1"] + runtimes["R5"]
                )
            rank_one = np.asarray(
                stage1_estimate.get("stage1_rank1_ratios", []), dtype=float
            )
            clock_replicas = np.asarray(
                stage1_estimate.get("selected_clock_offsets", []), dtype=float
            )
            ris_residuals = np.asarray(
                stage1_estimate.get("stage1_ris_residuals", []), dtype=float
            )
            valid_arrays = [
                np.asarray(stage1_estimate.get(key, [True]), dtype=bool)
                for key in ("stage1_delay_valid", "stage1_local_geometry_valid")
            ]
            reliability = {
                "assignment_margin": float(stage1_estimate.get("assignment_margin", np.nan)),
                "max_rank_one_ratio": float(np.max(rank_one)) if rank_one.size else np.nan,
                "clock_dispersion_ns": float(np.std(clock_replicas) * 1.0e9) if clock_replicas.size else np.nan,
                "max_ris_residual": float(np.nanmax(ris_residuals)) if ris_residuals.size else np.nan,
                "boundary_hit": bool(row["boundary_hit"]),
                "stage1_valid": bool(all(np.all(values) for values in valid_arrays)),
            }
            stratum = reliability_stratum(
                reliability,
                {
                    "assignment_margin_min": float(config.get("reliability_assignment_low", 0.3)),
                    "rank_one_ratio_max": float(config.get("mhr_rank1_ratio_threshold", 0.9)),
                    "clock_dispersion_ns_max": float(config.get("reliability_clock_bad_ns", 0.5)),
                    "ris_residual_max": float(config.get("reliability_ris_bad", 0.7)),
                },
            )
            cp_record = {
                "trial_id": trial_id,
                "seed": seed,
                "route_id": route_id,
                "route": route_name,
                **reliability,
                "reliability_stratum": stratum,
                "correct_basin_evaluation_label": bool(not row["outlier_flag"]),
                "error": covariance_error,
            }
            if covariance_c1 is not None:
                try:
                    cp_result = cp_ngc_statistic(
                        cp_ngc_stage1_vector(stage1_estimate, data["scene"]),
                        estimate["p_u"],
                        covariance_c1["covariance_z"],
                        data["scene"],
                    )
                    cert_eigenvalues = np.asarray(
                        cp_result["cert_information_eigenvalues"], dtype=float
                    )
                    cp_record.update(
                        {
                            "statistic": float(cp_result["statistic"]),
                            "dof": int(cp_result["dof"]),
                            "delta_t_gls_s": float(cp_result["delta_t_gls"]),
                            "projected_geometry_rank": int(cp_result["projected_geometry_rank"]),
                            "minimum_certificate_eigenvalue": float(np.min(cert_eigenvalues)),
                            "full_3d_certificate": bool(cp_result["full_3d_certificate"]),
                            "uncertifiable_direction_count": int(cp_result["uncertifiable_position_directions"].shape[1]),
                            "covariance_source": covariance_c1["source"],
                            "covariance_fisher_rank": int(covariance_c1["fisher_rank_before_floor"]),
                            "covariance_full_rank": bool(covariance_c1["full_fisher_rank"]),
                            "covariance_hard_certificate_reliable": bool(covariance_c1["covariance_reliable_for_hard_certificate"]),
                            "raw_fit_residual_to_noise_ratio": float(covariance_c1["raw_fit_residual_to_noise_ratio"]),
                            "error": "",
                        }
                    )
                except Exception as error:  # noqa: BLE001
                    cp_record["error"] = f"{type(error).__name__}: {error}"
            cp_diagnostics.append(cp_record)
            diagnostics.append(
                {
                    "trial_id": trial_id,
                    "seed": seed,
                    "route_id": route_id,
                    "route": route_name,
                    "candidate_hash": _candidate_hash(estimate),
                    "resolved_config_hash": resolved_config_hash,
                    "y_noisy_hash": y_hash,
                    "stage1_output_hash": stage1_hash,
                    "clock_coordinate": estimate.get("clock_coordinate", "profiled" if route_id in {"R3", "R5"} else "seconds"),
                    "clock_coordinate_scale_per_second": estimate.get("clock_coordinate_scale_per_second", ""),
                    "evaluation_budget": "" if budget is None else budget,
                    "budget_source": budget_source,
                    "objective_evaluations": optimizer.get("n_eval", ""),
                    "position_orbit_evaluations": estimate.get("ccop_position_evaluations", estimate.get("vp_position_evaluations", "")),
                    "clock_interval_evaluations": estimate.get("ccop_clock_interval_evaluations", ""),
                    "selected_candidate": estimate.get("selected_candidate", ""),
                    "peak_rss_mb_after_route": float(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    ),
                }
            )
            rows.append(row)
        except Exception as error:  # noqa: BLE001 - a route failure is experimental data.
            rows.append(_failed_row(trial_id, seed, route_name, spec, error))

    paired_wall = time.perf_counter() - trial_start
    for row in rows:
        row["paired_trial_wall_runtime_s"] = float(paired_wall)
    return {
        "rows": rows,
        "diagnostics": diagnostics,
        "cp_ngc": cp_diagnostics,
        "trial": {
            "trial_id": trial_id,
            "seed": seed,
            "resolved_config_hash": resolved_config_hash,
            "y_noisy_hash": y_hash,
            "stage1_output_hash": stage1_hash,
            "environment": environment,
        },
    }


def _safe_values(rows: list[dict], route: str, field: str) -> np.ndarray:
    values = []
    for row in rows:
        if row["route"] != route or bool(row["failed"]):
            continue
        try:
            value = float(row[field])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _summaries(rows: list[dict]) -> list[dict]:
    output = []
    for route_id, route in ROUTES.items():
        selected = [row for row in rows if row["route"] == route]
        successful = [row for row in selected if not bool(row["failed"])]
        record: dict[str, Any] = {
            "route_id": route_id,
            "route": route,
            "n": len(selected),
            "n_success": len(successful),
            "failure_rate": float(1.0 - len(successful) / len(selected)) if selected else np.nan,
            "outlier_rate": float(np.mean([bool(row["outlier_flag"]) for row in successful])) if successful else np.nan,
        }
        for field in ("position_error_m", "clock_error_ns", "channel_y_nmse", "raw_objective_final", "regularized_objective_final", "route_runtime_s", "deployment_runtime_s"):
            values = _safe_values(rows, route, field)
            for name, function in (
                ("median", np.median),
                ("p90", lambda x: np.percentile(x, 90)),
                ("p95", lambda x: np.percentile(x, 95)),
                ("rmse", lambda x: np.sqrt(np.mean(x**2))),
            ):
                record[f"{field}_{name}"] = float(function(values)) if values.size else np.nan
        output.append(record)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared-Stage-I CCOP R1--R5 validation runner")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument(
        "--preset", choices=("fast", "balanced", "accuracy", "custom"), default="fast"
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--seed-list",
        default="",
        help="Optional comma-separated explicit trial seeds; overrides --n-trials/--seed.",
    )
    parser.add_argument(
        "--split-role",
        choices=("development", "validation", "heldout", "regression"),
        default="development",
    )
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--diagnostic-mode", choices=("fast", "performance"), default="fast")
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    parser.add_argument("--jones-mode", choices=("jones_regularized", "jones_free"), default="jones_regularized")
    parser.add_argument("--old-max-iter", type=int, default=8)
    parser.add_argument("--ccop-outer-max-iter", type=int, default=5)
    parser.add_argument("--clock-fft-size", type=int, default=1024)
    parser.add_argument("--clock-abs-tol", type=float, default=1.0e-12)
    parser.add_argument("--clock-rel-tol", type=float, default=1.0e-10)
    parser.add_argument("--clock-max-intervals", type=int, default=5000)
    parser.add_argument("--old-vp-backend", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--cp-ngc-covariance", choices=("none", "c1"), default="c1"
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results/ccop_full_validation/stage3_smoke"))
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_arguments = list(argv or sys.argv[1:])
    if args.preset != "custom":
        chosen = validation_preset(str(args.preset))
        mapping = {
            "clock_fft_size": "--clock-fft-size",
            "clock_rel_tol": "--clock-rel-tol",
            "clock_max_intervals": "--clock-max-intervals",
            "ccop_outer_max_iter": "--ccop-outer-max-iter",
        }
        for attribute, flag in mapping.items():
            if flag not in raw_arguments:
                setattr(args, attribute, chosen[attribute])
    if args.n_trials <= 0 or args.blas_threads <= 0:
        raise ValueError("--n-trials and --blas-threads must be positive")
    out_dir = args.out_dir
    protected = [out_dir / name for name in ("route_trials.csv", "route_diagnostics.csv", "cp_ngc_diagnostics.csv", "route_summary.csv", "config.json")]
    if not args.force_rerun and any(path.exists() for path in protected):
        raise FileExistsError(f"outputs already exist under {out_dir}; use --force-rerun")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "oldcode.ccop_validation.experiments.run_ccop_full_validation",
            *(argv or sys.argv[1:]),
        ]
    )
    environment = validation_environment(command, repo_root=pathlib.Path(__file__).resolve().parents[3])
    if str(args.seed_list).strip():
        seeds = [
            int(value.strip())
            for value in str(args.seed_list).split(",")
            if value.strip()
        ]
        if not seeds:
            raise ValueError("--seed-list did not contain any seeds")
    else:
        seed_sequence = np.random.SeedSequence(int(args.seed))
        seeds = [_trial_seed(child) for child in seed_sequence.spawn(int(args.n_trials))]
    if len(set(seeds)) != len(seeds):
        raise ValueError("trial seeds must be unique within a split")
    spec = {
        "snr_db": float(args.snr_db),
        "diagnostic_mode": str(args.diagnostic_mode),
        "outlier_threshold_m": float(args.outlier_threshold_m),
        "jones_mode": str(args.jones_mode),
        "old_max_iter": int(args.old_max_iter),
        "ccop_outer_max_iter": int(args.ccop_outer_max_iter),
        "clock_fft_size": int(args.clock_fft_size),
        "clock_abs_tol": float(args.clock_abs_tol),
        "clock_rel_tol": float(args.clock_rel_tol),
        "clock_max_intervals": int(args.clock_max_intervals),
        "use_old_incumbent": False,
        "old_vp_backend": str(args.old_vp_backend),
        "gpu_device": int(args.gpu_device),
        "blas_threads": int(args.blas_threads),
        "cp_ngc_covariance": str(args.cp_ngc_covariance),
    }
    rows: list[dict] = []
    diagnostics: list[dict] = []
    cp_diagnostics: list[dict] = []
    trial_records: list[dict] = []
    for trial_id, seed in enumerate(seeds):
        result = _run_trial(trial_id, seed, spec, environment)
        rows.extend(result["rows"])
        diagnostics.extend(result["diagnostics"])
        cp_diagnostics.extend(result["cp_ngc"])
        trial_records.append(result["trial"])
        print(f"completed trial {trial_id + 1}/{len(seeds)} seed={seed}", flush=True)
    summaries = _summaries(rows)
    _write_csv(out_dir / "route_trials.csv", rows, TRIAL_FIELDS)
    _write_csv(out_dir / "route_diagnostics.csv", diagnostics, DIAGNOSTIC_FIELDS)
    _write_csv(out_dir / "cp_ngc_diagnostics.csv", cp_diagnostics, CP_NGC_FIELDS)
    summary_fields = list(summaries[0]) if summaries else ["route_id", "route"]
    _write_csv(out_dir / "route_summary.csv", summaries, summary_fields)
    config_record = {
        "experiment": "CCOP full validation Stage-III isolation R1--R5",
        "preset": str(args.preset),
        "preset_record": (
            validation_preset(str(args.preset)) if args.preset != "custom" else {}
        ),
        "split_role": str(args.split_role),
        "heldout_policy": (
            "run once after preset and thresholds are frozen"
            if args.split_role == "heldout"
            else "not a held-out test run"
        ),
        "shared_trial_contract": "one Y_noisy and one Stage-I output per seed",
        "r4_budget_contract": "raw-seconds 4-D exact objective capped at R1 objective evaluations plus R3 position-orbit evaluations",
        "arguments": _jsonable(vars(args)),
        "spec": spec,
        "seeds": seeds,
        "environment": environment,
        "trials": trial_records,
    }
    (out_dir / "config.json").write_text(json.dumps(_jsonable(config_record), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(summaries), indent=2), flush=True)


if __name__ == "__main__":
    main()
