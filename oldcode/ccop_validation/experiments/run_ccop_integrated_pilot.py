"""Reproducible pilot runner for the integrated experimental CCOP route."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import pathlib
import resource
import shlex
import sys
import time

import numpy as np

from ..ccop_certified_route import run_ccop_certified_route
from src.global_vp import build_jones_vp_dictionary, distance_to_box_boundary
from src.main_single_proposed import _make_data, run_stage1_only
from src.metrics import relative_nmse
from src.validation_artifacts import (
    array_sha256,
    canonical_hash,
    deterministic_stage1_output,
    validation_environment,
)
from .ccop_validation_presets import preset as validation_preset
from .run_ccop_full_validation import _candidate_hash
from .run_ccop_paired_mc import _build_config, _jsonable


FIELDS = [
    "seed",
    "failed",
    "error",
    "y_noisy_hash",
    "stage1_output_hash",
    "direct_position_error_m",
    "final_position_error_m",
    "direct_clock_error_ns",
    "final_clock_error_ns",
    "direct_channel_nmse",
    "final_channel_nmse",
    "direct_raw_objective",
    "final_raw_objective",
    "direct_total_objective",
    "final_total_objective",
    "direct_boundary_hit",
    "final_boundary_hit",
    "triggered",
    "selected_branch",
    "raw_non_degradation",
    "total_non_degradation",
    "outlier_rescued",
    "outlier_introduced",
    "cp_ngc_statistic",
    "cp_ngc_geometry_rank",
    "cp_ngc_covariance_fisher_rank",
    "cp_ngc_covariance_reliable",
    "cp_ngc_deployment_status",
    "direct_ccop_runtime_s",
    "cp_ngc_runtime_s",
    "conditional_recovery_runtime_s",
    "route_runtime_s",
    "data_runtime_s",
    "stage1_runtime_s",
    "deployment_runtime_s",
    "peak_rss_mb",
]


def _candidate_metrics(candidate: dict, data: dict, config: dict) -> dict:
    dictionary = build_jones_vp_dictionary(
        candidate["p_u"], candidate["delta_t"], data["scene"], config
    )
    y_hat = (dictionary @ np.asarray(candidate["x_hat"], dtype=complex)).reshape(
        data["Y_noisy"].shape
    )
    boundary = distance_to_box_boundary(
        candidate["p_u"],
        np.asarray(config["ue_bounds"], dtype=float),
        float(config["global_vp"].get("boundary_tol_m", 0.02)),
    )
    return {
        "position_error_m": float(
            np.linalg.norm(
                np.asarray(candidate["p_u"]) - np.asarray(data["scene"]["p_u_true"])
            )
        ),
        "clock_error_ns": float(
            abs(float(candidate["delta_t"]) - float(data["scene"]["delta_t_true"]))
            * 1.0e9
        ),
        "channel_nmse": float(relative_nmse(y_hat, data["Y_true"])),
        "raw_objective": float(candidate["raw_objective_final"]),
        "total_objective": float(candidate["total_objective_final"]),
        "boundary_hit": bool(boundary["boundary_hit"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--preset", choices=("fast", "balanced", "accuracy"), default="balanced")
    parser.add_argument("--diagnostic-mode", choices=("fast", "performance"), default="performance")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--split-role",
        choices=("development", "validation", "regression"),
        default="development",
    )
    parser.add_argument(
        "--reference-diagnostics",
        type=pathlib.Path,
        default=None,
        help="R1--R5 route_diagnostics.csv used to verify Y, Stage-I, and direct R3 hashes.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(value.strip()) for value in args.seed_list.split(",") if value.strip()]
    chosen = validation_preset(args.preset)
    spec = {
        "snr_db": -10.0,
        "diagnostic_mode": str(args.diagnostic_mode),
        "outlier_threshold_m": 0.1,
        "jones_mode": "jones_regularized",
        "old_max_iter": 80,
        "ccop_outer_max_iter": int(chosen["ccop_outer_max_iter"]),
        "clock_fft_size": int(chosen["clock_fft_size"]),
        "clock_abs_tol": 1.0e-12,
        "clock_rel_tol": float(chosen["clock_rel_tol"]),
        "clock_max_intervals": int(chosen["clock_max_intervals"]),
        "use_old_incumbent": False,
        "old_vp_backend": "cpu",
        "gpu_device": 0,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "integrated_trials.csv"
    config_path = args.out_dir / "config.json"
    if not args.force_rerun and (csv_path.exists() or config_path.exists()):
        raise FileExistsError("outputs exist; use --force-rerun")
    references: dict[int, dict] = {}
    if args.reference_diagnostics is not None:
        with args.reference_diagnostics.open(newline="", encoding="utf-8") as handle:
            for reference in csv.DictReader(handle):
                if reference.get("route_id") == "R3":
                    references[int(reference["seed"])] = reference
    if args.split_role == "validation" and args.reference_diagnostics is None:
        raise ValueError("validation replay requires --reference-diagnostics")
    missing_references = sorted(set(seeds) - set(references)) if references else []
    if missing_references:
        raise ValueError(f"missing R3 reference hashes for seeds {missing_references}")
    rows = []
    verified_reference_seeds: list[int] = []
    for seed in seeds:
        row = {field: "" for field in FIELDS}
        row.update({"seed": seed, "failed": True, "error": "not_run"})
        try:
            config = _build_config(spec, seed)
            data_start = time.perf_counter()
            data = _make_data(config)
            data_runtime = time.perf_counter() - data_start
            config["noise_variance"] = float(data["noise_variance"])
            stage1_start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                stage1 = run_stage1_only(data, config)["estimate"]
            stage1_runtime = time.perf_counter() - stage1_start
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_ccop_certified_route(
                    data, stage1, config, preset_name=args.preset
                )
            y_hash = array_sha256(data["Y_noisy"])
            stage1_hash = canonical_hash(deterministic_stage1_output(stage1))
            if references:
                reference = references[seed]
                checks = {
                    "Y_noisy": (y_hash, reference["y_noisy_hash"]),
                    "Stage-I": (stage1_hash, reference["stage1_output_hash"]),
                    "direct R3 candidate": (
                        _candidate_hash(result["direct"]),
                        reference["candidate_hash"],
                    ),
                }
                mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
                if mismatches:
                    raise RuntimeError(
                        "reference hash mismatch: " + ", ".join(mismatches)
                    )
                verified_reference_seeds.append(seed)
            direct = _candidate_metrics(result["direct"], data, config)
            final = _candidate_metrics(result["final"], data, config)
            direct_outlier = direct["position_error_m"] > 0.1
            final_outlier = final["position_error_m"] > 0.1
            cp = result["cp_ngc"]
            covariance = result["covariance"]
            row.update(
                {
                    "failed": False,
                    "error": "",
                    "y_noisy_hash": y_hash,
                    "stage1_output_hash": stage1_hash,
                    **{f"direct_{key}": value for key, value in direct.items()},
                    **{f"final_{key}": value for key, value in final.items()},
                    "triggered": bool(result["triggered"]),
                    "selected_branch": result["selected_branch"],
                    "raw_non_degradation": bool(result["raw_non_degradation"]),
                    "total_non_degradation": bool(result["total_non_degradation"]),
                    "outlier_rescued": bool(direct_outlier and not final_outlier),
                    "outlier_introduced": bool(not direct_outlier and final_outlier),
                    "cp_ngc_statistic": float(cp["statistic"]) if cp is not None else "",
                    "cp_ngc_geometry_rank": int(cp["projected_geometry_rank"]) if cp is not None else "",
                    "cp_ngc_covariance_fisher_rank": int(covariance["fisher_rank_before_floor"]) if covariance is not None else "",
                    "cp_ngc_covariance_reliable": bool(covariance["covariance_reliable_for_hard_certificate"]) if covariance is not None else False,
                    "cp_ngc_deployment_status": result["cp_ngc_deployment_status"],
                    "direct_ccop_runtime_s": result["runtime"]["direct_ccop_s"],
                    "cp_ngc_runtime_s": result["runtime"]["cp_ngc_c1_s"],
                    "conditional_recovery_runtime_s": result["runtime"]["conditional_recovery_s"],
                    "route_runtime_s": result["runtime"]["total_route_s"],
                    "data_runtime_s": data_runtime,
                    "stage1_runtime_s": stage1_runtime,
                    "deployment_runtime_s": data_runtime + stage1_runtime + result["runtime"]["total_route_s"],
                    "peak_rss_mb": float(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    ),
                }
            )
        except Exception as error:  # noqa: BLE001
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        print(f"completed seed={seed} failed={row['failed']}", flush=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    command = shlex.join([sys.executable, "-m", "src.experiments.run_ccop_integrated_pilot", *(argv or sys.argv[1:])])
    config_path.write_text(
        json.dumps(
            _jsonable(
                {
                    "split_role": str(args.split_role),
                    "selection_scope": (
                        "conditional subset replay" if references else "standalone pilot"
                    ),
                    "reference_diagnostics": (
                        str(args.reference_diagnostics)
                        if args.reference_diagnostics is not None
                        else None
                    ),
                    "reference_hashes_verified": len(verified_reference_seeds),
                    "seeds": seeds,
                    "preset": chosen,
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
