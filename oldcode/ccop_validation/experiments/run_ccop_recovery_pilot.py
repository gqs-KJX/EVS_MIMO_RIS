"""One-factor S0--S4 conditional recovery ablation on shared realizations."""

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
from ..ccop_recovery import run_recovery_ablation
from src.global_vp import build_jones_vp_dictionary, distance_to_box_boundary
from src.main_single_proposed import _make_data, run_stage1_only
from src.metrics import relative_nmse
from src.validation_artifacts import (
    array_sha256,
    canonical_hash,
    deterministic_stage1_output,
    validation_environment,
)
from .run_ccop_paired_mc import _build_config, _jsonable


FIELDS = [
    "seed",
    "variant",
    "failed",
    "error",
    "y_noisy_hash",
    "stage1_output_hash",
    "direct_position_error_m",
    "direct_clock_error_ns",
    "direct_channel_nmse",
    "direct_raw_objective",
    "direct_total_objective",
    "direct_boundary_hit",
    "rescue_position_error_m",
    "rescue_clock_error_ns",
    "rescue_channel_nmse",
    "rescue_raw_objective",
    "rescue_total_objective",
    "rescue_boundary_hit",
    "objective_non_degradation",
    "raw_non_degradation",
    "position_improved_evaluation_only",
    "outlier_rescued_evaluation_only",
    "outlier_introduced_evaluation_only",
    "selector_accepted",
    "selector_reason",
    "selection_path",
    "boundary_override",
    "relative_raw_improvement",
    "alternative_assignment_support",
    "independent_lg_support",
    "num_assignment_hypotheses",
    "num_short_ccop",
    "num_full_ccop",
    "lg_refinement_used",
    "rdc_is_final_estimator",
    "permutation_annealing_used",
    "runtime_s",
]


def _metrics(candidate: dict, data: dict, config: dict) -> dict:
    position_error = float(
        np.linalg.norm(
            np.asarray(candidate["p_u"]) - np.asarray(data["scene"]["p_u_true"])
        )
    )
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
        "position_error_m": position_error,
        "clock_error_ns": float(
            abs(float(candidate["delta_t"]) - float(data["scene"]["delta_t_true"]))
            * 1.0e9
        ),
        "channel_nmse": float(relative_nmse(y_hat, data["Y_true"])),
        "raw_objective": float(candidate["raw_objective_final"]),
        "total_objective": float(candidate["total_objective_final"]),
        "boundary_hit": bool(boundary["boundary_hit"]),
    }


def _run_seed(seed: int, spec: dict, variants: list[str]) -> list[dict]:
    config = _build_config(spec, seed)
    data = _make_data(config)
    config["noise_variance"] = float(data["noise_variance"])
    with contextlib.redirect_stdout(io.StringIO()):
        stage1 = run_stage1_only(data, config)["estimate"]
        direct = refine_ccop_jvp(
            data["Y_noisy"], stage1, data["scene"], config, incumbent=None
        )
    direct_metrics = _metrics(direct, data, config)
    y_hash = array_sha256(data["Y_noisy"])
    stage1_hash = canonical_hash(deterministic_stage1_output(stage1))
    rows = []
    for variant in variants:
        row = {field: "" for field in FIELDS}
        row.update(
            {
                "seed": seed,
                "variant": variant,
                "failed": True,
                "error": "not_run",
                "y_noisy_hash": y_hash,
                "stage1_output_hash": stage1_hash,
                **{f"direct_{key}": value for key, value in direct_metrics.items()},
            }
        )
        try:
            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                recovery = run_recovery_ablation(
                    data["Y_noisy"],
                    stage1,
                    direct,
                    data["scene"],
                    config,
                    variant=variant,
                    top_l=int(spec["top_l"]),
                    short_outer_max_iter=int(spec["short_outer_max_iter"]),
                    full_hypotheses=int(spec["full_hypotheses"]),
                    noise_variance=None,
                    z_noisy=data["Z_noisy"],
                )
            candidate = direct if recovery["best_rescue"] is None else recovery["best_rescue"]
            rescue_metrics = _metrics(candidate, data, config)
            direct_outlier = direct_metrics["position_error_m"] > float(spec["outlier_threshold_m"])
            rescue_outlier = rescue_metrics["position_error_m"] > float(spec["outlier_threshold_m"])
            lg_used = any(
                record.get("lg_diagnostics", {}).get("ris_only_refinement_used", False)
                for record in recovery.get("candidate_records", [])
            )
            row.update(
                {
                    "failed": False,
                    "error": "",
                    **{f"rescue_{key}": value for key, value in rescue_metrics.items()},
                    "objective_non_degradation": bool(
                        rescue_metrics["total_objective"]
                        <= direct_metrics["total_objective"] + 1.0e-12
                    ),
                    "raw_non_degradation": bool(
                        rescue_metrics["raw_objective"]
                        <= direct_metrics["raw_objective"] + 1.0e-12
                    ),
                    "position_improved_evaluation_only": bool(
                        rescue_metrics["position_error_m"] < direct_metrics["position_error_m"]
                    ),
                    "outlier_rescued_evaluation_only": bool(direct_outlier and not rescue_outlier),
                    "outlier_introduced_evaluation_only": bool(not direct_outlier and rescue_outlier),
                    "selector_accepted": bool(recovery["accepted"]),
                    "selector_reason": str(recovery["acceptance_reason"]),
                    "selection_path": str(recovery.get("selection_path", "direct_incumbent")),
                    "boundary_override": bool(recovery.get("boundary_override", False)),
                    "relative_raw_improvement": float(recovery.get("relative_raw_improvement", 0.0)),
                    "alternative_assignment_support": bool(recovery.get("alternative_assignment_support", False)),
                    "independent_lg_support": bool(recovery.get("independent_lg_support", False)),
                    "num_assignment_hypotheses": int(recovery.get("num_assignment_hypotheses", 0)),
                    "num_short_ccop": int(recovery.get("num_short_ccop", 0)),
                    "num_full_ccop": int(recovery.get("num_full_ccop", 0)),
                    "lg_refinement_used": bool(lg_used),
                    "rdc_is_final_estimator": bool(recovery.get("rdc_is_final_estimator", False)),
                    "permutation_annealing_used": bool(recovery.get("permutation_annealing_used", False)),
                    "runtime_s": float(time.perf_counter() - start),
                }
            )
        except Exception as error:  # noqa: BLE001 - failed ablation is a result.
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--variants", default="S0,S1,S2,S3,S4")
    parser.add_argument("--diagnostic-mode", choices=("fast", "performance"), default="performance")
    parser.add_argument("--top-l", type=int, default=3)
    parser.add_argument("--short-outer-max-iter", type=int, default=3)
    parser.add_argument("--full-hypotheses", type=int, default=1)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(value.strip()) for value in args.seed_list.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
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
        "top_l": int(args.top_l),
        "short_outer_max_iter": int(args.short_outer_max_iter),
        "full_hypotheses": int(args.full_hypotheses),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "recovery_ablation.csv"
    config_path = args.out_dir / "config.json"
    if not args.force_rerun and (csv_path.exists() or config_path.exists()):
        raise FileExistsError("outputs exist; use --force-rerun")
    rows = []
    for seed in seeds:
        seed_rows = _run_seed(seed, spec, variants)
        rows.extend(seed_rows)
        print(f"completed seed={seed} failures={sum(bool(row['failed']) for row in seed_rows)}", flush=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    command = shlex.join([sys.executable, "-m", "src.experiments.run_ccop_recovery_pilot", *(argv or sys.argv[1:])])
    config_path.write_text(
        json.dumps(
            _jsonable(
                {
                    "split_role": "development_or_regression_only",
                    "seeds": seeds,
                    "variants": variants,
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
