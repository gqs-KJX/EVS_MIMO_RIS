"""Paired pilot for frozen 4-D/CCOP and physics-subspace Stage-I routes.

The raw observation is generated once per seed.  The independent CCOP routes
use no incumbent, CP-NGC selector, or rescue.  The explicit frozen 4-D route
is retained as the original-route paired control.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import pathlib
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..ccop_jvp import refine_ccop_jvp, refine_four_dimensional_jvp_experimental
from ..ccop_stage1_initializer import (
    apply_ccop_stage1_preset,
    initialize_ccop_stage1,
    initialize_ccop_stage1_joint_geometry,
    refresh_ccop_stage1_jones_anchor,
    refine_ccop_stage1_joint_geometry,
)
from ..global_vp import _initial_xi_from_stage1, build_jones_vp_dictionary
from ..main_single_proposed import _make_data, run_stage1_only
from ..metrics import relative_nmse
from ..projections_delay import tau_from_pole
from ..validation_artifacts import (
    array_sha256,
    canonical_hash,
    deterministic_stage1_output,
    validation_environment,
)
from .resource_control import apply_thread_limits
from .final_mksc_seed_splits import seed_splits
from .ccop_experiment_config import build_ccop_experiment_config


ROUTES = (
    "frozen_stage1_4d_jones_vp",
    "frozen_stage1_ccop",
    "evs_subspace_stage1_ccop",
    "evs_subspace_joint_geometry_stage1_ccop",
)
FIELDS = (
    "seed",
    "route",
    "y_noisy_hash",
    "resolved_config_hash",
    "stage1_output_hash",
    "candidate_hash",
    "stage1_position_error_m",
    "stage1_delay_rmse_ns",
    "stage1_runtime_s",
    "stage1_projection_runtime_s",
    "stage1_evs_rank",
    "stage1_evs_retained_energy_fraction",
    "stage1_max_rank1_ratio",
    "stage1_assignment_margin",
    "stage1_jones_anchor_refresh",
    "stage1_jones_anchor_refresh_runtime_s",
    "stage1_jones_anchor_refresh_clock_certified",
    "position_error_m",
    "clock_error_ns",
    "channel_y_nmse",
    "outlier",
    "raw_objective_final",
    "total_objective_final",
    "ccop_runtime_s",
    "clock_certified",
    "selected_candidate",
)


def _spec(args: argparse.Namespace) -> dict:
    return {
        "snr_db": float(args.snr_db),
        "diagnostic_mode": str(args.diagnostic_mode),
        "outlier_threshold_m": float(args.outlier_threshold_m),
        "jones_mode": "jones_regularized",
        "old_max_iter": 80,
        "ccop_outer_max_iter": int(args.ccop_outer_max_iter),
        "clock_fft_size": int(args.clock_fft_size),
        "clock_abs_tol": 1.0e-12,
        "clock_rel_tol": 1.0e-10,
        "clock_max_intervals": 20000,
        "use_old_incumbent": False,
        "old_vp_backend": "cpu",
        "gpu_device": 0,
    }


def _stage1_delay_rmse_ns(estimate: dict, data: dict) -> float:
    scene = data["scene"]
    tau_hat = np.array(
        [tau_from_pole(pole, scene["delta_f"]) for pole in estimate["poles"]],
        dtype=float,
    )
    tau_true = np.asarray(data["true_components"]["taus"], dtype=float)
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)) * 1.0e9)


def _run_route(
    route: str,
    *,
    data: dict,
    config: dict,
    outlier_threshold_m: float,
    stage1_override: dict | None = None,
    stage1_runtime_override: float | None = None,
) -> dict:
    if stage1_override is None:
        stage1_start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            if route in ("frozen_stage1_4d_jones_vp", "frozen_stage1_ccop"):
                stage1 = run_stage1_only(data, config)["estimate"]
            elif route == "evs_subspace_stage1_ccop":
                stage1 = initialize_ccop_stage1(data["Z_noisy"], data["scene"], config)
            elif route == "evs_subspace_joint_geometry_stage1_ccop":
                stage1 = initialize_ccop_stage1_joint_geometry(
                    data["Z_noisy"], data["scene"], config, y_raw=data["Y_noisy"]
                )
            else:  # pragma: no cover
                raise ValueError(f"unknown route {route}")
        stage1_runtime = time.perf_counter() - stage1_start
    else:
        stage1 = copy.deepcopy(stage1_override)
        stage1_runtime = float(stage1_runtime_override)
    xi0 = _initial_xi_from_stage1(stage1, data["scene"], config)

    ccop_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if route == "frozen_stage1_4d_jones_vp":
            final = refine_four_dimensional_jvp_experimental(
                data["Y_noisy"],
                copy.deepcopy(stage1),
                data["scene"],
                config,
                clock_coordinate="seconds",
                max_iter=int(config["global_vp"].get("max_iter", 80)),
            )
        else:
            final = refine_ccop_jvp(
                data["Y_noisy"],
                copy.deepcopy(stage1),
                data["scene"],
                config,
                incumbent=None,
            )
    ccop_runtime = time.perf_counter() - ccop_start
    dictionary = build_jones_vp_dictionary(
        final["p_u"], final["delta_t"], data["scene"], config
    )
    y_hat = (dictionary @ np.asarray(final["x_hat"], dtype=complex)).reshape(
        data["Y_noisy"].shape
    )
    p_true = np.asarray(data["scene"]["p_u_true"], dtype=float)
    position_error = float(np.linalg.norm(np.asarray(final["p_u"]) - p_true))
    return {
        "seed": int(config["seed"]),
        "route": route,
        "y_noisy_hash": array_sha256(data["Y_noisy"]),
        "resolved_config_hash": canonical_hash(config),
        "stage1_output_hash": canonical_hash(deterministic_stage1_output(stage1)),
        "candidate_hash": canonical_hash(
            {
                "p_u": np.asarray(final["p_u"], dtype=float),
                "delta_t": float(final["delta_t"]),
                "x_hat": np.asarray(final["x_hat"], dtype=complex),
                "raw_objective_final": float(final["raw_objective_final"]),
            }
        ),
        "stage1_position_error_m": float(np.linalg.norm(xi0[:3] - p_true)),
        "stage1_delay_rmse_ns": _stage1_delay_rmse_ns(stage1, data),
        "stage1_runtime_s": float(stage1_runtime),
        "stage1_projection_runtime_s": float(
            stage1.get("stage1_time_evs_sufficient_projection", 0.0)
        ),
        "stage1_evs_rank": int(stage1.get("stage1_evs_union_rank", data["scene"]["I"])),
        "stage1_evs_retained_energy_fraction": float(
            stage1.get("stage1_evs_retained_energy_fraction", 1.0)
        ),
        "stage1_max_rank1_ratio": float(stage1.get("stage1_max_rank1_ratio", np.nan)),
        "stage1_assignment_margin": float(stage1.get("stage1_assignment_margin", np.nan)),
        "stage1_jones_anchor_refresh": str(
            stage1.get("stage1_jones_anchor_refresh", "disabled")
        ),
        "stage1_jones_anchor_refresh_runtime_s": float(
            stage1.get("stage1_jones_anchor_refresh_runtime_s", 0.0)
        ),
        "stage1_jones_anchor_refresh_clock_certified": bool(
            stage1.get("stage1_jones_anchor_refresh_clock_certified", False)
        ),
        "position_error_m": position_error,
        "clock_error_ns": float(
            abs(float(final["delta_t"]) - float(data["scene"]["delta_t_true"])) * 1.0e9
        ),
        "channel_y_nmse": float(relative_nmse(y_hat, data["Y_true"])),
        "outlier": bool(position_error > outlier_threshold_m),
        "raw_objective_final": float(final["raw_objective_final"]),
        "total_objective_final": float(final["total_objective_final"]),
        "ccop_runtime_s": float(ccop_runtime),
        "clock_certified": bool(final.get("clock_certified", False)),
        "selected_candidate": str(final.get("selected_candidate", "")),
    }


def _summary(rows: list[dict]) -> dict:
    result = {}
    for route in ROUTES:
        selected = [row for row in rows if row["route"] == route]
        if not selected:
            continue
        result[route] = {
            "n": len(selected),
            "outliers": int(sum(bool(row["outlier"]) for row in selected)),
            "position_rmse_m": float(
                np.sqrt(np.mean([row["position_error_m"] ** 2 for row in selected]))
            ),
            "position_median_m": float(np.median([row["position_error_m"] for row in selected])),
            "delay_rmse_median_ns": float(
                np.median([row["stage1_delay_rmse_ns"] for row in selected])
            ),
            "stage1_runtime_median_s": float(
                np.median([row["stage1_runtime_s"] for row in selected])
            ),
            "channel_nmse_median": float(np.median([row["channel_y_nmse"] for row in selected])),
        }
    frozen = {
        int(row["seed"]): row
        for row in rows
        if row["route"] == "frozen_stage1_4d_jones_vp"
    }
    proposed = {int(row["seed"]): row for row in rows if row["route"] == ROUTES[-1]}
    common = sorted(set(frozen) & set(proposed))
    result["paired"] = {
        "n": len(common),
        "rescued_outliers": int(
            sum(bool(frozen[s]["outlier"]) and not bool(proposed[s]["outlier"]) for s in common)
        ),
        "introduced_outliers": int(
            sum(not bool(frozen[s]["outlier"]) and bool(proposed[s]["outlier"]) for s in common)
        ),
        "position_wins": int(
            sum(proposed[s]["position_error_m"] < frozen[s]["position_error_m"] for s in common)
        ),
    }
    return result


def _write_checkpoint(
    output_csv: pathlib.Path,
    rows: list[dict],
    selected_seeds: list[int],
    selected_routes: list[str],
) -> None:
    """Atomically persist all completed paired rows in deterministic order."""
    seed_order = {int(seed): index for index, seed in enumerate(selected_seeds)}
    route_order = {route: index for index, route in enumerate(selected_routes)}
    ordered = sorted(
        rows,
        key=lambda row: (
            seed_order[int(row["seed"])],
            route_order[row["route"]],
        ),
    )
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(output_csv)


def _checkpoint_row(row: dict) -> dict:
    parsed = dict(row)
    parsed["seed"] = int(parsed["seed"])
    parsed["stage1_evs_rank"] = int(parsed["stage1_evs_rank"])
    for key in (
        "stage1_position_error_m",
        "stage1_delay_rmse_ns",
        "stage1_runtime_s",
        "stage1_projection_runtime_s",
        "stage1_evs_retained_energy_fraction",
        "stage1_max_rank1_ratio",
        "stage1_assignment_margin",
        "stage1_jones_anchor_refresh_runtime_s",
        "position_error_m",
        "clock_error_ns",
        "channel_y_nmse",
        "raw_objective_final",
        "total_objective_final",
        "ccop_runtime_s",
    ):
        parsed[key] = float(parsed[key])
    for key in (
        "stage1_jones_anchor_refresh_clock_certified",
        "outlier",
        "clock_certified",
    ):
        parsed[key] = str(parsed[key]).strip().lower() == "true"
    return parsed


def _run_seed(seed: int, spec: dict, options: dict) -> list[dict]:
    """Run selected paired routes for one reproducible raw observation."""
    apply_thread_limits(int(options["blas_threads"]))
    config = apply_ccop_stage1_preset(
        build_ccop_experiment_config(spec, int(seed)), "balanced"
    )
    if options["training_blocks"] is not None:
        config["T"] = int(options["training_blocks"])
    config["ccop_stage1_joint_geometry"] = {
        "num_starts": int(options["joint_num_starts"]),
        "max_iter": int(options["joint_max_iter"]),
        "use_leave_one_out": False,
    }
    if options["refresh_jones_anchor"] is not None:
        config["ccop_stage1_refresh_jones_anchor"] = bool(
            options["refresh_jones_anchor"]
        )
    data = _make_data(config)
    config["noise_variance"] = float(data["noise_variance"])
    selected_routes = tuple(options["routes"])
    stage1_routes = {}
    if any(
        route in selected_routes
        for route in ("frozen_stage1_4d_jones_vp", "frozen_stage1_ccop")
    ):
        frozen_start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            frozen_stage1 = run_stage1_only(data, config)["estimate"]
        frozen_runtime = time.perf_counter() - frozen_start
        for route in ("frozen_stage1_4d_jones_vp", "frozen_stage1_ccop"):
            if route in selected_routes:
                stage1_routes[route] = (frozen_stage1, frozen_runtime)
    if (
        "evs_subspace_stage1_ccop" in selected_routes
        or "evs_subspace_joint_geometry_stage1_ccop" in selected_routes
    ):
        subspace_start = time.perf_counter()
        subspace_stage1 = initialize_ccop_stage1(
            data["Z_noisy"], data["scene"], config
        )
        subspace_runtime = time.perf_counter() - subspace_start
        stage1_routes["evs_subspace_stage1_ccop"] = (
            subspace_stage1,
            subspace_runtime,
        )
    if "evs_subspace_joint_geometry_stage1_ccop" in selected_routes:
        joint_start = time.perf_counter()
        joint_stage1 = refine_ccop_stage1_joint_geometry(
            copy.deepcopy(subspace_stage1), data["scene"], config
        )
        if bool(config["ccop_stage1_refresh_jones_anchor"]):
            joint_stage1 = refresh_ccop_stage1_jones_anchor(
                data["Y_noisy"], joint_stage1, data["scene"], config
            )
        joint_runtime = subspace_runtime + (time.perf_counter() - joint_start)
        stage1_routes["evs_subspace_joint_geometry_stage1_ccop"] = (
            joint_stage1,
            joint_runtime,
        )

    rows = []
    for route in selected_routes:
        stage1_estimate, stage1_runtime = stage1_routes[route]
        rows.append(
            _run_route(
                route,
                data=data,
                config=config,
                outlier_threshold_m=float(options["outlier_threshold_m"]),
                stage1_override=stage1_estimate,
                stage1_runtime_override=stage1_runtime,
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seeds", nargs="+", type=int)
    seed_group.add_argument(
        "--seed-split", choices=("development", "validation", "heldout")
    )
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument(
        "--training-blocks",
        type=int,
        default=None,
        help="optional RIS training-block count for stress ablations",
    )
    parser.add_argument("--diagnostic-mode", choices=("fast", "performance"), default="performance")
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    parser.add_argument("--ccop-outer-max-iter", type=int, default=20)
    parser.add_argument("--clock-fft-size", type=int, default=4096)
    parser.add_argument("--joint-num-starts", type=int, default=4)
    parser.add_argument("--joint-max-iter", type=int, default=30)
    parser.add_argument(
        "--refresh-jones-anchor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="ablate one fixed-position free-Jones clock profile before Stage-III",
    )
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/ccop_full_validation/stage1_paired_pilot"),
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted run from its atomically checkpointed CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_seeds = (
        [int(seed) for seed in args.seeds]
        if args.seeds is not None
        else [int(seed) for seed in seed_splits()[str(args.seed_split)]]
    )
    apply_thread_limits(int(args.blas_threads))
    output_csv = args.out_dir / "paired_routes.csv"
    output_summary = args.out_dir / "summary.json"
    run_spec_path = args.out_dir / "run_spec.json"
    if (
        not args.force_rerun
        and not args.resume
        and (output_csv.exists() or output_summary.exists())
    ):
        raise FileExistsError(f"outputs already exist under {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = _spec(args)
    options = {
        "routes": list(args.routes),
        "joint_num_starts": int(args.joint_num_starts),
        "joint_max_iter": int(args.joint_max_iter),
        "refresh_jones_anchor": args.refresh_jones_anchor,
        "training_blocks": args.training_blocks,
        "outlier_threshold_m": float(args.outlier_threshold_m),
        "blas_threads": int(args.blas_threads),
    }
    run_spec = {
        "seeds": selected_seeds,
        "spec": spec,
        "options": options,
    }
    if args.resume and run_spec_path.exists():
        recorded_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        if canonical_hash(recorded_spec) != canonical_hash(run_spec):
            raise RuntimeError("resume run_spec does not match the recorded experiment")
    else:
        run_spec_path.write_text(
            json.dumps(run_spec, indent=2, sort_keys=True), encoding="utf-8"
        )
    rows = []
    if args.resume and output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = [_checkpoint_row(row) for row in csv.DictReader(handle)]
        expected_routes = set(args.routes)
        if any(
            int(row["seed"]) not in selected_seeds
            or row["route"] not in expected_routes
            for row in rows
        ):
            raise RuntimeError("checkpoint contains seeds or routes outside run_spec")
        completed_by_seed: dict[int, set[str]] = {}
        for row in rows:
            completed_by_seed.setdefault(int(row["seed"]), set()).add(row["route"])
        completed_seeds = {
            seed
            for seed, routes in completed_by_seed.items()
            if routes == expected_routes
        }
    else:
        completed_seeds = set()
    pending_seeds = [seed for seed in selected_seeds if seed not in completed_seeds]

    command = shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]])
    environment = validation_environment(
        command,
        repo_root=pathlib.Path(__file__).resolve().parents[2],
    )
    (args.out_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )
    if int(args.jobs) == 1:
        for seed in pending_seeds:
            seed_rows = _run_seed(int(seed), spec, options)
            rows.extend(seed_rows)
            _write_checkpoint(output_csv, rows, selected_seeds, list(args.routes))
            for row in seed_rows:
                print(json.dumps(row, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {
                executor.submit(_run_seed, int(seed), spec, options): int(seed)
                for seed in pending_seeds
            }
            for future in as_completed(futures):
                seed_rows = future.result()
                rows.extend(seed_rows)
                _write_checkpoint(output_csv, rows, selected_seeds, list(args.routes))
                for row in seed_rows:
                    print(json.dumps(row, sort_keys=True), flush=True)
    seed_order = {int(seed): index for index, seed in enumerate(selected_seeds)}
    route_order = {route: index for index, route in enumerate(args.routes)}
    rows.sort(key=lambda row: (seed_order[int(row["seed"])], route_order[row["route"]]))
    _write_checkpoint(output_csv, rows, selected_seeds, list(args.routes))
    summary = _summary(rows)
    summary["experiment"] = {
        "seed_split": args.seed_split,
        "seeds": selected_seeds,
        "routes": list(args.routes),
        "snr_db": float(args.snr_db),
        "training_blocks": (
            int(args.training_blocks)
            if args.training_blocks is not None
            else "config_default"
        ),
        "diagnostic_mode": str(args.diagnostic_mode),
        "joint_num_starts": int(args.joint_num_starts),
        "joint_max_iter": int(args.joint_max_iter),
        "refresh_jones_anchor": (
            bool(args.refresh_jones_anchor)
            if args.refresh_jones_anchor is not None
            else "preset_default"
        ),
        "ccop_outer_max_iter": int(args.ccop_outer_max_iter),
        "clock_fft_size": int(args.clock_fft_size),
        "blas_threads": int(args.blas_threads),
        "jobs": int(args.jobs),
        "runtime_policy": (
            "single-process timing"
            if int(args.jobs) == 1
            else "parallel throughput run; use single-process development timing for claims"
        ),
    }
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
