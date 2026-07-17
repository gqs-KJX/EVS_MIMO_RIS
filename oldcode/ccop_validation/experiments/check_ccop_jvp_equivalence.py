"""Minimal equivalence checks for experimental CCOP-JVP and CP-NGC.

The channel realization and Stage-I estimate come from
``main_single_proposed``.  This is not a performance Monte Carlo and does not
replace the frozen proposed pipeline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from src.ccop_jvp import CommonClockJonesProfiler, refine_ccop_jvp
from src.config import default_config
from ..cp_ngc import (
    cp_ngc_clock_vector,
    cp_ngc_geometry,
    cp_ngc_statistic,
)
from src.global_vp import (
    _global_exact_spherical_vp_refinement_lbfgsb_reduced,
    _initial_xi_from_stage1,
    _vp_objective_parts_and_grad,
)
from src.main_single_proposed import (
    _apply_main_single_defaults,
    _make_data,
    run_stage1_only,
)
from src.utils import scipy_is_available


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _git_value(arguments: list[str], default: str = "unavailable") -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=pathlib.Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return default
    return result.stdout.strip() or default


def _environment_record(command: str) -> dict:
    scipy_version = "unavailable"
    if scipy_is_available():
        import scipy

        scipy_version = scipy.__version__
    return {
        "command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
        "git_dirty": bool(_git_value(["status", "--porcelain"], default="")),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "scipy_optimizer_available": bool(scipy_is_available()),
    }


def _configured_problem(args: argparse.Namespace) -> dict:
    config = default_config()
    config["seed"] = int(args.seed)
    config["SNR_dB"] = float(args.snr_db)
    config["diagnostic_mode"] = "smoke"
    config["diagnostic_fast_problem_size"] = True
    config["diagnostic_fast_stage1_search"] = True
    config = _apply_main_single_defaults(config)
    config["global_vp"] = dict(config.get("global_vp", {}))
    config["global_vp"].update(
        {
            "solver": "lbfgsb_reduced",
            "mode": "jones_regularized",
            "vp_dictionary_mode": "matrix_free",
            "use_weight": False,
            "use_delay_prior": False,
            "jones_diagonal_loading": 0.0,
            "enable_z_rescue_multistart": False,
            "use_multistart": False,
            "max_iter": int(args.old_max_iter),
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
        }
    )
    config["ccop_jvp"] = {
        "clock_fft_size": int(args.clock_fft_size),
        "clock_abs_tol_objective": float(args.clock_abs_tol),
        "clock_rel_tol": float(args.clock_rel_tol),
        "clock_max_intervals": int(args.clock_max_intervals),
        "outer_max_iter": int(args.outer_max_iter),
        "outer_ftol": 1.0e-12,
        "outer_gtol": 1.0e-8,
    }
    return config


def _pointwise_equivalence(
    profiler: CommonClockJonesProfiler,
    y_raw: np.ndarray,
    stage1_estimate: dict,
    scene: dict,
    config: dict,
    old_result: dict,
) -> tuple[list[dict], dict]:
    xi_stage1 = _initial_xi_from_stage1(stage1_estimate, scene, config)
    clock_low, clock_high = np.asarray(config["delta_t_bounds"], dtype=float)
    points = [
        ("truth", np.asarray(scene["p_u_true"], dtype=float), float(scene["delta_t_true"])),
        ("stage1_low", xi_stage1[:3], float(clock_low)),
        ("stage1_mid", xi_stage1[:3], float(0.5 * (clock_low + clock_high))),
        ("old_solution", np.asarray(old_result["p_u"], dtype=float), float(old_result["delta_t"])),
        ("old_position_high", np.asarray(old_result["p_u"], dtype=float), float(clock_high)),
    ]
    rows = []
    max_abs = 0.0
    max_rel = 0.0
    max_trig_abs = 0.0
    for label, position, clock in points:
        old_parts, _ = _vp_objective_parts_and_grad(
            np.r_[position, clock],
            np.asarray(y_raw, dtype=complex).reshape(-1),
            stage1_estimate,
            scene,
            config,
        )
        new_parts = profiler.evaluate_clock(position, clock)
        old_value = float(old_parts["total_objective"])
        new_value = float(new_parts["total_objective"])
        abs_error = float(abs(old_value - new_value))
        rel_error = float(
            abs_error / max(abs(old_value), abs(new_value), 1.0e-300)
        )
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
        max_trig_abs = max(max_trig_abs, float(new_parts["objective_trig_abs_error"]))
        rows.append(
            {
                "label": label,
                "p_x": float(position[0]),
                "p_y": float(position[1]),
                "p_z": float(position[2]),
                "clock_s": clock,
                "old_total_objective": old_value,
                "ccop_total_objective": new_value,
                "absolute_error": abs_error,
                "relative_error": rel_error,
                "trig_absolute_error": float(new_parts["objective_trig_abs_error"]),
            }
        )
    summary = {
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "max_trig_absolute_error": max_trig_abs,
        "passed": bool(max_rel <= 1.0e-9 or max_abs <= 1.0e-12),
    }
    return rows, summary


def _envelope_gradient_check(
    profiler: CommonClockJonesProfiler,
    scene: dict,
    step_m: float,
) -> dict:
    position = np.asarray(scene["p_u_true"], dtype=float).copy()
    profile = profiler.profile_clock(position)
    analytic = np.asarray(profile["gradient_p"], dtype=float)
    finite_difference = np.empty(3, dtype=float)
    for dim in range(3):
        direction = np.zeros(3, dtype=float)
        direction[dim] = float(step_m)
        plus = profiler.profile_clock(position + direction)
        minus = profiler.profile_clock(position - direction)
        finite_difference[dim] = (
            float(plus["total_objective"]) - float(minus["total_objective"])
        ) / (2.0 * float(step_m))
    absolute = np.abs(analytic - finite_difference)
    relative_norm = float(
        np.linalg.norm(analytic - finite_difference)
        / max(np.linalg.norm(analytic), np.linalg.norm(finite_difference), 1.0e-12)
    )
    return {
        "position": position,
        "profiled_clock_s": float(profile["delta_t"]),
        "clock_certified": bool(profile["clock_certified"]),
        "clock_certificate_gap_objective": float(
            profile["clock_certificate_gap_objective"]
        ),
        "analytic": analytic,
        "finite_difference": finite_difference,
        "max_absolute_error": float(np.max(absolute)),
        "relative_norm_error": relative_norm,
        "passed": bool(
            profile["clock_certified"]
            and (relative_norm <= 2.0e-3 or float(np.max(absolute)) <= 1.0e-8)
        ),
    }


def _cp_ngc_empirical_check(
    scene: dict,
    *,
    trials: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    k_paths = int(scene["K"])
    dimension = 4 * k_paths
    standard_deviation = np.concatenate(
        [
            np.full(k_paths, 0.25e-9, dtype=float),
            np.tile(np.array([0.08, 0.010, 0.012], dtype=float), k_paths),
        ]
    )
    indices = np.arange(dimension)
    correlation = 0.18 ** np.abs(indices[:, None] - indices[None, :])
    correlation_factor = np.linalg.cholesky(correlation)
    covariance_factor = np.diag(standard_deviation) @ correlation_factor
    covariance = covariance_factor @ covariance_factor.T
    geometry = cp_ngc_geometry(scene["p_u_true"], scene)
    clock_vector = cp_ngc_clock_vector(scene)
    mean = geometry + clock_vector * float(scene["delta_t_true"])
    rng = np.random.default_rng(int(seed))
    standard_normal = rng.standard_normal((int(trials), dimension))
    observations = mean[None, :] + standard_normal @ covariance_factor.T
    statistics = np.empty(int(trials), dtype=float)
    first_diagnostic = None
    for trial in range(int(trials)):
        diagnostic = cp_ngc_statistic(
            observations[trial], scene["p_u_true"], covariance, scene
        )
        statistics[trial] = float(diagnostic["statistic"])
        if first_diagnostic is None:
            first_diagnostic = diagnostic
    assert first_diagnostic is not None
    dof = int(first_diagnostic["dof"])
    empirical_mean = float(np.mean(statistics))
    empirical_variance = float(np.var(statistics, ddof=1))
    empirical_q95 = float(np.quantile(statistics, 0.95))
    if scipy_is_available():
        from scipy.stats import chi2, kstest

        theoretical_q95 = float(chi2.ppf(0.95, dof))
        ks = kstest(statistics, chi2(df=dof).cdf)
        ks_statistic = float(ks.statistic)
        ks_pvalue = float(ks.pvalue)
    else:
        theoretical_q95 = float("nan")
        ks_statistic = float("nan")
        ks_pvalue = float("nan")
    mean_relative_error = float(abs(empirical_mean - dof) / dof)
    variance_relative_error = float(
        abs(empirical_variance - 2.0 * dof) / (2.0 * dof)
    )
    q95_relative_error = float(
        abs(empirical_q95 - theoretical_q95) / theoretical_q95
    )
    if scipy_is_available():
        distribution_pass = bool(
            ks_pvalue >= 0.01
            and mean_relative_error <= 0.05
            and q95_relative_error <= 0.08
        )
    else:
        distribution_pass = bool(
            mean_relative_error <= 0.05 and variance_relative_error <= 0.10
        )
    summary = {
        "trials": int(trials),
        "dof": dof,
        "empirical_mean": empirical_mean,
        "theoretical_mean": float(dof),
        "mean_relative_error": mean_relative_error,
        "empirical_variance": empirical_variance,
        "theoretical_variance": float(2 * dof),
        "variance_relative_error": variance_relative_error,
        "empirical_q95": empirical_q95,
        "theoretical_q95": theoretical_q95,
        "q95_relative_error": q95_relative_error,
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "projected_geometry_rank": int(
            first_diagnostic["projected_geometry_rank"]
        ),
        "projected_geometry_singular_values": np.asarray(
            first_diagnostic["projected_geometry_singular_values"], dtype=float
        ),
        "distribution_passed": distribution_pass,
        "rank_passed": bool(first_diagnostic["projected_geometry_rank"] == 3),
        "passed": bool(
            distribution_pass and first_diagnostic["projected_geometry_rank"] == 3
        ),
        "covariance": covariance,
    }
    return statistics, summary


def _write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    output_dir: pathlib.Path,
    config_record: dict,
    point_rows: list[dict],
    samples: np.ndarray,
    summaries: dict,
    timing_s: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(_jsonable(config_record), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "pointwise_objectives.csv",
        point_rows,
        list(point_rows[0].keys()),
    )
    _write_csv(
        output_dir / "cp_ngc_samples.csv",
        [
            {"trial": int(index), "statistic": float(value)}
            for index, value in enumerate(samples)
        ],
        ["trial", "statistic"],
    )
    check_rows = []
    for name, summary in summaries.items():
        check_rows.append(
            {
                "check": name,
                "passed": bool(summary["passed"]),
                "primary_value": float(
                    summary.get(
                        "max_relative_error",
                        summary.get(
                            "relative_norm_error",
                            summary.get(
                                "objective_gain",
                                summary.get("ks_pvalue", np.nan),
                            ),
                        ),
                    )
                ),
            }
        )
    _write_csv(
        output_dir / "checks.csv",
        check_rows,
        ["check", "passed", "primary_value"],
    )
    lines = [
        "# CCOP-JVP / CP-NGC equivalence summary",
        "",
        "This is a smoke/equivalence experiment, not a performance claim.",
        "",
        "| Check | Pass | Key diagnostics |",
        "|---|---:|---|",
        (
            "| Pointwise old/new objective | "
            f"{summaries['pointwise']['passed']} | "
            f"max rel {summaries['pointwise']['max_relative_error']:.3e}, "
            f"max abs {summaries['pointwise']['max_absolute_error']:.3e} |"
        ),
        (
            "| Envelope gradient | "
            f"{summaries['gradient']['passed']} | "
            f"relative norm {summaries['gradient']['relative_norm_error']:.3e}, "
            f"max abs {summaries['gradient']['max_absolute_error']:.3e} |"
        ),
        (
            "| Old incumbent non-degradation | "
            f"{summaries['incumbent']['passed']} | "
            f"old-new gain {summaries['incumbent']['objective_gain']:.3e} |"
        ),
        (
            "| CP-NGC truth chi-square | "
            f"{summaries['cp_ngc']['passed']} | "
            f"df {summaries['cp_ngc']['dof']}, mean "
            f"{summaries['cp_ngc']['empirical_mean']:.3f}, KS p "
            f"{summaries['cp_ngc']['ks_pvalue']:.3g}, geometry rank "
            f"{summaries['cp_ngc']['projected_geometry_rank']} |"
        ),
        "",
        "The CP-NGC distribution check uses calibrated correlated Gaussian "
        "perturbations of the joint delay/RIS-geometry vector at the fixed true "
        "position. It does not validate Stage-I covariance calibration or the "
        "pipeline false-green rate.",
        "",
        "Runtime is diagnostic only: "
        f"old 4-D VP {timing_s['old_4d_vp']:.3f} s, "
        f"CCOP 3-D refinement {timing_s['ccop_refinement']:.3f} s, "
        f"total {timing_s['total']:.3f} s.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Run four strict CCOP-JVP/CP-NGC equivalence checks."
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--old-max-iter", type=int, default=8)
    parser.add_argument("--outer-max-iter", type=int, default=6)
    parser.add_argument("--clock-fft-size", type=int, default=4096)
    parser.add_argument("--clock-abs-tol", type=float, default=1.0e-12)
    parser.add_argument("--clock-rel-tol", type=float, default=1.0e-10)
    parser.add_argument("--clock-max-intervals", type=int, default=20000)
    parser.add_argument("--gradient-step-m", type=float, default=1.0e-4)
    parser.add_argument("--cp-ngc-trials", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/ccop_jvp_equivalence"),
    )
    parser.add_argument(
        "--allow-failed-checks",
        action="store_true",
        help="Write diagnostics and exit zero even if a check fails.",
    )
    args = parser.parse_args(argv)
    if args.cp_ngc_trials < 200:
        raise ValueError("--cp-ngc-trials must be at least 200 for an empirical check")
    command = shlex.join(
        [sys.executable, "-m", "src.experiments.check_ccop_jvp_equivalence", *(argv or sys.argv[1:])]
    )
    print(f"command = {command}", flush=True)
    config = _configured_problem(args)
    data_start = time.perf_counter()
    data = _make_data(config)
    data_runtime = time.perf_counter() - data_start
    config["noise_variance"] = float(data["noise_variance"])
    stage1_start = time.perf_counter()
    stage1_record = run_stage1_only(data, config)
    stage1_runtime = time.perf_counter() - stage1_start
    stage1_estimate = stage1_record["estimate"]
    scene = data["scene"]

    old_start = time.perf_counter()
    old_result = _global_exact_spherical_vp_refinement_lbfgsb_reduced(
        data["Y_noisy"], copy.deepcopy(stage1_estimate), scene, config
    )
    old_runtime = time.perf_counter() - old_start
    profiler = CommonClockJonesProfiler(
        data["Y_noisy"], stage1_estimate, scene, config
    )
    point_start = time.perf_counter()
    point_rows, pointwise = _pointwise_equivalence(
        profiler,
        data["Y_noisy"],
        stage1_estimate,
        scene,
        config,
        old_result,
    )
    point_runtime = time.perf_counter() - point_start
    gradient_start = time.perf_counter()
    gradient = _envelope_gradient_check(
        profiler, scene, float(args.gradient_step_m)
    )
    gradient_runtime = time.perf_counter() - gradient_start
    ccop_start = time.perf_counter()
    ccop_result = refine_ccop_jvp(
        data["Y_noisy"],
        copy.deepcopy(stage1_estimate),
        scene,
        config,
        incumbent=old_result,
    )
    ccop_runtime = time.perf_counter() - ccop_start
    incumbent_gain = float(
        ccop_result["incumbent_old_objective"]
        - ccop_result["total_objective_final"]
    )
    incumbent = {
        "old_objective": float(ccop_result["incumbent_old_objective"]),
        "old_position_profiled_objective": float(
            ccop_result["incumbent_profiled_objective"]
        ),
        "new_objective": float(ccop_result["total_objective_final"]),
        "objective_gain": incumbent_gain,
        "clock_certificate_gap_objective": float(
            ccop_result["clock_certificate_gap_objective"]
        ),
        "passed": bool(ccop_result["incumbent_non_degradation"]),
    }
    cp_start = time.perf_counter()
    cp_samples, cp_summary = _cp_ngc_empirical_check(
        scene,
        trials=int(args.cp_ngc_trials),
        seed=int(args.seed) + 1,
    )
    cp_runtime = time.perf_counter() - cp_start
    summaries = {
        "pointwise": pointwise,
        "gradient": gradient,
        "incumbent": incumbent,
        "cp_ngc": cp_summary,
    }
    environment = _environment_record(command)
    timing_s = {
        "data_generation_and_hankelization": float(data_runtime),
        "stage1": float(stage1_runtime),
        "old_4d_vp": float(old_runtime),
        "pointwise_checks": float(point_runtime),
        "gradient_check": float(gradient_runtime),
        "ccop_refinement": float(ccop_runtime),
        "cp_ngc_empirical": float(cp_runtime),
        "total": float(time.perf_counter() - total_start),
    }
    config_record = {
        "experiment": "CCOP-JVP four equivalence checks on main_single_proposed smoke model",
        "environment": environment,
        "arguments": vars(args),
        "repository_config": config,
        "summaries": summaries,
        "timing_s": timing_s,
    }
    _write_outputs(
        args.output_dir,
        config_record,
        point_rows,
        cp_samples,
        summaries,
        timing_s,
    )

    for name, summary in summaries.items():
        print(f"{name}: passed={summary['passed']}")
    print(
        "pointwise max rel/abs = "
        f"{pointwise['max_relative_error']:.3e} / {pointwise['max_absolute_error']:.3e}"
    )
    print(
        "gradient relative/max abs = "
        f"{gradient['relative_norm_error']:.3e} / {gradient['max_absolute_error']:.3e}"
    )
    print(f"incumbent objective gain = {incumbent_gain:.3e}")
    print(
        "CP-NGC df/mean/KS-p/rank = "
        f"{cp_summary['dof']} / {cp_summary['empirical_mean']:.3f} / "
        f"{cp_summary['ks_pvalue']:.3g} / {cp_summary['projected_geometry_rank']}"
    )
    print(f"outputs = {args.output_dir}")
    all_passed = all(bool(summary["passed"]) for summary in summaries.values())
    if not all_passed and not args.allow_failed_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
