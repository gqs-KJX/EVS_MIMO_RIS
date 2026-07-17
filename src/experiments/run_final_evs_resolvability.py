"""Paired delay--polarization resolvability experiment for the frozen estimator.

The experiment varies the achieved minimum physical path-delay separation and
the Jones-vector overlap.  Scalar, dual-polarized, and full-EVS observations
are nested masks of one full-6D noisy realization.  A path set is declared
resolved only when the panel ordering is correct, every Stage-I delay is within
the predeclared tolerance, and no estimated delay pair collapses.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..channel_model import (
    add_awgn,
    channel_components,
    generate_scene,
    synthesize_raw_tensor,
)
from ..geometry import polarization_vector
from ..tensor_utils import hankelize_frequency
from ..validation_artifacts import array_sha256, validation_environment
from .final_mksc_ccop_common import (
    TRIAL_FIELDS,
    Stage1Cache,
    _binomial_interval,
    make_paper_config,
    run_paper_variant,
    save_resolved_config,
    save_run_manifest,
    write_csv,
)
from .resource_control import apply_thread_limits
from .run_paper_ablation_figures import make_nested_receiver_mode_data
from .run_robustness_and_scaling_figures import adjust_config_for_resolvability


RESOLUTION_FIELDS = TRIAL_FIELDS + (
    "target_delay_separation_ns",
    "achieved_delay_separation_ns",
    "polarization_overlap_target",
    "polarization_overlap_achieved",
    "delay_error_tolerance_ns",
    "pole_collapse_tolerance_ns",
    "estimated_path_count",
    "resolution_success",
    "resolution_failure_reason",
)

SUMMARY_FIELDS = (
    "target_delay_separation_ns",
    "achieved_delay_separation_ns_mean",
    "polarization_overlap_target",
    "receiver_mode",
    "n",
    "n_failed",
    "n_resolved",
    "resolution_probability",
    "resolution_ci_low",
    "resolution_ci_high",
    "resolution_ci_method",
    "panel_pairing_accuracy",
    "pole_collapse_rate",
    "stage1_delay_rmse_ns_median",
    "stage1_delay_rmse_ns_p95",
    "position_rmse_m",
    "position_p95_m",
    "clock_rmse_ns",
    "channel_nmse_p95",
)


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _trial_seeds(root_seed: int, count: int) -> list[int]:
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(int(root_seed)).spawn(int(count))
    ]


def _closest_pair(values: np.ndarray) -> tuple[int, int]:
    candidates = [
        (abs(float(values[left] - values[right])), left, right)
        for left in range(values.size)
        for right in range(left + 1, values.size)
    ]
    _, left, right = min(candidates)
    return int(left), int(right)


def _make_full_resolution_data(config: dict, overlap: float) -> tuple[dict, float]:
    """Generate one full-EVS realization with a controlled Jones overlap."""
    rng = np.random.default_rng(int(config["seed"]))
    scene = generate_scene(config, rng)
    provisional = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    left, right = _closest_pair(np.asarray(provisional["taus"], dtype=float))
    scene["gamma_true"][left] = np.pi / 2.0
    scene["eta_true"][left] = 0.0
    scene["gamma_true"][right] = float(np.arcsin(np.clip(overlap, 0.0, 1.0)))
    scene["eta_true"][right] = 0.0
    first = polarization_vector(scene["gamma_true"][left], scene["eta_true"][left])
    second = polarization_vector(scene["gamma_true"][right], scene["eta_true"][right])
    achieved_overlap = float(
        abs(np.vdot(first, second))
        / max(np.linalg.norm(first) * np.linalg.norm(second), 1.0e-300)
    )
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(components, scene["beta_true"])
    y_noisy, noise_variance = add_awgn(
        y_true,
        float(config["SNR_dB"]),
        rng,
        active_mask=scene["evs_observation_mask"],
    )
    data = {
        "scene": scene,
        "true_components": components,
        "Y_true": y_true,
        "Y_noisy": y_noisy,
        "Z_true": hankelize_frequency(y_true, scene["P"]),
        "Z_noisy": hankelize_frequency(y_noisy, scene["P"]),
        "noise_variance": float(noise_variance),
        "timing": {},
    }
    return data, achieved_overlap


def _resolution_failure_reason(
    row: dict[str, Any],
    true_path_count: int,
    delay_tolerance_ns: float,
    collapse_tolerance_ns: float,
) -> str:
    if bool(row.get("failed", False)):
        return "estimator_failure"
    if int(row.get("estimated_path_count", 0)) != int(true_path_count):
        return "path_count_error"
    if not bool(row.get("stage1_panel_pairing_correct", False)):
        return "panel_pairing_error"
    if float(row.get("stage1_delay_max_abs_error_ns", np.inf)) > delay_tolerance_ns:
        return "delay_error"
    if float(row.get("stage1_min_delay_separation_ns", -np.inf)) < collapse_tolerance_ns:
        return "pole_collapse"
    return ""


def _run_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    apply_thread_limits(int(task["blas_threads"]))
    base_config = make_paper_config(
        int(task["seed"]),
        float(task["snr_db"]),
        diagnostic_mode=str(task["diagnostic_mode"]),
        overrides={"receiver_mode": "full_6d"},
    )
    base_config = adjust_config_for_resolvability(
        base_config, float(task["delay_separation_ns"])
    )
    base_data, achieved_overlap = _make_full_resolution_data(
        base_config, float(task["polarization_overlap"])
    )
    base_hash = array_sha256(base_data["Y_noisy"])
    true_taus = np.asarray(base_data["true_components"]["taus"], dtype=float)
    achieved_delay = min(
        abs(float(true_taus[left] - true_taus[right])) * 1.0e9
        for left in range(true_taus.size)
        for right in range(left + 1, true_taus.size)
    )
    rows: list[dict[str, Any]] = []
    for mode in task["receiver_modes"]:
        config = dict(base_config)
        config["receiver_mode"] = str(mode)
        data = make_nested_receiver_mode_data(base_data, str(mode), config)
        config["noise_variance"] = float(data["noise_variance"])
        cache = Stage1Cache(data, config)
        row = run_paper_variant(
            "proposed",
            data=data,
            config=config,
            cache=cache,
            suite="evs_resolvability",
            x_name="delay_separation_ns",
            x_value=float(task["delay_separation_ns"]),
            trial_id=int(task["trial_id"]),
            outlier_threshold_m=float(task["outlier_threshold_m"]),
        )
        try:
            stage1, _ = cache.joint(4, True)
            estimated_path_count = int(
                np.asarray(stage1.get("poles", []), dtype=complex).size
            )
        except Exception:  # noqa: BLE001 - estimator failures remain MC outcomes.
            estimated_path_count = 0
        row["estimated_path_count"] = estimated_path_count
        reason = _resolution_failure_reason(
            row,
            int(base_data["scene"]["K"]),
            float(task["delay_error_tolerance_ns"]),
            float(task["pole_collapse_tolerance_ns"]),
        )
        row.update(
            {
                "variant": f"proposed_{mode}",
                "variant_label": f"MKSC-GI-balanced + CCOP-JVP [{mode}]",
                "receiver_mode": str(mode),
                "base_y_noisy_hash": base_hash,
                "target_delay_separation_ns": float(task["delay_separation_ns"]),
                "achieved_delay_separation_ns": float(achieved_delay),
                "polarization_overlap_target": float(task["polarization_overlap"]),
                "polarization_overlap_achieved": achieved_overlap,
                "delay_error_tolerance_ns": float(task["delay_error_tolerance_ns"]),
                "pole_collapse_tolerance_ns": float(task["pole_collapse_tolerance_ns"]),
                "resolution_success": reason == "",
                "resolution_failure_reason": reason,
            }
        )
        rows.append(row)
    if {str(row["base_y_noisy_hash"]) for row in rows} != {base_hash}:
        raise RuntimeError("receiver modes did not share the full-EVS realization")
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["target_delay_separation_ns"]),
            float(row["polarization_overlap_target"]),
            str(row["receiver_mode"]),
        )
        grouped.setdefault(key, []).append(row)
    result = []
    for (delay, overlap, mode), selected in sorted(grouped.items()):
        resolved = sum(bool(row["resolution_success"]) for row in selected)
        failures = sum(bool(row["failed"]) for row in selected)
        low, high, method = _binomial_interval(resolved, len(selected))

        def finite(key: str) -> np.ndarray:
            values = np.asarray(
                [float(row.get(key, np.nan)) for row in selected], dtype=float
            )
            return values[np.isfinite(values)]

        def percentile(values: np.ndarray, q: float) -> float:
            return float(np.percentile(values, q)) if values.size else float("nan")

        position = finite("position_error_m")
        clock = finite("clock_error_ns")
        channel = finite("channel_nmse")
        delay_rmse = finite("stage1_delay_rmse_ns")
        result.append(
            {
                "target_delay_separation_ns": delay,
                "achieved_delay_separation_ns_mean": float(
                    np.mean(finite("achieved_delay_separation_ns"))
                ),
                "polarization_overlap_target": overlap,
                "receiver_mode": mode,
                "n": len(selected),
                "n_failed": failures,
                "n_resolved": resolved,
                "resolution_probability": resolved / len(selected),
                "resolution_ci_low": low,
                "resolution_ci_high": high,
                "resolution_ci_method": method,
                "panel_pairing_accuracy": float(
                    np.mean(
                        [bool(row["stage1_panel_pairing_correct"]) for row in selected]
                    )
                ),
                "pole_collapse_rate": float(
                    np.mean(
                        [
                            float(row.get("stage1_min_delay_separation_ns", np.inf))
                            < float(row["pole_collapse_tolerance_ns"])
                            for row in selected
                        ]
                    )
                ),
                "stage1_delay_rmse_ns_median": percentile(delay_rmse, 50),
                "stage1_delay_rmse_ns_p95": percentile(delay_rmse, 95),
                "position_rmse_m": float(np.sqrt(np.mean(position**2)))
                if position.size
                else float("nan"),
                "position_p95_m": percentile(position, 95),
                "clock_rmse_ns": float(np.sqrt(np.mean(clock**2)))
                if clock.size
                else float("nan"),
                "channel_nmse_p95": percentile(channel, 95),
            }
        )
    return result


def _plot(summary_csv: pathlib.Path, out_dir: pathlib.Path) -> None:
    mpl_dir = out_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for overlap in sorted({float(row["polarization_overlap_target"]) for row in rows}):
        selected = [
            row for row in rows if np.isclose(float(row["polarization_overlap_target"]), overlap)
        ]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), constrained_layout=True)
        for mode in ("scalar", "dual_pol", "full_6d"):
            mode_rows = [row for row in selected if row["receiver_mode"] == mode]
            if not mode_rows:
                continue
            mode_rows.sort(key=lambda row: float(row["target_delay_separation_ns"]))
            x = [float(row["target_delay_separation_ns"]) for row in mode_rows]
            axes[0].plot(
                x,
                [float(row["resolution_probability"]) for row in mode_rows],
                marker="o",
                label=mode,
            )
            axes[1].plot(
                x,
                [float(row["position_p95_m"]) for row in mode_rows],
                marker="o",
                label=mode,
            )
        axes[0].set_ylabel("Resolution probability")
        axes[0].set_ylim(-0.02, 1.02)
        axes[1].set_ylabel("Position p95 (m)")
        axes[1].set_yscale("log")
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlabel("Minimum path-delay separation (ns)")
            axis.grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Jones-vector overlap = {overlap:g}")
        fig.savefig(
            out_dir / f"evs_resolvability_overlap_{overlap:g}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument(
        "--delay-separation-grid-ns", default="0.1,0.2,0.5,1,2,5"
    )
    parser.add_argument(
        "--polarization-overlap-grid", default="0.1,0.5,0.9,1.0"
    )
    parser.add_argument("--receiver-modes", default="scalar,dual_pol,full_6d")
    parser.add_argument("--delay-error-tolerance-ns", type=float, default=0.5)
    parser.add_argument("--pole-collapse-tolerance-ns", type=float, default=0.05)
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    parser.add_argument(
        "--diagnostic-mode", choices=("fast", "performance"), default="performance"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/final_mksc_ccop/evs_resolvability_smoke"),
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    args.delay_separation_grid_ns = _float_grid(args.delay_separation_grid_ns)
    args.polarization_overlap_grid = _float_grid(args.polarization_overlap_grid)
    args.receiver_modes = _csv_list(args.receiver_modes)
    if args.n_trials < 1 or args.jobs < 1:
        parser.error("--n-trials and --jobs must be positive")
    if any(value <= 0.0 for value in args.delay_separation_grid_ns):
        parser.error("delay separations must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.polarization_overlap_grid):
        parser.error("polarization overlaps must lie in [0, 1]")
    if set(args.receiver_modes) - {"scalar", "dual_pol", "full_6d"}:
        parser.error("receiver modes must be scalar, dual_pol, or full_6d")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.out_dir
    trials_csv = out_dir / "evs_resolvability_trials.csv"
    summary_csv = out_dir / "evs_resolvability_summary.csv"
    if args.plot_only:
        _plot(summary_csv, out_dir)
        return
    if trials_csv.exists() and not args.force_rerun:
        raise FileExistsError(f"{trials_csv} exists; choose a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, "-m", __spec__.name, *(argv or sys.argv[1:])])
    environment = validation_environment(
        command, repo_root=pathlib.Path(__file__).resolve().parents[2]
    )
    save_run_manifest(out_dir, command=command, arguments=vars(args), environment=environment)
    first_seed = _trial_seeds(args.seed, args.n_trials)[0]
    first_config = adjust_config_for_resolvability(
        make_paper_config(
            first_seed,
            args.snr_db,
            diagnostic_mode=args.diagnostic_mode,
            overrides={"receiver_mode": "full_6d"},
        ),
        args.delay_separation_grid_ns[0],
    )
    save_resolved_config(out_dir, first_config, "resolved_base_config.json")
    tasks = []
    for trial_id, seed in enumerate(_trial_seeds(args.seed, args.n_trials)):
        for delay in args.delay_separation_grid_ns:
            for overlap in args.polarization_overlap_grid:
                tasks.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "snr_db": args.snr_db,
                        "delay_separation_ns": delay,
                        "polarization_overlap": overlap,
                        "receiver_modes": list(args.receiver_modes),
                        "delay_error_tolerance_ns": args.delay_error_tolerance_ns,
                        "pole_collapse_tolerance_ns": args.pole_collapse_tolerance_ns,
                        "outlier_threshold_m": args.outlier_threshold_m,
                        "diagnostic_mode": args.diagnostic_mode,
                        "blas_threads": args.blas_threads,
                    }
                )
    rows: list[dict[str, Any]] = []
    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            rows.extend(_run_task(task))
            print(
                f"[{index}/{len(tasks)}] delay={task['delay_separation_ns']} ns "
                f"overlap={task['polarization_overlap']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {executor.submit(_run_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                rows.extend(future.result())
                print(f"[{index}/{len(tasks)}] resolvability", flush=True)
    rows.sort(
        key=lambda row: (
            float(row["polarization_overlap_target"]),
            float(row["target_delay_separation_ns"]),
            int(row["trial_id"]),
            str(row["receiver_mode"]),
        )
    )
    write_csv(trials_csv, rows, RESOLUTION_FIELDS)
    summary = _summarize(rows)
    write_csv(summary_csv, summary, SUMMARY_FIELDS)
    _plot(summary_csv, out_dir)
    (out_dir / "summary.md").write_text(
        "# EVS delay--polarization resolvability\n\n"
        "Resolution requires correct panel pairing, bounded per-path delay error, "
        "and no pole collapse. Failures remain in the denominator.\n",
        encoding="utf-8",
    )
    (out_dir / "completion.json").write_text(
        json.dumps(
            {
                "complete": True,
                "tasks": len(tasks),
                "rows": len(rows),
                "failed_rows": sum(bool(row["failed"]) for row in rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
