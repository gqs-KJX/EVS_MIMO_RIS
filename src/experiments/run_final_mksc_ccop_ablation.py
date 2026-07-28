"""Publication-grade paired ablations for frozen MKSC-GI--CCOP-JVP.

Large runs are intentionally opt-in.  The default is a one-realization smoke
configuration; paper commands should explicitly set ``--n-trials`` and a new
output directory.  Plots are generated only from the saved summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..validation_artifacts import array_sha256, canonical_hash, validation_environment
from .final_mksc_ccop_common import (
    PAIRED_FIELDS,
    PAPER_VARIANTS,
    SUMMARY_FIELDS,
    TRIAL_FIELDS,
    make_paper_config,
    make_shared_data,
    paired_summary,
    run_paired_variants,
    save_run_manifest,
    save_resolved_config,
    summarize_rows,
    write_csv,
)
from .resource_control import apply_thread_limits
from .run_paper_ablation_figures import _peb_from_efim, make_nested_receiver_mode_data


DEFAULT_SUITES = "components"
DEFAULT_COMPONENT_VARIANTS = (
    "scaled_4d",
    "old_stage1_ccop",
    "mksc_delay_ccop",
    "mksc_gi_1_no_refresh_ccop",
    "mksc_gi_4_no_refresh_ccop",
    "proposed",
)
DEFAULT_SNR_VARIANTS = (
    "scaled_4d",
    "old_stage1_ccop",
    "mksc_gi_4_no_refresh_ccop",
    "proposed",
)
DEFAULT_RECEIVER_VARIANTS = ("proposed",)
DEFAULT_C3_MATRIX_VARIANTS = (
    "old_4d",
    "scaled_4d",
    "old_stage1_ccop",
    "mksc_gi_refresh_4d_seconds",
    "mksc_gi_refresh_4d_distance_m",
    "proposed",
)
DEFAULT_CLOCK_UNIT_VARIANTS = (
    "mksc_gi_refresh_4d_seconds",
    "mksc_gi_refresh_4d_nanoseconds",
    "mksc_gi_refresh_4d_distance_m",
    "mksc_gi_refresh_ccop_seconds",
    "mksc_gi_refresh_ccop_nanoseconds",
    "mksc_gi_refresh_ccop_distance_m",
)


def _float_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _int_grid(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _trial_seeds(root_seed: int, n_trials: int) -> list[int]:
    sequence = np.random.SeedSequence(int(root_seed))
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in sequence.spawn(int(n_trials))
    ]


def _task_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    apply_thread_limits(int(task["blas_threads"]))
    if str(task["suite"]) == "receiver":
        return _receiver_task_rows(task)
    overrides = dict(task.get("overrides", {}))
    config = make_paper_config(
        int(task["seed"]),
        float(task["snr_db"]),
        diagnostic_mode=str(task["diagnostic_mode"]),
        overrides=overrides,
    )
    data = make_shared_data(config)
    rows = run_paired_variants(
        task["variants"],
        data=data,
        config=config,
        suite=str(task["suite"]),
        x_name=str(task["x_name"]),
        x_value=task["x_value"],
        trial_id=int(task["trial_id"]),
        outlier_threshold_m=float(task["outlier_threshold_m"]),
        profile_memory=bool(task.get("profile_memory", False)),
    )
    if str(task["suite"]) == "snr" and bool(task.get("include_peb", False)):
        rows.append(
            _peb_reference_row(
                data,
                config,
                task,
                variant="free_jones_peb",
                label="Data-only free-Jones PEB",
            )
        )
    return rows


def _peb_reference_row(
    data: dict,
    config: dict,
    task: dict[str, Any],
    *,
    variant: str,
    label: str,
) -> dict[str, Any]:
    """Build one matched data-only PEB reference row for saved-CSV plotting."""
    start = time.perf_counter()
    metrics = _peb_from_efim(data, config)
    runtime = time.perf_counter() - start
    row = {field: "" for field in TRIAL_FIELDS}
    scene = data["scene"]
    truth = np.asarray(scene["p_u_true"], dtype=float)
    row.update(
        {
            "suite": str(task["suite"]),
            "x_name": str(task["x_name"]),
            "x_value": task["x_value"],
            "trial_id": int(task["trial_id"]),
            "seed": int(config["seed"]),
            "snr_db": float(config["SNR_dB"]),
            "variant": str(variant),
            "variant_label": str(label),
            "failed": False,
            "error": "",
            "resolved_config_hash": canonical_hash(config),
            "y_noisy_hash": array_sha256(data["Y_noisy"]),
            "base_y_noisy_hash": str(
                data.get("nested_base_y_noisy_hash", array_sha256(data["Y_noisy"]))
            ),
            "nested_receiver_noise_convention": str(
                data.get("nested_receiver_noise_convention", "")
            ),
            "reference_sigma2": float(
                data.get("reference_sigma2", data["noise_variance"])
            ),
            "reference_type": "matched_data_only_free_jones",
            "peb_position_m": float(
                metrics.get("peb_free_jones_m", metrics["peb_position_m"])
            ),
            "deployment_runtime_s": float(runtime),
            "stage3_runtime_s": float(runtime),
            "receiver_mode": str(scene.get("receiver_mode", "")),
            "N": int(scene["N"]),
            "T": int(scene["T"]),
            "K": int(scene["K"]),
            "M_A": int(scene["M_A"]),
            "M_R": int(scene["M_R"]),
            "p_true_x": float(truth[0]),
            "p_true_y": float(truth[1]),
            "p_true_z": float(truth[2]),
        }
    )
    return row


def _receiver_task_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate nested receiver modes from one shared full-6D realization."""
    common_overrides = dict(task.get("overrides", {}))
    common_overrides["receiver_mode"] = "full_6d"
    base_config = make_paper_config(
        int(task["seed"]),
        float(task["snr_db"]),
        diagnostic_mode=str(task["diagnostic_mode"]),
        overrides=common_overrides,
    )
    base_data = make_shared_data(base_config)
    base_hash = array_sha256(base_data["Y_noisy"])
    rows: list[dict[str, Any]] = []
    for mode in task["receiver_modes"]:
        mode_overrides = dict(task.get("overrides", {}))
        mode_overrides["receiver_mode"] = str(mode)
        config = make_paper_config(
            int(task["seed"]),
            float(task["snr_db"]),
            diagnostic_mode=str(task["diagnostic_mode"]),
            overrides=mode_overrides,
        )
        data = make_nested_receiver_mode_data(base_data, str(mode), config)
        config["noise_variance"] = float(data["noise_variance"])
        mode_rows = run_paired_variants(
            task["variants"],
            data=data,
            config=config,
            suite="receiver",
            x_name="snr_db",
            x_value=float(task["snr_db"]),
            trial_id=int(task["trial_id"]),
            outlier_threshold_m=float(task["outlier_threshold_m"]),
            profile_memory=bool(task.get("profile_memory", False)),
        )
        for row in mode_rows:
            original_variant = str(row["variant"])
            row["variant"] = f"{original_variant}_{mode}"
            row["variant_label"] = f"{row['variant_label']} [{mode}]"
            row["base_y_noisy_hash"] = base_hash
        rows.extend(mode_rows)

        row = _peb_reference_row(
            data,
            config,
            {
                **task,
                "suite": "receiver",
                "x_name": "snr_db",
                "x_value": float(task["snr_db"]),
            },
            variant=f"free_jones_peb_{mode}",
            label=f"Data-only free-Jones PEB [{mode}]",
        )
        row["base_y_noisy_hash"] = base_hash
        rows.append(row)
    if {str(row["base_y_noisy_hash"]) for row in rows} != {base_hash}:
        raise RuntimeError("nested receiver modes did not share the full-6D realization")
    return rows


def _tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = _trial_seeds(args.seed, args.n_trials)
    suites = set(args.suites)
    tasks: list[dict[str, Any]] = []

    # Every suite of this runner evaluates the reference RIS geometry, so the
    # acquisition setting is a property of the run, not of a suite.  It is
    # applied here rather than in ``make_paper_config`` because that builder is
    # shared with the robustness runner, which sweeps the RIS side length and
    # must keep the fixed-count dictionary at the small array sizes where the
    # beam-space direction-cosine grid becomes the coarser of the two.
    acquisition_override = {
        "ris_search": {"coarse_codebook_mode": str(args.coarse_codebook_mode)}
    }

    def append(
        suite: str,
        x_name: str,
        x_value: Any,
        snr_db: float,
        variants: list[str] | tuple[str, ...],
        overrides: dict | None = None,
        **extra: Any,
    ) -> None:
        merged_overrides = dict(overrides or {})
        merged_overrides["ris_search"] = {
            **dict(merged_overrides.get("ris_search", {})),
            **acquisition_override["ris_search"],
        }
        for trial_id, seed in enumerate(seeds):
            tasks.append(
                {
                    "suite": suite,
                    "x_name": x_name,
                    "x_value": x_value,
                    "trial_id": trial_id,
                    "seed": seed,
                    "snr_db": float(snr_db),
                    "variants": list(variants),
                    "overrides": dict(merged_overrides),
                    "diagnostic_mode": args.diagnostic_mode,
                    "outlier_threshold_m": args.outlier_threshold_m,
                    "blas_threads": args.blas_threads,
                    "profile_memory": bool(args.profile_memory),
                    **extra,
                }
            )

    if "components" in suites:
        append(
            "components",
            "component_set",
            "all",
            args.focus_snr_db,
            args.component_variants,
        )
    if "snr" in suites:
        for snr_db in args.snr_grid:
            append(
                "snr",
                "snr_db",
                snr_db,
                snr_db,
                args.snr_variants,
                include_peb=bool(args.snr_peb),
            )
    if "compression" in suites:
        for snr_db in args.snr_grid:
            append(
                "compression",
                "snr_db",
                snr_db,
                snr_db,
                ("raw_delay_gi_ccop", "proposed"),
            )
    if "clock_units" in suites:
        append(
            "clock_units",
            "clock_parameterization",
            "seconds_vs_nanoseconds_vs_distance",
            args.focus_snr_db,
            args.clock_unit_variants,
        )
    if "c3_matrix" in suites:
        for snr_db in args.snr_grid:
            append(
                "c3_matrix",
                "snr_db",
                snr_db,
                snr_db,
                args.c3_variants,
            )
    if "receiver" in suites:
        for snr_db in args.snr_grid:
            append(
                "receiver",
                "snr_db",
                snr_db,
                snr_db,
                args.receiver_variants,
                {"receiver_mode": "full_6d"},
                receiver_modes=list(args.receiver_modes),
            )
    if "training" in suites:
        for training_blocks in args.training_grid:
            append(
                "training",
                "T",
                training_blocks,
                args.focus_snr_db,
                ["old_4d", "old_stage1_ccop", "proposed"],
                {"T": int(training_blocks)},
            )
    return tasks


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_markdown(path: pathlib.Path, summary: list[dict], paired: list[dict]) -> None:
    lines = [
        "# Final MKSC-GI--CCOP ablation summary",
        "",
        "Failures are included in the outlier/catastrophic rate. Position tails, channel tails, and deployment runtime are reported separately.",
        "",
        "| Suite | x | Variant | n | Success | Position RMSE (m) | Position p95 (m) | Catastrophic | Clock RMSE (ns) | Channel NMSE p95 | Runtime median (s) | Peak RSS max (MiB) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {suite} | {x_value} | {variant} | {n} | {success_rate:.3%} | {position_rmse_m:.6g} | "
            "{position_p95_m:.6g} | {catastrophic_rate:.3%} | {clock_rmse_ns:.6g} | {channel_nmse_p95:.6g} | "
            "{runtime_median_s:.6g} | {peak_rss_max_mb:.6g} |".format(**row)
        )
    c3_rows = [
        row for row in summary if row["suite"] in {"c3_matrix", "clock_units"}
    ]
    if c3_rows:
        lines.extend(
            [
                "",
                "## C3 initializer/solver and clock-unit controls",
                "",
                (
                    "CCOP unit-labelled rows deliberately invoke the same "
                    "profiled 3-D solver: the unit label is not exposed as "
                    "an optimization coordinate."
                ),
                "",
                (
                    "| Suite | SNR/x | Variant | Initializer | Inner solver | "
                    "Clock units | Dim. | Eval. median | Clock RMSE (ns) | "
                    "All-pos. cert. | Trajectory max gap/tol. |"
                ),
                "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in c3_rows:
            lines.append(
                "| {suite} | {x_value} | {variant} | {initializer_family} | "
                "{inner_solver} | {clock_parameterization} | "
                "{nonlinear_dim_median:.0f} | "
                "{optimizer_evaluations_median:.3g} | {clock_rmse_ns:.6g} | "
                "{clock_all_positions_certified_rate:.3g} | "
                "{clock_certificate_gap_ratio_max_over_positions_max:.3g} |".format(
                    **row
                )
            )
    if paired:
        lines.extend(
            [
                "",
                "## Configured paired comparison",
                "",
                "| Suite | x | Reference | Candidate | pairs | rescued | introduced | McNemar p | runtime ratio |",
                "|---|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in paired:
            lines.append(
                "| {suite} | {x_value} | {reference_variant} | {candidate_variant} | "
                "{n_pairs} | {rescued_outliers} | "
                "{introduced_outliers} | {mcnemar_exact_p:.4g} | "
                "{paired_runtime_ratio_median:.4g} |".format(**row)
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_from_summary(
    summary_csv: pathlib.Path,
    out_dir: pathlib.Path,
    trials_csv: pathlib.Path | None = None,
) -> None:
    """Create paper-ready vector plots only from the saved summary CSV."""
    mpl_dir = out_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = _read_csv(summary_csv)
    for suite in sorted({row["suite"] for row in rows}):
        selected = [row for row in rows if row["suite"] == suite]
        if not selected:
            continue
        variants = list(dict.fromkeys(row["variant"] for row in selected))
        x_values = list(dict.fromkeys(row["x_value"] for row in selected))
        numeric_x = []
        numeric = True
        for value in x_values:
            try:
                numeric_x.append(float(value))
            except ValueError:
                numeric = False
                break
        x_plot = numeric_x if numeric else list(range(len(x_values)))
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), constrained_layout=True)
        metrics = (
            ("position_rmse_m", "Position RMSE (m)", True),
            ("catastrophic_rate", "Outlier / failure probability", False),
            ("clock_rmse_ns", "Clock RMSE (ns)", True),
            ("channel_nmse_p95", "Channel NMSE p95", True),
        )
        for axis, (metric, label, log_y) in zip(axes.flat, metrics):
            for variant in variants:
                by_x = {row["x_value"]: row for row in selected if row["variant"] == variant}
                if not all(value in by_x for value in x_values):
                    continue
                if variant.startswith("free_jones_peb") or variant.startswith(
                    "constrained_jones_peb"
                ):
                    if metric != "position_rmse_m":
                        continue
                    source_metric = "peb_position_m_rms"
                else:
                    source_metric = metric
                y = [float(by_x[value][source_metric]) for value in x_values]
                axis.plot(x_plot, y, marker="o", label=variant)
            if log_y:
                axis.set_yscale("log")
            axis.set_ylabel(label)
            axis.grid(True, which="both", alpha=0.3)
            if not numeric:
                axis.set_xticks(x_plot, x_values, rotation=20, ha="right")
        axes[0, 0].legend(fontsize=7)
        fig.savefig(out_dir / f"ablation_{suite}.pdf", bbox_inches="tight")
        plt.close(fig)
    components = [row for row in rows if row["suite"] == "components"]
    components = [
        row
        for row in components
        if row["variant"] not in {"free_jones_peb", "constrained_jones_peb"}
    ]
    if components:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
        for axis, metric, ylabel in (
            (axes[0], "position_p95_m", "Position p95 (m)"),
            (axes[1], "outlier_rate", "Wrong-basin probability"),
        ):
            for row in components:
                axis.scatter(
                    float(row["runtime_median_s"]),
                    float(row[metric]),
                    s=45,
                    label=row["variant"],
                )
                axis.annotate(
                    row["variant"],
                    (float(row["runtime_median_s"]), float(row[metric])),
                    fontsize=6,
                )
            axis.set_xlabel("Median deployment runtime (s)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
        axes[0].set_yscale("log")
        fig.savefig(out_dir / "ablation_components_pareto.pdf", bbox_inches="tight")
        plt.close(fig)
    if trials_csv is None or not trials_csv.exists():
        return
    with trials_csv.open(newline="", encoding="utf-8") as handle:
        trial_rows = [
            row
            for row in csv.DictReader(handle)
            if row["suite"] == "components"
            and row["failed"].lower() != "true"
            and row["variant"] not in {"free_jones_peb", "constrained_jones_peb"}
        ]
    if not trial_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
    for variant in dict.fromkeys(row["variant"] for row in trial_rows):
        selected = [row for row in trial_rows if row["variant"] == variant]
        for axis, key, xlabel in (
            (axes[0], "position_error_m", "Position error (m)"),
            (axes[1], "channel_nmse", "Channel NMSE"),
        ):
            values = np.sort(np.asarray([float(row[key]) for row in selected]))
            probability = np.arange(1, values.size + 1) / values.size
            axis.step(values, probability, where="post", label=variant)
            axis.set_xscale("log")
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Empirical CDF")
            axis.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.savefig(out_dir / "ablation_components_ecdf.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", default=DEFAULT_SUITES)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--snr-grid", default="-30,-25,-20,-15,-10,-5,0,5,10,15,20"
    )
    parser.add_argument("--focus-snr-db", type=float, default=-10.0)
    parser.add_argument(
        "--coarse-codebook-mode",
        choices=("beamspace_only", "union"),
        default="beamspace_only",
        help=(
            "RIS acquisition candidate set. 'beamspace_only' (default) matches "
            "run_benchmark_comparison.make_config, so this runner and the "
            "external benchmark measure the same estimator; 'union' also scores "
            "the legacy fixed-count angular dictionary, which is the published "
            "acquisition and is redundant at the reference array size."
        ),
    )
    parser.add_argument("--training-grid", default="32,64,128,256")
    parser.add_argument("--receiver-modes", default="scalar,dual_pol,full_6d")
    parser.add_argument(
        "--component-variants", default=",".join(DEFAULT_COMPONENT_VARIANTS)
    )
    parser.add_argument("--snr-variants", default=",".join(DEFAULT_SNR_VARIANTS))
    parser.add_argument(
        "--receiver-variants", default=",".join(DEFAULT_RECEIVER_VARIANTS)
    )
    parser.add_argument(
        "--c3-variants", default=",".join(DEFAULT_C3_MATRIX_VARIANTS)
    )
    parser.add_argument(
        "--clock-unit-variants",
        default=",".join(DEFAULT_CLOCK_UNIT_VARIANTS),
    )
    parser.add_argument("--paired-reference", default="scaled_4d")
    parser.add_argument("--paired-candidate", default="proposed")
    parser.add_argument(
        "--snr-peb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include a matched data-only free-Jones PEB row in the SNR suite",
    )
    parser.add_argument("--diagnostic-mode", choices=("fast", "performance"), default="performance")
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="sample per-route process RSS every 10 ms; use jobs=1 for tables",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/final_mksc_ccop/ablation_smoke"),
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    args.suites = _csv_list(args.suites)
    unknown_suites = set(args.suites) - {
        "components",
        "snr",
        "compression",
        "clock_units",
        "c3_matrix",
        "receiver",
        "training",
    }
    if unknown_suites:
        parser.error(f"unknown suites: {sorted(unknown_suites)}")
    args.snr_grid = _float_grid(args.snr_grid)
    args.training_grid = _int_grid(args.training_grid)
    args.receiver_modes = _csv_list(args.receiver_modes)
    args.component_variants = _csv_list(args.component_variants)
    args.snr_variants = _csv_list(args.snr_variants)
    args.receiver_variants = _csv_list(args.receiver_variants)
    args.c3_variants = _csv_list(args.c3_variants)
    args.clock_unit_variants = _csv_list(args.clock_unit_variants)
    unknown_variants = (
        set(args.component_variants)
        | set(args.snr_variants)
        | set(args.receiver_variants)
        | set(args.c3_variants)
        | set(args.clock_unit_variants)
        | {str(args.paired_reference), str(args.paired_candidate)}
    ) - set(PAPER_VARIANTS)
    if unknown_variants:
        parser.error(f"unknown variants: {sorted(unknown_variants)}")
    if args.n_trials < 1 or args.jobs < 1:
        parser.error("--n-trials and --jobs must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_thread_limits(int(args.blas_threads))
    out_dir = args.out_dir
    trials_csv = out_dir / "ablation_trials.csv"
    summary_csv = out_dir / "ablation_summary.csv"
    paired_csv = out_dir / "ablation_paired.csv"
    if args.plot_only:
        _plot_from_summary(summary_csv, out_dir, trials_csv)
        return
    if trials_csv.exists() and not args.force_rerun:
        raise FileExistsError(f"{trials_csv} exists; choose a new directory or use --force-rerun")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, "-m", __spec__.name, *(argv or sys.argv[1:])])
    environment = validation_environment(
        command, repo_root=pathlib.Path(__file__).resolve().parents[2]
    )
    save_run_manifest(
        out_dir,
        command=command,
        arguments=vars(args),
        environment=environment,
    )
    first_seed = _trial_seeds(args.seed, args.n_trials)[0]
    save_resolved_config(
        out_dir,
        make_paper_config(
            first_seed,
            args.focus_snr_db,
            diagnostic_mode=args.diagnostic_mode,
            overrides={
                "ris_search": {
                    "coarse_codebook_mode": str(args.coarse_codebook_mode)
                }
            },
        ),
        "resolved_base_config.json",
    )
    tasks = _tasks(args)
    rows: list[dict[str, Any]] = []
    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            rows.extend(_task_rows(task))
            print(f"[{index}/{len(tasks)}] {task['suite']} seed={task['seed']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {executor.submit(_task_rows, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                rows.extend(future.result())
                print(f"[{index}/{len(tasks)}] {task['suite']} seed={task['seed']}", flush=True)
    rows.sort(key=lambda row: (row["suite"], str(row["x_value"]), int(row["trial_id"]), row["variant"]))
    write_csv(trials_csv, rows, TRIAL_FIELDS)
    summary = summarize_rows(rows)
    paired = paired_summary(
        rows,
        reference_variant=str(args.paired_reference),
        candidate_variant=str(args.paired_candidate),
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.seed) + 1,
    )
    write_csv(summary_csv, summary, SUMMARY_FIELDS)
    write_csv(paired_csv, paired, PAIRED_FIELDS)
    _write_markdown(out_dir / "summary.md", summary, paired)
    _plot_from_summary(summary_csv, out_dir, trials_csv)
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
