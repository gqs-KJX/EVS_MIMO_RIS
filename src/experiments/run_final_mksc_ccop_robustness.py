"""Robustness, geometry-generalization, and scaling experiments for MKSC-GI--CCOP.

The script covers the experiments not present in the legacy robustness runner:
Maxwell-subspace mismatch, colored-noise boundary tests, multi-position maps,
and N/M_A/K/RIS-size scaling.  Existing Fig.8--Fig.11 experiments can include
the frozen route via ``--baselines mksc_ccop,...`` in
``run_robustness_and_scaling_figures``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pathlib
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..ccop_stage1_initializer import known_evs_union_basis
from ..channel_model import channel_components, generate_scene, synthesize_raw_tensor
from ..geometry import maxwell_matrix, ula_steering
from ..tensor_utils import hankelize_frequency
from ..validation_artifacts import validation_environment
from .final_mksc_ccop_common import (
    PAPER_VARIANTS,
    SUMMARY_FIELDS,
    TRIAL_FIELDS,
    make_paper_config,
    make_shared_data,
    run_paired_variants,
    save_run_manifest,
    save_resolved_config,
    summarize_rows,
    write_csv,
)
from .resource_control import apply_thread_limits
from .run_paper_ablation_figures import _peb_from_efim, set_number_of_ris_paths
from .run_robustness_and_scaling_figures import (
    _data_from_scenes,
)


ROBUST_FIELDS = TRIAL_FIELDS + (
    "scenario",
    "mismatch_type",
    "mismatch_value",
    "subspace_leakage",
    "colored_noise_rho",
    "noise_handling",
    "position_index",
)
POSITION_SUMMARY_FIELDS = (
    "position_index",
    "p_true_x",
    "p_true_y",
    "p_true_z",
) + SUMMARY_FIELDS

DEFAULT_SUITES = "subspace_mismatch"
DEFAULT_VARIANTS = ("raw_delay_gi_ccop", "proposed")
DEFAULT_POSITION_VARIANTS = ("scaled_4d", "proposed")
DEFAULT_SCALING_VARIANTS = ("scaled_4d", "proposed")


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _int_grid(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _positions(value: str) -> list[np.ndarray]:
    result = []
    for item in value.split(";"):
        if not item.strip():
            continue
        coordinate = np.asarray([float(v) for v in item.split(":")], dtype=float)
        if coordinate.shape != (3,):
            raise ValueError("each position must be x:y:z")
        result.append(coordinate)
    return result


def _trial_seeds(root_seed: int, n_trials: int) -> list[int]:
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(int(root_seed)).spawn(int(n_trials))
    ]


def _subspace_leakage(y_true: np.ndarray, nominal_scene: dict) -> float:
    basis, _ = known_evs_union_basis(nominal_scene, relative_tolerance=1.0e-12)
    compressed = np.einsum("ir,int->rnt", basis.conj(), y_true, optimize=True)
    energy = float(np.linalg.norm(y_true) ** 2)
    return float(
        max(0.0, 1.0 - np.linalg.norm(compressed) ** 2 / max(energy, 1.0e-300))
    )


def _perturbed_unit_direction(
    direction: np.ndarray, std_deg: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply a small isotropic angular calibration error to a unit direction."""
    unit = np.asarray(direction, dtype=float)
    unit /= np.linalg.norm(unit)
    perturbation = rng.normal(0.0, np.deg2rad(float(std_deg)), size=3)
    perturbation -= unit * float(np.dot(unit, perturbation))
    result = unit + perturbation
    return result / np.linalg.norm(result)


def _inject_ris_bs_angle_error(
    scene: dict, std_deg: float, rng: np.random.Generator
) -> dict:
    """Perturb only generation-side calibrated RIS--BS directions."""
    generation = copy.deepcopy(scene)
    wavelength = float(scene["wavelength"])
    wavenumber = 2.0 * np.pi / wavelength
    for path in range(int(scene["K"])):
        nominal = (
            np.asarray(scene["p_B"], dtype=float)
            - np.asarray(scene["ris_centers"][path], dtype=float)
        )
        perturbed = _perturbed_unit_direction(nominal, std_deg, rng)
        direction_local = np.asarray(scene["rotations"][path]) @ perturbed
        generation["a_RB"][path] = np.exp(
            -1j * wavenumber * (np.asarray(scene["ris_grid"]) @ direction_local)
        )
        generation["v_B"][path] = ula_steering(
            int(scene["M_A"]), wavelength / 2.0, wavelength, -perturbed
        )
        generation["Theta"][path] = maxwell_matrix(perturbed)
    return generation


def _inject_bs_sensor_position_error(
    scene: dict, std_mm: float, rng: np.random.Generator
) -> dict:
    """Perturb generation-side BS sensor locations while retaining nominal estimation."""
    generation = copy.deepcopy(scene)
    wavelength = float(scene["wavelength"])
    count = int(scene["M_A"])
    nominal = np.zeros((count, 3), dtype=float)
    nominal[:, 0] = (
        np.arange(count) - (count - 1) / 2.0
    ) * (wavelength / 2.0)
    offsets = rng.normal(0.0, float(std_mm) * 1.0e-3, size=nominal.shape)
    positions = nominal + offsets
    wavenumber = 2.0 * np.pi / wavelength
    for path in range(int(scene["K"])):
        arrival = (
            np.asarray(scene["ris_centers"][path], dtype=float)
            - np.asarray(scene["p_B"], dtype=float)
        )
        arrival /= np.linalg.norm(arrival)
        generation["v_B"][path] = np.exp(
            -1j * wavenumber * (positions @ arrival)
        )
    return generation


def _mismatch_data(config: dict, mismatch_type: str, value: float) -> tuple[dict, float]:
    sequence = np.random.SeedSequence(int(config["seed"]))
    scene_seed, mismatch_seed, noise_seed = sequence.spawn(3)
    nominal = generate_scene(config, np.random.default_rng(scene_seed))
    rng = np.random.default_rng(mismatch_seed)
    generation = copy.deepcopy(nominal)
    if mismatch_type == "evs_phase_deg":
        phase = rng.normal(0.0, np.deg2rad(float(value)), size=(nominal["K"], 6, 1))
        generation["Theta"] = nominal["Theta"] * np.exp(1j * phase)
    elif mismatch_type == "evs_gain_std":
        sigma = float(value)
        gain = np.exp(sigma * rng.standard_normal((nominal["K"], 6, 1)) - 0.5 * sigma**2)
        generation["Theta"] = nominal["Theta"] * gain
    elif mismatch_type == "ris_bs_angle_deg":
        generation = _inject_ris_bs_angle_error(nominal, value, rng)
    elif mismatch_type == "bs_sensor_position_std_mm":
        generation = _inject_bs_sensor_position_error(nominal, value, rng)
    else:
        raise ValueError(f"unknown mismatch type {mismatch_type!r}")
    data = _data_from_scenes(
        nominal, generation, config, np.random.default_rng(noise_seed)
    )
    return data, _subspace_leakage(data["Y_true"], nominal)


def _colored_noise_data(config: dict, rho: float) -> dict:
    """Generate EVS-correlated noise at the requested observed-tensor SNR."""
    sequence = np.random.SeedSequence(int(config["seed"]))
    scene_seed, noise_seed = sequence.spawn(2)
    scene = generate_scene(config, np.random.default_rng(scene_seed))
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(components, scene["beta_true"])
    i_dim = int(scene["I"])
    indices = np.arange(i_dim)
    covariance = float(rho) ** np.abs(indices[:, None] - indices[None, :])
    chol = np.linalg.cholesky(covariance + 1.0e-12 * np.eye(i_dim))
    rng = np.random.default_rng(noise_seed)
    white = (
        rng.standard_normal(y_true.shape) + 1j * rng.standard_normal(y_true.shape)
    ) / np.sqrt(2.0)
    colored = np.einsum("ij,jnt->int", chol, white, optimize=True)
    active = np.asarray(scene.get("evs_observation_mask", np.ones(i_dim)), dtype=bool)
    colored[~active, :, :] = 0.0
    signal_power = float(np.mean(np.abs(y_true[active, :, :]) ** 2))
    target_noise_power = signal_power / (10.0 ** (float(config["SNR_dB"]) / 10.0))
    current_noise_power = float(np.mean(np.abs(colored[active, :, :]) ** 2))
    colored *= np.sqrt(target_noise_power / max(current_noise_power, 1.0e-300))
    y_noisy = y_true + colored
    return {
        "scene": scene,
        "true_components": components,
        "Y_true": y_true,
        "Y_noisy": y_noisy,
        "Z_true": hankelize_frequency(y_true, scene["P"]),
        "Z_noisy": hankelize_frequency(y_noisy, scene["P"]),
        "noise_variance": target_noise_power,
        "timing": {},
    }


def _task_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    apply_thread_limits(int(task["blas_threads"]))
    overrides = dict(task.get("overrides", {}))
    config = make_paper_config(
        int(task["seed"]),
        float(task["snr_db"]),
        diagnostic_mode=str(task["diagnostic_mode"]),
        overrides=overrides,
    )
    if "K" in task:
        set_number_of_ris_paths(config, int(task["K"]))
    scenario = str(task["scenario"])
    leakage = 0.0
    if scenario == "subspace_mismatch":
        data, leakage = _mismatch_data(
            config, str(task["mismatch_type"]), float(task["mismatch_value"])
        )
    elif scenario == "colored_noise":
        data = _colored_noise_data(config, float(task["colored_noise_rho"]))
    else:
        data = make_shared_data(config)
    config["noise_variance"] = float(data["noise_variance"])
    rows = run_paired_variants(
        task["variants"],
        data=data,
        config=config,
        suite=scenario,
        x_name=str(task["x_name"]),
        x_value=task["x_value"],
        trial_id=int(task["trial_id"]),
        outlier_threshold_m=float(task["outlier_threshold_m"]),
        profile_memory=bool(task.get("profile_memory", False)),
    )
    geometry_peb = float("nan")
    if scenario == "positions" and bool(task.get("position_peb", False)):
        try:
            geometry_peb = float(
                _peb_from_efim(data, config).get("peb_free_jones_m", np.nan)
            )
        except (KeyError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            geometry_peb = float("nan")
    for row in rows:
        row.update(
            {
                "scenario": scenario,
                "mismatch_type": str(task.get("mismatch_type", "")),
                "mismatch_value": task.get("mismatch_value", ""),
                "subspace_leakage": float(leakage),
                "colored_noise_rho": task.get("colored_noise_rho", ""),
                "noise_handling": (
                    "unwhitened_boundary_test" if scenario == "colored_noise" else "matched_white"
                ),
                "position_index": task.get("position_index", ""),
                "peb_position_m": geometry_peb,
                "reference_type": (
                    "matched_data_only_free_jones_geometry_diagnostic"
                    if np.isfinite(geometry_peb)
                    else ""
                ),
            }
        )
    return rows


def _tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = _trial_seeds(args.seed, args.n_trials)
    tasks: list[dict[str, Any]] = []

    def append(
        scenario: str,
        x_name: str,
        x_value: Any,
        *,
        overrides: dict | None = None,
        variants: list[str] | tuple[str, ...] | None = None,
        **extra,
    ) -> None:
        for trial_id, seed in enumerate(seeds):
            tasks.append(
                {
                    "scenario": scenario,
                    "x_name": x_name,
                    "x_value": x_value,
                    "trial_id": trial_id,
                    "seed": seed,
                    "snr_db": args.snr_db,
                    "variants": list(variants or args.variants),
                    "overrides": dict(overrides or {}),
                    "diagnostic_mode": args.diagnostic_mode,
                    "outlier_threshold_m": args.outlier_threshold_m,
                    "blas_threads": args.blas_threads,
                    "profile_memory": bool(args.profile_memory),
                    **extra,
                }
            )

    suites = set(args.suites)
    if "subspace_mismatch" in suites:
        for mismatch_type, values in (
            ("evs_phase_deg", args.phase_grid),
            ("evs_gain_std", args.gain_grid),
            ("ris_bs_angle_deg", args.ris_bs_angle_grid),
            ("bs_sensor_position_std_mm", args.bs_sensor_position_mm_grid),
        ):
            for value in values:
                append(
                    "subspace_mismatch",
                    mismatch_type,
                    value,
                    mismatch_type=mismatch_type,
                    mismatch_value=value,
                    variants=args.mismatch_variants,
                )
    if "colored_noise" in suites:
        for rho in args.colored_noise_grid:
            append(
                "colored_noise",
                "colored_noise_rho",
                rho,
                colored_noise_rho=rho,
                variants=args.colored_noise_variants,
            )
    if "positions" in suites:
        reference = make_paper_config(0, args.snr_db)
        bounds = np.asarray(reference["ue_bounds"], dtype=float)
        positions = list(args.positions)
        if args.position_grid_shape:
            shape = _int_grid(args.position_grid_shape)
            if len(shape) != 3 or any(size < 1 for size in shape):
                raise ValueError("--position-grid-shape must contain three positive integers")
            margin = float(args.position_grid_margin_m)
            axes = [
                np.linspace(bounds[dim, 0] + margin, bounds[dim, 1] - margin, shape[dim])
                for dim in range(3)
            ]
            positions = [
                np.array([x, y, z], dtype=float)
                for x in axes[0]
                for y in axes[1]
                for z in axes[2]
            ]
        for index, position in enumerate(positions):
            if not np.all((position >= bounds[:, 0]) & (position <= bounds[:, 1])):
                raise ValueError(f"position {position.tolist()} lies outside ue_bounds")
            append(
                "positions",
                "position_index",
                index,
                overrides={"p_u_true": position},
                position_index=index,
                position_peb=bool(args.position_peb),
                variants=args.position_variants,
            )
    if "scaling" in suites:
        for n_dim in args.n_grid:
            append(
                "scaling",
                "N",
                n_dim,
                overrides={"N": n_dim, "P": (n_dim + 1) // 2},
                variants=args.scaling_variants,
            )
        for t_dim in args.training_grid:
            append(
                "scaling",
                "T",
                t_dim,
                overrides={"T": t_dim},
                variants=args.scaling_variants,
            )
        for m_a in args.array_grid:
            append(
                "scaling",
                "M_A",
                m_a,
                overrides={"M_A": m_a},
                variants=args.scaling_variants,
            )
        for ris_side in args.ris_side_grid:
            append(
                "scaling",
                "M_R",
                ris_side * ris_side,
                overrides={"ris_shape": (ris_side, ris_side)},
                variants=args.scaling_variants,
            )
        for k_paths in args.k_grid:
            append(
                "scaling",
                "K",
                k_paths,
                K=k_paths,
                variants=args.scaling_variants,
            )
    return tasks


def _plot(
    summary_csv: pathlib.Path,
    out_dir: pathlib.Path,
    trials_csv: pathlib.Path | None = None,
) -> None:
    mpl_dir = out_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for scenario in sorted({row["suite"] for row in rows}):
        scenario_rows = [row for row in rows if row["suite"] == scenario]
        for x_name in sorted({row["x_name"] for row in scenario_rows}):
            selected = [row for row in scenario_rows if row["x_name"] == x_name]
            variants = list(dict.fromkeys(row["variant"] for row in selected))
            x_values = sorted({float(row["x_value"]) for row in selected})
            fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
            for axis, metric, label, log_y in (
                (axes[0, 0], "position_rmse_m", "Position RMSE (m)", True),
                (axes[0, 1], "catastrophic_rate", "Outlier / failure probability", False),
                (axes[1, 0], "clock_p95_ns", "Clock p95 (ns)", True),
                (axes[1, 1], "channel_nmse_p95", "Channel NMSE p95", True),
            ):
                for variant in variants:
                    by_x = {
                        float(row["x_value"]): row
                        for row in selected
                        if row["variant"] == variant
                    }
                    axis.plot(
                        x_values,
                        [float(by_x[value][metric]) for value in x_values],
                        marker="o",
                        label=variant,
                    )
                if log_y:
                    axis.set_yscale("log")
                axis.set_xlabel(x_name)
                axis.set_ylabel(label)
                axis.grid(True, which="both", alpha=0.3)
            axes[0, 0].legend(fontsize=7)
            fig.savefig(out_dir / f"robustness_{scenario}_{x_name}.pdf", bbox_inches="tight")
            plt.close(fig)
    if trials_csv is None or not trials_csv.exists():
        return
    with trials_csv.open(newline="", encoding="utf-8") as handle:
        trials = [
            row
            for row in csv.DictReader(handle)
            if row["suite"] == "positions"
            and row["variant"] in {"scaled_4d", "proposed"}
            and row["failed"].lower() != "true"
        ]
    grouped: dict[
        tuple[float, float, float], dict[str, list[dict[str, str]]]
    ] = {}
    for row in trials:
        coordinate = (
            float(row["p_true_x"]),
            float(row["p_true_y"]),
            float(row["p_true_z"]),
        )
        grouped.setdefault(coordinate, {}).setdefault(row["variant"], []).append(row)
    for z_value in sorted({coordinate[2] for coordinate in grouped}):
        layer = [
            (coordinate, by_variant)
            for coordinate, by_variant in grouped.items()
            if np.isclose(coordinate[2], z_value) and "proposed" in by_variant
        ]
        if not layer:
            continue
        x = np.asarray([item[0][0] for item in layer])
        y = np.asarray([item[0][1] for item in layer])
        p95 = np.asarray(
            [
                np.percentile(
                    [
                        float(row["position_error_m"])
                        for row in by_variant["proposed"]
                    ],
                    95,
                )
                for _, by_variant in layer
            ]
        )
        outlier = np.asarray(
            [
                np.mean(
                    [
                        row["outlier"].lower() == "true"
                        for row in by_variant["proposed"]
                    ]
                )
                for _, by_variant in layer
            ]
        )
        delta_p95 = np.asarray(
            [
                (
                    np.percentile(
                        [
                            float(row["position_error_m"])
                            for row in by_variant["proposed"]
                        ],
                        95,
                    )
                    - np.percentile(
                        [
                            float(row["position_error_m"])
                            for row in by_variant.get("scaled_4d", [])
                        ],
                        95,
                    )
                )
                if by_variant.get("scaled_4d")
                else np.nan
                for _, by_variant in layer
            ]
        )
        peb = np.asarray(
            [
                np.nanmedian(
                    [float(row.get("peb_position_m", "nan")) for row in by_variant["proposed"]]
                )
                for _, by_variant in layer
            ]
        )
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.8), constrained_layout=True)
        first = axes[0, 0].scatter(x, y, c=p95, s=110, cmap="viridis")
        second = axes[0, 1].scatter(
            x, y, c=outlier, s=110, cmap="magma", vmin=0.0, vmax=1.0
        )
        third = axes[1, 0].scatter(x, y, c=delta_p95, s=110, cmap="coolwarm")
        axes[1, 1].scatter(peb, p95, c=outlier, s=70, cmap="magma")
        fig.colorbar(first, ax=axes[0, 0], label="Proposed position p95 (m)")
        fig.colorbar(second, ax=axes[0, 1], label="Proposed catastrophic rate")
        fig.colorbar(third, ax=axes[1, 0], label="Proposed - scaled 4-D p95 (m)")
        axes[1, 1].set_xlabel("Free-Jones geometry PEB (m)")
        axes[1, 1].set_ylabel("Proposed position p95 (m)")
        for axis in axes.flat[:3]:
            axis.set_xlabel("UE x (m)")
            axis.set_ylabel("UE y (m)")
            axis.grid(True, alpha=0.25)
        axes[1, 1].grid(True, alpha=0.25)
        fig.suptitle(f"Frozen proposed route, UE z={z_value:.3f} m")
        fig.savefig(
            out_dir / f"robustness_positions_heatmap_z{z_value:.3f}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def _position_summary(
    rows: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach physical coordinates to per-position statistical summaries."""
    coordinate_by_index: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        if str(row.get("suite")) != "positions":
            continue
        coordinate_by_index[str(row["x_value"])] = (
            float(row["p_true_x"]),
            float(row["p_true_y"]),
            float(row["p_true_z"]),
        )
    result = []
    for row in summary:
        if str(row.get("suite")) != "positions":
            continue
        coordinate = coordinate_by_index[str(row["x_value"])]
        result.append(
            {
                "position_index": row["x_value"],
                "p_true_x": coordinate[0],
                "p_true_y": coordinate[1],
                "p_true_z": coordinate[2],
                **row,
            }
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", default=DEFAULT_SUITES)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--mismatch-variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument(
        "--colored-noise-variants", default=",".join(DEFAULT_VARIANTS)
    )
    parser.add_argument(
        "--position-variants", default=",".join(DEFAULT_POSITION_VARIANTS)
    )
    parser.add_argument(
        "--scaling-variants", default=",".join(DEFAULT_SCALING_VARIANTS)
    )
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--phase-grid", default="0")
    parser.add_argument("--gain-grid", default="0")
    parser.add_argument("--ris-bs-angle-grid", default="0")
    parser.add_argument("--bs-sensor-position-mm-grid", default="0")
    parser.add_argument("--colored-noise-grid", default="0,0.2,0.5,0.8")
    parser.add_argument("--n-grid", default="31,47,63,95")
    parser.add_argument("--training-grid", default="32,64,128,256")
    parser.add_argument("--array-grid", default="4,8,16,24")
    parser.add_argument("--ris-side-grid", default="16,32,48,64")
    parser.add_argument("--k-grid", default="2,3,4")
    parser.add_argument(
        "--positions",
        default=(
            "0.45:-1.1:0.45;0.45:0.05:0.9;0.45:1.2:1.35;"
            "1.5:-1.1:0.9;1.5:0.05:0.45;1.5:1.2:1.35;"
            "2.55:-1.1:1.35;2.55:0.05:0.9;2.55:1.2:0.45"
        ),
    )
    parser.add_argument(
        "--position-grid-shape",
        default="",
        help="optional nx,ny,nz grid inside ue_bounds; overrides --positions",
    )
    parser.add_argument("--position-grid-margin-m", type=float, default=0.1)
    parser.add_argument(
        "--position-peb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute the matched free-Jones geometry PEB for each position",
    )
    parser.add_argument("--diagnostic-mode", choices=("fast", "performance"), default="performance")
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
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
        default=pathlib.Path("results/final_mksc_ccop/robustness_smoke"),
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    args.suites = _csv_list(args.suites)
    args.variants = _csv_list(args.variants)
    args.mismatch_variants = _csv_list(args.mismatch_variants)
    args.colored_noise_variants = _csv_list(args.colored_noise_variants)
    args.position_variants = _csv_list(args.position_variants)
    args.scaling_variants = _csv_list(args.scaling_variants)
    args.phase_grid = _float_grid(args.phase_grid)
    args.gain_grid = _float_grid(args.gain_grid)
    args.ris_bs_angle_grid = _float_grid(args.ris_bs_angle_grid)
    args.bs_sensor_position_mm_grid = _float_grid(
        args.bs_sensor_position_mm_grid
    )
    args.colored_noise_grid = _float_grid(args.colored_noise_grid)
    args.n_grid = _int_grid(args.n_grid)
    args.training_grid = _int_grid(args.training_grid)
    args.array_grid = _int_grid(args.array_grid)
    args.ris_side_grid = _int_grid(args.ris_side_grid)
    args.k_grid = _int_grid(args.k_grid)
    args.positions = _positions(args.positions)
    unknown = set(args.suites) - {
        "subspace_mismatch", "colored_noise", "positions", "scaling"
    }
    if unknown:
        parser.error(f"unknown suites: {sorted(unknown)}")
    unknown_variants = (
        set(args.variants)
        | set(args.mismatch_variants)
        | set(args.colored_noise_variants)
        | set(args.position_variants)
        | set(args.scaling_variants)
    ) - set(PAPER_VARIANTS)
    if unknown_variants:
        parser.error(f"unknown variants: {sorted(unknown_variants)}")
    if args.n_trials < 1 or args.jobs < 1:
        parser.error("--n-trials and --jobs must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.out_dir
    trials_csv = out_dir / "robustness_trials.csv"
    summary_csv = out_dir / "robustness_summary.csv"
    if args.plot_only:
        _plot(summary_csv, out_dir, trials_csv)
        return
    if trials_csv.exists() and not args.force_rerun:
        raise FileExistsError(f"{trials_csv} exists; choose a new directory or use --force-rerun")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, "-m", __spec__.name, *(argv or sys.argv[1:])])
    environment = validation_environment(
        command, repo_root=pathlib.Path(__file__).resolve().parents[2]
    )
    save_run_manifest(out_dir, command=command, arguments=vars(args), environment=environment)
    first_seed = _trial_seeds(args.seed, args.n_trials)[0]
    save_resolved_config(
        out_dir,
        make_paper_config(
            first_seed,
            args.snr_db,
            diagnostic_mode=args.diagnostic_mode,
        ),
        "resolved_base_config.json",
    )
    tasks = _tasks(args)
    rows: list[dict[str, Any]] = []
    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            rows.extend(_task_rows(task))
            print(f"[{index}/{len(tasks)}] {task['scenario']} {task['x_name']}={task['x_value']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as executor:
            futures = {executor.submit(_task_rows, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                rows.extend(future.result())
                print(f"[{index}/{len(tasks)}] {task['scenario']} {task['x_name']}={task['x_value']}", flush=True)
    rows.sort(key=lambda row: (row["suite"], row["x_name"], float(row["x_value"]), int(row["trial_id"]), row["variant"]))
    write_csv(trials_csv, rows, ROBUST_FIELDS)
    summary = summarize_rows(rows)
    write_csv(summary_csv, summary, SUMMARY_FIELDS)
    position_summary = _position_summary(rows, summary)
    if position_summary:
        write_csv(
            out_dir / "position_generalization_summary.csv",
            position_summary,
            POSITION_SUMMARY_FIELDS,
        )
    _plot(summary_csv, out_dir, trials_csv)
    (out_dir / "summary.md").write_text(
        "# Final robustness summary\n\n"
        "The CSV files are authoritative. Colored-noise results are an unwhitened model-boundary test; they are not evidence for a prewhitened estimator.\n\n"
        + "\n".join(
            f"- {row['suite']} / {row['x_name']}={row['x_value']} / {row['variant']}: "
            f"RMSE={row['position_rmse_m']:.6g} m, p95={row['position_p95_m']:.6g} m, "
            f"outlier={row['outlier_rate']:.3%}, channel NMSE p95={row['channel_nmse_p95']:.6g}"
            for row in summary
        )
        + "\n",
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
