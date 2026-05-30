"""Run Stage-II module and guarded-mode ablations."""

from __future__ import annotations

import argparse
import copy
import csv
import pathlib
import sys
import time
import traceback
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from src.config import default_config
    from src.estimators import (
        estimate_position_from_local_ris,
        initialize_from_hankel,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from src.projections_delay import tau_from_pole
    from src.tensor_utils import hankelize_frequency
else:
    from ..channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from ..config import default_config
    from ..estimators import (
        estimate_position_from_local_ris,
        initialize_from_hankel,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from ..projections_delay import tau_from_pole
    from ..tensor_utils import hankelize_frequency


FIELDNAMES = [
    "trial_id",
    "seed",
    "variant",
    "snr_db",
    "failed",
    "error",
    "y_nmse",
    "z_rmse_noisy",
    "position_rmse_m",
    "range_rmse_m",
    "tau_rmse_s",
    "num_iterations_run",
    "num_iteration_accepted",
    "num_evs_accepted",
    "num_delay_accepted",
    "num_ris_accepted",
    "mean_ris_local_improvement",
    "mean_ris_best_rho",
    "runtime_s",
    "vp_position_rmse_m",
    "vp_y_nmse",
]


def _variant_specs(enable_vp: bool) -> list[tuple[str, dict]]:
    specs = [
        (
            "stage1_only",
            {
                "stage2_enable_evs": False,
                "stage2_enable_delay": False,
                "stage2_enable_ris": False,
                "num_structured_iters": 0,
                "enable_global_vp": False,
            },
        ),
        (
            "evs_only",
            {
                "stage2_enable_evs": True,
                "stage2_enable_delay": False,
                "stage2_enable_ris": False,
                "stage2_guarded": False,
                "enable_global_vp": False,
            },
        ),
        (
            "delay_only",
            {
                "stage2_enable_evs": False,
                "stage2_enable_delay": True,
                "stage2_enable_ris": False,
                "stage2_guarded": False,
                "enable_global_vp": False,
            },
        ),
        (
            "ris_only",
            {
                "stage2_enable_evs": False,
                "stage2_enable_delay": False,
                "stage2_enable_ris": True,
                "stage2_guarded": False,
                "enable_global_vp": False,
            },
        ),
        (
            "current_full_stage2",
            {
                "stage2_enable_evs": True,
                "stage2_enable_delay": True,
                "stage2_enable_ris": True,
                "stage2_guarded": False,
                "enable_global_vp": False,
            },
        ),
        (
            "guarded_full_stage2",
            {
                "stage2_enable_evs": True,
                "stage2_enable_delay": True,
                "stage2_enable_ris": True,
                "stage2_guarded": True,
                "stage2_strict_accept_rel": 1.0e-6,
                "ris_min_relative_improvement": 5.0e-3,
                "enable_global_vp": False,
            },
        ),
    ]
    if enable_vp:
        specs.append(
            (
                "guarded_full_stage2_vp",
                {
                    "stage2_enable_evs": True,
                    "stage2_enable_delay": True,
                    "stage2_enable_ris": True,
                    "stage2_guarded": True,
                    "stage2_strict_accept_rel": 1.0e-6,
                    "ris_min_relative_improvement": 5.0e-3,
                    "enable_global_vp": True,
                },
            )
        )
    return specs


def _nmse(x_hat: np.ndarray, x_ref: np.ndarray, eps: float = 1.0e-12) -> float:
    return float(np.linalg.norm(x_hat - x_ref) ** 2 / (np.linalg.norm(x_ref) ** 2 + eps))


def _rmse(x_hat: np.ndarray, x_ref: np.ndarray) -> float:
    diff = np.asarray(x_hat, dtype=float) - np.asarray(x_ref, dtype=float)
    return float(np.linalg.norm(diff) / np.sqrt(diff.size))


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def _trial_seed(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _make_data(config: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    scene = generate_scene(config, rng)
    true_components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(true_components, scene["beta_true"])
    y_noisy, noise_variance = add_awgn(y_true, config["SNR_dB"], rng)
    z_noisy = hankelize_frequency(y_noisy, scene["P"])
    return {
        "scene": scene,
        "true_components": true_components,
        "Y_true": y_true,
        "Y_noisy": y_noisy,
        "Z_noisy": z_noisy,
        "noise_variance": noise_variance,
    }


def _diagnostic_counts(diagnostics: dict) -> dict:
    updates = diagnostics.get("updates", [])
    evs_accepted = 0
    delay_accepted = 0
    ris_accepted = 0
    ris_improvements = []
    ris_best_rhos = []
    for update in updates:
        evs_accepted += sum(
            bool(item.get("accepted", False))
            for item in update.get("evs_projection_details", [])
        )
        delay_details = update.get("delay_projection_details", {})
        delay_accepted += int(bool(delay_details.get("accepted", False)))
        for item in update.get("ris_projection_details", []):
            ris_accepted += int(bool(item.get("accepted", False)))
            ris_improvements.append(float(item.get("local_relative_improvement", np.nan)))
            ris_best_rhos.append(float(item.get("best_rho", np.nan)))
    return {
        "num_iterations_run": len(updates),
        "num_iteration_accepted": sum(bool(item.get("iteration_accepted", False)) for item in updates),
        "num_evs_accepted": evs_accepted,
        "num_delay_accepted": delay_accepted,
        "num_ris_accepted": ris_accepted,
        "mean_ris_local_improvement": _finite_mean(ris_improvements),
        "mean_ris_best_rho": _finite_mean(ris_best_rhos),
    }


def _z_rmse_noisy(estimate: dict, diagnostics: dict, z_noisy: np.ndarray) -> float:
    residuals = diagnostics.get("residuals_noisy_rmse", [])
    if residuals:
        return float(residuals[-1])
    z_hat = estimate.get("Z_hat")
    if z_hat is None:
        return float("nan")
    return float(np.linalg.norm(z_hat - z_noisy) / np.sqrt(z_noisy.size))


def _failed_row(trial_id: int, seed: int, variant: str, snr_db: float, error: BaseException) -> dict:
    row = {field: float("nan") for field in FIELDNAMES}
    row.update(
        {
            "trial_id": trial_id,
            "seed": seed,
            "variant": variant,
            "snr_db": snr_db,
            "failed": True,
            "error": f"{type(error).__name__}: {error}".replace("\n", " | ")[:2000],
        }
    )
    return row


def _evaluate_variant(
    *,
    trial_id: int,
    seed: int,
    variant: str,
    variant_updates: dict,
    base_config: dict,
    data: dict,
    estimate_initial: dict,
    enable_vp: bool,
) -> dict:
    config = copy.deepcopy(base_config)
    config.update(variant_updates)
    start = time.perf_counter()
    estimate, diagnostics = structured_refinement(
        data["Z_noisy"],
        data["scene"],
        config,
        copy.deepcopy(estimate_initial),
    )
    y_hat = reconstruct_raw_tensor_from_structured_estimate(estimate, data["scene"])
    p_hat = estimate_position_from_local_ris(data["scene"], estimate, config)
    tau_hat = np.array([tau_from_pole(pole, data["scene"]["delta_f"]) for pole in estimate["poles"]])
    counts = _diagnostic_counts(diagnostics)
    vp_position = float("nan")
    vp_y_nmse = float("nan")
    if enable_vp and variant == "guarded_full_stage2_vp":
        vp = refine_global_raw(data["Y_noisy"], data["scene"], config, copy.deepcopy(estimate))
        vp_position = float(np.linalg.norm(vp["p_u"] - data["scene"]["p_u_true"]))
        vp_y_nmse = _nmse(vp["Y_hat"], data["Y_true"])

    runtime = time.perf_counter() - start
    row = {
        "trial_id": trial_id,
        "seed": seed,
        "variant": variant,
        "snr_db": float(config["SNR_dB"]),
        "failed": False,
        "error": "",
        "y_nmse": _nmse(y_hat, data["Y_true"]),
        "z_rmse_noisy": _z_rmse_noisy(estimate, diagnostics, data["Z_noisy"]),
        "position_rmse_m": float(np.linalg.norm(p_hat - data["scene"]["p_u_true"])),
        "range_rmse_m": _rmse(estimate["ris_eta"][:, 0], data["true_components"]["ranges"]),
        "tau_rmse_s": _rmse(tau_hat, data["true_components"]["taus"]),
        "runtime_s": float(runtime),
        "vp_position_rmse_m": vp_position,
        "vp_y_nmse": vp_y_nmse,
    }
    row.update(counts)
    return row


def _run_trial(trial_id: int, seed: int, args: argparse.Namespace) -> list[dict]:
    base_config = default_config()
    base_config["seed"] = seed
    base_config["SNR_dB"] = float(args.snr_db)
    base_config["enable_global_vp"] = False
    data = _make_data(base_config, seed)
    estimate_initial = initialize_from_hankel(data["Z_noisy"], data["scene"], base_config)
    rows = []
    for variant, updates in _variant_specs(args.enable_vp):
        try:
            rows.append(
                _evaluate_variant(
                    trial_id=trial_id,
                    seed=seed,
                    variant=variant,
                    variant_updates=updates,
                    base_config=base_config,
                    data=data,
                    estimate_initial=estimate_initial,
                    enable_vp=args.enable_vp,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one variant should not stop the ablation.
            rows.append(_failed_row(trial_id, seed, variant, args.snr_db, RuntimeError(traceback.format_exc(limit=8))))
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        pd.DataFrame(rows, columns=FIELDNAMES).to_csv(path, index=False)
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict]) -> None:
    print("\nGrouped summary")
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        frame = pd.DataFrame(rows)
        frame = frame[~frame["failed"].astype(bool)]
        if frame.empty:
            print("No successful rows.")
            return
        summary = frame.groupby("variant").agg(
            median_position_rmse_m=("position_rmse_m", "median"),
            p90_position_rmse_m=("position_rmse_m", lambda x: float(np.percentile(x, 90.0))),
            median_y_nmse=("y_nmse", "median"),
            median_range_rmse_m=("range_rmse_m", "median"),
            median_tau_rmse_s=("tau_rmse_s", "median"),
            mean_num_ris_accepted=("num_ris_accepted", "mean"),
        )
        print(summary.to_string())
        return

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("failed"):
            continue
        by_variant.setdefault(str(row["variant"]), []).append(row)
    if not by_variant:
        print("No successful rows.")
        return
    header = (
        "variant, median_position_rmse_m, p90_position_rmse_m, median_y_nmse, "
        "median_range_rmse_m, median_tau_rmse_s, mean_num_ris_accepted"
    )
    print(header)
    for variant in sorted(by_variant):
        group = by_variant[variant]
        pos = np.asarray([row["position_rmse_m"] for row in group], dtype=float)
        y_nmse = np.asarray([row["y_nmse"] for row in group], dtype=float)
        ranges = np.asarray([row["range_rmse_m"] for row in group], dtype=float)
        taus = np.asarray([row["tau_rmse_s"] for row in group], dtype=float)
        ris_count = np.asarray([row["num_ris_accepted"] for row in group], dtype=float)
        print(
            f"{variant}, {np.median(pos):.6e}, {np.percentile(pos, 90.0):.6e}, "
            f"{np.median(y_nmse):.6e}, {np.median(ranges):.6e}, "
            f"{np.median(taus):.6e}, {np.mean(ris_count):.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-II ablation and guarded-mode experiment.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--enable-vp", action="store_true", default=False)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/stage2_ablation.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    seed_sequence = np.random.SeedSequence(args.seed)
    child_sequences = seed_sequence.spawn(args.n_trials)
    rows = []
    for trial_id, child in enumerate(child_sequences):
        seed = _trial_seed(child)
        print(f"Running trial {trial_id + 1}/{args.n_trials} seed={seed}", flush=True)
        rows.extend(_run_trial(trial_id, seed, args))
    _write_csv(args.out, rows)
    print(f"\nWrote {args.out}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
