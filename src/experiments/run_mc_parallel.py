"""Parallel Monte Carlo driver for the proposed RIS-EVS-OFDM estimator."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.channel_model import (
        add_awgn,
        channel_components,
        generate_scene,
        synthesize_raw_tensor,
    )
    from src.config import default_config
    from src.diagnostics import estimate_position_from_ris_eta
    from src.estimators import initialize_from_hankel, refine_global_raw, structured_refinement
    from src.metrics import position_rmse
    from src.projections_delay import tau_from_pole
    from src.tensor_utils import hankelize_frequency, reconstruct_z
else:
    from ..channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from ..config import default_config
    from ..diagnostics import estimate_position_from_ris_eta
    from ..estimators import initialize_from_hankel, refine_global_raw, structured_refinement
    from ..metrics import position_rmse
    from ..projections_delay import tau_from_pole
    from ..tensor_utils import hankelize_frequency, reconstruct_z


FIELDNAMES = [
    "trial_id",
    "seed",
    "snr_db",
    "stage2_position_rmse",
    "stage2_range_rmse",
    "stage2_tau_rmse",
    "initial_z_residual",
    "final_z_sse",
    "runtime_s",
    "vp_position_rmse",
    "failed",
    "error",
]


def _run_trial_worker(args: tuple[int, int, float, bool]) -> dict[str, Any]:
    trial_id, seed, snr_db, enable_global_vp = args
    start = time.perf_counter()
    row: dict[str, Any] = {
        "trial_id": int(trial_id),
        "seed": int(seed),
        "snr_db": float(snr_db),
        "stage2_position_rmse": "",
        "stage2_range_rmse": "",
        "stage2_tau_rmse": "",
        "initial_z_residual": "",
        "final_z_sse": "",
        "runtime_s": "",
        "vp_position_rmse": "",
        "failed": False,
        "error": "",
    }
    try:
        config = default_config()
        config["seed"] = int(seed)
        config["SNR_dB"] = float(snr_db)
        config["enable_global_vp"] = bool(enable_global_vp)

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
        y_noisy, _ = add_awgn(y_true, config["SNR_dB"], rng)
        z_noisy = hankelize_frequency(y_noisy, scene["P"])

        estimate_initial = initialize_from_hankel(z_noisy, scene, config)
        estimate_stage2, _ = structured_refinement(z_noisy, scene, config, estimate_initial)

        p_hat = estimate_position_from_ris_eta(scene, estimate_stage2)
        tau_hat = np.array(
            [tau_from_pole(pole, scene["delta_f"]) for pole in estimate_stage2["poles"]]
        )
        range_hat = estimate_stage2["ris_eta"][:, 0]
        z_hat_stage2 = reconstruct_z(
            estimate_stage2["beta_z"],
            estimate_stage2["A"],
            estimate_stage2["B"],
            estimate_stage2["Q"],
            estimate_stage2["C"],
        )

        row.update(
            {
                "stage2_position_rmse": position_rmse(p_hat, scene["p_u_true"]),
                "stage2_range_rmse": float(
                    np.linalg.norm(range_hat - true_components["ranges"])
                    / np.sqrt(scene["K"])
                ),
                "stage2_tau_rmse": float(
                    np.linalg.norm(tau_hat - true_components["taus"]) / np.sqrt(scene["K"])
                ),
                "initial_z_residual": estimate_initial["initial_z_residual"],
                "final_z_sse": float(np.linalg.norm(z_hat_stage2 - z_noisy) ** 2),
            }
        )
        if enable_global_vp:
            final = refine_global_raw(y_noisy, scene, config, estimate_stage2)
            row["vp_position_rmse"] = position_rmse(final["p_u"], scene["p_u_true"])
    except Exception as exc:  # noqa: BLE001 - one failed trial should not stop MC.
        row["failed"] = True
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["runtime_s"] = time.perf_counter() - start
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Monte Carlo trials in parallel.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--enable-global-vp", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/mc_parallel.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")

    seed_sequence = np.random.SeedSequence()
    child_sequences = seed_sequence.spawn(args.n_trials)
    seeds = [int(seq.generate_state(1, dtype=np.uint32)[0]) for seq in child_sequences]
    tasks = [
        (trial_id, seeds[trial_id], float(args.snr_db), bool(args.enable_global_vp))
        for trial_id in range(args.n_trials)
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = [executor.submit(_run_trial_worker, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["trial_id"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    failed = sum(bool(row["failed"]) for row in rows)
    print(f"Wrote {args.out}")
    print(f"trials = {len(rows)}, failed = {failed}")


if __name__ == "__main__":
    main()
