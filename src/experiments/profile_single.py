"""Profile one default RIS-EVS-OFDM estimation trial."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

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
    from src.estimators import initialize_from_hankel, refine_global_raw, structured_refinement
    from src.tensor_utils import hankelize_frequency
else:
    from ..channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from ..config import default_config
    from ..estimators import initialize_from_hankel, refine_global_raw, structured_refinement
    from ..tensor_utils import hankelize_frequency


def _time_block(label: str, timings: list[tuple[str, float]], fn):
    start = time.perf_counter()
    result = fn()
    timings.append((label, time.perf_counter() - start))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one default_config trial.")
    parser.add_argument("--enable-global-vp", action="store_true")
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_config()
    if args.enable_global_vp:
        config["enable_global_vp"] = True
    if args.snr_db is not None:
        config["SNR_dB"] = float(args.snr_db)
    if args.seed is not None:
        config["seed"] = int(args.seed)

    timings: list[tuple[str, float]] = []
    rng = np.random.default_rng(config["seed"])

    scene = _time_block("generate_scene", timings, lambda: generate_scene(config, rng))

    def make_noisy_data():
        true_components = channel_components(
            scene,
            scene["p_u_true"],
            scene["delta_t_true"],
            scene["gamma_true"],
            scene["eta_true"],
        )
        y_true = synthesize_raw_tensor(true_components, scene["beta_true"])
        y_noisy, noise_variance = add_awgn(y_true, config["SNR_dB"], rng)
        return true_components, y_true, y_noisy, noise_variance

    true_components, y_true, y_noisy, noise_variance = _time_block(
        "channel_components + synthesize_raw_tensor + add_awgn",
        timings,
        make_noisy_data,
    )
    z_noisy = _time_block(
        "hankelize_frequency",
        timings,
        lambda: hankelize_frequency(y_noisy, scene["P"]),
    )
    estimate_initial = _time_block(
        "initialization",
        timings,
        lambda: initialize_from_hankel(z_noisy, scene, config),
    )
    estimate_stage2, structured_diag = _time_block(
        "structured_refinement",
        timings,
        lambda: structured_refinement(z_noisy, scene, config, estimate_initial),
    )

    if config.get("enable_global_vp", False):
        _time_block(
            "refine_global_raw",
            timings,
            lambda: refine_global_raw(y_noisy, scene, config, estimate_stage2),
        )
    else:
        timings.append(("refine_global_raw", float("nan")))

    print("Profile: one default_config trial")
    print(f"seed = {config['seed']}")
    print(f"SNR_dB = {config['SNR_dB']}")
    print(f"noise_variance = {noise_variance:.6e}")
    print(f"structured_iters_recorded = {len(structured_diag['updates'])}")
    print(f"Y_true_shape = {y_true.shape}")
    print(f"Z_noisy_shape = {z_noisy.shape}")
    for label, elapsed in timings:
        if np.isnan(elapsed):
            print(f"{label}: skipped")
        else:
            print(f"{label}: {elapsed:.6f} s")


if __name__ == "__main__":
    main()
