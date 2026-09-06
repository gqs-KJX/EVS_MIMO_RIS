#!/usr/bin/env python3
"""Recompute the retained-polarization weight alpha of Theorem C1(i).

    alpha_{S_mode,k} = ||S_mode Theta_k x_k||^2 / ||Theta_k x_k||^2

Provenance for the alpha values quoted in the main-paper C1 results.

Why no observation is synthesized
---------------------------------
alpha is a noiseless, per-panel ratio.  The complex gain beta_k scales the
numerator and the denominator alike and cancels, and the resolvability sweep
fixes the Jones states of the two controlled paths analytically from the
overlap: gamma_left = pi/2, gamma_right = arcsin(overlap), eta = 0 (see
``_make_full_resolution_data``).  Only Theta_k, the Jones vectors and the
receiver mask are therefore needed -- no observation and no estimator.

The separation sweep moves the panels, so the geometry is rebuilt for every
delay separation rather than evaluated once at the default geometry.

Usage
-----
    python scripts/recompute_alpha_retained_energy.py \
        --out results/alpha_retained/alpha_retained.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel_model import (  # noqa: E402
    channel_components,
    evs_component_selection,
    generate_scene,
)
from src.experiments.final_mksc_ccop_common import make_paper_config  # noqa: E402
from src.experiments.run_final_evs_resolvability import _closest_pair  # noqa: E402
from src.experiments.run_robustness_and_scaling_figures import (  # noqa: E402
    adjust_config_for_resolvability,
)
from src.geometry import polarization_vector  # noqa: E402

MODES = ("scalar", "dual_pol", "full_6d")


def alpha_for_draw(config: dict, overlap: float, seed: int) -> dict[str, dict[str, float]]:
    scene = generate_scene(config, np.random.default_rng(seed))
    provisional = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    left, right = _closest_pair(np.asarray(provisional["taus"], dtype=float))
    # Theta depends on the propagation geometry only, so it is already correct
    # for this scene; the overlap sets the Jones vectors, not Theta.
    scene["gamma_true"][left], scene["eta_true"][left] = np.pi / 2.0, 0.0
    scene["gamma_true"][right] = float(np.arcsin(np.clip(overlap, 0.0, 1.0)))
    scene["eta_true"][right] = 0.0
    theta = np.asarray(scene["Theta"], dtype=complex)
    out: dict[str, dict[str, float]] = {}
    for mode in MODES:
        mask = np.asarray(evs_component_selection(mode), dtype=float)
        entry = {}
        for tag, k in (("reference", left), ("competing", right)):
            x = polarization_vector(scene["gamma_true"][k], scene["eta_true"][k])
            v = theta[k] @ x
            denominator = float(np.vdot(v, v).real)
            numerator = float(np.vdot(mask * v, mask * v).real)
            entry[tag] = numerator / denominator if denominator > 0.0 else float("nan")
        out[mode] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-root", type=int, default=20260721)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--snr-db", type=float, default=-10.0)
    ap.add_argument("--separations-ns", type=float, nargs="+",
                    default=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--overlaps", type=float, nargs="+", default=[0.1, 0.5, 0.9, 1.0])
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "results" / "alpha_retained" / "alpha_retained.json")
    args = ap.parse_args()

    seeds = range(args.seed_root, args.seed_root + args.draws)
    header = "".join(f"{m + ' ref':>13s}{m + ' cmp':>13s}" for m in MODES)
    print(f"{'sep[ns]':>8s}{'ovlp':>6s}  " + header)
    records = []
    for separation in args.separations_ns:
        config = adjust_config_for_resolvability(
            make_paper_config(args.seed_root, args.snr_db,
                              overrides={"receiver_mode": "full_6d"}),
            separation,
        )
        for overlap in args.overlaps:
            draws = [alpha_for_draw(config, overlap, s) for s in seeds]
            median = {
                m: {t: statistics.median(d[m][t] for d in draws)
                    for t in ("reference", "competing")}
                for m in MODES
            }
            records.append({"separation_ns": separation, "overlap": overlap,
                            "n_draws": args.draws, "median": median})
            print(f"{separation:8.1f}{overlap:6.1f}  " + "".join(
                f"{median[m]['reference']:13.4f}{median[m]['competing']:13.4f}"
                for m in MODES))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"seed_root": args.seed_root, "draws": args.draws, "snr_db": args.snr_db,
         "definition": "||S_mode Theta_k x_k||^2 / ||Theta_k x_k||^2",
         "records": records}, indent=1) + "\n")
    print(f"\n  written to {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
