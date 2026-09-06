#!/usr/bin/env python3
"""Recompute the free-Jones clock error bound (CEB) used as the Fig. 2 anchor.

Provenance for the CEB curve in the main-paper benchmark figure, which is the
one plotted quantity with no per-trial column in the released benchmark suite.

What is held fixed and what is redrawn
--------------------------------------
The geometry -- panel centres, rotations, BS array and the UE position -- is
the frozen paper scene and is identical across draws.  Advancing the seed
re-runs ``generate_scene``, which redraws the per-panel RIS phase codes
(``omega``) *and* the Jones states and complex gains.  These draws therefore
sample the training-code/Jones/gain ensemble at fixed geometry, not a
Jones/gain ensemble at fixed training.  Fixing the codes would require
freezing ``omega`` explicitly before the Jones draw.

Usage
-----
    python scripts/recompute_ceb_anchor.py \
        --out results/ceb_anchor/ceb_anchor.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.channel_model import (  # noqa: E402
    channel_components,
    generate_scene,
    synthesize_raw_tensor,
)
from src.experiments.audit_bs_geometry import (  # noqa: E402
    _signal_noise_variance,
    _truth_init_estimate,
)
from src.experiments.final_mksc_ccop_common import make_paper_config  # noqa: E402
from src.global_vp import data_only_efim_diagnostic  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-root", type=int, default=20260727)
    ap.add_argument("--snr-db", type=float, default=-10.0)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "results" / "ceb_anchor" / "ceb_anchor.json",
    )
    args = ap.parse_args()

    config = make_paper_config(args.seed_root, args.snr_db)
    rows = []
    for index in range(args.draws):
        seed = args.seed_root + index
        scene = generate_scene(config, np.random.default_rng(seed))
        position = np.asarray(scene["p_u_true"], dtype=float)
        components = channel_components(
            scene,
            position,
            float(config["delta_t_true"]),
            scene["gamma_true"],
            scene["eta_true"],
        )
        y_true = synthesize_raw_tensor(components, scene["beta_true"])
        sigma2 = _signal_noise_variance(y_true, scene, args.snr_db)
        diagnostic = data_only_efim_diagnostic(
            y_true,
            position,
            config["delta_t_true"],
            _truth_init_estimate(scene, components),
            scene,
            config,
            sigma2=sigma2,
        )
        efim = np.asarray(diagnostic["data_only_scaled_efim"], dtype=float)
        efim = 0.5 * (efim + efim.T)
        covariance = np.linalg.pinv(efim, rcond=1.0e-12)
        ceb_ps = (
            np.sqrt(max(float(covariance[3, 3]), 0.0))
            / float(scene["c0"])
            * 1.0e12
        )
        rows.append({"seed": seed, "ceb_ps": float(ceb_ps), "sigma2": float(sigma2)})
        print(f"  draw {index:2d}  seed={seed}  CEB={ceb_ps:8.4f} ps")

    values = np.array([r["ceb_ps"] for r in rows])
    summary = {
        "seed_root": args.seed_root,
        "snr_db": args.snr_db,
        "n_draws": args.draws,
        "redrawn_per_seed": ["ris_phase_codes", "jones_states", "complex_gains"],
        "held_fixed": ["panel_geometry", "bs_array", "ue_position"],
        "ceb_ps_median": float(np.median(values)),
        "ceb_ps_min": float(values.min()),
        "ceb_ps_max": float(values.max()),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"\n  n={len(values)}  median={summary['ceb_ps_median']:.4f} ps"
        f"  min={summary['ceb_ps_min']:.4f}  max={summary['ceb_ps_max']:.4f}"
        f"\n  written to {args.out.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
