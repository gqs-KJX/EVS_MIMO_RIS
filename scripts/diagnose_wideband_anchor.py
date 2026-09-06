#!/usr/bin/env python3
"""Repeat the wideband local check with an estimated, rather than oracle, anchor.

The oracle-anchor diagnostic fixes the directional penalty at the true Jones
states, which no pipeline can do.  Here the anchor is taken from a real Stage-I
run and then held fixed across the matched and mismatched refinements, so the
comparison still isolates the mismatch rather than an anchor change.

Only ``gamma`` and ``eta_pol`` are replaced: those are the two keys the
directional regularizer reads.  The position initialization stays at truth, so
this remains a local refinement diagnostic and not an acquisition test.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.channel_model import (  # noqa: E402
    add_awgn,
    channel_components,
    generate_scene,
    synthesize_raw_tensor,
)
from src.experiments.final_mksc_ccop_common import Stage1Cache, make_paper_config  # noqa: E402
from src.experiments.run_paper_ablation_figures import _truth_init_estimate  # noqa: E402
from src.tensor_utils import hankelize_frequency  # noqa: E402
from src.validation_artifacts import (  # noqa: E402
    canonical_hash,
    validation_environment,
)

from diagnose_spatial_wideband import wideband_raw_tensor  # noqa: E402
from diagnose_wideband_nonlinear import run_one, summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--snr-db", type=float, default=-10.0)
    ap.add_argument("--anchor-snr-db", type=float, default=-10.0,
                    help="SNR of the observation Stage-I sees when producing the "
                         "anchor; the refinement observations stay noiseless")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "results" / "spatial_wideband" / "anchor_check.json")
    args = ap.parse_args()

    config = make_paper_config(args.seed, args.snr_db)
    scene = generate_scene(config, np.random.default_rng(args.seed))
    p_true = np.asarray(scene["p_u_true"], dtype=float)
    dt_true = float(scene["delta_t_true"])
    components = channel_components(scene, p_true, dt_true,
                                    scene["gamma_true"], scene["eta_true"])
    y_c = synthesize_raw_tensor(components, scene["beta_true"])
    y_wb = wideband_raw_tensor(scene, p_true, dt_true)

    # Stage-I sees a NOISY matched observation, so the anchor carries a
    # realistic Stage-I error.  A noiseless Stage-I run recovers the true Jones
    # states to ~1e-8 rad and the anchor check would be vacuous.  The
    # refinement observations themselves stay noiseless, so the mismatch is
    # still the only perturbation between the matched and wideband runs, and
    # the one anchor is shared by both.
    y_anchor, noise_variance = add_awgn(
        y_c, float(args.anchor_snr_db), np.random.default_rng(args.seed + 100000),
        active_mask=scene["evs_observation_mask"],
    )
    data = {
        "scene": scene, "true_components": components,
        "Y_true": y_c, "Y_noisy": y_anchor,
        "Z_true": hankelize_frequency(y_c, scene["P"]),
        "Z_noisy": hankelize_frequency(y_anchor, scene["P"]),
        "noise_variance": float(noise_variance), "timing": {},
    }
    print("  running Stage-I for the anchor ...", flush=True)
    stage1, _ = Stage1Cache(data, config).joint(4, True)

    init_oracle = _truth_init_estimate(scene, components)
    init_est = copy.deepcopy(init_oracle)
    init_est["gamma"] = np.asarray(stage1["gamma"], dtype=float).copy()
    init_est["eta_pol"] = np.asarray(stage1["eta_pol"], dtype=float).copy()

    anchor_shift = {
        "gamma_true": scene["gamma_true"].tolist(),
        "gamma_stage1": init_est["gamma"].tolist(),
        "eta_true": scene["eta_true"].tolist(),
        "eta_pol_stage1": init_est["eta_pol"].tolist(),
        "max_abs_gamma_shift_rad": float(
            np.max(np.abs(init_est["gamma"] - scene["gamma_true"]))),
    }
    print(f"  max |gamma - gamma_true| = {anchor_shift['max_abs_gamma_shift_rad']:.3e} rad",
          flush=True)

    results = {}
    for anchor_name, init in (("oracle", init_oracle), ("stage1_estimated", init_est)):
        for mode in ("jones_free", "jones_regularized"):
            for tag, y in (("matched", y_c), ("wideband", y_wb)):
                key = f"{anchor_name}/{mode}/{tag}"
                print(f"  running {key} ...", flush=True)
                results[key] = summarize(
                    run_one(y, init, scene, config, mode), scene, p_true, dt_true)

    provenance = {
        "environment": validation_environment(
            " ".join([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]),
            repo_root=ROOT),
        "resolved_config_hash": canonical_hash(
            {k: str(v) for k, v in sorted(config.items())}),
        "args": {k: (str(v) if isinstance(v, pathlib.Path) else v)
                 for k, v in vars(args).items()},
    }
    payload = {"seed": args.seed, "snr_db": args.snr_db, "anchor_snr_db": args.anchor_snr_db,
               "position_init": "truth",
               "anchor_note": "matched and wideband share one anchor within each block",
               "anchor_shift": anchor_shift, "results": results,
               "provenance": provenance}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"\n{'run':44s}{'|dp| mm':>13s}{'dt ps':>12s}")
    for k, r in results.items():
        print(f"  {k:42s}{r['dp_norm_mm']:13.3e}{r['dt_ps']:12.4f}")
    print(f"\n  written to {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
