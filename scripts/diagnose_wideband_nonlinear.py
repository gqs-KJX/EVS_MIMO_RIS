#!/usr/bin/env python3
"""Nonlinear cross-check for the spatial-wideband first-order bias prediction.

Four local refinements at one sample, all sharing the same truth position
initialization and the same Jones anchor:

                       no penalty (jones_free)   paper penalty (jones_regularized)
  matched  (centre f)  zero-mismatch control     penalty / numerical control
  wideband (per tone)  checks the first order    response of the actual refiner

The no-penalty column is the objective the first-order projection corresponds
to.  The penalized column answers a different question and a disagreement
between the two columns is not evidence that the linearization failed.

Anchor provenance is recorded explicitly.  ``oracle`` uses the true Jones
states as the directional anchor; it is a local diagnostic, not an achievable
pipeline.  Matched and mismatched runs always share one anchor so that the
comparison isolates the mismatch.
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

from src.ccop_jvp import refine_ccop_jvp  # noqa: E402
from src.channel_model import channel_components, generate_scene, synthesize_raw_tensor  # noqa: E402
from src.experiments.final_mksc_ccop_common import make_paper_config  # noqa: E402
from src.experiments.run_paper_ablation_figures import _truth_init_estimate  # noqa: E402
from src.validation_artifacts import (  # noqa: E402
    canonical_hash,
    validation_environment,
)

sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_spatial_wideband import wideband_raw_tensor  # noqa: E402


def run_one(y_raw, init, scene, config, mode, tol_scale=1.0):
    cfg = copy.deepcopy(config)
    cfg.setdefault("global_vp", {})["mode"] = mode
    if tol_scale != 1.0:
        ccop = dict(cfg.get("ccop_jvp", {}))
        ccop["outer_ftol"] = 1.0e-12 * tol_scale
        ccop["outer_gtol"] = 1.0e-8 * tol_scale
        cfg["ccop_jvp"] = ccop
    out = refine_ccop_jvp(np.asarray(y_raw, complex), init, scene, cfg)
    return out


def summarize(out, scene, p_true, dt_true):
    p_hat = np.asarray(out["p_u"], dtype=float).reshape(3)
    dt_hat = float(out["delta_t"])
    opt = out.get("optimizer", {})
    return {
        "p_hat": p_hat.tolist(),
        "delta_t_hat_s": dt_hat,
        "dp_vec_m": (p_hat - p_true).tolist(),
        "dp_norm_mm": float(np.linalg.norm(p_hat - p_true) * 1e3),
        "dt_ps": float((dt_hat - dt_true) * 1e12),
        "raw_objective_final": float(out.get("raw_objective_final", np.nan)),
        "jones_regularizer_objective_final": float(
            out.get("jones_regularizer_objective_final", np.nan)),
        "total_objective_final": float(out.get("total_objective_final", np.nan)),
        "position_evaluations": out.get("ccop_position_evaluations"),
        "clock_profile_evaluations": out.get("ccop_clock_profile_evaluations"),
        "clock_all_certified": out.get("ccop_clock_profiles_all_certified"),
        "clock_certificate_gap_ratio_max": out.get(
            "ccop_clock_profile_max_certificate_gap_ratio"),
        "optimizer_success": opt.get("success"),
        "optimizer_message": str(opt.get("message", ""))[:100],
        "optimizer_n_iter": opt.get("n_iter"),
        "selected_candidate": out.get("selected_candidate"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--snr-db", type=float, default=-10.0)
    ap.add_argument("--epsilons", type=float, nargs="*", default=[])
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "results" / "spatial_wideband" / "nonlinear_check.json")
    args = ap.parse_args()

    config = make_paper_config(args.seed, args.snr_db)
    scene = generate_scene(config, np.random.default_rng(args.seed))
    p_true = np.asarray(scene["p_u_true"], dtype=float)
    dt_true = float(scene["delta_t_true"])

    components = channel_components(scene, p_true, dt_true,
                                    scene["gamma_true"], scene["eta_true"])
    y_c = synthesize_raw_tensor(components, scene["beta_true"])
    y_wb = wideband_raw_tensor(scene, p_true, dt_true)
    init = _truth_init_estimate(scene, components)      # oracle anchor + truth position

    results = {}
    for mode in ("jones_free", "jones_regularized"):
        for tag, y in (("matched", y_c), ("wideband", y_wb)):
            key = f"{mode}/{tag}"
            print(f"  running {key} ...", flush=True)
            results[key] = summarize(
                run_one(y, init, scene, config, mode), scene, p_true, dt_true)
    # tolerance stability on the decisive cell
    print("  running jones_free/wideband @ tightened tolerance ...", flush=True)
    results["jones_free/wideband_tight"] = summarize(
        run_one(y_wb, init, scene, config, "jones_free", tol_scale=1e-2),
        scene, p_true, dt_true)

    for eps in args.epsilons:
        key = f"jones_free/eps_{eps}"
        print(f"  running {key} ...", flush=True)
        results[key] = summarize(
            run_one(y_c + eps * (y_wb - y_c), init, scene, config, "jones_free"),
            scene, p_true, dt_true)

    # increments relative to the matched control of the same column
    for mode in ("jones_free", "jones_regularized"):
        base = np.asarray(results[f"{mode}/matched"]["p_hat"], float)
        bdt = results[f"{mode}/matched"]["delta_t_hat_s"]
        for tag in list(results):
            if not tag.startswith(mode) or tag.endswith("matched"):
                continue
            r = results[tag]
            r["dp_incremental_mm"] = float(
                np.linalg.norm(np.asarray(r["p_hat"], float) - base) * 1e3)
            r["dt_incremental_ps"] = float((r["delta_t_hat_s"] - bdt) * 1e12)

    provenance = {
        "environment": validation_environment(
            " ".join([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]),
            repo_root=ROOT),
        "resolved_config_hash": canonical_hash(
            {k: str(v) for k, v in sorted(config.items())}),
        "args": {k: (str(v) if isinstance(v, pathlib.Path) else v)
                 for k, v in vars(args).items()},
    }
    payload = {"seed": args.seed, "snr_db": args.snr_db,
               "anchor_provenance": "oracle (true Jones states); local diagnostic only",
               "position_init": "truth", "results": results,
               "provenance": provenance}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"\n{'run':34s}{'|dp| mm':>12s}{'dt ps':>11s}{'incr mm':>11s}{'incr ps':>11s}")
    for k, r in results.items():
        print(f"  {k:32s}{r['dp_norm_mm']:12.6f}{r['dt_ps']:11.4f}"
              f"{r.get('dp_incremental_mm', float('nan')):11.6f}"
              f"{r.get('dt_incremental_ps', float('nan')):11.4f}")
    print(f"\n  written to {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
