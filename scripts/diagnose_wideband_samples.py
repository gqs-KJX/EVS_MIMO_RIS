#!/usr/bin/env python3
"""Spatial-wideband mismatch sensitivity over a pre-specified sample set.

Nine samples: three positions x three seeds, all fixed before any wideband
result was inspected.  The positions are the reference scene position, a
median-difficulty grid position and the hardest grid position, both taken from
the catastrophic rates already released in results/paper_v4/positions50x480.
The seeds continue the campaign seed root.

Per sample this runs the first-order projection and four local refinements
(no penalty / paper penalty) x (matched / wideband), sharing one anchor, and
records the per-sample PEB and CEB at that position and draw.

Scope: local mismatch sensitivity of the refinement at the tested samples.
These runs say nothing about Stage-I acquisition or end-to-end catastrophic
rates, which are not exercised here.
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

from src.ccop_stage1_initializer import known_evs_union_basis  # noqa: E402
from src.channel_model import channel_components, generate_scene, synthesize_raw_tensor  # noqa: E402
from src.experiments.audit_bs_geometry import _signal_noise_variance  # noqa: E402
from src.experiments.final_mksc_ccop_common import make_paper_config  # noqa: E402
from src.experiments.run_paper_ablation_figures import _truth_init_estimate  # noqa: E402
from src.geometry import polarization_vector  # noqa: E402
from src.global_vp import _build_global_dictionary, data_only_efim_diagnostic  # noqa: E402
from src.validation_artifacts import (  # noqa: E402
    canonical_hash,
    validation_environment,
)

from diagnose_spatial_wideband import project_off, wideband_raw_tensor  # noqa: E402
from diagnose_wideband_nonlinear import run_one, summarize  # noqa: E402

POSITIONS = {                       # pre-specified, existing metrics only
    "reference": None,                          # scene default
    "typical": np.array([1.500, 0.050, 0.450]),  # median grid P_cat = 0.62 %
    "hard": np.array([2.600, 1.400, 1.350]),     # worst grid P_cat = 29.58 %
}
SEEDS = (20260727, 20260728, 20260729)


def first_order(scene, config, p_true, dt_true):
    components = channel_components(scene, p_true, dt_true,
                                    scene["gamma_true"], scene["eta_true"])
    y_c = synthesize_raw_tensor(components, scene["beta_true"])
    y_wb = wideband_raw_tensor(scene, p_true, dt_true)
    basis, _ = known_evs_union_basis(scene)
    bh = basis.conj().T
    comp = lambda y: np.einsum("ri,int->rnt", bh, y).reshape(-1)
    yc_v, ywb_v = comp(y_c), comp(y_wb)
    c0 = float(scene["c0"])
    phi, aux = _build_global_dictionary(
        np.concatenate([p_true, [dt_true]]), {"p_hat": p_true, "delta_t_hat": dt_true},
        scene, config, need_jacobian=True, evs_basis=basis)
    x_star = np.concatenate([
        scene["beta_true"][k] * polarization_vector(
            scene["gamma_true"][k], scene["eta_true"][k])
        for k in range(int(scene["K"]))])
    d_z = np.column_stack([dphi @ x_star for dphi in aux["dPhi_dx"]])
    d_z[:, 3] /= c0
    pd, pdelta = project_off(phi, d_z), project_off(phi, ywb_v - yc_v)
    j_real = np.vstack([pd.real, pd.imag])
    sv = np.linalg.svd(j_real, compute_uv=False)
    dz, *_ = np.linalg.lstsq(
        j_real, np.concatenate([pdelta.real, pdelta.imag]), rcond=None)
    return {
        "y_c": y_c, "y_wb": y_wb, "components": components,
        "lin_dp_norm_mm": float(np.linalg.norm(dz[:3]) * 1e3),
        "lin_dt_ps": float(dz[3] / c0 * 1e12),
        "jacobian_rank": int(np.linalg.matrix_rank(j_real)),
        "jacobian_condition": float(sv[0] / sv[-1]),
        "mismatch_energy_fraction_raw": float(
            np.linalg.norm(y_wb - y_c) ** 2 / np.linalg.norm(y_c) ** 2),
    }


def bounds_at(scene, config, p_true, dt_true, components, snr_db):
    y_true = synthesize_raw_tensor(components, scene["beta_true"])
    sigma2 = _signal_noise_variance(y_true, scene, snr_db)
    diag = data_only_efim_diagnostic(
        y_true, p_true, dt_true, _truth_init_estimate(scene, components),
        scene, config, sigma2=sigma2)
    efim = np.asarray(diag["data_only_scaled_efim"], float)
    efim = 0.5 * (efim + efim.T)
    cov = np.linalg.pinv(efim, rcond=1e-12)
    return (float(np.sqrt(max(np.trace(cov[:3, :3]), 0.0)) * 1e3),
            float(np.sqrt(max(cov[3, 3], 0.0)) / float(scene["c0"]) * 1e12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snr-db", type=float, default=-10.0)
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "results" / "spatial_wideband" / "sample_set.json")
    args = ap.parse_args()

    samples = []
    for pos_name, pos in POSITIONS.items():
        for seed in SEEDS:
            config = make_paper_config(seed, args.snr_db)
            scene = generate_scene(config, np.random.default_rng(seed))
            if pos is not None:
                scene["p_u_true"] = np.asarray(pos, dtype=float)
            p_true = np.asarray(scene["p_u_true"], dtype=float)
            dt_true = float(scene["delta_t_true"])
            print(f"  [{pos_name}/{seed}] p={np.round(p_true,3).tolist()}", flush=True)

            fo = first_order(scene, config, p_true, dt_true)
            init = _truth_init_estimate(scene, fo["components"])
            peb_mm, ceb_ps = bounds_at(
                scene, config, p_true, dt_true, fo["components"], args.snr_db)

            runs = {}
            for mode in ("jones_free", "jones_regularized"):
                for tag, y in (("matched", fo["y_c"]), ("wideband", fo["y_wb"])):
                    runs[f"{mode}/{tag}"] = summarize(
                        run_one(y, init, scene, config, mode), scene, p_true, dt_true)
            samples.append({
                "position_name": pos_name, "seed": seed,
                "p_true": p_true.tolist(),
                "anchor_provenance": "oracle (true Jones states)",
                "peb_position_mm": peb_mm, "ceb_ps": ceb_ps,
                **{k: v for k, v in fo.items()
                   if k not in ("y_c", "y_wb", "components")},
                "runs": runs,
            })
            r = runs["jones_free/wideband"]
            print(f"      lin {fo['lin_dp_norm_mm']:.5f} mm / {fo['lin_dt_ps']:.4f} ps"
                  f" | nonlin {r['dp_norm_mm']:.5f} mm / {r['dt_ps']:.4f} ps"
                  f" | PEB {peb_mm:.4f} mm  CEB {ceb_ps:.4f} ps", flush=True)

    provenance = {
        "environment": validation_environment(
            " ".join([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]),
            repo_root=ROOT),
        "resolved_config_hash": canonical_hash(
            {k: str(v) for k, v in sorted(config.items())}),
        "args": {k: (str(v) if isinstance(v, pathlib.Path) else v)
                 for k, v in vars(args).items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"snr_db": args.snr_db, "positions": {k: (v.tolist() if v is not None else None)
                                              for k, v in POSITIONS.items()},
         "seeds": list(SEEDS), "samples": samples,
         "provenance": provenance}, indent=1) + "\n")
    print(f"\n  written to {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
