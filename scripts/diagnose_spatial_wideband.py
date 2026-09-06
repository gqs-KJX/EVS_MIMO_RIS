#!/usr/bin/env python3
"""Deterministic spatial-wideband model-mismatch diagnostic.

Generates the noiseless observation with a per-subcarrier spatial manifold and
processes it with the center-frequency estimator, then reports the induced
position and clock bias.  This is the effect the frozen campaigns cannot show:
every released suite generates and estimates on the same center-frequency
manifold, so the residual is identically zero there by construction.

Isolation, not theory protection
--------------------------------
The per-tone model lives here rather than in ``src/channel_model.py`` so that
the released campaigns stay reproducible and the mismatch stays attributable.
Whether the theory needs extending is for the result to decide: spatial
wideband breaks the space--frequency separability used by the current
compression, but not necessarily every Maxwell factorization, and the common
clock still enters as a per-tone common unitary phase.

Physical array is held fixed across tones
-----------------------------------------
f_n = f_c + nu_n df with nu_n the archived (centered) subcarrier indices and
lambda_n = c0/f_n.  RIS element coordinates, panel rotations and centres are
fixed; the ULA spacing stays at the physical lambda_c/2 rather than becoming
lambda_n/2; Omega, the Jones states and the complex gains are one draw shared
by all tones; and the centre-path baseband delay phase is unchanged, so the
spatial responses add only element-relative path differences.
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

from src.ccop_stage1_initializer import known_evs_union_basis  # noqa: E402
from src.channel_model import (  # noqa: E402
    channel_components,
    generate_scene,
    synthesize_raw_tensor,
)
from src.experiments.final_mksc_ccop_common import make_paper_config  # noqa: E402
from src.geometry import (  # noqa: E402
    far_field_ris_response,
    local_geometry_from_position,
    near_field_spherical_response,
    polarization_vector,
    ula_steering,
)
from src.global_vp import _build_global_dictionary  # noqa: E402
from src.validation_artifacts import (  # noqa: E402
    canonical_hash,
    validation_environment,
)


def wideband_raw_tensor(scene: dict, p_u: np.ndarray, delta_t: float) -> np.ndarray:
    """Noiseless observation with a per-subcarrier spatial manifold, I x N x T."""
    c0 = float(scene["c0"])
    lambda_c = float(scene["wavelength"])
    f_c = c0 / lambda_c
    nu = np.asarray(scene["subcarrier_indices"], dtype=float)
    f_n = f_c + nu * float(scene["delta_f"])
    lambda_n = c0 / f_n
    spacing = lambda_c / 2.0                      # physical, frequency independent
    p_b = np.asarray(scene["p_B"], dtype=float)

    i_dim, n_dim, t_dim = int(scene["I"]), int(nu.size), int(scene["T"])
    y = np.zeros((i_dim, n_dim, t_dim), dtype=complex)

    for k in range(int(scene["K"])):
        range_m, elev, az, _ = local_geometry_from_position(
            p_u, scene["ris_centers"][k], scene["rotations"][k]
        )
        tau = (range_m + scene["d_RB"][k]) / c0 + delta_t
        d_delay = np.exp(-1j * 2.0 * np.pi * scene["delta_f"] * tau * nu)
        pol = scene["Theta"][k] @ polarization_vector(
            scene["gamma_true"][k], scene["eta_true"][k]
        )
        arrival = (scene["ris_centers"][k] - p_b) / float(scene["d_RB"][k])
        for n in range(n_dim):
            lam = float(lambda_n[n])
            a_ur = near_field_spherical_response(
                range_m, elev, az, scene["ris_grid"], lam
            )
            a_rb = far_field_ris_response(
                scene["ris_centers"][k], p_b, scene["rotations"][k],
                scene["ris_grid"], lam,
            )
            c_train = scene["Omega"][k] @ (a_rb * a_ur)
            v_b = ula_steering(int(scene["M_A"]), spacing, lam, arrival)
            a_evs = np.kron(v_b, pol) * scene["evs_observation_mask"]
            y[:, n, :] += (
                scene["beta_true"][k] * d_delay[n]
                * a_evs[:, None] * c_train[None, :]
            )
    return y


def project_off(phi: np.ndarray, v: np.ndarray) -> np.ndarray:
    """P v = v - phi lstsq(phi, v), without forming P."""
    return v - phi @ np.linalg.lstsq(phi, v, rcond=None)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--snr-db", type=float, default=-10.0)
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "results" / "spatial_wideband" / "wideband_bias.json")
    args = ap.parse_args()

    config = make_paper_config(args.seed, args.snr_db)
    scene = generate_scene(config, np.random.default_rng(args.seed))
    p_true = np.asarray(scene["p_u_true"], dtype=float)
    dt_true = float(scene["delta_t_true"])
    c0 = float(scene["c0"])
    xi_true = np.concatenate([p_true, [dt_true]])

    components = channel_components(
        scene, p_true, dt_true, scene["gamma_true"], scene["eta_true"]
    )
    y_c = synthesize_raw_tensor(components, scene["beta_true"])
    y_wb = wideband_raw_tensor(scene, p_true, dt_true)

    basis, _ = known_evs_union_basis(scene)
    bh = basis.conj().T
    comp = lambda y: np.einsum("ri,int->rnt", bh, y).reshape(-1)
    yc_v, ywb_v = comp(y_c), comp(y_wb)

    # energy the calibrated receive subspace does not capture under mismatch
    def off_subspace(y):
        full = y.reshape(int(scene["I"]), -1)
        return float(np.linalg.norm(full - basis @ (bh @ full)) ** 2)
    off_c, off_wb = off_subspace(y_c), off_subspace(y_wb)
    tot_wb = float(np.linalg.norm(y_wb) ** 2)

    init = {"p_hat": p_true, "delta_t_hat": dt_true}
    phi, aux = _build_global_dictionary(
        xi_true, init, scene, config, need_jacobian=True, evs_basis=basis
    )

    # x* is CONSTRUCTED from the generator in dictionary column order, not
    # inferred.  For vp mode 'jones_regularized' the atoms of path k are
    # kron(v_B[k], Theta_k[:, j]), j = 0, 1, so column 2k+j carries
    # beta_k * polarization_vector(gamma_k, eta_k)[j].
    x_star = np.concatenate([
        scene["beta_true"][k]
        * polarization_vector(scene["gamma_true"][k], scene["eta_true"][k])
        for k in range(int(scene["K"]))
    ])
    # Independent recovery from the MATCHED observation, as a cross-check only.
    x_fit, *_ = np.linalg.lstsq(phi, yc_v, rcond=None)
    x_star_agreement = float(
        np.linalg.norm(x_fit - x_star) / np.linalg.norm(x_star)
    )
    resid_matched = float(
        np.linalg.norm(phi @ x_star - yc_v) / np.linalg.norm(yc_v)
    )
    phi_sv = np.linalg.svd(phi, compute_uv=False)

    delta = ywb_v - yc_v                       # = (A_wb - A_c) x*
    d_chi = np.column_stack([dphi @ x_star for dphi in aux["dPhi_dx"]])
    d_z = d_chi.copy()
    d_z[:, 3] /= c0                            # z = [p, c0 dt]

    pd, pdelta = project_off(phi, d_z), project_off(phi, delta)
    j_real = np.vstack([pd.real, pd.imag])
    r_real = np.concatenate([pdelta.real, pdelta.imag])
    sv = np.linalg.svd(j_real, compute_uv=False)
    dz, *_ = np.linalg.lstsq(j_real, r_real, rcond=None)
    dp, dt_s = dz[:3], dz[3] / c0

    out = {
        "seed": args.seed, "snr_db": args.snr_db,
        "matched_dictionary_residual": resid_matched,
        "x_star_constructed_vs_recovered_rel": x_star_agreement,
        "phi_columns": int(phi.shape[1]),
        "phi_rank": int(np.linalg.matrix_rank(phi)),
        "phi_condition": float(phi_sv[0] / phi_sv[-1]),
        "off_subspace_energy_fraction_matched": off_c / tot_wb,
        "off_subspace_energy_fraction_wideband": off_wb / tot_wb,
        "mismatch_energy_fraction_compressed": float(
            np.linalg.norm(delta) ** 2 / np.linalg.norm(yc_v) ** 2
        ),
        "mismatch_energy_fraction_raw": float(
            np.linalg.norm(y_wb - y_c) ** 2 / np.linalg.norm(y_c) ** 2
        ),
        "jacobian_singular_values": sv.tolist(),
        "jacobian_condition": float(sv[0] / sv[-1]),
        "jacobian_rank": int(np.linalg.matrix_rank(j_real)),
        "bias_position_m": dp.tolist(),
        "bias_position_norm_mm": float(np.linalg.norm(dp) * 1e3),
        "bias_clock_ps": float(dt_s * 1e12),
    }
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
    out["provenance"] = provenance
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for key, value in out.items():
        if key != "jacobian_singular_values":
            print(f"  {key} = {value}")
    print(f"  singular values = {np.array2string(sv, precision=4)}")
    print(f"\n  written to {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
