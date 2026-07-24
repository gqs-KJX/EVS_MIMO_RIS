"""Numerical verification of the polarization--geometry information splitting.

Structural premise of the paper: the per-panel dictionary block factorizes as
``A_k = c_k(chi) (x) Theta_k`` where ``c_k = h_k (x) d_k (x) v_{B,k}`` carries
the whole dependence on ``chi = [p; Delta t]`` and the Maxwell--Jones manifold
``Theta_k`` is fixed by the *known* RIS->BS direction (see
``src/channel_model.py``: ``theta[k] = maxwell_matrix((p_B - r_k)/d_k)``).
Hence ``dTheta_k/dchi = 0`` exactly and the Jacobian columns are
``D_i = (d c_k/d chi_i) (x) s_k`` with ``s_k = Theta_k x_k``.  Combined with
the Kronecker projector splitting
``Pi_A^perp = P_c (x) P_Theta^perp + P_c^perp (x) I`` and ``P_Theta^perp s = 0``
this gives, for a single panel, the exact one-term free-Jones EFIM

    J_chi^mode = (2/sigma^2) * alpha_mode * G,
    G = Re{ Cdot^H Pi_c^perp Cdot }   (mode independent),

i.e. the receiver mode enters only as a positive scalar multiplying a common
geometric information matrix.  For K > 1 the same algebra gives a *weighted*
panel sum, so exact proportionality holds only when the per-panel weights
coincide.

Tests reported by this script
-----------------------------
T1 (K=1)  ``J^mode`` is a scalar multiple of ``J^full`` -> relative deviation
          at machine precision, and ``cond(J)`` is mode invariant.
T2        The scalar is bounded by the retained component energy ratio,
          ``alpha_mode <= E_mode / E_full`` (the free-Jones projector removes
          a further, mode-dependent part of the polarization subspace).
T3 (K=3)  Proportionality degrades into a weighted panel sum; the reported
          relative deviation and the change in ``cond(J)`` quantify how much
          the reduced modes *distort* rather than merely *scale* the
          information geometry.

The EFIM is evaluated on the noiseless tensor so that every mode is profiled
at the same true Jones state; evaluating on ``Y_noisy`` instead replaces the
true ``x`` by a mode-dependent plug-in estimate and changes ``alpha`` only.

Usage:
    python scripts/verify_evs_information_splitting.py
    python scripts/verify_evs_information_splitting.py --out results/evs_splitting.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.channel_model import evs_component_selection  # noqa: E402
from src.experiments.final_mksc_ccop_common import (  # noqa: E402
    make_paper_config,
    make_shared_data,
)
from src.experiments.run_paper_ablation_figures import (  # noqa: E402
    make_nested_receiver_mode_data,
)
from src.global_vp import data_only_efim_diagnostic  # noqa: E402

MODES = ("scalar", "dual_pol", "full_6d")
DEFAULT_SEEDS = (20260721, 20260722, 20260723)


def _efim(data: dict, mode: str, config: dict) -> tuple[np.ndarray, float]:
    """Clock-scaled free-Jones EFIM of one receiver mode at the true state."""
    masked = (
        data
        if mode == "full_6d"
        else make_nested_receiver_mode_data(data, mode, config)
    )
    scene = masked["scene"]
    cfg = dict(config)
    cfg["receiver_mode"] = mode
    p_true = np.asarray(scene["p_u_true"], dtype=float)
    dt_true = float(scene["delta_t_true"])
    diag = data_only_efim_diagnostic(
        masked["Y_true"],
        p_true,
        dt_true,
        {"p_u": p_true, "delta_t": dt_true},
        scene,
        cfg,
        sigma2=float(data["noise_variance"]),
    )
    return (
        np.asarray(diag["data_only_scaled_efim"], dtype=float),
        float(diag["data_only_scaled_efim_condition_number"]),
    )


def _component_energy(data: dict, mode: str) -> float:
    """Noiseless energy retained by the component mask of ``mode``."""
    scene = data["scene"]
    mask = np.tile(evs_component_selection(mode), int(scene["M_A"])).astype(bool)
    return float(np.linalg.norm(np.asarray(data["Y_true"])[mask]) ** 2)


def _cross_panel_separation(data: dict, mode: str) -> dict:
    """Polarimetric separability of distinct panels under a component mask.

    At a competitor state whose space--frequency signature matches panel k
    with panel k', the free-Jones residual factorizes as
    ``||c_k|| * ||Pi_{Theta_k'}^perp Theta_k x_k||``.  With ``d <= 2`` retained
    components ``col(S Theta_k') = C^d`` for every panel, so the residual
    vanishes identically and the cross-panel ambiguity is exact; the full
    six-component receiver has generically distinct two-dimensional
    polarization subspaces and a strictly positive residual.
    """
    scene = data["scene"]
    mask = evs_component_selection(mode).astype(bool)
    theta = np.asarray(scene["Theta"], dtype=complex)
    gamma = np.asarray(scene["gamma_true"], dtype=float)
    eta = np.asarray(scene["eta_true"], dtype=float)
    beta = np.asarray(scene["beta_true"], dtype=complex)
    k_paths = theta.shape[0]

    residuals: list[float] = []
    angles: list[float] = []
    for k in range(k_paths):
        e_k = np.array(
            [np.sin(gamma[k]) * np.exp(1j * eta[k]), np.cos(gamma[k])],
            dtype=complex,
        )
        s_k = theta[k][mask] @ (beta[k] * e_k)
        for k_other in range(k_paths):
            if k_other == k:
                continue
            other = theta[k_other][mask]
            basis, _ = np.linalg.qr(other)
            basis = basis[:, : np.linalg.matrix_rank(other)]
            residual = np.linalg.norm(s_k - basis @ (basis.conj().T @ s_k))
            residuals.append(float(residual / max(np.linalg.norm(s_k), 1e-300)))
            own, _ = np.linalg.qr(theta[k][mask])
            own = own[:, : np.linalg.matrix_rank(theta[k][mask])]
            singular = np.linalg.svd(basis.conj().T @ own, compute_uv=False)
            angles.append(
                float(np.degrees(np.arccos(np.clip(singular, 0.0, 1.0))).min())
            )
    return {
        "min_relative_residual": float(np.min(residuals)) if residuals else float("nan"),
        "median_relative_residual": (
            float(np.median(residuals)) if residuals else float("nan")
        ),
        "min_principal_angle_deg": float(np.min(angles)) if angles else float("nan"),
        "max_principal_angle_deg": float(np.max(angles)) if angles else float("nan"),
    }


def _position_bounds(j_scaled: np.ndarray) -> tuple[list[float], float]:
    """Per-axis CRB and PEB in metres after Schur elimination of the clock."""
    j_p = (
        j_scaled[:3, :3]
        - np.outer(j_scaled[:3, 3], j_scaled[3, :3]) / j_scaled[3, 3]
    )
    cov = np.linalg.inv(j_p)
    return np.sqrt(np.diag(cov)).tolist(), float(np.sqrt(np.trace(cov)))


def run_case(k_paths: int, snr_db: float, seed: int) -> dict:
    config = make_paper_config(int(seed), float(snr_db))
    config["K"] = int(k_paths)
    data = make_shared_data(config)

    efim = {}
    cond = {}
    for mode in MODES:
        efim[mode], cond[mode] = _efim(data, mode, config)
    reference = efim["full_6d"]
    energy = {mode: _component_energy(data, mode) for mode in MODES}

    case: dict = {"K": int(k_paths), "snr_db": float(snr_db), "seed": int(seed)}
    for mode in MODES:
        alpha = float(np.sum(efim[mode] * reference) / np.sum(reference**2))
        deviation = float(
            np.linalg.norm(efim[mode] - alpha * reference)
            / np.linalg.norm(efim[mode])
        )
        crb, peb = _position_bounds(efim[mode])
        case[mode] = {
            "alpha_vs_full": alpha,
            "relative_deviation_from_scalar_multiple": deviation,
            "condition_number": cond[mode],
            "component_energy_ratio": energy[mode] / energy["full_6d"],
            "per_axis_crb_m": crb,
            "peb_m": peb,
            "cross_panel_separation": _cross_panel_separation(data, mode),
        }
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--k-paths", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cases = [
        run_case(k_paths, args.snr_db, seed)
        for k_paths in args.k_paths
        for seed in args.seeds
    ]
    payload = json.dumps(cases, indent=1)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)

    for case in cases:
        head = f"K={case['K']} seed={case['seed']}"
        for mode in MODES:
            item = case[mode]
            print(
                f"{head:>20s}  {mode:<9s} "
                f"alpha={item['alpha_vs_full']:.8f} "
                f"dev={item['relative_deviation_from_scalar_multiple']:.2e} "
                f"cond={item['condition_number']:.4f} "
                f"E={item['component_energy_ratio']:.6f} "
                f"PEB={item['peb_m'] * 1e3:.4f}mm "
                f"xpanel_res={item['cross_panel_separation']['min_relative_residual']:.4f}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
