#!/usr/bin/env python
"""Build the publication figures of the MKSC-GI + CCOP-JVP manuscript.

Reads only the frozen result CSVs under ``results/`` and writes vector PDFs to
``tex/figs/``.  No experiment is re-run and no metric is recomputed: every
plotted quantity is a column of a released summary/trial CSV, except for the
paired PEB efficiency ratio, which is formed from per-trial columns of the same
files.

The manuscript reports the operating region ``SNR >= -10 dB``; the released
CSVs also contain the threshold region down to ``-30 dB``, which is plotted
only in the supplementary material.  ``--snr-min`` selects the floor: the
default ``-10`` produces the main-text figures, ``--snr-min -30`` reproduces
the full-range supplementary versions.  Panels that are not SNR sweeps (fixed
``-10`` dB studies) are unaffected.

Usage:  python scripts/make_paper_figures.py [--out-dir tex/figs] [--snr-min -10]
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Patch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FINAL = RES / "final_mksc_ccop"

# Lowest SNR shown in the SNR sweeps; set from the command line.
SNR_MIN = -10.0

# ------------------------------------------------------------- campaigns ----
# Each figure names its inputs by logical dataset key, never by directory, so
# that a re-run of the campaign is a one-line change here rather than an edit
# scattered over nine plotting functions.  ``frozen`` is the 2026-07-19/21
# release; ``v3`` is the 2026-07-28 re-run on the optimized source tree, whose
# external benchmark is a single 13-point artifact instead of the three-way
# split (main grid / high-SNR completion / external clock) the frozen campaign
# needed.  Any key mapped to a list is concatenated in order.
CAMPAIGNS = {
    "frozen": {
        "snr_internal": [FINAL / "snr_internal_paper480"],
        "components": [FINAL / "components_paper480"],
        "components_m20": [FINAL / "components_paper480"],
        "receiver": [FINAL / "receiver_information_paper480"],
        "compression": [FINAL / "compression_matched_paper480"],
        "benchmark": [
            RES / "benchmark_full_k3_medium-480-final",
            RES / "benchmark_full_k3_medium-480-final-snr15to30",
        ],
        "benchmark_clock": [RES / "benchmark_clock_external_480"],
        "maxwell_mismatch": [FINAL / "maxwell_mismatch_paper480"],
        "colored_noise": [FINAL / "colored_noise_boundary480"],
        "ris_calibration": [FINAL / "ris_bs_calibration_boundary480"],
        "model_order": [FINAL / "model_order_mismatch480"],
        "positions": [FINAL / "positions50x480"],
        "resolvability": [FINAL / "evs_resolvability_paper480"],
        "scaling": [FINAL / "robustness_scaling480"],
        "benchmark_runtime": [FINAL / "benchmark_runtime30_cpu"],
        "components_cost": [FINAL / "components_cost30_cpu"],
    },
    "v3": {
        "snr_internal": [RES / "paper_v3" / "snr_internal_480"],
        "components": [RES / "paper_v3" / "components_480"],
        "components_m20": [RES / "paper_v3" / "components_480_m20"],
        "receiver": [RES / "paper_v3" / "receiver_information_480"],
        "compression": [RES / "paper_v3" / "compression_matched_480"],
        "benchmark": [RES / "paper_v3" / "benchmark_refinement_matched_960"],
        "benchmark_clock": [RES / "paper_v3" / "benchmark_refinement_matched_960"],
        "maxwell_mismatch": [RES / "paper_v3" / "maxwell_mismatch_480"],
        "colored_noise": [RES / "paper_v3" / "colored_noise_boundary_480"],
        "ris_calibration": [RES / "paper_v3" / "ris_bs_calibration_boundary_480"],
        "model_order": [RES / "paper_v3" / "model_order_mismatch_480"],
        "positions": [RES / "paper_v3" / "positions50x480"],
        "resolvability": [RES / "paper_v3" / "evs_resolvability_480"],
        "scaling": [RES / "paper_v3" / "robustness_scaling_480"],
        "benchmark_runtime": [RES / "paper_v3" / "benchmark_runtime30_cpu"],
        "components_cost": [RES / "paper_v3" / "components_cost30_cpu"],
    },
    # 10 trials per cell.  A pipeline smoke test, NOT a source of paper
    # numbers: at n = 10 a Clopper-Pearson interval on a rate spans roughly
    # +-30 percentage points and the paired bootstrap / McNemar statistics the
    # frozen protocol reports are meaningless.  The two cost entries point at
    # the 30-trial serialized suites, which are authoritative.
    "verify10": {
        "snr_internal": [RES / "paper_verify10" / "snr_internal"],
        "components": [RES / "paper_verify10" / "components_m10"],
        "components_m20": [RES / "paper_verify10" / "components_m20"],
        "receiver": [RES / "paper_verify10" / "receiver_information"],
        "compression": [RES / "paper_verify10" / "compression_matched"],
        "benchmark": [RES / "paper_verify10" / "benchmark_matched"],
        "benchmark_clock": [RES / "paper_verify10" / "benchmark_matched"],
        "maxwell_mismatch": [RES / "paper_verify10" / "maxwell_mismatch"],
        "colored_noise": [RES / "paper_verify10" / "colored_noise_boundary"],
        "ris_calibration": [RES / "paper_verify10" / "ris_bs_calibration_boundary"],
        "model_order": [RES / "paper_verify10" / "model_order_mismatch"],
        "positions": [RES / "paper_verify10" / "positions50"],
        "resolvability": [RES / "paper_verify10" / "evs_resolvability"],
        "scaling": [RES / "paper_verify10" / "robustness_scaling"],
        "benchmark_runtime": [RES / "paper_v3" / "benchmark_runtime30_cpu"],
        "components_cost": [RES / "paper_v3" / "components_cost30_cpu"],
    },
}

# Active dataset map; set from the command line by ``--campaign``.  The paper
# source of record is v3, so a plain invocation cannot silently fall back to
# an earlier campaign.
DATA = CAMPAIGNS["v3"]


def ds(key: str, filename: str) -> list[dict]:
    """Load and concatenate ``filename`` across every directory of dataset ``key``."""
    rows: list[dict] = []
    for directory in DATA[key]:
        rows.extend(load(directory / filename))
    return rows

# ----------------------------------------------------------------- style ----
COL_W = 3.5   # IEEE single column [in]
DBL_W = 7.16  # IEEE double column [in]

C = {
    "proposed": "#0B5FA5",
    "gi4": "#2E8B57",
    "r3": "#E08A00",
    "r2": "#C8102E",
    "a1": "#8C8C8C",
    "a2": "#6A3D9A",
    "peb": "#000000",
    "als": "#7B3294",
    "vbi": "#00868B",
    "omp": "#A0522D",
    "raw": "#C8102E",
    "scalar": "#C8102E",
    "dual": "#E08A00",
    "full": "#0B5FA5",
}
MK = {
    "proposed": "o",
    "gi4": "v",
    "r3": "^",
    "r2": "s",
    "a1": "P",
    "a2": "D",
    "als": "D",
    "vbi": "P",
    "omp": "X",
    "raw": "s",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            # Figures are exported at their IEEE final widths.  These sizes
            # therefore survive in the typeset PDF without a second, large
            # LaTeX down-scaling step.
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "0.7",
            "legend.borderpad": 0.3,
            "legend.labelspacing": 0.25,
            "legend.handlelength": 1.9,
            "legend.handletextpad": 0.5,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def grid(ax, which="major") -> None:
    ax.grid(True, which=which)
    ax.set_axisbelow(True)


def fig_legend(fig, ax, ncol=3, y=-0.02, extra=()) -> None:
    """Place one shared legend under the whole figure."""
    h, l = ax.get_legend_handles_labels()
    for hh, ll in extra:
        h.append(hh)
        l.append(ll)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, fontsize=7.2, columnspacing=1.2)


# ------------------------------------------------------------------ data ----
def load(path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def snr_rows(rows, col="x_value") -> list[dict]:
    """Keep only the rows of an SNR sweep at or above ``SNR_MIN``."""
    return [r for r in rows if fnum(r[col]) >= SNR_MIN]


def clip_ylim(ax, main, wide) -> None:
    """Hand-tuned limits: ``main`` for the operating region, ``wide`` when the
    threshold region is also plotted (supplementary)."""
    ax.set_ylim(*(main if SNR_MIN >= -10.0 else wide))


def rate_ylim(ax, values, *, pad=0.14, minimum=1.0) -> None:
    """Percentage axis sized by the data instead of by a frozen constant.

    Rates move by an order of magnitude between campaigns, so a hand-tuned
    ceiling silently turns into a mostly empty panel.  ``values`` are the
    percentages actually drawn, including any interval upper limits.
    """
    finite = [v for v in values if math.isfinite(v)]
    top = max(max(finite, default=0.0), minimum)
    ax.set_ylim(-pad * top / 2.0, (1.0 + pad) * top)


def sel(rows, **kw) -> list[dict]:
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def curve(rows, xcol, ycol, variant_col=None, variant=None, sort=True):
    rs = rows if variant is None else sel(rows, **{variant_col: variant})
    if sort:
        rs = sorted(rs, key=lambda r: fnum(r[xcol]))
    x = np.array([fnum(r[xcol]) for r in rs])
    y = np.array([fnum(r[ycol]) for r in rs])
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def band(rows, xcol, lo, hi, variant_col=None, variant=None):
    rs = rows if variant is None else sel(rows, **{variant_col: variant})
    rs = sorted(rs, key=lambda r: fnum(r[xcol]))
    x = np.array([fnum(r[xcol]) for r in rs])
    a = np.array([fnum(r[lo]) for r in rs])
    b = np.array([fnum(r[hi]) for r in rs])
    return x, a, b


def rms(a) -> float:
    a = [x for x in a if math.isfinite(x)]
    return math.sqrt(sum(x * x for x in a) / len(a)) if a else float("nan")


def nearest_rank(values, q: float) -> float:
    """Nearest-rank percentile, the convention every quantile in the paper uses.

    ``benchmark_summary.csv`` stores linearly interpolated percentiles, which
    disagree with the text by two orders of magnitude in the bimodal tail (the
    4-D route at -10 dB: 126 mm interpolated against 734 mm nearest-rank).  Any
    quantile drawn beside a quoted one is therefore recomputed here from the
    per-trial column with ``ceil(q/100 * n) - 1`` on the ascending sort.
    """
    v = sorted(x for x in values if math.isfinite(x))
    if not v:
        return float("nan")
    return v[max(0, min(len(v) - 1, int(math.ceil(q / 100.0 * len(v))) - 1))]


# Clock error bound at the default UE position and -10 dB: median over the 20
# Jones/gain draws of the supplementary bound table.  It is the one plotted
# quantity that is not a released CSV column, because no per-trial CEB is
# logged in the benchmark suite.  Its SNR dependence is *not* assumed: the
# curve is anchored here and scaled by the released free-Jones PEB, which is
# exact because the EFIM scales as 2/sigma^2 and the geometry is fixed.
CEB_REF_PS_AT_MINUS10 = 3.35


# =========================================================== fig: geometry ==
def _resolved_scene() -> dict:
    """Panel centres, orientations, BS and UE box as the campaign resolved them.

    Read from the released ``resolved_base_config.json`` rather than restated
    here, so the figure cannot drift from the geometry the trials were run on.
    """
    import json

    path = DATA["snr_internal"][0] / "resolved_base_config.json"
    cfg = json.loads(path.read_text())["resolved_config"]
    return {
        "ris": np.asarray(cfg["ris_centers"], float),
        "rot": np.asarray(cfg["ris_rotations"], float),
        "bs": np.asarray(cfg["p_B"], float),
        "ue": np.asarray(cfg["p_u_true"], float),
        "box": np.asarray(cfg["ue_bounds"], float),
        "side": int(cfg["ris_shape"][0]),
        "lam": float(cfg["wavelength"]),
        "c0": float(cfg["c0"]),
    }


def fig_geometry(out: pathlib.Path) -> None:
    sc = _resolved_scene()
    side = sc["side"]
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(DBL_W, 2.55), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # (a) Keep the topology schematic orthographic.  Perspective in the old
    # 3-D drawing consumed space without encoding an additional model degree
    # of freedom at final print size.
    a.set_xlim(0, 10)
    a.set_ylim(0, 6)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("(a) System topology", pad=2)

    bs = (0.8, 3.0)
    for yy in np.linspace(1.55, 4.45, 6):
        a.plot(bs[0], yy, "o", ms=4.6, mfc="white", mec="0.1", mew=1.0)
    a.plot([bs[0], bs[0]], [1.55, 4.45], color="0.1", lw=1.0)
    a.text(0.15, 5.18, "BS: EVS--ULA\n" + r"($M_A$ sensors)",
           ha="left", va="top", linespacing=0.9)
    a.text(0.08, 1.02, r"$[E_x,E_y,E_z,H_x,H_y,H_z]^{\mathsf{T}}$", fontsize=7.4)

    def draw_ris(ax, xy, k):
        x0, y0 = xy
        ax.add_patch(Rectangle((x0 - 0.38, y0 - 0.48), 0.76, 0.96,
                               fc="#EAF2F8", ec="#243746", lw=1.0, zorder=4))
        for ii in range(3):
            for jj in range(4):
                ax.add_patch(Rectangle((x0 - 0.29 + 0.19 * jj,
                                        y0 - 0.36 + 0.22 * ii),
                                       0.12, 0.13, fc="#9EC1DF", ec="#243746",
                                       lw=0.35, zorder=5))
        ax.text(x0, y0 + 0.62, rf"RIS$_{k}$", ha="center", weight="bold")

    ris_xy = [(4.15, 1.25), (4.05, 3.0), (4.35, 4.75)]
    for k, xy in enumerate(ris_xy, 1):
        draw_ris(a, xy, k)

    a.add_patch(Rectangle((7.55, 1.35), 1.85, 3.25, fc="#F5F7F9",
                          ec="#8795A1", lw=0.9, ls="--"))
    a.text(8.47, 4.78, "UE search region", ha="center", color="0.35")
    ue = (8.05, 3.0)
    a.plot(*ue, "o", color="0.08", ms=5.5, zorder=7)
    a.text(8.05, 3.28, "UE", ha="center", weight="bold")
    a.text(8.48, 4.14, r"$\mathbf{p}_u,\Delta t$ unknown", ha="center", fontsize=7.6)

    for xy in ris_xy:
        a.add_patch(FancyArrowPatch(ue, xy, arrowstyle="-|>", mutation_scale=8,
                                    color=C["proposed"], lw=1.25, zorder=2))
        a.add_patch(FancyArrowPatch(xy, bs, arrowstyle="-|>", mutation_scale=8,
                                    color=C["r2"], lw=1.1, ls="--", zorder=1))
    a.plot([bs[0], ue[0]], [bs[1], ue[1]], color="0.45", lw=1.0, ls=":")
    a.plot(5.95, 3.0, marker="x", ms=8, mew=1.5, color="0.1")
    a.text(5.95, 2.67, "blocked LoS", ha="center", fontsize=7.4)

    # Assumption (A2) is a quantitative claim about two link ranges, and the
    # schematic is where a reader would look for it: put both ranges and the
    # Fraunhofer distance of the panel aperture on the drawing itself.
    d_ur = np.linalg.norm(sc["ris"] - sc["ue"][None, :], axis=1)
    d_rb = np.linalg.norm(sc["ris"] - sc["bs"][None, :], axis=1)
    diag = math.sqrt(2.0) * side * sc["lam"] / 2.0
    a.text(0.05, 0.44,
           rf"UE--RIS $={d_ur.min():.1f}$--${d_ur.max():.1f}$ m: near field",
           ha="left", va="center", fontsize=6.8, color=C["proposed"])
    a.text(0.05, 0.06,
           rf"RIS--BS $={d_rb.min():.1f}$--${d_rb.max():.1f}$ m "
           rf"$>2D^2/\lambda_c={2 * diag ** 2 / sc['lam']:.1f}$ m",
           ha="left", va="center", fontsize=6.8, color=C["r2"])

    # (b) The local path view carries the parameterization that was illegible
    # when embedded below the old single-column topology drawing.
    b.set_xlim(-2.6, 2.65)
    b.set_ylim(-1.55, 2.15)
    b.set_aspect("equal")
    b.axis("off")
    b.set_title(r"(b) $k$th cascaded path", pad=2)
    origin, p_bs, p_ue = (0.0, 0.0), (-2.05, 1.25), (2.15, 1.45)
    b.plot(*origin, "o", color="0.1", ms=4.5, zorder=5)
    b.text(-0.18, -0.27, r"RIS$_k$", ha="right", weight="bold")
    b.add_patch(FancyArrowPatch(origin, p_ue, arrowstyle="-|>", mutation_scale=9,
                                color=C["proposed"], lw=1.35))
    b.add_patch(FancyArrowPatch(origin, p_bs, arrowstyle="-|>", mutation_scale=9,
                                color=C["r2"], lw=1.2, ls="--"))
    b.plot(*p_ue, "o", color="0.1", ms=4.5)
    b.text(p_ue[0] + 0.08, p_ue[1] + 0.02, "UE", weight="bold")
    b.text(p_bs[0] - 0.05, p_bs[1] + 0.08, "BS", ha="right", weight="bold")
    b.text(1.05, 0.95, r"$d_{\rm UR,k}$", color=C["proposed"])
    b.text(-1.34, 0.90, r"$d_{\rm RB,k}$", color=C["r2"])

    # Local axes and angular coordinates.
    b.add_patch(FancyArrowPatch(origin, (1.05, 0), arrowstyle="-|>",
                                mutation_scale=7, color="0.25", lw=0.8))
    b.add_patch(FancyArrowPatch(origin, (0, 1.05), arrowstyle="-|>",
                                mutation_scale=7, color="0.25", lw=0.8))
    b.add_patch(FancyArrowPatch(origin, (-0.75, -0.72), arrowstyle="-|>",
                                mutation_scale=7, color="0.25", lw=0.8))
    b.text(1.06, -0.08, r"$x_k$")
    b.text(0.05, 1.06, r"$z_k$")
    b.text(-0.85, -0.82, r"$y_k$")
    b.add_patch(Arc(origin, 1.25, 1.25, theta1=0, theta2=34,
                    color=C["proposed"], lw=0.8))
    b.add_patch(Arc(origin, 1.05, 1.05, theta1=149, theta2=180,
                    color=C["r2"], lw=0.8))
    b.text(0.62, 0.19, r"$\phi_{\rm UR,k}$", color=C["proposed"], fontsize=7.6)
    b.text(-0.86, 0.16, r"$\phi_{\rm RB,k}$", color=C["r2"], fontsize=7.6)
    b.text(0.92, 1.34, r"$\theta_{\rm UR,k}$", color=C["proposed"], fontsize=7.6)
    b.text(-1.70, 1.42, r"$\theta_{\rm RB,k}$", color=C["r2"], fontsize=7.6)
    b.text(0.0, -1.02,
           r"$\boldsymbol{\eta}_k=(d_{\rm UR,k},u_k,v_k),\quad"
           r"\tau_k=(d_{\rm UR,k}+d_{\rm RB,k})/c_0+\Delta t$",
           ha="center", va="center", fontsize=8.0,
           bbox=dict(boxstyle="round,pad=0.25", fc="#F5F7F9", ec="0.65", lw=0.6))

    handles = [
        Line2D([], [], color=C["proposed"], lw=1.35,
               label="exact spherical UE--RIS"),
        Line2D([], [], color=C["r2"], lw=1.2, ls="--",
               label="calibrated plane-wave RIS--BS"),
        Line2D([], [], color="0.45", lw=1.0, ls=":", label="blocked LoS"),
        Patch(fc="#EAF2F8", ec="#243746", label=rf"RIS panel (${side}\times{side}$)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=4, frameon=False, fontsize=7.3, handlelength=2.2,
               columnspacing=1.1)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.91, bottom=0.18, wspace=0.08)
    fig.savefig(out / "fig_geometry.pdf")
    plt.close(fig)


# ============================================================ fig: internal ==
def fig_internal(out: pathlib.Path) -> None:
    S = snr_rows(ds("snr_internal", "ablation_summary.csv"))
    P = snr_rows(ds("snr_internal", "ablation_paired.csv"))
    series = [
        ("proposed", "Proposed MKSC-GI + CCOP-JVP", C["proposed"], MK["proposed"], "-"),
        ("mksc_gi_4_no_refresh_ccop", "MKSC-GI, no anchor refresh", C["gi4"], MK["gi4"], "--"),
        ("old_stage1_ccop", "R3: frozen Stage-I + CCOP-JVP", C["r3"], MK["r3"], "-."),
        ("scaled_4d", "R2: frozen Stage-I + 4-D Jones-VP", C["r2"], MK["r2"], ":"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(DBL_W, 4.0))
    (a, b), (c, d) = axes

    for v, lab, col, mk, ls in series:
        x, y = curve(S, "x_value", "position_rmse_m", "variant", v)
        mfc = "none" if v == "old_stage1_ccop" else col
        a.semilogy(x, y, color=col, marker=mk, ls=ls, mfc=mfc, label=lab)
        x, y = curve(S, "x_value", "position_conditional_rmse_m", "variant", v)
        a.semilogy(x, y, color=col, marker=mk, mfc="white", ls="--", lw=1.05)
    x, y = curve(S, "x_value", "peb_position_m_rms", "variant", "free_jones_peb")
    a.semilogy(x, y, color=C["peb"], ls=(0, (4, 2)), lw=1.0, label="Free-Jones PEB")
    a.annotate("inlier-conditional RMSE\n(all routes $\\approx$ PEB)", xy=(5, 2.3e-4),
               xytext=(-1, 8e-6), fontsize=6.0, ha="center",
               arrowprops=dict(arrowstyle="->", lw=0.5, color="0.35"))
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Position RMSE [m]")
    a.set_title("(a) Overall and conditional RMSE", pad=3)
    clip_ylim(a, (2e-6, 2), (3e-7, 3))
    grid(a)

    tops = []
    for v, lab, col, mk, ls in series:
        mfc = "none" if v == "old_stage1_ccop" else col
        x, y = curve(S, "x_value", "catastrophic_rate", "variant", v)
        xl, lo, hi = band(S, "x_value", "outlier_ci_low", "outlier_ci_high", "variant", v)
        b.plot(x, 100 * y, color=col, marker=mk, ls=ls, mfc=mfc, label=lab)
        b.fill_between(xl, 100 * lo, 100 * hi, color=col, alpha=0.13, lw=0)
        tops.extend(100 * np.asarray(hi))
    b.set_xlabel("SNR [dB]")
    b.set_ylabel("$P_{\\rm cat}$ [%]")
    b.set_title("(b) catastrophic rate, exact 95% Clopper--Pearson", pad=3)
    rate_ylim(b, tops)
    grid(b)

    for v, lab, col, mk, ls in series:
        mfc = "none" if v == "old_stage1_ccop" else col
        x, y = curve(S, "x_value", "clock_rmse_ns", "variant", v)
        c.semilogy(x, y, color=col, marker=mk, ls=ls, mfc=mfc, label=lab)
        x, y = curve(S, "x_value", "clock_p95_ns", "variant", v)
        c.semilogy(x, y, color=col, marker=mk, mfc="white", ls="--", lw=1.05)
    c.set_xlabel("SNR [dB]")
    c.set_ylabel("Clock error [ns]")
    c.set_title("(c) Clock RMSE and p95", pad=3)
    grid(c)

    for v, lab, col, mk, ls in series:
        mfc = "none" if v == "old_stage1_ccop" else col
        x, y = curve(S, "x_value", "channel_nmse_mean", "variant", v)
        d.semilogy(x, y, color=col, marker=mk, ls=ls, mfc=mfc, label=lab)
    d.set_xlabel("SNR [dB]")
    d.set_ylabel("Channel NMSE")
    d.set_title("(d) Jones-domain channel reconstruction NMSE", pad=3)
    grid(d)

    fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.0)
    fig_legend(fig, a, ncol=5, y=0.005)
    fig.savefig(out / "fig_internal_snr.pdf")
    plt.close(fig)

    # paired deltas ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(DBL_W, 1.85))
    x, y = curve(P, "x_value", "paired_position_rmse_difference_m")
    xl, lo, hi = band(P, "x_value", "paired_position_rmse_difference_ci_low_m",
                      "paired_position_rmse_difference_ci_high_m")
    ax[0].plot(x, y, color=C["proposed"], marker="o")
    ax[0].fill_between(xl, lo, hi, color=C["proposed"], alpha=0.18, lw=0)
    ax[0].axhline(0.0, color="k", lw=0.6, ls="--")
    ax[0].set_xlabel("SNR [dB]")
    ax[0].set_ylabel("$\\Delta$RMSE [m]")
    ax[0].set_title("(a) paired RMSE difference vs. R2 (10$^4$ bootstrap)", pad=3)
    grid(ax[0])

    events = sorted(
        [r for r in P if fnum(r.get("rescued_outliers", 0))
         + fnum(r.get("introduced_outliers", 0)) > 0],
        key=lambda r: fnum(r["x_value"]))
    xpos = np.arange(len(events), dtype=float)
    rescued = [fnum(r["rescued_outliers"]) for r in events]
    introduced = [fnum(r["introduced_outliers"]) for r in events]
    w = 0.34
    ax[1].bar(xpos - w / 2, rescued, width=w, color=C["gi4"], label="rescued")
    ax[1].bar(xpos + w / 2, introduced, width=w, color=C["r2"], label="introduced")
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels([f"{fnum(r['x_value']):g}" for r in events])
    ax[1].set_xlabel("SNR [dB]")
    ax[1].set_ylabel("trials (of 480)")
    ymax = max(rescued + introduced + [1.0])
    for xx, r, rr, ii in zip(xpos, events, rescued, introduced):
        pv = fnum(r.get("mcnemar_exact_p", "nan"))
        ax[1].text(xx, max(rr, ii) + 0.05 * ymax, f"$p={pv:.2g}$",
                   ha="center", fontsize=7.0)
    ax[1].set_ylim(0, 1.25 * ymax)
    ax[1].set_title("(b) McNemar outlier exchange", pad=3)
    ax[1].legend(loc="upper right")
    grid(ax[1], which="major")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(out / "fig_internal_paired.pdf")
    plt.close(fig)


# =========================================================== fig: ablation ==
def fig_ablation(out: pathlib.Path) -> None:
    """Nested ladder at two operating points.

    Inside the operating region the pipeline is no longer initialization
    limited, so every rung sits at the same rate; the ladder only separates
    once the delay poles start to collapse.  Showing one operating point would
    therefore report either a flat panel or a claim outside the declared
    region, so both are drawn: ``-10`` dB is the operating-region result and
    ``-20`` dB exhibits the mechanism.
    """
    A = ds("components", "ablation_summary.csv")
    T = ds("components", "ablation_trials.csv")
    A20 = ds("components_m20", "ablation_summary.csv")
    T20 = ds("components_m20", "ablation_trials.csv")
    order = [
        ("old_stage1_ccop", "R3", "R3: frozen Stage-I", C["r3"]),
        ("mksc_delay_ccop", "A1", "A1: + MKSC delay compression", C["a1"]),
        ("mksc_gi_1_no_refresh_ccop", "A2", "A2: + common geometry (1 start)", C["a2"]),
        ("mksc_gi_4_no_refresh_ccop", "A3", "A3: + 4 deterministic starts", C["gi4"]),
        ("proposed", "Prop.", "Proposed: + anchor refresh", C["proposed"]),
    ]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DBL_W, 2.15))

    xs = np.arange(len(order))
    snr_styles = [(A20, r"$-20$ dB", C["r2"], "s", "--"),
                  (A, r"$-10$ dB", C["proposed"], "o", "-")]
    ceiling = 1.0
    for rows, snr_lab, col, mk, ls in snr_styles:
        vals, lo, hi = [], [], []
        for v, *_ in order:
            r = sel(rows, variant=v)[0]
            vals.append(100 * fnum(r["catastrophic_rate"]))
            lo.append(100 * fnum(r["outlier_ci_low"]))
            hi.append(100 * fnum(r["outlier_ci_high"]))
        vals, lo, hi = map(np.asarray, (vals, lo, hi))
        ceiling = max(ceiling, float(max(hi)))
        a.plot(xs, vals, color=col, marker=mk, ls=ls, label=snr_lab)
        a.fill_between(xs, lo, hi, color=col, alpha=0.10, lw=0)
        for xx, yy in zip(xs, vals):
            a.text(xx, yy + 0.035 * ceiling, f"{yy:.1f}", ha="center", fontsize=6.2)
    vals20 = [100 * fnum(sel(A20, variant=v)[0]["catastrophic_rate"])
              for v, *_ in order]
    for i in range(1, len(vals20)):
        delta = vals20[i] - vals20[i - 1]
        if abs(delta) >= 0.3:
            a.text(i - 0.5, 0.5 * (vals20[i] + vals20[i - 1]),
                   f"{delta:+.1f}", color=C["r2"], fontsize=6.2, ha="center")
    a.set_xticks(xs)
    a.set_xticklabels([o[1] for o in order], fontsize=7)
    a.set_ylabel("$P_{\\rm cat}$ [%]")
    a.set_ylim(0, 1.20 * ceiling)
    a.set_title("(a) Catastrophic-rate ladder", pad=3)
    grid(a, which="major")

    for rows, snr_lab, col, mk, ls in snr_styles:
        vals = [100 * fnum(sel(rows, variant=v)[0]["stage1_basin_acquisition_rate"])
                for v, *_ in order]
        b.plot(xs, vals, color=col, marker=mk, ls=ls)
        for xx, yy in zip(xs, vals):
            b.text(xx, yy + 1.4, f"{yy:.1f}", ha="center", fontsize=6.2)
    b.set_xticks(xs)
    b.set_xticklabels([o[1] for o in order], fontsize=7)
    b.set_ylabel("Stage-I basin acquisition [%]")
    b.set_ylim(0, 116)
    b.set_title("(b) Basin-acquisition ladder", pad=3)
    grid(b, which="major")

    cdf_order = [order[i] for i in (0, 1, 2, 4)]
    for v, short, lab, col in cdf_order:
        for rows, ls, lw in ((T20, "--", 0.9), (T, "-", 1.1)):
            e = sorted(fnum(r["position_error_m"]) for r in sel(rows, variant=v)
                       if r["failed"].lower() == "false")
            e = [x for x in e if math.isfinite(x)]
            if not e:
                continue
            y = np.arange(1, len(e) + 1) / len(e)
            c.semilogx(e, y, color=col, lw=lw, ls=ls)
    c.axvline(0.1, color="k", lw=0.7, ls="--")
    c.text(0.115, 0.10, "10 cm\nthreshold", fontsize=6.0)
    c.set_xlabel("Position error [m]")
    c.set_ylabel("Empirical CDF")
    c.set_ylim(0, 1.02)
    c.set_title("(c) Position-error CDF", pad=3)
    grid(c)

    snr_handles = [Line2D([], [], color=col, marker=mk, ls=ls, label=lab)
                   for _, lab, col, mk, ls in snr_styles]
    step_handles = [Line2D([], [], color=col, lw=1.5, label=short)
                    for _, short, _, col in cdf_order]
    fig.legend(handles=snr_handles + step_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=6, frameon=False,
               fontsize=6.9, columnspacing=0.9, handlelength=1.6)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.26, wspace=0.30)
    fig.savefig(out / "fig_ablation.pdf")
    plt.close(fig)


# =========================================================== fig: receiver ==
def fig_receiver(out: pathlib.Path) -> None:
    S = snr_rows(ds("receiver", "ablation_summary.csv"))
    T = snr_rows(ds("receiver", "ablation_trials.csv"),
                 "snr_db")
    modes = [
        ("scalar", "scalar", C["scalar"], "s", ":"),
        ("dual_pol", "dual-pol", C["dual"], "^", "--"),
        ("full_6d", "full 6-comp. EVS", C["full"], "o", "-"),
    ]

    fig, (a, b) = plt.subplots(1, 2, figsize=(DBL_W, 2.30))
    for m, lab, col, mk, ls in modes:
        x, y = curve(S, "x_value", "position_conditional_rmse_m", "variant", f"proposed_{m}")
        a.semilogy(x, y, color=col, marker=mk, ls="-", label=lab)
        x, y = curve(S, "x_value", "peb_position_m_rms", "variant", f"free_jones_peb_{m}")
        a.semilogy(x, y, color=col, ls="--", lw=1.05)
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Conditional RMSE [m]")
    a.set_title("(a) Matched RMSE and PEB", pad=3)
    # The one place the frozen directional penalty is visible: with a single
    # retained component the two-dimensional Jones anchor is unidentifiable,
    # so the scalar curve leaves its own bound instead of tracking it.
    a.annotate("scalar: regularization\nfloor", xy=(28, 9.4e-5), xytext=(15.5, 6.0e-4),
               fontsize=6.2, ha="center", color=C["scalar"], linespacing=0.95,
               arrowprops=dict(arrowstyle="->", lw=0.5, color=C["scalar"]))
    grid(a)

    TOP = 1.14          # efficiency-axis ceiling; the scalar curve exceeds it
    exits: list[tuple[float, float]] = []
    for m, lab, col, mk, ls in modes:
        xs, rr = [], []
        for s in sorted({fnum(r["x_value"]) for r in S}):
            E = {r["trial_id"]: r for r in T
                 if r["variant"] == f"proposed_{m}" and fnum(r["snr_db"]) == s}
            P = {r["trial_id"]: r for r in T
                 if r["variant"] == f"free_jones_peb_{m}" and fnum(r["snr_db"]) == s}
            ids = [t for t in E if t in P and E[t]["outlier"].lower() == "false"]
            if len(ids) < 30:
                continue
            xs.append(s)
            rr.append(rms([fnum(E[t]["position_error_m"]) for t in ids])
                      / rms([fnum(P[t]["peb_position_m"]) for t in ids]))
        b.plot(xs, rr, color=col, marker=mk, ls="-", label=lab)
        if m == "scalar":
            exits = [(s, r) for s, r in zip(xs, rr) if r > TOP]
    # The claim the abstract makes is "within 2% of the free-Jones reference",
    # which is a band, not a line, so draw it.  Letting the scalar excursion
    # set the range would compress that band to a hairline: the axis is clipped
    # and the scalar curve is reported where it leaves.
    b.axhspan(0.98, 1.02, color="0.45", alpha=0.14, lw=0)
    b.axhline(1.0, color="k", lw=0.8, ls="--")
    b.set_ylim(0.955, TOP)
    if exits:
        b.text(0.5, 0.02,
               rf"scalar leaves the axis: {exits[0][1]:.2f} at "
               rf"{exits[0][0]:.0f} dB, {exits[-1][1]:.2f} at {exits[-1][0]:.0f} dB",
               transform=b.transAxes, ha="center", va="bottom", fontsize=6.3,
               color=C["scalar"])
    b.text(0.985, 1.0, r"$\pm2\%$", transform=b.get_yaxis_transform(),
           va="center", ha="right", fontsize=6.6, color="0.25",
           bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
    b.set_xlabel("SNR [dB]")
    b.set_ylabel("RMSE / PEB (inliers)")
    b.set_title("(b) Efficiency ratio", pad=3)
    grid(b)

    mode_handles = [Line2D([], [], color=col, marker=mk, ls="-", label=lab)
                    for _, lab, col, mk, _ in modes]
    metric_handles = [Line2D([], [], color="0.15", marker="o", ls="-",
                             label="estimator RMSE"),
                      Line2D([], [], color="0.15", ls="--",
                             label="free-Jones PEB")]
    fig.legend(handles=mode_handles + metric_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=5, frameon=False,
               fontsize=7.1, columnspacing=1.0, handlelength=1.8)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.25, wspace=0.25)
    fig.savefig(out / "fig_receiver.pdf")
    plt.close(fig)


# ======================================================== fig: compression ==
def fig_compression(out: pathlib.Path) -> None:
    S = snr_rows(ds("compression", "ablation_summary.csv"))
    T = snr_rows(ds("snr_internal", "ablation_trials.csv"), "snr_db")
    pairs = [
        ("proposed", "MKSC-compressed delay subspace", C["proposed"], "o", "-"),
        ("raw_delay_gi_ccop", "raw (uncompressed) delay subspace", C["raw"], "s", "--"),
    ]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DBL_W, 2.15))
    for v, lab, col, mk, ls in pairs:
        x, y = curve(S, "x_value", "stage1_delay_rmse_ns_median", "variant", v)
        a.semilogy(x, y, color=col, marker=mk, ls="-", label=lab)
        x, y = curve(S, "x_value", "stage1_delay_rmse_ns_p95", "variant", v)
        a.semilogy(x, y, color=col, marker=mk, mfc="white", ls="--", lw=1.05)
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Delay RMSE [ns]")
    a.set_title("(a) Delay median and p95", pad=3)
    clip_ylim(a, (3e-4, 6e1), (3e-4, 4e2))
    # The single factor separates in the tail, not in the middle of the
    # distribution: mark the p95 gap at the low edge of the operating region
    # rather than leaving it to be read off two decades of log axis.
    p95 = {v: dict(zip(*curve(S, "x_value", "stage1_delay_rmse_ns_p95",
                              "variant", v))) for v, *_ in pairs}
    edge = min(p95["proposed"])
    lo_v, hi_v = p95["proposed"][edge], p95["raw_delay_gi_ccop"][edge]
    a.annotate("", xy=(edge + 1.4, lo_v), xytext=(edge + 1.4, hi_v),
               arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.25"))
    a.text(edge + 2.2, math.sqrt(lo_v * hi_v), rf"${hi_v / lo_v:.0f}\times$ at p95",
           fontsize=6.4, va="center", ha="left", color="0.2",
           bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
    grid(a)

    tops = []
    for v, lab, col, mk, ls in pairs:
        x, y = curve(S, "x_value", "catastrophic_rate", "variant", v)
        b.plot(x, 100 * y, color=col, marker=mk, ls=ls, label=lab)
        xl, lo, hi = band(S, "x_value", "outlier_ci_low", "outlier_ci_high",
                          "variant", v)
        b.fill_between(xl, 100 * lo, 100 * hi, color=col, alpha=0.11, lw=0)
        tops.extend(100 * np.asarray(hi))
    b.set_xlabel("SNR [dB]")
    b.set_ylabel("$P_{\\rm cat}$ [%]")
    rate_ylim(b, tops, pad=0.18)
    b.set_title("(b) Catastrophic rate", pad=3)
    grid(b)

    snrs = sorted({fnum(r["snr_db"]) for r in T})
    keep, sig = [], []
    for s in snrs:
        rs = [r for r in T if r["variant"] == "proposed" and fnum(r["snr_db"]) == s]
        keep.append(st.median([fnum(r["stage1_evs_retained_energy_fraction"]) for r in rs]))
        sig.append(st.median([fnum(r["stage1_delay_sigma_kplus1_over_k"]) for r in rs]))
    c.plot(snrs, keep, color=C["proposed"], marker="o")
    c.plot(snrs, sig, color=C["gi4"], marker="v", ls="--")
    c.annotate("retained energy\nfraction", xy=(10, 0.915), xytext=(12, 0.52),
               fontsize=6.0, ha="center", color=C["proposed"],
               arrowprops=dict(arrowstyle="->", lw=0.5, color=C["proposed"]))
    sig_text = (-4.5, 0.72) if SNR_MIN >= -10.0 else (-14, 0.28)
    c.annotate(r"$\sigma_{K+1}/\sigma_K$", xy=(-5, 0.34), xytext=sig_text,
               fontsize=6.6, ha="center", color=C["gi4"],
               arrowprops=dict(arrowstyle="->", lw=0.5, color=C["gi4"]))
    c.axhline(6 / 96, color="k", lw=0.7, ls=":")
    c.text(21, 0.105, "$r/I=1/16$\n(noise floor)", fontsize=6.0, ha="center")
    c.set_xlabel("SNR [dB]")
    c.set_ylabel("fraction")
    c.set_ylim(0, 1.06)
    c.set_title("(c) Subspace diagnostics", pad=3)
    grid(c)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.28, wspace=0.34)
    fig_legend(fig, a, ncol=2, y=0.035)
    fig.savefig(out / "fig_compression.pdf")
    plt.close(fig)


# ========================================================= fig: benchmark ===
def fig_benchmark(out: pathlib.Path) -> None:
    B = snr_rows(
        ds("benchmark", "benchmark_summary.csv"),
        "snr_db")
    series = [
        ("mksc_ccop", "Proposed", C["proposed"], "o", "-"),
        ("scaled_4d", "4-D VP", C["r2"], "s", ":"),
        ("als_cpd", "ALS-CPD", C["als"], "D", "--"),
        ("ris_vbi_sbl", "VBI/SBL", C["vbi"], "P", "-."),
        ("nf_ris_groupomp_localgrid_wls", "OMP-SAGE-WLS", C["omp"], "X", "--"),
    ]

    # Per-trial errors are needed for the p99 panel: the summary carries only
    # interpolated percentiles, and p99 is exactly where that convention
    # disagrees with the text (see ``nearest_rank``).
    BT = snr_rows(ds("benchmark", "benchmark_trials.csv"), "snr_db")
    snr_grid = sorted({fnum(r["snr_db"]) for r in BT})

    def quantile_curve(base: str, q: float):
        xs, ys = [], []
        for s in snr_grid:
            e = [fnum(r["position_error_m"]) for r in BT
                 if r["baseline"] == base and fnum(r["snr_db"]) == s]
            if e:
                xs.append(s)
                ys.append(nearest_rank(e, q))
        return np.array(xs), np.array(ys)

    # The three columns answer three different questions in the same order on
    # both rows: how precise inside the basin (left), how heavy the tail
    # (middle), how often the basin is lost (right).
    fig, axes = plt.subplots(2, 3, figsize=(DBL_W, 3.95))
    (a, b, c), (d, e, f) = axes

    for v, lab, col, mk, ls in series:
        x, y = curve(B, "snr_db", "position_error_m_median", "baseline", v)
        a.semilogy(x, 1e3 * y, color=col, marker=mk, ls=ls, label=lab)
    x, y = curve(B, "snr_db", "peb_position_m_rms", "baseline", "peb")
    a.semilogy(x, 1e3 * y, color="k", ls=(0, (4, 2)), lw=1.0, label="Free-Jones PEB")
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Error [mm]")
    a.set_title("(a) Position, median", pad=3)
    grid(a)

    for v, lab, col, mk, ls in series:
        x, y = quantile_curve(v, 99.0)
        b.semilogy(x, 1e3 * y, color=col, marker=mk, ls=ls, label=lab)
    b.set_xlabel("SNR [dB]")
    b.set_ylabel("Error [mm]")
    b.set_title("(b) Position, p99", pad=3)
    # p99 is the tenth-largest of 960 trials, so it enters the wrong-basin
    # population as soon as that mass exceeds ~1%: the decades between the
    # proposed curve and the rest are wrong-basin mass, not precision.
    b.axhspan(1e-4, 100.0, color=C["proposed"], alpha=0.055, lw=0)
    b.text(0.03, 0.05, r"inlier region ($\leq 0.1$ m)", transform=b.transAxes,
           ha="left", fontsize=6.2, color="0.35")
    grid(b)

    # The rates span two decades here -- ALS-CPD holds a ~30% mass at every
    # SNR while three routes sit at or below one trial per cell -- so a linear
    # axis would collapse the interesting part.  Exact zeros are drawn at the
    # floor with a hollow marker and their interval still starts there.
    # Cell size is read from the data: the benchmark runs at a different trial
    # count from the other suites, so a hardcoded reference line would lie.
    n_cell = max(int(fnum(r["n"])) for r in B if math.isfinite(fnum(r.get("n"))))
    # Exact zeros are parked half a trial below the one-trial line, so "no
    # failure" stays visibly distinct from "one failure" at any cell size.  A
    # fixed 0.1 floor collided with the one-trial line once n reached 960.
    floor = 100.0 / n_cell / 2.0
    for v, lab, col, mk, ls in series:
        x, y = curve(B, "snr_db", "catastrophic_rate", "baseline", v)
        pct_ = np.clip(100 * y, floor, None)
        c.semilogy(x, pct_, color=col, marker=mk, ls=ls, label=lab)
        zero = 100 * y <= 0.0
        if zero.any():
            c.semilogy(x[zero], np.full(int(zero.sum()), floor), color=col,
                       marker=mk, ls="none", mfc="white", ms=3.6)
        xl, lo, hi = band(B, "snr_db", "outlier_ci_low", "outlier_ci_high", "baseline", v)
        c.fill_between(xl, np.clip(100 * lo, floor, None),
                       np.clip(100 * hi, floor, None), color=col, alpha=0.10, lw=0)
    c.axhline(100.0 / n_cell, color="k", lw=0.6, ls=":")
    c.text(0.97, 100.0 / n_cell * 1.25, f"1 trial in {n_cell}", fontsize=5.8,
           ha="right", va="bottom", transform=c.get_yaxis_transform())
    c.set_xlabel("SNR [dB]")
    c.set_ylabel("$P_{\\rm cat}$ [%]")
    c.set_ylim(floor * 0.75, 90)
    c.set_yticks([floor, 1, 10])
    c.set_yticklabels(["$0$", "1", "10"])
    c.set_title(r"(c) Position, $P_{\rm cat}$", pad=3)
    grid(c, which="major")

    # Clock median only: the p95 of every route is tabulated in the supplement,
    # and drawing ten curves in one column-width panel cost more than it said.
    # Panel (d) instead gains the reference the position panels already had.
    for v, lab, col, mk, ls in series:
        med_col = ("clock_error_ns_median" if v in ("mksc_ccop", "scaled_4d")
                   else "clock_median_abs_error_ns")
        x, y = curve(B, "snr_db", med_col, "baseline", v)
        d.semilogy(x, 1e3 * y, color=col, marker=mk, ls=ls, label=lab)
    xp, yp = curve(B, "snr_db", "peb_position_m_rms", "baseline", "peb")
    if xp.size:
        anchor = yp[np.argmin(np.abs(xp + 10.0))]
        d.semilogy(xp, CEB_REF_PS_AT_MINUS10 * yp / anchor, color="k",
                   ls=(0, (4, 1.4, 1, 1.4)), lw=1.0,
                   label=r"CEB ($\sigma$-scaled)")
    d.set_xlabel("SNR [dB]")
    d.set_ylabel("Error [ps]")
    d.set_title("(d) Clock, median", pad=3)
    grid(d)

    for v, lab, col, mk, ls in series:
        x, y = curve(B, "snr_db", "y_nmse_mean", "baseline", v)
        e.semilogy(x, y, color=col, marker=mk, ls=ls, label=lab)
    e.set_xlabel("SNR [dB]")
    e.set_ylabel("NMSE")
    e.set_title("(e) Observation NMSE, mean", pad=3)
    grid(e)

    # Same construction as (c) so the two rate panels read identically: log
    # axis, exact zeros parked half a trial below the one-trial line.
    for v, lab, col, mk, ls in series:
        x, y = curve(B, "snr_db", "clock_catastrophic_rate", "baseline", v)
        f.semilogy(x, np.clip(100 * y, floor, None), color=col, marker=mk, ls=ls)
        zero = 100 * y <= 0.0
        if zero.any():
            f.semilogy(x[zero], np.full(int(zero.sum()), floor), color=col,
                       marker=mk, ls="none", mfc="white", ms=3.6)
        xl, lo, hi = band(B, "snr_db", "clock_catastrophic_ci_low",
                          "clock_catastrophic_ci_high", "baseline", v)
        f.fill_between(xl, np.clip(100 * lo, floor, None),
                       np.clip(100 * hi, floor, None), color=col, alpha=0.10, lw=0)
    f.axhline(100.0 / n_cell, color="k", lw=0.6, ls=":")
    f.set_xlabel("SNR [dB]")
    f.set_ylabel(r"$P_{\rm cat}^{\rm clk}$ [%]")
    f.set_ylim(floor * 0.75, 30)
    f.set_yticks([floor, 1, 10])
    f.set_yticklabels(["$0$", "1", "10"])
    f.set_title(r"(f) Clock, $P_{\rm cat}$ ($>1$ ns)", pad=3)
    grid(f, which="major")

    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.175,
                        wspace=0.38, hspace=0.62)
    ceb_handle = [(Line2D([], [], color="k", ls=(0, (4, 1.4, 1, 1.4)), lw=1.0),
                   r"CEB ($\sigma$-scaled)")]
    fig_legend(fig, a, ncol=3, y=0.038, extra=ceb_handle)
    fig.savefig(out / "fig_benchmark.pdf")
    plt.close(fig)


# =================================================== fig: benchmark clock ===
def fig_benchmark_clock(out: pathlib.Path) -> None:
    """External synchronization comparison.

    Proposed / R2 clock statistics come from the position benchmark summary
    columns (``clock_error_ns_{median,p95}``); the three external routes come
    from the paired ``benchmark_clock_external_480`` suite, which shares the
    same noise realizations (identical ``y_noisy_hash``).  The clock
    catastrophic rate (|error| > 1 ns) is a released column of the external
    suite and, for the two internal routes, is counted from their per-trial
    ``clock_error_ns`` column of the same shared realizations.
    """
    Bs = snr_rows(
        ds("benchmark", "benchmark_summary.csv"),
        "snr_db")
    Bt = snr_rows(
        ds("benchmark", "benchmark_trials.csv"),
        "snr_db")
    Cx = snr_rows(ds("benchmark_clock", "benchmark_summary.csv"),
                  "snr_db")

    # (proposed, R2) internal: median/p95 from summary [ns]; externals: from Cx.
    internal = [
        ("mksc_ccop", "Proposed", C["proposed"], "o", "-"),
        ("scaled_4d", "4-D VP", C["r2"], "s", ":"),
    ]
    external = [
        ("als_cpd", "ALS-CPD", C["als"], "D", "--"),
        ("ris_vbi_sbl", "VBI/SBL", C["vbi"], "P", "-."),
        ("nf_ris_groupomp_localgrid_wls", "OMP-SAGE-WLS", C["omp"], "X", "--"),
    ]

    def clk_cat_internal(base):
        xs = sorted({fnum(r["snr_db"]) for r in Bt})
        out_x, out_y = [], []
        for s in xs:
            e = [fnum(r["clock_error_ns"]) for r in Bt
                 if r["baseline"] == base and fnum(r["snr_db"]) == s
                 and r["clock_error_ns"] not in ("", "nan")]
            if not e:
                continue
            out_x.append(s)
            out_y.append(100.0 * sum(1 for v in e if v > 1.0) / len(e))
        return np.array(out_x), np.array(out_y)

    fig, (a, b) = plt.subplots(1, 2, figsize=(DBL_W, 2.25))

    for v, lab, col, mk, ls in internal:
        x, y = curve(Bs, "snr_db", "clock_error_ns_median", "baseline", v)
        a.semilogy(x, 1e3 * y, color=col, marker=mk, ls="-", label=lab)
        x, y = curve(Bs, "snr_db", "clock_error_ns_p95", "baseline", v)
        a.semilogy(x, 1e3 * y, color=col, marker=mk, mfc="white", ls="--", lw=1.05)
    for v, lab, col, mk, ls in external:
        x, y = curve(Cx, "snr_db", "clock_median_abs_error_ns", "baseline", v)
        a.semilogy(x, 1e3 * y, color=col, marker=mk, ls="-", label=lab)
        x, y = curve(Cx, "snr_db", "clock_p95_abs_error_ns", "baseline", v)
        a.semilogy(x, 1e3 * y, color=col, marker=mk, mfc="white", ls="--", lw=1.05)
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Clock error [ps]")
    a.set_title("(a) Clock median and p95", pad=3)
    grid(a)

    for v, lab, col, mk, ls in internal:
        x, y = clk_cat_internal(v)
        b.plot(x, y, color=col, marker=mk, ls=ls, label=lab)
    for v, lab, col, mk, ls in external:
        x, y = curve(Cx, "snr_db", "clock_catastrophic_rate", "baseline", v)
        b.plot(x, 100 * y, color=col, marker=mk, ls=ls, label=lab)
        xl, lo, hi = band(Cx, "snr_db", "clock_catastrophic_ci_low",
                          "clock_catastrophic_ci_high", "baseline", v)
        b.fill_between(xl, 100 * lo, 100 * hi, color=col, alpha=0.10, lw=0)
    b.set_xlabel("SNR [dB]")
    b.set_ylabel("$P_{\\rm cat}^{\\rm clk}$ [\\%] ($>1$ ns)")
    clip_ylim(b, (-1, 24), (-3, 55))
    b.set_title("(b) Clock-catastrophic rate", pad=3)
    grid(b)

    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig_legend(fig, a, ncol=3, y=0.02)
    fig.savefig(out / "fig_benchmark_clock.pdf")
    plt.close(fig)


# ======================================================== fig: robustness ===
def fig_robustness(out: pathlib.Path) -> None:
    M = ds("maxwell_mismatch", "robustness_summary.csv")
    N = ds("colored_noise", "robustness_summary.csv")
    F8 = ds("ris_calibration", "fig8_calibration_summary.csv")
    F9 = ds("model_order", "fig9_k_mismatch_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(DBL_W, 4.0))
    (a, b), (c, d) = axes

    for v, lab, col, mk, ls in [("proposed", "MKSC compressed", C["proposed"], "o", "-"),
                                ("raw_delay_gi_ccop", "raw delay subspace", C["raw"], "s", "--")]:
        rs = sel(M, x_name="ris_bs_angle_deg", variant=v)
        x, y = curve(rs, "x_value", "catastrophic_rate")
        a.plot(x, 100 * y, color=col, marker=mk, ls="-", label=lab)
        x, y = curve(rs, "x_value", "stage1_basin_acquisition_rate")
        a.plot(x, 100 * y, color=col, marker=mk, mfc="white", ls="--", lw=1.05)
    a.set_xlabel("RIS--BS incidence-angle error [deg]")
    a.set_ylabel("$P_{\\rm cat}$ / basin rate [%]")
    a.set_title("(a) RIS--BS angle mismatch", pad=3)
    h, l = a.get_legend_handles_labels()
    h += [Line2D([], [], color="0.15", marker="o", ls="-", label=r"$P_{\rm cat}$"),
          Line2D([], [], color="0.15", marker="o", mfc="white", ls="--",
                 label="basin acquisition")]
    l += [r"$P_{\rm cat}$", "basin acquisition"]
    a.legend(h, l, loc="center left", fontsize=6.7)
    grid(a)

    styles = {"evs_gain_std": ("EVS gain, 0--0.1", C["gi4"], "v", "-"),
              "evs_phase_deg": (r"EVS phase, 0--10$^\circ$", C["r3"], "^", "--"),
              "bs_sensor_position_std_mm": ("sensor position, 0--0.5 mm", C["a2"], "D", "-.")}
    for xn, (lab, col, mk, ls) in styles.items():
        rs = sel(M, x_name=xn, variant="proposed")
        x, y = curve(rs, "x_value", "catastrophic_rate")
        xn_norm = x / max(x.max(), 1e-12)
        b.plot(xn_norm, 100 * y, color=col, marker=mk, ls=ls, label=lab)
    b.set_xlabel("mismatch level, normalized to sweep maximum")
    b.set_ylabel("$P_{\\rm cat}$ [%]")
    rate_ylim(b, [100 * fnum(r["catastrophic_rate"])
                  for xn in styles for r in sel(M, x_name=xn, variant="proposed")],
              pad=0.25)
    b.set_title("(b) Receiver calibration", pad=3)
    b.legend(loc="upper left", fontsize=6.7)
    grid(b)

    for v, lab, col, mk, ls in [("proposed", "MKSC compressed", C["proposed"], "o", "-"),
                                ("raw_delay_gi_ccop", "raw delay subspace", C["raw"], "s", "--")]:
        rs = sel(N, variant=v)
        x, y = curve(rs, "x_value", "catastrophic_rate")
        c.plot(x, 100 * y, color=col, marker=mk, ls=ls, label=lab)
    c.set_xlabel(r"noise correlation coefficient $\rho$")
    c.set_ylabel("$P_{\\rm cat}$ [%]")
    c.set_title("(c) Unwhitened colored noise", pad=3)
    c.legend(loc="upper left", fontsize=6.7)
    grid(c)

    rs = [r for r in F9 if r["baseline"] == "mksc_ccop"]
    x, y = curve(rs, "x_value", "outlier_rate")
    d.bar(x, 100 * y, width=0.32, color=C["proposed"], alpha=0.25,
          edgecolor=C["proposed"], label="$P_{\\rm cat}$ (left)")
    d.plot(x, 100 * y, color=C["proposed"], marker="o", ls="none")
    d.set_xlabel(r"assumed model order $\hat K$ (true $K=3$)")
    d.set_ylabel("$P_{\\rm cat}$ [%]")
    d.set_xticks([2, 3, 4, 5])
    grid(d)
    d2 = d.twinx()
    x, y = curve(rs, "x_value", "median_m")
    d2.semilogy(x, y, color=C["gi4"], marker="v", ls="none", label="median error (right)")
    x, y = curve(rs, "x_value", "p95_m")
    d2.semilogy(x, y, color=C["r2"], marker="s", mfc="white", ls="none",
                 label="p95 error (right)")
    d2.set_ylabel("position error [m]")
    d.set_title("(d) Model-order mismatch", pad=3)
    h1, l1 = d.get_legend_handles_labels()
    h2, l2 = d2.get_legend_handles_labels()
    d.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=6.7)

    fig.tight_layout(pad=0.4, w_pad=1.4, h_pad=1.0)
    fig.savefig(out / "fig_robustness.pdf")
    plt.close(fig)

    # RIS element-phase calibration -----------------------------------------
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    for v, lab, col, mk, ls in [("mksc_ccop", "full estimator (Stage-I + CCOP-JVP)", C["proposed"], "o", "-"),
                                ("stage1_only", "Stage-I output only", C["r2"], "s", "--")]:
        rs = [r for r in F8 if r["baseline"] == v]
        x, y = curve(rs, "x_value", "median_m")
        ax.semilogy(x, y, color=col, marker=mk, ls=ls, label=lab)
        x, y = curve(rs, "x_value", "p95_m")
        ax.semilogy(x, y, color=col, marker=mk, mfc="white", ls="--", lw=1.05)
    ax.set_xlabel("RIS element-phase calibration error std [deg]")
    ax.set_ylabel("position error [m]")
    ax.set_title("RIS element-phase sensitivity", pad=3)
    ax.legend(loc="center left", fontsize=6.7)
    grid(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_ris_calibration.pdf")
    plt.close(fig)


# ================================================ fig: main boundaries ====
def fig_boundaries(out: pathlib.Path) -> None:
    """Main-text claim boundaries, with one panel per scientific question.

    Mild receiver mismatch, colored noise, and resource scaling remain in the
    supplement.  The main figure instead aligns directly with the three
    limitations stated in the conclusion: angular-calibration bias, known
    model order, and acquisition failure near delay coincidence despite a
    full-rank local EFIM.
    """
    M = ds("maxwell_mismatch", "robustness_summary.csv")
    F9 = ds("model_order", "fig9_k_mismatch_summary.csv")
    G = ds("positions", "position_generalization_summary.csv")
    Tp = [r for r in ds("positions", "robustness_trials.csv")
          if r["variant"] == "proposed"]

    fig, axes = plt.subplots(2, 2, figsize=(DBL_W, 3.35))
    (a, b), (c, d) = axes

    angle = sel(M, x_name="ris_bs_angle_deg", variant="proposed")
    x, y = curve(angle, "x_value", "position_conditional_rmse_m")
    a.semilogy(x, 1e3 * y, color=C["proposed"], marker="o", ls="-",
               label="conditional RMSE")
    x, y = curve(angle, "x_value", "position_p95_m")
    a.semilogy(x, 1e3 * y, color=C["r2"], marker="s", mfc="white", ls="--",
               label="p95 (all trials)")
    a.axvspan(0.1, 0.25, color=C["r3"], alpha=0.08, lw=0)
    a.text(0.028, 260.0, "capture retained,\nprecision lost", ha="left",
           va="top", fontsize=6.8, color="0.35", linespacing=0.95)
    a.set_xlabel("RIS--BS angle error [deg]")
    a.set_ylabel("Error [mm]")
    a.set_title(r"(a) Calibration accuracy floor ($-10$ dB)", pad=3)
    a.legend(loc="lower right", fontsize=7.0)
    grid(a)

    x, y = curve(angle, "x_value", "catastrophic_rate")
    b.plot(x, 100 * y, color=C["proposed"], marker="o", ls="-",
           label=r"$P_{\rm cat}$")
    x, y = curve(angle, "x_value", "stage1_basin_acquisition_rate")
    b.plot(x, 100 * y, color=C["r3"], marker="^", mfc="white", ls="--",
           label="basin acquisition")
    b.set_xlabel("RIS--BS angle error [deg]")
    b.set_ylabel("Rate [%]")
    b.set_ylim(-3, 104)
    b.set_title(r"(b) Calibration capture boundary ($-10$ dB)", pad=3)
    b.legend(loc="center left", fontsize=7.0)
    grid(b)

    order = [r for r in F9 if r["baseline"] == "mksc_ccop"]
    x, y = curve(order, "x_value", "outlier_rate")
    c.bar(x, 100 * y, width=0.32, color=C["proposed"], alpha=0.25,
          edgecolor=C["proposed"], label=r"$P_{\rm cat}$")
    c.plot(x, 100 * y, color=C["proposed"], marker="o", ls="none")
    c.set_xlabel(r"Assumed order $\widehat K$ (true $K=3$)")
    c.set_ylabel(r"$P_{\rm cat}$ [%]")
    c.set_xticks([2, 3, 4, 5])
    # Headroom for the legend: at the frozen 72% ceiling the K=2 bar ran into
    # it.  The order sweep is the one panel of this figure measured at 0 dB.
    c.set_ylim(-2, 96)
    c2 = c.twinx()
    x, y = curve(order, "x_value", "median_m")
    c2.semilogy(x, y, color=C["gi4"], marker="v", ls="none",
                label="median error")
    x, y = curve(order, "x_value", "p95_m")
    c2.semilogy(x, y, color=C["r2"], marker="s", mfc="white", ls="none",
                label="p95 error")
    c2.set_ylabel("Error [m]", labelpad=1)
    c2.set_ylim(5e-5, 40.0)
    c.set_title(r"(c) Model-order mismatch ($0$ dB)", pad=3)
    h1, l1 = c.get_legend_handles_labels()
    h2, l2 = c2.get_legend_handles_labels()
    c.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=6.6, ncol=3,
             columnspacing=0.9, handletextpad=0.4, borderpad=0.25)
    grid(c)

    # True cascaded-delay separations are derived only from the released v3
    # position coordinates and resolved geometry.  The common clock cancels
    # in pairwise differences.
    import json

    cfg_path = DATA["positions"][0] / "resolved_base_config.json"
    cfg = json.loads(cfg_path.read_text())["resolved_config"]
    ris = np.asarray(cfg["ris_centers"], float)
    bs = np.asarray(cfg["p_B"], float)
    c0 = float(cfg["c0"])

    by_pos: dict[str, list[dict]] = {}
    for r in Tp:
        by_pos.setdefault(r["position_index"], []).append(r)
    diagnostics = {}
    defic = 0
    for idx, rows in by_pos.items():
        diagnostics[idx] = st.median(
            [fnum(r["efim_condition_number"]) for r in rows
             if math.isfinite(fnum(r["efim_condition_number"]))])
        defic += sum(r["efim_rank_deficient"].lower() in ("true", "1")
                     for r in rows)

    def delay_sep_ns(row):
        p = np.array([fnum(row["p_true_x"]), fnum(row["p_true_y"]),
                      fnum(row["p_true_z"])])
        lengths = np.linalg.norm(ris - p[None, :], axis=1)
        lengths += np.linalg.norm(ris - bs[None, :], axis=1)
        tau = np.sort(lengths / c0 * 1e9)
        return float(np.min(np.diff(tau)))

    prop = sorted(sel(G, variant="proposed"), key=lambda r: int(r["position_index"]))
    r2 = sorted(sel(G, variant="scaled_4d"), key=lambda r: int(r["position_index"]))
    xp = np.array([delay_sep_ns(r) for r in prop])
    yp = np.array([100 * fnum(r["catastrophic_rate"]) for r in prop])
    cp = np.array([diagnostics[r["position_index"]] for r in prop])
    xr = np.array([delay_sep_ns(r) for r in r2])
    yr = np.array([100 * fnum(r["catastrophic_rate"]) for r in r2])
    d.axvspan(0, 1.2, color=C["r2"], alpha=0.07, lw=0)
    sc = d.scatter(xp, yp, c=cp, cmap="viridis", marker="o", s=27,
                   edgecolor="0.2", linewidth=0.35, label="Proposed", zorder=3)
    d.scatter(xr, yr, facecolors="none", edgecolors=C["r2"], marker="s",
              s=25, linewidth=0.9, label="4-D VP", zorder=2)
    d.axvline(1.2, color=C["r2"], lw=0.8, ls="--")
    d.text(0.99, 0.60, f"free-Jones EFIM full rank\nin {len(Tp) - defic}/{len(Tp)} trials",
           ha="right", va="top", transform=d.transAxes, fontsize=6.4,
           linespacing=0.95, color="0.25")
    # Anchor the grid against the scene every other figure and table uses, so
    # the reader can place the reported operating point on this boundary.
    sc_ref = _resolved_scene()
    tau_ref = np.sort((np.linalg.norm(sc_ref["ris"] - sc_ref["ue"][None, :], axis=1)
                       + np.linalg.norm(sc_ref["ris"] - sc_ref["bs"][None, :], axis=1))
                      / sc_ref["c0"] * 1e9)
    dtau_ref = float(np.min(np.diff(tau_ref)))
    d.axvline(dtau_ref, color="0.25", lw=0.8, ls="-.")
    d.text(dtau_ref + 0.13, 0.955, "reference scene", rotation=90, va="top",
           ha="left", fontsize=6.0, color="0.25", transform=d.get_xaxis_transform())
    d.set_xlabel("Minimum true delay separation [ns]")
    d.set_ylabel(r"$P_{\rm cat}$ [%]", labelpad=1)
    d.set_xlim(left=0)
    d.set_title(r"(d) Delay-coincidence boundary ($-10$ dB)", pad=3)
    d.legend(loc="upper right", fontsize=7.0)
    cb = fig.colorbar(sc, ax=d, fraction=0.047, pad=0.025)
    cb.set_label("EFIM condition number", fontsize=7.2)
    cb.ax.tick_params(labelsize=7)
    grid(d)

    fig.subplots_adjust(left=0.08, right=0.965, top=0.94, bottom=0.12,
                        wspace=0.48, hspace=0.54)
    fig.savefig(out / "fig_boundaries.pdf")
    plt.close(fig)


# =========================================== fig: geometry gen. / scaling ===
def fig_generalization(out: pathlib.Path) -> None:
    G = ds("positions", "position_generalization_summary.csv")
    Rv = ds("resolvability", "evs_resolvability_summary.csv")
    S = ds("scaling", "robustness_summary.csv")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DBL_W, 2.2))

    for v, lab, col, mk in [("proposed", "Proposed", C["proposed"], "o"),
                            ("scaled_4d", "R2: 4-D Jones-VP", C["r2"], "s")]:
        rs = sel(G, variant=v)
        e = sorted(100 * fnum(r["catastrophic_rate"]) for r in rs)
        y = np.arange(1, len(e) + 1) / len(e)
        a.plot(e, y, color=col, marker=mk, ms=2.2, lw=1.0, label=lab)
    a.set_xlabel("per-position $P_{\\rm cat}$ [%]")
    a.set_ylabel("CDF over 50 UE positions")
    a.set_title("(a) geometry generalization", pad=3)
    a.legend(loc="lower right", fontsize=6.0)
    grid(a)

    Tp = [r for r in ds("positions", "robustness_trials.csv")
          if r["variant"] == "proposed"]
    by_pos: dict[str, list[dict]] = {}
    for r in Tp:
        by_pos.setdefault(r["position_index"], []).append(r)
    lam, cond, defic, ntr = [], [], 0, 0
    for rs in by_pos.values():
        lam.append(st.median([fnum(r["efim_min_eigenvalue"]) for r in rs
                              if r["efim_min_eigenvalue"] not in ("", "nan")]))
        cond.append(st.median([fnum(r["efim_condition_number"]) for r in rs
                               if r["efim_condition_number"] not in ("", "nan")]))
        defic += sum(1 for r in rs if r["efim_rank_deficient"].lower() in ("true", "1"))
        ntr += len(rs)
    idx = np.argsort(lam)
    xs = np.arange(1, len(lam) + 1)
    b.semilogy(xs, np.array(lam)[idx], color=C["proposed"], marker="o", ms=2.4, lw=0.9,
               label=r"$\lambda_{\min}(\mathbf{J}^{\rm free}_{\chi})$ (left)")
    b.set_xlabel("UE position, sorted by $\\lambda_{\\min}$")
    b.set_ylabel(r"$\lambda_{\min}$ of free-Jones EFIM")
    b.set_ylim(0.5 * min(lam), 4.0 * max(lam))
    grid(b)
    b2 = b.twinx()
    b2.plot(xs, np.array(cond)[idx], color=C["r3"], marker="^", ms=2.4, lw=0.9, ls="--",
            label="condition number (right)")
    b2.set_ylabel("EFIM condition number")
    b2.set_ylim(0, 1.35 * max(cond))
    b.set_title("(b) local identifiability over the grid", pad=3)
    b.text(0.97, 0.05, f"rank-deficient trials: {defic}/{ntr}", fontsize=6.0,
           ha="right", transform=b.transAxes)
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.0)

    axes_map = [("N", "$N$", C["proposed"], "o", "-"),
                ("T", "$T$", C["gi4"], "v", "--"),
                ("M_A", "$M_A$", C["r3"], "^", "-."),
                ("M_R", "$M_R$", C["a2"], "D", ":")]
    reference = {"N": 63.0, "T": 256.0, "M_A": 16.0, "M_R": 4096.0}
    for xn, lab, col, mk, ls in axes_map:
        rs = sel(S, x_name=xn, variant="proposed")
        x, y = curve(rs, "x_value", "catastrophic_rate")
        c.semilogx(x / reference[xn], 100 * y, color=col, marker=mk, ls="-", label=lab)
        rs = sel(S, x_name=xn, variant="scaled_4d")
        x, y = curve(rs, "x_value", "catastrophic_rate")
        c.semilogx(x / reference[xn], 100 * y, color=col, marker=mk,
                   mfc="white", ls="--", lw=1.05)
    c.axvline(1.0, color="0.25", lw=0.7, ls=":")
    c.set_xlabel("resource / reference value")
    c.set_ylabel("$P_{\\rm cat}$ [%]")
    c.set_title("(c) Relative resource scaling", pad=3)
    c.legend(loc="upper right", ncol=2, fontsize=6.7)
    grid(c)

    fig.tight_layout(pad=0.4, w_pad=1.3)
    fig.savefig(out / "fig_generalization.pdf")
    plt.close(fig)

    # resolvability heat maps ------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(DBL_W, 2.10), sharey=True)
    seps = sorted({fnum(r["target_delay_separation_ns"]) for r in Rv})
    ovs = sorted({fnum(r["polarization_overlap_target"]) for r in Rv})

    # Where the reference scene sits on this map.  Without it the reader cannot
    # tell whether the operating point is in the resolvable region or on the
    # cliff; it is recomputed from the released geometry so it cannot drift.
    sc = _resolved_scene()
    tau_ref = np.sort((np.linalg.norm(sc["ris"] - sc["ue"][None, :], axis=1)
                       + np.linalg.norm(sc["ris"] - sc["bs"][None, :], axis=1))
                      / sc["c0"] * 1e9)
    dtau_ref = float(np.min(np.diff(tau_ref)))
    lo = max(i for i, s in enumerate(seps) if s <= dtau_ref)
    x_ref = lo + (math.log(dtau_ref) - math.log(seps[lo])) / (
        math.log(seps[lo + 1]) - math.log(seps[lo])) if lo + 1 < len(seps) else lo
    for ax, (m, lab) in zip(axs, [("scalar", "(c) Scalar"),
                                  ("dual_pol", "(d) Dual-polarized"),
                                  ("full_6d", "(e) Full EVS")]):
        Z = np.full((len(ovs), len(seps)), np.nan)
        for i, o in enumerate(ovs):
            for j, s in enumerate(seps):
                rr = [r for r in Rv if r["receiver_mode"] == m
                      and abs(fnum(r["polarization_overlap_target"]) - o) < 1e-9
                      and abs(fnum(r["target_delay_separation_ns"]) - s) < 1e-9]
                if rr:
                    Z[i, j] = 100 * fnum(rr[0]["resolution_probability"])
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(len(seps)))
        ax.set_xticklabels([f"{s:g}" for s in seps], fontsize=6.2)
        ax.set_yticks(range(len(ovs)))
        ax.set_yticklabels([f"{o:g}" for o in ovs], fontsize=6.2)
        ax.set_xlabel(r"$\Delta\tau$ [ns]")
        ax.set_title(lab, pad=3)
        for i in range(len(ovs)):
            for j in range(len(seps)):
                if math.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center",
                            fontsize=6.5, color="w" if Z[i, j] < 60 else "k")
        ax.axvline(x_ref, color="w", lw=1.1, ls="--", zorder=6)
        ax.axvline(x_ref, color="0.15", lw=0.5, ls="--", zorder=7)
    # A header strip above the maps carries the operating-point marker, so it
    # never has to be drawn over a cell value or a panel title.
    axs[0].set_ylim(-0.5, len(ovs) + 0.10)
    axs[-1].annotate(rf"$\Delta\tau_{{\min}}$ of the reference scene "
                     rf"(${dtau_ref:.2f}$ ns)",
                     xy=(x_ref, len(ovs) - 0.45), xytext=(x_ref - 0.12, len(ovs) - 0.15),
                     ha="right", va="center", fontsize=5.8, color="0.15",
                     arrowprops=dict(arrowstyle="->", lw=0.5, color="0.15"))
    axs[0].set_ylabel("Jones overlap")
    fig.subplots_adjust(left=0.07, right=0.89, top=0.87, bottom=0.235, wspace=0.12)
    cax = fig.add_axes([0.915, 0.235, 0.014, 0.605])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("resolution probability [%]", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(out / "fig_resolvability.pdf")
    plt.close(fig)


# ============================================================== fig: cost ===
def fig_cost(out: pathlib.Path) -> None:
    T1 = ds("benchmark_runtime", "table1_runtime_memory.csv")
    A = ds("components_cost", "ablation_trials.csv")
    B = ds("benchmark", "benchmark_summary.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(DBL_W, 2.1))

    blocks = [("mksc_projection_runtime_s", "MKSC projection", "#9ecae1"),
              ("stage1_core_s", "Hankel / delay / coupled LS / assignment", "#4292c6"),
              ("common_geometry_runtime_s", "common-geometry init", "#2E8B57"),
              ("anchor_refresh_runtime_s", "anchor refresh", "#E08A00"),
              ("stage3_runtime_s", "CCOP-JVP refinement", "#C8102E")]
    variants = [("old_stage1_ccop", "R3"), ("mksc_delay_ccop", "A1"),
                ("mksc_gi_1_no_refresh_ccop", "A2"), ("mksc_gi_4_no_refresh_ccop", "A3"),
                ("proposed", "Proposed")]
    totals: list[float] = []
    for i, (v, lab) in enumerate(variants):
        rs = sel(A, variant=v)
        vals = {}
        for col, _, _ in blocks:
            if col == "stage1_core_s":
                vals[col] = st.median([
                    fnum(r["stage1_runtime_s"]) - fnum(r["mksc_projection_runtime_s"])
                    - fnum(r["common_geometry_runtime_s"]) - fnum(r["anchor_refresh_runtime_s"])
                    for r in rs])
            else:
                vals[col] = st.median([fnum(r[col]) for r in rs])
        bottom = 0.0
        for col, blab, colr in blocks:
            a.bar(i, vals[col], bottom=bottom, color=colr, width=0.62,
                  label=blab if i == 0 else None)
            bottom += vals[col]
        totals.append(bottom)
        a.text(i, bottom + 0.015 * max(totals), f"{bottom:.2f}",
               ha="center", fontsize=6.2)
    a.set_xticks(range(len(variants)))
    a.set_xticklabels([v[1] for v in variants], fontsize=6.5)
    a.set_ylabel("sum of component medians [s]")
    # Data-driven, because the budget dropped by ~5x between campaigns and a
    # hand-tuned ceiling silently clips the bars instead of failing.
    a.set_ylim(0, 1.14 * max(totals))
    a.set_title("(a) single-thread cost decomposition, 30 trials", pad=3)
    grid(a, which="major")

    # The p95 of the error distribution, not its RMSE: at -10 dB the RMSE of
    # every route is set by a handful of wrong-basin trials, so it separates
    # the routes by their tail mass twice over and hides the resolution
    # difference that the runtime is being traded against.
    rmse = {r["baseline"]: fnum(r["position_error_m_p95"])
            for r in B if abs(fnum(r["snr_db"]) + 10.0) < 1e-9}
    runtimes: list[float] = []
    rmses: list[float] = []
    names = {"mksc_ccop": ("Proposed", C["proposed"], "o"),
             "scaled_4d": ("R2 4-D Jones-VP", C["r2"], "s"),
             "als_cpd": ("ALS-CPD", C["als"], "D"),
             "ris_vbi_sbl": ("VBI/SBL", C["vbi"], "P"),
             "nf_ris_groupomp_localgrid_wls": ("NF-RIS OMP-SAGE-WLS", C["omp"], "X")}
    for r in T1:
        key = r["baseline"]
        if key not in names or key not in rmse:
            continue
        lab, colr, mk = names[key]
        t = fnum(r["mean_runtime_s_at_minus10_db"])
        mem = fnum(r["peak_memory_mb"]) / 1024.0
        runtimes.append(t)
        rmses.append(rmse[key])
        b.scatter(t, rmse[key], s=18 + 34 * mem, color=colr, marker=mk,
                  alpha=0.85, edgecolor="k", linewidth=0.4, label=f"{lab} ({mem:.1f} GB)")
        offsets = {
            "mksc_ccop": (8, 9),
            "scaled_4d": (-55, -22),
            "als_cpd": (6, 5),
            "ris_vbi_sbl": (5, -13),
            "nf_ris_groupomp_localgrid_wls": (6, 5),
        }
        dx, dy = offsets[key]
        b.annotate(f"{lab}\n{mem:.2f} GB", (t, rmse[key]),
                   textcoords="offset points", xytext=(dx, dy), fontsize=6.2,
                   linespacing=0.9)
    b.set_yscale("log")
    b.set_xlabel("single-thread mean runtime at $-10$ dB [s]")
    b.set_ylabel("p95 position error at $-10$ dB [m]")
    b.set_title("(b) Accuracy--cost--memory trade-off", pad=3)
    b.set_xlim(0, 1.30 * max(runtimes))
    lo, hi = min(rmses), max(rmses)
    b.set_ylim(0.55 * lo, 1.8 * hi)
    grid(b)

    fig.tight_layout(pad=0.4, w_pad=1.3)
    fig_legend(fig, a, ncol=5, y=0.03)
    fig.savefig(out / "fig_cost.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "tex" / "figs"))
    ap.add_argument("--snr-min", type=float, default=-10.0,
                    help="lowest SNR [dB] shown in the SNR sweeps "
                         "(-10: main text, -30: supplementary)")
    ap.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="v3",
                    help="which released campaign to plot (default: 'v3', "
                         "the paper source of record)")
    args = ap.parse_args()
    global SNR_MIN, DATA
    SNR_MIN = float(args.snr_min)
    DATA = CAMPAIGNS[args.campaign]
    missing = [
        str(d) for dirs in DATA.values() for d in dirs if not d.exists()
    ]
    if missing:
        raise SystemExit(
            "campaign "
            f"{args.campaign!r} is incomplete; missing:\n  "
            + "\n  ".join(sorted(set(missing)))
        )
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_style()
    fig_geometry(out)
    fig_internal(out)
    fig_ablation(out)
    fig_receiver(out)
    fig_compression(out)
    fig_benchmark(out)
    fig_benchmark_clock(out)
    fig_robustness(out)
    fig_boundaries(out)
    fig_generalization(out)
    fig_cost(out)
    print("figures written to", out)
    for p in sorted(out.glob("*.pdf")):
        print("   ", p.name, f"{p.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
