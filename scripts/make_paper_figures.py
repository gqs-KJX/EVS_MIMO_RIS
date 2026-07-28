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
        "receiver": [RES / "paper_v3" / "receiver_information_480"],
        "compression": [RES / "paper_v3" / "compression_matched_480"],
        "benchmark": [RES / "paper_v3" / "benchmark_matched_480"],
        "benchmark_clock": [RES / "paper_v3" / "benchmark_matched_480"],
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

# Active dataset map; set from the command line by ``--campaign``.
DATA = CAMPAIGNS["frozen"]


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
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "lines.linewidth": 1.1,
            "lines.markersize": 3.2,
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


def grid(ax, which="both") -> None:
    ax.grid(True, which=which)
    ax.set_axisbelow(True)


def fig_legend(fig, ax, ncol=3, y=-0.02, extra=()) -> None:
    """Place one shared legend under the whole figure."""
    h, l = ax.get_legend_handles_labels()
    for hh, ll in extra:
        h.append(hh)
        l.append(ll)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               frameon=False, fontsize=6.6, columnspacing=1.3)


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


# =========================================================== fig: geometry ==
def fig_geometry(out: pathlib.Path) -> None:
    ris = np.array([[4.20, -2.20, 1.05], [5.10, 2.10, 1.15], [4.80, 0.0, 1.25]])
    bs = np.array([0.0, 0.0, 1.0])
    ue = np.array([1.25, 0.55, 0.75])
    box = np.array([[0.30, 2.70], [-1.40, 1.50], [0.35, 1.45]])

    fig = plt.figure(figsize=(COL_W, 2.75))
    ax = fig.add_subplot(111, projection="3d")

    # search box wireframe
    xs, ys, zs = box
    for a in xs:
        for b in ys:
            ax.plot([a, a], [b, b], zs, color="0.6", lw=0.5, ls=":")
    for a in xs:
        for c in zs:
            ax.plot([a, a], ys, [c, c], color="0.6", lw=0.5, ls=":")
    for b in ys:
        for c in zs:
            ax.plot(xs, [b, b], [c, c], color="0.6", lw=0.5, ls=":")

    # panels as small squares in the y-z plane
    for k, r in enumerate(ris):
        d = 0.223 / 2
        yy = np.array([r[1] - d, r[1] + d, r[1] + d, r[1] - d, r[1] - d])
        zz = np.array([r[2] - d, r[2] - d, r[2] + d, r[2] + d, r[2] - d])
        ax.plot(np.full(5, r[0]), yy, zz, color=C["proposed"], lw=1.2)
        ax.text(r[0] - 0.15, r[1] + 0.10, r[2] + 0.30, f"RIS {k + 1}",
                color=C["proposed"], fontsize=6.4)
        ax.plot(*[[ue[i], r[i]] for i in range(3)], color="0.45", lw=0.6, ls="-")
        ax.plot(*[[r[i], bs[i]] for i in range(3)], color="0.45", lw=0.6, ls="--")

    ax.scatter(*bs, s=26, marker="^", color=C["r2"], depthshade=False)
    ax.text(bs[0] - 0.1, bs[1] - 0.9, bs[2] + 0.42, "BS (EVS-ULA)", color=C["r2"], fontsize=6.4)
    ax.scatter(*ue, s=22, marker="o", color="k", depthshade=False)
    ax.text(ue[0] - 0.1, ue[1], ue[2] - 0.62, "UE", fontsize=6.4)
    ax.text(0.6, 1.5, 1.62, "UE search box", color="0.45", fontsize=6.0)

    ax.set_xlabel("$x$ [m]", labelpad=-7)
    ax.set_ylabel("$y$ [m]", labelpad=-7)
    ax.set_zlabel("$z$ [m]", labelpad=-7)
    ax.tick_params(pad=-2)
    ax.set_box_aspect((1.55, 1.35, 0.62))
    ax.view_init(elev=20, azim=-62)
    ax.set_zlim(0.2, 1.8)
    ax.set_xlim(0.0, 5.4)
    handles = [Line2D([], [], color=C["r2"], marker="^", ls="none", ms=4, label="BS"),
               Line2D([], [], color=C["proposed"], lw=1.2, label=r"RIS panel ($64\times64$)"),
               Line2D([], [], color="0.45", lw=0.6, label="UE--RIS (near field)"),
               Line2D([], [], color="0.45", lw=0.6, ls="--", label="RIS--BS (calibrated)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.10, 1.03),
              fontsize=5.8, handlelength=1.4)
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
        a.semilogy(x, y, color=col, ls=ls, alpha=0.40, lw=0.8)
    x, y = curve(S, "x_value", "peb_position_m_rms", "variant", "free_jones_peb")
    a.semilogy(x, y, color=C["peb"], ls=(0, (4, 2)), lw=1.0, label="Free-Jones PEB")
    a.annotate("inlier-conditional RMSE\n(all routes $\\approx$ PEB)", xy=(5, 2.3e-4),
               xytext=(-1, 8e-6), fontsize=6.0, ha="center",
               arrowprops=dict(arrowstyle="->", lw=0.5, color="0.35"))
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Position RMSE [m]")
    a.set_title("(a) overall (bold) and inlier-conditional (faint) RMSE", pad=3)
    clip_ylim(a, (2e-6, 2), (3e-7, 3))
    grid(a)

    for v, lab, col, mk, ls in series:
        mfc = "none" if v == "old_stage1_ccop" else col
        x, y = curve(S, "x_value", "catastrophic_rate", "variant", v)
        xl, lo, hi = band(S, "x_value", "outlier_ci_low", "outlier_ci_high", "variant", v)
        b.plot(x, 100 * y, color=col, marker=mk, ls=ls, mfc=mfc, label=lab)
        b.fill_between(xl, 100 * lo, 100 * hi, color=col, alpha=0.13, lw=0)
    b.set_xlabel("SNR [dB]")
    b.set_ylabel("$P_{\\rm cat}$ [%]")
    b.set_title("(b) catastrophic rate, exact 95% Clopper--Pearson", pad=3)
    clip_ylim(b, (-1.5, 32), (-2, 65))
    grid(b)

    for v, lab, col, mk, ls in series:
        mfc = "none" if v == "old_stage1_ccop" else col
        x, y = curve(S, "x_value", "clock_rmse_ns", "variant", v)
        c.semilogy(x, y, color=col, marker=mk, ls=ls, mfc=mfc, label=lab)
        x, y = curve(S, "x_value", "clock_p95_ns", "variant", v)
        c.semilogy(x, y, color=col, ls=ls, alpha=0.40, lw=0.8)
    c.set_xlabel("SNR [dB]")
    c.set_ylabel("Clock error [ns]")
    c.set_title("(c) clock error, RMSE (bold) and p95 (faint)", pad=3)
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

    xr, resc = curve(P, "x_value", "rescued_outliers")
    xi, intro = curve(P, "x_value", "introduced_outliers")
    w = 1.6
    ax[1].bar(xr - w / 2, resc, width=w, color=C["gi4"], label="rescued outliers")
    ax[1].bar(xi + w / 2, intro, width=w, color=C["r2"], label="introduced outliers")
    ax[1].set_xlabel("SNR [dB]")
    ax[1].set_ylabel("trials (of 480)")
    ax[1].set_title("(b) McNemar outlier exchange, $p<10^{-23}$ everywhere", pad=3)
    ax[1].legend(loc="upper right")
    grid(ax[1], which="major")
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(out / "fig_internal_paired.pdf")
    plt.close(fig)


# =========================================================== fig: ablation ==
def fig_ablation(out: pathlib.Path) -> None:
    A = ds("components", "ablation_summary.csv")
    T = ds("components", "ablation_trials.csv")
    order = [
        ("old_stage1_ccop", "R3", "R3: frozen Stage-I", C["r3"]),
        ("mksc_delay_ccop", "A1", "A1: + MKSC delay compression", C["a1"]),
        ("mksc_gi_1_no_refresh_ccop", "A2", "A2: + common geometry (1 start)", C["a2"]),
        ("mksc_gi_4_no_refresh_ccop", "A3", "A3: + 4 deterministic starts", C["gi4"]),
        ("proposed", "Prop.", "Proposed: + anchor refresh", C["proposed"]),
    ]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DBL_W, 2.05))

    xs = np.arange(len(order))
    for i, (v, short, lab, col) in enumerate(order):
        r = sel(A, variant=v)[0]
        p = 100 * fnum(r["catastrophic_rate"])
        lo = 100 * fnum(r["outlier_ci_low"])
        hi = 100 * fnum(r["outlier_ci_high"])
        a.bar(i, p, color=col, width=0.66, label=lab)
        a.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="k", lw=0.7, capsize=2)
        a.text(i, hi + 0.9, f"{p:.1f}", ha="center", fontsize=6.2)
    a.set_xticks(xs)
    a.set_xticklabels([o[1] for o in order], fontsize=7)
    a.set_ylabel("$P_{\\rm cat}$ [%]")
    a.set_ylim(0, 30)
    a.set_title("(a) catastrophic rate at $-10$ dB", pad=3)
    grid(a, which="major")

    for i, (v, short, lab, col) in enumerate(order):
        r = sel(A, variant=v)[0]
        rate = 100 * fnum(r["stage1_basin_acquisition_rate"])
        b.bar(i, rate, color=col, width=0.66)
        b.text(i, rate + 1.2, f"{rate:.1f}", ha="center", fontsize=6.2)
    b.set_xticks(xs)
    b.set_xticklabels([o[1] for o in order], fontsize=7)
    b.set_ylabel("Stage-I basin acquisition [%]")
    b.set_ylim(0, 112)
    b.set_title("(b) trials reaching the correct basin", pad=3)
    grid(b, which="major")

    for v, short, lab, col in order:
        e = sorted(fnum(r["position_error_m"]) for r in sel(T, variant=v)
                   if r["failed"].lower() == "false")
        e = [x for x in e if math.isfinite(x)]
        y = np.arange(1, len(e) + 1) / len(e)
        c.semilogx(e, y, color=col, lw=1.1)
    c.axvline(0.1, color="k", lw=0.7, ls="--")
    c.text(0.115, 0.10, "10 cm\nthreshold", fontsize=6.0)
    c.set_xlabel("Position error [m]")
    c.set_ylabel("Empirical CDF")
    c.set_ylim(0, 1.02)
    c.set_title("(c) error CDF, 480 paired trials", pad=3)
    grid(c)

    fig.tight_layout(pad=0.4, w_pad=1.1)
    fig_legend(fig, a, ncol=3, y=0.015)
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

    fig, (a, b) = plt.subplots(1, 2, figsize=(DBL_W, 2.2))
    for m, lab, col, mk, ls in modes:
        x, y = curve(S, "x_value", "position_conditional_rmse_m", "variant", f"proposed_{m}")
        a.semilogy(x, y, color=col, marker=mk, ls=ls, label=f"estimator, {lab}")
        x, y = curve(S, "x_value", "peb_position_m_rms", "variant", f"free_jones_peb_{m}")
        a.semilogy(x, y, color=col, ls=(0, (3, 2)), lw=0.9, alpha=0.75,
                   label=f"free-Jones PEB, {lab}")
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Inlier-conditional RMSE [m]")
    a.set_title("(a) matched bounds (dashed) and attained accuracy (solid)", pad=3)
    grid(a)

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
        b.semilogy(xs, rr, color=col, marker=mk, ls=ls, label=lab)
    b.axhline(1.0, color="k", lw=0.7, ls="--")
    b.set_xlabel("SNR [dB]")
    b.set_ylabel("RMSE / PEB (paired, inliers)")
    b.set_title("(b) efficiency ratio: bias floor of information-poor modes", pad=3)
    b.legend(loc="upper left", fontsize=6.2)
    grid(b)

    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig_legend(fig, a, ncol=3, y=0.02)
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

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DBL_W, 2.05))
    for v, lab, col, mk, ls in pairs:
        x, y = curve(S, "x_value", "stage1_delay_rmse_ns_median", "variant", v)
        a.semilogy(x, y, color=col, marker=mk, ls=ls, label=lab)
        x, y = curve(S, "x_value", "stage1_delay_rmse_ns_p95", "variant", v)
        a.semilogy(x, y, color=col, ls=ls, alpha=0.40, lw=0.8)
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Stage-I delay RMSE [ns]")
    a.set_title("(a) delay accuracy: median (bold), p95 (faint)", pad=3)
    clip_ylim(a, (3e-4, 6e1), (3e-4, 4e2))
    grid(a)

    for v, lab, col, mk, ls in pairs:
        x, y = curve(S, "x_value", "stage1_basin_acquisition_rate", "variant", v)
        b.plot(x, 100 * y, color=col, marker=mk, ls=ls, label=lab)
    b.set_xlabel("SNR [dB]")
    b.set_ylabel("Basin acquisition [%]", color="k")
    clip_ylim(b, (92, 101), (45, 103))
    grid(b)
    b2 = b.twinx()
    for v, lab, col, mk, ls in pairs:
        x, y = curve(S, "x_value", "catastrophic_rate", "variant", v)
        b2.plot(x, 100 * y, color=col, ls=ls, alpha=0.40, lw=0.8)
    b2.set_ylabel("$P_{\\rm cat}$ [%] (faint)")
    clip_ylim(b2, (-0.4, 9), (-1, 45))
    b.set_title("(b) basin acquisition (bold), $P_{\\rm cat}$ (faint)", pad=3)

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
    c.set_title("(c) subspace-match diagnostics, $r=2K=6$", pad=3)
    grid(c)

    fig.tight_layout(pad=0.4, w_pad=1.5)
    fig_legend(fig, a, ncol=2, y=0.02)
    fig.savefig(out / "fig_compression.pdf")
    plt.close(fig)


# ========================================================= fig: benchmark ===
def fig_benchmark(out: pathlib.Path) -> None:
    B = snr_rows(
        ds("benchmark", "benchmark_summary.csv"),
        "snr_db")
    series = [
        ("mksc_ccop", "Proposed MKSC-GI + CCOP-JVP", C["proposed"], "o", "-"),
        ("scaled_4d", "Scale-normalized 4-D Jones-VP", C["r2"], "s", ":"),
        ("als_cpd", "ALS-CPD + joint mapping", C["als"], "D", "--"),
        ("ris_vbi_sbl", "VBI/SBL joint loc.+rec.", C["vbi"], "P", "-."),
        ("nf_ris_groupomp_localgrid_wls", "NF-RIS CPD-OMP-SAGE-WLS", C["omp"], "X", "--"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(DBL_W, 4.1))
    (a, b), (c, d) = axes

    for v, lab, col, mk, ls in series:
        x, y = curve(B, "snr_db", "position_rmse_m", "baseline", v)
        a.semilogy(x, y, color=col, marker=mk, ls=ls, label=lab)
    x, y = curve(B, "snr_db", "peb_position_m_rms", "baseline", "peb")
    a.semilogy(x, y, color="k", ls=(0, (4, 2)), lw=1.0, label="Free-Jones PEB")
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Position RMSE [m]")
    a.set_title("(a) position RMSE (all trials)", pad=3)
    grid(a)

    for v, lab, col, mk, ls in series:
        x, y = curve(B, "snr_db", "position_error_m_median", "baseline", v)
        b.semilogy(x, y, color=col, marker=mk, ls=ls, label=lab)
    x, y = curve(B, "snr_db", "peb_position_m_rms", "baseline", "peb")
    b.semilogy(x, y, color="k", ls=(0, (4, 2)), lw=1.0)
    b.set_xlabel("SNR [dB]")
    b.set_ylabel("Median position error [m]")
    b.set_title("(b) median error: grid floors of the sparse baselines", pad=3)
    grid(b)

    for v, lab, col, mk, ls in series:
        x, y = curve(B, "snr_db", "catastrophic_rate", "baseline", v)
        c.plot(x, 100 * y, color=col, marker=mk, ls=ls, label=lab)
        xl, lo, hi = band(B, "snr_db", "outlier_ci_low", "outlier_ci_high", "baseline", v)
        c.fill_between(xl, 100 * lo, 100 * hi, color=col, alpha=0.10, lw=0)
    c.set_xlabel("SNR [dB]")
    c.set_ylabel("$P_{\\rm cat}$ [%]")
    clip_ylim(c, (-1, 22), (-3, 102))
    c.set_title("(c) catastrophic rate with 95% CP intervals", pad=3)
    grid(c)

    for v, lab, col, mk, ls in series:
        x, y = curve(B, "snr_db", "y_nmse_mean", "baseline", v)
        d.semilogy(x, y, color=col, marker=mk, ls=ls, label=lab)
    d.set_xlabel("SNR [dB]")
    d.set_ylabel("Observation-domain NMSE")
    d.set_title("(d) channel/observation reconstruction NMSE", pad=3)
    grid(d)

    fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.0)
    fig_legend(fig, a, ncol=3, y=0.005)
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
        ("mksc_ccop", "Proposed MKSC-GI + CCOP-JVP", C["proposed"], "o", "-"),
        ("scaled_4d", "Scale-normalized 4-D Jones-VP", C["r2"], "s", ":"),
    ]
    external = [
        ("als_cpd", "ALS-CPD + joint mapping", C["als"], "D", "--"),
        ("ris_vbi_sbl", "VBI/SBL joint loc.+rec.", C["vbi"], "P", "-."),
        ("nf_ris_groupomp_localgrid_wls", "NF-RIS CPD-OMP-SAGE-WLS", C["omp"], "X", "--"),
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
        a.semilogy(x, 1e3 * y, color=col, marker=mk, ls=ls, label=lab)
        x, y = curve(Bs, "snr_db", "clock_error_ns_p95", "baseline", v)
        a.semilogy(x, 1e3 * y, color=col, ls=ls, alpha=0.40, lw=0.8)
    for v, lab, col, mk, ls in external:
        x, y = curve(Cx, "snr_db", "clock_median_abs_error_ns", "baseline", v)
        a.semilogy(x, 1e3 * y, color=col, marker=mk, ls=ls, label=lab)
        x, y = curve(Cx, "snr_db", "clock_p95_abs_error_ns", "baseline", v)
        a.semilogy(x, 1e3 * y, color=col, ls=ls, alpha=0.40, lw=0.8)
    a.set_xlabel("SNR [dB]")
    a.set_ylabel("Clock error [ps]")
    a.set_title("(a) clock error: median (bold) and p95 (faint)", pad=3)
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
    b.set_title("(b) clock-catastrophic rate", pad=3)
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
        a.plot(x, 100 * y, color=col, marker=mk, ls=ls, label=lab)
        x, y = curve(rs, "x_value", "stage1_basin_acquisition_rate")
        a.plot(x, 100 * y, color=col, marker=mk, ls=ls, alpha=0.4, lw=0.8, ms=2.2)
    a.set_xlabel("RIS--BS incidence-angle error [deg]")
    a.set_ylabel("$P_{\\rm cat}$ / basin rate [%]")
    a.set_title("(a) infrastructure-geometry mismatch (bold: $P_{\\rm cat}$)", pad=3)
    a.legend(loc="center left", fontsize=6.0)
    grid(a)

    styles = {"evs_gain_std": ("EVS gain std", C["gi4"], "v", "-"),
              "evs_phase_deg": ("EVS phase [deg]", C["r3"], "^", "--"),
              "bs_sensor_position_std_mm": ("sensor position [mm]", C["a2"], "D", "-.")}
    for xn, (lab, col, mk, ls) in styles.items():
        rs = sel(M, x_name=xn, variant="proposed")
        x, y = curve(rs, "x_value", "catastrophic_rate")
        xn_norm = x / max(x.max(), 1e-12)
        b.plot(xn_norm, 100 * y, color=col, marker=mk, ls=ls, label=lab)
    b.set_xlabel("mismatch level, normalized to sweep maximum")
    b.set_ylabel("$P_{\\rm cat}$ [%]")
    b.set_ylim(0, 8)
    b.set_title("(b) EVS/Maxwell calibration perturbations", pad=3)
    b.legend(loc="upper left", fontsize=6.0)
    grid(b)

    for v, lab, col, mk, ls in [("proposed", "MKSC compressed", C["proposed"], "o", "-"),
                                ("raw_delay_gi_ccop", "raw delay subspace", C["raw"], "s", "--")]:
        rs = sel(N, variant=v)
        x, y = curve(rs, "x_value", "catastrophic_rate")
        c.plot(x, 100 * y, color=col, marker=mk, ls=ls, label=lab)
    c.set_xlabel(r"noise correlation coefficient $\rho$")
    c.set_ylabel("$P_{\\rm cat}$ [%]")
    c.set_title("(c) unwhitened colored-noise boundary", pad=3)
    c.legend(loc="upper left", fontsize=6.0)
    grid(c)

    rs = [r for r in F9 if r["baseline"] == "mksc_ccop"]
    x, y = curve(rs, "x_value", "outlier_rate")
    d.plot(x, 100 * y, color=C["proposed"], marker="o", label="$P_{\\rm cat}$ (left)")
    d.set_xlabel(r"assumed model order $\hat K$ (true $K=3$)")
    d.set_ylabel("$P_{\\rm cat}$ [%]")
    d.set_xticks([2, 3, 4, 5])
    grid(d)
    d2 = d.twinx()
    x, y = curve(rs, "x_value", "median_m")
    d2.semilogy(x, y, color=C["gi4"], marker="v", ls="--", label="median error (right)")
    x, y = curve(rs, "x_value", "p95_m")
    d2.semilogy(x, y, color=C["r2"], marker="s", ls=":", label="p95 error (right)")
    d2.set_ylabel("position error [m]")
    d.set_title("(d) model-order mismatch", pad=3)
    h1, l1 = d.get_legend_handles_labels()
    h2, l2 = d2.get_legend_handles_labels()
    d.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=6.0)

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
        ax.semilogy(x, y, color=col, marker=mk, ls=ls, alpha=0.4, lw=0.8, ms=2.2)
    ax.set_xlabel("RIS element-phase calibration error std [deg]")
    ax.set_ylabel("position error [m]")
    ax.set_title("median (solid) and p95 (faint) error", pad=3)
    ax.legend(loc="center left", fontsize=6.0)
    grid(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_ris_calibration.pdf")
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
    b.set_ylim(6e4, 3e6)
    grid(b)
    b2 = b.twinx()
    b2.plot(xs, np.array(cond)[idx], color=C["r3"], marker="^", ms=2.4, lw=0.9, ls="--",
            label="condition number (right)")
    b2.set_ylabel("EFIM condition number")
    b2.set_ylim(0, 700)
    b.set_title("(b) local identifiability over the grid", pad=3)
    b.text(30, 2.0e6, f"rank-deficient trials: {defic}/{ntr}", fontsize=6.0, ha="center")
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.0)

    axes_map = [("N", "$N$", C["proposed"], "o", "-"),
                ("T", "$T$", C["gi4"], "v", "--"),
                ("M_A", "$M_A$", C["r3"], "^", "-."),
                ("M_R", "$M_R$", C["a2"], "D", ":")]
    for xn, lab, col, mk, ls in axes_map:
        rs = sel(S, x_name=xn, variant="proposed")
        x, y = curve(rs, "x_value", "catastrophic_rate")
        c.semilogx(x, 100 * y, color=col, marker=mk, ls=ls, label=lab)
        rs = sel(S, x_name=xn, variant="scaled_4d")
        x, y = curve(rs, "x_value", "catastrophic_rate")
        c.semilogx(x, 100 * y, color=col, marker=mk, ls=ls, alpha=0.32, lw=0.8, ms=2.2)
    c.set_xlabel("resource value (log scale)")
    c.set_ylabel("$P_{\\rm cat}$ [%]")
    c.set_title("(c) resource scaling", pad=3)
    c.legend(loc="upper right", ncol=2, fontsize=6.0)
    grid(c)

    fig.tight_layout(pad=0.4, w_pad=1.3)
    fig.savefig(out / "fig_generalization.pdf")
    plt.close(fig)

    # resolvability heat maps ------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(DBL_W, 1.9), sharey=True)
    seps = sorted({fnum(r["target_delay_separation_ns"]) for r in Rv})
    ovs = sorted({fnum(r["polarization_overlap_target"]) for r in Rv})
    for ax, (m, lab) in zip(axs, [("scalar", "scalar"), ("dual_pol", "dual-pol"),
                                  ("full_6d", "full 6-comp. EVS")]):
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
                            fontsize=5.6, color="w" if Z[i, j] < 60 else "k")
    axs[0].set_ylabel("polarization overlap")
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.012)
    cb.set_label("resolution probability [%]", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
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
    a.set_ylabel("median wall-clock time [s]")
    # Data-driven, because the budget dropped by ~5x between campaigns and a
    # hand-tuned ceiling silently clips the bars instead of failing.
    a.set_ylim(0, 1.14 * max(totals))
    a.set_title("(a) single-thread cost decomposition, 30 trials", pad=3)
    grid(a, which="major")

    rmse = {r["baseline"]: fnum(r["position_rmse_m"])
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
        b.annotate(lab, (t, rmse[key]), textcoords="offset points", xytext=(4, 5),
                   fontsize=5.8)
    b.set_yscale("log")
    b.set_xlabel("single-thread mean runtime at $-10$ dB [s]")
    b.set_ylabel("position RMSE at $-10$ dB [m]")
    b.set_title("(b) accuracy--cost trade-off (marker area $\\propto$ peak RSS)", pad=3)
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
    ap.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="frozen",
                    help="which released campaign to plot: 'frozen' is the "
                         "2026-07-19/21 release, 'v3' the 2026-07-28 re-run")
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
    fig_generalization(out)
    fig_cost(out)
    print("figures written to", out)
    for p in sorted(out.glob("*.pdf")):
        print("   ", p.name, f"{p.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
