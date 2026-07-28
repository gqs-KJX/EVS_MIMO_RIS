#!/usr/bin/env python
"""Print, in one place, every measured quantity the manuscript quotes.

The manuscript's Results section and Table~I are dense with scalars that each
descend from one column of one released CSV.  After a re-run, checking them by
hand is error prone, so this script re-reads the CSVs and prints the numbers in
the same grouping the text uses.  It computes nothing that the released summary
files do not already contain, except for the quantiles of the raw per-trial
error columns, which the paper reports and the summary CSV does not always
carry at the requested percentile.

Usage:
    python scripts/extract_paper_numbers.py [--results results/paper_v3]
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics as st


def load(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def finite(values) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile, the convention the released summaries use."""
    vals = sorted(finite(values))
    if not vals:
        return float("nan")
    idx = max(0, min(len(vals) - 1, int(math.ceil(q / 100.0 * len(vals))) - 1))
    return vals[idx]


def mm(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{1000.0 * value:.3g}"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


LABELS = {
    "mksc_ccop": "Proposed MKSC-GI + CCOP-JVP",
    "scaled_4d": "R2 scale-normalized 4-D Jones-VP",
    "als_cpd": "ALS-CPD + joint mapping",
    "ris_vbi_sbl": "VBI/SBL joint loc.+rec.",
    "nf_ris_groupomp_localgrid_wls": "NF-RIS CPD-OMP-SAGE-WLS",
    "peb": "Free-Jones PEB",
    "constrained_jones_peb": "Constrained-Jones PEB",
}
ORDER = [
    "als_cpd",
    "ris_vbi_sbl",
    "nf_ris_groupomp_localgrid_wls",
    "scaled_4d",
    "mksc_ccop",
    "peb",
]


def benchmark_block(root: pathlib.Path, snrs: list[float]) -> None:
    trials = load(root / "benchmark_trials.csv")
    summary = load(root / "benchmark_summary.csv")
    if not trials:
        print(f"  (no benchmark_trials.csv under {root})")
        return

    for snr in snrs:
        section(f"External benchmark @ {snr:+.0f} dB  [{root.name}]")
        print(
            f"  {'method':34s} {'median':>9s} {'p90':>9s} {'p95':>9s} "
            f"{'p99':>9s} {'RMSE':>9s} {'Pcat%':>7s} {'NMSE':>10s}"
        )
        for base in ORDER:
            rows = [
                r
                for r in trials
                if r.get("baseline") == base
                and abs(fnum(r.get("snr_db")) - snr) < 1e-9
            ]
            if not rows:
                continue
            err = [fnum(r.get("position_error_m")) for r in rows]
            peb = finite([fnum(r.get("peb_position_m")) for r in rows])
            nmse = finite([fnum(r.get("y_nmse")) for r in rows])
            srow = [
                r
                for r in summary
                if r.get("baseline") == base
                and abs(fnum(r.get("snr_db")) - snr) < 1e-9
            ]
            if base == "peb":
                rms = math.sqrt(st.fmean([p * p for p in peb])) if peb else float("nan")
                print(f"  {LABELS[base]:34s} {'---':>9s} {'---':>9s} {'---':>9s} "
                      f"{'---':>9s} {mm(rms):>9s} {'---':>7s} {'---':>10s}")
                continue
            rmse = (
                math.sqrt(st.fmean([e * e for e in finite(err)]))
                if finite(err)
                else float("nan")
            )
            cat = fnum(srow[0].get("catastrophic_rate")) if srow else float("nan")
            print(
                f"  {LABELS.get(base, base):34s} {mm(st.median(finite(err))):>9s} "
                f"{mm(pct(err, 90)):>9s} {mm(pct(err, 95)):>9s} "
                f"{mm(pct(err, 99)):>9s} {mm(rmse):>9s} "
                f"{100.0 * cat:>7.2f} "
                f"{(st.fmean(nmse) if nmse else float('nan')):>10.2e}"
            )
        print("  (position columns in mm)")


def clock_block(root: pathlib.Path, snrs: list[float]) -> None:
    summary = load(root / "benchmark_summary.csv")
    if not summary:
        return
    section(f"External synchronization  [{root.name}]")
    print(
        f"  {'method':34s} {'SNR':>5s} {'median|e|ns':>12s} {'p95 ns':>10s} "
        f"{'RMSE ns':>10s} {'clk Pcat%':>10s}"
    )
    for base in ORDER:
        for snr in snrs:
            rows = [
                r
                for r in summary
                if r.get("baseline") == base
                and abs(fnum(r.get("snr_db")) - snr) < 1e-9
            ]
            if not rows:
                continue
            r = rows[0]
            print(
                f"  {LABELS.get(base, base):34s} {snr:>5.0f} "
                f"{fnum(r.get('clock_median_abs_error_ns')):>12.4g} "
                f"{fnum(r.get('clock_p95_abs_error_ns')):>10.4g} "
                f"{fnum(r.get('clock_rmse_ns')):>10.4g} "
                f"{100.0 * fnum(r.get('clock_catastrophic_rate')):>10.2f}"
            )


def runtime_block(root: pathlib.Path) -> None:
    table = load(root / "table1_runtime_memory.csv")
    if not table:
        print(f"  (no table1_runtime_memory.csv under {root})")
        return
    section(f"Table I cost column  [{root.name}]")
    print(f"  {'method':34s} {'s @-10dB':>10s} {'s @0dB':>10s} {'peak GB':>9s}")
    for r in table:
        base = r.get("baseline", "")
        mem = fnum(r.get("peak_memory_mb")) / 1024.0
        print(
            f"  {LABELS.get(base, base):34s} "
            f"{fnum(r.get('mean_runtime_s_at_minus10_db')):>10.3f} "
            f"{fnum(r.get('mean_runtime_s_at_0_db')):>10.3f} "
            f"{mem:>9.3f}"
        )


def cost_decomposition(root: pathlib.Path) -> None:
    rows = load(root / "ablation_trials.csv")
    if not rows:
        print(f"  (no ablation_trials.csv under {root})")
        return
    section(f"Serial cost decomposition  [{root.name}]")
    blocks = [
        ("mksc_projection_runtime_s", "MKSC projection"),
        ("stage1_core_s", "Hankel / delay / coupled LS / assignment"),
        ("common_geometry_runtime_s", "common-geometry init"),
        ("anchor_refresh_runtime_s", "anchor refresh"),
        ("stage3_runtime_s", "CCOP-JVP refinement"),
    ]
    variants = sorted({r.get("variant", "") for r in rows})
    for variant in variants:
        rs = [r for r in rows if r.get("variant") == variant]
        vals = {}
        for col, _ in blocks:
            if col == "stage1_core_s":
                vals[col] = st.median(
                    [
                        fnum(r["stage1_runtime_s"])
                        - fnum(r["mksc_projection_runtime_s"])
                        - fnum(r["common_geometry_runtime_s"])
                        - fnum(r["anchor_refresh_runtime_s"])
                        for r in rs
                    ]
                )
            else:
                vals[col] = st.median([fnum(r.get(col)) for r in rs])
        total = sum(vals.values())
        rss = finite([fnum(r.get("rss_mb_peak")) for r in rs])
        print(f"  {variant:32s} total={total:7.3f} s  "
              f"peakRSS={(max(rss) / 1024.0 if rss else float('nan')):5.2f} GB")
        for col, label in blocks:
            share = 100.0 * vals[col] / total if total > 0 else float("nan")
            print(f"      {label:32s} {vals[col]:7.3f} s  ({share:5.1f}%)")


def snr_curve(root: pathlib.Path, variant: str, snrs: list[float]) -> None:
    rows = load(root / "ablation_summary.csv")
    if not rows:
        print(f"  (no ablation_summary.csv under {root})")
        return
    section(f"Internal SNR curve, variant={variant}  [{root.name}]")
    print(
        f"  {'SNR':>5s} {'cond RMSE mm':>13s} {'PEB rms mm':>11s} {'ratio':>7s} "
        f"{'Pcat%':>7s} {'clk RMSE ns':>12s}"
    )
    for snr in snrs:
        rs = [
            r
            for r in rows
            if r.get("variant") == variant
            and abs(fnum(r.get("x_value")) - snr) < 1e-9
        ]
        if not rs:
            continue
        r = rs[0]
        cond = fnum(r.get("position_conditional_rmse_m"))
        # The matched bound is carried by its own ``free_jones_peb`` variant row
        # at the same SNR, not as a column on the estimator rows.
        peb_rows = [
            q
            for q in rows
            if q.get("variant") == "free_jones_peb"
            and abs(fnum(q.get("x_value")) - snr) < 1e-9
        ]
        peb = fnum(peb_rows[0].get("peb_position_m_rms")) if peb_rows else float("nan")
        print(
            f"  {snr:>5.0f} {1000 * cond:>13.4g} {1000 * peb:>11.4g} "
            f"{(cond / peb if peb else float('nan')):>7.3f} "
            f"{100 * fnum(r.get('catastrophic_rate')):>7.2f} "
            f"{fnum(r.get('clock_rmse_ns')):>12.4g}"
        )


def compression_block(root: pathlib.Path, snrs: list[float]) -> None:
    rows = load(root / "ablation_summary.csv")
    if not rows:
        print(f"  (no ablation_summary.csv under {root})")
        return
    section(f"Matched compression evidence  [{root.name}]")
    print(
        f"  {'variant':28s} {'SNR':>5s} {'delay med ns':>13s} {'delay p95 ns':>13s} "
        f"{'basin%':>8s} {'Pcat%':>7s}"
    )
    for variant in sorted({r.get("variant", "") for r in rows}):
        for snr in snrs:
            rs = [
                r
                for r in rows
                if r.get("variant") == variant
                and abs(fnum(r.get("x_value")) - snr) < 1e-9
            ]
            if not rs:
                continue
            r = rs[0]
            print(
                f"  {variant:28s} {snr:>5.0f} "
                f"{fnum(r.get('stage1_delay_rmse_ns_median')):>13.4g} "
                f"{fnum(r.get('stage1_delay_rmse_ns_p95')):>13.4g} "
                f"{100 * fnum(r.get('stage1_basin_acquisition_rate')):>8.2f} "
                f"{100 * fnum(r.get('catastrophic_rate')):>7.2f}"
            )


ABLATION_ORDER = [
    ("scaled_4d", "R2: frozen Stage-I + 4-D Jones-VP"),
    ("old_stage1_ccop", "R3: frozen Stage-I + CCOP-JVP"),
    ("mksc_delay_ccop", "A1: + MKSC delay compression"),
    ("mksc_gi_1_no_refresh_ccop", "A2: + common geometry (1 start)"),
    ("mksc_gi_4_no_refresh_ccop", "A3: + four deterministic starts"),
    ("proposed", "Proposed: + anchor refresh"),
    ("mksc_gi_7_refresh_ccop", "(7 starts)"),
    ("oracle_position_start_ccop", "(oracle position start)"),
]


def ablation_block(root: pathlib.Path) -> None:
    """Table~\\ref{tab:ablation}: one row per rung of the nested ladder."""
    rows = load(root / "ablation_summary.csv")
    if not rows:
        print(f"  (no ablation_summary.csv under {root})")
        return
    # The components suite indexes by ``component_set`` ("all"), not by SNR, so
    # the focus SNR lives in the run manifest rather than in the summary rows.
    focus = "?"
    command = (root / "command.txt")
    if command.exists():
        tokens = command.read_text().split()
        if "--focus-snr-db" in tokens:
            focus = tokens[tokens.index("--focus-snr-db") + 1]
    section(f"Component ablation  [{root.name}]  focus SNR {focus} dB")
    print(
        f"  {'route':34s} {'Pcat%':>7s} {'CI':>16s} {'basin%':>7s} "
        f"{'condRMSE mm':>12s} {'p95 err m':>10s} {'clkRMSE ns':>11s} "
        f"{'NMSE mean':>10s} {'NMSE p95':>10s} {'t med s':>8s}"
    )
    for key, label in ABLATION_ORDER:
        rs = [r for r in rows if r.get("variant") == key]
        if not rs:
            continue
        r = rs[0]
        ci = (
            f"[{100 * fnum(r.get('outlier_ci_low')):.2f},"
            f"{100 * fnum(r.get('outlier_ci_high')):.2f}]"
        )
        print(
            f"  {label:34s} {100 * fnum(r.get('catastrophic_rate')):>7.2f} {ci:>16s} "
            f"{100 * fnum(r.get('stage1_basin_acquisition_rate')):>7.2f} "
            f"{1000 * fnum(r.get('position_conditional_rmse_m')):>12.4g} "
            f"{fnum(r.get('position_p95_m')):>10.4g} "
            f"{fnum(r.get('clock_rmse_ns')):>11.4g} "
            f"{fnum(r.get('channel_nmse_mean')):>10.2e} "
            f"{fnum(r.get('channel_nmse_p95')):>10.2e} "
            f"{fnum(r.get('runtime_median_s')):>8.3f}"
        )
    print("  Stage-I delay error (median / p95 ns):")
    for key, label in ABLATION_ORDER:
        rs = [r for r in rows if r.get("variant") == key]
        if not rs:
            continue
        print(
            f"    {label:34s} {fnum(rs[0].get('stage1_delay_rmse_ns_median')):>9.4g}"
            f" / {fnum(rs[0].get('stage1_delay_rmse_ns_p95')):>9.4g}"
        )


def paired_block(root: pathlib.Path) -> None:
    """Paired evidence against the declared reference variant."""
    rows = load(root / "ablation_paired.csv")
    if not rows:
        print(f"  (no ablation_paired.csv under {root})")
        return
    section(f"Paired evidence  [{root.name}]")
    print(
        f"  {'x':>7s} {'ref -> cand':34s} {'dRMSE m':>10s} "
        f"{'CI':>22s} {'resc':>5s} {'intro':>6s} {'McNemar p':>11s} "
        f"{'t ratio':>8s} {'chan win':>9s}"
    )
    for r in rows:
        pair = f"{r.get('reference_variant','')} -> {r.get('candidate_variant','')}"
        ci = (
            f"[{fnum(r.get('paired_position_rmse_difference_ci_low_m')):.4g},"
            f"{fnum(r.get('paired_position_rmse_difference_ci_high_m')):.4g}]"
        )
        x_raw = r.get("x_value", "")
        x_num = fnum(x_raw)
        x_label = f"{x_num:.1f}" if math.isfinite(x_num) else str(x_raw)[:7]
        print(
            f"  {x_label:>7s} {pair[:34]:34s} "
            f"{fnum(r.get('paired_position_rmse_difference_m')):>10.4g} {ci:>22s} "
            f"{r.get('rescued_outliers',''):>5s} {r.get('introduced_outliers',''):>6s} "
            f"{fnum(r.get('mcnemar_exact_p')):>11.2e} "
            f"{fnum(r.get('paired_runtime_ratio_median')):>8.3f} "
            f"{fnum(r.get('channel_win_rate')):>9.4g}"
        )


def receiver_block(root: pathlib.Path, snrs: list[float]) -> None:
    """Receiver-mode study.

    This suite encodes the mode in the variant name -- ``proposed_scalar`` and
    its matched bound ``free_jones_peb_scalar`` -- rather than in a column, so
    each estimator row has to be paired with the bound row carrying the same
    suffix.
    """
    rows = load(root / "ablation_summary.csv")
    if not rows:
        print(f"  (no ablation_summary.csv under {root})")
        return
    modes = ["scalar", "dual_pol", "full_6d"]
    section(f"Receiver modes  [{root.name}]")
    print(
        f"  {'mode':10s} {'SNR':>5s} {'condRMSE mm':>12s} {'PEB mm':>10s} "
        f"{'ratio':>7s} {'Pcat%':>7s}"
    )

    def find(prefix: str, mode: str, snr: float) -> dict | None:
        for r in rows:
            if r.get("variant") != f"{prefix}_{mode}":
                continue
            if abs(fnum(r.get("x_value")) - snr) < 1e-9:
                return r
        return None

    for mode in modes:
        for snr in snrs:
            est = find("proposed", mode, snr)
            bound = find("free_jones_peb", mode, snr)
            if est is None:
                continue
            cond = fnum(est.get("position_conditional_rmse_m"))
            peb = fnum(bound.get("peb_position_m_rms")) if bound else float("nan")
            print(
                f"  {mode:10s} {snr:>5.0f} {1000 * cond:>12.4g} "
                f"{1000 * peb:>10.4g} "
                f"{(cond / peb if peb else float('nan')):>7.3f} "
                f"{100 * fnum(est.get('catastrophic_rate')):>7.2f}"
            )
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/paper_v3")
    args = ap.parse_args()
    root = pathlib.Path(args.results)

    # Directory names differ between the 480-trial campaign and the 10-trial
    # verification campaign; try both spellings and use whichever exists.
    def pick(*names: str) -> pathlib.Path:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
        return root / names[0]

    benchmark_block(pick("benchmark_matched_480", "benchmark_matched"),
                    [-10.0, 0.0, 30.0])
    clock_block(pick("benchmark_matched_480", "benchmark_matched"),
                [-10.0, 0.0, 10.0, 30.0])
    benchmark_block(pick("benchmark_as_published_480", "benchmark_as_published"),
                    [-15.0, -10.0, 0.0, 10.0])
    runtime_block(pick("benchmark_runtime30_cpu"))
    cost_decomposition(pick("components_cost30_cpu"))
    internal = pick("snr_internal_480", "snr_internal")
    for variant in ("proposed", "scaled_4d"):
        snr_curve(internal, variant, [-10.0, -5.0, 0.0, 10.0, 20.0, 30.0])
    paired_block(internal)
    for name in (("components_480", "components_m10"), ("components_480_m20", "components_m20")):
        directory = pick(*name)
        ablation_block(directory)
        paired_block(directory)
    receiver_block(pick("receiver_information_480", "receiver_information"),
                   [-10.0, 0.0, 10.0, 30.0])
    compression_block(pick("compression_matched_480", "compression_matched"),
                      [-10.0, -5.0, 0.0, 5.0])


if __name__ == "__main__":
    main()
