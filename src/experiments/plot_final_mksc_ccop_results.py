"""Plot final MKSC--CCOP results only from saved CSV/JSON artifacts.

This module deliberately contains no simulation or estimator calls.  Each
experiment has a dedicated plotting function, while
``plot_all_final_mksc_ccop_results`` provides one entry point for the complete
result directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


DEFAULT_RESULT_ROOT = pathlib.Path("result/final_mksc_ccop")

EXPERIMENT_DIRECTORIES = {
    "components": "components_paper400",
    "snr": "snr_internal_paper200",
    "receiver": "receiver_information_paper200",
    "compression": "compression_matched_paper200",
    "benchmark": "benchmark_cpu",
    "maxwell_mismatch": "maxwell_mismatch_paper150",
    "colored_noise": "colored_noise_boundary150",
    "positions": "positions50x30",
    "scaling": "robustness_scaling50",
    "evs_resolvability": "evs_resolvability_paper200",
}

DISPLAY_LABELS = {
    "scaled_4d": "Scaled 4-D",
    "old_stage1_ccop": "Stage-I + CCOP",
    "mksc_delay_ccop": "MKSC delay + CCOP",
    "mksc_gi_1_no_refresh_ccop": "MKSC-GI (1 start)",
    "mksc_gi_4_no_refresh_ccop": "MKSC-GI (4 starts)",
    "proposed": "Proposed",
    "mksc_ccop": "Proposed",
    "raw_delay_gi_ccop": "Raw-delay GI + CCOP",
    "free_jones_peb": "Free-Jones PEB",
    "peb": "Free-Jones PEB",
    "constrained_jones_peb": "Constrained-Jones PEB",
    "als_cpd": "ALS-CPD",
    "ris_momp": "RIS-MOMP adaptation",
    "nf_ris_groupomp_localgrid_wls": "NF-RIS CPD-OMP-SAGE-WLS adaptation",
    "proposed_scalar": "Scalar",
    "proposed_dual_pol": "Dual-polarized",
    "proposed_full_6d": "Full 6-D",
    "free_jones_peb_scalar": "Scalar PEB",
    "free_jones_peb_dual_pol": "Dual-polarized PEB",
    "free_jones_peb_full_6d": "Full 6-D PEB",
}

MODE_COLORS = {
    "scalar": "#E69F00",
    "dual_pol": "#009E73",
    "full_6d": "#0072B2",
}

SERIES_COLORS = {
    "proposed": "#D55E00",
    "mksc_ccop": "#D55E00",
    "scaled_4d": "#0072B2",
    "old_stage1_ccop": "#999999",
    "mksc_delay_ccop": "#E69F00",
    "mksc_gi_1_no_refresh_ccop": "#56B4E9",
    "mksc_gi_4_no_refresh_ccop": "#009E73",
    "raw_delay_gi_ccop": "#0072B2",
    "free_jones_peb": "#222222",
    "peb": "#222222",
    "constrained_jones_peb": "#7A5195",
    "als_cpd": "#009E73",
    "ris_momp": "#CC79A7",
    "nf_ris_groupomp_localgrid_wls": "#7A5195",
}

FALLBACK_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7A5195",
    "#555555",
)

X_LABELS = {
    "snr_db": "SNR (dB)",
    "colored_noise_rho": r"Noise correlation $\rho$",
    "bs_sensor_position_std_mm": "BS sensor-position std. (mm)",
    "evs_gain_std": "EVS gain-error std.",
    "evs_phase_deg": "EVS phase-error std. (deg)",
    "ris_bs_angle_deg": "RIS--BS angle-error std. (deg)",
    "K": "Number of paths, K",
    "M_A": "BS array size, M_A",
    "M_R": "RIS elements, M_R",
    "N": "Subcarriers, N",
    "T": "Training slots, T",
}


def load_csv_or_json(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Load tabular records from CSV or a record-oriented JSON file.

    JSON inputs may be a list of objects, a single object, or an object whose
    ``rows``, ``records``, ``data``, or ``results`` member is a list of objects.
    """

    source = pathlib.Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix != ".json":
        raise ValueError(f"unsupported data format: {source.suffix}")
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("rows", "records", "data", "results"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
        if records is None:
            records = [payload]
    else:
        raise ValueError(f"JSON data must contain objects: {source}")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"JSON records must all be objects: {source}")
    return [dict(record) for record in records]


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _first_finite(row: Mapping[str, Any], *fields: str) -> float:
    for field in fields:
        value = _as_float(row.get(field))
        if math.isfinite(value):
            return value
    return float("nan")


def _to_db(value: Any) -> float:
    number = _as_float(value)
    if not number > 0.0:
        return float("nan")
    return 10.0 * math.log10(number)


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _display_name(name: str) -> str:
    return DISPLAY_LABELS.get(name, name.replace("_", " "))


def _series_color(name: str, index: int) -> str:
    for mode, color in MODE_COLORS.items():
        if name == mode or name.endswith(f"_{mode}"):
            return color
    return SERIES_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def _is_reference(name: str) -> bool:
    return "peb" in name


def _line_style(name: str, index: int) -> dict[str, Any]:
    reference = _is_reference(name)
    proposed = name in {"proposed", "mksc_ccop"} or name.startswith("proposed_")
    markers = ("o", "s", "^", "D", "v", "P", "X", "h")
    return {
        "color": _series_color(name, index),
        "linestyle": "--" if reference else "-",
        "linewidth": 2.2 if proposed else 1.65,
        "marker": None if reference else markers[index % len(markers)],
        "markersize": 4.8,
        "markerfacecolor": "white" if not proposed else _series_color(name, index),
        "markeredgewidth": 0.9,
        "zorder": 5 if proposed else (4 if reference else 3),
    }


def _configure_matplotlib(output_dir: pathlib.Path):
    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    return plt


def _save_figure(fig: Any, output_pdf: pathlib.Path) -> pathlib.Path:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        format="pdf",
        metadata={"Creator": "plot_final_mksc_ccop_results.py"},
    )
    return output_pdf


def _curve_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_field: str,
    y_getter: Callable[[Mapping[str, Any]], float],
    ci_fields: tuple[str, str] | None = None,
    positive_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    points: list[tuple[float, float, float, float]] = []
    for row in rows:
        x_value = _as_float(row.get(x_field))
        y_value = y_getter(row)
        if positive_only and not y_value > 0.0:
            y_value = float("nan")
        low = high = float("nan")
        if ci_fields is not None:
            low = _as_float(row.get(ci_fields[0]))
            high = _as_float(row.get(ci_fields[1]))
        if math.isfinite(x_value):
            points.append((x_value, y_value, low, high))
    points.sort(key=lambda item: item[0])
    if not points:
        empty = np.asarray([], dtype=float)
        return empty, empty, None, None
    values = np.asarray(points, dtype=float)
    low = values[:, 2] if ci_fields is not None else None
    high = values[:, 3] if ci_fields is not None else None
    return values[:, 0], values[:, 1], low, high


def _draw_grouped_curves(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    x_field: str,
    group_field: str,
    y_getter: Callable[[Mapping[str, Any]], float],
    group_order: Sequence[str] | None = None,
    ci_fields: tuple[str, str] | None = None,
    positive_only: bool = False,
) -> int:
    available = _ordered_unique([row.get(group_field, "") for row in rows])
    if group_order is None:
        groups = available
    else:
        groups = [name for name in group_order if name in available]
        groups.extend(name for name in available if name not in groups)
    lines = 0
    for index, group in enumerate(groups):
        selected = [row for row in rows if str(row.get(group_field, "")) == group]
        x, y, low, high = _curve_points(
            selected,
            x_field=x_field,
            y_getter=y_getter,
            ci_fields=ci_fields,
            positive_only=positive_only,
        )
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        style = _line_style(group, index)
        ax.plot(x, y, label=_display_name(group), **style)
        if low is not None and high is not None:
            band = finite & np.isfinite(low) & np.isfinite(high)
            if np.any(band):
                ax.fill_between(
                    x[band],
                    low[band],
                    high[band],
                    color=style["color"],
                    alpha=0.12,
                    linewidth=0.0,
                    zorder=1,
                )
        lines += 1
    return lines


def _style_curve_axis(
    ax: Any,
    *,
    x_label: str,
    y_label: str,
    y_scale: str = "linear",
    y_limits: tuple[float, float] | None = None,
) -> None:
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yscale(y_scale)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, which="major", color="#B0B0B0", linestyle=":", linewidth=0.7)
    if y_scale == "log":
        ax.grid(True, which="minor", color="#D0D0D0", linestyle=":", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.035)


def plot_csv_or_json(
    source: str | pathlib.Path,
    output_pdf: str | pathlib.Path,
    *,
    x_field: str,
    y_field: str,
    group_field: str,
    x_label: str,
    y_label: str,
    y_scale: str = "linear",
    transform: str | None = None,
    group_order: Sequence[str] | None = None,
    ci_fields: tuple[str, str] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> pathlib.Path:
    """Draw one sorted grouped-curve PDF from a CSV or JSON data file."""

    rows = load_csv_or_json(source)
    destination = pathlib.Path(output_pdf)
    plt = _configure_matplotlib(destination.parent)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    converter = _to_db if transform == "db" else _as_float
    line_count = _draw_grouped_curves(
        ax,
        rows,
        x_field=x_field,
        group_field=group_field,
        y_getter=lambda row: converter(row.get(y_field)),
        group_order=group_order,
        ci_fields=ci_fields,
        positive_only=y_scale == "log",
    )
    if line_count == 0:
        plt.close(fig)
        raise ValueError(f"no finite {y_field!r} curves found in {source}")
    _style_curve_axis(
        ax,
        x_label=x_label,
        y_label=y_label,
        y_scale=y_scale,
        y_limits=y_limits,
    )
    ax.legend(frameon=False, ncol=1)
    result = _save_figure(fig, destination)
    plt.close(fig)
    return result


def _set_categorical_ticks(ax: Any, x: np.ndarray, labels: Sequence[str]) -> None:
    ax.set_xticks(x, labels, rotation=24, ha="right", rotation_mode="anchor")


def plot_components_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot the component ablation as categorical comparisons, not one-point lines."""

    source = pathlib.Path(result_dir) / "ablation_summary.csv"
    rows = load_csv_or_json(source)
    destination = pathlib.Path(output_dir)
    plt = _configure_matplotlib(destination)
    preferred_order = (
        "scaled_4d",
        "old_stage1_ccop",
        "mksc_delay_ccop",
        "mksc_gi_1_no_refresh_ccop",
        "mksc_gi_4_no_refresh_ccop",
        "proposed",
    )
    by_variant = {str(row["variant"]): row for row in rows}
    variants = [name for name in preferred_order if name in by_variant]
    variants.extend(name for name in by_variant if name not in variants)
    if not variants:
        raise ValueError(f"no component rows found in {source}")
    x = np.arange(len(variants), dtype=float)
    labels = [_display_name(name) for name in variants]
    colors = [_series_color(name, index) for index, name in enumerate(variants)]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6), constrained_layout=True)

    width = 0.37
    rmse = np.asarray([_as_float(by_variant[name].get("position_rmse_m")) for name in variants])
    p95 = np.asarray([_as_float(by_variant[name].get("position_p95_m")) for name in variants])
    axes[0, 0].bar(x - width / 2, rmse, width, color="#0072B2", label="RMSE")
    axes[0, 0].bar(
        x + width / 2,
        p95,
        width,
        color="#E69F00",
        hatch="//",
        label="p95",
    )
    _style_curve_axis(
        axes[0, 0], x_label="", y_label="Position error (m)", y_scale="log"
    )
    axes[0, 0].legend(frameon=False, ncol=2)

    channel_mean = np.asarray(
        [_to_db(by_variant[name].get("channel_nmse_mean")) for name in variants]
    )
    channel_p95 = np.asarray(
        [_to_db(by_variant[name].get("channel_nmse_p95")) for name in variants]
    )
    axes[0, 1].bar(
        x - width / 2, channel_mean, width, color="#0072B2", label="Mean"
    )
    axes[0, 1].bar(
        x + width / 2,
        channel_p95,
        width,
        color="#E69F00",
        hatch="//",
        label="p95",
    )
    _style_curve_axis(axes[0, 1], x_label="", y_label="Channel NMSE (dB)")
    axes[0, 1].legend(frameon=False, ncol=2)

    outlier = np.asarray([_as_float(by_variant[name].get("outlier_rate")) for name in variants])
    ci_low = np.asarray([_as_float(by_variant[name].get("outlier_ci_low")) for name in variants])
    ci_high = np.asarray([_as_float(by_variant[name].get("outlier_ci_high")) for name in variants])
    lower = np.maximum(0.0, outlier - ci_low)
    upper = np.maximum(0.0, ci_high - outlier)
    valid_ci = np.isfinite(lower) & np.isfinite(upper)
    yerr = np.vstack(
        [np.where(valid_ci, lower, 0.0), np.where(valid_ci, upper, 0.0)]
    )
    axes[1, 0].bar(x, outlier, color=colors, yerr=yerr, capsize=2.5)
    _style_curve_axis(
        axes[1, 0],
        x_label="",
        y_label="Outlier probability",
        y_limits=(0.0, min(1.0, max(0.05, float(np.nanmax(ci_high)) * 1.12))),
    )

    runtime = np.asarray(
        [_as_float(by_variant[name].get("runtime_median_s")) for name in variants]
    )
    axes[1, 1].bar(x, runtime, color=colors)
    _style_curve_axis(
        axes[1, 1], x_label="", y_label="Median deployment runtime (s)"
    )
    for ax in axes.flat:
        _set_categorical_ticks(ax, x, labels)
    output = destination / "components_ablation.pdf"
    result = _save_figure(fig, output)
    plt.close(fig)
    return [result]


def _plot_ablation_snr_curves(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    prefix: str,
    group_order: Sequence[str] | None = None,
) -> list[pathlib.Path]:
    rows = load_csv_or_json(source)
    plt = _configure_matplotlib(destination)
    outputs: list[pathlib.Path] = []

    fig, ax = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
    _draw_grouped_curves(
        ax,
        rows,
        x_field="x_value",
        group_field="variant",
        y_getter=lambda row: (
            _first_finite(
                row, "peb_position_m_rms", "peb_position_m_mean"
            )
            if _is_reference(str(row.get("variant", "")))
            else _as_float(row.get("position_rmse_m"))
        ),
        group_order=group_order,
        positive_only=True,
    )
    _style_curve_axis(
        ax, x_label="SNR (dB)", y_label="Position RMSE / PEB (m)", y_scale="log"
    )
    ax.legend(frameon=False, ncol=2)
    outputs.append(_save_figure(fig, destination / f"{prefix}_position_rmse.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
    _draw_grouped_curves(
        ax,
        rows,
        x_field="x_value",
        group_field="variant",
        y_getter=lambda row: (
            _first_finite(
                row, "peb_position_m_rms", "peb_position_m_mean"
            )
            if _is_reference(str(row.get("variant", "")))
            else _as_float(row.get("position_conditional_rmse_m"))
        ),
        group_order=group_order,
        positive_only=True,
    )
    _style_curve_axis(
        ax,
        x_label="SNR (dB)",
        y_label="Correct-basin conditional RMSE / PEB (m)",
        y_scale="log",
    )
    ax.legend(frameon=False, ncol=2)
    outputs.append(
        _save_figure(
            fig, destination / f"{prefix}_position_conditional_rmse.pdf"
        )
    )
    plt.close(fig)

    outputs.append(
        plot_csv_or_json(
            source,
            destination / f"{prefix}_channel_nmse.pdf",
            x_field="x_value",
            y_field="channel_nmse_mean",
            group_field="variant",
            x_label="SNR (dB)",
            y_label="Channel NMSE (dB)",
            transform="db",
            group_order=group_order,
        )
    )
    outputs.append(
        plot_csv_or_json(
            source,
            destination / f"{prefix}_outlier_rate.pdf",
            x_field="x_value",
            y_field="outlier_rate",
            group_field="variant",
            x_label="SNR (dB)",
            y_label="Outlier probability",
            group_order=group_order,
            ci_fields=("outlier_ci_low", "outlier_ci_high"),
            y_limits=(0.0, 1.0),
        )
    )
    return outputs


def plot_snr_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot the internal SNR ablation with numerically sorted SNR values."""

    source = pathlib.Path(result_dir) / "ablation_summary.csv"
    order = (
        "scaled_4d",
        "old_stage1_ccop",
        "mksc_gi_4_no_refresh_ccop",
        "proposed",
        "free_jones_peb",
    )
    return _plot_ablation_snr_curves(
        source, pathlib.Path(output_dir), prefix="snr", group_order=order
    )


def plot_compression_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot the matched-compression SNR sweep."""

    source = pathlib.Path(result_dir) / "ablation_summary.csv"
    order = ("raw_delay_gi_ccop", "proposed")
    return _plot_ablation_snr_curves(
        source, pathlib.Path(output_dir), prefix="compression", group_order=order
    )


def plot_receiver_information_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot receiver-mode curves and their matching data-only PEB references."""

    source = pathlib.Path(result_dir) / "ablation_summary.csv"
    rows = load_csv_or_json(source)
    destination = pathlib.Path(output_dir)
    plt = _configure_matplotlib(destination)
    mode_order = (
        "proposed_scalar",
        "proposed_dual_pol",
        "proposed_full_6d",
        "free_jones_peb_scalar",
        "free_jones_peb_dual_pol",
        "free_jones_peb_full_6d",
    )
    outputs: list[pathlib.Path] = []

    fig, ax = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
    _draw_grouped_curves(
        ax,
        rows,
        x_field="x_value",
        group_field="variant",
        y_getter=lambda row: (
            _first_finite(
                row, "peb_position_m_rms", "peb_position_m_mean"
            )
            if _is_reference(str(row.get("variant", "")))
            else _as_float(row.get("position_rmse_m"))
        ),
        group_order=mode_order,
        positive_only=True,
    )
    _style_curve_axis(
        ax, x_label="SNR (dB)", y_label="Position RMSE / PEB (m)", y_scale="log"
    )
    ax.legend(frameon=False, ncol=2)
    outputs.append(_save_figure(fig, destination / "receiver_position_rmse.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
    _draw_grouped_curves(
        ax,
        rows,
        x_field="x_value",
        group_field="variant",
        y_getter=lambda row: (
            _first_finite(
                row, "peb_position_m_rms", "peb_position_m_mean"
            )
            if _is_reference(str(row.get("variant", "")))
            else _as_float(row.get("position_conditional_rmse_m"))
        ),
        group_order=mode_order,
        positive_only=True,
    )
    _style_curve_axis(
        ax,
        x_label="SNR (dB)",
        y_label="Correct-basin conditional RMSE / PEB (m)",
        y_scale="log",
    )
    ax.legend(frameon=False, ncol=2)
    outputs.append(
        _save_figure(fig, destination / "receiver_position_conditional_rmse.pdf")
    )
    plt.close(fig)

    proposed_order = ("proposed_scalar", "proposed_dual_pol", "proposed_full_6d")
    outputs.append(
        plot_csv_or_json(
            source,
            destination / "receiver_channel_nmse.pdf",
            x_field="x_value",
            y_field="channel_nmse_mean",
            group_field="variant",
            x_label="SNR (dB)",
            y_label="Channel NMSE (dB)",
            transform="db",
            group_order=proposed_order,
        )
    )
    outputs.append(
        plot_csv_or_json(
            source,
            destination / "receiver_outlier_rate.pdf",
            x_field="x_value",
            y_field="outlier_rate",
            group_field="variant",
            x_label="SNR (dB)",
            y_label="Outlier probability",
            group_order=proposed_order,
            ci_fields=("outlier_ci_low", "outlier_ci_high"),
            y_limits=(0.0, 1.0),
        )
    )
    return outputs


def plot_benchmark_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot every baseline present in benchmark_summary.csv, including bounds."""

    source = pathlib.Path(result_dir) / "benchmark_summary.csv"
    rows = load_csv_or_json(source)
    selected = {
        "als_cpd",
        "scaled_4d",
        "nf_ris_groupomp_localgrid_wls",
        "ris_momp",
        "mksc_ccop",
        "peb",
        "constrained_jones_peb",
    }
    rows = [row for row in rows if str(row.get("baseline", "")) in selected]
    destination = pathlib.Path(output_dir)
    plt = _configure_matplotlib(destination)
    available = _ordered_unique([row.get("baseline", "") for row in rows])
    preferred = (
        "als_cpd",
        "scaled_4d",
        "nf_ris_groupomp_localgrid_wls",
        "ris_momp",
        "mksc_ccop",
        "peb",
        "constrained_jones_peb",
    )
    order = [name for name in preferred if name in available]
    order.extend(name for name in available if name not in order)
    outputs: list[pathlib.Path] = []

    fig, ax = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
    _draw_grouped_curves(
        ax,
        rows,
        x_field="snr_db",
        group_field="baseline",
        y_getter=lambda row: (
            _first_finite(
                row, "peb_position_m_rms", "peb_position_m_mean"
            )
            if _is_reference(str(row.get("baseline", "")))
            else _first_finite(
                row, "position_rmse_m", "position_rmse_m_mean"
            )
        ),
        group_order=order,
        positive_only=True,
    )
    _style_curve_axis(
        ax, x_label="SNR (dB)", y_label="Position RMSE / PEB (m)", y_scale="log"
    )
    ax.legend(frameon=False, ncol=2)
    outputs.append(_save_figure(fig, destination / "benchmark_position_rmse.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
    _draw_grouped_curves(
        ax,
        rows,
        x_field="snr_db",
        group_field="baseline",
        y_getter=lambda row: (
            _first_finite(
                row, "peb_position_m_rms", "peb_position_m_mean"
            )
            if _is_reference(str(row.get("baseline", "")))
            else _as_float(row.get("position_conditional_rmse_m"))
        ),
        group_order=order,
        positive_only=True,
    )
    _style_curve_axis(
        ax,
        x_label="SNR (dB)",
        y_label="Correct-basin conditional RMSE / PEB (m)",
        y_scale="log",
    )
    ax.legend(frameon=False, ncol=2)
    outputs.append(
        _save_figure(fig, destination / "benchmark_position_conditional_rmse.pdf")
    )
    plt.close(fig)

    outputs.append(
        plot_csv_or_json(
            source,
            destination / "benchmark_channel_nmse.pdf",
            x_field="snr_db",
            y_field="y_nmse_mean",
            group_field="baseline",
            x_label="SNR (dB)",
            y_label="Observed-data NMSE",
            y_scale="log",
            group_order=order,
        )
    )
    outputs.append(
        plot_csv_or_json(
            source,
            destination / "benchmark_outlier_rate.pdf",
            x_field="snr_db",
            y_field="outlier_rate",
            group_field="baseline",
            x_label="SNR (dB)",
            y_label="Outlier probability",
            group_order=order,
            ci_fields=("outlier_ci_low", "outlier_ci_high"),
            y_limits=(0.0, 1.0),
        )
    )
    return outputs


def _unique_legend(fig: Any, axes: Sequence[Any], *, ncol: int) -> None:
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.015),
            frameon=False,
            ncol=ncol,
        )


def _plot_faceted_sweep(
    rows: Sequence[Mapping[str, Any]],
    destination: pathlib.Path,
    *,
    output_name: str,
    x_names: Sequence[str],
    metric_getter: Callable[[Mapping[str, Any]], float],
    y_label: str,
    y_scale: str = "linear",
    y_limits: tuple[float, float] | None = None,
    ci_fields: tuple[str, str] | None = None,
) -> pathlib.Path:
    plt = _configure_matplotlib(destination)
    columns = 2 if len(x_names) <= 4 else 3
    rows_count = int(math.ceil(len(x_names) / columns))
    fig, axes_array = plt.subplots(
        rows_count,
        columns,
        figsize=(4.25 * columns, 3.2 * rows_count),
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axes_array.flat)
    for ax, x_name in zip(axes, x_names):
        selected = [row for row in rows if str(row.get("x_name", "")) == x_name]
        _draw_grouped_curves(
            ax,
            selected,
            x_field="x_value",
            group_field="variant",
            y_getter=metric_getter,
            ci_fields=ci_fields,
            positive_only=y_scale == "log",
        )
        _style_curve_axis(
            ax,
            x_label=X_LABELS.get(x_name, x_name),
            y_label=y_label,
            y_scale=y_scale,
            y_limits=y_limits,
        )
    for ax in axes[len(x_names) :]:
        ax.set_visible(False)
    _unique_legend(fig, axes[: len(x_names)], ncol=3)
    result = _save_figure(fig, destination / output_name)
    plt.close(fig)
    return result


def plot_maxwell_mismatch_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot the four Maxwell/model-mismatch axes with one metric per PDF."""

    source = pathlib.Path(result_dir) / "robustness_summary.csv"
    rows = load_csv_or_json(source)
    destination = pathlib.Path(output_dir)
    x_names = (
        "bs_sensor_position_std_mm",
        "evs_gain_std",
        "evs_phase_deg",
        "ris_bs_angle_deg",
    )
    return [
        _plot_faceted_sweep(
            rows,
            destination,
            output_name="maxwell_mismatch_position_rmse.pdf",
            x_names=x_names,
            metric_getter=lambda row: _as_float(row.get("position_rmse_m")),
            y_label="Position RMSE (m)",
            y_scale="log",
        ),
        _plot_faceted_sweep(
            rows,
            destination,
            output_name="maxwell_mismatch_channel_nmse.pdf",
            x_names=x_names,
            metric_getter=lambda row: _to_db(row.get("channel_nmse_mean")),
            y_label="Channel NMSE (dB)",
        ),
        _plot_faceted_sweep(
            rows,
            destination,
            output_name="maxwell_mismatch_outlier_rate.pdf",
            x_names=x_names,
            metric_getter=lambda row: _as_float(row.get("outlier_rate")),
            y_label="Outlier probability",
            y_limits=(0.0, 1.0),
            ci_fields=("outlier_ci_low", "outlier_ci_high"),
        ),
    ]


def plot_colored_noise_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot the colored-noise boundary experiment."""

    source = pathlib.Path(result_dir) / "robustness_summary.csv"
    destination = pathlib.Path(output_dir)
    order = ("raw_delay_gi_ccop", "proposed")
    return [
        plot_csv_or_json(
            source,
            destination / "colored_noise_position_rmse.pdf",
            x_field="x_value",
            y_field="position_rmse_m",
            group_field="variant",
            x_label=X_LABELS["colored_noise_rho"],
            y_label="Position RMSE (m)",
            y_scale="log",
            group_order=order,
        ),
        plot_csv_or_json(
            source,
            destination / "colored_noise_channel_nmse.pdf",
            x_field="x_value",
            y_field="channel_nmse_mean",
            group_field="variant",
            x_label=X_LABELS["colored_noise_rho"],
            y_label="Channel NMSE (dB)",
            transform="db",
            group_order=order,
        ),
        plot_csv_or_json(
            source,
            destination / "colored_noise_outlier_rate.pdf",
            x_field="x_value",
            y_field="outlier_rate",
            group_field="variant",
            x_label=X_LABELS["colored_noise_rho"],
            y_label="Outlier probability",
            group_order=order,
            ci_fields=("outlier_ci_low", "outlier_ci_high"),
            y_limits=(0.0, 1.0),
        ),
    ]


def plot_scaling_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot accuracy, runtime, and failures over all five scaling axes."""

    source = pathlib.Path(result_dir) / "robustness_summary.csv"
    rows = load_csv_or_json(source)
    destination = pathlib.Path(output_dir)
    x_names = ("N", "T", "M_A", "M_R", "K")
    return [
        _plot_faceted_sweep(
            rows,
            destination,
            output_name="scaling_position_rmse.pdf",
            x_names=x_names,
            metric_getter=lambda row: _as_float(row.get("position_rmse_m")),
            y_label="Position RMSE (m)",
            y_scale="log",
        ),
        _plot_faceted_sweep(
            rows,
            destination,
            output_name="scaling_runtime.pdf",
            x_names=x_names,
            metric_getter=lambda row: _as_float(row.get("runtime_median_s")),
            y_label="Median deployment runtime (s)",
            y_scale="log",
        ),
        _plot_faceted_sweep(
            rows,
            destination,
            output_name="scaling_outlier_rate.pdf",
            x_names=x_names,
            metric_getter=lambda row: _as_float(row.get("outlier_rate")),
            y_label="Outlier probability",
            y_limits=(0.0, 1.0),
            ci_fields=("outlier_ci_low", "outlier_ci_high"),
        ),
    ]


def _position_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    z_value: float,
    variant: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if str(row.get("variant", "")) == variant
        and np.isclose(_as_float(row.get("p_true_z")), z_value)
    ]
    x_values = np.asarray(sorted({_as_float(row.get("p_true_x")) for row in selected}))
    y_values = np.asarray(sorted({_as_float(row.get("p_true_y")) for row in selected}))
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    for row in selected:
        x_value = _as_float(row.get("p_true_x"))
        y_value = _as_float(row.get("p_true_y"))
        grid[y_index[y_value], x_index[x_value]] = _as_float(row.get(metric))
    return x_values, y_values, grid


def _positive_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([array[np.isfinite(array) & (array > 0.0)] for array in arrays])
    if values.size == 0:
        raise ValueError("heatmap has no positive finite values")
    lower = float(np.min(values))
    upper = float(np.max(values))
    if np.isclose(lower, upper):
        lower *= 0.9
        upper *= 1.1
    return lower, upper


def plot_position_generalization_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot gridded UE-position maps directly from the position summary CSV."""

    source = pathlib.Path(result_dir) / "position_generalization_summary.csv"
    rows = load_csv_or_json(source)
    destination = pathlib.Path(output_dir)
    plt = _configure_matplotlib(destination)
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

    z_values = sorted({_as_float(row.get("p_true_z")) for row in rows})
    outputs: list[pathlib.Path] = []
    for z_value in z_values:
        x, y, proposed_rmse = _position_grid(
            rows, z_value=z_value, variant="proposed", metric="position_rmse_m"
        )
        _, _, scaled_rmse = _position_grid(
            rows, z_value=z_value, variant="scaled_4d", metric="position_rmse_m"
        )
        _, _, proposed_outlier = _position_grid(
            rows, z_value=z_value, variant="proposed", metric="outlier_rate"
        )
        _, _, scaled_outlier = _position_grid(
            rows, z_value=z_value, variant="scaled_4d", metric="outlier_rate"
        )
        _, _, peb = _position_grid(
            rows, z_value=z_value, variant="proposed", metric="peb_position_m_rms"
        )
        rmse_limits = _positive_limits(proposed_rmse, scaled_rmse)
        peb_limits = _positive_limits(peb)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log10(proposed_rmse / scaled_rmse)
        finite_ratio = np.abs(log_ratio[np.isfinite(log_ratio)])
        ratio_limit = max(0.05, float(np.max(finite_ratio)))

        fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), constrained_layout=True)
        heatmaps = (
            (
                axes[0, 0],
                proposed_rmse,
                "Proposed RMSE (m)",
                "viridis",
                LogNorm(*rmse_limits),
            ),
            (
                axes[0, 1],
                scaled_rmse,
                "Scaled 4-D RMSE (m)",
                "viridis",
                LogNorm(*rmse_limits),
            ),
            (
                axes[0, 2],
                log_ratio,
                r"$\log_{10}$(Proposed / Scaled 4-D RMSE)",
                "coolwarm",
                TwoSlopeNorm(vmin=-ratio_limit, vcenter=0.0, vmax=ratio_limit),
            ),
            (
                axes[1, 0],
                proposed_outlier,
                "Proposed outlier probability",
                "magma",
                Normalize(0.0, 1.0),
            ),
            (
                axes[1, 1],
                scaled_outlier,
                "Scaled 4-D outlier probability",
                "magma",
                Normalize(0.0, 1.0),
            ),
            (
                axes[1, 2],
                peb,
                "Data-only PEB (m)",
                "cividis",
                LogNorm(*peb_limits),
            ),
        )
        for ax, values, title, cmap, norm in heatmaps:
            image = ax.pcolormesh(x, y, values, shading="nearest", cmap=cmap, norm=norm)
            fig.colorbar(image, ax=ax, pad=0.02)
            ax.set_title(title)
            ax.set_xlabel("UE x (m)")
            ax.set_ylabel("UE y (m)")
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle(f"UE position generalization, z = {z_value:g} m")
        output = destination / f"position_generalization_z{z_value:g}.pdf"
        outputs.append(_save_figure(fig, output))
        plt.close(fig)
    return outputs


def plot_evs_resolvability_experiment(
    result_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Plot EVS resolvability by receiver mode, delay separation, and overlap."""

    source = pathlib.Path(result_dir) / "evs_resolvability_summary.csv"
    rows = load_csv_or_json(source)
    destination = pathlib.Path(output_dir)
    plt = _configure_matplotlib(destination)
    modes = ("scalar", "dual_pol", "full_6d")
    overlaps = sorted(
        {_as_float(row.get("polarization_overlap_target")) for row in rows}
    )
    overlap_colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, max(1, len(overlaps)))
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), constrained_layout=True)
    for column, mode in enumerate(modes):
        selected_mode = [row for row in rows if str(row.get("receiver_mode")) == mode]
        for index, overlap in enumerate(overlaps):
            selected = [
                row
                for row in selected_mode
                if np.isclose(_as_float(row.get("polarization_overlap_target")), overlap)
            ]
            selected.sort(key=lambda row: _as_float(row.get("target_delay_separation_ns")))
            x = np.asarray(
                [_as_float(row.get("target_delay_separation_ns")) for row in selected]
            )
            resolution = np.asarray(
                [_as_float(row.get("resolution_probability")) for row in selected]
            )
            low = np.asarray([_as_float(row.get("resolution_ci_low")) for row in selected])
            high = np.asarray([_as_float(row.get("resolution_ci_high")) for row in selected])
            position_p95 = np.asarray(
                [_as_float(row.get("position_p95_m")) for row in selected]
            )
            label = f"Overlap = {overlap:g}"
            axes[0, column].plot(
                x,
                resolution,
                color=overlap_colors[index],
                marker=("o", "s", "^", "D")[index % 4],
                markersize=4.5,
                linewidth=1.7,
                label=label,
            )
            band = np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
            if np.any(band):
                axes[0, column].fill_between(
                    x[band],
                    low[band],
                    high[band],
                    color=overlap_colors[index],
                    alpha=0.12,
                    linewidth=0.0,
                )
            axes[1, column].plot(
                x,
                position_p95,
                color=overlap_colors[index],
                marker=("o", "s", "^", "D")[index % 4],
                markersize=4.5,
                linewidth=1.7,
                label=label,
            )
        axes[0, column].set_title(_display_name(f"proposed_{mode}"))
        _style_curve_axis(
            axes[0, column],
            x_label="Delay separation (ns)",
            y_label="Resolution probability",
            y_limits=(0.0, 1.0),
        )
        axes[0, column].set_xscale("log")
        _style_curve_axis(
            axes[1, column],
            x_label="Delay separation (ns)",
            y_label="Position-error p95 (m)",
            y_scale="log",
        )
        axes[1, column].set_xscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.035),
            frameon=False,
            ncol=len(overlaps),
        )
    output = destination / "evs_resolvability.pdf"
    result = _save_figure(fig, output)
    plt.close(fig)
    return [result]


PLOTTERS: dict[
    str,
    Callable[[str | pathlib.Path, str | pathlib.Path], list[pathlib.Path]],
] = {
    "components": plot_components_experiment,
    "snr": plot_snr_experiment,
    "receiver": plot_receiver_information_experiment,
    "compression": plot_compression_experiment,
    "benchmark": plot_benchmark_experiment,
    "maxwell_mismatch": plot_maxwell_mismatch_experiment,
    "colored_noise": plot_colored_noise_experiment,
    "positions": plot_position_generalization_experiment,
    "scaling": plot_scaling_experiment,
    "evs_resolvability": plot_evs_resolvability_experiment,
}


def plot_all_final_mksc_ccop_results(
    result_root: str | pathlib.Path = DEFAULT_RESULT_ROOT,
    output_root: str | pathlib.Path | None = None,
    *,
    experiments: Sequence[str] | None = None,
    strict: bool = True,
) -> dict[str, list[pathlib.Path]]:
    """Run the experiment-specific plotters over one final-result directory."""

    root = pathlib.Path(result_root)
    destination = pathlib.Path(output_root) if output_root is not None else root / "plots"
    selected = list(experiments) if experiments is not None else list(PLOTTERS)
    unknown = [name for name in selected if name not in PLOTTERS]
    if unknown:
        raise ValueError(f"unknown experiments: {', '.join(unknown)}")
    generated: dict[str, list[pathlib.Path]] = {}
    for name in selected:
        experiment_dir = root / EXPERIMENT_DIRECTORIES[name]
        if not experiment_dir.is_dir():
            if strict:
                raise FileNotFoundError(experiment_dir)
            continue
        generated[name] = PLOTTERS[name](experiment_dir, destination / name)
    return generated


def _parse_experiments(value: str) -> list[str] | None:
    if value.strip().lower() == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=pathlib.Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=None,
        help="default: <result-root>/plots",
    )
    parser.add_argument(
        "--experiments",
        default="all",
        help="comma-separated keys or 'all': " + ",".join(PLOTTERS),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="skip experiment directories that are not present",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generated = plot_all_final_mksc_ccop_results(
        args.result_root,
        args.output_root,
        experiments=_parse_experiments(args.experiments),
        strict=not args.skip_missing,
    )
    for name, paths in generated.items():
        print(f"{name}: {len(paths)} PDF(s)")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
