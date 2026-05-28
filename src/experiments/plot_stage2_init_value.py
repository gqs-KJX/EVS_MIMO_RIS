"""Plot Stage-II initialization-value experiment summaries.

Reads the default clean summary and, when present, the weak-initialization
summary. Figures are written under results/stage2_init_value/figures by
default.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PIPELINE_STAGE1_VP = "Stage-I + VP"
PIPELINE_STAGE1_STAGE2 = "Stage-I + Stage-II"
PIPELINE_STAGE1_STAGE2_VP = "Stage-I + Stage-II + VP"

COMPARE_PIPELINES = [PIPELINE_STAGE1_VP, PIPELINE_STAGE1_STAGE2_VP]
PHYSICAL_PIPELINES = [
    PIPELINE_STAGE1_VP,
    PIPELINE_STAGE1_STAGE2,
    PIPELINE_STAGE1_STAGE2_VP,
]

PIPELINE_STYLES = {
    PIPELINE_STAGE1_VP: {"marker": "o", "linestyle": "-", "color": "tab:blue"},
    PIPELINE_STAGE1_STAGE2: {"marker": "^", "linestyle": ":", "color": "tab:green"},
    PIPELINE_STAGE1_STAGE2_VP: {"marker": "s", "linestyle": "--", "color": "tab:orange"},
}

PHYSICAL_METRICS = [
    (
        "delay_geometry_consistency_RMSE_mean",
        "Delay-Geometry RMSE",
        "RMSE (s)",
    ),
    (
        "compressed_ris_manifold_residual_mean",
        "RIS Manifold Residual",
        "Relative residual",
    ),
    (
        "evs_maxwell_consistency_residual_mean",
        "EVS Maxwell Residual",
        "Relative residual",
    ),
]


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_token(value: Any) -> str:
    text = str(value).strip().replace("+", "plus").replace("-", "m")
    text = text.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_") or "value"


def _format_number(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return str(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def _format_context(source: str, init_mode: str, t_dim: Any | None, snr_db: Any | None, ris_shape: str) -> str:
    parts = [f"source={source}", f"init={init_mode}", f"RIS={ris_shape}"]
    if t_dim is not None:
        parts.append(f"T={_format_number(t_dim)}")
    if snr_db is not None:
        parts.append(f"SNR={_format_number(snr_db)} dB")
    return ", ".join(parts)


def _read_summary(path: pathlib.Path, source: str, *, required: bool) -> list[dict]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required summary not found: {path}")
        print(f"Optional summary not found, skipping: {path}")
        return []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row = dict(row)
            row["source"] = source
            row.setdefault("init_mode", "clean" if source == "main" else "weak")
            if not row["init_mode"]:
                row["init_mode"] = "clean" if source == "main" else "weak"
            row.setdefault("init_mode_effective", row["init_mode"])
            row.setdefault("init_warning", "")
            rows.append(row)
    print(f"Read {len(rows)} rows from {path}")
    return rows


def _filter_rows(rows: list[dict], **filters: Any) -> list[dict]:
    filtered = []
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            filtered.append(row)
    return filtered


def _unique_sorted(rows: list[dict], key: str, numeric: bool = False) -> list[Any]:
    values = {row.get(key) for row in rows if row.get(key) not in ("", None)}
    if numeric:
        return sorted(values, key=lambda value: float(value))
    return sorted(values)


def _series(
    rows: list[dict],
    *,
    pipeline: str,
    x_key: str,
    y_key: str,
    scale: float = 1.0,
    **filters: Any,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in _filter_rows(rows, pipeline=pipeline, **filters):
        x_value = _as_float(row.get(x_key))
        y_value = _as_float(row.get(y_key))
        if x_value is None or y_value is None:
            continue
        grouped[x_value].append(scale * y_value)
    if not grouped:
        return np.asarray([]), np.asarray([])
    xs = np.asarray(sorted(grouped), dtype=float)
    ys = np.asarray([float(np.mean(grouped[x])) for x in xs], dtype=float)
    return xs, ys


def _plot_pipeline_lines(
    ax: plt.Axes,
    rows: list[dict],
    *,
    pipelines: list[str],
    x_key: str,
    y_key: str,
    scale: float = 1.0,
    **filters: Any,
) -> bool:
    plotted = False
    for pipeline in pipelines:
        xs, ys = _series(
            rows,
            pipeline=pipeline,
            x_key=x_key,
            y_key=y_key,
            scale=scale,
            **filters,
        )
        if xs.size == 0:
            continue
        ax.plot(xs, ys, label=pipeline, **PIPELINE_STYLES.get(pipeline, {}))
        plotted = True
    return plotted


def _maybe_log_y(ax: plt.Axes, values: list[float]) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size > 0 and np.all(finite > 0.0):
        ax.set_yscale("log")


def _save(fig: plt.Figure, path: pathlib.Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _figure_name(prefix: str, source: str, init_mode: str, **parts: Any) -> str:
    tokens = [prefix, _safe_token(source), _safe_token(init_mode)]
    for key, value in parts.items():
        tokens.append(f"{_safe_token(key)}{_safe_token(value)}")
    return "_".join(tokens) + ".png"


def _plot_position_vs_snr(rows: list[dict], out_dir: pathlib.Path, dpi: int) -> int:
    count = 0
    contexts = sorted(
        {
            (row["source"], row["init_mode"], row["T"], row["ris_shape"])
            for row in rows
            if row.get("pipeline") in COMPARE_PIPELINES
        },
        key=lambda item: (item[0], item[1], int(float(item[2])), item[3]),
    )
    for source, init_mode, t_dim, ris_shape in contexts:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        plotted = _plot_pipeline_lines(
            ax,
            rows,
            pipelines=COMPARE_PIPELINES,
            x_key="SNR_dB",
            y_key="position_error_mean",
            source=source,
            init_mode=init_mode,
            T=t_dim,
            ris_shape=ris_shape,
        )
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title("Position RMSE vs SNR\n" + _format_context(source, init_mode, t_dim, None, ris_shape))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Position RMSE (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = out_dir / _figure_name(
            "position_rmse_vs_snr", source, init_mode, T=t_dim, RIS=ris_shape
        )
        _save(fig, path, dpi)
        count += 1
    return count


def _plot_success_vs_snr(rows: list[dict], out_dir: pathlib.Path, dpi: int) -> int:
    count = 0
    contexts = sorted(
        {
            (row["source"], row["init_mode"], row["T"], row["ris_shape"])
            for row in rows
            if row.get("pipeline") in COMPARE_PIPELINES
        },
        key=lambda item: (item[0], item[1], int(float(item[2])), item[3]),
    )
    for source, init_mode, t_dim, ris_shape in contexts:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
        plotted_any = False
        for ax, key, title in [
            (axes[0], "success_pos_20cm_rate", "Success: position < 20 cm"),
            (axes[1], "success_pos_50cm_rate", "Success: position < 50 cm"),
        ]:
            plotted = _plot_pipeline_lines(
                ax,
                rows,
                pipelines=COMPARE_PIPELINES,
                x_key="SNR_dB",
                y_key=key,
                scale=100.0,
                source=source,
                init_mode=init_mode,
                T=t_dim,
                ris_shape=ris_shape,
            )
            plotted_any = plotted_any or plotted
            ax.set_title(title)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("Success rate (%)")
            ax.set_ylim(-2.0, 102.0)
            ax.grid(True, alpha=0.3)
            ax.legend()
        if not plotted_any:
            plt.close(fig)
            continue
        fig.suptitle("VP Success Rate vs SNR\n" + _format_context(source, init_mode, t_dim, None, ris_shape))
        path = out_dir / _figure_name(
            "vp_success_vs_snr", source, init_mode, T=t_dim, RIS=ris_shape
        )
        _save(fig, path, dpi)
        count += 1
    return count


def _plot_p90_vs_snr(rows: list[dict], out_dir: pathlib.Path, dpi: int) -> int:
    count = 0
    contexts = sorted(
        {
            (row["source"], row["init_mode"], row["T"], row["ris_shape"])
            for row in rows
            if row.get("pipeline") in COMPARE_PIPELINES
        },
        key=lambda item: (item[0], item[1], int(float(item[2])), item[3]),
    )
    for source, init_mode, t_dim, ris_shape in contexts:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        plotted = _plot_pipeline_lines(
            ax,
            rows,
            pipelines=COMPARE_PIPELINES,
            x_key="SNR_dB",
            y_key="position_error_p90",
            source=source,
            init_mode=init_mode,
            T=t_dim,
            ris_shape=ris_shape,
        )
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title("90th Percentile Position Error vs SNR\n" + _format_context(source, init_mode, t_dim, None, ris_shape))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("90th percentile position error (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = out_dir / _figure_name(
            "position_p90_vs_snr", source, init_mode, T=t_dim, RIS=ris_shape
        )
        _save(fig, path, dpi)
        count += 1
    return count


def _plot_physical_consistency_vs_snr(rows: list[dict], out_dir: pathlib.Path, dpi: int) -> int:
    count = 0
    contexts = sorted(
        {
            (row["source"], row["init_mode"], row["T"], row["ris_shape"])
            for row in rows
            if row.get("pipeline") in PHYSICAL_PIPELINES
        },
        key=lambda item: (item[0], item[1], int(float(item[2])), item[3]),
    )
    for source, init_mode, t_dim, ris_shape in contexts:
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        plotted_any = False
        for ax, (key, title, ylabel) in zip(axes, PHYSICAL_METRICS):
            plotted = _plot_pipeline_lines(
                ax,
                rows,
                pipelines=PHYSICAL_PIPELINES,
                x_key="SNR_dB",
                y_key=key,
                source=source,
                init_mode=init_mode,
                T=t_dim,
                ris_shape=ris_shape,
            )
            plotted_any = plotted_any or plotted
            values = [
                y
                for pipeline in PHYSICAL_PIPELINES
                for y in _series(
                    rows,
                    pipeline=pipeline,
                    x_key="SNR_dB",
                    y_key=key,
                    source=source,
                    init_mode=init_mode,
                    T=t_dim,
                    ris_shape=ris_shape,
                )[1]
            ]
            _maybe_log_y(ax, values)
            ax.set_title(title)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        if not plotted_any:
            plt.close(fig)
            continue
        fig.suptitle("Physical Consistency Residuals vs SNR\n" + _format_context(source, init_mode, t_dim, None, ris_shape))
        path = out_dir / _figure_name(
            "physical_consistency_vs_snr", source, init_mode, T=t_dim, RIS=ris_shape
        )
        _save(fig, path, dpi)
        count += 1
    return count


def _plot_position_vs_t(rows: list[dict], out_dir: pathlib.Path, dpi: int) -> int:
    count = 0
    contexts = sorted(
        {
            (row["source"], row["init_mode"], row["SNR_dB"], row["ris_shape"])
            for row in rows
            if row.get("pipeline") in COMPARE_PIPELINES
        },
        key=lambda item: (item[0], item[1], float(item[2]), item[3]),
    )
    for source, init_mode, snr_db, ris_shape in contexts:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        plotted = _plot_pipeline_lines(
            ax,
            rows,
            pipelines=COMPARE_PIPELINES,
            x_key="T",
            y_key="position_error_mean",
            source=source,
            init_mode=init_mode,
            SNR_dB=snr_db,
            ris_shape=ris_shape,
        )
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title("Position RMSE vs T\n" + _format_context(source, init_mode, None, snr_db, ris_shape))
        ax.set_xlabel("RIS training length T")
        ax.set_ylabel("Position RMSE (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = out_dir / _figure_name(
            "position_rmse_vs_t", source, init_mode, SNR=snr_db, RIS=ris_shape
        )
        _save(fig, path, dpi)
        count += 1
    return count


def _plot_improvement_heatmap(rows: list[dict], out_dir: pathlib.Path, dpi: int) -> int:
    count = 0
    contexts = sorted(
        {
            (row["source"], row["init_mode"], row["ris_shape"])
            for row in rows
            if row.get("improvement_rate_position") not in ("", None)
        },
        key=lambda item: (item[0], item[1], item[2]),
    )
    for source, init_mode, ris_shape in contexts:
        context_rows = _filter_rows(
            rows,
            source=source,
            init_mode=init_mode,
            ris_shape=ris_shape,
            pipeline=PIPELINE_STAGE1_STAGE2_VP,
        )
        if not context_rows:
            context_rows = _filter_rows(rows, source=source, init_mode=init_mode, ris_shape=ris_shape)
        snrs = _unique_sorted(context_rows, "SNR_dB", numeric=True)
        t_values = _unique_sorted(context_rows, "T", numeric=True)
        if not snrs or not t_values:
            continue

        matrix = np.full((len(t_values), len(snrs)), np.nan, dtype=float)
        for row in context_rows:
            value = _as_float(row.get("improvement_rate_position"))
            if value is None:
                continue
            try:
                i = t_values.index(row["T"])
                j = snrs.index(row["SNR_dB"])
            except ValueError:
                continue
            matrix[i, j] = value
        if not np.any(np.isfinite(matrix)):
            continue

        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="lightgray")
        image = ax.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
        ax.set_title("Position Improvement Rate: Stage-II+VP over Stage-I+VP\n" + _format_context(source, init_mode, None, None, ris_shape))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("RIS training length T")
        ax.set_xticks(np.arange(len(snrs)))
        ax.set_xticklabels([_format_number(value) for value in snrs])
        ax.set_yticks(np.arange(len(t_values)))
        ax.set_yticklabels([_format_number(value) for value in t_values])
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Improvement rate")
        path = out_dir / _figure_name(
            "improvement_rate_position_heatmap", source, init_mode, RIS=ris_shape
        )
        _save(fig, path, dpi)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stage-II initialization-value figures from summary CSV files."
    )
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        default=pathlib.Path("results/stage2_init_value/summary.csv"),
        help="Main experiment summary CSV.",
    )
    parser.add_argument(
        "--weak-summary",
        type=pathlib.Path,
        default=pathlib.Path("results/stage2_init_value_weak/summary.csv"),
        help="Optional weak-initialization summary CSV.",
    )
    parser.add_argument(
        "--no-weak",
        action="store_true",
        help="Do not try to read the optional weak-initialization summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/stage2_init_value/figures"),
        help="Directory where PNG figures are written.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    try:
        rows = _read_summary(args.summary, "main", required=True)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    if not args.no_weak:
        rows.extend(_read_summary(args.weak_summary, "weak", required=False))

    if not rows:
        print("No summary rows available; no figures generated.", file=sys.stderr)
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    total += _plot_position_vs_snr(rows, args.output_dir, args.dpi)
    total += _plot_success_vs_snr(rows, args.output_dir, args.dpi)
    total += _plot_p90_vs_snr(rows, args.output_dir, args.dpi)
    total += _plot_physical_consistency_vs_snr(rows, args.output_dir, args.dpi)
    total += _plot_position_vs_t(rows, args.output_dir, args.dpi)
    total += _plot_improvement_heatmap(rows, args.output_dir, args.dpi)

    print(f"Generated {total} figure(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
