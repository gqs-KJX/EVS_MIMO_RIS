"""Run benchmark comparison baselines for EVS-MIMO-RIS paper figures."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
import pathlib
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.baselines.als_cpd import run_als_cpd_baseline
    from src.baselines.common import BaselineResult, data_hash, make_baseline_row, y_noisy_hash
    from src.baselines.far_field_omp import run_far_field_omp_baseline
    from src.baselines.near_field_mmpsr import run_near_field_mmpsr_baseline
    from src.baselines.nf_ris_groupomp_localgrid_wls import run_nf_ris_groupomp_localgrid_wls_baseline
    from src.baselines.ris_momp import run_ris_momp_baseline
    from src.config import default_config
    from src.main_single_proposed import _make_data, run_single_proposed_diagnostic
    from src.metrics import relative_nmse, position_rmse
    from src.experiments.run_paper_ablation_figures import _peb_from_efim, set_number_of_ris_paths
    from src.experiments.resource_control import (
        apply_thread_limits,
        assert_row_is_light,
        memory_snapshot_mb,
        resolve_hybrid_resources,
        thread_limit_context,
        trim_memory,
    )
    from src.experiments.progress_logger import ProgressLogger
    from src.experiments.cli_common import (
        add_io_args,
        add_mc_args,
        add_progress_args,
        add_resource_args,
        normalize_blas_threads,
    )
else:
    from ..baselines.als_cpd import run_als_cpd_baseline
    from ..baselines.common import BaselineResult, data_hash, make_baseline_row, y_noisy_hash
    from ..baselines.far_field_omp import run_far_field_omp_baseline
    from ..baselines.near_field_mmpsr import run_near_field_mmpsr_baseline
    from ..baselines.nf_ris_groupomp_localgrid_wls import run_nf_ris_groupomp_localgrid_wls_baseline
    from ..baselines.ris_momp import run_ris_momp_baseline
    from ..config import default_config
    from ..main_single_proposed import _make_data, run_single_proposed_diagnostic
    from ..metrics import relative_nmse, position_rmse
    from .run_paper_ablation_figures import _peb_from_efim, set_number_of_ris_paths
    from .resource_control import (
        apply_thread_limits,
        assert_row_is_light,
        memory_snapshot_mb,
        resolve_hybrid_resources,
        thread_limit_context,
        trim_memory,
    )
    from .progress_logger import ProgressLogger
    from .cli_common import (
        add_io_args,
        add_mc_args,
        add_progress_args,
        add_resource_args,
        normalize_blas_threads,
    )


DEFAULT_SNR_GRID = "-30,-25,-20,-15,-10,-5,0,5,10"
DEFAULT_BASELINES = "als_cpd,ff_omp,ris_momp,nf_mmpsr,proposed,peb"
TRIAL_CSV = "benchmark_trials.csv"
SUMMARY_CSV = "benchmark_summary.csv"
RMSE_PDF = "fig7_benchmark_rmse_vs_snr.pdf"
NMSE_PDF = "fig7_benchmark_nmse_vs_snr.pdf"
FIELDNAMES = [
    "baseline",
    "trial_id",
    "seed",
    "snr_db",
    "K",
    "data_hash",
    "y_noisy_hash",
    "failed",
    "error",
    "runtime_s",
    "position_rmse_m",
    "y_nmse",
    "range_rmse_m",
    "tau_rmse_s",
    "raw_objective_final",
    "support_size",
    "grid_size",
    "dictionary_mode",
    "group_omp",
    "offgrid_refinement",
    "refinement_objective",
    "model_variant",
    "selected_support",
    "peb_position_m",
    "peb_is_data_only",
    "peb_uses_regularization",
    "nuisance_model",
    "clock_eliminated",
    "efim_condition_number",
    "efim_parameter_order",
    "peb_reference_type",
    "peb_reference_data_hash",
    "batch_size",
    "max_batch_memory_mb",
    "num_batches",
    "baseline_backend",
    "gpu_used",
    "gpu_device",
    "gpu_num_batches",
    "gpu_batch_size",
    "cache_enabled",
    "cache_hits",
    "cache_misses",
    "cache_estimated_bytes",
    "scoring_time_s",
    "backend_warning",
    "selected_grid_index",
    "momp_group_omp_enabled",
    "momp_score_mode",
    "momp_group_size",
    "momp_max_groups",
    "momp_selected_groups",
    "momp_local_refinement_used",
    "momp_refinement_levels",
    "momp_refinement_num_evals",
    "nf_mmpsr_cc_metric",
    "nf_mmpsr_top_candidates",
    "nf_mmpsr_local_refinement_used",
    "nf_mmpsr_refinement_levels",
    "nf_mmpsr_refinement_num_evals",
    "nf_mmpsr_coarse_best_score",
    "nf_mmpsr_refined_best_score",
    "reference_algorithm",
    "cpd_omp_adapted_used",
    "near_field_l1_refinement_used",
    "sage_enabled",
    "sage_iterations",
    "sage_num_evals",
    "local_grid_enabled",
    "local_grid_iterations",
    "local_grid_num_evals",
    "wls_enabled",
    "wls_final_cost",
    "subris_mode",
    "subris_shape",
    "subris_fallback_used",
    "adaptation_note",
    "rss_mb_before",
    "rss_mb_after",
    "warning",
]
BASELINE_LABELS = {
    "als_cpd": "ALS-CPD",
    "ff_omp": "FF-OMP",
    "ris_momp": "RIS-MOMP",
    "nf_mmpsr": "NF-MMPSR",
    "nf_ris_groupomp_localgrid_wls": "NF-RIS-GroupOMP-LocalGrid-WLS",
    "proposed": "Proposed",
    "peb": "PEB",
}


_WORKER_BLAS_THREADS = 1
_WORKER_RESPECT_EXISTING_BLAS_ENV = False
_WORKER_TRIM_MEMORY = True
_WORKER_PROFILE_MEMORY = False


def _init_worker(
    blas_threads: int,
    respect_existing_blas_env: bool = False,
    trim_memory_enabled: bool = True,
    profile_memory: bool = False,
) -> None:
    global _WORKER_BLAS_THREADS, _WORKER_RESPECT_EXISTING_BLAS_ENV
    global _WORKER_TRIM_MEMORY, _WORKER_PROFILE_MEMORY
    _WORKER_BLAS_THREADS = int(blas_threads)
    _WORKER_RESPECT_EXISTING_BLAS_ENV = bool(respect_existing_blas_env)
    _WORKER_TRIM_MEMORY = bool(trim_memory_enabled)
    _WORKER_PROFILE_MEMORY = bool(profile_memory)
    apply_thread_limits(
        _WORKER_BLAS_THREADS,
        respect_existing=_WORKER_RESPECT_EXISTING_BLAS_ENV,
    )


def parse_snr_grid(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def parse_baselines(value: str) -> list[str]:
    baselines = [item.strip() for item in value.split(",") if item.strip()]
    allowed = set(BASELINE_LABELS)
    unknown = [item for item in baselines if item not in allowed]
    if unknown:
        raise ValueError(f"unknown baselines: {unknown}")
    return baselines


def _trial_seed(seed: int, trial_id: int) -> int:
    sequence = np.random.SeedSequence(int(seed))
    return int(sequence.spawn(int(trial_id) + 1)[int(trial_id)].generate_state(1, dtype=np.uint32)[0])


def _proposed_policy_log_fragment(config: dict) -> str:
    return (
        f"proposed_stage2_policy={config.get('proposed_stage2_policy', '')} "
        f"ngc_lambda_ris={config.get('ngc_lambda_ris', 1.0)} "
        f"ngc_clock_green_quantile={config.get('ngc_clock_green_quantile', 0.99)} "
        f"ngc_clock_red_quantile={config.get('ngc_clock_red_quantile', 0.999)} "
        "rescue_accept_min_rel_improvement="
        f"{config.get('rescue_accept_min_rel_improvement', '')} "
        "rescue_accept_min_abs_improvement="
        f"{config.get('rescue_accept_min_abs_improvement', '')}"
    )


def _apply_grid_profile(config: dict, profile: str) -> dict:
    baselines = dict(config.get("baselines", {}))
    if profile == "coarse":
        baselines.update(
            {
                "als_cpd": {"position_grid_shape": (3, 3, 2)},
                "ff_omp": {"angle_grid_size": 5, "delay_grid_size": 5, "max_groups": config["K"], "offgrid_refinement": True, "batch_size": 64, "max_batch_memory_mb": 64.0},
                "ris_momp": {"direction_grid_size": 5, "delay_grid_size": 5, "max_groups": config["K"], "local_refinement": False, "refinement_levels": 0, "refinement_shrink": 0.5, "offgrid_refinement": False, "batch_size": 64, "max_batch_memory_mb": 64.0},
                "nf_mmpsr": {"grid_shape": (3, 3, 2), "clock_grid_size": 3, "top_candidates": 2, "local_refinement": True, "refinement_levels": 1, "local_position_grid_shape": (3, 3, 3), "local_clock_grid_size": 3, "offgrid_refinement": True, "batch_size": 16, "max_batch_memory_mb": 64.0},
                "nf_ris_groupomp_localgrid_wls": {"direction_grid_size": 5, "delay_grid_size": 5, "max_groups": config["K"], "coarse_to_nf_refinement_levels": 1, "local_grid_iterations": 1, "local_grid_refinement_levels": 1, "batch_size": 32, "max_batch_memory_mb": 64.0, "wls_enabled": True},
            }
        )
    elif profile == "medium":
        baselines.update(
            {
                "als_cpd": {"position_grid_shape": (5, 5, 3)},
                "ff_omp": {"angle_grid_size": 31, "delay_grid_size": 41, "max_groups": config["K"], "offgrid_refinement": True, "batch_size": 256, "max_batch_memory_mb": 256.0},
                "ris_momp": {"direction_grid_size": 31, "delay_grid_size": 41, "max_groups": config["K"], "local_refinement": True, "refinement_levels": 2, "refinement_shrink": 0.5, "offgrid_refinement": True, "batch_size": 256, "max_batch_memory_mb": 256.0},
                "nf_mmpsr": {"grid_shape": (11, 11, 5), "clock_grid_size": 11, "top_candidates": 8, "local_refinement": True, "refinement_levels": 3, "local_position_grid_shape": (3, 3, 3), "local_clock_grid_size": 5, "offgrid_refinement": True, "batch_size": 64, "max_batch_memory_mb": 256.0},
                "nf_ris_groupomp_localgrid_wls": {"direction_grid_size": 31, "delay_grid_size": 41, "max_groups": config["K"], "coarse_to_nf_refinement_levels": 2, "local_grid_iterations": 5, "local_grid_refinement_levels": 2, "batch_size": 128, "max_batch_memory_mb": 256.0, "wls_enabled": True},
            }
        )
    elif profile == "fine":
        baselines.update(
            {
                "als_cpd": {"position_grid_shape": (7, 7, 5)},
                "ff_omp": {"angle_grid_size": 45, "delay_grid_size": 61, "max_groups": config["K"], "offgrid_refinement": True, "batch_size": 256, "max_batch_memory_mb": 256.0},
                "ris_momp": {"direction_grid_size": 45, "delay_grid_size": 61, "max_groups": config["K"], "local_refinement": True, "refinement_levels": 3, "refinement_shrink": 0.5, "offgrid_refinement": True, "batch_size": 256, "max_batch_memory_mb": 256.0},
                "nf_mmpsr": {"grid_shape": (15, 15, 7), "clock_grid_size": 15, "top_candidates": 16, "local_refinement": True, "refinement_levels": 4, "local_position_grid_shape": (3, 3, 3), "local_clock_grid_size": 5, "offgrid_refinement": True, "batch_size": 64, "max_batch_memory_mb": 256.0},
                "nf_ris_groupomp_localgrid_wls": {"direction_grid_size": 45, "delay_grid_size": 61, "max_groups": config["K"], "coarse_to_nf_refinement_levels": 3, "local_grid_iterations": 7, "local_grid_refinement_levels": 3, "batch_size": 128, "max_batch_memory_mb": 256.0, "wls_enabled": True},
            }
        )
    else:
        raise ValueError(f"unknown grid profile {profile!r}")
    config["baselines"] = baselines
    return config


def make_config(
    seed: int,
    snr_db: float,
    paper_k: int,
    grid_profile: str,
    *,
    strict_ris_geometry: bool = False,
) -> dict:
    config = default_config()
    config["seed"] = int(seed)
    config["SNR_dB"] = float(snr_db)
    config["receiver_mode"] = "full_6d"
    config["print_progress"] = False
    config["verbose_stage2"] = False
    config["run_full_legacy_comparison"] = False
    config["stage2_adaptive"] = True
    config["stage2_rescue_type"] = "ris_only"
    config["proposed_stage2_policy"] = "ngc_certified_ris_only"
    config["rescue_accept_min_rel_improvement"] = 0.0
    config["rescue_accept_min_abs_improvement"] = 1.0e-8
    if strict_ris_geometry and np.asarray(config.get("ris_centers", [])).shape[0] < int(paper_k):
        raise ValueError(
            f"--strict-ris-geometry requested K={paper_k}, but config has only "
            f"{np.asarray(config.get('ris_centers', [])).shape[0]} RIS centers"
        )
    set_number_of_ris_paths(config, int(paper_k))
    _apply_grid_profile(config, grid_profile)
    return config


def _proposed_row(data: dict, config: dict, trial_id: int, baseline: str) -> dict[str, Any]:
    start = time.perf_counter()
    result = run_single_proposed_diagnostic(config, data_override=data)
    runtime_s = time.perf_counter() - start
    final = result.get("final", {})
    y_hat = final.get("Y_hat")
    p_hat = final.get("p_u")
    raw_obj = final.get("raw_objective_final", final.get("raw_objective", float("nan")))
    baseline_result = BaselineResult(
        name=baseline,
        p_u=None if p_hat is None else np.asarray(p_hat, dtype=float),
        delta_t=final.get("delta_t"),
        Y_hat=y_hat,
        raw_objective_final=float(raw_obj) if raw_obj is not None else float("nan"),
        components=final.get("components", {}),
        selected_support=[],
        runtime_s=runtime_s,
        diagnostics={"dictionary_mode": "proposed_ngc_adaptive_jones_vp"},
    )
    return make_baseline_row(
        baseline_result,
        data,
        config,
        baseline=baseline,
        trial_id=trial_id,
        seed=int(config["seed"]),
        snr_db=float(config["SNR_dB"]),
    )


def _peb_row(data: dict, config: dict, trial_id: int) -> dict[str, Any]:
    start = time.perf_counter()
    metrics = _peb_from_efim(data, config)
    runtime_s = time.perf_counter() - start
    return {
        "baseline": "peb",
        "trial_id": int(trial_id),
        "seed": int(config["seed"]),
        "snr_db": float(config["SNR_dB"]),
        "K": int(config["K"]),
        "data_hash": data_hash(data),
        "y_noisy_hash": y_noisy_hash(data),
        "failed": False,
        "error": "",
        "runtime_s": runtime_s,
        "position_rmse_m": float("nan"),
        "y_nmse": float("nan"),
        "range_rmse_m": float("nan"),
        "tau_rmse_s": float("nan"),
        "raw_objective_final": float("nan"),
        "support_size": 0,
        "grid_size": "",
        "dictionary_mode": "data_only_efim_peb",
        "selected_support": "",
        "peb_position_m": metrics.get("peb_position_m", float("nan")),
        "peb_is_data_only": bool(metrics.get("peb_is_data_only", True)),
        "peb_uses_regularization": bool(metrics.get("peb_uses_regularization", False)),
        "nuisance_model": str(metrics.get("nuisance_model", "jones_linear")),
        "clock_eliminated": bool(metrics.get("clock_eliminated", True)),
        "efim_condition_number": metrics.get("efim_condition_number", float("inf")),
        "efim_parameter_order": metrics.get("efim_parameter_order", []),
        "peb_reference_type": metrics.get(
            "peb_reference_type", "matched_model"
        ),
        "peb_reference_data_hash": data_hash(data),
        "warning": metrics.get("warning", ""),
    }


BASELINE_RUNNERS = {
    "als_cpd": run_als_cpd_baseline,
    "ff_omp": run_far_field_omp_baseline,
    "ris_momp": run_ris_momp_baseline,
    "nf_mmpsr": run_near_field_mmpsr_baseline,
    "nf_ris_groupomp_localgrid_wls": run_nf_ris_groupomp_localgrid_wls_baseline,
}


def _failure_row(
    baseline: str,
    trial_id: int,
    config: dict,
    exc: BaseException,
    data: dict | None = None,
) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "trial_id": int(trial_id),
        "seed": int(config.get("seed", 0)),
        "snr_db": float(config.get("SNR_dB", float("nan"))),
        "K": int(config.get("K", 0)),
        "data_hash": data_hash(data) if data is not None else "",
        "y_noisy_hash": y_noisy_hash(data) if data is not None else "",
        "failed": True,
        "error": f"{type(exc).__name__}: {exc}",
        "runtime_s": float("nan"),
        "position_rmse_m": float("nan"),
        "y_nmse": float("nan"),
        "range_rmse_m": float("nan"),
        "tau_rmse_s": float("nan"),
        "raw_objective_final": float("nan"),
        "support_size": 0,
        "grid_size": "",
        "dictionary_mode": "",
        "selected_support": "",
        "peb_position_m": float("nan"),
        "peb_is_data_only": "",
        "peb_uses_regularization": "",
        "nuisance_model": "",
        "clock_eliminated": "",
        "efim_condition_number": float("nan"),
        "efim_parameter_order": "",
        "peb_reference_type": "",
        "peb_reference_data_hash": "",
        "warning": "",
    }


def _run_trial_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    apply_thread_limits(
        int(task.get("blas_threads", _WORKER_BLAS_THREADS)),
        respect_existing=bool(
            task.get(
                "respect_existing_blas_env",
                _WORKER_RESPECT_EXISTING_BLAS_ENV,
            )
        ),
    )
    config = make_config(
        seed=int(task["seed"]),
        snr_db=float(task["snr_db"]),
        paper_k=int(task["paper_k"]),
        grid_profile=str(task["grid_profile"]),
        strict_ris_geometry=bool(task.get("strict_ris_geometry", False)),
    )
    config["baselines"]["backend_config"] = dict(task.get("backend_config", {}))
    config["baselines"]["trim_memory"] = bool(
        task.get("trim_memory", _WORKER_TRIM_MEMORY)
    )
    data = _make_data(config)
    rows: list[dict[str, Any]] = []
    for baseline in task["baselines"]:
        rss_before = (
            memory_snapshot_mb()
            if bool(task.get("profile_memory", _WORKER_PROFILE_MEMORY))
            else float("nan")
        )
        try:
            with thread_limit_context(
                int(task.get("blas_threads", _WORKER_BLAS_THREADS))
            ):
                if baseline in BASELINE_RUNNERS:
                    result = BASELINE_RUNNERS[baseline](data, config)
                    row = make_baseline_row(
                        result,
                        data,
                        config,
                        baseline=baseline,
                        trial_id=int(task["trial_id"]),
                        seed=int(config["seed"]),
                        snr_db=float(config["SNR_dB"]),
                    )
                    del result
                elif baseline == "proposed":
                    row = _proposed_row(data, config, int(task["trial_id"]), baseline)
                elif baseline == "peb":
                    row = _peb_row(data, config, int(task["trial_id"]))
                else:
                    raise ValueError(f"unknown baseline {baseline!r}")
        except Exception as exc:  # noqa: BLE001 - failed baseline becomes CSV row.
            row = _failure_row(baseline, int(task["trial_id"]), config, exc, data)
        if bool(task.get("trim_memory", _WORKER_TRIM_MEMORY)):
            trim_memory()
        rss_after = (
            memory_snapshot_mb()
            if bool(task.get("profile_memory", _WORKER_PROFILE_MEMORY))
            else float("nan")
        )
        row["rss_mb_before"] = rss_before
        row["rss_mb_after"] = rss_after
        rows.append(assert_row_is_light(row))
    del data
    if bool(task.get("trim_memory", _WORKER_TRIM_MEMORY)):
        trim_memory()
    return rows


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class StreamingCsvWriter:
    """Write row batches to a temporary CSV and atomically publish it."""

    def __init__(self, final_path: pathlib.Path, fieldnames: list[str]):
        self.final_path = pathlib.Path(final_path)
        self.tmp_path = self.final_path.with_name(f"{self.final_path.name}.tmp")
        self.fieldnames = fieldnames
        self.handle: Any | None = None
        self.writer: csv.DictWriter | None = None

    def __enter__(self) -> "StreamingCsvWriter":
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.tmp_path.open("w", newline="")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=self.fieldnames,
            extrasaction="ignore",
        )
        self.writer.writeheader()
        self.handle.flush()
        return self

    def writerows(self, rows: Iterable[dict[str, Any]]) -> None:
        if self.writer is None or self.handle is None:
            raise RuntimeError("StreamingCsvWriter is not open")
        self.writer.writerows(rows)
        self.handle.flush()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
        if exc_type is None:
            os.replace(self.tmp_path, self.final_path)
        return False


def _read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_csv(
    path: pathlib.Path,
    outlier_threshold_m: float | None = None,
) -> list[dict[str, Any]]:
    """Summarize a trial CSV without retaining complete rows."""
    groups: dict[tuple[str, float], dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["baseline"]), _to_float(row["snr_db"]))
            group = groups.setdefault(
                key,
                {
                    "n": 0,
                    "success": 0,
                    "rmse": [],
                    "nmse": [],
                    "peb": [],
                    "runtime": [],
                },
            )
            group["n"] += 1
            if str(row.get("failed")).lower() == "true":
                continue
            group["success"] += 1
            group["rmse"].append(_to_float(row.get("position_rmse_m")))
            group["nmse"].append(_to_float(row.get("y_nmse")))
            group["peb"].append(_to_float(row.get("peb_position_m")))
            group["runtime"].append(_to_float(row.get("runtime_s")))

    summary: list[dict[str, Any]] = []
    for (baseline, snr_db), group in sorted(groups.items()):
        rmse_stats = _summary_stats(group["rmse"])
        nmse_stats = _summary_stats(group["nmse"], percentiles=False)
        peb_stats = _summary_stats(group["peb"])
        runtime_stats = _summary_stats(group["runtime"], percentiles=False)
        finite_rmse = np.asarray(group["rmse"], dtype=float)
        finite_rmse = finite_rmse[np.isfinite(finite_rmse)]
        outlier_rate = float("nan")
        if outlier_threshold_m is not None and finite_rmse.size:
            outlier_rate = float(
                np.mean(finite_rmse > float(outlier_threshold_m))
            )
        row_summary = {
            "baseline": baseline,
            "snr_db": snr_db,
            "n": int(group["n"]),
            "success_rate": float(group["success"] / max(group["n"], 1)),
            "outlier_rate": outlier_rate,
            "runtime_s_mean": runtime_stats["mean"],
        }
        for name, value in rmse_stats.items():
            row_summary[f"position_rmse_m_{name}"] = value
        for name, value in nmse_stats.items():
            row_summary[f"y_nmse_{name}"] = value
        for name, value in peb_stats.items():
            row_summary[f"peb_position_m_{name}"] = value
        summary.append(row_summary)
    return summary


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _summary_stats(values: list[float], percentiles: bool = True) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        stats = {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
        if percentiles:
            stats.update({"p10": float("nan"), "p90": float("nan")})
        return stats
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }
    if percentiles:
        stats.update({"p10": float(np.percentile(arr, 10.0)), "p90": float(np.percentile(arr, 90.0))})
    return stats


def get_plot_metric(baseline: str, plot_kind: str) -> str | None:
    if baseline == "peb":
        return "peb_position_m" if plot_kind == "rmse" else None
    if plot_kind == "rmse":
        return "position_rmse_m"
    if plot_kind == "nmse":
        return "y_nmse"
    raise ValueError(f"unknown plot kind {plot_kind!r}")


def summarize_rows(rows: list[dict[str, Any]], outlier_threshold_m: float | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["baseline"]), _to_float(row["snr_db"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (baseline, snr_db), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        success = [row for row in group if str(row.get("failed")).lower() != "true"]
        rmse_stats = _summary_stats([_to_float(row.get("position_rmse_m")) for row in success])
        nmse_stats = _summary_stats([_to_float(row.get("y_nmse")) for row in success], percentiles=False)
        peb_stats = _summary_stats([_to_float(row.get("peb_position_m")) for row in success])
        runtime_stats = _summary_stats([_to_float(row.get("runtime_s")) for row in success], percentiles=False)
        outlier_rate = float("nan")
        if outlier_threshold_m is not None:
            rmse = np.asarray([_to_float(row.get("position_rmse_m")) for row in success], dtype=float)
            finite = rmse[np.isfinite(rmse)]
            outlier_rate = float(np.mean(finite > float(outlier_threshold_m))) if finite.size else float("nan")
        row_summary = {
            "baseline": baseline,
            "snr_db": snr_db,
            "n": len(group),
            "success_rate": float(len(success) / max(len(group), 1)),
            "outlier_rate": outlier_rate,
            "runtime_s_mean": runtime_stats["mean"],
        }
        for name, value in rmse_stats.items():
            row_summary[f"position_rmse_m_{name}"] = value
        for name, value in nmse_stats.items():
            row_summary[f"y_nmse_{name}"] = value
        for name, value in peb_stats.items():
            row_summary[f"peb_position_m_{name}"] = value
        summary.append(row_summary)
    return summary


def validate_same_data_hashes(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[int, float, int], set[str]] = {}
    for row in rows:
        if str(row.get("failed")).lower() == "true":
            continue
        key = (int(row["trial_id"]), _to_float(row["snr_db"]), int(row["K"]))
        groups.setdefault(key, set()).add(str(row.get("y_noisy_hash", "")))
    mismatches = {key: values for key, values in groups.items() if len(values) > 1}
    if mismatches:
        raise RuntimeError(f"benchmark same-data hash mismatch: {mismatches}")


def _plot(summary_rows: list[dict[str, Any]], out_dir: pathlib.Path, plot_kind: str) -> None:
    mpl_config = out_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    baselines = [name for name in BASELINE_LABELS if any(row["baseline"] == name for row in summary_rows)]
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, baseline in enumerate(baselines):
        metric = get_plot_metric(baseline, plot_kind)
        if metric is None:
            continue
        rows = [row for row in summary_rows if row["baseline"] == baseline]
        xs = np.asarray([_to_float(row["snr_db"]) for row in rows], dtype=float)
        ys = np.asarray([_to_float(row[f"{metric}_mean"]) for row in rows], dtype=float)
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker=markers[idx % len(markers)], linewidth=1.5, label=BASELINE_LABELS[baseline])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Position RMSE / PEB (m)" if plot_kind == "rmse" else "Channel NMSE")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / (RMSE_PDF if plot_kind == "rmse" else NMSE_PDF))
    plt.close(fig)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _cache_signature(args: argparse.Namespace, snr_grid: list[float], baselines: list[str]) -> dict[str, Any]:
    return {
        "n_trials": int(args.n_trials),
        "snr_grid": [float(value) for value in snr_grid],
        "paper_k": int(args.paper_k),
        "baselines": list(baselines),
        "grid_profile": str(args.grid_profile),
        "seed": int(args.seed),
        "git_commit": _git_commit(),
        "strict_ris_geometry": bool(args.strict_ris_geometry),
        "baseline_backend": str(args.baseline_backend),
        "gpu_device": args.gpu_device,
        "gpu_batch_size": args.gpu_batch_size,
        "cpu_batch_size": args.cpu_batch_size,
        "cache_baseline_grids": bool(args.cache_baseline_grids),
        "cache_memory_budget_gb": args.cache_memory_budget_gb,
        "gpu_memory_fraction": args.gpu_memory_fraction,
    }


def _metadata(args: argparse.Namespace, snr_grid: list[float], baselines: list[str]) -> dict[str, Any]:
    signature = _cache_signature(args, snr_grid, baselines)
    return {
        **signature,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(sys.argv),
        "jobs": int(args.jobs),
        "process_workers": int(args.process_workers),
        "blas_threads": int(args.blas_threads),
        "estimated_cpu_slots": int(args.resource_plan["estimated_cpu_slots"]),
        "memory_budget_gb": args.memory_budget_gb,
        "memory_per_worker_gb": args.memory_per_worker_gb,
        "trim_memory": bool(args.trim_memory),
        "profile_memory": bool(args.profile_memory),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "baseline_backend": str(args.baseline_backend),
        "gpu_device": args.gpu_device,
        "gpu_batch_size": args.gpu_batch_size,
        "cpu_batch_size": args.cpu_batch_size,
        "cache_baseline_grids": bool(args.cache_baseline_grids),
        "cache_memory_budget_gb": args.cache_memory_budget_gb,
        "gpu_memory_fraction": args.gpu_memory_fraction,
    }


def _read_metadata(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _metadata_matches(metadata: dict[str, Any] | None, args: argparse.Namespace, snr_grid: list[float], baselines: list[str]) -> bool:
    if not metadata:
        return False
    expected = _cache_signature(args, snr_grid, baselines)
    return all(metadata.get(key) == value for key, value in expected.items())


def _tasks(args: argparse.Namespace, snr_grid: list[float], baselines: list[str]) -> list[dict[str, Any]]:
    tasks = []
    for snr_db in snr_grid:
        for trial_id in range(int(args.n_trials)):
            tasks.append(
                {
                    "trial_id": int(trial_id),
                    "seed": _trial_seed(int(args.seed), int(trial_id)),
                    "snr_db": float(snr_db),
                    "paper_k": int(args.paper_k),
                    "baselines": list(baselines),
                    "grid_profile": str(args.grid_profile),
                    "blas_threads": (
                        1
                        if str(args.blas_threads).lower() == "auto"
                        else int(args.blas_threads)
                    ),
                    "respect_existing_blas_env": bool(args.respect_existing_blas_env),
                    "trim_memory": bool(args.trim_memory),
                    "profile_memory": bool(args.profile_memory),
                    "strict_ris_geometry": bool(args.strict_ris_geometry),
                    "backend_config": {
                        "backend": str(args.baseline_backend),
                        "gpu_device": args.gpu_device,
                        "gpu_batch_size": args.gpu_batch_size,
                        "cpu_batch_size": args.cpu_batch_size,
                        "cache_enabled": bool(args.cache_baseline_grids),
                        "cache_memory_budget_gb": args.cache_memory_budget_gb,
                        "gpu_memory_fraction": args.gpu_memory_fraction,
                        "dtype": "complex128",
                    },
                }
            )
    return tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EVS-MIMO-RIS benchmark comparisons.")
    add_mc_args(parser, n_trials_default=50, paper_k_default=3, outlier_threshold_default=0.1)
    parser.add_argument("--snr-grid", default=DEFAULT_SNR_GRID)
    add_io_args(parser, default_out_dir="results/benchmark_comparison")
    add_resource_args(parser, jobs_default=10, blas_threads_default="auto")
    add_progress_args(parser)
    parser.add_argument(
        "--baseline-backend",
        choices=("cpu", "cupy", "auto"),
        default="cpu",
    )
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--gpu-batch-size", type=int, default=None)
    parser.add_argument("--cpu-batch-size", type=int, default=None)
    parser.add_argument(
        "--cache-baseline-grids",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cache-memory-budget-gb", type=float, default=None)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--reuse-incompatible-cache", action="store_true")
    parser.add_argument("--baselines", default=DEFAULT_BASELINES)
    parser.add_argument("--strict-ris-geometry", action="store_true")
    parser.add_argument("--grid-profile", choices=("coarse", "medium", "fine"), default="medium")
    args = parser.parse_args(argv)
    normalize_blas_threads(args)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.process_workers is not None and args.process_workers <= 0:
        raise ValueError("--process-workers must be positive")
    if args.paper_k <= 0:
        raise ValueError("--paper-k must be positive")
    if args.blas_threads != "auto" and int(args.blas_threads) <= 0:
        raise ValueError("--blas-threads must be positive or 'auto'")
    if args.gpu_batch_size is not None and args.gpu_batch_size <= 0:
        raise ValueError("--gpu-batch-size must be positive")
    if args.cpu_batch_size is not None and args.cpu_batch_size <= 0:
        raise ValueError("--cpu-batch-size must be positive")
    if args.cache_memory_budget_gb is not None and args.cache_memory_budget_gb <= 0:
        raise ValueError("--cache-memory-budget-gb must be positive")
    if args.gpu_memory_fraction is not None and not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be in (0, 1]")
    if args.baseline_backend in {"cupy", "auto"} and args.process_workers is None:
        args.process_workers = 1
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = (
        pathlib.Path(args.progress_log)
        if args.progress_log is not None
        else out_dir / "progress.jsonl"
    )
    trial_csv = out_dir / TRIAL_CSV
    summary_csv = out_dir / SUMMARY_CSV
    metadata_path = out_dir / "benchmark_metadata.json"
    baselines = parse_baselines(args.baselines)
    snr_grid = parse_snr_grid(args.snr_grid)
    if "proposed" in baselines:
        preview_config = make_config(
            seed=int(args.seed),
            snr_db=float(snr_grid[0]),
            paper_k=int(args.paper_k),
            grid_profile=str(args.grid_profile),
            strict_ris_geometry=bool(args.strict_ris_geometry),
        )
        print(f"baseline=proposed {_proposed_policy_log_fragment(preview_config)}")
    tasks = _tasks(args, snr_grid, baselines)
    progress = ProgressLogger(
        progress_path, len(tasks), "run_benchmark_comparison"
    )
    if not args.quiet_progress:
        print(f"Progress log: {progress_path}")
        print("Monitor with:")
        print(f"    tail -f {progress_path}")
        print(
            "    python -m src.experiments.monitor_progress "
            f"--progress-log {progress_path}"
        )
    progress.log("start", "running", message="benchmark experiment started")
    args.resource_plan = resolve_hybrid_resources(
        jobs=args.jobs,
        process_workers=args.process_workers,
        blas_threads=args.blas_threads,
        n_tasks=max(len(tasks), 1),
        memory_budget_gb=args.memory_budget_gb,
        memory_per_worker_gb=args.memory_per_worker_gb,
    )
    args.process_workers = int(args.resource_plan["process_workers"])
    args.blas_threads = int(args.resource_plan["blas_threads"])
    if args.baseline_backend in {"cupy", "auto"} and args.process_workers > 2:
        print(
            "Multiple worker processes may contend for one GPU; prefer "
            "--process-workers 1 or 2."
        )
    for task in tasks:
        task["blas_threads"] = args.blas_threads
    apply_thread_limits(
        args.blas_threads,
        respect_existing=bool(args.respect_existing_blas_env),
    )
    print(
        "Resource plan: "
        f"jobs={args.jobs} "
        f"process_workers={args.process_workers} "
        f"blas_threads={args.blas_threads} "
        f"estimated_cpu_slots={args.resource_plan['estimated_cpu_slots']} "
        f"memory_budget_gb={args.memory_budget_gb} "
        f"memory_per_worker_gb={args.memory_per_worker_gb}"
    )
    metadata = _read_metadata(metadata_path)
    cache_compatible = _metadata_matches(metadata, args, snr_grid, baselines)
    can_reuse = (
        trial_csv.exists()
        and summary_csv.exists()
        and not args.force_rerun
        and (cache_compatible or bool(args.reuse_incompatible_cache))
    )
    if can_reuse:
        summary = _read_csv(summary_csv)
    else:
        hash_groups: dict[tuple[int, float, int], set[str]] = {}
        initargs = (
            args.blas_threads,
            bool(args.respect_existing_blas_env),
            bool(args.trim_memory),
            bool(args.profile_memory),
        )
        with StreamingCsvWriter(trial_csv, FIELDNAMES) as writer:
            if args.process_workers == 1:
                _init_worker(*initargs)
                row_batches = map(_run_trial_task, tasks)
                pool_context = contextlib.nullcontext()
            else:
                pool_context = mp.Pool(
                    processes=args.process_workers,
                    initializer=_init_worker,
                    initargs=initargs,
                )
                pool = pool_context.__enter__()
                row_batches = pool.imap_unordered(
                    _run_trial_task,
                    tasks,
                    chunksize=1,
                )
            try:
                for row_batch in row_batches:
                    writer.writerows(row_batch)
                    representative = row_batch[0] if row_batch else {}
                    failed_rows = [
                        row
                        for row in row_batch
                        if str(row.get("failed")).lower() == "true"
                    ]
                    progress.log(
                        "task_failed" if failed_rows else "task_done",
                        "failed" if failed_rows else "completed",
                        figure="fig7",
                        baseline_or_variant=",".join(
                            str(row.get("baseline", "")) for row in row_batch
                        ),
                        snr_db=representative.get("snr_db", ""),
                        trial_id=representative.get("trial_id", ""),
                        seed=representative.get("seed", ""),
                        K=representative.get("K", ""),
                        message="benchmark trial batch completed",
                        error="; ".join(
                            str(row.get("error", "")) for row in failed_rows
                        ),
                    )
                    for row in row_batch:
                        if str(row.get("failed")).lower() == "true":
                            continue
                        key = (
                            int(row["trial_id"]),
                            _to_float(row["snr_db"]),
                            int(row["K"]),
                        )
                        hash_groups.setdefault(key, set()).add(
                            str(row.get("y_noisy_hash", ""))
                        )
            finally:
                if args.process_workers != 1:
                    pool_context.__exit__(None, None, None)
        mismatches = {
            key: values for key, values in hash_groups.items() if len(values) > 1
        }
        if mismatches:
            raise RuntimeError(f"benchmark same-data hash mismatch: {mismatches}")
        summary = summarize_csv(
            trial_csv,
            outlier_threshold_m=float(args.outlier_threshold_m),
        )
        _write_csv(
            summary_csv,
            summary,
            list(summary[0].keys()) if summary else [],
        )
    with metadata_path.open("w") as handle:
        json.dump(_metadata(args, snr_grid, baselines), handle, indent=2)
    if not args.no_plots:
        _plot(summary, out_dir, "rmse")
        _plot(summary, out_dir, "nmse")
    progress.log("finished", "completed", message="benchmark experiment finished")
    progress.close()
    print(f"Wrote benchmark comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
