"""Generate robustness and system-scaling figures for the EVS-MIMO-RIS paper."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
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
    from src.baselines.common import BaselineResult, data_hash, make_baseline_row, y_noisy_hash
    from src.baselines.far_field_omp import run_far_field_omp_baseline
    from src.baselines.near_field_mmpsr import run_near_field_mmpsr_baseline
    from src.baselines.ris_momp import run_ris_momp_baseline
    from src.channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from src.config import default_config
    from src.experiments.resource_control import (
        apply_thread_limits,
        assert_row_is_light,
        memory_snapshot_mb,
        resolve_hybrid_resources,
        thread_limit_context,
        trim_memory,
    )
    from src.experiments.progress_logger import ProgressLogger
    from src.experiments.run_benchmark_comparison import _apply_grid_profile
    from src.experiments.run_paper_ablation_figures import _peb_from_efim, set_number_of_ris_paths
    from src.main_single_proposed import _make_data, run_single_proposed_diagnostic
    from src.tensor_utils import hankelize_frequency
    from src.utils import scipy_is_available
else:
    from ..baselines.common import BaselineResult, data_hash, make_baseline_row, y_noisy_hash
    from ..baselines.far_field_omp import run_far_field_omp_baseline
    from ..baselines.near_field_mmpsr import run_near_field_mmpsr_baseline
    from ..baselines.ris_momp import run_ris_momp_baseline
    from ..channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from ..config import default_config
    from .resource_control import (
        apply_thread_limits,
        assert_row_is_light,
        memory_snapshot_mb,
        resolve_hybrid_resources,
        thread_limit_context,
        trim_memory,
    )
    from .progress_logger import ProgressLogger
    from .run_benchmark_comparison import _apply_grid_profile
    from .run_paper_ablation_figures import _peb_from_efim, set_number_of_ris_paths
    from ..main_single_proposed import _make_data, run_single_proposed_diagnostic
    from ..tensor_utils import hankelize_frequency
    from ..utils import scipy_is_available


FIGURE_ORDER = ["fig8", "fig9", "fig10a", "fig10b", "fig10c"]
DEFAULT_FIGURES = ",".join(FIGURE_ORDER)
DEFAULT_BASELINES = "proposed,ff_omp,ris_momp,nf_mmpsr,peb"
ESTIMATOR_BASELINES = ("proposed", "ff_omp", "ris_momp", "nf_mmpsr")
REFERENCE_BASELINES = ("peb", "oracle_calibrated_peb", "trueK_peb_reference")
BASELINE_LABELS = {
    "proposed": "Proposed",
    "ff_omp": "FF-OMP",
    "ris_momp": "RIS-MOMP",
    "nf_mmpsr": "NF-MMPSR",
    "peb": "PEB",
    "oracle_calibrated_peb": "Oracle-calibrated PEB (reference)",
    "trueK_peb_reference": "True-K PEB (reference only)",
}
FIGURE_FILES = {
    "fig8": (
        "fig8_calibration_trials.csv",
        "fig8_calibration_summary.csv",
        "fig8_calibration_rmse_vs_phase_error",
    ),
    "fig9": (
        "fig9_k_mismatch_trials.csv",
        "fig9_k_mismatch_summary.csv",
        "fig9_k_mismatch_rmse_vs_Khat",
    ),
    "fig10a": ("fig10a_T_trials.csv", "fig10a_T_summary.csv", "fig10a_rmse_vs_T"),
    "fig10b": ("fig10b_MR_trials.csv", "fig10b_MR_summary.csv", "fig10b_rmse_vs_MR"),
    "fig10c": ("fig10c_rUE_trials.csv", "fig10c_rUE_summary.csv", "fig10c_rmse_vs_rUE"),
}
X_FIELDS = {
    "fig8": ("calibration_std_deg", "Calibration phase error std. (deg)"),
    "fig9": ("assumed_K", r"Assumed path number $\hat{K}$"),
    "fig10a": ("T", "Training length T"),
    "fig10b": ("M_R", r"RIS element number $M_R$"),
    "fig10c": ("r_ue_m", r"UE distance from RIS centroid $r_{\rm UE}$ (m)"),
}
FIELDNAMES = [
    "figure",
    "trial_id",
    "seed",
    "snr_db",
    "true_K",
    "assumed_K",
    "calibration_std_deg",
    "T",
    "ris_side",
    "M_R",
    "r_ue_m",
    "baseline",
    "position_rmse_m",
    "y_nmse",
    "raw_objective_final",
    "runtime_s",
    "failed",
    "error",
    "y_noisy_hash",
    "data_hash",
    "warning",
    "peb_position_m",
    "peb_is_data_only",
    "peb_uses_regularization",
    "nuisance_model",
    "clock_eliminated",
    "efim_condition_number",
    "efim_parameter_order",
    "peb_reference_type",
    "peb_reference_data_hash",
    "k_mismatch_scene_mode",
    "true_scene_hash",
    "estimator_scene_hash",
    "first_trueK_preserved",
    "rss_mb_before",
    "rss_mb_after",
]

_PATH_ARRAY_KEYS = (
    "ris_centers",
    "rotations",
    "Omega",
    "a_RB",
    "v_B",
    "Theta",
    "d_RB",
    "gamma_true",
    "eta_true",
    "beta_true",
    "gamma",
    "eta",
    "eta_pol",
    "beta",
    "gains",
    "path_gains",
    "ranges",
    "path_ranges",
    "taus",
    "path_delays",
    "poles",
    "path_poles",
    "elevations",
    "azimuths",
)
_WORKER_BLAS_THREADS = 1
_WORKER_PROFILE_MEMORY = False


def parse_float_grid(value: str | Iterable[float]) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()] if isinstance(value, str) else [float(item) for item in value]
    if not values:
        raise ValueError("grid must not be empty")
    return values


def parse_int_grid(value: str | Iterable[int]) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()] if isinstance(value, str) else [int(item) for item in value]
    if not values or any(item <= 0 for item in values):
        raise ValueError("integer grid must contain positive values")
    return values


def parse_figures(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(FIGURE_ORDER))
    if unknown:
        raise ValueError(f"unknown figures: {unknown}")
    return [figure for figure in FIGURE_ORDER if figure in requested]


def parse_baselines(value: str) -> list[str]:
    baselines = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(baselines) - set(ESTIMATOR_BASELINES) - {"peb"})
    if unknown:
        raise ValueError(f"unknown baselines: {unknown}")
    return list(dict.fromkeys(baselines))


def baselines_for_figure(
    figure: str,
    requested: Iterable[str],
    *,
    include_calibration_oracle_peb: bool = False,
    include_trueK_peb_reference: bool = False,
) -> list[str]:
    selected = [name for name in requested if name in ESTIMATOR_BASELINES]
    if figure == "fig8":
        if include_calibration_oracle_peb:
            selected.append("oracle_calibrated_peb")
    elif figure == "fig9":
        if include_trueK_peb_reference:
            selected.append("trueK_peb_reference")
    elif "peb" in requested:
        selected.append("peb")
    return selected


def _trial_seed(seed: int, figure: str, trial_id: int) -> int:
    figure_index = FIGURE_ORDER.index(figure)
    sequence = np.random.SeedSequence([int(seed), int(figure_index), int(trial_id)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _array_hash(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).view(np.uint8)).hexdigest()[:16]


def scene_hash(scene: dict) -> str:
    """Hash a complete scene, including complex calibration/model arrays."""
    digest = hashlib.sha256()
    for key in sorted(scene):
        value = scene[key]
        digest.update(str(key).encode())
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.view(np.uint8))
        elif isinstance(value, np.generic):
            digest.update(repr(value.item()).encode())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()[:16]


def reference_data_hash(data: dict) -> str:
    payload = f"{data_hash(data)}:{scene_hash(data['scene'])}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def inject_ris_bs_calibration_phase_error(
    data_or_scene: dict,
    std_deg: float,
    rng: np.random.Generator,
) -> dict:
    """Return a deep copy with phase error applied only to the generated a_RB."""
    perturbed = copy.deepcopy(data_or_scene)
    scene = perturbed["scene"] if "scene" in perturbed else perturbed
    nominal = np.asarray(scene["a_RB"], dtype=complex)
    sigma = np.deg2rad(float(std_deg))
    epsilon = rng.normal(0.0, sigma, size=nominal.shape)
    scene["a_RB"] = nominal * np.exp(1j * epsilon)
    return perturbed


def ue_position_on_centroid_ray(config: dict, r_ue_m: float) -> np.ndarray:
    centers = np.asarray(config["ris_centers"], dtype=float)[: int(config["K"])]
    centroid = np.mean(centers, axis=0)
    p0 = np.asarray(config["p_u_true"], dtype=float)
    direction = p0 - centroid
    direction /= np.linalg.norm(direction)
    return centroid + float(r_ue_m) * direction


def default_rue_grid(config: dict) -> list[float]:
    centers = np.asarray(config["ris_centers"], dtype=float)[: int(config["K"])]
    r0 = float(np.linalg.norm(np.asarray(config["p_u_true"], dtype=float) - np.mean(centers, axis=0)))
    return [factor * r0 for factor in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]


def make_config(
    seed: int,
    snr_db: float,
    k_paths: int,
    *,
    T: int | None = None,
    ris_side: int | None = None,
    r_ue_m: float | None = None,
    strict_ris_geometry: bool = False,
) -> dict:
    config = default_config()
    config.update(
        {
            "seed": int(seed),
            "SNR_dB": float(snr_db),
            "receiver_mode": "full_6d",
            "print_progress": False,
            "verbose_stage2": False,
            "run_full_legacy_comparison": False,
        }
    )
    if strict_ris_geometry and np.asarray(config["ris_centers"]).shape[0] < int(k_paths):
        raise ValueError(
            f"--strict-ris-geometry requested K={k_paths}, but only "
            f"{np.asarray(config['ris_centers']).shape[0]} RIS centers are configured"
        )
    set_number_of_ris_paths(config, int(k_paths))
    if T is not None:
        config["T"] = int(T)
    if ris_side is not None:
        config["ris_shape"] = (int(ris_side), int(ris_side))
    if r_ue_m is not None:
        config["p_u_true"] = ue_position_on_centroid_ray(config, float(r_ue_m))
    # This follows the benchmark runner's resource-safe profile. The baseline
    # mathematics is unchanged; only its documented dictionary resolution is set.
    _apply_grid_profile(config, "coarse")
    return config


def _data_from_scenes(
    nominal_scene: dict,
    generation_scene: dict,
    config: dict,
    noise_rng: np.random.Generator,
) -> dict:
    true_components = channel_components(
        generation_scene,
        generation_scene["p_u_true"],
        generation_scene["delta_t_true"],
        generation_scene["gamma_true"],
        generation_scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(true_components, generation_scene["beta_true"])
    y_noisy, noise_variance = add_awgn(
        y_true,
        config["SNR_dB"],
        noise_rng,
        active_mask=generation_scene.get("evs_observation_mask"),
    )
    return {
        "scene": nominal_scene,
        "true_components": true_components,
        "Y_true": y_true,
        "Y_noisy": y_noisy,
        "Z_true": hankelize_frequency(y_true, nominal_scene["P"]),
        "Z_noisy": hankelize_frequency(y_noisy, nominal_scene["P"]),
        "noise_variance": noise_variance,
        "timing": {},
    }


def make_calibration_mismatch_data(config: dict, std_deg: float) -> tuple[dict, dict]:
    sequence = np.random.SeedSequence(int(config["seed"]))
    scene_seed, mismatch_seed, noise_seed = sequence.spawn(3)
    nominal_scene = generate_scene(config, np.random.default_rng(scene_seed))
    generation_scene = inject_ris_bs_calibration_phase_error(
        nominal_scene, std_deg, np.random.default_rng(mismatch_seed)
    )
    data = _data_from_scenes(
        nominal_scene,
        generation_scene,
        config,
        np.random.default_rng(noise_seed),
    )
    oracle_data = copy.copy(data)
    oracle_data["scene"] = generation_scene
    return data, oracle_data


def _slice_scene(scene: dict, k_paths: int) -> dict:
    sliced = copy.deepcopy(scene)
    original_k = int(scene["K"])
    if not 0 < int(k_paths) <= original_k:
        raise ValueError(f"cannot slice scene K={original_k} to K={k_paths}")
    sliced["K"] = int(k_paths)
    for key in _scene_path_keys(scene):
        if key in sliced:
            sliced[key] = np.asarray(sliced[key])[: int(k_paths)].copy()
    return sliced


def _scene_path_keys(scene: dict) -> list[str]:
    """Return known and explicitly named per-path scene fields."""
    keys = {key for key in _PATH_ARRAY_KEYS if key in scene}
    for key, value in scene.items():
        lowered = str(key).lower()
        array = np.asarray(value)
        explicitly_per_path = (
            lowered.startswith("path_")
            or lowered.endswith("_per_path")
            or "jones" in lowered
            or "polarization" in lowered
        )
        if explicitly_per_path and array.ndim > 0:
            keys.add(str(key))
    return sorted(keys)


def _path_prefix_equal(
    scene_true: dict,
    scene_other: dict,
    true_k: int,
) -> bool:
    for key in _scene_path_keys(scene_true):
        if key not in scene_true:
            continue
        if key not in scene_other:
            return False
        true_value = np.asarray(scene_true[key])
        other_value = np.asarray(scene_other[key])
        if true_value.ndim == 0 or true_value.shape[0] < int(true_k):
            raise ValueError(
                f"scene path field {key!r} has fewer than {true_k} path entries"
            )
        if other_value.ndim == 0 or other_value.shape[0] < int(true_k):
            return False
        if not np.array_equal(
            true_value[: int(true_k)],
            other_value[: int(true_k)],
        ):
            return False
    return True


def extend_scene_preserving_true_paths(
    scene_true: dict,
    target_K: int,
    config: dict,
    seed: int,
) -> dict:
    """Append deterministic candidate paths without changing physical paths."""
    true_k = int(scene_true["K"])
    target_k = int(target_K)
    if target_k < true_k:
        raise ValueError("target_K must be at least the physical scene K")
    if target_k == true_k:
        return copy.deepcopy(scene_true)

    candidate_config = copy.deepcopy(config)
    set_number_of_ris_paths(candidate_config, target_k)
    candidate_scene = generate_scene(
        candidate_config,
        np.random.default_rng(int(seed)),
    )
    candidate_components: dict[str, Any] | None = None
    extended = copy.deepcopy(scene_true)
    extended["K"] = target_k
    component_aliases = {
        "ranges": "ranges",
        "path_ranges": "ranges",
        "taus": "taus",
        "path_delays": "taus",
        "poles": "poles",
        "path_poles": "poles",
        "elevations": "elevations",
        "azimuths": "azimuths",
    }
    for key in _scene_path_keys(scene_true):
        if key not in scene_true:
            continue
        if key in candidate_scene:
            candidate_field = candidate_scene[key]
        elif key in component_aliases:
            if candidate_components is None:
                candidate_components = channel_components(
                    candidate_scene,
                    candidate_scene["p_u_true"],
                    candidate_scene["delta_t_true"],
                    candidate_scene["gamma_true"],
                    candidate_scene["eta_true"],
                )
            candidate_field = candidate_components[component_aliases[key]]
        else:
            raise ValueError(
                f"cannot safely extend scene path field {key!r}: no deterministic "
                "candidate-field generator is defined"
            )
        true_value = np.asarray(scene_true[key])
        candidate_value = np.asarray(candidate_field)
        if true_value.ndim == 0 or true_value.shape[0] < true_k:
            raise ValueError(
                f"cannot safely extend {key!r}: expected at least {true_k} entries"
            )
        if (
            candidate_value.ndim != true_value.ndim
            or candidate_value.shape[0] < target_k
            or candidate_value.shape[1:] != true_value.shape[1:]
        ):
            raise ValueError(
                f"cannot safely extend {key!r}: incompatible generated shape "
                f"{candidate_value.shape}, physical shape {true_value.shape}"
            )
        extended[key] = np.concatenate(
            [
                true_value[:true_k].copy(),
                candidate_value[true_k:target_k].copy(),
            ],
            axis=0,
        )
    if not _path_prefix_equal(scene_true, extended, true_k):
        raise RuntimeError("extended estimator scene changed a physical true-K path")
    return extended


def make_k_mismatch_data(
    true_config: dict,
    assumed_k: int,
    max_assumed_k: int | None = None,
) -> tuple[dict, dict]:
    """Generate true-K data and return a K-hat estimator view of the same tensor."""
    del max_assumed_k  # Compatibility with older callers; no catalog is generated.
    true_config = copy.deepcopy(true_config)
    true_k = int(true_config["K"])
    assumed_k = int(assumed_k)
    physical_data = _make_data(true_config)
    physical_scene = physical_data["scene"]
    if assumed_k < true_k:
        estimator_scene = _slice_scene(physical_scene, assumed_k)
        scene_mode = "slice_physical_scene"
    elif assumed_k == true_k:
        estimator_scene = physical_scene
        scene_mode = "matched_physical_scene"
    else:
        estimator_scene = extend_scene_preserving_true_paths(
            physical_scene,
            assumed_k,
            true_config,
            int(true_config["seed"]),
        )
        scene_mode = "extended_physical_scene"
    estimator_data = dict(physical_data)
    estimator_data["scene"] = estimator_scene
    estimator_data["physical_ris_centroid"] = np.mean(
        np.asarray(physical_scene["ris_centers"], dtype=float)[:true_k],
        axis=0,
    )
    estimator_data["k_mismatch_scene_mode"] = scene_mode
    estimator_data["true_scene_hash"] = scene_hash(physical_scene)
    estimator_data["estimator_scene_hash"] = scene_hash(estimator_scene)
    estimator_data["first_trueK_preserved"] = _path_prefix_equal(
        physical_scene,
        estimator_scene,
        min(true_k, assumed_k),
    )
    return estimator_data, physical_data


def _configure_assumed_k(config: dict, assumed_k: int) -> dict:
    configured = copy.deepcopy(config)
    configured["K"] = int(assumed_k)
    baselines = copy.deepcopy(configured.get("baselines", {}))
    baselines.setdefault("ff_omp", {})["max_atoms"] = int(assumed_k)
    baselines.setdefault("ris_momp", {})["max_atoms"] = int(assumed_k)
    # NF-MMPSR creates two Jones nuisance columns per panel; its model
    # dimension and support therefore follow scene["K"] == assumed_K.
    baselines.setdefault("nf_mmpsr", {})["assumed_K"] = int(assumed_k)
    configured["baselines"] = baselines
    return configured


def _common_row_fields(task: dict[str, Any], data: dict, baseline: str) -> dict[str, Any]:
    scene = data["scene"]
    centers = np.asarray(scene["ris_centers"], dtype=float)[: int(scene["K"])]
    centroid = np.asarray(
        data.get("physical_ris_centroid", np.mean(centers, axis=0)),
        dtype=float,
    )
    return {
        "figure": str(task["figure"]),
        "trial_id": int(task["trial_id"]),
        "seed": int(task["seed"]),
        "snr_db": float(task["snr_db"]),
        "true_K": int(task["true_K"]),
        "assumed_K": int(task.get("assumed_K", task["true_K"])),
        "calibration_std_deg": float(task.get("calibration_std_deg", 0.0)),
        "T": int(scene["T"]),
        "ris_side": int(scene["M_Rx"]),
        "M_R": int(scene["M_R"]),
        "r_ue_m": float(np.linalg.norm(np.asarray(scene["p_u_true"]) - centroid)),
        "baseline": baseline,
        "data_hash": data_hash(data),
        "y_noisy_hash": y_noisy_hash(data),
        "k_mismatch_scene_mode": str(data.get("k_mismatch_scene_mode", "")),
        "true_scene_hash": str(data.get("true_scene_hash", "")),
        "estimator_scene_hash": str(data.get("estimator_scene_hash", "")),
        "first_trueK_preserved": data.get("first_trueK_preserved", ""),
    }


def _proposed_result_row(data: dict, config: dict, task: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    result = run_single_proposed_diagnostic(config, data_override=data)
    runtime_s = time.perf_counter() - start
    final = result.get("final", {})
    baseline_result = BaselineResult(
        name="proposed",
        p_u=final.get("p_u"),
        delta_t=final.get("delta_t"),
        Y_hat=final.get("Y_hat"),
        raw_objective_final=float(final.get("raw_objective_final", final.get("raw_objective", np.nan))),
        components=final.get("components", {}),
        runtime_s=runtime_s,
        diagnostics={"dictionary_mode": "proposed_assumed_K"},
    )
    return make_baseline_row(
        baseline_result,
        data,
        config,
        baseline="proposed",
        trial_id=int(task["trial_id"]),
        seed=int(task["seed"]),
        snr_db=float(task["snr_db"]),
    )


BASELINE_RUNNERS = {
    "ff_omp": run_far_field_omp_baseline,
    "ris_momp": run_ris_momp_baseline,
    "nf_mmpsr": run_near_field_mmpsr_baseline,
}


def _peb_result_row(
    data: dict,
    config: dict,
    task: dict[str, Any],
    baseline: str,
    *,
    peb_reference_type: str,
    peb_reference_data_hash: str,
    reference_warning: str = "",
) -> dict[str, Any]:
    start = time.perf_counter()
    metrics = _peb_from_efim(data, config)
    warning = str(metrics.get("warning", ""))
    if reference_warning:
        warning = reference_warning + (f"; {warning}" if warning else "")
    return {
        "baseline": baseline,
        "trial_id": int(task["trial_id"]),
        "seed": int(task["seed"]),
        "snr_db": float(task["snr_db"]),
        "position_rmse_m": float("nan"),
        "y_nmse": float("nan"),
        "raw_objective_final": float("nan"),
        "runtime_s": time.perf_counter() - start,
        "failed": False,
        "error": "",
        **metrics,
        "warning": warning,
        "peb_reference_type": str(peb_reference_type),
        "peb_reference_data_hash": str(peb_reference_data_hash),
    }


def _failure_row(task: dict[str, Any], data: dict, baseline: str, exc: BaseException) -> dict[str, Any]:
    return {
        **_common_row_fields(task, data, baseline),
        "position_rmse_m": float("nan"),
        "y_nmse": float("nan"),
        "raw_objective_final": float("nan"),
        "runtime_s": float("nan"),
        "failed": True,
        "error": f"{type(exc).__name__}: {exc}",
        "warning": "",
        "peb_position_m": float("nan"),
        "peb_reference_type": "",
        "peb_reference_data_hash": "",
    }


def _prepare_task_data(task: dict[str, Any]) -> tuple[dict, dict | None, dict]:
    true_k = int(task["true_K"])
    config = make_config(
        int(task["seed"]),
        float(task["snr_db"]),
        true_k,
        T=task.get("T"),
        ris_side=task.get("ris_side"),
        r_ue_m=task.get("r_ue_m"),
        strict_ris_geometry=bool(task.get("strict_ris_geometry", False)),
    )
    figure = str(task["figure"])
    reference_data = None
    if figure == "fig8":
        data, reference_data = make_calibration_mismatch_data(
            config, float(task["calibration_std_deg"])
        )
    elif figure == "fig9":
        data, reference_data = make_k_mismatch_data(
            config,
            int(task["assumed_K"]),
            int(task["max_assumed_K"]),
        )
        config = _configure_assumed_k(config, int(task["assumed_K"]))
    else:
        data = _make_data(config)
    return data, reference_data, config


def run_shared_trial(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate all requested methods on one immutable noisy realization."""
    apply_thread_limits(int(task.get("blas_threads", _WORKER_BLAS_THREADS)))
    data, reference_data, config = _prepare_task_data(task)
    rows: list[dict[str, Any]] = []
    for baseline in task["baselines"]:
        rss_before = memory_snapshot_mb() if bool(task.get("profile_memory", _WORKER_PROFILE_MEMORY)) else float("nan")
        try:
            with thread_limit_context(int(task.get("blas_threads", _WORKER_BLAS_THREADS))):
                if baseline == "proposed":
                    row = _proposed_result_row(data, config, task)
                elif baseline in BASELINE_RUNNERS:
                    result = BASELINE_RUNNERS[baseline](data, config)
                    row = make_baseline_row(
                        result,
                        data,
                        config,
                        baseline=baseline,
                        trial_id=int(task["trial_id"]),
                        seed=int(task["seed"]),
                        snr_db=float(task["snr_db"]),
                    )
                elif baseline == "peb":
                    row = _peb_result_row(
                        data,
                        config,
                        task,
                        baseline,
                        peb_reference_type="matched_model",
                        peb_reference_data_hash=reference_data_hash(data),
                    )
                elif baseline == "oracle_calibrated_peb":
                    if reference_data is None:
                        raise RuntimeError("oracle calibration data is unavailable")
                    row = _peb_result_row(
                        reference_data,
                        config,
                        task,
                        baseline,
                        peb_reference_type="oracle_calibrated",
                        peb_reference_data_hash=reference_data_hash(reference_data),
                        reference_warning=(
                            "Oracle-calibrated PEB reference only; not a CRB "
                            "for mismatched estimators."
                        ),
                    )
                elif baseline == "trueK_peb_reference":
                    if reference_data is None:
                        raise RuntimeError("true-K reference data is unavailable")
                    true_config = copy.deepcopy(config)
                    true_config["K"] = int(task["true_K"])
                    row = _peb_result_row(
                        reference_data,
                        true_config,
                        task,
                        baseline,
                        peb_reference_type="trueK_matched_reference",
                        peb_reference_data_hash=reference_data_hash(reference_data),
                        reference_warning=(
                            "True-K PEB reference only; not a bound under "
                            "model-order mismatch."
                        ),
                    )
                else:
                    raise ValueError(f"unknown baseline {baseline!r}")
                row = {**_common_row_fields(task, data, baseline), **row}
        except Exception as exc:  # noqa: BLE001 - failures are recorded per method.
            row = _failure_row(task, data, baseline, exc)
        if bool(task.get("trim_memory", True)):
            trim_memory()
        row["rss_mb_before"] = rss_before
        row["rss_mb_after"] = memory_snapshot_mb() if bool(task.get("profile_memory", _WORKER_PROFILE_MEMORY)) else float("nan")
        rows.append(assert_row_is_light(row))
    validate_same_y_noisy_hash(rows)
    return rows


def validate_same_y_noisy_hash(rows: Iterable[dict[str, Any]]) -> None:
    hashes = {str(row.get("y_noisy_hash", "")) for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"shared-trial y_noisy_hash mismatch: {sorted(hashes)}")


def _init_worker(blas_threads: int, profile_memory: bool) -> None:
    global _WORKER_BLAS_THREADS, _WORKER_PROFILE_MEMORY
    _WORKER_BLAS_THREADS = int(blas_threads)
    _WORKER_PROFILE_MEMORY = bool(profile_memory)
    apply_thread_limits(_WORKER_BLAS_THREADS)


class StreamingCsvWriter:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.handle: Any | None = None
        self.writer: csv.DictWriter | None = None

    def __enter__(self) -> "StreamingCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.tmp_path.open("w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        self.writer.writeheader()
        return self

    def writerows(self, rows: Iterable[dict[str, Any]]) -> None:
        assert self.writer is not None and self.handle is not None
        self.writer.writerows(rows)
        self.handle.flush()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.handle is not None:
            self.handle.close()
        if exc_type is None:
            os.replace(self.tmp_path, self.path)
        return False


def _read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_rows(rows: Iterable[dict[str, Any]], figure: str) -> list[dict[str, Any]]:
    x_field, _ = X_FIELDS[figure]
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["baseline"]), _to_float(row[x_field])), []).append(row)
    summary = []
    for (baseline, x_value), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        successful = [row for row in group if str(row.get("failed", "")).lower() != "true"]
        metric = "peb_position_m" if baseline in REFERENCE_BASELINES else "position_rmse_m"
        values = np.asarray([_to_float(row.get(metric)) for row in successful], dtype=float)
        values = values[np.isfinite(values)]
        rmse = float(np.sqrt(np.mean(values**2))) if values.size else float("nan")
        median = float(np.median(values)) if values.size else float("nan")
        p90 = float(np.percentile(values, 90.0)) if values.size else float("nan")
        p95 = float(np.percentile(values, 95.0)) if values.size else float("nan")
        runtime = np.asarray([_to_float(row.get("runtime_s")) for row in successful], dtype=float)
        runtime = runtime[np.isfinite(runtime)]
        summary.append(
            {
                "figure": figure,
                "baseline": baseline,
                "x_name": x_field,
                "x_value": x_value,
                "n": len(group),
                "success_rate": len(successful) / max(len(group), 1),
                "plot_metric": metric,
                "rmse_m": rmse,
                "median_m": median,
                "p90_m": p90,
                "p95_m": p95,
                "runtime_s_mean": float(np.mean(runtime)) if runtime.size else float("nan"),
            }
        )
    return summary


def plot_summary(summary: list[dict[str, Any]], figure: str, out_dir: pathlib.Path) -> None:
    mpl_dir = out_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib.pyplot as plt

    _, xlabel = X_FIELDS[figure]
    _, _, stem = FIGURE_FILES[figure]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    styles = {
        "proposed": ("o", "-"),
        "ff_omp": ("s", "-"),
        "ris_momp": ("^", "-"),
        "nf_mmpsr": ("D", "-"),
        "peb": ("v", "--"),
        "oracle_calibrated_peb": ("P", "--"),
        "trueK_peb_reference": ("X", "--"),
    }
    for baseline in BASELINE_LABELS:
        selected = [row for row in summary if row["baseline"] == baseline]
        if not selected:
            continue
        selected.sort(key=lambda row: float(row["x_value"]))
        marker, linestyle = styles[baseline]
        ax.plot(
            [float(row["x_value"]) for row in selected],
            [float(row["rmse_m"]) for row in selected],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            label=BASELINE_LABELS[baseline],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Position RMSE / PEB (m)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    plt.close(fig)


def write_summary_markdown(summary: list[dict[str, Any]], figure: str, out_dir: pathlib.Path) -> None:
    lines = [
        f"# {figure} summary",
        "",
        "The one-trial smoke configuration is diagnostic only; paper claims require the configured multi-trial run.",
        "",
        "| Baseline | x | RMSE (m) | Median (m) | p90 (m) | p95 (m) | Success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {BASELINE_LABELS.get(str(row['baseline']), row['baseline'])} | "
            f"{float(row['x_value']):.6g} | {float(row['rmse_m']):.6g} | "
            f"{float(row['median_m']):.6g} | {float(row['p90_m']):.6g} | "
            f"{float(row['p95_m']):.6g} | {float(row['success_rate']):.3f} |"
        )
    (out_dir / f"{figure}_summary.md").write_text("\n".join(lines) + "\n")


def _metadata_signature(args: argparse.Namespace, figure: str, baselines: list[str]) -> dict[str, Any]:
    return {
        "figure": figure,
        "n_trials": int(args.n_trials),
        "snr_db": float(args.snr_db),
        "true_K": int(args.true_k),
        "assumed_K_grid": list(args.assumed_k_grid),
        "calibration_std_grid": list(args.calibration_std_grid),
        "T_grid": list(args.T_grid),
        "ris_side_grid": list(args.ris_side_grid),
        "rue_grid": list(args.rue_grid),
        "baselines": list(baselines),
        "git_commit": _git_commit(),
        "seed": int(args.seed),
    }


def _metadata(args: argparse.Namespace, figure: str, baselines: list[str]) -> dict[str, Any]:
    return {
        **_metadata_signature(args, figure, baselines),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(sys.argv),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy_optimizer_available": bool(scipy_is_available()),
        "resource_plan": dict(args.resource_plan),
        "profile_memory": bool(args.profile_memory),
        "grid_profile": "coarse",
        "peb_policy": (
            "oracle-calibrated reference only; not a mismatched CRB"
            if figure == "fig8" and "oracle_calibrated_peb" in baselines
            else "true-K reference only; not a K-mismatch bound"
            if figure == "fig9" and "trueK_peb_reference" in baselines
            else "ordinary matched-model data-only PEB"
            if figure.startswith("fig10")
            else "no PEB"
        ),
    }


def _metadata_matches(path: pathlib.Path, expected: dict[str, Any]) -> bool:
    try:
        current = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all(current.get(key) == value for key, value in expected.items())


def build_tasks(args: argparse.Namespace, figure: str, baselines: list[str]) -> list[dict[str, Any]]:
    if figure == "fig8":
        grid = [("calibration_std_deg", value) for value in args.calibration_std_grid]
    elif figure == "fig9":
        grid = [("assumed_K", value) for value in args.assumed_k_grid]
    elif figure == "fig10a":
        grid = [("T", value) for value in args.T_grid]
    elif figure == "fig10b":
        grid = [("ris_side", value) for value in args.ris_side_grid]
    else:
        grid = [("r_ue_m", value) for value in args.rue_grid]
    tasks = []
    for name, value in grid:
        for trial_id in range(int(args.n_trials)):
            task = {
                "figure": figure,
                "trial_id": trial_id,
                "seed": _trial_seed(args.seed, figure, trial_id),
                "snr_db": args.snr_db,
                "true_K": args.true_k,
                "assumed_K": args.true_k,
                "calibration_std_deg": 0.0,
                "baselines": baselines,
                "max_assumed_K": max(args.assumed_k_grid),
                "blas_threads": args.blas_threads,
                "profile_memory": args.profile_memory,
                "trim_memory": True,
                "strict_ris_geometry": args.strict_ris_geometry,
                name: value,
            }
            tasks.append(task)
    return tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", default=DEFAULT_FIGURES)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--true-k", type=int, default=3)
    parser.add_argument("--calibration-std-grid", default="0,1,2,5,10,20")
    parser.add_argument("--assumed-k-grid", default="2,3,4,5")
    parser.add_argument("--T-grid", default="64,128,256,512")
    parser.add_argument("--ris-side-grid", default="16,24,32,48,64")
    parser.add_argument("--rue-grid", default="")
    parser.add_argument("--baselines", default=DEFAULT_BASELINES)
    parser.add_argument("--include-calibration-oracle-peb", action="store_true")
    parser.add_argument("--include-trueK-peb-reference", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results/robustness_and_scaling"))
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--process-workers", type=int, default=None)
    parser.add_argument("--blas-threads", default="auto")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--strict-ris-geometry", action="store_true")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress-log", type=pathlib.Path, default=None)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args(argv)
    args.figures = parse_figures(args.figures)
    args.calibration_std_grid = parse_float_grid(args.calibration_std_grid)
    args.assumed_k_grid = parse_int_grid(args.assumed_k_grid)
    args.T_grid = parse_int_grid(args.T_grid)
    args.ris_side_grid = parse_int_grid(args.ris_side_grid)
    args.baselines = parse_baselines(args.baselines)
    rue_config = default_config()
    set_number_of_ris_paths(rue_config, args.true_k)
    args.rue_grid = parse_float_grid(args.rue_grid) if args.rue_grid.strip() else default_rue_grid(rue_config)
    if str(args.blas_threads).lower() != "auto":
        args.blas_threads = int(args.blas_threads)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_trials <= 0 or args.true_k <= 0 or args.jobs <= 0:
        raise ValueError("n-trials, true-k, and jobs must be positive")
    if args.process_workers is not None and args.process_workers <= 0:
        raise ValueError("process-workers must be positive")
    all_task_count = sum(
        len(
            args.calibration_std_grid
            if figure == "fig8"
            else args.assumed_k_grid
            if figure == "fig9"
            else args.T_grid
            if figure == "fig10a"
            else args.ris_side_grid
            if figure == "fig10b"
            else args.rue_grid
        )
        * args.n_trials
        for figure in args.figures
    )
    args.resource_plan = resolve_hybrid_resources(
        args.jobs, args.process_workers, args.blas_threads, max(all_task_count, 1)
    )
    args.process_workers = args.resource_plan["process_workers"]
    args.blas_threads = args.resource_plan["blas_threads"]
    apply_thread_limits(args.blas_threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = (
        pathlib.Path(args.progress_log)
        if args.progress_log is not None
        else args.out_dir / "progress.jsonl"
    )
    progress = ProgressLogger(
        progress_path,
        all_task_count,
        "run_robustness_and_scaling_figures",
    )
    if not args.quiet_progress:
        print(f"Progress log: {progress_path}")
        print("Monitor with:")
        print(f"    tail -f {progress_path}")
        print(
            "    python -m src.experiments.monitor_progress "
            f"--progress-log {progress_path}"
        )
    progress.log("start", "running", message="robustness/scaling experiment started")
    print("Command:", " ".join(sys.argv))
    print("Resource plan:", json.dumps(args.resource_plan, sort_keys=True))

    for figure in args.figures:
        baselines = baselines_for_figure(
            figure,
            args.baselines,
            include_calibration_oracle_peb=args.include_calibration_oracle_peb,
            include_trueK_peb_reference=args.include_trueK_peb_reference,
        )
        trials_name, summary_name, _ = FIGURE_FILES[figure]
        trial_path = args.out_dir / trials_name
        summary_path = args.out_dir / summary_name
        metadata_path = args.out_dir / f"{figure}_metadata.json"
        expected = _metadata_signature(args, figure, baselines)
        compatible = _metadata_matches(metadata_path, expected)
        if trial_path.exists() and summary_path.exists() and compatible and not args.force_rerun:
            print(f"{figure}: reusing compatible CSV cache")
            summary = _read_csv(summary_path)
        else:
            if trial_path.exists() and not compatible and not args.force_rerun:
                print(f"{figure}: metadata mismatch; recomputing incompatible CSV")
            tasks = build_tasks(args, figure, baselines)
            initargs = (args.blas_threads, args.profile_memory)
            with StreamingCsvWriter(trial_path) as writer:
                if args.process_workers == 1:
                    _init_worker(*initargs)
                    batches = map(run_shared_trial, tasks)
                    pool_context = contextlib.nullcontext()
                else:
                    pool_context = mp.Pool(
                        processes=args.process_workers,
                        initializer=_init_worker,
                        initargs=initargs,
                    )
                    pool = pool_context.__enter__()
                    batches = pool.imap_unordered(run_shared_trial, tasks, chunksize=1)
                try:
                    for batch in batches:
                        writer.writerows(batch)
                        representative = batch[0] if batch else {}
                        failed_rows = [
                            row
                            for row in batch
                            if str(row.get("failed")).lower() == "true"
                        ]
                        progress.log(
                            "task_failed" if failed_rows else "task_done",
                            "failed" if failed_rows else "completed",
                            figure=representative.get("figure", figure),
                            baseline_or_variant=",".join(
                                str(row.get("baseline", "")) for row in batch
                            ),
                            snr_db=representative.get("snr_db", ""),
                            trial_id=representative.get("trial_id", ""),
                            seed=representative.get("seed", ""),
                            K=representative.get("assumed_K", ""),
                            message="robustness/scaling trial batch completed",
                            error="; ".join(
                                str(row.get("error", ""))
                                for row in failed_rows
                            ),
                        )
                finally:
                    if args.process_workers != 1:
                        pool_context.__exit__(None, None, None)
            rows = _read_csv(trial_path)
            summary = summarize_rows(rows, figure)
            _write_csv(summary_path, summary)
            metadata_path.write_text(json.dumps(_metadata(args, figure, baselines), indent=2))
        if not args.no_plots:
            plot_summary(summary, figure, args.out_dir)
        write_summary_markdown(summary, figure, args.out_dir)
        print(f"{figure}: wrote {trial_path.name}, {summary_path.name}")
    progress.log(
        "finished", "completed", message="robustness/scaling experiment finished"
    )
    progress.close()


if __name__ == "__main__":
    main()
