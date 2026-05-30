"""Monte Carlo experiment: Stage-II value as VP-WNLS initialization.

The primary comparison is Stage-I + VP versus Stage-I + Stage-II + VP.
True parameters are used only for synthetic data generation and offline metrics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import pathlib
import sys
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from src.config import default_config
    from src.diagnostics import estimate_position_from_ris_eta
    from src.estimators import (
        initialize_from_hankel,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from src.geometry import polarization_vector, position_from_local_geometry
    from src.metrics import position_rmse, relative_nmse
    from src.projections_delay import bq_from_poles, pole_from_tau, tau_from_pole
    from src.projections_evs import project_evs_factor
    from src.projections_ris import (
        compressed_exact_response,
        local_ris_search_config,
        project_ris_factor,
        scaled_residual,
    )
    from src.tensor_utils import hankelize_frequency, reconstruct_z, z_design_column
    from src.utils import scipy_is_available, solve_lstsq
else:
    from ..channel_model import add_awgn, channel_components, generate_scene, synthesize_raw_tensor
    from ..config import default_config
    from ..diagnostics import estimate_position_from_ris_eta
    from ..estimators import (
        initialize_from_hankel,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from ..geometry import polarization_vector, position_from_local_geometry
    from ..metrics import position_rmse, relative_nmse
    from ..projections_delay import bq_from_poles, pole_from_tau, tau_from_pole
    from ..projections_evs import project_evs_factor
    from ..projections_ris import (
        compressed_exact_response,
        local_ris_search_config,
        project_ris_factor,
        scaled_residual,
    )
    from ..tensor_utils import hankelize_frequency, reconstruct_z, z_design_column
    from ..utils import scipy_is_available, solve_lstsq


PIPELINE_STAGE1 = "Stage-I only"
PIPELINE_STAGE1_VP = "Stage-I + VP"
PIPELINE_STAGE1_STAGE2 = "Stage-I + Stage-II"
PIPELINE_STAGE1_STAGE2_VP = "Stage-I + Stage-II + VP"

INIT_MODE_CHOICES = ("clean", "weak", "perturbed", "unstructured")
_WARNED_MESSAGES: set[str] = set()

SUCCESS_FIELDS = [
    "success_pos_10cm",
    "success_pos_20cm",
    "success_pos_50cm",
    "success_nmse_1e3",
    "success_nmse_1e2",
]

PHYSICAL_CONSISTENCY_FIELDS = [
    "delay_geometry_consistency_RMSE",
    "compressed_ris_manifold_residual",
    "evs_maxwell_consistency_residual",
    "common_delay_pole_consistency",
]

VP_DIAGNOSTIC_FIELDS = [
    "optimizer_converged",
    "optimizer_status",
    "optimizer_message",
    "nfev",
    "initial_noisy_objective",
    "final_noisy_objective",
    "relative_noisy_objective_decrease",
    "final_position_error",
    "final_Y_NMSE",
]

VP_FAILURE_FIELDS = [
    "fail_convergence",
    "fail_position_50cm",
    "fail_position_1m",
    "fail_nmse_1e2",
    "fail_noisy_objective_increase",
]

FIELDNAMES = [
    "init_mode",
    "init_mode_effective",
    "init_warning",
    "SNR_dB",
    "T",
    "ris_shape",
    "K",
    "N",
    "trial",
    "seed",
    "pipeline",
    "failed",
    "error",
    "Y_NMSE",
    "Z_NMSE",
    "raw_noisy_residual",
    "Z_noisy_residual",
    "position_error",
    "range_RMSE",
    "delay_RMSE",
    "polarization_angle_RMSE",
    *PHYSICAL_CONSISTENCY_FIELDS,
    "vp_converged",
    "vp_nfev",
    "vp_final_noisy_objective",
    "vp_initial_noisy_objective",
    "vp_initial_Y_NMSE",
    "vp_worsened_noisy_objective",
    "vp_worsened_true_nmse",
    *VP_DIAGNOSTIC_FIELDS,
    *VP_FAILURE_FIELDS,
    "stage2_worsened_noisy_z_residual",
    "stage2_worsened_true_z_nmse",
    "stage2_initial_Z_noisy_residual",
    "stage2_final_Z_noisy_residual",
    "stage2_initial_Z_NMSE",
    "stage2_final_Z_NMSE",
    *SUCCESS_FIELDS,
]

SUMMARY_FIELDNAMES = [
    "init_mode",
    "init_mode_effective",
    "init_warning",
    "SNR_dB",
    "T",
    "ris_shape",
    "pipeline",
    "num_rows",
    "num_failed",
    "failure_rate",
    "Y_NMSE_mean",
    "Y_NMSE_std",
    "Y_NMSE_median",
    "position_error_mean",
    "position_error_std",
    "position_error_median",
    "position_error_p90",
    "position_error_p95",
    "range_RMSE_mean",
    "range_RMSE_std",
    "range_RMSE_median",
    "delay_RMSE_mean",
    "delay_RMSE_std",
    "delay_RMSE_median",
    "delay_geometry_consistency_RMSE_mean",
    "delay_geometry_consistency_RMSE_std",
    "delay_geometry_consistency_RMSE_median",
    "compressed_ris_manifold_residual_mean",
    "compressed_ris_manifold_residual_std",
    "compressed_ris_manifold_residual_median",
    "evs_maxwell_consistency_residual_mean",
    "evs_maxwell_consistency_residual_std",
    "evs_maxwell_consistency_residual_median",
    "common_delay_pole_consistency_mean",
    "common_delay_pole_consistency_std",
    "common_delay_pole_consistency_median",
    "success_pos_10cm_rate",
    "success_pos_20cm_rate",
    "success_pos_50cm_rate",
    "success_nmse_1e3_rate",
    "success_nmse_1e2_rate",
    "vp_convergence_rate",
    "vp_worsened_noisy_rate",
    "vp_worsened_true_nmse_rate",
    "fail_convergence_rate",
    "fail_position_50cm_rate",
    "fail_position_1m_rate",
    "fail_nmse_1e2_rate",
    "fail_noisy_objective_increase_rate",
    "stage2_worsened_noisy_z_rate",
    "stage2_worsened_true_z_nmse_rate",
    "improvement_rate_position",
    "improvement_rate_Y_NMSE",
    "VP_success_gain_20cm",
    "VP_success_gain_50cm",
]


def _parse_shape(value: str) -> tuple[int, int]:
    text = value.lower().replace(" ", "")
    parts = text.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"RIS shape must look like 16x16, got {value!r}")
    try:
        shape = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid RIS shape {value!r}") from exc
    if shape[0] <= 0 or shape[1] <= 0:
        raise argparse.ArgumentTypeError(f"RIS shape entries must be positive, got {value!r}")
    return shape


def _shape_label(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _warn_once(message: str) -> None:
    if message not in _WARNED_MESSAGES:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _WARNED_MESSAGES.add(message)


def _relative_sse(x_hat: np.ndarray, x_ref: np.ndarray, eps: float = 1e-12) -> float:
    assert x_hat.shape == x_ref.shape, "relative SSE inputs must have matching shapes"
    return float(np.linalg.norm(x_hat - x_ref) ** 2 / (np.linalg.norm(x_ref) ** 2 + eps))


def _circular_rmse(angle_hat: np.ndarray | None, angle_true: np.ndarray) -> float:
    if angle_hat is None:
        return float("nan")
    angle_hat = np.asarray(angle_hat, dtype=float)
    angle_true = np.asarray(angle_true, dtype=float)
    if angle_hat.shape != angle_true.shape:
        return float("nan")
    diff = np.angle(np.exp(1j * (angle_hat - angle_true)))
    return float(np.linalg.norm(diff) / np.sqrt(diff.size))


def _estimate_unit_pole_from_factor(vector: np.ndarray, eps: float) -> complex:
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    if vector.size < 2:
        return 1.0 + 0.0j
    numerator = np.vdot(vector[:-1], vector[1:])
    denominator = float(np.vdot(vector[:-1], vector[:-1]).real)
    if denominator <= eps or abs(numerator) <= eps:
        return 1.0 + 0.0j
    pole = numerator / denominator
    return pole / abs(pole)


def _delay_geometry_consistency_rmse(
    tau_hat: np.ndarray,
    range_hat: np.ndarray,
    scene: dict,
    delta_t_hat: float | None,
) -> float:
    tau_hat = np.asarray(tau_hat, dtype=float)
    range_hat = np.asarray(range_hat, dtype=float)
    implied_dt = tau_hat - (range_hat + scene["d_RB"]) / scene["c0"]
    if delta_t_hat is None:
        delta_t_hat = float(np.median(implied_dt))
    residual = implied_dt - float(delta_t_hat)
    return float(np.linalg.norm(residual) / np.sqrt(residual.size))


def _compressed_ris_manifold_residual(
    c_mat: np.ndarray,
    ris_eta: np.ndarray | None,
    scene: dict,
    config: dict,
) -> float:
    values = []
    for k in range(scene["K"]):
        search = local_ris_search_config(scene, config, k)
        current_eta = None if ris_eta is None else np.asarray(ris_eta[k], dtype=float)
        try:
            if current_eta is None:
                projection = project_ris_factor(
                    c_mat[:, k],
                    scene["Omega"][k],
                    scene["a_RB"][k],
                    scene["ris_grid"],
                    scene["wavelength"],
                    search,
                    config["eps"],
                )
                relative = projection.get("exact_relative_residual")
                if relative is None:
                    relative = projection["relative_residual"]
                values.append(float(relative) ** 2)
                continue

            lower = np.array(
                [
                    search["range_bounds"][0],
                    search["elev_bounds"][0],
                    search["az_bounds"][0],
                ],
                dtype=float,
            )
            upper = np.array(
                [
                    search["range_bounds"][1],
                    search["elev_bounds"][1],
                    search["az_bounds"][1],
                ],
                dtype=float,
            )
            eta_start = np.clip(current_eta, lower, upper)

            def exact_objective(eta_local: np.ndarray) -> float:
                h_model = compressed_exact_response(
                    eta_local,
                    scene["Omega"][k],
                    scene["a_RB"][k],
                    scene["ris_grid"],
                    scene["wavelength"],
                )
                value, _ = scaled_residual(c_mat[:, k], h_model, config["eps"])
                return float(value / (np.linalg.norm(c_mat[:, k]) ** 2 + config["eps"]))

            value = exact_objective(eta_start)
            if scipy_is_available():
                from scipy.optimize import minimize

                result = minimize(
                    exact_objective,
                    eta_start,
                    method="L-BFGS-B",
                    bounds=list(zip(lower, upper)),
                    options={"maxiter": 40, "ftol": 1e-12},
                )
                value = min(value, float(result.fun))
            values.append(float(value))
        except Exception:  # noqa: BLE001 - metric failure should not stop a Monte Carlo trial.
            values.append(float("nan"))
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def _evs_maxwell_consistency_residual(a_mat: np.ndarray, scene: dict, config: dict) -> float:
    values = []
    eps = config["eps"]
    for k in range(scene["K"]):
        try:
            projection = project_evs_factor(
                a_mat[:, k], scene["v_B"][k], scene["Theta"][k], eps
            )
            projected = projection["scale"] * projection["a"]
            residual = np.linalg.norm(a_mat[:, k] - projected) ** 2 / (
                np.linalg.norm(a_mat[:, k]) ** 2 + eps
            )
            values.append(float(residual))
        except Exception:  # noqa: BLE001
            values.append(float("nan"))
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def _common_delay_pole_consistency(b_mat: np.ndarray, q_mat: np.ndarray, config: dict) -> float:
    values = []
    for k in range(b_mat.shape[1]):
        z_b = _estimate_unit_pole_from_factor(b_mat[:, k], config["eps"])
        z_q = _estimate_unit_pole_from_factor(q_mat[:, k], config["eps"])
        values.append(abs(z_b - z_q))
    return float(np.mean(values)) if values else float("nan")


def _physical_consistency_metrics(
    *,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    q_mat: np.ndarray,
    c_mat: np.ndarray,
    tau_hat: np.ndarray,
    range_hat: np.ndarray,
    delta_t_hat: float | None,
    ris_eta: np.ndarray | None,
    scene: dict,
    config: dict,
) -> dict:
    return {
        "delay_geometry_consistency_RMSE": _delay_geometry_consistency_rmse(
            tau_hat, range_hat, scene, delta_t_hat
        ),
        "compressed_ris_manifold_residual": _compressed_ris_manifold_residual(
            c_mat, ris_eta, scene, config
        ),
        "evs_maxwell_consistency_residual": _evs_maxwell_consistency_residual(
            a_mat, scene, config
        ),
        "common_delay_pole_consistency": _common_delay_pole_consistency(
            b_mat, q_mat, config
        ),
    }


def _sanitize_scene_for_estimation(scene: dict) -> dict:
    forbidden = {"p_u_true", "gamma_true", "eta_true", "beta_true", "delta_t_true"}
    return {key: copy.deepcopy(value) for key, value in scene.items() if key not in forbidden}


def _sanitize_config_for_estimation(config: dict) -> dict:
    forbidden = {"p_u_true", "delta_t_true"}
    return {key: copy.deepcopy(value) for key, value in config.items() if key not in forbidden}


def _assert_blind_estimation_inputs(data: dict) -> None:
    scene_forbidden = {"p_u_true", "gamma_true", "eta_true", "beta_true", "delta_t_true"}
    config_forbidden = {"p_u_true", "delta_t_true"}
    leaked_scene = sorted(scene_forbidden.intersection(data["scene_est"]))
    leaked_config = sorted(config_forbidden.intersection(data["config_est"]))
    if leaked_scene or leaked_config:
        raise RuntimeError(
            "true parameters leaked into estimator inputs: "
            f"scene={leaked_scene}, config={leaked_config}"
        )


def _make_config(
    *,
    snr_db: float,
    t_dim: int,
    ris_shape: tuple[int, int],
    seed: int,
    k_paths: int,
    n_dim: int,
) -> dict:
    config = default_config()
    config["SNR_dB"] = float(snr_db)
    config["T"] = int(t_dim)
    config["ris_shape"] = tuple(ris_shape)
    config["seed"] = int(seed)
    config["K"] = int(k_paths)
    config["N"] = int(n_dim)
    config["P"] = min(int(config["P"]), int(n_dim))
    config["enable_global_vp"] = False
    return config


def _make_trial_data(config: dict) -> dict:
    rng = np.random.default_rng(config["seed"])
    scene = generate_scene(config, rng)
    true_components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_true = synthesize_raw_tensor(true_components, scene["beta_true"])
    y_noisy, noise_variance = add_awgn(y_true, config["SNR_dB"], rng)
    return {
        "scene_true": scene,
        "scene_est": _sanitize_scene_for_estimation(scene),
        "config_est": _sanitize_config_for_estimation(config),
        "true_components": true_components,
        "Y_true": y_true,
        "Y_noisy": y_noisy,
        "Z_true": hankelize_frequency(y_true, scene["P"]),
        "Z_noisy": hankelize_frequency(y_noisy, scene["P"]),
        "noise_variance": noise_variance,
    }


def _weak_stage1_config(config: dict) -> dict:
    weak_config = copy.deepcopy(config)
    ris_search = dict(weak_config["ris_search"])
    ris_search["num_range"] = min(int(ris_search.get("num_range", 15)), 5)
    ris_search["num_elev"] = min(int(ris_search.get("num_elev", 9)), 3)
    ris_search["num_az"] = min(int(ris_search.get("num_az", 25)), 7)
    ris_search["num_exact_refine_starts"] = min(
        int(ris_search.get("num_exact_refine_starts", 6)), 1
    )
    ris_search["num_lift_candidates"] = min(int(ris_search.get("num_lift_candidates", 4)), 1)
    ris_search["num_lift_steps"] = min(int(ris_search.get("num_lift_steps", 4)), 1)
    weak_config["ris_search"] = ris_search
    return weak_config


def _normalize_complex_vector(vector: np.ndarray, eps: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= eps:
        return vector.copy()
    return vector / norm


def _refit_structured_z_weights(estimate: dict, data: dict) -> dict:
    scene = data["scene_est"]
    b_mat, q_mat = bq_from_poles(estimate["poles"], scene["P"], scene["L"])
    design = np.column_stack(
        [
            z_design_column(
                estimate["A"][:, k],
                b_mat[:, k],
                q_mat[:, k],
                estimate["C"][:, k],
            )
            for k in range(scene["K"])
        ]
    )
    beta_z = solve_lstsq(design, data["Z_noisy"].reshape(-1), reg=1e-12)
    estimate["B"] = b_mat
    estimate["Q"] = q_mat
    estimate["beta_z"] = beta_z
    estimate["Z_hat"] = reconstruct_z(beta_z, estimate["A"], b_mat, q_mat, estimate["C"])
    return estimate


def _rebuild_structured_factors_from_physical_params(estimate: dict, data: dict) -> dict:
    scene = data["scene_est"]
    config = data["config_est"]
    for k in range(scene["K"]):
        pol = scene["Theta"][k] @ polarization_vector(estimate["gamma"][k], estimate["eta_pol"][k])
        a_model = np.kron(scene["v_B"][k], pol)
        estimate["A"][:, k] = _normalize_complex_vector(a_model, config["eps"])

        c_model = compressed_exact_response(
            estimate["ris_eta"][k],
            scene["Omega"][k],
            scene["a_RB"][k],
            scene["ris_grid"],
            scene["wavelength"],
        )
        estimate["C"][:, k] = _normalize_complex_vector(c_model, config["eps"])
    return _refit_structured_z_weights(estimate, data)


def _perturb_structured_estimate(estimate: dict, data: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    scene = data["scene_est"]
    config = data["config_est"]
    perturbed = copy.deepcopy(estimate)

    range_std = float(config.get("range_perturb_std_m", 0.5))
    angle_std = float(config.get("angle_perturb_std_rad", np.deg2rad(5.0)))
    delay_std = float(config.get("delay_perturb_std_s", 0.5e-9))
    pol_std = float(config.get("pol_perturb_std_rad", np.deg2rad(5.0)))

    for k in range(scene["K"]):
        search = local_ris_search_config(scene, config, k)
        lower = np.array(
            [
                search["range_bounds"][0],
                search["elev_bounds"][0],
                search["az_bounds"][0],
            ],
            dtype=float,
        )
        upper = np.array(
            [
                search["range_bounds"][1],
                search["elev_bounds"][1],
                search["az_bounds"][1],
            ],
            dtype=float,
        )
        perturb = np.array(
            [
                rng.normal(0.0, range_std),
                rng.normal(0.0, angle_std),
                rng.normal(0.0, angle_std),
            ],
            dtype=float,
        )
        perturbed["ris_eta"][k] = np.clip(perturbed["ris_eta"][k] + perturb, lower, upper)

    tau_hat = np.array([tau_from_pole(pole, scene["delta_f"]) for pole in perturbed["poles"]])
    tau_period = 1.0 / scene["delta_f"]
    tau_hat = tau_hat + rng.normal(0.0, delay_std, size=scene["K"])
    tau_hat = np.clip(tau_hat, 0.0, np.nextafter(tau_period, 0.0))
    perturbed["poles"] = np.array([pole_from_tau(tau, scene["delta_f"]) for tau in tau_hat])

    perturbed["eta_pol"] = np.angle(
        np.exp(1j * (perturbed["eta_pol"] + rng.normal(0.0, pol_std, size=scene["K"])))
    )
    return _rebuild_structured_factors_from_physical_params(perturbed, data)


def _prepare_initial_estimate(data: dict, config: dict, seed: int) -> dict:
    init_mode = str(config.get("init_mode", "clean"))
    if init_mode == "unstructured":
        message = "TODO: unstructured CPD/ALS initialization is not available; skipping trials."
        _warn_once(message)
        raise NotImplementedError(message)

    stage1_config = data["config_est"]
    if init_mode == "weak":
        stage1_config = _weak_stage1_config(data["config_est"])
    estimate_initial = initialize_from_hankel(data["Z_noisy"], data["scene_est"], stage1_config)
    if init_mode == "perturbed":
        return _perturb_structured_estimate(estimate_initial, data, seed + 77_777)
    return estimate_initial


def _base_row(config: dict, pipeline: str, trial: int, seed: int) -> dict:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "init_mode": config.get("init_mode", "clean"),
            "init_mode_effective": config.get("init_mode_effective", config.get("init_mode", "clean")),
            "init_warning": config.get("init_warning", ""),
            "SNR_dB": float(config["SNR_dB"]),
            "T": int(config["T"]),
            "ris_shape": _shape_label(tuple(config["ris_shape"])),
            "K": int(config["K"]),
            "N": int(config["N"]),
            "trial": int(trial),
            "seed": int(seed),
            "pipeline": pipeline,
            "failed": False,
            "error": "",
        }
    )
    return row


def _failed_row(config: dict, pipeline: str, trial: int, seed: int, error: BaseException) -> dict:
    row = _base_row(config, pipeline, trial, seed)
    row["failed"] = True
    row["error"] = f"{type(error).__name__}: {error}".replace("\n", " | ")[:2000]
    for field in SUCCESS_FIELDS:
        row[field] = False
    if "VP" in pipeline:
        row["vp_converged"] = False
        row["optimizer_converged"] = False
        row["optimizer_message"] = row["error"]
        row["fail_convergence"] = True
        row["fail_position_50cm"] = True
        row["fail_position_1m"] = True
        row["fail_nmse_1e2"] = True
    return row


def _success_flags(y_nmse: float, position_error: float) -> dict:
    finite_y = np.isfinite(y_nmse)
    finite_pos = np.isfinite(position_error)
    return {
        "success_pos_10cm": bool(finite_pos and position_error < 0.10),
        "success_pos_20cm": bool(finite_pos and position_error < 0.20),
        "success_pos_50cm": bool(finite_pos and position_error < 0.50),
        "success_nmse_1e3": bool(finite_y and y_nmse < 1.0e-3),
        "success_nmse_1e2": bool(finite_y and y_nmse < 1.0e-2),
    }


def _fill_common_metrics(
    row: dict,
    *,
    y_hat: np.ndarray,
    p_hat: np.ndarray,
    range_hat: np.ndarray,
    tau_hat: np.ndarray,
    eta_hat: np.ndarray | None,
    data: dict,
) -> dict:
    scene_true = data["scene_true"]
    true_components = data["true_components"]
    z_hat = hankelize_frequency(y_hat, scene_true["P"])
    y_nmse = relative_nmse(y_hat, data["Y_true"])
    z_nmse = relative_nmse(z_hat, data["Z_true"])
    raw_noisy = _relative_sse(y_hat, data["Y_noisy"])
    z_noisy = _relative_sse(z_hat, data["Z_noisy"])
    position_error = position_rmse(np.asarray(p_hat, dtype=float), scene_true["p_u_true"])
    range_rmse = float(
        np.linalg.norm(np.asarray(range_hat) - true_components["ranges"]) / np.sqrt(scene_true["K"])
    )
    delay_rmse = float(
        np.linalg.norm(np.asarray(tau_hat) - true_components["taus"]) / np.sqrt(scene_true["K"])
    )
    angle_rmse = _circular_rmse(eta_hat, scene_true["eta_true"])
    row.update(
        {
            "Y_NMSE": y_nmse,
            "Z_NMSE": z_nmse,
            "raw_noisy_residual": raw_noisy,
            "Z_noisy_residual": z_noisy,
            "position_error": position_error,
            "range_RMSE": range_rmse,
            "delay_RMSE": delay_rmse,
            "polarization_angle_RMSE": angle_rmse,
        }
    )
    row.update(_success_flags(y_nmse, position_error))
    return row


def _structured_metrics_row(
    *,
    config: dict,
    pipeline: str,
    trial: int,
    seed: int,
    estimate: dict,
    data: dict,
) -> dict:
    scene_est = data["scene_est"]
    y_hat = reconstruct_raw_tensor_from_structured_estimate(estimate, scene_est)
    tau_hat = np.array([tau_from_pole(pole, scene_est["delta_f"]) for pole in estimate["poles"]])
    row = _base_row(config, pipeline, trial, seed)
    row = _fill_common_metrics(
        row,
        y_hat=y_hat,
        p_hat=estimate_position_from_ris_eta(scene_est, estimate),
        range_hat=estimate["ris_eta"][:, 0],
        tau_hat=tau_hat,
        eta_hat=estimate.get("eta_pol"),
        data=data,
    )
    b_mat = estimate.get("B")
    q_mat = estimate.get("Q")
    if b_mat is None or q_mat is None:
        b_mat, q_mat = bq_from_poles(estimate["poles"], scene_est["P"], scene_est["L"])
    row.update(
        _physical_consistency_metrics(
            a_mat=estimate["A"],
            b_mat=b_mat,
            q_mat=q_mat,
            c_mat=estimate["C"],
            tau_hat=tau_hat,
            range_hat=estimate["ris_eta"][:, 0],
            delta_t_hat=None,
            ris_eta=estimate["ris_eta"],
            scene=scene_est,
            config=data["config_est"],
        )
    )
    return row


def _raw_design_matrix(components: dict) -> np.ndarray:
    a_mat = components["a_EVS"].T
    d_mat = components["d"].T
    c_mat = components["c"].T
    i_dim, k_paths = a_mat.shape
    n_dim = d_mat.shape[0]
    t_dim = c_mat.shape[0]
    design = np.empty((i_dim * n_dim * t_dim, k_paths), dtype=complex)
    for k in range(k_paths):
        design[:, k] = (
            a_mat[:, k, None, None]
            * d_mat[None, :, k, None]
            * c_mat[None, None, :, k]
        ).reshape(-1)
    return design


def _vp_initial_model_from_estimate(estimate: dict, data: dict) -> dict:
    """Return the raw-domain VP objective model at the optimizer's initial point."""
    scene = data["scene_est"]
    config = data["config_est"]
    positions = []
    for k in range(scene["K"]):
        positions.append(
            position_from_local_geometry(
                scene["ris_centers"][k],
                scene["rotations"][k],
                estimate["ris_eta"][k, 0],
                estimate["ris_eta"][k, 1],
                estimate["ris_eta"][k, 2],
            )
        )
    p_init = np.mean(np.asarray(positions), axis=0)
    p_init = np.clip(p_init, config["ue_bounds"][:, 0], config["ue_bounds"][:, 1])

    dt_values = []
    for k in range(scene["K"]):
        tau_hat = tau_from_pole(estimate["poles"][k], scene["delta_f"])
        range_hat = estimate["ris_eta"][k, 0]
        dt_values.append(tau_hat - (range_hat + scene["d_RB"][k]) / scene["c0"])
    dt_init = float(np.median(dt_values))
    dt_init = float(np.clip(dt_init, *config["delta_t_bounds"]))

    components = channel_components(scene, p_init, dt_init, estimate["gamma"], estimate["eta_pol"])
    design = _raw_design_matrix(components)
    beta = solve_lstsq(design, data["Y_noisy"].reshape(-1), reg=1e-12)
    y_hat = synthesize_raw_tensor(components, beta)
    return {
        "Y_hat": y_hat,
        "p_u": p_init,
        "delta_t": dt_init,
        "components": components,
        "eta_pol": estimate.get("eta_pol"),
    }


def _vp_initial_reference_row(estimate: dict, data: dict) -> dict:
    model = _vp_initial_model_from_estimate(estimate, data)
    row = {field: "" for field in FIELDNAMES}
    return _fill_common_metrics(
        row,
        y_hat=model["Y_hat"],
        p_hat=model["p_u"],
        range_hat=model["components"]["ranges"],
        tau_hat=model["components"]["taus"],
        eta_hat=model["eta_pol"],
        data=data,
    )


def _vp_metrics_row(
    *,
    config: dict,
    pipeline: str,
    trial: int,
    seed: int,
    final: dict,
    init_row: dict,
    data: dict,
) -> dict:
    row = _base_row(config, pipeline, trial, seed)
    scene_est = data["scene_est"]
    row = _fill_common_metrics(
        row,
        y_hat=final["Y_hat"],
        p_hat=final["p_u"],
        range_hat=final["components"]["ranges"],
        tau_hat=final["components"]["taus"],
        eta_hat=final.get("eta_pol"),
        data=data,
    )
    b_mat, q_mat = bq_from_poles(
        final["components"]["poles"], scene_est["P"], scene_est["L"]
    )
    ris_eta = np.column_stack(
        [
            final["components"]["ranges"],
            final["components"]["elevations"],
            final["components"]["azimuths"],
        ]
    )
    row.update(
        _physical_consistency_metrics(
            a_mat=final["components"]["a_EVS"].T,
            b_mat=b_mat,
            q_mat=q_mat,
            c_mat=final["components"]["c"].T,
            tau_hat=final["components"]["taus"],
            range_hat=final["components"]["ranges"],
            delta_t_hat=final.get("delta_t"),
            ris_eta=ris_eta,
            scene=scene_est,
            config=data["config_est"],
        )
    )
    optimizer = final.get("optimizer", {})
    vp_initial_noisy = init_row["raw_noisy_residual"]
    vp_initial_true = init_row["Y_NMSE"]
    vp_final_noisy = row["raw_noisy_residual"]
    relative_decrease = (
        (vp_initial_noisy - vp_final_noisy) / max(vp_initial_noisy, 1.0e-300)
        if np.isfinite(vp_initial_noisy) and np.isfinite(vp_final_noisy)
        else float("nan")
    )
    optimizer_converged = bool(optimizer.get("success", False))
    row.update(
        {
            "vp_converged": optimizer_converged,
            "vp_nfev": optimizer.get("n_eval", ""),
            "vp_final_noisy_objective": vp_final_noisy,
            "vp_initial_noisy_objective": vp_initial_noisy,
            "vp_initial_Y_NMSE": vp_initial_true,
            "vp_worsened_noisy_objective": bool(vp_final_noisy > vp_initial_noisy + 1.0e-12),
            "vp_worsened_true_nmse": bool(row["Y_NMSE"] > vp_initial_true + 1.0e-12),
            "optimizer_converged": optimizer_converged,
            "optimizer_status": optimizer.get("status", ""),
            "optimizer_message": str(optimizer.get("message", "")).replace("\n", " | "),
            "nfev": optimizer.get("n_eval", ""),
            "initial_noisy_objective": vp_initial_noisy,
            "final_noisy_objective": vp_final_noisy,
            "relative_noisy_objective_decrease": relative_decrease,
            "final_position_error": row["position_error"],
            "final_Y_NMSE": row["Y_NMSE"],
            "fail_convergence": not optimizer_converged,
            "fail_position_50cm": bool(row["position_error"] > 0.50),
            "fail_position_1m": bool(row["position_error"] > 1.00),
            "fail_nmse_1e2": bool(row["Y_NMSE"] > 1.0e-2),
            "fail_noisy_objective_increase": bool(
                vp_final_noisy > vp_initial_noisy * (1.0 + 1.0e-6)
            ),
        }
    )
    return row


def _add_stage2_comparison_flags(row: dict, initial_row: dict, stage2_row: dict) -> dict:
    initial_noisy_z = initial_row["Z_noisy_residual"]
    stage2_noisy_z = stage2_row["Z_noisy_residual"]
    initial_true_z = initial_row["Z_NMSE"]
    stage2_true_z = stage2_row["Z_NMSE"]
    row.update(
        {
            "stage2_worsened_noisy_z_residual": bool(stage2_noisy_z > initial_noisy_z + 1.0e-12),
            "stage2_worsened_true_z_nmse": bool(stage2_true_z > initial_true_z + 1.0e-12),
            "stage2_initial_Z_noisy_residual": initial_noisy_z,
            "stage2_final_Z_noisy_residual": stage2_noisy_z,
            "stage2_initial_Z_NMSE": initial_true_z,
            "stage2_final_Z_NMSE": stage2_true_z,
        }
    )
    return row


def _run_trial(config: dict, trial: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    try:
        data = _make_trial_data(config)
        _assert_blind_estimation_inputs(data)
        estimate_initial = _prepare_initial_estimate(data, config, seed)
        initial_row = _structured_metrics_row(
            config=config,
            pipeline=PIPELINE_STAGE1,
            trial=trial,
            seed=seed,
            estimate=estimate_initial,
            data=data,
        )
        rows.append(initial_row)
    except Exception as exc:  # noqa: BLE001 - each failed trial must be recorded.
        error = RuntimeError(traceback.format_exc(limit=6))
        for pipeline in (
            PIPELINE_STAGE1,
            PIPELINE_STAGE1_VP,
            PIPELINE_STAGE1_STAGE2,
            PIPELINE_STAGE1_STAGE2_VP,
        ):
            rows.append(_failed_row(config, pipeline, trial, seed, error))
        return rows

    try:
        direct_vp_init_row = _vp_initial_reference_row(estimate_initial, data)
        direct_vp = refine_global_raw(
            data["Y_noisy"], data["scene_est"], data["config_est"], copy.deepcopy(estimate_initial)
        )
        rows.append(
            _vp_metrics_row(
                config=config,
                pipeline=PIPELINE_STAGE1_VP,
                trial=trial,
                seed=seed,
                final=direct_vp,
                init_row=direct_vp_init_row,
                data=data,
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(_failed_row(config, PIPELINE_STAGE1_VP, trial, seed, exc))

    try:
        estimate_stage2, _ = structured_refinement(
            data["Z_noisy"], data["scene_est"], data["config_est"], copy.deepcopy(estimate_initial)
        )
        stage2_row = _structured_metrics_row(
            config=config,
            pipeline=PIPELINE_STAGE1_STAGE2,
            trial=trial,
            seed=seed,
            estimate=estimate_stage2,
            data=data,
        )
        _add_stage2_comparison_flags(stage2_row, initial_row, stage2_row)
        rows.append(stage2_row)
    except Exception as exc:  # noqa: BLE001
        rows.append(_failed_row(config, PIPELINE_STAGE1_STAGE2, trial, seed, exc))
        rows.append(_failed_row(config, PIPELINE_STAGE1_STAGE2_VP, trial, seed, exc))
        return rows

    try:
        stage2_vp_init_row = _vp_initial_reference_row(estimate_stage2, data)
        stage2_vp = refine_global_raw(
            data["Y_noisy"], data["scene_est"], data["config_est"], copy.deepcopy(estimate_stage2)
        )
        stage2_vp_row = _vp_metrics_row(
            config=config,
            pipeline=PIPELINE_STAGE1_STAGE2_VP,
            trial=trial,
            seed=seed,
            final=stage2_vp,
            init_row=stage2_vp_init_row,
            data=data,
        )
        _add_stage2_comparison_flags(stage2_vp_row, initial_row, stage2_row)
        rows.append(stage2_vp_row)
    except Exception as exc:  # noqa: BLE001
        failed = _failed_row(config, PIPELINE_STAGE1_STAGE2_VP, trial, seed, exc)
        _add_stage2_comparison_flags(failed, initial_row, stage2_row)
        rows.append(failed)

    return rows


def _trial_options_from_args(args: argparse.Namespace) -> dict:
    return {
        "seed_base": int(args.seed_base),
        "k": int(args.k),
        "n": int(args.n),
        "init_mode": str(args.init_mode),
        "range_perturb_std_m": float(args.range_perturb_std),
        "angle_perturb_std_rad": float(np.deg2rad(args.angle_perturb_std)),
        "delay_perturb_std_s": float(args.delay_perturb_std * 1.0e-9),
        "pol_perturb_std_rad": float(np.deg2rad(args.pol_perturb_std)),
    }


def _config_for_trial(
    *,
    setting_index: int,
    snr_db: float,
    t_dim: int,
    ris_shape: tuple[int, int],
    trial: int,
    options: dict,
) -> tuple[dict, int]:
    seed = int(options["seed_base"] + setting_index * 100_000 + trial)
    config = _make_config(
        snr_db=snr_db,
        t_dim=t_dim,
        ris_shape=ris_shape,
        seed=seed,
        k_paths=options["k"],
        n_dim=options["n"],
    )
    config["init_mode"] = options["init_mode"]
    config["init_mode_effective"] = options["init_mode"]
    config["init_warning"] = (
        "TODO: unstructured CPD/ALS initialization is not available; trial skipped."
        if options["init_mode"] == "unstructured"
        else ""
    )
    config["range_perturb_std_m"] = float(options["range_perturb_std_m"])
    config["angle_perturb_std_rad"] = float(options["angle_perturb_std_rad"])
    config["delay_perturb_std_s"] = float(options["delay_perturb_std_s"])
    config["pol_perturb_std_rad"] = float(options["pol_perturb_std_rad"])
    return config, seed


def _failed_rows_for_trial_task(task: tuple, error: BaseException) -> tuple[int, int, list[dict]]:
    setting_index, snr_db, t_dim, ris_shape, trial, options = task
    config, seed = _config_for_trial(
        setting_index=setting_index,
        snr_db=snr_db,
        t_dim=t_dim,
        ris_shape=ris_shape,
        trial=trial,
        options=options,
    )
    rows = [
        _failed_row(config, pipeline, trial, seed, error)
        for pipeline in (
            PIPELINE_STAGE1,
            PIPELINE_STAGE1_VP,
            PIPELINE_STAGE1_STAGE2,
            PIPELINE_STAGE1_STAGE2_VP,
        )
    ]
    return setting_index, trial, rows


def _run_trial_task(task: tuple) -> tuple[int, int, list[dict]]:
    setting_index, snr_db, t_dim, ris_shape, trial, options = task
    try:
        config, seed = _config_for_trial(
            setting_index=setting_index,
            snr_db=snr_db,
            t_dim=t_dim,
            ris_shape=ris_shape,
            trial=trial,
            options=options,
        )
        return setting_index, trial, _run_trial(config, trial, seed)
    except Exception as exc:  # noqa: BLE001 - worker setup failures should be recorded.
        return _failed_rows_for_trial_task(task, exc)


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        if row.get("failed"):
            continue
        value = _as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _stats(rows: list[dict], key: str) -> tuple[float | str, float | str, float | str]:
    values = _finite_values(rows, key)
    if not values:
        return "", "", ""
    array = np.asarray(values, dtype=float)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return float(np.mean(array)), std, float(np.median(array))


def _percentile(rows: list[dict], key: str, percentile: float) -> float | str:
    values = _finite_values(rows, key)
    if not values:
        return ""
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _rate(rows: list[dict], key: str) -> float | str:
    values = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        values.append(bool(value))
    if not values:
        return ""
    return float(np.mean(values))


def _paired_improvement_metrics(rows: list[dict]) -> dict[tuple[Any, ...], dict[str, float | str]]:
    paired: dict[tuple[Any, ...], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (
            row.get("init_mode", "clean"),
            row.get("init_mode_effective", row.get("init_mode", "clean")),
            row.get("init_warning", ""),
            row["SNR_dB"],
            row["T"],
            row["ris_shape"],
            row["trial"],
            row["seed"],
        )
        paired[key][row["pipeline"]] = row

    by_setting: dict[tuple[Any, ...], dict[str, list]] = defaultdict(
        lambda: {
            "position_improved": [],
            "y_nmse_improved": [],
            "direct_success20": [],
            "staged_success20": [],
            "direct_success50": [],
            "staged_success50": [],
        }
    )
    for key, pipeline_rows in paired.items():
        direct = pipeline_rows.get(PIPELINE_STAGE1_VP)
        staged = pipeline_rows.get(PIPELINE_STAGE1_STAGE2_VP)
        if direct is None or staged is None:
            continue

        setting_key = key[:6]
        values = by_setting[setting_key]
        direct_pos = _as_float(direct.get("position_error"))
        staged_pos = _as_float(staged.get("position_error"))
        if direct_pos is not None and staged_pos is not None:
            values["position_improved"].append(staged_pos < direct_pos)

        direct_y = _as_float(direct.get("Y_NMSE"))
        staged_y = _as_float(staged.get("Y_NMSE"))
        if direct_y is not None and staged_y is not None:
            values["y_nmse_improved"].append(staged_y < direct_y)

        values["direct_success20"].append(bool(direct.get("success_pos_20cm")))
        values["staged_success20"].append(bool(staged.get("success_pos_20cm")))
        values["direct_success50"].append(bool(direct.get("success_pos_50cm")))
        values["staged_success50"].append(bool(staged.get("success_pos_50cm")))

    metrics = {}
    for setting_key, values in by_setting.items():
        pos_values = values["position_improved"]
        y_values = values["y_nmse_improved"]
        direct20 = values["direct_success20"]
        staged20 = values["staged_success20"]
        direct50 = values["direct_success50"]
        staged50 = values["staged_success50"]
        metrics[setting_key] = {
            "improvement_rate_position": ""
            if not pos_values
            else float(np.mean(pos_values)),
            "improvement_rate_Y_NMSE": "" if not y_values else float(np.mean(y_values)),
            "VP_success_gain_20cm": ""
            if not direct20
            else float(np.mean(staged20) - np.mean(direct20)),
            "VP_success_gain_50cm": ""
            if not direct50
            else float(np.mean(staged50) - np.mean(direct50)),
        }
    return metrics


def _summarize(rows: list[dict]) -> list[dict]:
    paired_metrics = _paired_improvement_metrics(rows)
    grouped: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("init_mode", "clean"),
            row.get("init_mode_effective", row.get("init_mode", "clean")),
            row.get("init_warning", ""),
            row["SNR_dB"],
            row["T"],
            row["ris_shape"],
            row["pipeline"],
        )
        grouped[key].append(row)

    summary_rows = []
    for key in sorted(
        grouped,
        key=lambda item: (item[0], item[1], float(item[3]), int(item[4]), item[5], item[6]),
    ):
        group_rows = grouped[key]
        y_mean, y_std, y_median = _stats(group_rows, "Y_NMSE")
        pos_mean, pos_std, pos_median = _stats(group_rows, "position_error")
        range_mean, range_std, range_median = _stats(group_rows, "range_RMSE")
        delay_mean, delay_std, delay_median = _stats(group_rows, "delay_RMSE")
        delay_geo_mean, delay_geo_std, delay_geo_median = _stats(
            group_rows, "delay_geometry_consistency_RMSE"
        )
        ris_mean, ris_std, ris_median = _stats(
            group_rows, "compressed_ris_manifold_residual"
        )
        evs_mean, evs_std, evs_median = _stats(
            group_rows, "evs_maxwell_consistency_residual"
        )
        pole_mean, pole_std, pole_median = _stats(
            group_rows, "common_delay_pole_consistency"
        )
        failed_count = sum(bool(row.get("failed")) for row in group_rows)
        summary_rows.append(
            {
                "init_mode": key[0],
                "init_mode_effective": key[1],
                "init_warning": key[2],
                "SNR_dB": key[3],
                "T": key[4],
                "ris_shape": key[5],
                "pipeline": key[6],
                "num_rows": len(group_rows),
                "num_failed": failed_count,
                "failure_rate": float(failed_count / max(len(group_rows), 1)),
                "Y_NMSE_mean": y_mean,
                "Y_NMSE_std": y_std,
                "Y_NMSE_median": y_median,
                "position_error_mean": pos_mean,
                "position_error_std": pos_std,
                "position_error_median": pos_median,
                "position_error_p90": _percentile(group_rows, "position_error", 90.0),
                "position_error_p95": _percentile(group_rows, "position_error", 95.0),
                "range_RMSE_mean": range_mean,
                "range_RMSE_std": range_std,
                "range_RMSE_median": range_median,
                "delay_RMSE_mean": delay_mean,
                "delay_RMSE_std": delay_std,
                "delay_RMSE_median": delay_median,
                "delay_geometry_consistency_RMSE_mean": delay_geo_mean,
                "delay_geometry_consistency_RMSE_std": delay_geo_std,
                "delay_geometry_consistency_RMSE_median": delay_geo_median,
                "compressed_ris_manifold_residual_mean": ris_mean,
                "compressed_ris_manifold_residual_std": ris_std,
                "compressed_ris_manifold_residual_median": ris_median,
                "evs_maxwell_consistency_residual_mean": evs_mean,
                "evs_maxwell_consistency_residual_std": evs_std,
                "evs_maxwell_consistency_residual_median": evs_median,
                "common_delay_pole_consistency_mean": pole_mean,
                "common_delay_pole_consistency_std": pole_std,
                "common_delay_pole_consistency_median": pole_median,
                "success_pos_10cm_rate": _rate(group_rows, "success_pos_10cm"),
                "success_pos_20cm_rate": _rate(group_rows, "success_pos_20cm"),
                "success_pos_50cm_rate": _rate(group_rows, "success_pos_50cm"),
                "success_nmse_1e3_rate": _rate(group_rows, "success_nmse_1e3"),
                "success_nmse_1e2_rate": _rate(group_rows, "success_nmse_1e2"),
                "vp_convergence_rate": _rate(group_rows, "vp_converged"),
                "vp_worsened_noisy_rate": _rate(group_rows, "vp_worsened_noisy_objective"),
                "vp_worsened_true_nmse_rate": _rate(group_rows, "vp_worsened_true_nmse"),
                "fail_convergence_rate": _rate(group_rows, "fail_convergence"),
                "fail_position_50cm_rate": _rate(group_rows, "fail_position_50cm"),
                "fail_position_1m_rate": _rate(group_rows, "fail_position_1m"),
                "fail_nmse_1e2_rate": _rate(group_rows, "fail_nmse_1e2"),
                "fail_noisy_objective_increase_rate": _rate(
                    group_rows, "fail_noisy_objective_increase"
                ),
                "stage2_worsened_noisy_z_rate": _rate(
                    group_rows, "stage2_worsened_noisy_z_residual"
                ),
                "stage2_worsened_true_z_nmse_rate": _rate(
                    group_rows, "stage2_worsened_true_z_nmse"
                ),
                **paired_metrics.get(key[:6], {}),
            }
        )
    return summary_rows


def _write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, precision: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{precision}e}"


def _fmt_rate(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{100.0 * number:.1f}%"


def _print_primary_comparison(summary_rows: list[dict]) -> None:
    lookup = {
        (row["init_mode"], row["SNR_dB"], row["T"], row["ris_shape"], row["pipeline"]): row
        for row in summary_rows
    }
    settings = sorted(
        {(row["init_mode"], row["SNR_dB"], row["T"], row["ris_shape"]) for row in summary_rows},
        key=lambda item: (item[0], float(item[1]), int(item[2]), item[3]),
    )
    print("\nPrimary comparison: Stage-I + VP vs Stage-I + Stage-II + VP")
    for init_mode, snr_db, t_dim, ris_shape in settings:
        direct = lookup.get((init_mode, snr_db, t_dim, ris_shape, PIPELINE_STAGE1_VP))
        staged = lookup.get((init_mode, snr_db, t_dim, ris_shape, PIPELINE_STAGE1_STAGE2_VP))
        if direct is None or staged is None:
            continue
        print(
            f"init={init_mode}, SNR={snr_db:g} dB, T={t_dim}, RIS={ris_shape}: "
            f"Y_NMSE median {_fmt(direct['Y_NMSE_median'])} -> {_fmt(staged['Y_NMSE_median'])}, "
            f"pos median {_fmt(direct['position_error_median'])} -> "
            f"{_fmt(staged['position_error_median'])} m, "
            f"pos p90 {_fmt(direct['position_error_p90'])} -> "
            f"{_fmt(staged['position_error_p90'])} m, "
            f"pos p95 {_fmt(direct['position_error_p95'])} -> "
            f"{_fmt(staged['position_error_p95'])} m, "
            f"success20 {_fmt_rate(direct['success_pos_20cm_rate'])} -> "
            f"{_fmt_rate(staged['success_pos_20cm_rate'])}, "
            f"VP conv {_fmt_rate(direct['vp_convergence_rate'])} -> "
            f"{_fmt_rate(staged['vp_convergence_rate'])}, "
            f"fail>50cm {_fmt_rate(direct['fail_position_50cm_rate'])} -> "
            f"{_fmt_rate(staged['fail_position_50cm_rate'])}, "
            f"fail>1m {_fmt_rate(direct['fail_position_1m_rate'])} -> "
            f"{_fmt_rate(staged['fail_position_1m_rate'])}, "
            f"improve(pos) {_fmt_rate(staged.get('improvement_rate_position'))}, "
            f"gain20 {_fmt_rate(staged.get('VP_success_gain_20cm'))}, "
            f"gain50 {_fmt_rate(staged.get('VP_success_gain_50cm'))}, "
            f"VP worse(noisy) {_fmt_rate(direct['vp_worsened_noisy_rate'])} -> "
            f"{_fmt_rate(staged['vp_worsened_noisy_rate'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-II HP-R1P-CPD as an initializer for raw-domain VP-WNLS."
    )
    parser.add_argument("--mc", type=int, default=50, help="Monte Carlo trials per grid point.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes for Monte Carlo trials.",
    )
    parser.add_argument("--seed-base", type=int, default=2026052800)
    parser.add_argument(
        "--init-mode",
        choices=INIT_MODE_CHOICES,
        default="weak",
        help="Initialization mode before optional Stage-II and VP refinement.",
    )
    parser.add_argument(
        "--range-perturb-std",
        type=float,
        default=0.5,
        help="Range perturbation standard deviation in meters for --init-mode perturbed.",
    )
    parser.add_argument(
        "--angle-perturb-std",
        type=float,
        default=5.0,
        help="RIS elevation/azimuth perturbation standard deviation in degrees.",
    )
    parser.add_argument(
        "--delay-perturb-std",
        type=float,
        default=0.5,
        help="Delay perturbation standard deviation in ns for --init-mode perturbed.",
    )
    parser.add_argument(
        "--pol-perturb-std",
        type=float,
        default=5.0,
        help="Polarization angle perturbation standard deviation in degrees.",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=[-10.0, -5.0, 0.0, 5.0, 10.0, 20.0],
        help="SNR grid in dB.",
    )
    parser.add_argument(
        "--t-values",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="RIS training-length grid.",
    )
    parser.add_argument(
        "--ris-shapes",
        type=_parse_shape,
        nargs="+",
        default=[(8, 8), (16, 16), (32, 32)],
        help="RIS shape grid, e.g. 8x8 16x16 32x32.",
    )
    parser.add_argument("--k", type=int, default=2, help="Number of paths/RIS panels.")
    parser.add_argument("--n", type=int, default=24, help="Number of OFDM tones.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/stage2_init_value_weak"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mc <= 0:
        raise ValueError("--mc must be positive")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")
    for name in (
        "range_perturb_std",
        "angle_perturb_std",
        "delay_perturb_std",
        "pol_perturb_std",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.k != 2:
        raise ValueError("This experiment currently assumes K=2, as requested.")
    if args.n != 24:
        raise ValueError("This experiment currently assumes N=24, as requested.")

    grid = list(itertools.product(args.snr_db, args.t_values, args.ris_shapes))
    options = _trial_options_from_args(args)
    trial_tasks = []
    total_trials = len(grid) * args.mc
    for setting_index, (snr_db, t_dim, ris_shape) in enumerate(grid):
        print(
            f"Running init={args.init_mode}, SNR={snr_db:g} dB, "
            f"T={t_dim}, RIS={_shape_label(ris_shape)} "
            f"({setting_index + 1}/{len(grid)})"
        )
        for trial in range(args.mc):
            trial_tasks.append((setting_index, snr_db, t_dim, ris_shape, trial, options))

    results: list[tuple[int, int, list[dict]]] = []
    completed_trials = 0
    progress_interval = max(1, min(args.mc, 10))

    if args.n_jobs == 1:
        for task in trial_tasks:
            results.append(_run_trial_task(task))
            completed_trials += 1
            if completed_trials % progress_interval == 0:
                print(f"  completed {completed_trials}/{total_trials} trials")
    else:
        print(f"Using {args.n_jobs} worker processes for {total_trials} trials")
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {executor.submit(_run_trial_task, task): task for task in trial_tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - record hard worker failures.
                    results.append(_failed_rows_for_trial_task(task, exc))
                completed_trials += 1
                if completed_trials % progress_interval == 0:
                    print(f"  completed {completed_trials}/{total_trials} trials")

    results.sort(key=lambda item: (item[0], item[1]))
    rows = [row for _, _, trial_rows in results for row in trial_rows]

    summary_rows = _summarize(rows)
    raw_path = args.output_dir / "raw_results.csv"
    summary_path = args.output_dir / "summary.csv"
    _write_csv(raw_path, FIELDNAMES, rows)
    _write_csv(summary_path, SUMMARY_FIELDNAMES, summary_rows)
    print(f"\nWrote {raw_path}")
    print(f"Wrote {summary_path}")
    _print_primary_comparison(summary_rows)


if __name__ == "__main__":
    main()
