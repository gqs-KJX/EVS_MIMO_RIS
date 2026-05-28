"""Quick sanity check for Stage-II as VP-WNLS initialization.

This is intentionally small and prints per-trial diagnostics instead of writing
CSV files. True parameters are used only for data generation and metrics; all
estimator calls receive sanitized scene/config dictionaries and noisy data.
"""

from __future__ import annotations

import copy
import pathlib
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.estimators import (
        initialize_from_hankel,
        refine_global_raw,
        structured_refinement,
    )
    from src.experiments.exp_stage2_as_vp_initialization import (
        PIPELINE_STAGE1,
        PIPELINE_STAGE1_STAGE2,
        PIPELINE_STAGE1_STAGE2_VP,
        PIPELINE_STAGE1_VP,
        _add_stage2_comparison_flags,
        _assert_blind_estimation_inputs,
        _make_config,
        _make_trial_data,
        _relative_sse,
        _structured_metrics_row,
        _vp_initial_reference_row,
        _vp_metrics_row,
    )
    from src.metrics import relative_nmse
    from src.projections_delay import bq_from_poles
    from src.projections_ris import local_ris_search_config
    from src.tensor_utils import reconstruct_z, z_design_column
    from src.utils import solve_lstsq
else:
    from ..estimators import initialize_from_hankel, refine_global_raw, structured_refinement
    from ..metrics import relative_nmse
    from ..projections_delay import bq_from_poles
    from ..projections_ris import local_ris_search_config
    from ..tensor_utils import reconstruct_z, z_design_column
    from ..utils import solve_lstsq
    from .exp_stage2_as_vp_initialization import (
        PIPELINE_STAGE1,
        PIPELINE_STAGE1_STAGE2,
        PIPELINE_STAGE1_STAGE2_VP,
        PIPELINE_STAGE1_VP,
        _add_stage2_comparison_flags,
        _assert_blind_estimation_inputs,
        _make_config,
        _make_trial_data,
        _relative_sse,
        _structured_metrics_row,
        _vp_initial_reference_row,
        _vp_metrics_row,
    )


MC = 3
SNR_DB = 0.0
T_DIM = 64
RIS_SHAPE = (16, 16)
K_PATHS = 2
N_DIM = 24
SEED_BASE = 2026052900
INIT_MODES = ("clean", "perturbed")

METRIC_FIELDS = [
    "Y_NMSE",
    "Z_NMSE",
    "raw_noisy_residual",
    "Z_noisy_residual",
    "position_error",
    "range_RMSE",
    "delay_RMSE",
    "polarization_angle_RMSE",
]


def _complex_perturb_columns(
    matrix: np.ndarray, relative_scale: float, rng: np.random.Generator, eps: float
) -> np.ndarray:
    perturbed = matrix.copy()
    for k in range(matrix.shape[1]):
        norm = np.linalg.norm(matrix[:, k])
        noise_scale = relative_scale * norm / np.sqrt(matrix.shape[0])
        noise = noise_scale * (
            rng.standard_normal(matrix.shape[0]) + 1j * rng.standard_normal(matrix.shape[0])
        ) / np.sqrt(2.0)
        column = matrix[:, k] + noise
        column_norm = np.linalg.norm(column)
        if norm > eps and column_norm > eps:
            column = column * (norm / column_norm)
        perturbed[:, k] = column
    return perturbed


def _refit_z_weights(estimate: dict, data: dict) -> dict:
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


def _perturb_initial_estimate(estimate: dict, data: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    scene = data["scene_est"]
    config = data["config_est"]
    eps = config["eps"]
    perturbed = copy.deepcopy(estimate)

    perturbed["A"] = _complex_perturb_columns(perturbed["A"], 0.06, rng, eps)
    perturbed["C"] = _complex_perturb_columns(perturbed["C"], 0.06, rng, eps)

    phase_jitter = rng.normal(0.0, 0.08, size=scene["K"])
    perturbed["poles"] = perturbed["poles"] * np.exp(1j * phase_jitter)
    perturbed["poles"] = perturbed["poles"] / np.maximum(np.abs(perturbed["poles"]), eps)

    perturbed["gamma"] = np.clip(
        perturbed["gamma"] + rng.normal(0.0, 0.06, size=scene["K"]), 0.05, 1.50
    )
    perturbed["eta_pol"] = np.angle(
        np.exp(1j * (perturbed["eta_pol"] + rng.normal(0.0, 0.12, size=scene["K"])))
    )

    for k in range(scene["K"]):
        search = local_ris_search_config(scene, config, k)
        lower = np.array(
            [
                search["range_bounds"][0],
                search["elev_bounds"][0],
                search["az_bounds"][0],
            ]
        )
        upper = np.array(
            [
                search["range_bounds"][1],
                search["elev_bounds"][1],
                search["az_bounds"][1],
            ]
        )
        jitter = np.array(
            [
                rng.normal(0.0, 0.25),
                rng.normal(0.0, 0.04),
                rng.normal(0.0, 0.08),
            ]
        )
        perturbed["ris_eta"][k] = np.clip(perturbed["ris_eta"][k] + jitter, lower, upper)

    return _refit_z_weights(perturbed, data)


def _float(value: Any) -> float:
    return float(value)


def _assert_finite_row(row: dict) -> None:
    for field in METRIC_FIELDS:
        value = _float(row[field])
        if not np.isfinite(value):
            raise AssertionError(f"{row['pipeline']} has nonfinite {field}: {value}")
    if "VP" in row["pipeline"]:
        value = _float(row["vp_final_noisy_objective"])
        if not np.isfinite(value):
            raise AssertionError(f"{row['pipeline']} has nonfinite VP final objective: {value}")


def _assert_intended_vp_initialization(row: dict, expected_init_row: dict) -> None:
    expected_noisy = _float(expected_init_row["raw_noisy_residual"])
    expected_true = _float(expected_init_row["Y_NMSE"])
    actual_noisy = _float(row["vp_initial_noisy_objective"])
    actual_true = _float(row["vp_initial_Y_NMSE"])
    if not np.isclose(actual_noisy, expected_noisy, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            f"{row['pipeline']} did not record the intended VP noisy initialization"
        )
    if not np.isclose(actual_true, expected_true, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            f"{row['pipeline']} did not record the intended VP true-domain initialization"
        )


def _assert_no_true_observation_passed(data: dict, *, y_arg=None, z_arg=None) -> None:
    if y_arg is data["Y_true"]:
        raise AssertionError("Y_true was passed to an estimator")
    if z_arg is data["Z_true"]:
        raise AssertionError("Z_true was passed to an estimator")
    if y_arg is not None and y_arg is not data["Y_noisy"]:
        raise AssertionError("VP estimator did not receive the expected Y_noisy array")
    if z_arg is not None and z_arg is not data["Z_noisy"]:
        raise AssertionError("Stage-I/Stage-II estimator did not receive the expected Z_noisy array")


def _format_row(row: dict) -> str:
    parts = [
        f"Y_NMSE={_float(row['Y_NMSE']):.3e}",
        f"Z_NMSE={_float(row['Z_NMSE']):.3e}",
        f"raw_noisy={_float(row['raw_noisy_residual']):.3e}",
        f"Z_noisy={_float(row['Z_noisy_residual']):.3e}",
        f"pos={_float(row['position_error']):.3e}m",
        f"range={_float(row['range_RMSE']):.3e}m",
        f"delay={_float(row['delay_RMSE']):.3e}s",
        f"pol={_float(row['polarization_angle_RMSE']):.3e}rad",
    ]
    if "VP" in row["pipeline"]:
        parts.extend(
            [
                f"vp_success={row['vp_converged']}",
                f"vp_nfev={row['vp_nfev']}",
                f"vp_obj={_float(row['vp_final_noisy_objective']):.3e}",
                f"vp_init_obj={_float(row['vp_initial_noisy_objective']):.3e}",
                f"vp_worse_noisy={row['vp_worsened_noisy_objective']}",
                f"vp_worse_true={row['vp_worsened_true_nmse']}",
            ]
        )
    if row["pipeline"] in (PIPELINE_STAGE1_STAGE2, PIPELINE_STAGE1_STAGE2_VP):
        parts.extend(
            [
                f"stage2_worse_noisy_z={row['stage2_worsened_noisy_z_residual']}",
                f"stage2_worse_true_z={row['stage2_worsened_true_z_nmse']}",
            ]
        )
    return " | ".join(parts)


def _run_one_mode(config: dict, trial: int, init_mode: str) -> list[dict]:
    data = _make_trial_data(config)
    _assert_blind_estimation_inputs(data)

    noise_nmse = relative_nmse(data["Y_noisy"], data["Y_true"])
    if not np.isfinite(noise_nmse) or abs(noise_nmse - 1.0) > 0.2:
        raise AssertionError(f"NMSE_Y_noisy={noise_nmse:.6e} is not within 0.2 of 1 at 0 dB")

    _assert_no_true_observation_passed(data, z_arg=data["Z_noisy"])
    estimate_initial = initialize_from_hankel(data["Z_noisy"], data["scene_est"], data["config_est"])
    if init_mode == "clean":
        estimate_start = copy.deepcopy(estimate_initial)
    elif init_mode == "perturbed":
        estimate_start = _perturb_initial_estimate(
            estimate_initial, data, seed=config["seed"] + 71_003
        )
    else:
        raise ValueError(f"unknown init_mode {init_mode!r}")

    rows = []
    stage1_row = _structured_metrics_row(
        config=config,
        pipeline=PIPELINE_STAGE1,
        trial=trial,
        seed=config["seed"],
        estimate=estimate_start,
        data=data,
    )
    rows.append(stage1_row)

    direct_vp_init = _vp_initial_reference_row(estimate_start, data)
    _assert_no_true_observation_passed(data, y_arg=data["Y_noisy"])
    direct_vp = refine_global_raw(
        data["Y_noisy"], data["scene_est"], data["config_est"], copy.deepcopy(estimate_start)
    )
    direct_vp_row = _vp_metrics_row(
        config=config,
        pipeline=PIPELINE_STAGE1_VP,
        trial=trial,
        seed=config["seed"],
        final=direct_vp,
        init_row=direct_vp_init,
        data=data,
    )
    _assert_intended_vp_initialization(direct_vp_row, direct_vp_init)
    rows.append(direct_vp_row)

    _assert_no_true_observation_passed(data, z_arg=data["Z_noisy"])
    estimate_stage2, _ = structured_refinement(
        data["Z_noisy"], data["scene_est"], data["config_est"], copy.deepcopy(estimate_start)
    )
    stage2_row = _structured_metrics_row(
        config=config,
        pipeline=PIPELINE_STAGE1_STAGE2,
        trial=trial,
        seed=config["seed"],
        estimate=estimate_stage2,
        data=data,
    )
    _add_stage2_comparison_flags(stage2_row, stage1_row, stage2_row)
    rows.append(stage2_row)

    stage2_vp_init = _vp_initial_reference_row(estimate_stage2, data)
    _assert_no_true_observation_passed(data, y_arg=data["Y_noisy"])
    stage2_vp = refine_global_raw(
        data["Y_noisy"], data["scene_est"], data["config_est"], copy.deepcopy(estimate_stage2)
    )
    stage2_vp_row = _vp_metrics_row(
        config=config,
        pipeline=PIPELINE_STAGE1_STAGE2_VP,
        trial=trial,
        seed=config["seed"],
        final=stage2_vp,
        init_row=stage2_vp_init,
        data=data,
    )
    _add_stage2_comparison_flags(stage2_vp_row, stage1_row, stage2_row)
    _assert_intended_vp_initialization(stage2_vp_row, stage2_vp_init)
    rows.append(stage2_vp_row)

    for row in rows:
        _assert_finite_row(row)
        if row.get("failed"):
            raise AssertionError(f"{row['pipeline']} unexpectedly failed")

    print(f"\ntrial={trial} init_mode={init_mode} seed={config['seed']}")
    print(f"NMSE_Y_noisy={noise_nmse:.6e}")
    for row in rows:
        print(f"  {row['pipeline']}: {_format_row(row)}")

    return rows


def main() -> None:
    all_rows = []
    for trial in range(MC):
        for init_index, init_mode in enumerate(INIT_MODES):
            seed = SEED_BASE + 10_000 * trial + init_index
            config = _make_config(
                snr_db=SNR_DB,
                t_dim=T_DIM,
                ris_shape=RIS_SHAPE,
                seed=seed,
                k_paths=K_PATHS,
                n_dim=N_DIM,
            )
            all_rows.extend(_run_one_mode(config, trial, init_mode))

    print(f"\nSanity check passed: {len(all_rows)} pipeline rows executed.")


if __name__ == "__main__":
    main()
