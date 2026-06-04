"""Run one diagnostic proposed RIS-EVS-OFDM estimation demo."""

from __future__ import annotations

import argparse
import copy
import itertools
import pathlib
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ""):
    project_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from src.channel_model import (
        add_awgn,
        channel_components,
        generate_scene,
        synthesize_raw_tensor,
    )
    from src.config import default_config
    from src.diagnostics import (
        format_float_list,
        hankel_metric_summary,
        noise_metric_summary,
        estimate_position_from_ris_eta,
        parameter_errors_for_structured,
        parameter_errors_for_vp,
        run_delay_projection_self_test,
        run_ris_projection_self_test,
        run_tensor_factorization_shape_self_test,
        y_metric_summary,
        z_metric_summary,
    )
    from src.estimators import (
        global_exact_spherical_vp_refinement,
        initialize_from_hankel,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from src.metrics import position_rmse, relative_nmse, rmse_abs
    from src.geometry import polarization_vector
    from src.projections_delay import tau_from_pole
    from src.projections_ris import compressed_exact_response, scaled_residual
    from src.tensor_utils import hankelize_frequency
    from src.utils import scipy_is_available
else:
    from .channel_model import (
        add_awgn,
        channel_components,
        generate_scene,
        synthesize_raw_tensor,
    )
    from .config import default_config
    from .diagnostics import (
        format_float_list,
        hankel_metric_summary,
        noise_metric_summary,
        estimate_position_from_ris_eta,
        parameter_errors_for_structured,
        parameter_errors_for_vp,
        run_delay_projection_self_test,
        run_ris_projection_self_test,
        run_tensor_factorization_shape_self_test,
        y_metric_summary,
        z_metric_summary,
    )
    from .estimators import (
        global_exact_spherical_vp_refinement,
        initialize_from_hankel,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from .metrics import position_rmse, relative_nmse, rmse_abs
    from .geometry import polarization_vector
    from .projections_delay import tau_from_pole
    from .projections_ris import compressed_exact_response, scaled_residual
    from .tensor_utils import hankelize_frequency
    from .utils import scipy_is_available


def _make_data(config: dict) -> dict:
    """Generate one reproducible synthetic channel and noisy observation."""
    data_start = time.perf_counter()
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
    data_generation_s = time.perf_counter() - data_start

    hankel_start = time.perf_counter()
    z_true = hankelize_frequency(y_true, scene["P"])
    z_noisy = hankelize_frequency(y_noisy, scene["P"])
    hankelization_s = time.perf_counter() - hankel_start

    assert y_true.shape == (scene["I"], scene["N"], scene["T"])
    assert y_noisy.shape == y_true.shape
    assert z_true.shape == (scene["I"], scene["P"], scene["L"], scene["T"])
    assert z_noisy.shape == z_true.shape

    return {
        "scene": scene,
        "true_components": true_components,
        "Y_true": y_true,
        "Y_noisy": y_noisy,
        "Z_true": z_true,
        "Z_noisy": z_noisy,
        "noise_variance": noise_variance,
        "timing": {
            "data_generation": data_generation_s,
            "hankelization": hankelization_s,
        },
    }


def _git_commit() -> str:
    """Return the current git commit hash when the script is run inside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _fmt(value, precision: int = 6) -> str:
    """Format scalars for compact diagnostic tables."""
    if value is None or value == "":
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return "NA"
    return f"{value_float:.{precision}e}"


def _fmt_fixed(value, precision: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return "NA"
    return f"{value_float:.{precision}f}"


def _fmt_vector(values: np.ndarray, scale: float = 1.0, precision: int = 4) -> str:
    arr = np.asarray(values).reshape(-1) * scale
    return "[" + ", ".join(_fmt_fixed(value, precision) for value in arr) + "]"


def _wrap_angle_rad(angle: float) -> float:
    return float(np.angle(np.exp(1j * angle)))


def _relative_complex_residual(target: np.ndarray, model: np.ndarray, eps: float) -> float:
    scale = np.vdot(model, target) / (np.vdot(model, model) + eps)
    return float(np.linalg.norm(target - scale * model) / (np.linalg.norm(target) + eps))


def _evs_model(scene: dict, path: int, gamma: float, eta_pol: float) -> np.ndarray:
    pol = scene["Theta"][path] @ polarization_vector(gamma, eta_pol)
    return np.kron(scene["v_B"][path], pol)


def _ris_local_residual(scene: dict, path: int, c_vec: np.ndarray, eta_local: np.ndarray) -> float:
    h_model = compressed_exact_response(
        eta_local,
        scene["Omega"][path],
        scene["a_RB"][path],
        scene["ris_grid"],
        scene["wavelength"],
    )
    value, _ = scaled_residual(c_vec, h_model, 1.0e-10)
    return float(np.sqrt(max(value, 0.0) / (np.linalg.norm(c_vec) ** 2 + 1.0e-10)))


def _evs_local_residual(
    scene: dict, path: int, a_vec: np.ndarray, gamma: float, eta_pol: float
) -> float:
    return _relative_complex_residual(
        a_vec, _evs_model(scene, path, gamma, eta_pol), 1.0e-10
    )


def _format_matrix(matrix: np.ndarray, precision: int = 4) -> str:
    arr = np.asarray(matrix, dtype=float)
    rows = []
    for row in arr:
        rows.append("[" + ", ".join(_fmt(value, precision) for value in row) + "]")
    return "[" + ", ".join(rows) + "]"


def _permutation_margin(
    costs: np.ndarray, orientation: str
) -> tuple[float, float, float] | None:
    arr = np.asarray(costs, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or not np.all(np.isfinite(arr)):
        return None
    scores = []
    k_paths = arr.shape[0]
    for perm in itertools.permutations(range(k_paths)):
        if orientation == "column_to_panel":
            score = sum(arr[col, perm[col]] for col in range(k_paths))
        elif orientation == "panel_to_column":
            score = sum(arr[perm[panel], panel] for panel in range(k_paths))
        else:
            raise ValueError(f"unknown assignment orientation {orientation!r}")
        scores.append(float(score))
    scores.sort()
    best = scores[0]
    second = scores[1] if len(scores) > 1 else float("inf")
    margin = (second - best) / max(abs(best), 1.0e-12)
    return best, second, float(margin)


def _inverse_assignment(column_to_panel: list[int]) -> list[int]:
    panel_to_column = [-1] * len(column_to_panel)
    for column, panel in enumerate(column_to_panel):
        panel_to_column[int(panel)] = int(column)
    return panel_to_column


def _print_ris_dimension_diagnostics(scene: dict, true_components: dict) -> None:
    """Print and assert RIS element-domain and compressed-domain dimensions."""
    print(f"M_Rx = {scene['M_Rx']}")
    print(f"M_Ry = {scene['M_Ry']}")
    print(f"M_R = {scene['M_R']}")
    for k in range(scene["K"]):
        g_k = true_components["g"][k]
        omega_k = scene["Omega"][k]
        c_k = true_components["c"][k]
        assert len(g_k) == scene["M_R"], "len(g_k) != M_R"
        assert omega_k.shape == (scene["T"], scene["M_R"]), "Omega_k shape mismatch"
        assert c_k.shape == (scene["T"],), "c_k shape mismatch"
        print(
            f"path {k}: len(g_k)={len(g_k)}, "
            f"Omega_k shape={omega_k.shape}, c_k shape={c_k.shape}"
        )


def _print_self_tests(scene: dict, config: dict, true_components: dict) -> None:
    """Run and print deterministic self-tests requested for diagnostics."""
    print("\n=== Self-tests ===")
    tensor_test = run_tensor_factorization_shape_self_test()
    print(f"tensor_unfolding_max_error = {tensor_test['max_mode_error']:.3e}")

    delay_test = run_delay_projection_self_test(scene["delta_f"])
    print(
        "delay_projection: "
        f"true_pole={delay_test['true_pole']:.6g}, "
        f"estimated_pole={delay_test['estimated_pole']:.6g}, "
        f"delay_error_s={delay_test['delay_error_s']:.3e}"
    )

    ris_test = run_ris_projection_self_test(scene, config, true_components)
    print(
        "ris_projection: "
        f"Phi_before={ris_test['phi_before']:.6e}, "
        f"Phi_after={ris_test['phi_after']:.6e}, "
        f"range_error={ris_test['range_error']:.3e}, "
        f"angle_error={ris_test['angle_error']:.3e}, "
        f"pinv_used={ris_test['used_pinv']}"
    )
    if ris_test["warning"]:
        print(f"WARNING OBJECTIVE_MISMATCH: {ris_test['warning']}")


def _weak_reasonable_stage1_config(config: dict) -> dict:
    """Return a moderately weakened Stage-I RIS search for main_single_proposed."""
    weak_config = copy.deepcopy(config)
    ris_search = dict(weak_config["ris_search"])
    ris_search["num_range"] = min(int(ris_search.get("num_range", 15)), 9)
    ris_search["num_elev"] = min(int(ris_search.get("num_elev", 9)), 5)
    ris_search["num_az"] = min(int(ris_search.get("num_az", 25)), 13)
    ris_search["num_exact_refine_starts"] = min(
        int(ris_search.get("num_exact_refine_starts", 6)), 3
    )
    ris_search["num_lift_candidates"] = min(int(ris_search.get("num_lift_candidates", 4)), 3)
    ris_search["num_lift_steps"] = min(int(ris_search.get("num_lift_steps", 4)), 3)
    weak_config["ris_search"] = ris_search
    return weak_config


def _print_stage1_initialization_diagnostics(results: dict) -> None:
    """Print the Stage-I initialization mode and RIS search strength."""
    ris_search = results["stage1_initialization"]["ris_search"]
    print("stage1_init_mode = weak_reasonable")
    print(
        f"range/elev/az = ({ris_search['num_range']}, "
        f"{ris_search['num_elev']}, {ris_search['num_az']})"
    )
    print(f"exact_refine_starts = {ris_search['num_exact_refine_starts']}")
    print(f"lift_candidates = {ris_search['num_lift_candidates']}")
    print(f"lift_steps = {ris_search['num_lift_steps']}")


def _print_run_configuration(config: dict, results: dict) -> None:
    """Print the structured run configuration block."""
    scene = results["scene"]
    ris_search = results["stage1_initialization"]["ris_search"]
    final_method = str(
        config.get("final_refinement_method", "global_exact_spherical_vp")
    ).lower()
    global_vp_options = dict(config.get("global_vp", {}))
    global_vp_solver = str(global_vp_options.get("solver", "least_squares"))
    if not config.get("enable_global_vp", True) or final_method == "none":
        vp_solver_type = "skipped_global_vp"
        vp_backend = "skipped"
    elif (
        final_method == "global_exact_spherical_vp"
        and global_vp_solver == "least_squares"
        and scipy_is_available()
    ):
        vp_solver_type = "scipy.optimize.least_squares"
        vp_backend = "scipy.optimize"
    elif final_method == "global_exact_spherical_vp" and global_vp_solver == "least_squares":
        vp_solver_type = "bounded_coordinate_search"
        vp_backend = "fallback"
    elif (
        final_method == "global_exact_spherical_vp"
        and global_vp_solver == "lbfgsb_reduced"
        and scipy_is_available()
    ):
        vp_solver_type = "scipy.optimize.minimize:L-BFGS-B"
        vp_backend = "scipy.optimize"
    elif final_method == "global_exact_spherical_vp" and global_vp_solver == "lbfgsb_reduced":
        vp_solver_type = "bounded_coordinate_search"
        vp_backend = "fallback"
    elif final_method == "legacy_raw_vp" and scipy_is_available():
        vp_solver_type = "scipy.optimize.least_squares"
        vp_backend = "scipy.optimize"
    else:
        vp_solver_type = "bounded_coordinate_search"
        vp_backend = "fallback"

    print("=== Run configuration ===")
    print(f"seed = {config['seed']}")
    print(f"git_commit = {_git_commit()}")
    print(f"SNR_dB = {config['SNR_dB']:.1f}")
    print(f"fc_Hz = {config['fc']:.6e}")
    print(f"fc_GHz = {config['fc'] / 1.0e9:.3f}")
    print(f"delta_f_Hz = {config['delta_f']:.6e}")
    print(f"delta_f_MHz = {config['delta_f'] / 1.0e6:.3f}")
    print(
        f"K={scene['K']}, I={scene['I']}, N={scene['N']}, "
        f"P={scene['P']}, L={scene['L']}, T={scene['T']}"
    )
    print(
        f"RIS_shape = ({scene['M_Rx']}, {scene['M_Ry']}), "
        f"M_R = {scene['M_R']}"
    )
    print(f"stage1_init_mode = {results['stage1_initialization']['mode']}")
    print(
        "stage1_grid = "
        f"range={ris_search['num_range']}, "
        f"elev={ris_search['num_elev']}, az={ris_search['num_az']}"
    )
    print(f"exact_refine_starts = {ris_search['num_exact_refine_starts']}")
    print(f"lift_candidates = {ris_search['num_lift_candidates']}")
    print(f"lift_steps = {ris_search['num_lift_steps']}")
    print(
        "stage2_enabled_flags = "
        f"EVS={config.get('stage2_enable_evs', True)}, "
        f"delay={config.get('stage2_enable_delay', True)}, "
        f"RIS={config.get('stage2_enable_ris', True)}"
    )
    print(f"stage2_guarded = {config.get('stage2_guarded', False)}")
    print(f"stage2_mode = {config.get('stage2_mode', 'none')}")
    print(f"num_structured_iters = {config['num_structured_iters']}")
    print(f"enable_global_vp = {config.get('enable_global_vp', True)}")
    print(f"final_refinement_method = {config.get('final_refinement_method', 'global_exact_spherical_vp')}")
    print(f"global_vp_solver = {global_vp_solver}")
    print(f"vp_solver_type = {vp_solver_type}")
    print(f"vp_solver_backend = {vp_backend}")


def _print_assignment_diagnostics(results: dict) -> None:
    """Print Stage-I and Stage-II assignment mappings with correct orientations."""
    print("\n=== Assignment diagnostics ===")
    column_to_panel = [int(item) for item in results["estimate_initial"]["assignment"]]
    panel_to_column = _inverse_assignment(column_to_panel)
    print(f"column_to_panel_assignment = {column_to_panel}")
    print(f"panel_to_column_assignment = {panel_to_column}")
    costs = results["estimate_initial"].get("assignment_costs")
    if costs is not None:
        print(f"stage1_assignment_cost_matrix_col_by_panel = {_format_matrix(costs)}")
        margin = _permutation_margin(costs, "column_to_panel")
        if margin is not None:
            best, second, rel_margin = margin
            print(
                "stage1_assignment_margin = "
                f"best={best:.6e}, second={second:.6e}, relative={rel_margin:.6e}"
            )
            if rel_margin < 1.0e-3:
                print(
                    "WARNING ASSIGNMENT_AMBIGUOUS: Stage-I column-to-panel "
                    f"assignment has relative margin {rel_margin:.3e}."
                )

    for iter_idx, update in enumerate(results["structured_diag"]["updates"], start=1):
        mode4_panel_order = update.get("mode4_assignment_order")
        print(f"iter_{iter_idx}_mode4_panel_order = {mode4_panel_order}")
        assignment_costs = update.get("mode4_assignment_costs")
        if assignment_costs is not None:
            print(
                f"iter_{iter_idx}_mode4_assignment_cost_matrix_col_by_panel = "
                f"{_format_matrix(assignment_costs)}"
            )
            margin = _permutation_margin(assignment_costs, "panel_to_column")
            if margin is not None:
                best, second, rel_margin = margin
                print(
                    f"iter_{iter_idx}_mode4_assignment_margin = "
                    f"best={best:.6e}, second={second:.6e}, relative={rel_margin:.6e}"
                )
                if rel_margin < 1.0e-3:
                    print(
                        "WARNING ASSIGNMENT_AMBIGUOUS: Stage-II mode-4 "
                        f"panel order has relative margin {rel_margin:.3e}."
                    )


def _run_single_pipeline(config: dict, use_structured: bool) -> dict:
    """Run Stage-I, optional legacy Stage-II ablation, and final raw-domain VP."""
    total_start = time.perf_counter()
    data = _make_data(config)
    timing = dict(data.get("timing", {}))
    scene = data["scene"]
    stage1_config = _weak_reasonable_stage1_config(config)
    stage1_start = time.perf_counter()
    estimate_initial = initialize_from_hankel(data["Z_noisy"], scene, stage1_config)
    timing["stage1"] = time.perf_counter() - stage1_start
    requested_stage2_mode = str(config.get("stage2_mode", "none")).lower()
    stage2_mode = requested_stage2_mode if use_structured else "none"
    if stage2_mode == "full_legacy":
        stage2_start = time.perf_counter()
        estimate_used, structured_diag = structured_refinement(
            data["Z_noisy"], scene, config, copy.deepcopy(estimate_initial)
        )
        timing["stage2"] = time.perf_counter() - stage2_start
    elif stage2_mode == "ris_only":
        raise NotImplementedError("stage2_mode='ris_only' is not a standalone pipeline")
    elif stage2_mode == "none":
        estimate_used = copy.deepcopy(estimate_initial)
        structured_diag = {
            "z_hat_history": [],
            "residuals_noisy_rmse": [],
            "updates": [],
            "ris_projection_total_s": 0.0,
        }
        timing["stage2"] = 0.0
    else:
        raise ValueError(f"unknown stage2_mode {stage2_mode!r}")
    timing["ris_projection_total"] = float(
        structured_diag.get("ris_projection_total_s", 0.0)
    )
    final_method = str(
        config.get("final_refinement_method", "global_exact_spherical_vp")
    ).lower()
    if not config.get("enable_global_vp", True):
        final_method = "none"

    if final_method == "global_exact_spherical_vp":
        vp_start = time.perf_counter()
        final = global_exact_spherical_vp_refinement(
            data["Y_noisy"], estimate_used, scene, config
        )
        timing["vp"] = time.perf_counter() - vp_start
        final["vp_enabled"] = True
        final["stage2_mode"] = stage2_mode
        final["final_refinement_method"] = "global_exact_spherical_vp"
    elif final_method == "legacy_raw_vp":
        vp_start = time.perf_counter()
        final = refine_global_raw(data["Y_noisy"], scene, config, estimate_used)
        timing["vp"] = time.perf_counter() - vp_start
        final["vp_enabled"] = True
        final["stage2_mode"] = stage2_mode
        final["final_refinement_method"] = "legacy_raw_vp"
    elif final_method == "none":
        vp_start = time.perf_counter()
        y_hat = reconstruct_raw_tensor_from_structured_estimate(estimate_used, scene)
        raw_residual = y_hat - data["Y_noisy"]
        raw_objective = float(
            np.vdot(raw_residual.reshape(-1), raw_residual.reshape(-1)).real
            / data["Y_noisy"].size
        )
        tau_hat = np.array(
            [
                ((-np.angle(pole)) % (2.0 * np.pi))
                / (2.0 * np.pi * scene["delta_f"])
                for pole in estimate_used["poles"]
            ]
        )
        final = {
            "Y_hat": y_hat,
            "p_u": estimate_position_from_ris_eta(scene, estimate_used),
            "components": {
                "taus": tau_hat,
                "ranges": estimate_used["ris_eta"][:, 0],
            },
            "raw_residual_rmse_noisy": float(
                np.linalg.norm(y_hat - data["Y_noisy"]) / np.sqrt(data["Y_noisy"].size)
            ),
            "raw_objective_initial": raw_objective,
            "raw_objective_final": raw_objective,
            "optimizer": {
                "success": True,
                "message": "global VP disabled by config",
                "n_eval": 0,
                "method": "skipped_global_vp",
                "solver_backend": "skipped",
            },
            "vp_enabled": False,
            "stage2_mode": stage2_mode,
            "final_refinement_method": "none",
        }
        timing["vp"] = time.perf_counter() - vp_start
    else:
        raise ValueError(f"unknown final_refinement_method {final_method!r}")
    timing["total"] = time.perf_counter() - total_start
    return {
        **data,
        "estimate_initial": estimate_initial,
        "estimate_used": estimate_used,
        "structured_diag": structured_diag,
        "stage1_initialization": {
            "mode": "weak_reasonable",
            "ris_search": dict(stage1_config["ris_search"]),
        },
        "final": final,
        "timing": timing,
    }


def _print_global_vp_diagnostics(results: dict) -> None:
    """Print the compact final-refinement diagnostics requested for the demo."""
    scene = results["scene"]
    final = results["final"]
    y_noisy = results["Y_noisy"]
    stage1_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_initial"], scene
    )
    stage1_residual = float(
        np.linalg.norm(stage1_y_hat - y_noisy) / np.sqrt(y_noisy.size)
    )
    timing = results.get("timing", {})

    print("\n=== Global VP diagnostics ===")
    print(f"stage1_raw_residual_rmse_noisy = {stage1_residual:.6e}")
    print(f"global_vp_initial_residual = {_fmt(final.get('raw_residual_initial'))}")
    print(f"global_vp_final_residual = {_fmt(final.get('raw_residual_final'))}")
    print(f"global_vp_raw_objective = {_fmt(final.get('raw_objective'))}")
    print(f"global_vp_delay_prior_objective = {_fmt(final.get('delay_prior_objective'))}")
    print(f"global_vp_total_objective = {_fmt(final.get('total_objective'))}")
    print(f"global_vp_solver = {final.get('global_vp_solver', 'unknown')}")
    print(f"global_vp_evs_mode = {final.get('global_vp_evs_mode', 'unknown')}")
    print(f"global_vp_use_delay_prior = {final.get('global_vp_use_delay_prior', 'NA')}")
    print(f"global_vp_trust_region_used = {final.get('global_vp_trust_region_used', 'NA')}")
    print(
        "global_vp_columns_are_panel_ordered = "
        f"{final.get('global_vp_columns_are_panel_ordered', 'NA')}"
    )
    print(
        "global_vp_used_panel_to_column = "
        f"{final.get('global_vp_used_panel_to_column', 'NA')}"
    )
    print(f"global_vp_panel_to_column = {final.get('global_vp_panel_to_column', 'NA')}")
    print(f"tau_stage1_ns = {_fmt_vector(final.get('tau_stage1', []), scale=1e9)}")
    print(
        "tau_after_global_vp_ns = "
        f"{_fmt_vector(final.get('tau_after_global_vp', []), scale=1e9)}"
    )
    print(f"global_vp_init_method = {final.get('global_vp_init_method', 'unknown')}")
    print(
        "global_vp_init_selected_candidate = "
        f"{final.get('global_vp_init_selected_candidate', 'unknown')}"
    )
    print(f"estimated_p_u_m = {_fmt_vector(final.get('p_u', []), precision=5)}")
    delta_t = final.get("delta_t")
    delta_t_ns = None if delta_t is None else float(delta_t) * 1.0e9
    print(f"estimated_Delta_t_ns = {_fmt(delta_t_ns, precision=6)}")
    print(f"stage1_runtime_s = {_fmt(timing.get('stage1'))}")
    print(f"legacy_stage2_runtime_s = {_fmt(timing.get('stage2'))}")
    print(f"global_vp_runtime_s = {_fmt(timing.get('vp'))}")
    print(f"total_runtime_s = {_fmt(timing.get('total'))}")


def _print_noise_and_y_metrics(results: dict, direct_results: dict, snr_db: float) -> dict:
    """Print noise and raw-domain metrics for default diagnostics."""
    scene = results["scene"]
    y_true = results["Y_true"]
    y_noisy = results["Y_noisy"]
    vp_enabled = bool(results["final"].get("vp_enabled", True))
    initial_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_initial"], scene
    )
    structured_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_used"], scene
    )
    final_y_hat = results["final"]["Y_hat"]

    noise_metrics = noise_metric_summary(y_true, y_noisy, snr_db)
    print("\n=== Noise and Y-domain metrics ===")
    for key in (
        "norm_Y_true",
        "norm_noise",
        "signal_power_Y",
        "noise_power_Y",
        "target_SNR_dB",
        "empirical_SNR_dB",
        "RMSE_Y_noisy_abs",
        "NMSE_Y_noisy",
    ):
        print(f"{key} = {noise_metrics[key]:.6e}")

    initial_metrics = y_metric_summary(initial_y_hat, y_true)
    structured_metrics = y_metric_summary(structured_y_hat, y_true)
    final_metrics = y_metric_summary(final_y_hat, y_true)
    direct_final_metrics = y_metric_summary(direct_results["final"]["Y_hat"], y_true)

    print(f"RMSE_Y_hat_initial_abs = {initial_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat_initial = {initial_metrics['nmse']:.6e}")
    print(f"RMSE_Y_hat_after_structured_abs = {structured_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat_after_structured = {structured_metrics['nmse']:.6e}")
    if vp_enabled:
        print(f"RMSE_Y_hat_after_VP_abs = {final_metrics['rmse_abs']:.6e}")
        print(f"NMSE_Y_hat_after_VP = {final_metrics['nmse']:.6e}")
    else:
        print(f"RMSE_Y_hat_final_stage2_only_abs = {final_metrics['rmse_abs']:.6e}")
        print(f"NMSE_Y_hat_final_stage2_only = {final_metrics['nmse']:.6e}")
    print(f"RMSE_Y_hat_abs = {final_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat = {final_metrics['nmse']:.6e}")
    print(f"after_structured_Y_RMSE_abs = {structured_metrics['rmse_abs']:.6e}")
    print(f"after_structured_Y_NMSE = {structured_metrics['nmse']:.6e}")
    if vp_enabled:
        print(f"after_VP_Y_RMSE_abs = {final_metrics['rmse_abs']:.6e}")
        print(f"after_VP_Y_NMSE = {final_metrics['nmse']:.6e}")

    if not 0.70 <= noise_metrics["NMSE_Y_noisy"] <= 1.30:
        print("WARNING OBJECTIVE_MISMATCH: NMSE_Y_noisy is not close to 1 at 0 dB; check AWGN scaling.")
    if vp_enabled and final_metrics["nmse"] > structured_metrics["nmse"]:
        print(
            "WARNING VP_NO_GAIN: Raw VP-WNLS worsened true-domain Y NMSE after Stage-II in this run; "
            "likely cause is noisy-domain fitting or weak nonlinear initialization."
        )

    return {
        "initial": initial_metrics,
        "structured": structured_metrics,
        "final": final_metrics,
        "direct_final": direct_final_metrics,
        "vp": final_metrics,
        "direct_vp": direct_final_metrics,
        "noise": noise_metrics,
    }


def _print_z_stage_metrics(results: dict) -> list[dict]:
    """Print true-domain and noisy-domain Z residuals for Stage II."""
    print("\n=== Z-domain structured-stage diagnostics ===")
    z_true = results["Z_true"]
    z_noisy = results["Z_noisy"]
    initial_metrics = z_metric_summary(results["estimate_initial"]["Z_hat"], z_true, z_noisy)
    print(f"initial_Z_RMSE_noisy = {initial_metrics['rmse_noisy']:.6e}")
    print(f"initial_Z_RMSE_true = {initial_metrics['rmse_true']:.6e}")
    print(f"initial_Z_NMSE_noisy = {initial_metrics['nmse_noisy']:.6e}")
    print(f"initial_Z_NMSE_true = {initial_metrics['nmse_true']:.6e}")

    history_metrics = []
    for idx, z_hat in enumerate(results["structured_diag"]["z_hat_history"], start=1):
        metrics = z_metric_summary(z_hat, z_true, z_noisy)
        history_metrics.append(metrics)
        print(f"structured_iter_{idx}_Z_RMSE_noisy = {metrics['rmse_noisy']:.6e}")
        print(f"structured_iter_{idx}_Z_RMSE_true = {metrics['rmse_true']:.6e}")
        print(f"structured_iter_{idx}_Z_NMSE_noisy = {metrics['nmse_noisy']:.6e}")
        print(f"structured_iter_{idx}_Z_NMSE_true = {metrics['nmse_true']:.6e}")

    if history_metrics and history_metrics[-1]["nmse_true"] >= initial_metrics["nmse_true"]:
        print("WARNING OBJECTIVE_MISMATCH: Stage-II did not reduce true-domain Z NMSE in this run.")
    return [initial_metrics] + history_metrics


def _structured_parameter_arrays(scene: dict, estimate: dict) -> dict:
    tau_hat = np.array([tau_from_pole(pole, scene["delta_f"]) for pole in estimate["poles"]])
    return {
        "tau": tau_hat,
        "range": np.asarray(estimate["ris_eta"][:, 0], dtype=float),
        "elev": np.asarray(estimate["ris_eta"][:, 1], dtype=float),
        "az": np.asarray(estimate["ris_eta"][:, 2], dtype=float),
        "gamma": np.asarray(estimate.get("gamma", []), dtype=float),
        "eta_pol": np.asarray(estimate.get("eta_pol", []), dtype=float),
    }


def _vp_parameter_arrays(final: dict) -> dict:
    components = final["components"]
    return {
        "tau": np.asarray(components["taus"], dtype=float),
        "range": np.asarray(components["ranges"], dtype=float),
        "elev": np.asarray(components.get("elevations", []), dtype=float),
        "az": np.asarray(components.get("azimuths", []), dtype=float),
        "gamma": np.asarray(final.get("gamma", []), dtype=float),
        "eta_pol": np.asarray(final.get("eta_pol", []), dtype=float),
    }


def _geometry_error_metrics(scene: dict, arrays: dict, p_hat: np.ndarray, true_components: dict) -> dict:
    az_err = np.array(
        [
            _wrap_angle_rad(arrays["az"][k] - true_components["azimuths"][k])
            for k in range(scene["K"])
        ]
    )
    return {
        "tau_RMSE_s": float(
            np.linalg.norm(arrays["tau"] - true_components["taus"]) / np.sqrt(scene["K"])
        ),
        "range_RMSE_m": float(
            np.linalg.norm(arrays["range"] - true_components["ranges"]) / np.sqrt(scene["K"])
        ),
        "elev_RMSE_deg": float(
            np.rad2deg(
                np.linalg.norm(arrays["elev"] - true_components["elevations"])
                / np.sqrt(scene["K"])
            )
        ),
        "az_RMSE_deg": float(np.rad2deg(np.linalg.norm(az_err) / np.sqrt(scene["K"]))),
        "position_RMSE_m": position_rmse(p_hat, scene["p_u_true"]),
    }


def _structured_geometry_metrics(scene: dict, estimate: dict, true_components: dict) -> dict:
    arrays = _structured_parameter_arrays(scene, estimate)
    p_hat = estimate_position_from_ris_eta(scene, estimate)
    return _geometry_error_metrics(scene, arrays, p_hat, true_components)


def _vp_geometry_metrics(scene: dict, final: dict, true_components: dict) -> dict:
    arrays = _vp_parameter_arrays(final)
    return _geometry_error_metrics(scene, arrays, final["p_u"], true_components)


def _parameter_table_rows(
    scene: dict,
    arrays: dict,
    c_rows: np.ndarray | None,
    a_rows: np.ndarray | None,
    true_components: dict | None,
) -> list[dict]:
    rows = []
    for k in range(scene["K"]):
        tau = arrays["tau"][k]
        range_m = arrays["range"][k]
        elev = arrays["elev"][k] if arrays["elev"].size else float("nan")
        az = arrays["az"][k] if arrays["az"].size else float("nan")
        gamma = arrays["gamma"][k] if arrays["gamma"].size else float("nan")
        eta_pol = arrays["eta_pol"][k] if arrays["eta_pol"].size else float("nan")
        if true_components is None:
            tau_err_ps = range_err_m = elev_err_deg = az_err_deg = float("nan")
        else:
            tau_err_ps = (tau - true_components["taus"][k]) * 1.0e12
            range_err_m = range_m - true_components["ranges"][k]
            elev_err_deg = np.rad2deg(elev - true_components["elevations"][k])
            az_err_deg = np.rad2deg(_wrap_angle_rad(az - true_components["azimuths"][k]))

        ris_residual = float("nan")
        if c_rows is not None and np.isfinite([range_m, elev, az]).all():
            c_vec = c_rows[k] if c_rows.shape[0] == scene["K"] else c_rows[:, k]
            ris_residual = _ris_local_residual(
                scene, k, np.asarray(c_vec), np.array([range_m, elev, az])
            )

        evs_residual = float("nan")
        if (
            a_rows is not None
            and np.isfinite(gamma)
            and np.isfinite(eta_pol)
        ):
            a_vec = a_rows[k] if a_rows.shape[0] == scene["K"] else a_rows[:, k]
            evs_residual = _evs_local_residual(
                scene, k, np.asarray(a_vec), gamma, eta_pol
            )

        rows.append(
            {
                "path": k,
                "panel": k,
                "tau_ns": tau * 1.0e9,
                "tau_err_ps": tau_err_ps,
                "range_m": range_m,
                "range_err_m": range_err_m,
                "elev_deg": np.rad2deg(elev),
                "elev_err_deg": elev_err_deg,
                "az_deg": np.rad2deg(az),
                "az_err_deg": az_err_deg,
                "gamma_deg": np.rad2deg(gamma),
                "eta_pol_deg": np.rad2deg(eta_pol),
                "RIS_local_residual": ris_residual,
                "EVS_local_residual": evs_residual,
            }
        )
    return rows


def _print_parameter_table(title: str, rows: list[dict]) -> None:
    columns = [
        "path",
        "panel",
        "tau_ns",
        "tau_err_ps",
        "range_m",
        "range_err_m",
        "elev_deg",
        "elev_err_deg",
        "az_deg",
        "az_err_deg",
        "gamma_deg",
        "eta_pol_deg",
        "RIS_local_residual",
        "EVS_local_residual",
    ]
    print(f"\n=== {title} ===")
    print(" | ".join(columns))
    for row in rows:
        print(
            " | ".join(
                str(row[col]) if col in ("path", "panel") else _fmt(row[col], 5)
                for col in columns
            )
        )


def _print_per_path_parameter_tables(results: dict) -> None:
    scene = results["scene"]
    true_components = results.get("true_components")
    initial_arrays = _structured_parameter_arrays(scene, results["estimate_initial"])
    structured_arrays = _structured_parameter_arrays(scene, results["estimate_used"])
    _print_parameter_table(
        "Per-path parameters after Stage-I",
        _parameter_table_rows(
            scene,
            initial_arrays,
            results["estimate_initial"]["C"],
            results["estimate_initial"]["A"],
            true_components,
        ),
    )
    _print_parameter_table(
        "Per-path parameters after Stage-II",
        _parameter_table_rows(
            scene,
            structured_arrays,
            results["estimate_used"]["C"],
            results["estimate_used"]["A"],
            true_components,
        ),
    )
    if bool(results["final"].get("vp_enabled", True)):
        final_arrays = _vp_parameter_arrays(results["final"])
        _print_parameter_table(
            "Per-path parameters after VP",
            _parameter_table_rows(
                scene,
                final_arrays,
                results["final"]["components"].get("c"),
                results["final"]["components"].get("a_EVS"),
                true_components,
            ),
        )


def _relative_change_scalar(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before) or abs(before) <= 1.0e-300:
        return float("nan")
    return float((after - before) / abs(before))


def _print_stage2_summary_table(results: dict) -> dict:
    """Print Stage-I versus after Stage-II summary metrics."""
    scene = results["scene"]
    true_components = results["true_components"]
    initial_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_initial"], scene
    )
    structured_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_used"], scene
    )
    initial_geom = _structured_geometry_metrics(
        scene, results["estimate_initial"], true_components
    )
    structured_geom = _structured_geometry_metrics(
        scene, results["estimate_used"], true_components
    )
    initial_values = {
        "Y_NMSE_true": relative_nmse(initial_y_hat, results["Y_true"]),
        "Z_NMSE_true": relative_nmse(results["estimate_initial"]["Z_hat"], results["Z_true"]),
        "global_Z_SSE": float(np.linalg.norm(results["estimate_initial"]["Z_hat"] - results["Z_noisy"]) ** 2),
        **initial_geom,
        "num_EVS_accepted": 0.0,
        "num_delay_accepted": 0.0,
        "num_RIS_accepted": 0.0,
        "num_iteration_rollbacks": 0.0,
    }
    updates = results["structured_diag"]["updates"]
    structured_values = {
        "Y_NMSE_true": relative_nmse(structured_y_hat, results["Y_true"]),
        "Z_NMSE_true": relative_nmse(results["estimate_used"]["Z_hat"], results["Z_true"]),
        "global_Z_SSE": float(np.linalg.norm(results["estimate_used"]["Z_hat"] - results["Z_noisy"]) ** 2),
        **structured_geom,
        "num_EVS_accepted": float(
            sum(
                bool(detail.get("accepted", False))
                for update in updates
                for detail in update.get("evs_projection_details", [])
            )
        ),
        "num_delay_accepted": float(
            sum(bool(update.get("delay_projection_details", {}).get("accepted", False)) for update in updates)
        ),
        "num_RIS_accepted": float(
            sum(
                bool(detail.get("accepted", False))
                for update in updates
                for detail in update.get("ris_projection_details", [])
            )
        ),
        "num_iteration_rollbacks": float(
            sum(not bool(update.get("iteration_accepted", True)) for update in updates)
        ),
    }
    rows = [
        "Y_NMSE_true",
        "Z_NMSE_true",
        "tau_RMSE_s",
        "range_RMSE_m",
        "elev_RMSE_deg",
        "az_RMSE_deg",
        "position_RMSE_m",
        "global_Z_SSE",
        "num_EVS_accepted",
        "num_delay_accepted",
        "num_RIS_accepted",
        "num_iteration_rollbacks",
    ]
    print("\n=== Stage-II summary ===")
    print("metric | Stage-I | After Stage-II | Relative change")
    for key in rows:
        before = initial_values[key]
        after = structured_values[key]
        print(
            f"{key} | {_fmt(before)} | {_fmt(after)} | "
            f"{_fmt(_relative_change_scalar(after, before))}"
        )

    if (
        structured_values["global_Z_SSE"] < initial_values["global_Z_SSE"]
        and structured_values["Z_NMSE_true"] > initial_values["Z_NMSE_true"]
    ):
        print(
            "WARNING OBJECTIVE_MISMATCH: Stage-II reduced noisy-domain global_Z_SSE "
            "but increased true-domain Z_NMSE_true."
        )
    for key in ("range_RMSE_m", "elev_RMSE_deg", "az_RMSE_deg", "position_RMSE_m"):
        if structured_values[key] > initial_values[key] + 1.0e-12:
            print(
                f"WARNING GEOM_DEGRADE: {key} increased from "
                f"{initial_values[key]:.6e} to {structured_values[key]:.6e}."
            )

    return {"initial": initial_values, "structured": structured_values}


def _print_parameter_diagnostics(results: dict) -> dict:
    """Print tau, range, position, and compact per-path diagnostics."""
    print("\n=== Parameter diagnostics ===")
    scene = results["scene"]
    true_components = results["true_components"]
    vp_enabled = bool(results["final"].get("vp_enabled", True))
    initial = parameter_errors_for_structured(scene, results["estimate_initial"], true_components)
    structured = parameter_errors_for_structured(scene, results["estimate_used"], true_components)
    final = parameter_errors_for_vp(scene, results["final"], true_components)

    print(f"tau_RMSE_initial = {initial['tau_rmse']:.6e}")
    print(f"tau_RMSE_after_structured = {structured['tau_rmse']:.6e}")
    if vp_enabled:
        print(f"tau_RMSE_after_VP = {final['tau_rmse']:.6e}")
    else:
        print(f"tau_RMSE_final_stage2_only = {final['tau_rmse']:.6e}")
    print(f"range_RMSE_initial = {initial['range_rmse']:.6e}")
    print(f"range_RMSE_after_structured = {structured['range_rmse']:.6e}")
    if vp_enabled:
        print(f"range_RMSE_after_VP = {final['range_rmse']:.6e}")
    else:
        print(f"range_RMSE_final_stage2_only = {final['range_rmse']:.6e}")
    print(f"position_RMSE_initial = {initial['position_rmse']:.6e}")
    print(f"position_RMSE_after_structured = {structured['position_rmse']:.6e}")
    if vp_enabled:
        print(f"position_RMSE_after_VP = {final['position_rmse']:.6e}")
    else:
        print(f"position_RMSE_final_stage2_only = {final['position_rmse']:.6e}")

    print(f"true_tau_ns = {format_float_list(true_components['taus'], scale=1e9)}")
    print(f"initial_tau_ns = {format_float_list(initial['tau_hat'], scale=1e9)}")
    print(f"structured_tau_ns = {format_float_list(structured['tau_hat'], scale=1e9)}")
    if vp_enabled:
        print(f"VP_tau_ns = {format_float_list(final['tau_hat'], scale=1e9)}")
    else:
        print(f"final_stage2_tau_ns = {format_float_list(final['tau_hat'], scale=1e9)}")
    print(f"true_range_m = {format_float_list(true_components['ranges'])}")
    print(f"initial_range_m = {format_float_list(initial['range_hat'])}")
    print(f"structured_range_m = {format_float_list(structured['range_hat'])}")
    if vp_enabled:
        print(f"VP_range_m = {format_float_list(final['range_hat'])}")
    else:
        print(f"final_stage2_range_m = {format_float_list(final['range_hat'])}")
    print(f"true_RIS_panel_assignment = {list(range(scene['K']))}")
    print(f"estimated_col_to_panel_assignment = {results['estimate_initial']['assignment']}")
    _print_per_path_parameter_tables(results)
    return {"initial": initial, "structured": structured, "final": final, "vp": final}


def _fmt_eta(eta: np.ndarray | None) -> str:
    if eta is None:
        return "range_m=NA,elev_deg=NA,az_deg=NA"
    arr = np.asarray(eta, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return "range_m=NA,elev_deg=NA,az_deg=NA"
    return (
        f"range_m={arr[0]:.6e},"
        f"elev_deg={np.rad2deg(arr[1]):.6e},"
        f"az_deg={np.rad2deg(arr[2]):.6e}"
    )


def _print_stage_two_update_diagnostics(results: dict) -> None:
    """Print whether Stage-II variables and projections are changing."""
    print("\n=== Stage-II update diagnostics ===")
    unchanged_ris_count = 0
    for idx, update in enumerate(results["structured_diag"]["updates"], start=1):
        print(
            f"iter {idx}: "
            f"delta_A={update['delta_A']:.3e}, "
            f"delta_B={update['delta_B']:.3e}, "
            f"delta_Q={update['delta_Q']:.3e}, "
            f"delta_C={update['delta_C']:.3e}, "
            f"delta_beta={update['delta_beta']:.3e}, "
            f"nonfinite(A,B,Q,C,beta)="
            f"({update['nonfinite_A']},{update['nonfinite_B']},"
            f"{update['nonfinite_Q']},{update['nonfinite_C']},"
            f"{update['nonfinite_beta']})"
        )
        if any(
            update[key] > 0
            for key in (
                "nonfinite_A",
                "nonfinite_B",
                "nonfinite_Q",
                "nonfinite_C",
                "nonfinite_beta",
            )
        ):
            print("  WARNING NONFINITE: Stage-II iterate contains nonfinite entries.")
        print(
            "  iteration_guard: "
            f"accepted={update.get('iteration_accepted', True)}, "
            f"global_SSE_before={_fmt(update.get('iteration_sse_before'))}, "
            f"global_SSE_proposed={_fmt(update.get('iteration_sse_proposed'))}, "
            f"global_SSE_after={_fmt(update.get('iteration_sse_after'))}, "
            f"relative_change={_fmt(update.get('relative_residual_change'))}"
        )

        for detail in update["evs_projection_details"]:
            path = detail.get("path", "?")
            gamma_before = np.rad2deg(detail.get("gamma_before", np.nan))
            gamma_after = np.rad2deg(detail.get("gamma_after", np.nan))
            eta_before = np.rad2deg(detail.get("eta_pol_before", np.nan))
            eta_after = np.rad2deg(detail.get("eta_pol_after", np.nan))
            print(
                f"  EVS path {path}: "
                f"accepted={detail.get('accepted', False)}, "
                f"reason={detail.get('reason', 'unknown')}, "
                f"local_res_before={_fmt(detail.get('local_res_before'))}, "
                f"local_res_after={_fmt(detail.get('local_res_after'))}, "
                f"relative_improvement={_fmt(detail.get('relative_improvement'))}, "
                f"global_SSE_before={_fmt(detail.get('global_sse_before'))}, "
                f"global_SSE_after={_fmt(detail.get('global_sse_after'))}, "
                f"damping_rho={_fmt(detail.get('best_rho'))}, "
                f"gamma_deg_before={_fmt(gamma_before)}, "
                f"gamma_deg_after={_fmt(gamma_after)}, "
                f"eta_pol_deg_before={_fmt(eta_before)}, "
                f"eta_pol_deg_after={_fmt(eta_after)}"
            )

        delay_detail = update["delay_projection_details"]
        print(
            "  delay structured LS: "
            f"skipped={delay_detail.get('skipped', False)}, "
            f"accepted={delay_detail.get('accepted', False)}, "
            f"reason={delay_detail.get('reason', 'unknown')}, "
            f"local_res_before={_fmt(delay_detail.get('local_res_before'))}, "
            f"local_res_after={_fmt(delay_detail.get('local_res_after'))}, "
            f"relative_improvement={_fmt(delay_detail.get('relative_improvement'))}, "
            f"global_SSE_before={_fmt(delay_detail.get('global_sse_before'))}, "
            f"global_SSE_after={_fmt(delay_detail.get('global_sse_after'))}, "
            f"damping_rho={_fmt(delay_detail.get('damping'))}, "
            f"tau_ns_before={_fmt_vector(delay_detail.get('tau_before', []), scale=1e9)}, "
            f"tau_ns_candidate={_fmt_vector(delay_detail.get('tau_candidate', []), scale=1e9)}, "
            f"tau_ns_after={_fmt_vector(delay_detail.get('tau_after', []), scale=1e9)}, "
            f"geom_accepted={delay_detail.get('geometry_correction_accepted', False)}"
        )
        print(f"  mode4_panel_order = {update.get('mode4_assignment_order')}")
        ris_accept = [
            "skipped" if detail.get("skipped", False) else detail.get("accepted", False)
            for detail in update["ris_projection_details"]
        ]
        print(f"  RIS projection accepted = {ris_accept}")
        for detail in update["ris_projection_details"]:
            eta = detail.get("selected_eta")
            if detail.get("skipped", False):
                print(
                    f"  RIS path {detail['path']}: "
                    f"skipped=True, "
                    f"accepted={detail.get('accepted', False)}, "
                    f"reason={detail.get('reason', 'unknown')}, "
                    f"geometry_after=({_fmt_eta(eta)})"
                )
                continue
            print(
                f"  RIS path {detail['path']}: "
                f"accepted={detail.get('accepted', False)}, "
                f"reason={detail.get('reason', 'unknown')}, "
                f"local_res_before={_fmt(detail.get('residual_before'))}, "
                f"local_res_after={_fmt(detail.get('residual_after'))}, "
                f"relative_improvement={_fmt(detail.get('relative_improvement'))}, "
                f"global_SSE_before={_fmt(detail.get('global_sse_before'))}, "
                f"global_SSE_after={_fmt(detail.get('global_sse_after'))}, "
                f"damping_rho={_fmt(detail.get('best_rho'))}, "
                f"projection_time_s={_fmt(detail.get('projection_time_s'))}, "
                f"selected_model={detail.get('selected_model')}, "
                f"lifted_used={detail.get('lifted_used', False)}, "
                f"c_delta={_fmt(detail.get('c_relative_change'))}, "
                f"geometry_before=({_fmt_eta(detail.get('eta_before'))}), "
                f"geometry_candidate=({_fmt_eta(detail.get('candidate_eta'))}), "
                f"geometry_after=({_fmt_eta(eta)})"
            )
            for candidate in detail.get("candidate_ranking", [])[:3]:
                print(
                    f"    RIS candidate rank {candidate.get('rank')}: "
                    f"model={candidate.get('model')}, "
                    f"range_m={_fmt(candidate.get('range_m'))}, "
                    f"elev_deg={_fmt(candidate.get('elev_deg'))}, "
                    f"az_deg={_fmt(candidate.get('az_deg'))}, "
                    f"local_residual={_fmt(candidate.get('local_residual'))}, "
                    f"exact_refined={candidate.get('exact_refined')}, "
                    f"selected={candidate.get('selected')}"
                )
            if detail.get("c_relative_change", 0.0) < 1e-8:
                unchanged_ris_count += 1
                print(
                    "  WARNING RIS_STAGNATION: RIS Mode-4 projection returned "
                    "an almost unchanged c_k."
                )
    if unchanged_ris_count:
        print(
            "WARNING RIS_STAGNATION: RIS Mode-4 projection stagnated in at least one path/iteration; "
            "likely cause is the compressed RIS projection selecting the same local grid optimum."
        )


def _print_structured_comparison(results: dict, direct_results: dict, y_metrics: dict) -> None:
    """Print final estimates with versus without the structured stage."""
    if str(results["final"].get("stage2_mode", "none")).lower() == "none":
        print("\n=== With vs without structured stage ===")
        print("legacy_structured_stage_enabled = False")
        print("comparison_note = default pipeline bypasses legacy factor-domain Stage-II")
        return

    y_true = results["Y_true"]
    _ = y_true
    vp_enabled = bool(results["final"].get("vp_enabled", True))
    direct_nmse = y_metrics["direct_final"]["nmse"]
    with_nmse = y_metrics["final"]["nmse"]
    direct_pos = position_rmse(direct_results["final"]["p_u"], results["scene"]["p_u_true"])
    with_pos = position_rmse(results["final"]["p_u"], results["scene"]["p_u_true"])
    improvement = direct_nmse - with_nmse

    print("\n=== With vs without structured stage ===")
    if vp_enabled:
        print(f"NMSE_Y_after_VP_without_structured = {direct_nmse:.6e}")
    else:
        print(f"NMSE_Y_final_without_structured_no_VP = {direct_nmse:.6e}")
    print(f"position_RMSE_without_structured = {direct_pos:.6e}")
    if vp_enabled:
        print(f"NMSE_Y_after_VP_with_structured = {with_nmse:.6e}")
    else:
        print(f"NMSE_Y_final_with_structured_no_VP = {with_nmse:.6e}")
    print(f"position_RMSE_with_structured = {with_pos:.6e}")
    print(f"improvement_from_structured = {improvement:.6e}")
    if abs(improvement) < 1e-4 and abs(direct_pos - with_pos) < 1e-3:
        if vp_enabled:
            print(
                "WARNING VP_NO_GAIN: Structured HP-R1P-CPD stage currently gives little improvement "
                "over direct VP-WNLS."
            )
        else:
            print(
                "WARNING OBJECTIVE_MISMATCH: Structured HP-R1P-CPD stage currently gives little improvement "
                "over initialization-only output."
            )


def _print_vp_branch_comparison(results: dict, direct_results: dict) -> None:
    """Print direct Stage-I+VP versus Stage-I+Stage-II+VP diagnostics."""
    if not bool(results["final"].get("vp_enabled", True)):
        return
    scene = results["scene"]
    true_components = results["true_components"]
    branches = [
        ("Stage-I+VP", direct_results["final"]),
        ("Stage-I+Stage-II+VP", results["final"]),
    ]
    print("\n=== VP branch comparison ===")
    print(
        "branch | solver_type | solver_backend | raw_objective_initial | "
        "raw_objective_final | nfev | success | position_RMSE_m | "
        "range_RMSE_m | tau_RMSE_s | message"
    )
    for label, final in branches:
        optimizer = final.get("optimizer", {})
        metrics = _vp_geometry_metrics(scene, final, true_components)
        raw_initial = final.get("raw_objective_initial")
        raw_final = final.get("raw_objective_final")
        print(
            f"{label} | {optimizer.get('method', 'unknown')} | "
            f"{optimizer.get('solver_backend', 'unknown')} | "
            f"{_fmt(raw_initial)} | {_fmt(raw_final)} | "
            f"{optimizer.get('n_eval', 'NA')} | {optimizer.get('success', 'NA')} | "
            f"{_fmt(metrics['position_RMSE_m'])} | {_fmt(metrics['range_RMSE_m'])} | "
            f"{_fmt(metrics['tau_RMSE_s'])} | {optimizer.get('message', '')}"
        )
        if (
            raw_initial is not None
            and raw_final is not None
            and np.isfinite(raw_initial)
            and np.isfinite(raw_final)
            and raw_final >= raw_initial - 1.0e-12
        ):
            print(
                f"WARNING VP_NO_GAIN: {label} raw objective did not decrease "
                f"({_fmt(raw_initial)} -> {_fmt(raw_final)})."
            )


def _print_runtime_profile(results: dict, direct_results: dict) -> None:
    """Print always-on runtime profiling collected with time.perf_counter."""
    timing = results.get("timing", {})
    print("\n=== Runtime profile ===")
    print(f"data_generation_s = {_fmt(timing.get('data_generation'))}")
    print(f"hankelization_s = {_fmt(timing.get('hankelization'))}")
    print(f"stage1_s = {_fmt(timing.get('stage1'))}")
    print(f"stage2_s = {_fmt(timing.get('stage2'))}")
    print(f"legacy_stage2_s = {_fmt(timing.get('stage2'))}")
    print(f"ris_projection_total_s = {_fmt(timing.get('ris_projection_total'))}")
    for iter_idx, update in enumerate(results["structured_diag"]["updates"], start=1):
        print(
            f"stage2_iter_{iter_idx}_mode4_assignment_time_s = "
            f"{_fmt(update.get('mode4_assignment_time_s'))}"
        )
        for detail in update.get("ris_projection_details", []):
            print(
                f"stage2_iter_{iter_idx}_ris_path_{detail.get('path')}_projection_time_s = "
                f"{_fmt(detail.get('projection_time_s'))}"
            )
    print(f"vp_s = {_fmt(timing.get('vp'))}")
    print(f"global_vp_s = {_fmt(timing.get('vp'))}")
    print(f"total_s = {_fmt(timing.get('total'))}")
    direct_timing = direct_results.get("timing", {})
    print(f"direct_stage1_vp_total_s = {_fmt(direct_timing.get('total'))}")
    print(f"direct_stage1_vp_vp_s = {_fmt(direct_timing.get('vp'))}")


def run_default_diagnostic() -> None:
    """Run and print the default SNR=0 diagnostic report."""
    config = default_config()
    results = _run_single_pipeline(config, use_structured=True)
    if str(config.get("stage2_mode", "none")).lower() == "none":
        direct_results = copy.deepcopy(results)
    else:
        direct_results = _run_single_pipeline(config, use_structured=False)
    scene = results["scene"]

    print("=== Single proposed diagnostic run ===")
    _print_run_configuration(config, results)
    _print_ris_dimension_diagnostics(scene, results["true_components"])
    _print_assignment_diagnostics(results)
    _print_global_vp_diagnostics(results)
    if not scipy_is_available():
        print("optimizer_note = scipy.optimize not found; using deterministic fallback optimizer")

    _print_self_tests(scene, config, results["true_components"])
    y_metrics = _print_noise_and_y_metrics(results, direct_results, config["SNR_dB"])
    _print_z_stage_metrics(results)
    _print_stage2_summary_table(results)
    param_metrics = _print_parameter_diagnostics(results)
    _print_stage_two_update_diagnostics(results)
    _print_vp_branch_comparison(results, direct_results)
    _print_structured_comparison(results, direct_results, y_metrics)
    _print_runtime_profile(results, direct_results)

    print("\n=== Final result ===")
    print(f"Y_true shape = {results['Y_true'].shape}")
    print(f"Y_noisy shape = {results['Y_noisy'].shape}")
    print(f"Y_hat shape = {results['final']['Y_hat'].shape}")
    print(f"global_VP_enabled = {results['final'].get('vp_enabled', True)}")
    print(f"RMSE_Y_abs = {y_metrics['final']['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat = {y_metrics['final']['nmse']:.6e}")
    print(f"UE_position_RMSE_m = {param_metrics['final']['position_rmse']:.6e}")


def _run_compact(config: dict) -> dict:
    """Run the full pipeline once and return compact sweep metrics."""
    results = _run_single_pipeline(config, use_structured=True)
    y_true = results["Y_true"]
    y_noisy = results["Y_noisy"]
    final_y = results["final"]["Y_hat"]
    true_components = results["true_components"]
    final_params = parameter_errors_for_vp(results["scene"], results["final"], true_components)
    structured_params = parameter_errors_for_structured(
        results["scene"], results["estimate_used"], true_components
    )
    return {
        "NMSE_Y_noisy": relative_nmse(y_noisy, y_true),
        "NMSE_Y_hat_final": relative_nmse(final_y, y_true),
        "position_RMSE_final": final_params["position_rmse"],
        "global_VP_enabled": bool(results["final"].get("vp_enabled", True)),
        "range_RMSE_after_structured": structured_params["range_rmse"],
    }


def run_snr_sweep() -> None:
    """Run a small one-seed SNR diagnostic sweep."""
    print("=== Diagnostic SNR sweep ===")
    snrs = [-10.0, 0.0, 10.0, 20.0, 30.0]
    position_errors = []
    for snr in snrs:
        config = default_config()
        config["SNR_dB"] = snr
        metrics = _run_compact(config)
        position_errors.append(metrics["position_RMSE_final"])
        print(
            f"SNR_dB={snr:.1f}, "
            f"NMSE_Y_noisy={metrics['NMSE_Y_noisy']:.6e}, "
            f"NMSE_Y_hat_final={metrics['NMSE_Y_hat_final']:.6e}, "
            f"position_RMSE_final={metrics['position_RMSE_final']:.6e}, "
            f"global_VP_enabled={metrics['global_VP_enabled']}"
        )
    if position_errors[-1] > position_errors[0]:
        print("WARNING GEOM_DEGRADE: UE position RMSE did not improve from -10 dB to 30 dB.")


def run_mr_sweep() -> None:
    """Run a small RIS-size diagnostic sweep."""
    print("=== Diagnostic M_R sweep ===")
    cases = [((4, 4), 18), ((8, 8), 32), ((16, 16), 64)]
    for ris_shape, t_dim in cases:
        config = default_config()
        config["ris_shape"] = ris_shape
        config["T"] = t_dim
        metrics = _run_compact(config)
        print(
            f"M_Rx={ris_shape[0]}, M_Ry={ris_shape[1]}, M_R={ris_shape[0] * ris_shape[1]}, "
            f"T={t_dim}, "
            f"range_RMSE_after_structured={metrics['range_RMSE_after_structured']:.6e}, "
            f"position_RMSE_final={metrics['position_RMSE_final']:.6e}, "
            f"global_VP_enabled={metrics['global_VP_enabled']}"
        )


def main() -> None:
    """CLI entrypoint for default diagnostics and optional sweeps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-snr-sweep", action="store_true")
    parser.add_argument("--diagnostic-mr-sweep", action="store_true")
    args = parser.parse_args()

    if args.diagnostic_snr_sweep:
        run_snr_sweep()
    elif args.diagnostic_mr_sweep:
        run_mr_sweep()
    else:
        run_default_diagnostic()


if __name__ == "__main__":
    main()
