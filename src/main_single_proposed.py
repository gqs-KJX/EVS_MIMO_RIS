"""Run one diagnostic proposed RIS-EVS-OFDM estimation demo."""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

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
        parameter_errors_for_structured,
        parameter_errors_for_vp,
        run_delay_projection_self_test,
        run_ris_projection_self_test,
        run_tensor_factorization_shape_self_test,
        y_metric_summary,
        z_metric_summary,
    )
    from src.estimators import (
        initialize_from_hankel,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from src.metrics import position_rmse, relative_nmse, rmse_abs
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
        parameter_errors_for_structured,
        parameter_errors_for_vp,
        run_delay_projection_self_test,
        run_ris_projection_self_test,
        run_tensor_factorization_shape_self_test,
        y_metric_summary,
        z_metric_summary,
    )
    from .estimators import (
        initialize_from_hankel,
        reconstruct_raw_tensor_from_structured_estimate,
        refine_global_raw,
        structured_refinement,
    )
    from .metrics import position_rmse, relative_nmse, rmse_abs
    from .tensor_utils import hankelize_frequency
    from .utils import scipy_is_available


def _make_data(config: dict) -> dict:
    """Generate one reproducible synthetic channel and noisy observation."""
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
    z_true = hankelize_frequency(y_true, scene["P"])
    z_noisy = hankelize_frequency(y_noisy, scene["P"])

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
    }


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
        print(f"WARNING: {ris_test['warning']}")


def _run_single_pipeline(config: dict, use_structured: bool) -> dict:
    """Run initialization, optional Stage-II, and final raw-domain VP."""
    data = _make_data(config)
    scene = data["scene"]
    estimate_initial = initialize_from_hankel(data["Z_noisy"], scene, config)
    if use_structured:
        estimate_used, structured_diag = structured_refinement(
            data["Z_noisy"], scene, config, copy.deepcopy(estimate_initial)
        )
    else:
        estimate_used = copy.deepcopy(estimate_initial)
        structured_diag = {"z_hat_history": [], "residuals_noisy_rmse": [], "updates": []}
    final = refine_global_raw(data["Y_noisy"], scene, config, estimate_used)
    return {
        **data,
        "estimate_initial": estimate_initial,
        "estimate_used": estimate_used,
        "structured_diag": structured_diag,
        "final": final,
    }


def _print_noise_and_y_metrics(results: dict, direct_results: dict, snr_db: float) -> dict:
    """Print noise and raw-domain metrics for default diagnostics."""
    scene = results["scene"]
    y_true = results["Y_true"]
    y_noisy = results["Y_noisy"]
    initial_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_initial"], scene
    )
    structured_y_hat = reconstruct_raw_tensor_from_structured_estimate(
        results["estimate_used"], scene
    )
    vp_y_hat = results["final"]["Y_hat"]

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
    vp_metrics = y_metric_summary(vp_y_hat, y_true)
    direct_vp_metrics = y_metric_summary(direct_results["final"]["Y_hat"], y_true)

    print(f"RMSE_Y_hat_initial_abs = {initial_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat_initial = {initial_metrics['nmse']:.6e}")
    print(f"RMSE_Y_hat_after_structured_abs = {structured_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat_after_structured = {structured_metrics['nmse']:.6e}")
    print(f"RMSE_Y_hat_after_VP_abs = {vp_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat_after_VP = {vp_metrics['nmse']:.6e}")
    print(f"RMSE_Y_hat_abs = {vp_metrics['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat = {vp_metrics['nmse']:.6e}")
    print(f"after_structured_Y_RMSE_abs = {structured_metrics['rmse_abs']:.6e}")
    print(f"after_structured_Y_NMSE = {structured_metrics['nmse']:.6e}")
    print(f"after_VP_Y_RMSE_abs = {vp_metrics['rmse_abs']:.6e}")
    print(f"after_VP_Y_NMSE = {vp_metrics['nmse']:.6e}")

    if not 0.70 <= noise_metrics["NMSE_Y_noisy"] <= 1.30:
        print("WARNING: NMSE_Y_noisy is not close to 1 at 0 dB; check AWGN scaling.")
    if vp_metrics["nmse"] > structured_metrics["nmse"]:
        print(
            "WARNING: Raw VP-WNLS worsened true-domain Y NMSE after Stage-II in this run; "
            "likely cause is noisy-domain fitting or weak nonlinear initialization."
        )

    return {
        "initial": initial_metrics,
        "structured": structured_metrics,
        "vp": vp_metrics,
        "direct_vp": direct_vp_metrics,
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
        print("WARNING: Stage-II did not reduce true-domain Z NMSE in this run.")
    return [initial_metrics] + history_metrics


def _print_parameter_diagnostics(results: dict) -> dict:
    """Print tau, range, position, and compact per-path diagnostics."""
    print("\n=== Parameter diagnostics ===")
    scene = results["scene"]
    true_components = results["true_components"]
    initial = parameter_errors_for_structured(scene, results["estimate_initial"], true_components)
    structured = parameter_errors_for_structured(scene, results["estimate_used"], true_components)
    vp = parameter_errors_for_vp(scene, results["final"], true_components)

    print(f"tau_RMSE_initial = {initial['tau_rmse']:.6e}")
    print(f"tau_RMSE_after_structured = {structured['tau_rmse']:.6e}")
    print(f"tau_RMSE_after_VP = {vp['tau_rmse']:.6e}")
    print(f"range_RMSE_initial = {initial['range_rmse']:.6e}")
    print(f"range_RMSE_after_structured = {structured['range_rmse']:.6e}")
    print(f"range_RMSE_after_VP = {vp['range_rmse']:.6e}")
    print(f"position_RMSE_initial = {initial['position_rmse']:.6e}")
    print(f"position_RMSE_after_structured = {structured['position_rmse']:.6e}")
    print(f"position_RMSE_after_VP = {vp['position_rmse']:.6e}")

    print(f"true_tau_ns = {format_float_list(true_components['taus'], scale=1e9)}")
    print(f"initial_tau_ns = {format_float_list(initial['tau_hat'], scale=1e9)}")
    print(f"structured_tau_ns = {format_float_list(structured['tau_hat'], scale=1e9)}")
    print(f"VP_tau_ns = {format_float_list(vp['tau_hat'], scale=1e9)}")
    print(f"true_range_m = {format_float_list(true_components['ranges'])}")
    print(f"initial_range_m = {format_float_list(initial['range_hat'])}")
    print(f"structured_range_m = {format_float_list(structured['range_hat'])}")
    print(f"VP_range_m = {format_float_list(vp['range_hat'])}")
    print(f"true_RIS_panel_assignment = {list(range(scene['K']))}")
    print(f"estimated_col_to_panel_assignment = {results['estimate_initial']['assignment']}")
    return {"initial": initial, "structured": structured, "vp": vp}


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
        evs_accept = [detail["accepted"] for detail in update["evs_projection_details"]]
        ris_accept = [detail["accepted"] for detail in update["ris_projection_details"]]
        print(f"  EVS projection accepted = {evs_accept}")
        print(f"  RIS projection accepted = {ris_accept}")
        for detail in update["ris_projection_details"]:
            eta = detail["selected_eta"]
            print(
                f"  RIS path {detail['path']}: "
                f"res_before={detail['residual_before']:.3e}, "
                f"res_after={detail['residual_after']:.3e}, "
                f"range/elev/az=({eta[0]:.3f}, {eta[1]:.3f}, {eta[2]:.3f}), "
                f"c_delta={detail['c_relative_change']:.3e}, "
                f"lifted_used={detail.get('lifted_used', False)}"
            )
            if detail["c_relative_change"] < 1e-8:
                unchanged_ris_count += 1
                print("  WARNING: RIS Mode-4 projection returned an almost unchanged c_k.")
    if unchanged_ris_count:
        print(
            "WARNING: RIS Mode-4 projection stagnated in at least one path/iteration; "
            "likely cause is the compressed RIS projection selecting the same local grid optimum."
        )


def _print_structured_comparison(results: dict, direct_results: dict, y_metrics: dict) -> None:
    """Print direct VP versus structured+VP comparison."""
    y_true = results["Y_true"]
    direct_nmse = y_metrics["direct_vp"]["nmse"]
    with_nmse = y_metrics["vp"]["nmse"]
    direct_pos = position_rmse(direct_results["final"]["p_u"], results["scene"]["p_u_true"])
    with_pos = position_rmse(results["final"]["p_u"], results["scene"]["p_u_true"])
    improvement = direct_nmse - with_nmse

    print("\n=== With vs without structured stage ===")
    print(f"NMSE_Y_after_VP_without_structured = {direct_nmse:.6e}")
    print(f"position_RMSE_without_structured = {direct_pos:.6e}")
    print(f"NMSE_Y_after_VP_with_structured = {with_nmse:.6e}")
    print(f"position_RMSE_with_structured = {with_pos:.6e}")
    print(f"improvement_from_structured = {improvement:.6e}")
    if abs(improvement) < 1e-4 and abs(direct_pos - with_pos) < 1e-3:
        print(
            "WARNING: Structured HP-R1P-CPD stage currently gives little improvement "
            "over direct VP-WNLS."
        )


def run_default_diagnostic() -> None:
    """Run and print the default SNR=0 diagnostic report."""
    config = default_config()
    results = _run_single_pipeline(config, use_structured=True)
    direct_results = _run_single_pipeline(config, use_structured=False)
    scene = results["scene"]

    print("=== Single proposed diagnostic run ===")
    print(f"seed = {config['seed']}")
    print(f"SNR_dB = {config['SNR_dB']:.1f}")
    print(f"K = {scene['K']}")
    print(f"I = {scene['I']}")
    print(f"N = {scene['N']}")
    print(f"P = {scene['P']}")
    print(f"L = {scene['L']}")
    print(f"T = {scene['T']}")
    _print_ris_dimension_diagnostics(scene, results["true_components"])
    if not scipy_is_available():
        print("optimizer_note = scipy.optimize not found; using deterministic fallback optimizer")

    _print_self_tests(scene, config, results["true_components"])
    y_metrics = _print_noise_and_y_metrics(results, direct_results, config["SNR_dB"])
    _print_z_stage_metrics(results)
    param_metrics = _print_parameter_diagnostics(results)
    _print_stage_two_update_diagnostics(results)
    _print_structured_comparison(results, direct_results, y_metrics)

    print("\n=== Final result ===")
    print(f"Y_true shape = {results['Y_true'].shape}")
    print(f"Y_noisy shape = {results['Y_noisy'].shape}")
    print(f"Y_hat shape = {results['final']['Y_hat'].shape}")
    print(f"RMSE_Y_abs = {y_metrics['vp']['rmse_abs']:.6e}")
    print(f"NMSE_Y_hat = {y_metrics['vp']['nmse']:.6e}")
    print(f"UE_position_RMSE_m = {param_metrics['vp']['position_rmse']:.6e}")


def _run_compact(config: dict) -> dict:
    """Run the full pipeline once and return compact sweep metrics."""
    results = _run_single_pipeline(config, use_structured=True)
    y_true = results["Y_true"]
    y_noisy = results["Y_noisy"]
    final_y = results["final"]["Y_hat"]
    true_components = results["true_components"]
    vp_params = parameter_errors_for_vp(results["scene"], results["final"], true_components)
    structured_params = parameter_errors_for_structured(
        results["scene"], results["estimate_used"], true_components
    )
    return {
        "NMSE_Y_noisy": relative_nmse(y_noisy, y_true),
        "NMSE_Y_hat_after_VP": relative_nmse(final_y, y_true),
        "position_RMSE_after_VP": vp_params["position_rmse"],
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
        position_errors.append(metrics["position_RMSE_after_VP"])
        print(
            f"SNR_dB={snr:.1f}, "
            f"NMSE_Y_noisy={metrics['NMSE_Y_noisy']:.6e}, "
            f"NMSE_Y_hat_after_VP={metrics['NMSE_Y_hat_after_VP']:.6e}, "
            f"position_RMSE_after_VP={metrics['position_RMSE_after_VP']:.6e}"
        )
    if position_errors[-1] > position_errors[0]:
        print("WARNING: UE position RMSE did not improve from -10 dB to 30 dB.")


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
            f"position_RMSE_after_VP={metrics['position_RMSE_after_VP']:.6e}"
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
