import copy
import json

import numpy as np
import pytest

from src.config import default_config
from src.experiments import run_paper_ablation_figures as figures
from src.main_single_proposed import (
    _apply_main_single_defaults,
    _make_data,
    run_single_proposed_diagnostic,
    run_stage1_only,
)
from src.metrics import relative_nmse


def _smoke_config():
    config = default_config()
    config.update(
        {
            "seed": 20260526,
            "SNR_dB": 0.0,
            "K": 3,
            "receiver_mode": "full_6d",
            "diagnostic_mode": "smoke",
            "diagnostic_fast_problem_size": True,
            "diagnostic_fast_stage1_search": True,
            "print_progress": False,
        }
    )
    return _apply_main_single_defaults(config)


def _result_metrics(result):
    final = result["final"]
    p_hat = np.asarray(final["p_u"], dtype=float)
    p_true = np.asarray(result["scene"]["p_u_true"], dtype=float)
    return {
        "p_hat": p_hat,
        "delta_t": float(final["delta_t"]),
        "raw_objective_final": float(final["raw_objective_final"]),
        "y_nmse": float(relative_nmse(final["Y_hat"], result["Y_true"])),
        "position_error_m": float(np.linalg.norm(p_hat - p_true)),
        "global_vp_mode": final.get("global_vp_mode"),
        "jones_mode": final.get("selected_vp_family_branch"),
        "adaptive_enabled": final.get("global_vp_mode") == "adaptive_jones",
    }


def test_fig1_adaptive_variant_dispatch_matches_main_single():
    config = _smoke_config()
    data = _make_data(config)
    result_main = run_single_proposed_diagnostic(
        copy.deepcopy(config), data_override=copy.deepcopy(data)
    )
    stage1 = run_stage1_only(copy.deepcopy(data), copy.deepcopy(config))
    result_fig1 = figures.run_fig1_adaptive_from_shared(
        copy.deepcopy(data), copy.deepcopy(stage1), copy.deepcopy(config)
    )
    main_metrics = _result_metrics(result_main)
    fig_metrics = _result_metrics(result_fig1)
    config_diff = figures._config_diff_summary(
        result_fig1.get("stage1_config", {}),
        result_main.get("stage1_config", {}),
    )
    diagnostics = {
        "config_diff": config_diff,
        "main": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in main_metrics.items()
        },
        "fig1": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in fig_metrics.items()
        },
    }
    message = json.dumps(diagnostics, indent=2)
    assert np.allclose(fig_metrics["p_hat"], main_metrics["p_hat"], atol=1e-10), message
    for key in (
        "delta_t",
        "raw_objective_final",
        "y_nmse",
        "position_error_m",
    ):
        assert np.isclose(fig_metrics[key], main_metrics[key], rtol=1e-10, atol=1e-12), message
    for key in ("global_vp_mode", "jones_mode", "adaptive_enabled"):
        assert fig_metrics[key] == main_metrics[key], message
    assert result_fig1["variant_diagnostics"]["used_main_single_proposed_path"]

    no_rescue_spec = figures._variant_specs(figures.FIG1_FIG2_SHARED_FIGURE)[
        "adaptive_jones_no_rescue"
    ]
    no_rescue_config = figures.apply_nested_update(
        copy.deepcopy(config), no_rescue_spec
    )
    result_no_rescue = figures.run_final_vp_from_shared_stage1(
        copy.deepcopy(data),
        copy.deepcopy(stage1),
        no_rescue_config,
        no_rescue_spec,
        allow_stage2=False,
    )
    proposed_direct = result_fig1["branches"]["direct_vp"]["final"]
    no_rescue_direct = result_no_rescue["branches"]["direct_vp"]["final"]
    np.testing.assert_allclose(
        proposed_direct["p_u"], no_rescue_direct["p_u"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        proposed_direct["delta_t"],
        no_rescue_direct["delta_t"],
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        proposed_direct["raw_objective_final"],
        no_rescue_direct["raw_objective_final"],
        rtol=1e-12,
        atol=1e-12,
    )
    for coefficient_key in ("x_hat", "beta_raw"):
        if proposed_direct.get(coefficient_key) is not None:
            np.testing.assert_allclose(
                proposed_direct[coefficient_key],
                no_rescue_direct[coefficient_key],
                rtol=1e-12,
                atol=1e-12,
            )


def test_ngc_rejects_position_boundary_candidate(monkeypatch):
    from src import main_single_proposed as proposed

    monkeypatch.setattr(
        proposed,
        "_stage1_clock_panel_order",
        lambda *args: (np.zeros(3), np.ones(3), True, [0, 1, 2]),
    )
    monkeypatch.setattr(
        proposed,
        "_ngc_clock_sigmas_s",
        lambda *args, **kwargs: (np.ones(3), "test"),
    )
    monkeypatch.setattr(
        proposed,
        "robust_jnpp_geometry_consistency_score",
        lambda *args, **kwargs: {
            "available": True,
            "score": 0.0,
            "score_norm": 0.0,
        },
    )
    config = default_config()
    branch = {
        "final": {
            "p_u": np.array([1.25, 0.55, config["ue_bounds"][2, 1]]),
            "global_vp_init_selected_candidate": "all_panel_mean",
            "global_vp_init_candidate_scores": [
                {
                    "name": "all_panel_mean",
                    "p_u": np.array([1.25, 0.55, 0.75]),
                }
            ],
        }
    }
    scene = {
        "K": 3,
        "c0": config["c0"],
        "ris_centers": np.zeros((3, 3)),
        "d_RB": np.zeros(3),
    }
    diagnostics = proposed._ngc_certificate(
        "direct", branch, {}, scene, config
    )
    assert diagnostics["ngc_direct_cert_status"] == "red"
    assert diagnostics["ngc_direct_position_boundary_hit"] is True
    assert diagnostics["ngc_direct_position_boundary_axis"] == "z"
    assert "position_boundary_hit" in diagnostics["ngc_direct_cert_reason"]
    assert np.isclose(diagnostics["ngc_direct_stage1_displacement_m"], 0.7)


def test_fig1_grouped_execution_does_not_reuse_final_result(monkeypatch):
    data = {
        "scene": {
            "K": 3,
            "receiver_mode": "full_6d",
            "p_u_true": np.zeros(3),
        },
        "Y_true": np.zeros((1, 1, 1), dtype=complex),
        "Y_noisy": np.zeros((1, 1, 1), dtype=complex),
        "Z_noisy": np.zeros((1, 1, 1, 1), dtype=complex),
        "true_components": {},
        "timing": {},
        "noise_variance": 1.0,
    }
    shared_result = {
        **data,
        "final": {
            "p_u": np.zeros(3),
            "Y_hat": np.zeros((1, 1, 1), dtype=complex),
            "raw_objective_final": 0.0,
            "global_vp_mode": "adaptive_jones",
        },
        "timing": {},
    }
    monkeypatch.setattr(figures, "_make_data", lambda config: copy.deepcopy(data))
    monkeypatch.setattr(
        figures,
        "run_stage1_only",
        lambda data, config: {"estimate": {}, "timing": {}},
    )
    monkeypatch.setattr(
        figures,
        "run_final_vp_from_shared_stage1",
        lambda *args, **kwargs: shared_result,
    )
    task = {
        "task_kind": "grouped",
        "figure": figures.FIG1_FIG2_SHARED_FIGURE,
        "group": "fig1_fig2",
        "trial_id": 0,
        "trial_seed": 123,
        "snr_db": 0.0,
        "x_name": "snr_db",
        "x_value": 0.0,
        "K": 3,
        "paper_k": 3,
        "outlier_threshold_m": 0.1,
        "store_large_arrays": False,
        "profile_memory": False,
        "blas_threads": 1,
        "out_dir": "",
        "validation_variants": ["fixed_pol_vp", "free_jones_vp"],
    }
    with pytest.raises(figures.GroupedResultReuseError):
        figures._run_grouped_task(task)


def test_fig1_adaptive_row_has_lambda_and_mode_diagnostics():
    config = _smoke_config()
    data = _make_data(config)
    stage1 = run_stage1_only(copy.deepcopy(data), copy.deepcopy(config))
    result = figures.run_fig1_adaptive_from_shared(data, stage1, config)
    row = figures.extract_metrics(result, 0.1)
    assert row["global_vp_mode"] == "adaptive_jones"
    assert row["adaptive_enabled"] is True
    assert row["adaptive_policy_name"]
    assert np.isfinite(row["lambda_path_min"])
    assert np.isfinite(row["lambda_path_max"])
    assert np.isfinite(row["lambda_path_mean"])
    assert row["final_runner_name"] == "main_single_proposed"


def test_duplicate_curve_report_written_for_exact_overlap(tmp_path):
    summary = [
        {"variant": "fixed_pol_vp", "x_value": 0.0, "plot_y_mean": 1.0},
        {
            "variant": "adaptive_jones_vp_proposed",
            "x_value": 0.0,
            "plot_y_mean": 1.0,
        },
    ]
    trials = [
        {
            "variant": "adaptive_jones_vp_proposed",
            "selected_vp_family_branch": "fixed_pol_anchor",
            "lambda_path_min": 1.0,
            "lambda_path_max": 2.0,
            "lambda_path_mean": 1.5,
            "lambda_clipped_fraction": 0.0,
        }
    ]
    report = figures._write_duplicate_curves_report(
        "fig1", summary, trials, tmp_path
    )
    assert report["exact_duplicate_variants"] == ["fixed_pol_vp"]
    assert (tmp_path / "duplicate_curves_report.json").exists()


def test_proposed_plotted_last(tmp_path, monkeypatch):
    import matplotlib.axes
    import matplotlib.figure

    labels = []
    original_plot = matplotlib.axes.Axes.plot

    def recording_plot(self, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *args, **kwargs: None)
    figures._plot_figure(
        "fig1",
        [
            {"variant": "adaptive_jones_vp_proposed", "x_value": 0.0, "plot_y_mean": 1.0},
            {"variant": "fixed_pol_vp", "x_value": 0.0, "plot_y_mean": 1.0},
        ],
        tmp_path,
    )
    assert labels[-1] == "Proposed NGC–LG-RDC"
