import copy
import json

import numpy as np
import pytest

from src.config import default_config
from src.experiments import run_paper_ablation_figures as figures


def test_help_includes_global_vp_backend(capsys):
    with pytest.raises(SystemExit):
        figures.parse_args(["--help"])
    assert "--global-vp-backend" in capsys.readouterr().out


def test_parse_args_accepts_global_vp_backend():
    args = figures.parse_args(
        ["--global-vp-backend", "cupy", "--global-vp-gpu-device", "0"]
    )
    assert args.global_vp_backend == "cupy"
    assert args.global_vp_gpu_device == 0


def test_global_vp_cli_override_sets_config_backend():
    args = figures.parse_args(
        ["--global-vp-backend", "cupy", "--global-vp-gpu-device", "0"]
    )
    config = default_config()
    figures.apply_global_vp_cli_overrides(config, args)
    assert config["global_vp"]["backend"] == "cupy"
    assert config["global_vp"]["gpu_device"] == 0
    assert config["global_vp"]["gpu_dtype"] == "complex128"
    assert config["global_vp"]["validate_gpu_against_cpu"] is False


def test_without_global_vp_backend_flag_leaves_config_unchanged():
    args = figures.parse_args([])
    config = default_config()
    before = copy.deepcopy(config["global_vp"])
    figures.apply_global_vp_cli_overrides(config, args)
    assert config["global_vp"] == before


def _fake_data(config):
    return {
        "scene": {
            "K": int(config["K"]),
            "receiver_mode": config.get("receiver_mode", "full_6d"),
            "p_u_true": np.zeros(3),
        },
        "Y_true": np.zeros((1, 1, 1), dtype=complex),
        "Y_noisy": np.zeros((1, 1, 1), dtype=complex),
        "true_components": {"ranges": np.zeros(int(config["K"]))},
        "timing": {},
        "noise_variance": 1.0,
    }


def _resolved_backend(config):
    requested = str(config.get("global_vp", {}).get("backend", "cpu"))
    return "cpu" if requested == "auto" else requested


def _fake_result(config):
    backend = _resolved_backend(config)
    gpu_used = backend == "cupy"
    return {
        **_fake_data(config),
        "final": {
            "Y_hat": np.zeros((1, 1, 1), dtype=complex),
            "p_u": np.zeros(3),
            "components": {
                "ranges": np.zeros(int(config["K"])),
                "taus": np.zeros(int(config["K"])),
            },
            "raw_objective_final": 0.0,
            "selected_branch": "direct_vp",
            "final_refinement_method": "global_exact_spherical_vp",
            "global_vp_mode": config.get("global_vp", {}).get("mode", ""),
            "global_vp_backend": backend,
            "global_vp_gpu_used": gpu_used,
            "global_vp_gpu_device": config.get("global_vp", {}).get("gpu_device", "")
            if gpu_used
            else "",
            "global_vp_objective_backend": "cupy" if gpu_used else "numpy",
            "global_vp_linear_solve_backend": "cupy.linalg.solve"
            if gpu_used
            else "numpy.linalg.solve",
        },
        "timing": {"stage1": 0.0, "vp": 0.0, "total": 0.0},
        "reliability": {"decision": "direct_vp", "trigger_reasons": []},
        "selected_branch": "direct_vp",
    }


def _install_fast_fig1_fakes(monkeypatch):
    monkeypatch.setattr(figures, "_make_data", lambda config: _fake_data(config))
    monkeypatch.setattr(
        figures,
        "run_stage1_only",
        lambda data, config: {"estimate": {}, "timing": {"stage1": 0.0}},
    )
    monkeypatch.setattr(
        figures,
        "run_final_vp_from_shared_stage1",
        lambda data, stage1, config, variant_spec, allow_stage2: _fake_result(config),
    )
    monkeypatch.setattr(
        figures,
        "_peb_metrics_result_for_config",
        lambda config, out_dir, data=None: {
            "scene": {
                "K": int(config["K"]),
                "receiver_mode": config.get("receiver_mode", "full_6d"),
            },
            "peb_position_m": 0.5,
            "peb_scalar_m": np.nan,
            "peb_dual_m": np.nan,
            "peb_evs_m": 0.5,
            "warning": "",
            "final": {},
            "timing": {},
        },
    )


def _run_tiny_fig1(tmp_path, monkeypatch, backend):
    _install_fast_fig1_fakes(monkeypatch)
    out_dir = tmp_path / f"fig1_{backend}"
    figures.main(
        [
            "--figures",
            "fig1",
            "--variant-filter",
            "free_jones_vp,PEB",
            "--n-trials",
            "1",
            "--paper-k",
            "1",
            "--snr-grid",
            "0",
            "--jobs",
            "1",
            "--process-workers",
            "1",
            "--blas-threads",
            "auto",
            "--global-vp-backend",
            backend,
            "--out-dir",
            str(out_dir),
            "--force-rerun",
            "--no-plots",
            "--quiet-progress",
        ]
    )
    return out_dir


def test_tiny_fig1_smoke_with_global_vp_backend_cpu(tmp_path, monkeypatch):
    out_dir = _run_tiny_fig1(tmp_path, monkeypatch, "cpu")
    rows = figures._read_csv(out_dir / figures.FIG1_FIG2_SHARED_TRIAL_CSV)
    vp_row = next(row for row in rows if row["variant"] == "free_jones_vp")
    assert vp_row["global_vp_backend"] == "cpu"
    assert vp_row["global_vp_objective_backend"] == "numpy"
    metadata = json.loads((out_dir / "experiment_metadata.json").read_text())
    assert metadata["global_vp_backend"] == "cpu"


def test_tiny_fig1_smoke_with_global_vp_backend_auto_falls_back(tmp_path, monkeypatch):
    out_dir = _run_tiny_fig1(tmp_path, monkeypatch, "auto")
    rows = figures._read_csv(out_dir / figures.FIG1_FIG2_SHARED_TRIAL_CSV)
    vp_row = next(row for row in rows if row["variant"] == "free_jones_vp")
    assert vp_row["global_vp_backend"] == "cpu"
    metadata = json.loads((out_dir / "experiment_metadata.json").read_text())
    assert metadata["global_vp_backend"] == "auto"
