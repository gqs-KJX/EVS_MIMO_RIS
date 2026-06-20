import math
import sqlite3

import numpy as np
import pytest

from src.config import default_config
from src.experiments import run_paper_ablation_figures as figures


def _fake_data(config):
    return {
        "scene": {
            "K": int(config["K"]),
            "receiver_mode": config.get("receiver_mode", "full_6d"),
            "p_u_true": np.zeros(3),
        },
        "Y_true": np.zeros((1, 1, 1), dtype=complex),
        "Y_noisy": np.zeros((1, 1, 1), dtype=complex),
        "true_components": {
            "ranges": np.zeros(int(config["K"])),
            "taus": np.zeros(int(config["K"])),
        },
        "timing": {"data_generation": 0.0},
        "noise_variance": 1.0,
    }


def _fake_result(config, selected_branch="direct_vp"):
    k_paths = int(config["K"])
    return {
        **_fake_data(config),
        "final": {
            "Y_hat": np.zeros((1, 1, 1), dtype=complex),
            "p_u": np.zeros(3),
            "components": {"ranges": np.zeros(k_paths), "taus": np.zeros(k_paths)},
            "raw_objective_final": 0.0,
            "selected_branch": selected_branch,
            "final_refinement_method": "global_exact_spherical_vp",
            "global_vp_mode": config.get("global_vp", {}).get("mode", "adaptive_jones"),
        },
        "timing": {"stage1": 0.0, "vp": 0.0, "total": 0.0},
        "reliability": {"decision": selected_branch, "trigger_reasons": []},
        "selected_branch": selected_branch,
    }


def _install_fast_grouped_fakes(monkeypatch):
    monkeypatch.setattr(figures, "_make_data", lambda config: _fake_data(config))
    monkeypatch.setattr(
        figures,
        "run_stage1_only",
        lambda data, config: {
            "estimate": {"poles": np.zeros(int(config["K"]))},
            "timing": {"stage1": 0.0},
        },
    )
    monkeypatch.setattr(
        figures,
        "_stage1_only_result_from_shared",
        lambda data, stage1, config: _fake_result(config, "stage1_only"),
    )
    monkeypatch.setattr(
        figures,
        "run_final_vp_from_shared_stage1",
        lambda data, stage1, config, variant_spec, allow_stage2: _fake_result(
            config, config.get("proposed_stage2_policy", "direct_vp")
        ),
    )
    monkeypatch.setattr(
        figures,
        "_oracle_result_from_shared",
        lambda data, config: _fake_result(config, "oracle_init_vp"),
    )
    monkeypatch.setattr(
        figures,
        "_peb_metrics_result_for_config",
        lambda config, out_dir, data=None: {
            "scene": {"K": int(config["K"]), "receiver_mode": config.get("receiver_mode", "full_6d")},
            "peb_position_m": 0.5,
            "peb_scalar_m": math.nan,
            "peb_dual_m": math.nan,
            "peb_evs_m": 0.5,
            "warning": "",
            "final": {},
            "timing": {},
        },
    )


def _group_task(group, figure=None):
    return {
        "task_kind": "grouped",
        "figure": figure or group,
        "group": group,
        "trial_id": 0,
        "trial_seed": 123,
        "snr_db": -30.0,
        "x_name": "snr_db",
        "x_value": -30.0,
        "K": 3,
        "paper_k": 3,
        "outlier_threshold_m": 0.1,
        "store_large_arrays": False,
        "profile_memory": False,
        "blas_threads": 1,
        "out_dir": "",
    }


def test_cli_default_jobs_is_ten():
    assert figures.parse_args([]).jobs == 10


def test_process_workers_and_max_workers_conflict_is_rejected():
    with pytest.raises(ValueError):
        figures.parse_args(["--process-workers", "10", "--max-workers", "30"])


def test_grouped_fig1_task_returns_vp_family_rows(monkeypatch):
    _install_fast_grouped_fakes(monkeypatch)
    rows, _ = figures._run_grouped_task(_group_task("fig1_fig2", figures.FIG1_FIG2_SHARED_FIGURE))
    variants = {row["variant"] for row in rows}
    assert {
        "stage1_only",
        "fixed_pol_vp",
        "free_jones_vp",
        "regularized_jones_vp",
        "adaptive_jones_vp_proposed",
    }.issubset(variants)
    assert {row["effective_K"] for row in rows} == {3}
    assert {row["paper_k"] for row in rows} == {3}


def test_grouped_fig6_rows_use_x_value_as_effective_k(monkeypatch):
    _install_fast_grouped_fakes(monkeypatch)
    task = _group_task("fig6")
    task.update({"x_name": "K", "x_value": 4.0, "K": 4})
    rows, _ = figures._run_grouped_task(task)
    assert rows
    assert {row["effective_K"] for row in rows} == {4}
    assert {row["paper_k"] for row in rows} == {3}


def test_grouped_fig5_task_returns_all_variants(monkeypatch):
    _install_fast_grouped_fakes(monkeypatch)
    rows, _ = figures._run_grouped_task(_group_task("fig5"))
    assert {row["variant"] for row in rows} == {
        "direct_vp",
        "jnpp_always",
        "reliability_gated_proposed",
        "oracle_init_vp",
    }


def test_worker_rows_do_not_contain_ndarrays(monkeypatch):
    _install_fast_grouped_fakes(monkeypatch)
    rows, _ = figures._run_grouped_task(_group_task("fig5"))
    assert rows
    assert all(not isinstance(value, np.ndarray) for row in rows for value in row.values())


def test_persistent_peb_cache_second_lookup_avoids_recompute(tmp_path, monkeypatch):
    config = default_config()
    config["K"] = 1
    config["receiver_mode"] = "full_6d"
    config["SNR_dB"] = -30.0
    calls = {"compute": 0}

    def fake_compute(data, config):
        calls["compute"] += 1
        return {
            "peb_position_m": 1.25,
            "peb_scalar_m": math.nan,
            "peb_dual_m": math.nan,
            "peb_evs_m": 1.25,
            "warning": "",
        }

    monkeypatch.setattr(figures, "_make_data", lambda config: _fake_data(config))
    monkeypatch.setattr(figures, "_peb_from_efim", fake_compute)
    figures._PEB_CACHE.clear()
    first = figures._peb_result(config, tmp_path)
    figures._PEB_CACHE.clear()
    second = figures._peb_result(config, tmp_path)
    assert first["peb_position_m"] == second["peb_position_m"] == 1.25
    assert calls["compute"] == 1


def test_peb_cache_init_is_safe_when_called_twice(tmp_path):
    db_path = tmp_path / ".cache" / "peb_cache.sqlite"
    figures._init_peb_cache(db_path)
    figures._init_peb_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='peb_cache'"
        ).fetchone()
    assert journal_mode.lower() == "wal"
    assert table == ("peb_cache",)


def test_worker_init_does_not_force_wal_concurrently(tmp_path, monkeypatch):
    monkeypatch.setattr(
        figures,
        "_init_peb_cache",
        lambda path: (_ for _ in ()).throw(
            AssertionError("worker must not initialize WAL")
        ),
    )
    figures._init_worker(
        default_config(),
        str(tmp_path),
        1,
        peb_cache_enabled=True,
    )
    assert figures._WORKER_OUT_DIR == tmp_path
    assert figures._peb_cache_is_enabled(figures._peb_cache_path(tmp_path))


def test_locked_peb_cache_falls_back_without_crashing(
    tmp_path, monkeypatch, capsys
):
    db_path = figures._peb_cache_path(tmp_path)
    figures._PEB_CACHE_DISABLED_PATHS.discard(
        figures._peb_cache_path_key(db_path)
    )
    monkeypatch.setattr(
        figures,
        "_init_peb_cache",
        lambda path: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
    )
    assert not figures._prepare_peb_cache(tmp_path)
    assert not figures._peb_cache_is_enabled(db_path)

    config = default_config()
    config["K"] = 1
    config["receiver_mode"] = "full_6d"
    config["SNR_dB"] = -30.0
    figures._PEB_CACHE.clear()
    monkeypatch.setattr(figures, "_make_data", lambda config: _fake_data(config))
    monkeypatch.setattr(
        figures,
        "_peb_from_efim",
        lambda data, config: {
            "peb_position_m": 1.0,
            "peb_scalar_m": math.nan,
            "peb_dual_m": math.nan,
            "peb_evs_m": 1.0,
            "warning": "",
        },
    )
    result = figures._peb_result(config, tmp_path)
    assert result["peb_position_m"] == 1.0
    assert "disabling persistent PEB cache" in capsys.readouterr().err


def test_variant_mode_tasks_still_work(tmp_path, monkeypatch):
    args = figures.parse_args(
        [
            "--figures",
            "fig1",
            "--task-grouping",
            "variant",
            "--out-dir",
            str(tmp_path),
        ]
    )
    variants = {"stage1_only": figures._variant_specs("fig1")["stage1_only"]}
    tasks = figures._tasks_for_figure(
        figure=figures.FIG1_FIG2_SHARED_FIGURE,
        grouped_group="fig1_fig2",
        x_name="snr_db",
        x_values=[-30.0],
        variants=variants,
        trial_seeds=[123],
        args=args,
    )
    assert len(tasks) == 1
    assert tasks[0].get("task_kind") != "grouped"
