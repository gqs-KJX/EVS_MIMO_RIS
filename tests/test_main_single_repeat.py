import csv
import json

import numpy as np

from src import main_single_proposed as main_single


class _SynchronousPool:
    def __init__(self, processes, initializer=None, initargs=()):
        self.processes = processes
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, function, tasks, chunksize=1):
        del chunksize
        for task in tasks:
            yield function(task)


def _fake_result(config):
    p_true = np.array([1.0, 2.0, 3.0])
    p_hat = p_true + np.array([0.1, -0.2, 0.3])
    y_true = np.ones((2, 2, 2), dtype=complex)
    y_hat = 0.9 * y_true
    return {
        "scene": {
            "p_u_true": p_true,
            "delta_t_true": 5.0e-9,
            "c0": 3.0e8,
        },
        "Y_true": y_true,
        "Y_noisy": y_true,
        "noise_variance": 0.01,
        "estimate_initial": {"p_u": p_true + 0.5},
        "estimate_used": {"p_u": p_true + 0.25},
        "stage1_config": config,
        "selected_branch": "direct_vp",
        "final": {
            "p_u": p_hat,
            "delta_t": 5.2e-9,
            "Y_hat": y_hat,
            "raw_objective_final": 0.02,
            "global_vp_mode": "adaptive_jones",
            "jones_mode": "adaptive_jones",
        },
    }


def _install_synchronous_repeat(monkeypatch):
    monkeypatch.setattr(main_single.mp, "Pool", _SynchronousPool)
    monkeypatch.setattr(
        main_single,
        "run_single_proposed_diagnostic",
        lambda config: _fake_result(config),
    )


def test_repeat_runs_one_preserves_single_run_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main_single,
        "run_default_diagnostic",
        lambda config: calls.append(config),
    )
    main_single.main(["--repeat-runs", "1"])
    assert len(calls) == 1


def test_run_repeated_writes_all_outputs(tmp_path, monkeypatch):
    _install_synchronous_repeat(monkeypatch)
    result = main_single.run_repeated_main_single(
        n_runs=2,
        jobs=1,
        base_seed=123,
        out_dir=tmp_path,
    )
    assert len(result["rows"]) == 2
    for name in (
        "main_single_repeat_trials.csv",
        "main_single_repeat_summary.csv",
        "main_single_repeat_metadata.json",
        "main_single_repeat.log",
    ):
        assert (tmp_path / name).exists()
    with (tmp_path / "main_single_repeat_trials.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_repeat_summary_rmse_is_correct():
    rows = [
        {
            "failed": False,
            "position_error_m": 5.0,
            "err_x_m": 3.0,
            "err_y_m": 4.0,
            "err_z_m": 0.0,
            "clock_error_ns": 2.0,
            "clock_range_error_m": 0.6,
            "y_nmse": 0.1,
            "raw_objective_final": 1.0,
        },
        {
            "failed": False,
            "position_error_m": 10.0,
            "err_x_m": 6.0,
            "err_y_m": 8.0,
            "err_z_m": 0.0,
            "clock_error_ns": 4.0,
            "clock_range_error_m": 1.2,
            "y_nmse": 0.3,
            "raw_objective_final": 3.0,
        },
    ]
    summary = main_single.summarize_repeated_main_single(rows)
    assert np.isclose(summary["position_rmse_m"], np.sqrt(62.5))
    assert np.isclose(summary["err_x_rmse_m"], np.sqrt(22.5))
    assert np.isclose(summary["err_y_rmse_m"], np.sqrt(40.0))
    assert np.isclose(summary["clock_error_ns_rmse"], np.sqrt(10.0))
    assert np.isclose(summary["y_nmse_mean"], 0.2)


def test_failed_trials_are_recorded_in_success_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(main_single.mp, "Pool", _SynchronousPool)

    def fake_worker(task):
        row = main_single._empty_repeat_trial_row(
            task["trial_id"], task["seed"], main_single.default_config()
        )
        if task["trial_id"] == 1:
            row.update({"failed": True, "error": "RuntimeError: expected"})
        else:
            row.update({"position_error_m": 1.0, "runtime_s": 0.1})
        return row

    monkeypatch.setattr(main_single, "_run_repeat_trial", fake_worker)
    result = main_single.run_repeated_main_single(2, 1, 123, tmp_path)
    assert result["summary"]["n_success"] == 1
    assert result["summary"]["n_failed"] == 1
    assert result["summary"]["success_rate"] == 0.5
    assert result["rows"][1]["error"] == "RuntimeError: expected"


def test_worker_return_row_has_no_ndarrays(monkeypatch):
    monkeypatch.setattr(
        main_single,
        "run_single_proposed_diagnostic",
        lambda config: _fake_result(config),
    )
    row = main_single._run_repeat_trial(
        {"trial_id": 0, "seed": 123, "config_overrides": {}}
    )
    assert not row["failed"]
    assert all(not isinstance(value, np.ndarray) for value in row.values())


def test_repeat_seeds_are_deterministic_and_distinct(tmp_path, monkeypatch):
    _install_synchronous_repeat(monkeypatch)
    first = main_single.run_repeated_main_single(3, 1, 456, tmp_path / "first")
    second = main_single.run_repeated_main_single(3, 1, 456, tmp_path / "second")
    first_seeds = [row["seed"] for row in first["rows"]]
    second_seeds = [row["seed"] for row in second["rows"]]
    assert first_seeds == second_seeds
    assert len(set(first_seeds)) == 3
    metadata = json.loads(
        (tmp_path / "first" / "main_single_repeat_metadata.json").read_text()
    )
    assert metadata["trial_seeds"] == first_seeds
