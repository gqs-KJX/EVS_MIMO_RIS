import argparse
import csv
import math

import numpy as np

from src.experiments import run_paper_ablation_figures as figures


def test_cli_memory_safe_defaults():
    args = figures.parse_args([])
    assert args.jobs == 10
    assert args.max_workers == 10
    assert args.maxtasksperchild == 1
    assert args.streaming_csv is True
    assert args.store_large_arrays is False
    assert args.blas_threads == 1


def test_streaming_csv_writer_writes_rows_immediately(tmp_path):
    final_path = tmp_path / "fig1_trials.csv"
    with figures.StreamingCsvWriter(final_path, ["figure", "value"]) as writer:
        writer.writerow({"figure": "fig1", "value": 1})
        tmp_path_csv = tmp_path / "fig1_trials.csv.tmp"
        assert tmp_path_csv.exists()
        with tmp_path_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows == [{"figure": "fig1", "value": "1"}]
        assert not final_path.exists()
    assert final_path.exists()
    assert not (tmp_path / "fig1_trials.csv.tmp").exists()


def test_compact_experiment_result_removes_large_arrays_and_keeps_scalars():
    result = {
        "Y_true": np.ones((2, 2)),
        "metric": 3.0,
        "nested": {
            "small_vector": np.arange(3),
            "huge": np.ones(12),
            "label": "direct_vp",
        },
    }
    compacted = figures.compact_experiment_result(
        result,
        keep_large_arrays=False,
        large_array_threshold=10,
    )
    assert compacted["Y_true"] is None
    assert compacted["metric"] == 3.0
    assert np.array_equal(compacted["nested"]["small_vector"], np.arange(3))
    assert compacted["nested"]["huge"] is None
    assert compacted["nested"]["label"] == "direct_vp"


def _shared_test_rows():
    rows = []
    for variant in figures._expected_variant_names(figures.FIG1_FIG2_SHARED_FIGURE):
        row = {field: "" for field in figures.FIELDNAMES}
        row.update(
            {
                "figure": figures.FIG1_FIG2_SHARED_FIGURE,
                "variant": variant,
                "trial_id": "0",
                "seed": "1",
                "snr_db": "-30.0",
                "x_name": "snr_db",
                "x_value": "-30.0",
                "K": "3",
                "failed": "False",
                "position_rmse_m": "1.0" if variant != "PEB" else "",
                "y_nmse": "0.1" if variant != "PEB" else "",
                "peb_position_m": "2.0" if variant == "PEB" else "",
            }
        )
        rows.append(row)
    return rows


def test_fig2_reuses_shared_trials_when_compatible(tmp_path, monkeypatch):
    args = figures.parse_args(
        [
            "--figures",
            "fig2",
            "--n-trials",
            "1",
            "--snr-grid",
            "-30",
            "--out-dir",
            str(tmp_path),
            "--no-plots",
        ]
    )
    args.force_rerun = False
    args.reuse_existing = False
    rows = _shared_test_rows()
    figures._write_rows_atomic_csv(
        tmp_path / figures.FIG1_FIG2_SHARED_TRIAL_CSV,
        rows,
        figures.FIELDNAMES,
    )
    monkeypatch.setattr(
        figures,
        "_can_reuse_csv",
        lambda path, figure, args, snr_grid, figures_arg, metadata: (
            True,
            rows,
        )
        if figure == figures.FIG1_FIG2_SHARED_FIGURE
        else (False, []),
    )

    out_rows = figures._run_figure(
        "fig2",
        args=args,
        snr_grid=[-30.0],
        trial_seeds=[1],
        figures=["fig2"],
        existing_metadata={},
        completed_figures=set(),
    )

    assert {row["variant"] for row in out_rows} == set(
        figures._expected_variant_names(figures.FIG1_FIG2_SHARED_FIGURE)
    )
    fig2_summary = figures.summarize_rows(out_rows, "fig2")
    assert "PEB" not in {row["variant"] for row in fig2_summary}
    assert (tmp_path / "fig2_trials.csv").exists()
    assert (tmp_path / figures.FIG1_FIG2_SHARED_SUMMARY_CSV).exists()


def test_fig1_fig2_together_generate_shared_trials_once(tmp_path, monkeypatch):
    calls = {"write_trial_results": 0}
    rows = _shared_test_rows()

    def fake_write_trial_results(trial_csv, tasks, log_path, args):
        calls["write_trial_results"] += 1
        figures._write_rows_atomic_csv(trial_csv, rows, figures.FIELDNAMES)
        return rows

    monkeypatch.setattr(figures, "_write_trial_results", fake_write_trial_results)
    monkeypatch.setattr(figures, "_plot_figure", lambda *args, **kwargs: None)

    figures.main(
        [
            "--figures",
            "fig1,fig2",
            "--n-trials",
            "1",
            "--snr-grid",
            "-30",
            "--out-dir",
            str(tmp_path),
            "--force-rerun",
            "--no-plots",
        ]
    )

    assert calls["write_trial_results"] == 1
    assert (tmp_path / figures.FIG1_FIG2_SHARED_TRIAL_CSV).exists()
    assert (tmp_path / figures.FIG1_FIG2_SHARED_SUMMARY_CSV).exists()


def test_fig1_fig2_cache_filename_is_shared(tmp_path):
    assert figures._fig1_fig2_shared_trial_csv(tmp_path) == (
        tmp_path / figures.FIG1_FIG2_SHARED_TRIAL_CSV
    )


def test_fig1_fig2_shared_plot_metric_mapping():
    rows = _shared_test_rows()
    fig1_summary = figures.summarize_rows(rows, "fig1")
    fig2_summary = figures.summarize_rows(rows, "fig2")
    fig1_peb = [row for row in fig1_summary if row["variant"] == "PEB"]
    assert fig1_peb and fig1_peb[0]["plot_metric_name"] == "peb_position_m"
    assert "PEB" not in {row["variant"] for row in fig2_summary}


def test_peb_cache_key_is_deterministic():
    config = figures._config_for_point(
        figure="fig1",
        variant_updates={"receiver_mode": "full_6d"},
        seed=123,
        snr_db=-30.0,
        x_value=-30.0,
        paper_k=3,
    )
    first = figures.peb_cache_key(config)
    second = figures.peb_cache_key(config)
    assert first == second
    assert not any(isinstance(item, float) and math.isnan(item) for item in first)
