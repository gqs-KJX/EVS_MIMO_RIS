import json

from src.experiments import monitor_progress
from src.experiments import run_benchmark_comparison as benchmark
from src.experiments.progress_logger import ProgressLogger


def test_progress_logger_writes_valid_jsonl(tmp_path):
    path = tmp_path / "progress.jsonl"
    logger = ProgressLogger(path, total_tasks=2, script_name="unit_test")
    logger.log("start", "running")
    logger.log(
        "task_done",
        "completed",
        figure="fig7",
        baseline_or_variant="peb",
        trial_id=0,
    )
    logger.log("finished", "completed")
    logger.close()

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "start",
        "task_done",
        "finished",
    ]
    assert events[-1]["done_tasks"] == events[-1]["total_tasks"] == 2
    assert events[-1]["percent"] == 100.0


def test_monitor_prints_latest_event_and_error(tmp_path, capsys):
    path = tmp_path / "progress.jsonl"
    logger = ProgressLogger(path, total_tasks=1, script_name="unit_test")
    logger.log("start", "running")
    logger.log(
        "task_failed",
        "failed",
        figure="fig7",
        baseline_or_variant="ff_omp",
        error="synthetic failure",
    )
    logger.close()

    monitor_progress.print_progress(path)
    output = capsys.readouterr().out
    assert "Progress: 1/1 (100.0%)" in output
    assert "ff_omp" in output
    assert "Latest error: synthetic failure" in output


def test_benchmark_accepts_progress_log(tmp_path):
    path = tmp_path / "custom.jsonl"
    args = benchmark.parse_args(["--progress-log", str(path)])
    assert args.progress_log == path


def test_tiny_benchmark_writes_progress_events(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress.jsonl"

    def fake_task(task):
        return [
            {
                "baseline": "peb",
                "trial_id": task["trial_id"],
                "seed": task["seed"],
                "snr_db": task["snr_db"],
                "K": task["paper_k"],
                "y_noisy_hash": "shared",
                "failed": False,
            }
        ]

    monkeypatch.setattr(benchmark, "_run_trial_task", fake_task)
    benchmark.main(
        [
            "--n-trials",
            "1",
            "--snr-grid",
            "-20",
            "--paper-k",
            "3",
            "--baselines",
            "peb",
            "--out-dir",
            str(tmp_path),
            "--jobs",
            "1",
            "--process-workers",
            "1",
            "--no-plots",
            "--force-rerun",
            "--quiet-progress",
            "--progress-log",
            str(progress_path),
        ]
    )
    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "start",
        "task_done",
        "finished",
    ]
