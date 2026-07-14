import copy

import numpy as np

from src.baselines import far_field_omp
from src.config import default_config
from src.experiments import run_benchmark_comparison as benchmark
from src.main_single_proposed import _make_data


def _tiny_config():
    config = default_config()
    config.update(
        {
            "seed": 123,
            "K": 1,
            "M_A": 1,
            "ris_shape": (2, 2),
            "N": 5,
            "P": 3,
            "T": 4,
            "SNR_dB": 60.0,
            "receiver_mode": "full_6d",
            "print_progress": False,
            "p_u_true": np.array([1.2, 0.4, 0.8]),
            "ris_centers": np.array([[4.2, -2.2, 1.05]]),
            "ue_bounds": np.array([[1.0, 1.4], [0.2, 0.6], [0.6, 1.0]]),
            "delta_t_true": 5.0e-9,
            "delta_t_bounds": np.array([4.0e-9, 6.0e-9]),
        }
    )
    config["baselines"] = {
        "ff_omp": {
            "angle_grid_size": 3,
            "delay_grid_size": 3,
            "max_atoms": 1,
            "batch_size": 1,
            "max_batch_memory_mb": 1.0,
        }
    }
    return config


def test_sparse_batching_matches_full_scoring_best_atom():
    config = _tiny_config()
    data = _make_data(config)
    scene = data["scene"]
    support = list(far_field_omp._far_field_supports(scene, config))[2]
    atom = far_field_omp.simple_atom_normalize(
        far_field_omp.raw_atom_from_support(scene, config, support)
    )
    data["Y_noisy"] = atom.reshape(scene["I"], scene["N"], scene["T"])
    data["Y_true"] = data["Y_noisy"].copy()

    batched = far_field_omp.run_far_field_omp_baseline(data, config)
    full_config = copy.deepcopy(config)
    full_config["baselines"]["ff_omp"]["batch_size"] = 10_000
    full = far_field_omp.run_far_field_omp_baseline(data, full_config)

    assert batched.selected_support[0]["direction_index"] == full.selected_support[0]["direction_index"]
    assert batched.selected_support[0]["tau_index"] == full.selected_support[0]["tau_index"]
    assert batched.diagnostics["num_batches"] > full.diagnostics["num_batches"]


def test_benchmark_pool_uses_process_workers(tmp_path, monkeypatch):
    calls = {}

    class FakePool:
        def __init__(self, processes, initializer=None, initargs=()):
            calls["processes"] = processes
            if initializer is not None:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, worker, tasks, chunksize=1):
            for task in tasks:
                yield [
                    {
                        "baseline": "peb",
                        "trial_id": task["trial_id"],
                        "seed": task["seed"],
                        "snr_db": task["snr_db"],
                        "K": task["paper_k"],
                        "y_noisy_hash": f"hash-{task['trial_id']}",
                        "failed": False,
                    }
                ]

    monkeypatch.setattr(benchmark.mp, "Pool", FakePool)
    benchmark.main(
        [
            "--n-trials",
            "3",
            "--snr-grid",
            "-20",
            "--baselines",
            "peb",
            "--jobs",
            "10",
            "--process-workers",
            "2",
            "--blas-threads",
            "auto",
            "--out-dir",
            str(tmp_path),
            "--force-rerun",
            "--no-plots",
        ]
    )
    assert calls["processes"] == 2


def test_one_benchmark_trial_uses_one_noisy_data_hash(monkeypatch):
    config = _tiny_config()
    data = _make_data(config)

    monkeypatch.setattr(benchmark, "make_config", lambda **kwargs: config)
    monkeypatch.setattr(benchmark, "_make_data", lambda config_arg: data)

    def row_for(name):
        return {
            "baseline": name,
            "trial_id": 0,
            "seed": config["seed"],
            "snr_db": config["SNR_dB"],
            "K": config["K"],
            "y_noisy_hash": benchmark.y_noisy_hash(data),
            "failed": False,
        }

    monkeypatch.setattr(benchmark, "_proposed_row", lambda *args: row_for("proposed"))
    monkeypatch.setattr(benchmark, "_peb_row", lambda *args: row_for("peb"))
    rows = benchmark._run_trial_task(
        {
            "trial_id": 0,
            "seed": config["seed"],
            "snr_db": config["SNR_dB"],
            "paper_k": 1,
            "baselines": ["proposed", "peb"],
            "grid_profile": "coarse",
            "blas_threads": 1,
            "strict_ris_geometry": False,
            "trim_memory": False,
        }
    )
    assert len({row["y_noisy_hash"] for row in rows}) == 1


def test_benchmark_groups_snr_by_trial_and_preserves_legacy_noisy_data():
    args = benchmark.parse_args(
        [
            "--n-trials",
            "2",
            "--snr-grid=-20,0",
            "--baselines",
            "peb",
        ]
    )
    tasks = benchmark._tasks(args, [-20.0, 0.0], ["peb"])
    assert len(tasks) == 2
    assert tasks[0]["snr_grid"] == [-20.0, 0.0]

    configs = []
    for snr_db in (-20.0, 0.0):
        config = _tiny_config()
        config["SNR_dB"] = snr_db
        configs.append(config)
    grouped = list(benchmark._iter_shared_scene_snr_data(configs))
    legacy = [_make_data(config) for config in configs]
    for grouped_data, legacy_data in zip(grouped, legacy):
        for key in ("Y_true", "Y_noisy", "Z_true", "Z_noisy"):
            assert np.array_equal(grouped_data[key], legacy_data[key])
        assert grouped_data["noise_variance"] == legacy_data["noise_variance"]
