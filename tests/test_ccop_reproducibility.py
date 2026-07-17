import numpy as np

from oldcode.ccop_validation.experiments.run_ccop_paired_mc import _run_trial
from src.validation_artifacts import canonical_hash, deterministic_stage1_output


def _task(artifact_dir):
    return {
        "trial_id": 0,
        "seed": 4191845062,
        "blas_threads": 1,
        "artifact_dir": str(artifact_dir),
        "environment": {
            "git_commit": "unit-test",
            "git_branch": "unit-test",
            "git_dirty": False,
            "worktree_fingerprint": "unit-test",
        },
        "spec": {
            "snr_db": -10.0,
            "diagnostic_mode": "fast",
            "outlier_threshold_m": 0.1,
            "jones_mode": "jones_regularized",
            "old_max_iter": 2,
            "ccop_outer_max_iter": 2,
            "clock_fft_size": 512,
            "clock_abs_tol": 1.0e-12,
            "clock_rel_tol": 1.0e-10,
            "clock_max_intervals": 2000,
            "use_old_incumbent": False,
            "old_vp_backend": "cpu",
            "gpu_device": 0,
        },
    }


def test_canonical_hash_is_independent_of_dictionary_insertion_order():
    left = {"b": np.array([1.0, 2.0]), "a": {"z": 3, "y": 4.0}}
    right = {"a": {"y": 4.0, "z": 3}, "b": np.array([1.0, 2.0])}
    assert canonical_hash(left) == canonical_hash(right)


def test_stage1_hash_excludes_all_top_level_runtime_buckets():
    first = {
        "p_u": np.array([1.0, 2.0, 3.0]),
        "stage1_time_delay_estimation": 0.1,
        "stage1_jones_anchor_refresh_runtime_s": 0.2,
    }
    second = {
        "p_u": np.array([1.0, 2.0, 3.0]),
        "stage1_time_delay_estimation": 9.1,
        "stage1_jones_anchor_refresh_runtime_s": 8.2,
    }
    assert canonical_hash(deterministic_stage1_output(first)) == canonical_hash(
        deterministic_stage1_output(second)
    )


def test_same_seed_config_and_single_thread_reproduce_all_pipeline_hashes(tmp_path):
    first = _run_trial(_task(tmp_path / "first"))
    second = _run_trial(_task(tmp_path / "second"))
    assert not first["difference_row"]["failed"]
    assert not second["difference_row"]["failed"]
    fields = (
        "resolved_config_hash",
        "y_noisy_hash",
        "stage1_input_hash",
        "stage1_output_hash",
        "old_candidate_hash",
        "ccop_candidate_hash",
        "artifact_payload_hash",
    )
    for field in fields:
        assert first["artifact_manifest"][field] == second["artifact_manifest"][field]
