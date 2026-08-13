import numpy as np

from src.validation_artifacts import canonical_hash, deterministic_stage1_output


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
