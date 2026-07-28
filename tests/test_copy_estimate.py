"""Bulk reconstruction tensors are shared, everything else is deep-copied."""

from __future__ import annotations

import copy

import numpy as np

from src.utils import BULK_RECONSTRUCTION_KEYS, copy_estimate


def _estimate() -> dict:
    return {
        "A": np.arange(6, dtype=complex).reshape(3, 2),
        "poles": np.array([1.0 + 1.0j, 2.0]),
        "ris_eta": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
        "Z_hat": np.zeros((2, 2, 2, 2), dtype=complex),
        "nested": {"gamma": np.array([0.5, 0.25])},
        "scalar": 3,
        "flag": True,
    }


def test_bulk_key_is_shared_by_reference():
    original = _estimate()
    duplicate = copy_estimate(original)
    for key in BULK_RECONSTRUCTION_KEYS:
        assert duplicate[key] is original[key]


def test_every_other_entry_is_independent():
    original = _estimate()
    duplicate = copy_estimate(original)

    duplicate["A"][0, 0] = 99.0
    duplicate["poles"][0] = -1.0
    duplicate["ris_eta"][0][0] = -1.0
    duplicate["nested"]["gamma"][0] = -1.0

    assert original["A"][0, 0] == 0.0
    assert original["poles"][0] == 1.0 + 1.0j
    assert original["ris_eta"][0][0] == 1.0
    assert original["nested"]["gamma"][0] == 0.5


def test_contents_and_key_order_match_deepcopy():
    original = _estimate()
    duplicate = copy_estimate(original)
    reference = copy.deepcopy(original)

    assert list(duplicate) == list(reference) == list(original)
    for key in reference:
        left, right = duplicate[key], reference[key]
        if isinstance(right, np.ndarray):
            assert np.array_equal(left, right)
        elif isinstance(right, list):
            assert all(np.array_equal(a, b) for a, b in zip(left, right))
        elif isinstance(right, dict):
            assert all(np.array_equal(left[k], right[k]) for k in right)
        else:
            assert left == right


def test_missing_bulk_key_is_tolerated():
    original = {"A": np.zeros(3)}
    duplicate = copy_estimate(original)
    assert "Z_hat" not in duplicate
    duplicate["A"][0] = 1.0
    assert original["A"][0] == 0.0


def test_non_dict_falls_back_to_deepcopy():
    original = [np.zeros(2)]
    duplicate = copy_estimate(original)
    duplicate[0][0] = 1.0
    assert original[0][0] == 0.0
