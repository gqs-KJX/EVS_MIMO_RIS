import numpy as np

from src.projections_ris import (
    _block_dehankel_2d,
    _block_hankel_2d,
    _block_hankel_counts_2d,
    _hankel_counts_1d,
    _hankel_window,
)


def test_ris_hankel_window_balanced():
    assert _hankel_window(6) == (3, 4)
    assert _hankel_window(7) == (4, 4)


def test_ris_hankel_counts_1d():
    np.testing.assert_allclose(
        _hankel_counts_1d(6),
        np.array([1, 2, 3, 3, 2, 1], dtype=float),
    )
    np.testing.assert_allclose(
        _hankel_counts_1d(7),
        np.array([1, 2, 3, 4, 3, 2, 1], dtype=float),
    )


def test_block_hankel_counts_are_outer_product():
    mx, my = 6, 7
    px, lx = _hankel_window(mx)
    py, ly = _hankel_window(my)
    counts2d = _block_hankel_counts_2d(mx, my, px, lx, py, ly)
    expected = np.outer(_hankel_counts_1d(mx), _hankel_counts_1d(my))
    np.testing.assert_allclose(counts2d, expected)


def test_block_hankel_dehankel_identity():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(6, 7)) + 1j * rng.normal(size=(6, 7))
    px, lx = _hankel_window(6)
    py, ly = _hankel_window(7)
    lifted = _block_hankel_2d(matrix, px, lx, py, ly)
    recovered = _block_dehankel_2d(lifted, 6, 7, px, lx, py, ly)
    np.testing.assert_allclose(recovered, matrix, atol=1e-12, rtol=1e-12)
