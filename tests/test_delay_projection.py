import numpy as np

from src.projections_delay import (
    _hankel_antidiagonal_counts,
    _hankel_shape,
    delay_matrix_from_poles,
    project_delay_mother_hankel_rank_one,
)


def test_es_cpd_hankel_shape_rule():
    assert _hankel_shape(6) == (3, 4)
    assert _hankel_shape(7) == (4, 4)


def test_es_cpd_inverse_weight_counts():
    np.testing.assert_allclose(
        _hankel_antidiagonal_counts(6),
        np.array([1, 2, 3, 3, 2, 1], dtype=float),
    )
    np.testing.assert_allclose(
        _hankel_antidiagonal_counts(7),
        np.array([1, 2, 3, 4, 3, 2, 1], dtype=float),
    )


def test_hankel_rank_one_projection_preserves_exact_vandermonde():
    poles = np.array([np.exp(1j * 0.37), np.exp(-1j * 0.91)])
    delay_mother = delay_matrix_from_poles(poles, 7)
    projected = project_delay_mother_hankel_rank_one(delay_mother)
    np.testing.assert_allclose(projected, delay_mother, atol=1e-10, rtol=1e-10)
