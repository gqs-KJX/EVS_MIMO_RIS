import numpy as np

from src.estimators import _fit_z_model, _fit_z_model_fast_sse


def test_fit_z_model_fast_sse_matches_full_fit():
    rng = np.random.default_rng(123)
    i_dim, p_dim, l_dim, t_dim, k_paths = 3, 4, 5, 6, 2

    def complex_normal(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    z_tensor = complex_normal((i_dim, p_dim, l_dim, t_dim))
    a_mat = complex_normal((i_dim, k_paths))
    b_mat = complex_normal((p_dim, k_paths))
    q_mat = complex_normal((l_dim, k_paths))
    c_mat = complex_normal((t_dim, k_paths))

    beta_full, _, sse_full = _fit_z_model(z_tensor, a_mat, b_mat, q_mat, c_mat)
    beta_fast, sse_fast = _fit_z_model_fast_sse(z_tensor, a_mat, b_mat, q_mat, c_mat)

    np.testing.assert_allclose(beta_fast, beta_full, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(sse_fast, sse_full, atol=1e-9, rtol=1e-11)
