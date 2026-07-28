"""The factorized Khatri-Rao variable projection must equal the dense design solve.

The Stage-I CP weight estimate and the Stage-II global VP residual both solve a
ridge least-squares problem whose design matrix has one rank-one Kronecker
column per path.  The factorized path never materializes that design; these
tests pin it to the legacy dense path it replaced, since the two are the same
minimizer and any drift would silently change reported objectives.
"""

import numpy as np

from src.estimators import (
    _estimate_weights_raw,
    _estimate_weights_z,
    _raw_design_matrix_from_factors,
)
from src.tensor_utils import (
    khatri_rao_gram,
    khatri_rao_synthesize,
    reconstruct_z,
    solve_khatri_rao_lstsq,
    z_design_column,
)
from src.utils import solve_lstsq

REG = 1.0e-12


def _complex_normal(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _factors(rng, dims, k_paths):
    return [_complex_normal(rng, (dim, k_paths)) for dim in dims]


def _dense_design(factors):
    k_paths = factors[0].shape[1]
    columns = []
    for k in range(k_paths):
        column = factors[0][:, k]
        for factor in factors[1:]:
            column = np.multiply.outer(column, factor[:, k])
        columns.append(column.reshape(-1))
    return np.column_stack(columns)


def test_gram_matches_dense_design():
    rng = np.random.default_rng(11)
    factors = _factors(rng, (5, 4, 3), 3)
    design = _dense_design(factors)
    assert np.allclose(khatri_rao_gram(factors), design.conj().T @ design, atol=1e-10)


def test_solve_matches_dense_ridge_lstsq():
    rng = np.random.default_rng(12)
    dims = (6, 5, 4, 3)
    factors = _factors(rng, dims, 3)
    tensor = _complex_normal(rng, dims)
    design = _dense_design(factors)

    expected = solve_lstsq(design, tensor.reshape(-1), reg=REG)
    actual = solve_khatri_rao_lstsq(factors, tensor, reg=REG)
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_solve_is_exact_for_a_noiseless_low_rank_tensor():
    rng = np.random.default_rng(13)
    dims = (7, 5, 4)
    factors = _factors(rng, dims, 3)
    weights = _complex_normal(rng, (3,))
    tensor = khatri_rao_synthesize(factors, weights)
    recovered = solve_khatri_rao_lstsq(factors, tensor, reg=REG)
    assert np.allclose(recovered, weights, rtol=1e-8, atol=1e-10)


def test_synthesize_matches_dense_design_product():
    rng = np.random.default_rng(14)
    dims = (5, 4, 3, 2)
    factors = _factors(rng, dims, 3)
    weights = _complex_normal(rng, (3,))
    expected = (_dense_design(factors) @ weights).reshape(dims)
    assert np.allclose(khatri_rao_synthesize(factors, weights), expected, atol=1e-12)


def test_solve_falls_back_when_paths_are_collinear():
    """A repeated column makes the Gram singular; the solve must stay finite."""
    rng = np.random.default_rng(15)
    dims = (6, 5, 4)
    factors = _factors(rng, dims, 3)
    for factor in factors:
        factor[:, 2] = factor[:, 1]
    tensor = _complex_normal(rng, dims)
    weights = solve_khatri_rao_lstsq(factors, tensor, reg=REG)
    assert np.all(np.isfinite(weights))
    design = _dense_design(factors)
    residual_fast = np.linalg.norm(design @ weights - tensor.reshape(-1))
    residual_dense = np.linalg.norm(
        design @ solve_lstsq(design, tensor.reshape(-1), reg=REG) - tensor.reshape(-1)
    )
    assert residual_fast <= residual_dense * (1.0 + 1e-6)


def test_estimate_weights_z_matches_legacy_column_stack():
    rng = np.random.default_rng(16)
    i_dim, p_dim, l_dim, t_dim, k_paths = 6, 4, 3, 5, 3
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    b_mat = _complex_normal(rng, (p_dim, k_paths))
    q_mat = _complex_normal(rng, (l_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    z_tensor = _complex_normal(rng, (i_dim, p_dim, l_dim, t_dim))

    legacy_design = np.column_stack(
        [
            z_design_column(a_mat[:, k], b_mat[:, k], q_mat[:, k], c_mat[:, k])
            for k in range(k_paths)
        ]
    )
    expected = solve_lstsq(legacy_design, z_tensor.reshape(-1), reg=REG)
    actual = _estimate_weights_z(z_tensor, a_mat, b_mat, q_mat, c_mat)
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_estimate_weights_raw_matches_legacy_design():
    rng = np.random.default_rng(17)
    i_dim, n_dim, t_dim, k_paths = 6, 5, 4, 3
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    d_mat = _complex_normal(rng, (n_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    y_tensor = _complex_normal(rng, (i_dim, n_dim, t_dim))

    design = _raw_design_matrix_from_factors(a_mat, d_mat, c_mat)
    expected = solve_lstsq(design, y_tensor.reshape(-1), reg=REG)
    actual = _estimate_weights_raw(y_tensor, a_mat, d_mat, c_mat)
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_reconstruct_z_matches_explicit_sum():
    rng = np.random.default_rng(18)
    i_dim, p_dim, l_dim, t_dim, k_paths = 5, 4, 3, 4, 3
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    b_mat = _complex_normal(rng, (p_dim, k_paths))
    q_mat = _complex_normal(rng, (l_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    beta = _complex_normal(rng, (k_paths,))

    expected = np.zeros((i_dim, p_dim, l_dim, t_dim), dtype=complex)
    for k in range(k_paths):
        expected += beta[k] * (
            a_mat[:, k, None, None, None]
            * b_mat[None, :, k, None, None]
            * q_mat[None, None, :, k, None]
            * c_mat[None, None, None, :, k]
        )
    assert np.allclose(reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat), expected, atol=1e-12)


def test_global_vp_residual_matches_dense_dictionary():
    """The Stage-II residual must equal the dense `dictionary @ beta - y` form."""
    from src.channel_model import generate_scene
    from src.config import default_config
    from src.estimators import (
        _dictionary_from_global_x,
        _initial_global_parameters,
        global_vp_residual,
    )

    config = default_config()
    config["T"] = 8
    config["N"] = 12
    config["P"] = 6
    config["M_A"] = 2
    config["M_R"] = 64
    scene = generate_scene(config, np.random.default_rng(19))
    k_paths = int(scene["K"])
    x = np.concatenate(
        [
            np.asarray(scene["p_u_true"], dtype=float),
            [float(scene["delta_t_true"])],
            np.full(k_paths, 0.7),
            np.full(k_paths, 0.3),
        ]
    )
    y_tensor = np.random.default_rng(20).normal(
        size=(int(scene["I"]), int(scene["N"]), int(scene["T"]))
    ).astype(complex)

    dictionary, _ = _dictionary_from_global_x(scene, x)
    y_vec = y_tensor.reshape(-1)
    beta_dense = solve_lstsq(dictionary, y_vec, reg=REG)
    residual_dense = dictionary @ beta_dense - y_vec

    residual, beta, _ = global_vp_residual(scene, x, y_tensor)
    assert np.allclose(beta, beta_dense, rtol=1e-9, atol=1e-12)
    assert np.allclose(residual, residual_dense, rtol=1e-9, atol=1e-12)
