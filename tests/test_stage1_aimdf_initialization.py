import itertools

import numpy as np

from src.channel_model import channel_components, generate_scene, synthesize_raw_tensor
from src.config import default_config
from src.estimators import _coupled_hankel_factor_initialization, initialize_from_hankel
from src.projections_delay import (
    bq_from_poles,
    estimate_poles_aimdf_tls_from_hankel,
)
from src.tensor_utils import hankelize_frequency, reconstruct_z


def _complex_normal(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _best_phase_error(estimated: np.ndarray, true: np.ndarray) -> float:
    best = np.inf
    for perm in itertools.permutations(range(true.size)):
        aligned = estimated[list(perm)]
        error = np.max(np.abs(np.angle(aligned / true)))
        best = min(best, float(error))
    return best


def _normalized_snapshot_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vec = left.reshape(-1)
    right_vec = right.reshape(-1)
    left_vec = left_vec / np.linalg.norm(left_vec)
    right_vec = right_vec / np.linalg.norm(right_vec)
    return float(abs(np.vdot(left_vec, right_vec)))


def test_aimdf_tls_recovers_noiseless_delay_poles_up_to_permutation():
    rng = np.random.default_rng(101)
    i_dim, p_dim, l_dim, t_dim, k_paths = 5, 5, 5, 7, 3
    poles = np.exp(1j * np.array([-0.91, 0.27, 1.13]))
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    beta = _complex_normal(rng, (k_paths,))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    z_tensor = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)

    estimated = estimate_poles_aimdf_tls_from_hankel(
        z_tensor, k_paths, forward_backward=True, tls=True
    )

    assert estimated.shape == (k_paths,)
    assert _best_phase_error(estimated, poles) < 1.0e-8


def test_coupled_hankel_factor_initialization_recovers_rank_one_snapshots():
    rng = np.random.default_rng(102)
    i_dim, p_dim, l_dim, t_dim, k_paths = 6, 5, 4, 8, 3
    poles = np.exp(1j * np.array([-0.74, 0.18, 0.96]))
    a_mat = _complex_normal(rng, (i_dim, k_paths))
    c_mat = _complex_normal(rng, (t_dim, k_paths))
    beta = _complex_normal(rng, (k_paths,))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    z_tensor = reconstruct_z(beta, a_mat, b_mat, q_mat, c_mat)

    a_proxy, c_proxy = _coupled_hankel_factor_initialization(
        z_tensor, poles, reg=1.0e-14
    )

    assert a_proxy.shape == (i_dim, k_paths)
    assert c_proxy.shape == (t_dim, k_paths)
    similarities = np.empty((k_paths, k_paths), dtype=float)
    for est_col in range(k_paths):
        estimated_snapshot = a_proxy[:, est_col, None] * c_proxy[None, :, est_col]
        for true_col in range(k_paths):
            true_snapshot = (
                beta[true_col] * a_mat[:, true_col, None] * c_mat[None, :, true_col]
            )
            similarities[est_col, true_col] = _normalized_snapshot_similarity(
                estimated_snapshot, true_snapshot
            )
    best = 0.0
    for perm in itertools.permutations(range(k_paths)):
        best = max(best, min(similarities[col, perm[col]] for col in range(k_paths)))
    assert best > 1.0 - 1.0e-8


def test_initialize_from_hankel_returns_expected_keys_and_rebuilds_bq_from_poles():
    rng = np.random.default_rng(103)
    config = default_config()
    config.update(
        {
            "K": 2,
            "M_A": 1,
            "ris_shape": (4, 4),
            "N": 9,
            "P": 5,
            "T": 12,
            "ris_centers": config["ris_centers"][:2].copy(),
            "stage1_delay_method": "aimdf_tls",
            "stage1_forward_backward": True,
            "stage1_tls": True,
            "stage1_factor_init": "hankel_coupled_ls",
            "stage1_factor_reg": 1.0e-12,
        }
    )
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_tensor = synthesize_raw_tensor(components, scene["beta_true"])
    z_tensor = hankelize_frequency(y_tensor, scene["P"])

    estimate = initialize_from_hankel(z_tensor, scene, config)

    expected = {
        "poles",
        "A",
        "B",
        "Q",
        "C",
        "beta_z",
        "gamma",
        "eta_pol",
        "ris_eta",
        "assignment",
        "initial_z_residual",
        "Z_hat",
    }
    assert expected.issubset(estimate.keys())
    b_expected, q_expected = bq_from_poles(estimate["poles"], scene["P"], scene["L"])
    np.testing.assert_allclose(estimate["B"], b_expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(estimate["Q"], q_expected, atol=0.0, rtol=0.0)
    assert estimate["stage1_delay_method"] == "aimdf_tls"
    assert estimate["stage1_factor_init"] == "hankel_coupled_ls"
    assert estimate["stage1_forward_backward"] is True
    assert estimate["stage1_tls"] is True
