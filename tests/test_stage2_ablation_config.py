import numpy as np

from src.config import default_config
from src.estimators import _accept_strict_sse, structured_refinement
from src.projections_delay import bq_from_poles


def test_default_config_has_guarded_stage2_keys_disabled_by_default():
    config = default_config()
    assert config["stage2_enable_evs"] is True
    assert config["stage2_enable_delay"] is True
    assert config["stage2_enable_ris"] is True
    assert config["stage2_guarded"] is False
    assert config["stage2_strict_accept_rel"] == 1.0e-6
    assert config["ris_min_relative_improvement"] == 5.0e-3
    assert tuple(config["stage2_damping_grid"]) == (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


def test_structured_refinement_all_modules_disabled_keeps_factors_unchanged():
    rng = np.random.default_rng(7)
    i_dim, p_dim, l_dim, t_dim, k_paths = 5, 3, 4, 6, 2
    z_tensor = rng.normal(size=(i_dim, p_dim, l_dim, t_dim)) + 1j * rng.normal(
        size=(i_dim, p_dim, l_dim, t_dim)
    )
    a_mat = rng.normal(size=(i_dim, k_paths)) + 1j * rng.normal(size=(i_dim, k_paths))
    c_mat = rng.normal(size=(t_dim, k_paths)) + 1j * rng.normal(size=(t_dim, k_paths))
    poles = np.exp(1j * np.array([0.23, -0.71]))
    b_mat, q_mat = bq_from_poles(poles, p_dim, l_dim)
    estimate = {
        "A": a_mat.copy(),
        "B": b_mat.copy(),
        "Q": q_mat.copy(),
        "C": c_mat.copy(),
        "poles": poles.copy(),
        "beta_z": np.ones(k_paths, dtype=complex),
        "gamma": np.zeros(k_paths),
        "eta_pol": np.zeros(k_paths),
        "ris_eta": np.zeros((k_paths, 3)),
    }
    config = default_config()
    config.update(
        {
            "num_structured_iters": 0,
            "stage2_enable_evs": False,
            "stage2_enable_delay": False,
            "stage2_enable_ris": False,
        }
    )
    scene = {"P": p_dim, "L": l_dim}

    refined, _ = structured_refinement(z_tensor, scene, config, estimate)

    np.testing.assert_allclose(refined["A"], a_mat)
    np.testing.assert_allclose(refined["C"], c_mat)
    np.testing.assert_allclose(refined["poles"], poles)
    assert refined["Z_hat"].shape == z_tensor.shape


def test_accept_strict_sse_requires_strict_relative_decrease():
    assert _accept_strict_sse(9.0, 10.0, 0.0, 1.0e-3)
    assert not _accept_strict_sse(10.0, 10.0, 0.0, 1.0e-3)
    assert not _accept_strict_sse(10.1, 10.0, 0.0, 1.0e-3)
