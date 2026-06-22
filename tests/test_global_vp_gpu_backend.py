import builtins
import copy

import numpy as np
import pytest

from src.baselines.backend import get_backend
from src.channel_model import channel_components, generate_scene, synthesize_raw_tensor
from src.config import default_config
from src.global_vp import (
    _CuPyReducedVPEvaluator,
    _global_vp_config,
    _make_global_vp_gpu_context,
    _solve_linear_vp_regularized,
    _vp_objective_parts_and_grad,
    _vp_objective_parts_and_grad_cupy,
    global_exact_spherical_vp_refinement,
)
from src.main_single_proposed import _make_data


def _tiny_problem():
    config = default_config()
    config.update(
        {
            "seed": 1234,
            "K": 1,
            "M_A": 1,
            "ris_shape": (2, 2),
            "N": 5,
            "P": 3,
            "T": 4,
            "p_u_true": np.array([1.2, 0.4, 0.8]),
            "ris_centers": np.array([[4.2, -2.2, 1.05]]),
            "ue_bounds": np.array([[1.0, 1.4], [0.2, 0.6], [0.6, 1.0]]),
            "delta_t_true": 5.0e-9,
            "delta_t_bounds": np.array([4.0e-9, 6.0e-9]),
        }
    )
    config["global_vp"] = dict(config["global_vp"])
    config["global_vp"].update(
        {
            "mode": "jones_free",
            "solver": "lbfgsb_reduced",
            "max_iter": 1,
            "use_delay_prior": False,
            "enable_z_rescue_multistart": False,
        }
    )
    rng = np.random.default_rng(config["seed"])
    scene = generate_scene(config, rng)
    components = channel_components(
        scene,
        scene["p_u_true"],
        scene["delta_t_true"],
        scene["gamma_true"],
        scene["eta_true"],
    )
    y_raw = synthesize_raw_tensor(components, scene["beta_true"])
    init = {
        "A": components["a_EVS"].T.copy(),
        "D": components["d"].T.copy(),
        "C": components["c"].T.copy(),
        "poles": components["poles"].copy(),
        "ris_eta": np.column_stack(
            [
                components["ranges"],
                components["elevations"],
                components["azimuths"],
            ]
        ),
        "gamma": scene["gamma_true"].copy(),
        "eta_pol": scene["eta_true"].copy(),
        "assignment": [0],
        "panel_to_column_assignment": [0],
        "columns_are_panel_ordered": True,
    }
    return config, scene, y_raw, init


def _cupy_or_skip():
    try:
        backend = get_backend({"backend": "cupy", "gpu_device": 0})
        backend.asarray([1.0])
        backend.synchronize()
        return backend
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_global_vp_cpu_backend_is_default():
    options = _global_vp_config(default_config())
    assert options["backend"] == "cpu"
    assert options["gpu_dtype"] == "complex128"


def test_global_vp_auto_falls_back_without_cupy(monkeypatch):
    original_import = builtins.__import__

    def rejecting_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("forced unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        backend = get_backend({"backend": "auto"})
    assert backend.name == "cpu"


def test_backend_change_does_not_change_numpy_rng_data_generation():
    cpu_config = default_config()
    cpu_config["diagnostic_fast_problem_size"] = True
    cpu_config["global_vp"]["backend"] = "cpu"
    auto_config = copy.deepcopy(cpu_config)
    auto_config["global_vp"]["backend"] = "auto"
    cpu = _make_data(cpu_config)
    auto = _make_data(auto_config)
    np.testing.assert_array_equal(cpu["Y_true"], auto["Y_true"])
    np.testing.assert_array_equal(cpu["Y_noisy"], auto["Y_noisy"])


def test_legacy_lstsq_path_remains_cpu():
    config, scene, y_raw, init = _tiny_problem()
    config["global_vp"].update({"mode": "fixed_pol", "solver": "least_squares"})
    result = global_exact_spherical_vp_refinement(y_raw, init, scene, config)
    assert result["global_vp_lstsq_backend"] == "cpu"
    assert result["global_vp_least_squares_gpu_partial"] is False


def test_cpu_gpu_regularized_solve_match():
    backend = _cupy_or_skip()
    rng = np.random.default_rng(8)
    phi = rng.normal(size=(12, 3)) + 1j * rng.normal(size=(12, 3))
    y = rng.normal(size=12) + 1j * rng.normal(size=12)
    reg = np.diag([0.1, 0.2, 0.3]).astype(complex)
    cpu, _ = _solve_linear_vp_regularized(phi, y, reg, 1.0e-10)
    gpu, _ = _solve_linear_vp_regularized(
        backend.asarray(phi),
        backend.asarray(y),
        backend.asarray(reg),
        1.0e-10,
        backend=backend,
    )
    np.testing.assert_allclose(backend.to_host(gpu), cpu, rtol=1.0e-11, atol=1.0e-12)


def test_cpu_gpu_reduced_objective_gradient_and_xhat_match():
    backend = _cupy_or_skip()
    config, scene, y_raw, init = _tiny_problem()
    xi = np.r_[scene["p_u_true"] + [0.01, -0.01, 0.01], scene["delta_t_true"]]
    y_vec = y_raw.reshape(-1)
    cpu_parts, cpu_grad = _vp_objective_parts_and_grad(
        xi, y_vec, init, scene, config
    )
    context = _make_global_vp_gpu_context(y_vec, init, scene, config, backend)
    gpu_parts, gpu_grad = _vp_objective_parts_and_grad_cupy(
        xi, init, scene, config, context
    )
    np.testing.assert_allclose(
        gpu_parts["total_objective"],
        cpu_parts["total_objective"],
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(gpu_grad, cpu_grad, rtol=1.0e-7, atol=1.0e-8)
    np.testing.assert_allclose(
        backend.to_host(gpu_parts["beta"]),
        cpu_parts["beta"],
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_cpu_gpu_tiny_final_position_match():
    _cupy_or_skip()
    cpu_config, scene, y_raw, init = _tiny_problem()
    gpu_config = copy.deepcopy(cpu_config)
    gpu_config["global_vp"].update(
        {
            "backend": "cupy",
            "gpu_device": 0,
            "validate_gpu_against_cpu": True,
        }
    )
    cpu = global_exact_spherical_vp_refinement(y_raw, init, scene, cpu_config)
    gpu = global_exact_spherical_vp_refinement(y_raw, init, scene, gpu_config)
    np.testing.assert_allclose(gpu["p_u"], cpu["p_u"], rtol=1.0e-8, atol=1.0e-9)
    assert gpu["global_vp_gpu_used"] is True
    assert gpu["global_vp_gpu_num_objective_calls"] > 0

