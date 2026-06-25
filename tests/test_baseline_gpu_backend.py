import builtins
import copy

import numpy as np
import pytest

from src.baselines import common, far_field_omp, near_field_mmpsr, ris_momp
from src.baselines.backend import BackendConfig, get_backend
from src.baselines.cache import BaselineCache, baseline_cache_key
from src.config import default_config
from src.experiments import run_benchmark_comparison as benchmark
from src.main_single_proposed import _make_data


def _tiny_config():
    config = default_config()
    config.update(
        {
            "seed": 123,
            "K": 1,
            "M_A": 1,
            "ris_shape": (2, 2),
            "N": 5,
            "P": 3,
            "T": 4,
            "SNR_dB": 60.0,
            "receiver_mode": "full_6d",
            "print_progress": False,
            "p_u_true": np.array([1.2, 0.4, 0.8]),
            "ris_centers": np.array([[4.2, -2.2, 1.05]]),
            "ue_bounds": np.array([[1.0, 1.4], [0.2, 0.6], [0.6, 1.0]]),
            "delta_t_true": 5.0e-9,
            "delta_t_bounds": np.array([4.0e-9, 6.0e-9]),
        }
    )
    config["baselines"] = {
        "ff_omp": {
            "angle_grid_size": 3,
            "delay_grid_size": 3,
            "max_atoms": 1,
            "batch_size": 2,
        },
        "ris_momp": {
            "direction_grid_size": 3,
            "delay_grid_size": 3,
            "max_atoms": 1,
            "batch_size": 2,
        },
        "nf_mmpsr": {
            "grid_shape": (3, 3, 3),
            "clock_grid_size": 3,
            "batch_size": 2,
        },
        "backend_config": {"backend": "cpu", "cache_enabled": True},
        "trim_memory": False,
    }
    return config


def _cupy_backend_or_skip():
    try:
        backend = get_backend({"backend": "cupy", "gpu_device": 0})
        backend.asarray([1.0])
        backend.synchronize()
        return backend
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_cpu_backend_imports_and_works_without_cupy():
    backend = get_backend(BackendConfig(backend="cpu"))
    values = backend.asarray([1.0, 2.0])
    assert backend.name == "cpu"
    assert np.array_equal(backend.to_host(values), np.array([1.0, 2.0]))
    assert backend.memory_info()["backend"] == "cpu"


def test_auto_backend_falls_back_when_cupy_import_fails(monkeypatch):
    original_import = builtins.__import__

    def rejecting_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("forced unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        backend = get_backend({"backend": "auto"})
    assert backend.name == "cpu"
    assert "falling back to CPU" in backend.warning


def test_cache_key_changes_for_required_scene_dimensions():
    config = _tiny_config()
    scene = _make_data(config)["scene"]
    base = baseline_cache_key("ff_omp", scene, config, grid_sizes=(3, 3))

    mode_scene = dict(scene)
    mode_scene["receiver_mode"] = "scalar"
    assert baseline_cache_key("ff_omp", mode_scene, config, grid_sizes=(3, 3)) != base
    assert baseline_cache_key("ff_omp", scene, config, grid_sizes=(5, 3)) != base

    shape_config = copy.deepcopy(config)
    shape_config["ris_shape"] = (1, 4)
    assert baseline_cache_key("ff_omp", scene, shape_config, grid_sizes=(3, 3)) != base

    k_scene = dict(scene)
    k_scene["K"] = 2
    assert baseline_cache_key("ff_omp", k_scene, config, grid_sizes=(3, 3)) != base


def test_cache_rejects_noisy_observations():
    cache = BaselineCache()
    with pytest.raises(ValueError, match="cannot store"):
        cache.put("bad", {"Y_noisy": np.ones(4)})


@pytest.mark.parametrize("backend_name", ["cpu", "auto", "cupy"])
def test_benchmark_cli_accepts_baseline_backend(backend_name):
    args = benchmark.parse_args(["--baseline-backend", backend_name])
    assert args.baseline_backend == backend_name


def test_benchmark_row_contains_backend_diagnostics():
    config = _tiny_config()
    data = _make_data(config)
    result = far_field_omp.run_far_field_omp_baseline(data, config)
    row = common.make_baseline_row(result, data, config)
    for field in (
        "baseline_backend",
        "gpu_used",
        "gpu_device",
        "gpu_num_batches",
        "gpu_batch_size",
        "group_omp",
        "offgrid_refinement",
        "refinement_objective",
        "model_variant",
        "cache_enabled",
        "cache_hits",
        "cache_misses",
        "cache_estimated_bytes",
        "scoring_time_s",
        "backend_warning",
    ):
        assert field in row
        assert field in benchmark.FIELDNAMES


def test_ff_omp_cpu_gpu_selected_support_match():
    _cupy_backend_or_skip()
    cpu_config = _tiny_config()
    data = _make_data(cpu_config)
    scene = data["scene"]
    support = list(far_field_omp._far_field_supports(scene, cpu_config))[2]
    atom = common.simple_atom_normalize(
        common.raw_atom_from_support(scene, cpu_config, support)
    )
    data["Y_noisy"] = atom.reshape(scene["I"], scene["N"], scene["T"])
    data["Y_true"] = data["Y_noisy"].copy()
    cpu = far_field_omp.run_far_field_omp_baseline(data, cpu_config)
    gpu_config = copy.deepcopy(cpu_config)
    gpu_config["baselines"]["backend_config"].update(
        {"backend": "cupy", "gpu_device": 0, "gpu_batch_size": 2}
    )
    gpu = far_field_omp.run_far_field_omp_baseline(data, gpu_config)
    assert gpu.selected_support[0]["direction_index"] == cpu.selected_support[0]["direction_index"]
    assert gpu.selected_support[0]["tau_index"] == cpu.selected_support[0]["tau_index"]
    assert gpu.diagnostics["group_omp"] is True
    assert len(gpu.diagnostics["expanded_supports"]) == 2 * len(gpu.selected_support)


def test_ris_momp_cpu_gpu_selected_support_match():
    _cupy_backend_or_skip()
    cpu_config = _tiny_config()
    data = _make_data(cpu_config)
    scene = data["scene"]
    support = list(ris_momp._ris_momp_supports(scene, cpu_config))[2]
    atom = common.simple_atom_normalize(
        common.raw_atom_from_support(scene, cpu_config, support)
    )
    data["Y_noisy"] = atom.reshape(scene["I"], scene["N"], scene["T"])
    data["Y_true"] = data["Y_noisy"].copy()
    cpu = ris_momp.run_ris_momp_baseline(data, cpu_config)
    gpu_config = copy.deepcopy(cpu_config)
    gpu_config["baselines"]["backend_config"].update(
        {"backend": "cupy", "gpu_device": 0, "gpu_batch_size": 2}
    )
    gpu = ris_momp.run_ris_momp_baseline(data, gpu_config)
    assert gpu.selected_support[0]["direction_index"] == cpu.selected_support[0]["direction_index"]
    assert gpu.selected_support[0]["tau_index"] == cpu.selected_support[0]["tau_index"]
    assert gpu.diagnostics["group_omp"] is True
    assert len(gpu.diagnostics["expanded_supports"]) == 2 * len(gpu.selected_support)


def test_nf_mmpsr_cpu_gpu_grid_and_position_match():
    _cupy_backend_or_skip()
    cpu_config = _tiny_config()
    data = _make_data(cpu_config)
    data["Y_noisy"] = data["Y_true"].copy()
    cpu = near_field_mmpsr.run_near_field_mmpsr_baseline(data, cpu_config)
    gpu_config = copy.deepcopy(cpu_config)
    gpu_config["baselines"]["backend_config"].update(
        {"backend": "cupy", "gpu_device": 0, "gpu_batch_size": 2}
    )
    gpu = near_field_mmpsr.run_near_field_mmpsr_baseline(data, gpu_config)
    assert gpu.diagnostics["selected_grid_index"] == cpu.diagnostics["selected_grid_index"]
    assert np.allclose(gpu.p_u, cpu.p_u)
