"""Shared configuration builder for paired CCOP paper experiments.

This small module keeps the frozen MKSC-GI--CCOP route independent of the
retired CP-NGC/conditional-recovery validation runners.  It changes no model
or estimator defaults; the implementation is the configuration block formerly
hosted by ``run_ccop_paired_mc``.
"""

from __future__ import annotations

from ..config import default_config
from ..main_single_proposed import _apply_main_single_defaults


def build_ccop_experiment_config(spec: dict, seed: int) -> dict:
    """Resolve one paired CCOP experiment configuration."""
    config = default_config()
    config["seed"] = int(seed)
    config["SNR_dB"] = float(spec["snr_db"])
    config["print_progress"] = False
    config["verbose_stage2"] = False
    config["run_full_legacy_comparison"] = False
    mode = str(spec["diagnostic_mode"])
    config["diagnostic_mode"] = "smoke" if mode == "fast" else "performance"
    config["diagnostic_fast_problem_size"] = mode == "fast"
    config["diagnostic_fast_stage1_search"] = mode == "fast"
    config = _apply_main_single_defaults(config)
    config["global_vp"] = dict(config.get("global_vp", {}))
    config["global_vp"].update(
        {
            "solver": "lbfgsb_reduced",
            "mode": str(spec["jones_mode"]),
            "backend": str(spec["old_vp_backend"]),
            "gpu_device": int(spec["gpu_device"]),
            "vp_dictionary_mode": "matrix_free",
            "use_weight": False,
            "use_delay_prior": False,
            "jones_diagonal_loading": 0.0,
            "enable_z_rescue_multistart": False,
            "use_multistart": False,
            "max_iter": int(spec["old_max_iter"]),
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
        }
    )
    config["ccop_jvp"] = {
        "clock_fft_size": int(spec["clock_fft_size"]),
        "clock_abs_tol_objective": float(spec["clock_abs_tol"]),
        "clock_rel_tol": float(spec["clock_rel_tol"]),
        "clock_max_intervals": int(spec["clock_max_intervals"]),
        "outer_max_iter": int(spec["ccop_outer_max_iter"]),
        "outer_ftol": 1.0e-12,
        "outer_gtol": 1.0e-8,
    }
    return config
