"""Default parameters for the single proposed-method demo."""

from __future__ import annotations

import numpy as np


def default_config() -> dict:
    """Return a small, deterministic configuration for one fast run."""
    c0 = 299_792_458.0
    fc = 28.0e9
    wavelength = c0 / fc

    return {
        "seed": 20260526,
        "SNR_dB": 0.0,
        "enable_global_vp": False,
        "K": 2,
        "M_A": 2,
        "ris_shape": (32, 32),
        "N": 24,
        "P": 12,
        "T": 128,
        "c0": c0,
        "fc": fc,
        "wavelength": wavelength,
        "delta_f": 20.0e6,
        "delta_t_true": 1.5e-9,
        "p_B": np.array([0.0, 0.0, 1.0]),
        "p_u_true": np.array([1.25, 0.55, 0.75]),
        "ris_centers": np.array(
            [
                [4.20, -2.20, 1.05],
                [5.10, 2.10, 1.15],
            ]
        ),
        "ue_bounds": np.array(
            [
                [0.30, 2.70],
                [-1.40, 1.50],
                [0.35, 1.45],
            ]
        ),
        "delta_t_bounds": np.array([0.0, 4.0e-9]),
        "ris_search": {
            "range_bounds": (2.5, 6.5),
            "elev_bounds": (-0.45, 0.25),
            "az_bounds": (-np.pi, np.pi),
            "num_range": 15,
            "num_elev": 9,
            "num_az": 25,
            "stage2_num_range": 3,
            "stage2_num_elev": 3,
            "stage2_num_az": 5,
            "stage2_range_span": 0.45,
            "stage2_angle_span": 0.12,
            "num_lift_candidates": 4,
            "num_lift_steps": 4,
            "lambda_phys": 1.0e-2,
            "num_exact_refine_starts": 6,
            "projection_mode": "paper",
        },
        "num_structured_iters": 4,
        "stage2_global_safeguard": True,
        "stage2_tol": 1.0e-5,
        "delay_lambda": 1.0e-2,
        "delay_num_pgd_steps": 10,
        "delay_step_scale": 0.8,
        "delay_damping": 0.8,
        "delay_geometry_rho": 0.2,
        "assignment_clock_weight": 0.5,
        "assignment_clock_scale_s": 1.0e-9,
        "delay_refine_phase_span": 0.35,
        "delay_refine_phase_grid": 9,
        "eps": 1.0e-10,
    }
