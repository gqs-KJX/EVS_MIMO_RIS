"""Integrated experimental CCOP route with gate-controlled recovery.

This module is separate from the frozen estimator.  The currently admissible
balanced route is independent CCOP-JVP only.  C1--C4 CP-NGC and conditional
LG/top-L recovery remain available in their diagnostic/ablation runners, but
are disabled here because their validation gates were not met.  The direct
candidate is therefore also the final candidate for the paper preset.
"""

from __future__ import annotations

import time

import numpy as np

from src.ccop_jvp import refine_ccop_jvp
from .ccop_recovery import run_recovery_ablation
from .cp_ngc import cp_ngc_stage1_vector, cp_ngc_statistic
from .cp_ngc_covariance import linearized_stage1_covariance
from .experiments.ccop_validation_presets import preset as validation_preset
from src.global_vp import distance_to_box_boundary


def run_ccop_certified_route(
    data: dict,
    stage1_estimate: dict,
    config: dict,
    *,
    preset_name: str = "balanced",
) -> dict:
    """Run the gate-controlled preset without changing the frozen estimator."""
    chosen = validation_preset(preset_name)
    total_start = time.perf_counter()
    direct_start = time.perf_counter()
    direct = refine_ccop_jvp(
        data["Y_noisy"], stage1_estimate, data["scene"], config, incumbent=None
    )
    direct_runtime = time.perf_counter() - direct_start
    boundary = distance_to_box_boundary(
        direct["p_u"],
        np.asarray(config["ue_bounds"], dtype=float),
        float(config["global_vp"].get("boundary_tol_m", 0.02)),
    )

    recovery_trigger = str(chosen.get("recovery_trigger", "direct_boundary_only"))
    trigger = bool(
        boundary["boundary_hit"] and recovery_trigger == "direct_boundary_only"
    )
    trigger_reasons = ["direct_boundary"] if trigger else []
    cp_policy = str(chosen.get("cp_ngc_policy", "triggered_only"))
    evaluate_cp = cp_policy == "always" or (cp_policy == "triggered_only" and trigger)
    cp_start = time.perf_counter()
    cp_result = None
    covariance_result = None
    cp_error = ""
    if evaluate_cp:
        try:
            covariance_result = linearized_stage1_covariance(
                data["Y_noisy"],
                stage1_estimate,
                data["scene"],
                data["noise_variance"],
            )
            cp_result = cp_ngc_statistic(
                cp_ngc_stage1_vector(stage1_estimate, data["scene"]),
                direct["p_u"],
                covariance_result["covariance_z"],
                data["scene"],
            )
        except Exception as error:  # noqa: BLE001 - CP is diagnostic at this gate.
            cp_error = f"{type(error).__name__}: {error}"
    cp_runtime = time.perf_counter() - cp_start

    recovery = None
    recovery_runtime = 0.0
    final = direct
    if trigger:
        recovery_start = time.perf_counter()
        recovery = run_recovery_ablation(
            data["Y_noisy"],
            stage1_estimate,
            direct,
            data["scene"],
            config,
            variant="S3",
            top_l=int(chosen["top_l"]),
            short_outer_max_iter=int(chosen["short_outer_max_iter"]),
            full_hypotheses=int(chosen["full_hypotheses"]),
            noise_variance=None,
            z_noisy=data["Z_noisy"],
        )
        recovery_runtime = time.perf_counter() - recovery_start
        final = recovery["selected"]

    raw_non_degradation = bool(
        float(final["raw_objective_final"])
        <= float(direct["raw_objective_final"]) + 1.0e-12
    )
    total_non_degradation = bool(
        float(final["total_objective_final"])
        <= float(direct["total_objective_final"]) + 1.0e-12
    )
    if not (raw_non_degradation and total_non_degradation):
        final = direct
        raw_non_degradation = True
        total_non_degradation = True
        rollback = True
    else:
        rollback = False
    return {
        "final": final,
        "direct": direct,
        "cp_ngc": cp_result,
        "covariance": covariance_result,
        "cp_ngc_error": cp_error,
        "cp_ngc_deployment_status": (
            "soft_diagnostic_on_trigger"
            if evaluate_cp
            else "soft_diagnostic_not_evaluated"
        ),
        "cp_ngc_hard_gate_passed": False,
        "triggered": trigger,
        "trigger_reasons": trigger_reasons,
        "recovery": recovery,
        "selected_branch": (
            "boundary_override"
            if recovery is not None and recovery["accepted"]
            else "direct_ccop"
        ),
        "raw_non_degradation": raw_non_degradation,
        "total_non_degradation": total_non_degradation,
        "rollback": rollback,
        "preset": preset_name,
        "disabled_modules": {
            "RDC": "diagnostic clock interval/fallback only; no deployment selector use",
            "S3_LG_topL": "validation boundary replay added runtime without an accepted rescue",
            "C2": "bootstrap cost/instability gate not passed",
            "C4": "correct full-data candidate may fail on fold A; gray diagnostic only",
            "CP_NGC_hard_threshold": "C1 development detection gate not passed; no hard threshold calibrated",
        },
        "runtime": {
            "direct_ccop_s": float(direct_runtime),
            "cp_ngc_c1_s": float(cp_runtime),
            "conditional_recovery_s": float(recovery_runtime),
            "total_route_s": float(time.perf_counter() - total_start),
        },
    }
