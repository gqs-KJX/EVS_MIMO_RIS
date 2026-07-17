"""Conditional LG/RDC/top-L candidate generation for experimental CCOP-JVP.

The module never replaces the frozen estimator.  It retains the direct CCOP
candidate as an incumbent, treats RDC only as a short-search interval/seed,
and enumerates discrete Stage-I assignments without a soft permutation
mixture or annealing objective.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from src.ccop_jvp import refine_ccop_jvp
from .cp_ngc import cp_ngc_stage1_vector, cp_ngc_statistic
from .cp_ngc_covariance import linearized_stage1_covariance
from src.estimators import estimate_position_from_local_ris
from src.global_vp import distance_to_box_boundary, extract_stage1_jones_directions
from src.main_single_proposed import refine_stage2_ris_factors, solve_stage2_rescue
from src.projections_delay import tau_from_pole


def _inverse_assignment(column_to_panel: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    assignment = tuple(int(value) for value in column_to_panel)
    inverse = [-1] * len(assignment)
    for column, panel in enumerate(assignment):
        if panel < 0 or panel >= len(assignment) or inverse[panel] >= 0:
            raise ValueError("assignment must be a permutation")
        inverse[panel] = column
    return tuple(inverse)


def top_l_assignment_hypotheses(stage1_estimate: dict, top_l: int) -> list[dict]:
    """Return Stage-I discrete assignment hypotheses in recorded score order."""
    count = max(1, int(top_l))
    k_paths = int(np.asarray(stage1_estimate["poles"]).size)
    records = list(stage1_estimate.get("all_assignment_scores", []))
    if not records:
        records = [
            {"assignment": assignment, "score": rank}
            for rank, assignment in enumerate(
                stage1_estimate.get("stage1_shortlisted_assignments", [])
            )
        ]
    direct = tuple(
        int(value)
        for value in stage1_estimate.get("assignment", range(k_paths))
    )
    records.append({"assignment": direct, "score": float("-inf")})
    ranked = []
    seen = set()
    for record in sorted(records, key=lambda item: float(item.get("score", np.inf))):
        assignment = tuple(int(value) for value in record["assignment"])
        if len(assignment) != k_paths or sorted(assignment) != list(range(k_paths)):
            continue
        if assignment in seen:
            continue
        seen.add(assignment)
        ranked.append(
            {
                "column_to_panel": assignment,
                "panel_to_column": _inverse_assignment(assignment),
                "stage1_score": float(record.get("score", np.nan)),
                "is_direct_assignment": bool(assignment == direct),
            }
        )
    return ranked[: min(count, len(ranked))]


def estimate_for_assignment(stage1_estimate: dict, hypothesis: dict) -> dict:
    """Rebuild a panel-ordered Stage-I estimate for one raw-column assignment."""
    estimate = copy.deepcopy(stage1_estimate)
    k_paths = int(np.asarray(stage1_estimate["poles"]).size)
    current_panel_to_column = tuple(
        int(value)
        for value in stage1_estimate.get("panel_to_column_assignment", range(k_paths))
    )
    if len(current_panel_to_column) != k_paths or sorted(current_panel_to_column) != list(range(k_paths)):
        raise ValueError("Stage-I panel_to_column_assignment is not a permutation")
    new_panel_to_column = tuple(int(value) for value in hypothesis["panel_to_column"])

    def reorder_matrix(value: Any) -> np.ndarray:
        physical = np.asarray(value)
        raw = np.empty_like(physical)
        for panel, raw_column in enumerate(current_panel_to_column):
            raw[:, raw_column] = physical[:, panel]
        return raw[:, new_panel_to_column].copy()

    def reorder_vector(value: Any) -> np.ndarray:
        physical = np.asarray(value)
        raw = np.empty_like(physical)
        for panel, raw_column in enumerate(current_panel_to_column):
            raw[raw_column] = physical[panel]
        return raw[list(new_panel_to_column)].copy()

    for key in ("A", "B", "Q", "C"):
        if key in estimate:
            estimate[key] = reorder_matrix(estimate[key])
    for key in (
        "poles",
        "beta_z",
        "gamma",
        "eta_pol",
        "stage1_ris_residuals",
        "stage1_rank1_ratios",
        "stage1_delay_valid",
        "stage1_local_geometry_valid",
    ):
        if key in estimate:
            estimate[key] = reorder_vector(estimate[key])
    if "ris_eta" in estimate:
        physical_eta = np.asarray(estimate["ris_eta"])
        raw_eta = np.empty_like(physical_eta)
        for panel, raw_column in enumerate(current_panel_to_column):
            raw_eta[raw_column] = physical_eta[panel]
        estimate["ris_eta"] = raw_eta[list(new_panel_to_column)].copy()
    estimate["assignment"] = list(hypothesis["column_to_panel"])
    estimate["column_to_panel_assignment"] = list(hypothesis["column_to_panel"])
    estimate["panel_to_column_assignment"] = list(new_panel_to_column)
    estimate["columns_are_panel_ordered"] = True
    estimate["ccop_assignment_hypothesis"] = list(hypothesis["column_to_panel"])
    return estimate


def local_geometry_seed(stage1_estimate: dict, scene: dict, config: dict) -> dict:
    """Return the deterministic LG position seed without changing geometry."""
    position = estimate_position_from_local_ris(scene, stage1_estimate, config)
    return {
        "available": bool(np.all(np.isfinite(position))),
        "p_u": np.asarray(position, dtype=float),
        "source": "Stage-I_RIS_local_geometry_mean",
    }


def rdc_clock_interval(stage1_estimate: dict, scene: dict, config: dict) -> dict:
    """Return robust delay-minus-range clock replicas and a short interval.

    The interval is a computational proposal only.  Full CCOP profiling is
    retained as fallback, so RDC is never the final clock estimator.
    """
    k_paths = int(scene["K"])
    poles = np.asarray(stage1_estimate["poles"], dtype=complex).reshape(k_paths)
    ranges = np.asarray(stage1_estimate["ris_eta"], dtype=float).reshape(k_paths, 3)[:, 0]
    tau = np.asarray(
        [tau_from_pole(pole, float(scene["delta_f"])) for pole in poles]
    )
    replicas = tau - (
        ranges + np.asarray(scene["d_RB"], dtype=float)
    ) / float(scene["c0"])
    center = float(np.median(replicas))
    mad = float(1.4826 * np.median(np.abs(replicas - center)))
    minimum_radius = float(config.get("ccop_rdc_min_radius_ns", 0.5)) * 1.0e-9
    radius = max(
        minimum_radius,
        float(config.get("ccop_rdc_mad_multiplier", 3.0)) * mad,
    )
    full_bounds = np.asarray(config["delta_t_bounds"], dtype=float)
    lower = max(float(full_bounds[0]), center - radius)
    upper = min(float(full_bounds[1]), center + radius)
    available = bool(
        np.all(np.isfinite(replicas))
        and np.isfinite(center)
        and lower < upper
    )
    return {
        "available": available,
        "clock_seed_s": center,
        "clock_replicas_s": replicas,
        "robust_scale_s": mad,
        "interval_s": np.array([lower, upper], dtype=float),
        "source": "RDC_short_interval_only",
        "is_final_estimator": False,
        "full_interval_fallback_retained": True,
    }


def _jones_leakage(estimate: dict, stage1_estimate: dict, scene: dict) -> float:
    x_hat = np.asarray(estimate.get("x_hat", []), dtype=complex)
    if x_hat.size != 2 * int(scene["K"]):
        return float("nan")
    anchors = extract_stage1_jones_directions(stage1_estimate, scene)
    leakage = []
    for path, block in enumerate(x_hat.reshape(int(scene["K"]), 2)):
        anchor = anchors[path]
        projection = anchor * np.vdot(anchor, block) / max(
            float(np.vdot(anchor, anchor).real), 1.0e-15
        )
        leakage.append(
            float(
                np.vdot(block - projection, block - projection).real
                / max(np.vdot(block, block).real, 1.0e-15)
            )
        )
    return float(np.max(leakage))


def run_recovery_ablation(
    y_raw: np.ndarray,
    stage1_estimate: dict,
    direct_candidate: dict,
    scene: dict,
    config: dict,
    *,
    variant: str,
    top_l: int = 3,
    short_outer_max_iter: int = 3,
    full_hypotheses: int = 1,
    noise_variance: float | None = None,
    z_noisy: np.ndarray | None = None,
) -> dict:
    """Generate S0--S4 candidates and apply a conservative selector.

    The best rescue candidate is returned for ablation even when it is not
    eligible to replace the direct incumbent.  Replacement additionally
    requires full-rank C1 covariance and a lower CP-NGC statistic; before
    validation calibration, this keeps CP-NGC diagnostic-only in rank-deficient
    cases.
    """
    if variant not in {"S0", "S1", "S2", "S3", "S4"}:
        raise ValueError("variant must be one of S0--S4")
    start = time.perf_counter()
    if variant == "S0":
        return {
            "variant": variant,
            "selected": copy.deepcopy(direct_candidate),
            "best_rescue": None,
            "accepted": False,
            "acceptance_reason": "no_rescue",
            "runtime_s": float(time.perf_counter() - start),
            "candidate_records": [],
        }
    use_lg = variant in {"S1", "S2", "S3", "S4"}
    use_rdc = variant in {"S2", "S4"}
    use_assignments = variant in {"S3", "S4"}
    hypotheses = (
        top_l_assignment_hypotheses(stage1_estimate, top_l)
        if use_assignments
        else top_l_assignment_hypotheses(stage1_estimate, 1)
    )
    short_records: list[dict] = []
    for hypothesis in hypotheses:
        hypothesis_estimate = estimate_for_assignment(stage1_estimate, hypothesis)
        lg = local_geometry_seed(hypothesis_estimate, scene, config)
        lg_diagnostics: dict[str, Any] = {
            "ris_only_refinement_used": False,
            "rescue_available": False,
            "failure_reason": "",
        }
        if use_lg and z_noisy is not None:
            try:
                common_state = refine_stage2_ris_factors(
                    np.asarray(z_noisy),
                    scene,
                    config,
                    hypothesis_estimate,
                )
                solution = solve_stage2_rescue(
                    common_state,
                    scene,
                    config,
                    impl=str(config.get("stage2_rescue_impl", "pllg")),
                )
                hypothesis_estimate = copy.deepcopy(solution["estimate"])
                lg = {
                    "available": bool(solution.get("rescue_available", False)),
                    "p_u": np.asarray(solution.get("position", np.full(3, np.nan))),
                    "source": "RIS_only_refinement_plus_LG_polish",
                }
                lg_diagnostics = {
                    "ris_only_refinement_used": True,
                    "rescue_available": bool(solution.get("rescue_available", False)),
                    "failure_reason": str(solution.get("failure_reason", "")),
                    "runtime_s": float(solution.get("runtime_s", 0.0)),
                    "num_valid_local_fixes": int(
                        sum(
                            bool(record.get("valid", False))
                            for record in common_state.local_fix_records
                        )
                    ),
                }
            except Exception as error:  # noqa: BLE001 - retain deterministic LG fallback.
                lg_diagnostics = {
                    "ris_only_refinement_used": True,
                    "rescue_available": False,
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
        if use_lg and lg["available"]:
            hypothesis_estimate["_global_vp_initial_p_u"] = lg["p_u"]
        short_config = copy.deepcopy(config)
        short_config["ccop_jvp"] = dict(short_config.get("ccop_jvp", {}))
        short_config["ccop_jvp"]["outer_max_iter"] = int(short_outer_max_iter)
        rdc = rdc_clock_interval(hypothesis_estimate, scene, config)
        if use_rdc and rdc["available"]:
            short_config["delta_t_bounds"] = rdc["interval_s"].copy()
            hypothesis_estimate["_global_vp_initial_delta_t"] = float(
                np.clip(rdc["clock_seed_s"], *rdc["interval_s"])
            )
        candidate = refine_ccop_jvp(
            y_raw, hypothesis_estimate, scene, short_config, incumbent=None
        )
        short_records.append(
            {
                "hypothesis": hypothesis,
                "stage1_estimate": hypothesis_estimate,
                "lg": lg,
                "lg_diagnostics": lg_diagnostics,
                "rdc": rdc,
                "short_candidate": candidate,
                "short_objective": float(candidate["total_objective_final"]),
            }
        )
    short_records.sort(key=lambda record: record["short_objective"])
    full_records = []
    for record in short_records[: max(1, int(full_hypotheses))]:
        full_estimate = copy.deepcopy(record["stage1_estimate"])
        full_estimate["_global_vp_initial_p_u"] = np.asarray(
            record["short_candidate"]["p_u"], dtype=float
        )
        # Restore the complete clock interval: RDC is never the final solver.
        full_candidate = refine_ccop_jvp(
            y_raw, full_estimate, scene, config, incumbent=None
        )
        boundary = distance_to_box_boundary(
            full_candidate["p_u"],
            np.asarray(config["ue_bounds"], dtype=float),
            float(config.get("global_vp", {}).get("boundary_tol_m", 0.02)),
        )
        full_records.append(
            {
                **record,
                "full_candidate": full_candidate,
                "boundary": boundary,
                "jones_leakage": _jones_leakage(
                    full_candidate, full_estimate, scene
                ),
            }
        )
    full_records.sort(
        key=lambda record: float(record["full_candidate"]["total_objective_final"])
    )
    best_record = full_records[0]
    best = best_record["full_candidate"]
    direct_total = float(direct_candidate["total_objective_final"])
    direct_raw = float(direct_candidate["raw_objective_final"])
    objective_tolerance = float(config.get("ccop_recovery_objective_tolerance", 1.0e-12))
    finite = bool(
        np.all(np.isfinite(best["p_u"]))
        and np.isfinite(float(best["delta_t"]))
        and np.isfinite(float(best["raw_objective_final"]))
    )
    geometry_bounds = bool(
        np.all(np.asarray(best["p_u"]) >= np.asarray(config["ue_bounds"])[:, 0])
        and np.all(np.asarray(best["p_u"]) <= np.asarray(config["ue_bounds"])[:, 1])
    )
    raw_non_degradation = bool(
        float(best["raw_objective_final"]) <= direct_raw + objective_tolerance
    )
    total_non_degradation = bool(
        float(best["total_objective_final"]) <= direct_total + objective_tolerance
    )

    cp_improved = False
    cp_available = False
    covariance_reliable = False
    direct_cp = rescue_cp = None
    if noise_variance is not None:
        try:
            direct_covariance = linearized_stage1_covariance(
                y_raw, stage1_estimate, scene, float(noise_variance)
            )
            rescue_covariance = linearized_stage1_covariance(
                y_raw,
                best_record["stage1_estimate"],
                scene,
                float(noise_variance),
            )
            direct_cp = cp_ngc_statistic(
                cp_ngc_stage1_vector(stage1_estimate, scene),
                direct_candidate["p_u"],
                direct_covariance["covariance_z"],
                scene,
            )
            rescue_cp = cp_ngc_statistic(
                cp_ngc_stage1_vector(best_record["stage1_estimate"], scene),
                best["p_u"],
                rescue_covariance["covariance_z"],
                scene,
            )
            cp_available = True
            covariance_reliable = bool(
                direct_covariance["covariance_reliable_for_hard_certificate"]
                and rescue_covariance["covariance_reliable_for_hard_certificate"]
            )
            cp_improved = bool(rescue_cp["statistic"] < direct_cp["statistic"])
        except Exception:  # noqa: BLE001 - failure keeps the incumbent.
            cp_available = False
    leakage_threshold = float(
        config.get("global_vp", {}).get("jones_leakage_threshold", 0.25)
    )
    leakage_ok = bool(
        not np.isfinite(best_record["jones_leakage"])
        or best_record["jones_leakage"] <= leakage_threshold
    )
    direct_boundary = distance_to_box_boundary(
        direct_candidate["p_u"],
        np.asarray(config["ue_bounds"], dtype=float),
        float(config.get("global_vp", {}).get("boundary_tol_m", 0.02)),
    )
    relative_raw_improvement = float(
        (direct_raw - float(best["raw_objective_final"]))
        / max(abs(direct_raw), 1.0e-15)
    )
    alternative_assignment = bool(
        not best_record["hypothesis"].get("is_direct_assignment", True)
    )
    independent_lg_support = bool(
        best_record["lg"].get("available", False)
        and best_record["lg_diagnostics"].get("ris_only_refinement_used", False)
    )
    boundary_override = bool(
        direct_boundary["boundary_hit"]
        and not best_record["boundary"]["boundary_hit"]
        and alternative_assignment
        and independent_lg_support
        and relative_raw_improvement
        >= float(
            config.get(
                "ccop_boundary_override_min_rel_raw_improvement", 1.0e-3
            )
        )
    )
    common_eligibility = bool(
        finite
        and geometry_bounds
        and not best_record["boundary"]["boundary_hit"]
        and raw_non_degradation
        and total_non_degradation
        and leakage_ok
    )
    cp_calibrated_path = bool(cp_available and covariance_reliable and cp_improved)
    eligible = bool(common_eligibility and (boundary_override or cp_calibrated_path))
    selected = copy.deepcopy(best if eligible else direct_candidate)
    failed_conditions = [
        name
        for name, passed in (
            ("finite", finite),
            ("geometry_bounds", geometry_bounds),
            ("nonboundary", not best_record["boundary"]["boundary_hit"]),
            ("raw_non_degradation", raw_non_degradation),
            ("total_non_degradation", total_non_degradation),
            ("jones_leakage", leakage_ok),
            ("boundary_override_or_cp_calibrated", boundary_override or cp_calibrated_path),
        )
        if not passed
    ]
    return {
        "variant": variant,
        "selected": selected,
        "best_rescue": copy.deepcopy(best),
        "accepted": eligible,
        "acceptance_reason": "accepted" if eligible else ";".join(failed_conditions),
        "runtime_s": float(time.perf_counter() - start),
        "candidate_records": full_records,
        "num_assignment_hypotheses": len(hypotheses),
        "num_short_ccop": len(short_records),
        "num_full_ccop": len(full_records),
        "direct_incumbent_retained": not eligible,
        "raw_non_degradation": raw_non_degradation,
        "total_non_degradation": total_non_degradation,
        "cp_available": cp_available,
        "covariance_reliable": covariance_reliable,
        "cp_improved": cp_improved,
        "boundary_override": boundary_override,
        "relative_raw_improvement": relative_raw_improvement,
        "alternative_assignment_support": alternative_assignment,
        "independent_lg_support": independent_lg_support,
        "selection_path": (
            "boundary_override"
            if eligible and boundary_override
            else "cp_calibrated"
            if eligible
            else "direct_incumbent"
        ),
        "direct_cp_ngc": direct_cp,
        "rescue_cp_ngc": rescue_cp,
        "rdc_is_final_estimator": False,
        "permutation_annealing_used": False,
    }
