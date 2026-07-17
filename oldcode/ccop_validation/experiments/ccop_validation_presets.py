"""Frozen candidate presets and deterministic seed splits for CCOP validation."""

from __future__ import annotations

import copy

import numpy as np


PRESETS = {
    "fast": {
        "status": "experimental_unvalidated",
        "clock_fft_size": 1024,
        "clock_rel_tol": 1.0e-8,
        "clock_max_intervals": 5000,
        "ccop_outer_max_iter": 8,
        "top_l": 1,
        "short_outer_max_iter": 2,
        "full_hypotheses": 1,
        "covariance": "disabled_in_deployment",
        "cp_ngc_policy": "disabled",
        "recovery_trigger": "disabled",
        "heldout": "disabled",
    },
    "balanced": {
        "status": "paper_candidate_stage3_validation_passed_recovery_rejected",
        "clock_fft_size": 4096,
        "clock_rel_tol": 1.0e-10,
        "clock_max_intervals": 20000,
        "ccop_outer_max_iter": 20,
        "top_l": 3,
        "short_outer_max_iter": 3,
        "full_hypotheses": 1,
        "covariance": "disabled_in_deployment_diagnostic_only",
        "cp_ngc_policy": "disabled",
        "recovery_trigger": "disabled_after_validation_no_gain",
        "bootstrap_replicates": 24,
        "heldout": "disabled",
    },
    "accuracy": {
        "status": "experimental_not_admissible_without_recovery_validation",
        "clock_fft_size": 8192,
        "clock_rel_tol": 1.0e-12,
        "clock_max_intervals": 50000,
        "ccop_outer_max_iter": 40,
        "top_l": 6,
        "short_outer_max_iter": 5,
        "full_hypotheses": 2,
        "covariance": "C1_soft_diagnostic_on_boundary_only",
        "cp_ngc_policy": "triggered_only",
        "recovery_trigger": "direct_boundary_only",
        "bootstrap_replicates": 48,
        "heldout": "gray_and_red",
    },
}


REGRESSION_SEEDS = {
    # These seeds are diagnostic replays only.  The old saved outputs are not
    # evidence because their worktree/observation hashes were not recorded.
    "trial_46_correlated_false_green": 2935528983,
    "trial_51_boundary_override": 2563947575,
    "z_upper_boundary_trial_1": 1963740598,
    "z_upper_boundary_trial_5": 2003832857,
    "z_upper_boundary_trial_14": 3257551814,
    "channel_nmse_near_one_smoke": 4191845062,
}


def _spawn(root_seed: int, count: int) -> list[int]:
    root = np.random.SeedSequence(int(root_seed))
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in root.spawn(int(count))
    ]


def seed_splits() -> dict:
    """Return disjoint development, validation, and one-shot held-out splits."""
    development = _spawn(20260715, 20)
    validation = _spawn(20260716, 100)
    heldout = _spawn(20260717, 400)
    if set(development) & set(validation) or set(development) & set(heldout) or set(validation) & set(heldout):
        raise RuntimeError("generated CCOP seed splits overlap")
    return {
        "development": development,
        "validation": validation,
        "heldout": heldout,
        "regression": copy.deepcopy(REGRESSION_SEEDS),
        "policy": {
            "development": "debug only",
            "validation": "threshold, L, covariance and budget selection",
            "heldout": "run once after preset freeze; no retuning",
            "regression": "failure replay only; never pooled into final statistics",
        },
    }


def preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"unknown CCOP validation preset {name!r}")
    return copy.deepcopy(PRESETS[name])
