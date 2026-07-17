"""Deterministic development, validation, and held-out seeds for MKSC-GI."""

from __future__ import annotations

import copy

import numpy as np


REGRESSION_SEEDS = {
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
    """Return the frozen, mutually disjoint paper-development seed splits."""
    development = _spawn(20260715, 20)
    validation = _spawn(20260716, 100)
    heldout = _spawn(20260717, 400)
    if (
        set(development) & set(validation)
        or set(development) & set(heldout)
        or set(validation) & set(heldout)
    ):
        raise RuntimeError("generated MKSC-GI seed splits overlap")
    return {
        "development": development,
        "validation": validation,
        "heldout": heldout,
        "regression": copy.deepcopy(REGRESSION_SEEDS),
        "policy": {
            "development": "debug only",
            "validation": "budget and preset selection",
            "heldout": "run once after preset freeze; no retuning",
            "regression": "failure replay only; never pooled into final statistics",
        },
    }
