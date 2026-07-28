"""Small numerical helpers used by the demo."""

from __future__ import annotations

import copy
import importlib.util
import numpy as np

# Estimate entries that are bulk model reconstructions rather than optimizer
# state.  They are the largest objects in an estimate by two orders of
# magnitude: ``Z_hat`` alone is ``I * P * L * T`` complex entries (403 MB in the
# paper configuration, against under 1 MB for every parameter array combined).
BULK_RECONSTRUCTION_KEYS = ("Z_hat",)


def copy_estimate(estimate: dict) -> dict:
    """Deep-copy a Stage-I/II estimate, sharing the bulk reconstructions.

    Branches copy an estimate to protect the *parameters* from in-place edits by
    a competing branch.  The reconstruction tensors in
    :data:`BULK_RECONSTRUCTION_KEYS` are only ever replaced wholesale or read for
    a norm --- never written element-wise --- so duplicating them is pure memory
    traffic.  Sharing them by reference removes it and leaves the deep copy of
    everything else, including every parameter array, unchanged.
    """
    if not isinstance(estimate, dict):
        return copy.deepcopy(estimate)
    shared = {
        key: estimate[key] for key in BULK_RECONSTRUCTION_KEYS if key in estimate
    }
    stripped = {key: value for key, value in estimate.items() if key not in shared}
    duplicate = copy.deepcopy(stripped)
    # Rebuild in the original key order so serialization is byte-identical.
    return {
        key: (shared[key] if key in shared else duplicate[key]) for key in estimate
    }


def check_finite(name: str, array: np.ndarray) -> None:
    """Raise a clear error if an array contains NaN or Inf values."""
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} contains NaN or Inf values")


def complex_awgn(shape: tuple[int, ...], variance: float, rng: np.random.Generator) -> np.ndarray:
    """Generate circular complex Gaussian noise with E[|w|^2] = variance."""
    sigma = np.sqrt(variance / 2.0)
    return sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def normalize_columns(matrix: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Normalize columns of a 2-D matrix and return the column norms."""
    assert matrix.ndim == 2, "matrix must be 2-D"
    norms = np.linalg.norm(matrix, axis=0)
    safe = np.maximum(norms, eps)
    return matrix / safe[None, :], norms


def solve_lstsq(design: np.ndarray, target: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """Solve a small possibly regularized least-squares problem."""
    assert design.ndim == 2, "design must be 2-D"
    if reg > 0.0:
        rows = design.shape[1]
        design_aug = np.vstack([design, np.sqrt(reg) * np.eye(rows, dtype=design.dtype)])
        if target.ndim == 1:
            target_aug = np.concatenate([target, np.zeros(rows, dtype=target.dtype)])
        else:
            target_aug = np.vstack(
                [target, np.zeros((rows, target.shape[1]), dtype=target.dtype)]
            )
        return np.linalg.lstsq(design_aug, target_aug, rcond=None)[0]
    return np.linalg.lstsq(design, target, rcond=None)[0]


def scipy_is_available() -> bool:
    """Return True when scipy.optimize can be imported."""
    if importlib.util.find_spec("scipy") is None:
        return False
    return importlib.util.find_spec("scipy.optimize") is not None


def bounded_coordinate_search(
    objective,
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    step0: float = 0.08,
    max_iter: int = 70,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float, dict]:
    """Simple bounded pattern search used only when SciPy is unavailable.

    The search variables are expected to be scaled to comparable ranges.
    """
    x = np.clip(np.asarray(x0, dtype=float), lower, upper)
    step = np.full_like(x, step0, dtype=float)
    best = float(objective(x))
    n_eval = 1

    for it in range(max_iter):
        improved = False
        for dim in range(x.size):
            for sign in (1.0, -1.0):
                trial = x.copy()
                trial[dim] = np.clip(trial[dim] + sign * step[dim], lower[dim], upper[dim])
                value = float(objective(trial))
                n_eval += 1
                if value + 1e-14 < best:
                    x = trial
                    best = value
                    improved = True
        if not improved:
            step *= 0.5
        if np.max(step) < tol:
            break

    info = {
        "success": True,
        "message": "SciPy unavailable; used bounded coordinate-search fallback",
        "n_eval": n_eval,
        "iterations": it + 1,
    }
    return x, best, info
