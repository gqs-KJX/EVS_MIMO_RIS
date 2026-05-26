"""Metric helpers for the single run."""

from __future__ import annotations

import numpy as np


def rmse_tensor(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """Return ||Y_hat - Y_true||_F / sqrt(Y_true.size)."""
    assert y_hat.shape == y_true.shape, "Y_hat and Y_true shapes must match"
    return float(np.linalg.norm(y_hat - y_true) / np.sqrt(y_true.size))


def rmse_abs(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Return ||x_hat - x_true||_F / sqrt(x_true.size)."""
    assert x_hat.shape == x_true.shape, "input shapes must match"
    return float(np.linalg.norm(x_hat - x_true) / np.sqrt(x_true.size))


def relative_nmse(x_hat: np.ndarray, x_true: np.ndarray, eps: float = 1e-12) -> float:
    """Return ||x_hat - x_true||_F^2 / ||x_true||_F^2."""
    assert x_hat.shape == x_true.shape, "input shapes must match"
    return float(np.linalg.norm(x_hat - x_true) ** 2 / (np.linalg.norm(x_true) ** 2 + eps))


def position_rmse(p_hat: np.ndarray, p_true: np.ndarray) -> float:
    """Return Euclidean position error in meters."""
    assert p_hat.shape == p_true.shape == (3,), "positions must have shape (3,)"
    return float(np.linalg.norm(p_hat - p_true))
