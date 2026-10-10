"""Baseline estimation for evenly sampled scientific signals."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def estimate_baseline_asls(
    signal: np.ndarray,
    *,
    smoothness: float = 100_000.0,
    asymmetry: float = 0.01,
    iterations: int = 10,
) -> np.ndarray:
    """Estimate a smooth lower baseline using asymmetric least squares (AsLS).

    The method is intended for positive-going peaks on approximately evenly
    spaced x values. ``smoothness`` controls baseline curvature and
    ``asymmetry`` controls how strongly points above the baseline are downweighted.
    """
    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or len(y) < 3:
        raise ValueError("Baseline estimation requires a one-dimensional signal with at least three points.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Signal must contain only finite values for baseline estimation.")
    if not np.isfinite(smoothness) or smoothness <= 0:
        raise ValueError("Baseline smoothness must be a finite positive number.")
    if not np.isfinite(asymmetry) or not 0 < asymmetry < 0.5:
        raise ValueError("Baseline asymmetry must be greater than zero and less than 0.5.")
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("Baseline iterations must be a positive integer.")

    size = len(y)
    difference = sparse.diags(
        [np.ones(size - 2), -2 * np.ones(size - 2), np.ones(size - 2)],
        [0, 1, 2],
        shape=(size - 2, size),
        format="csc",
    )
    penalty = smoothness * (difference.T @ difference)
    weights = np.ones(size)
    baseline = np.zeros(size)

    for _ in range(iterations):
        weight_matrix = sparse.diags(weights, format="csc")
        baseline = spsolve(weight_matrix + penalty, weights * y)
        weights = np.where(y > baseline, asymmetry, 1.0 - asymmetry)

    return np.asarray(baseline)
