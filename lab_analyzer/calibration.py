"""Linear calibration models for concentration versus analytical response."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationFit:
    slope: float
    intercept: float
    r_squared: float


def prepare_calibration_data(
    frame: pd.DataFrame, concentration_column: str, response_column: str
) -> tuple[pd.DataFrame, int]:
    """Select numeric standards, drop invalid pairs, and retain replicates."""
    if concentration_column == response_column:
        raise ValueError("Choose different columns for concentration and response.")
    missing = [name for name in (concentration_column, response_column) if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing selected column(s): {', '.join(missing)}")
    selected = frame[[concentration_column, response_column]].copy()
    selected.columns = ["concentration", "response"]
    selected["concentration"] = pd.to_numeric(selected["concentration"], errors="coerce")
    selected["response"] = pd.to_numeric(selected["response"], errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    removed = len(frame) - len(selected)
    if len(selected) < 3:
        raise ValueError("At least three valid standard measurements are required.")
    if selected["concentration"].nunique() < 3:
        raise ValueError("Standards must cover at least three distinct concentrations.")
    if (selected["concentration"] < 0).any():
        raise ValueError("Standard concentrations cannot be negative.")
    if (selected["response"] < 0).any():
        raise ValueError("Standard responses cannot be negative.")
    return selected.reset_index(drop=True), removed


def fit_linear_calibration(concentration: np.ndarray, response: np.ndarray) -> CalibrationFit:
    """Fit response = slope * concentration + intercept and calculate R²."""
    x = np.asarray(concentration, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Concentration and response must be one-dimensional arrays of equal length.")
    if len(x) < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("At least three finite standard measurements are required.")
    if np.unique(x).size < 3:
        raise ValueError("Standards must cover at least three distinct concentrations.")
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Standard concentrations and responses must be nonnegative.")
    if np.ptp(y) == 0:
        raise ValueError("Responses must vary to fit a calibration curve.")

    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError("Calibration response must increase with concentration (positive slope).")
    predicted = slope * x + intercept
    residual_sum_squares = float(np.sum((y - predicted) ** 2))
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return CalibrationFit(float(slope), float(intercept), float(r_squared))


def estimate_concentration(response: float, fit: CalibrationFit) -> float:
    """Invert a fitted positive-slope calibration line for an unknown response."""
    response = float(response)
    if not np.isfinite(response) or response < 0:
        raise ValueError("Unknown response must be a finite nonnegative number.")
    return (response - fit.intercept) / fit.slope
