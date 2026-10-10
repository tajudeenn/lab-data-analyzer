"""CSV preparation, peak detection, and baseline-corrected integration."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, peak_prominences

@dataclass(frozen=True)
class CleaningSummary:
    input_rows: int
    removed_rows: int
    duplicate_x_values: int

def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    """Read a CSV with common instrument export encodings and delimiters."""
    try:
        return pd.read_csv(BytesIO(contents), sep=None, engine="python", encoding="utf-8-sig", comment="#")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not read this CSV: {exc}") from exc

def clean_signal(frame: pd.DataFrame, x_column: str, y_column: str) -> tuple[pd.DataFrame, CleaningSummary]:
    """Select numeric columns, discard invalid rows, average duplicate x, and sort."""
    if x_column == y_column:
        raise ValueError("Choose different columns for the x and signal values.")
    missing = [name for name in (x_column, y_column) if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing selected column(s): {', '.join(missing)}")
    if len(frame) < 3:
        raise ValueError("At least three data rows are required for analysis.")
    input_rows = len(frame)
    selected = frame[[x_column, y_column]].copy()
    selected.columns = ["x", "signal"]
    selected["x"] = pd.to_numeric(selected["x"], errors="coerce")
    selected["signal"] = pd.to_numeric(selected["signal"], errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
    removed_rows = input_rows - len(selected)
    if len(selected) < 3:
        raise ValueError("Fewer than three valid numeric x/signal pairs remain after cleaning.")
    duplicate_x_values = int(selected["x"].duplicated(keep=False).sum())
    cleaned = selected.groupby("x", as_index=False, sort=True)["signal"].mean()
    if len(cleaned) < 3:
        raise ValueError("At least three distinct x values are required after duplicate values are combined.")
    return cleaned, CleaningSummary(input_rows, removed_rows, duplicate_x_values)

def detect_peaks(x: np.ndarray, signal: np.ndarray, *, prominence: float | None = None, distance: int = 1) -> pd.DataFrame:
    """Detect local maxima and integrate above a linear baseline between prominence bases."""
    x, y = np.asarray(x, dtype=float), np.asarray(signal, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and signal must be one-dimensional arrays of equal length.")
    if len(x) < 3:
        raise ValueError("At least three points are required to detect peaks.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and signal must contain only finite values.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing.")
    if distance < 1:
        raise ValueError("Minimum peak distance must be at least one point.")
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance)
    columns = ["peak", "x", "height", "prominence", "left_x", "right_x", "area"]
    if len(peaks) == 0:
        return pd.DataFrame(columns=columns)
    prominences, left_bases, right_bases = peak_prominences(y, peaks)
    rows = []
    for number, (peak, prom, left, right) in enumerate(zip(peaks, prominences, left_bases, right_bases), start=1):
        if right <= left:
            continue
        sx, sy = x[left:right + 1], y[left:right + 1]
        baseline = np.interp(sx, [x[left], x[right]], [y[left], y[right]])
        area = float(trapezoid(sy - baseline, sx))
        rows.append({"peak": number, "x": float(x[peak]), "height": float(y[peak]), "prominence": float(prom),
                     "left_x": float(x[left]), "right_x": float(x[right]), "area": max(0.0, area)})
    return pd.DataFrame(rows, columns=columns)
