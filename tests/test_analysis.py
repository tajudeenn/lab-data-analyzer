import numpy as np
import pandas as pd
import pytest
from lab_analyzer.analysis import clean_signal, detect_peaks, read_csv_bytes
from lab_analyzer.baseline import estimate_baseline_asls

def test_clean_signal_coerces_drops_sorts_and_averages_duplicates():
    frame = pd.DataFrame({"time": [2, 1, 1, 3, 4, 5], "absorbance": [4, 2, 4, "bad", np.inf, 7]})
    cleaned, summary = clean_signal(frame, "time", "absorbance")
    assert cleaned.to_dict("list") == {"x": [1, 2, 5], "signal": [3.0, 4.0, 7.0]}
    assert summary.removed_rows == 2
    assert summary.duplicate_x_values == 2

def test_clean_signal_rejects_too_few_distinct_values():
    with pytest.raises(ValueError, match="distinct x"):
        clean_signal(pd.DataFrame({"x": [1, 1, 2], "y": [2, 3, 4]}), "x", "y")

def test_clean_signal_requires_different_columns():
    with pytest.raises(ValueError, match="different columns"):
        clean_signal(pd.DataFrame({"x": [1, 2, 3]}), "x", "x")

def test_detect_peaks_finds_peak_and_integrates_above_baseline():
    x = np.linspace(0, 10, 1001)
    y = 2 + 5 * np.exp(-0.5 * ((x - 5) / 0.5) ** 2)
    result = detect_peaks(x, y, prominence=1, distance=10)
    assert len(result) == 1
    assert result.iloc[0]["x"] == pytest.approx(5.0)
    assert result.iloc[0]["area"] == pytest.approx(5 * 0.5 * np.sqrt(2 * np.pi), rel=0.03)

def test_detect_peaks_returns_empty_table_for_flat_signal():
    result = detect_peaks(np.arange(10), np.ones(10))
    assert result.empty
    assert "area" in result.columns

def test_detect_peaks_rejects_non_increasing_x():
    with pytest.raises(ValueError, match="strictly increasing"):
        detect_peaks(np.array([0, 1, 1, 3]), np.array([0, 1, 0, 0]))

def test_read_csv_bytes_supports_semicolon_delimiter_and_bom():
    frame = read_csv_bytes("time;signal\n0;1\n1;2\n".encode("utf-8-sig"))
    assert list(frame.columns) == ["time", "signal"]
    assert frame["signal"].tolist() == [1, 2]

def test_asls_estimates_sloped_baseline_under_positive_peaks():
    x = np.linspace(0, 100, 501)
    baseline = 0.2 + 0.002 * x
    signal = baseline + np.exp(-0.5 * ((x - 30) / 3) ** 2) + 0.6 * np.exp(-0.5 * ((x - 72) / 5) ** 2)
    estimated = estimate_baseline_asls(signal, smoothness=1_000_000, asymmetry=0.01)
    corrected = signal - estimated
    assert np.mean(np.abs(estimated - baseline)) < 0.02
    assert corrected.max() == pytest.approx(1.0, rel=0.05)

def test_asls_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="asymmetry"):
        estimate_baseline_asls(np.ones(5), asymmetry=0.5)
