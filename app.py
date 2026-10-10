"""Streamlit interface for exploratory HPLC and spectroscopy CSV analysis."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from lab_analyzer.analysis import clean_signal, detect_peaks, read_csv_bytes
from lab_analyzer.baseline import estimate_baseline_asls

st.set_page_config(page_title="Lab Data Analyzer", page_icon="🧪", layout="wide")
st.title("🧪 Lab Data Analyzer")
st.caption("Explore HPLC chromatograms and UV–Vis / FTIR spectra from CSV exports.")
with st.sidebar:
    st.header("Analysis settings")
    mode = st.selectbox("Data type", ["HPLC chromatogram", "UV–Vis spectrum", "FTIR spectrum", "Other"])
    st.caption("The mode labels the plot; analysis uses your selected x and signal columns.")
    prominence = st.number_input("Minimum peak prominence", min_value=0.0, value=0.0, step=0.1,
                                 help="Set to zero to detect every local maximum.")
    distance = st.number_input("Minimum peak spacing (points)", min_value=1, value=5, step=1)
    apply_baseline = st.checkbox(
        "Apply baseline correction",
        value=False,
        help="Estimate a smooth lower baseline with asymmetric least squares before peak analysis.",
    )
    if apply_baseline:
        baseline_smoothness = st.select_slider(
            "Baseline smoothness (λ)",
            options=[1_000, 10_000, 100_000, 1_000_000, 10_000_000],
            value=1_000_000,
            format_func=lambda value: f"{value:,}",
            help="Higher values produce a smoother, less flexible baseline.",
        )
        baseline_asymmetry = st.select_slider(
            "Baseline asymmetry (p)",
            options=[0.001, 0.01, 0.05, 0.1],
            value=0.01,
            help="Lower values push the estimated baseline farther below positive peaks.",
        )
upload = st.file_uploader("Upload an instrument CSV", type=["csv", "txt"])
use_sample = st.checkbox("Use bundled demonstration chromatogram", value=upload is None)
if upload is not None and not use_sample:
    source_name, contents = upload.name, upload.getvalue()
elif use_sample:
    source_name = "sample_hplc.csv"
    with open("data/sample_hplc.csv", "rb") as sample_file:
        contents = sample_file.read()
else:
    st.info("Upload a CSV or select the bundled demonstration data to begin.")
    st.stop()
try:
    raw = read_csv_bytes(contents)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if len(raw.columns) < 2:
    st.error("The CSV needs at least two columns: an x value and a signal value.")
    st.stop()
st.write(f"**File:** {source_name} · **Rows:** {len(raw):,} · **Columns:** {len(raw.columns)}")
left, right = st.columns(2)
x_column = left.selectbox("X-axis column", options=list(raw.columns), index=0)
y_column = right.selectbox("Signal column", options=list(raw.columns), index=1)
try:
    cleaned, summary = clean_signal(raw, x_column, y_column)
    x_values = cleaned["x"].to_numpy()
    raw_signal = cleaned["signal"].to_numpy()
    estimated_baseline = None
    analysis_signal = raw_signal
    if apply_baseline:
        estimated_baseline = estimate_baseline_asls(
            raw_signal,
            smoothness=float(baseline_smoothness),
            asymmetry=float(baseline_asymmetry),
        )
        analysis_signal = raw_signal - estimated_baseline
    peaks = detect_peaks(x_values, analysis_signal,
                         prominence=prominence if prominence > 0 else None, distance=int(distance))
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if summary.removed_rows or summary.duplicate_x_values:
    st.warning(f"Cleaning: removed {summary.removed_rows:,} invalid row(s); found {summary.duplicate_x_values:,} rows with repeated x values and averaged them.")
else:
    st.success("All selected data points passed numeric and finite-value checks.")
fig = go.Figure()
if estimated_baseline is None:
    fig.add_trace(go.Scatter(x=x_values, y=raw_signal, mode="lines", name="Cleaned signal"))
else:
    fig.add_trace(go.Scatter(x=x_values, y=raw_signal, mode="lines", name="Raw signal",
                             line={"color": "#8a8f98", "dash": "dot"}))
    fig.add_trace(go.Scatter(x=x_values, y=estimated_baseline, mode="lines", name="Estimated baseline",
                             line={"color": "#8b5cf6", "dash": "dash"}))
    fig.add_trace(go.Scatter(x=x_values, y=analysis_signal, mode="lines", name="Baseline-corrected signal",
                             line={"color": "#2563eb"}))
if not peaks.empty:
    fig.add_trace(go.Scatter(x=peaks["x"], y=peaks["height"], mode="markers", name="Detected peaks",
                             marker={"color": "#e4572e", "size": 9}, customdata=peaks[["area", "prominence"]],
                             hovertemplate="x=%{x:.4g}<br>signal=%{y:.4g}<br>area=%{customdata[0]:.4g}<br>prominence=%{customdata[1]:.4g}<extra></extra>"))
fig.update_layout(title=f"{mode}: {y_column} vs {x_column}", xaxis_title=x_column, yaxis_title=y_column,
                  hovermode="x unified", height=470, margin={"l": 10, "r": 10, "t": 55, "b": 10})
st.plotly_chart(fig, width="stretch", alt=f"{mode} plot of {y_column} versus {x_column}")
if estimated_baseline is None:
    st.caption("Areas use trapezoidal integration above a straight baseline between each peak's prominence bases. Units are x-units × signal-units.")
else:
    steps = np.diff(x_values)
    if not np.allclose(steps, np.median(steps), rtol=0.05, atol=1e-12):
        st.warning("AsLS baseline smoothness assumes roughly even x-axis spacing; results may be less reliable for irregularly spaced measurements.")
    st.caption("AsLS baseline correction is enabled. Peak detection and integration use the corrected signal; areas use trapezoidal integration between each peak's prominence bases.")
st.subheader(f"Detected peaks ({len(peaks)})")
if peaks.empty:
    st.info("No peaks met these settings. Lower the minimum prominence or spacing to inspect more local maxima.")
else:
    st.dataframe(peaks, width="stretch", hide_index=True, alt="Detected peak positions, heights, prominence, integration bounds, and areas")
col1, col2 = st.columns(2)
if estimated_baseline is None:
    signal_export = cleaned
    signal_download_label = "Download cleaned signal CSV"
    signal_download_name = "cleaned_signal.csv"
else:
    signal_export = cleaned.assign(
        raw_signal=raw_signal,
        estimated_baseline=estimated_baseline,
        corrected_signal=analysis_signal,
    ).drop(columns="signal")
    signal_download_label = "Download raw and corrected signal CSV"
    signal_download_name = "baseline_corrected_signal.csv"
col1.download_button(signal_download_label, signal_export.to_csv(index=False).encode("utf-8"),
                      file_name=signal_download_name, mime="text/csv")
col2.download_button("Download peak results CSV", peaks.to_csv(index=False).encode("utf-8"),
                      file_name="peak_results.csv", mime="text/csv", disabled=peaks.empty)
