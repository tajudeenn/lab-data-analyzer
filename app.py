"""Streamlit interface for exploratory HPLC and spectroscopy CSV analysis."""
import plotly.graph_objects as go
import streamlit as st
from lab_analyzer.analysis import clean_signal, detect_peaks, read_csv_bytes

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
    peaks = detect_peaks(cleaned["x"].to_numpy(), cleaned["signal"].to_numpy(),
                         prominence=prominence if prominence > 0 else None, distance=int(distance))
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if summary.removed_rows or summary.duplicate_x_values:
    st.warning(f"Cleaning: removed {summary.removed_rows:,} invalid row(s); found {summary.duplicate_x_values:,} rows with repeated x values and averaged them.")
else:
    st.success("All selected data points passed numeric and finite-value checks.")
fig = go.Figure()
fig.add_trace(go.Scatter(x=cleaned["x"], y=cleaned["signal"], mode="lines", name="Cleaned signal"))
if not peaks.empty:
    fig.add_trace(go.Scatter(x=peaks["x"], y=peaks["height"], mode="markers", name="Detected peaks",
                             marker={"color": "#e4572e", "size": 9}, customdata=peaks[["area", "prominence"]],
                             hovertemplate="x=%{x:.4g}<br>signal=%{y:.4g}<br>area=%{customdata[0]:.4g}<br>prominence=%{customdata[1]:.4g}<extra></extra>"))
fig.update_layout(title=f"{mode}: {y_column} vs {x_column}", xaxis_title=x_column, yaxis_title=y_column,
                  hovermode="x unified", height=470, margin={"l": 10, "r": 10, "t": 55, "b": 10})
st.plotly_chart(fig, use_container_width=True)
st.caption("Areas use trapezoidal integration above a straight baseline between each peak's prominence bases. Units are x-units × signal-units.")
st.subheader(f"Detected peaks ({len(peaks)})")
if peaks.empty:
    st.info("No peaks met these settings. Lower the minimum prominence or spacing to inspect more local maxima.")
else:
    st.dataframe(peaks, use_container_width=True, hide_index=True)
col1, col2 = st.columns(2)
col1.download_button("Download cleaned signal CSV", cleaned.to_csv(index=False).encode("utf-8"),
                      file_name="cleaned_signal.csv", mime="text/csv")
col2.download_button("Download peak results CSV", peaks.to_csv(index=False).encode("utf-8"),
                      file_name="peak_results.csv", mime="text/csv", disabled=peaks.empty)
