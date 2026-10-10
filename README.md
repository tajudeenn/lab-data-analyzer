# Lab Data Analyzer

A small Python and Streamlit application for exploring instrument CSV exports. The first milestone supports HPLC chromatograms and UV–Vis / FTIR spectra through the same two-column analysis workflow.

## Current capabilities

- Upload CSV data or use the bundled HPLC demonstration.
- Select x and signal columns from multi-column exports.
- Read common comma, tab, or semicolon delimiters, including UTF-8 BOM.
- Coerce numeric values, remove missing/non-finite rows, sort x values, and average repeated x values.
- Plot the cleaned signal interactively and mark detected peaks.
- Set minimum peak prominence and spacing.
- Optionally estimate and subtract a smooth asymmetric-least-squares (AsLS) baseline, with a raw/baseline/corrected overlay and export of all three signals.
- Integrate peaks above a straight line between their prominence bases using trapezoidal integration.
- Download the cleaned signal and peak table as CSV files.
- Run numerical unit tests with pytest.

## Setup

Python 3.10 or newer is required.

```bash
# Windows PowerShell
py -3.13 -m venv .venv
\.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS / Linux: source .venv/bin/activate
python3 -m venv .venv
python -m pip install -r requirements.txt
```

## Run the app

```bash
# Windows PowerShell
\.venv\Scripts\python.exe -m streamlit run app.py
# macOS / Linux
streamlit run app.py
```

Choose **Use bundled demonstration chromatogram** to try the app without an instrument file. For your data, upload a CSV and select the x and signal columns. The x-axis may be time, wavelength, or wavenumber; the signal may be intensity, absorbance, or another numeric response.

## Run tests

```bash
# Windows PowerShell
\.venv\Scripts\python.exe -m pytest
# macOS / Linux
pytest
```

## Project layout

```text
app.py                    Streamlit interface
lab_analyzer/analysis.py  CSV cleaning, peak detection, and integration
data/                     Demonstration instrument data
tests/                    Numerical unit tests
```

## Scientific notes and limitations

Peak detection uses SciPy local maxima and prominence. Each area is the trapezoidal integral of signal above a straight baseline joining that peak's prominence bases; it is reported in x-units × signal-units. Optional AsLS correction estimates a smooth lower baseline for positive-going peaks and works best with roughly evenly spaced x values. It is exploratory and does not replace review of raw and corrected signals. Baseline choice, co-elution, noise, and instrument-specific exports can materially change results. This is not a validated regulated workflow. CSV parsing expects a header row and does not handle every vendor-specific preamble or locale-specific decimal format.

## Roadmap

1. Add calibration curves, R², and LOD / LOQ calculations with validated examples.
2. Support vendor preambles, more export conventions, and user-selected integration boundaries.
3. Add downloadable PDF reports, expanded scientific validation, and portfolio documentation.
