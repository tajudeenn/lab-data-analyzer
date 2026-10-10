# Lab Data Analyzer

A small Python and Streamlit application for exploring instrument CSV exports. The first milestone supports HPLC chromatograms and UV–Vis / FTIR spectra through the same two-column analysis workflow.

## Current capabilities

- Upload CSV data or use the bundled HPLC demonstration.
- Select x and signal columns from multi-column exports.
- Read common comma, tab, or semicolon delimiters, including UTF-8 BOM.
- Coerce numeric values, remove missing/non-finite rows, sort x values, and average repeated x values.
- Plot the cleaned signal interactively and mark detected peaks.
- Set minimum peak prominence and spacing.
- Integrate peaks above a straight line between their prominence bases using trapezoidal integration.
- Download the cleaned signal and peak table as CSV files.
- Run numerical unit tests with pytest.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Choose **Use bundled demonstration chromatogram** to try the app without an instrument file. For your data, upload a CSV and select the x and signal columns. The x-axis may be time, wavelength, or wavenumber; the signal may be intensity, absorbance, or another numeric response.

## Run tests

```bash
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

Peak detection uses SciPy local maxima and prominence. Each area is the trapezoidal integral of signal above a straight baseline joining that peak's prominence bases; it is reported in x-units × signal-units. This is a transparent exploratory method, not a validated regulated workflow. Baseline choice, co-elution, noise, and instrument-specific exports can materially change results. Review settings and inspect the plot before interpreting areas. CSV parsing expects a header row and does not handle every vendor-specific preamble or locale-specific decimal format.

## Roadmap

1. Add configurable baseline correction and compare raw versus corrected signal.
2. Add calibration curves, R², and LOD / LOQ calculations with validated examples.
3. Support vendor preambles, more export conventions, and user-selected integration boundaries.
4. Add downloadable PDF reports, expanded scientific validation, and portfolio documentation.
