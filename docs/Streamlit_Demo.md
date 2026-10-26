# Streamlit demonstration

Run from the repository root using the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501. Stop it with Ctrl+C in its terminal. The app sets up its own src import path. It reads existing outputs and never trains on page navigation or prediction submission.

## Optional frontend dependency proposal

Shared `requirements.txt` is unchanged. `requirements-streamlit.txt` is a separate, reviewable frontend dependency file referencing the existing requirements plus the tested Streamlit version. On a newly approved environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-streamlit.txt
```

The current laptop already has the approved installation. Do not install globally.

## Five-page presentation

1. **Dashboard / Overview:** live counts from the canonical cleaned CSV, source description and preview.
2. **Data Cleaning & Transformation:** official data-quality audit, missing values, duplicates, flagged outliers, actual scaled/encoded/binned columns. Dirty demonstrations are not represented as defects in the official data.
3. **Statistical Analysis:** three generated CSV tables, sensor selection and denominator explanations.
4. **Data Visualization:** six Amirul PNGs and three Yahya PNGs, grouped by owner; select a chart for a larger view. Click its image fullscreen control if needed.
5. **Machine Failure Prediction:** edit the six inputs to update live Plotly comparisons. Numeric edits take effect on Enter or when leaving the field; no Predict click is needed for graphs. A torque/RPM scatter shows all recorded observations and a large current-input diamond; air/process histograms and a wear histogram show moving input lines. Charts support hover and zoom. Historical populations include all product types and are never artificially balanced. Click **Predict Machine Failure** to call the existing `predict_one`; it is never called just because a sensor changed. Changing any input hides the old prediction until resubmission. Prefilling a dataset observation is a demonstration, not an independent model evaluation.

The source is synthetic snapshot data. The application does not predict breakdown dates, establish causation or provide a certified industrial diagnosis. Reported probability is not a calibrated risk estimate.

## Artifact handling

Paths resolve from the source repository, independently of notebook/terminal location. Required artifacts are the canonical cleaned CSV, transformed CSV, three statistics tables, quality/model/manifest reports, nine PNGs and saved pipeline. The app identifies missing paths and explains setup; it never silently fabricates outputs or retrains.

If analysis artifacts are missing, use the existing pipeline from the project root:

```powershell
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python.exe scripts/run_pipeline.py
```

Without a readings file the pipeline generates analytical artifacts and reports `awaiting_readings`. To include CLI prediction, supply `--readings-file` pointing to an approved six-field JSON object. Alternatively, use the app's prediction form after artifacts exist. No refresh/training button is included, avoiding accidental artifact replacement during a demo.

CSV/JSON reads are cached using absolute path, modification timestamp and file size; regenerated files invalidate their cache entry. The backend reloads its saved pipeline on an explicit prediction call, keeping existing validation intact. Changing the form does not retrain. Artifact consistency across independent pipeline runs still requires generating them together; the existing reports do not contain a full provenance hash chain.

## Verification

```powershell
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -q
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

AppTest checks real existing artifacts and the saved classifier, with temporary copies for destructive error cases. Mocked prediction responses are explicitly test-only rendering checks. UI tests skip with a reason if Streamlit or prerequisite artifacts are absent. They do not train a classifier. Existing backend files and the notebook are hash-checked for preservation during UI tests.

Frontend changes are published separately on `feature/streamlit-dashboard` for team review. Review the branch against `main` before an explicitly authorized merge. Keep local notebook edits and generated artifacts out of the frontend commit.
