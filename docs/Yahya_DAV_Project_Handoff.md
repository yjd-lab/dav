# Industrial Machine Failure Analyzer
## Project and Implementation Handoff for Yahya

**Project:** Industrial Machine Failure Analyzer  
**Subject:** Data Analytics and Visualization (DAV)  
**Team:** Yahya and Amirul | Third year, fifth semester  
**Repository:** https://github.com/yjd-lab/dav  
**Branches:** `main` (integrated), `yahya` (Yahya), `amirul` (Amirul)  
**Delivery window:** approximately 10 days beginning 10 October 2026  
**Status:** Planning and contract agreed in discussion. The present state of GitHub files and branches has **not** been verified by this document.

> **Purpose:** This is Yahya's stand-alone specification and coding-assistant handoff for the DAV project. It matches the integration contract in Amirul's report, but concentrates on Yahya's advanced analytics, failure-classification module, independent testing and integration support. **Do not silently change the public data schema, paths, or function signatures.**

## 1. What we're building

The project imports a **synthetic industrial machinery dataset**, verifies and cleans the observations, computes descriptive statistics and transformations, and creates plots that help explain relationships between machine operating measurements and recorded failures. A separate callable classifier takes a machine's readings and predicts the dataset's `machine_failure` label. The central DAV deliverable is **data preparation + statistical exploration + visualizations + interpretable output**, not a web site.

**Research question:** Which measured operating conditions and engineered features are associated with recorded machinery failures, and how can the patterns be communicated using analytics and visualizations?

**DAV syllabus mapping:** Lab 2 = missing data, duplicates and outliers; Lab 3 = Pandas/NumPy transformations, statistics, encoding and scaling; Lab 7 = line/scatter/distribution/relationship visualizations. Prediction is an added analytical feature. We are **not** implementing a real-time IoT device, external API, SQL warehouse, or ARIMA time-series system in the initial deliverable.

```text
Official UCI CSV (immutable)
        |
        v
Amirul: import -> validate -> audit -> clean
        |
        +--> data/processed/ai4i_cleaned.csv  <--- COMMON DATA CONTRACT
        |                  |
        |                  +--> Yahya: engineer_features(df)
        |                  +--> Yahya: train_failure_model(df, model_path)
        |                  +--> Yahya: plot_failure_insights(df, output_dir)
        |                                          |
        |                                          +--> models/failure_model.joblib
        |                                          +--> model_metrics.json + fig_y_*.png
        |
Amirul: DAV transformations/stats/EDA -> main DAV charts
        |
Amirul: integrates all public functions in scripts/run_pipeline.py
        |
Final notebook + processed CSVs + JSON reports + PNG graphs
```

## 2. Dataset: exact scientific facts and limitations

**Source:** [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), distributed with CC BY 4.0 attribution. There are **10,000 synthetic observations** and **14 source columns**. UCI reports no missing values in the official original; a separately labeled deliberately dirty copy may be used to demonstrate cleaning. **Do not claim real measured factory data or real-time readings.**

| Original field | Shared canonical field | Meaning |
|---|---|---|
| UID | `uid` | Unique integer record identifier (not a real date/time) |
| Product ID | `product_id` | Product identifier; do not use as predictive signal |
| Type | `type` | Categorical product type: `L`, `M`, `H` |
| Air temperature [K] | `air_temperature_k` | Kelvin |
| Process temperature [K] | `process_temperature_k` | Kelvin |
| Rotational speed [rpm] | `rotational_speed_rpm` | Revolutions/minute |
| Torque [Nm] | `torque_nm` | Newton-metres |
| Tool wear [min] | `tool_wear_min` | Minutes |
| Machine failure | `machine_failure` | Binary target, 0 or 1 |
| TWF, HDF, PWF, OSF, RNF | `twf`, `hdf`, `pwf`, `osf`, `rnf` | Outcome-derived failure-mode indicators; descriptive use only |

**Critical caveats:** No time stamps exist; `uid` ordering is **not proof of elapsed time**. The task is contemporaneous **failure classification**, not predicting how many days remain before failure. Failure-class imbalance means accuracy alone is inadequate. Extreme sensor readings could represent meaningful failure conditions: **flag, don't indiscriminately remove**. Do not include any target label or mode flag in model inputs, otherwise the model will leak the answer.

## 3. Responsibilities and boundaries

### Yahya's primary deliverables

1. **Advanced derived features:** implement `engineer_features(df)` using physically interpretable combinations (e.g., temperature difference `process_temperature_k - air_temperature_k` and rotational power proxy `torque_nm * rotational_speed_rpm`), with clear units/meaning and no label leakage. Keep the original columns intact. Avoid questionable interpretations of units.
2. **Failure-focused analytics:** explore differences in temperature/torque/speed/wear between failing and non-failing observations; report descriptive associations, not causation.
3. **Prediction module:** implement a reproducible **Decision Tree** baseline using the six approved raw inputs. Optionally add SVM later only after end-to-end integration is stable; no SVM requirement for minimum deliverable.
4. **Sound evaluation:** stratified train/test split, fitted preprocessing inside a scikit-learn Pipeline/ColumnTransformer (training split only), confusion matrix, precision, recall, F1 and accuracy; clearly label evaluated test-set results.
5. **Stable prediction entry point:** write `predict_one(readings: dict, model_path: str) -> dict` with exactly the agreed payload and result keys; save trained artifact at an agreed path.
6. **Advanced PNG graphs:** export callable failure-oriented figures with prefix `fig_y_` so they do not overwrite Amirul's figures.
7. **Unit/contract tests:** verify correct feature set, no leakage, schema validation and JSON-safe return types. When Amirul encounters integration defects in Yahya code, **Yahya fixes those defects**, then sends an updated branch/PR.
8. **Shared responsibilities:** review pull requests, explain analytic choices in documentation, help rehearse demonstration and presentation.

### Amirul's primary deliverables (context, not Yahya's implementation assignment)

- Raw dataset download/loading, strict schema validation, data-quality report, cleaning/dirty-data demonstration.
- DAV transformations, statistical summaries, and six core Matplotlib/Seaborn figures.
- **Final integration ownership:** `scripts/run_pipeline.py`, `notebooks/04_final_demo.ipynb`, end-to-end runs, final merge coordination to `main`.

**No workload shares or percentages** are assigned. Files belong to their module owners; both students should understand the complete system. Amirul integrates, but he is **not** responsible for fixing Yahya's code internally.

## 4. Project layout and file ownership

```text
dav/
├── README.md                         # coordinated edits
├── requirements.txt                  # coordinated edits
├── .gitignore
├── docs/
│   ├── contracts.md                  # same contract as this handoff
│   └── data_dictionary.md
├── data/
│   ├── raw/ai4i2020.csv              # official source, immutable
│   ├── demo/ai4i_dirty_demo.csv      # explicitly artificial errors
│   └── processed/
│       ├── ai4i_cleaned.csv          # Amirul produces; Yahya consumes
│       └── ai4i_transformed.csv      # Amirul's DAV showcase only
├── src/industrial_analyzer/
│   ├── __init__.py
│   ├── data_loader.py                 # Amirul
│   ├── validation.py                  # Amirul
│   ├── cleaning.py                    # Amirul
│   ├── transformation.py              # Amirul
│   ├── statistics.py                  # Amirul
│   ├── visualization.py               # Amirul
│   ├── features.py                    # YAHYA
│   ├── prediction.py                  # YAHYA
│   └── advanced_analysis.py           # YAHYA
├── scripts/
│   ├── create_dirty_demo.py          # Amirul
│   └── run_pipeline.py               # Amirul, final orchestration
├── notebooks/
│   ├── 01_data_quality_and_cleaning.ipynb # Amirul
│   ├── 02_transformation_and_statistics.ipynb # Amirul
│   ├── 03_failure_insights.ipynb      # YAHYA
│   └── 04_final_demo.ipynb            # Amirul
├── models/failure_model.joblib        # generated by Yahya's module
├── outputs/figures/                 # fig_*.png vs fig_y_*.png
├── outputs/tables/
├── outputs/reports/                 # quality and model JSON
└── tests/
    ├── test_contract.py             # shared, coordinate changes
    └── test_prediction.py           # YAHYA
```

**Git rule:** Do not both edit the final notebook. Yahya's module should be plain importable Python functions, not code copied from his notebook. The `models/` and most `outputs/` contents are generated; avoid committing large/binary/temporary artifacts unless required for final demonstration and intentionally reviewed.

## 5. AUTHORITATIVE V1 DATA CONTRACT (must match Amirul)

### A. Canonical shared clean input file

**Path:** `data/processed/ai4i_cleaned.csv`  
**Encoding/CSV rules:** UTF-8, comma delimiter, exactly one header row, `index=False`, no extra `Unnamed: 0` field.  
**Rows:** `uid` is unique integer, generally sorted ascending; no nulls in required 14 fields.  
**Schema:** the fourteen canonical fields listed in section 2, sensor measurements in **original units**.  
**`type`:** only `L`, `M`, `H`; failure fields are integers 0 or 1.  
**Model inputs:** **only** `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`.  
**Model target:** `machine_failure`.  
**Never use as predictors:** `uid`, `product_id`, `machine_failure`, `twf`, `hdf`, `pwf`, `osf`, `rnf`.

Example row is **illustrative only**, not proof of a source data value:

```csv
uid,product_id,type,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_nm,tool_wear_min,machine_failure,twf,hdf,pwf,osf,rnf
1,L00001,L,298.1,308.6,1551,42.8,0,0,0,0,0,0,0
```

**Very important:** Amirul's `ai4i_transformed.csv` has separate standardization, min-max encoding and bins for DAV display. **Do not use it as your model-training input**: it may contain statistics fitted on the full dataset. Use `ai4i_cleaned.csv`, then fit any model preprocessing **inside each training fold** only.

### B. Python function API contract: Yahya publishes these EXACT signatures

```python
# src/industrial_analyzer/features.py
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a NEW dataframe; preserve all original columns and row order."""

# src/industrial_analyzer/prediction.py
def train_failure_model(df: pd.DataFrame, model_path: str) -> dict:
    """Train/evaluate model and save pipeline; return JSON-safe artifact metadata."""

def predict_one(readings: dict, model_path: str) -> dict:
    """Classify one row of six raw physical-value inputs using saved pipeline."""

# src/industrial_analyzer/advanced_analysis.py
def plot_failure_insights(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Write fig_y_*.png and return repo-relative or documented paths."""
```

**Required `train_failure_model` return keys:** `model_path` (str), `metrics` (dict of computed test-set values), `schema_version` (string literal `"1.0"`). The function must also write `outputs/reports/model_metrics.json` as valid JSON (coordinate any alternate path before coding). Use consistent results keys and native Python scalar types; avoid leaving `numpy.int64` values that cannot be serialized by the JSON encoder.

### C. Prediction input and output

**Input to `predict_one`: exactly these six keys**, in original units:

```json
{
  "type": "L",
  "air_temperature_k": 298.1,
  "process_temperature_k": 308.6,
  "rotational_speed_rpm": 1551,
  "torque_nm": 42.8,
  "tool_wear_min": 0
}
```

**Output shape (illustrative result; actual values must come from running the model):**

```json
{
  "schema_version": "1.0",
  "predicted_failure": 0,
  "failure_probability": null,
  "model_name": "decision_tree",
  "message": "Model classification; not a certified equipment diagnosis"
}
```

**Validation rules:** reject missing or unexpected input keys, invalid categories, nonnumeric/NaN/infinite numeric values or impossible physical values with a helpful `ValueError`. Never quietly substitute default readings. Return `predicted_failure` as Python integer `0` or `1`, and `failure_probability` as Python float in `[0,1]` **only if** a valid `predict_proba` estimate is available; otherwise `None` (JSON `null`). Distinguish model probability from guaranteed empirical risk and do not hardcode or make up predictions.

### D. Additional agreed generated outputs

| File / pattern | Owner | Purpose |
|---|---|---|
| `data/processed/ai4i_cleaned.csv` | Amirul | Canonical model and analytics source |
| `data/processed/ai4i_transformed.csv` | Amirul | Scaling/normalization/encoding/binning demo |
| `outputs/reports/data_quality.json` | Amirul | Source audit; before/after quality counts |
| `outputs/tables/summary_statistics.csv` | Amirul | Numeric summary for notebook |
| `outputs/tables/failure_by_type.csv` | Amirul | Group comparisons |
| `outputs/tables/failure_modes.csv` | Amirul | Descriptive failure-mode frequencies |
| `outputs/figures/fig_*.png` | Amirul | Core plots |
| `models/failure_model.joblib` | Yahya | Saved fitted prediction pipeline |
| `outputs/reports/model_metrics.json` | Yahya | Real computed test-set metrics |
| `outputs/figures/fig_y_*.png` | Yahya | Advanced failure-oriented plots |
| `outputs/reports/run_manifest.json` | Amirul | End-to-end integration outputs manifest |

**No HTTP server or REST contract**. A plain Python import + agreed CSV/JSON is the integration interface.

## 6. Yahya's exact development plan

### Stage 1 — Inspect and validate inputs

1. Clone/open repo and check whether `main`, `yahya`, and `amirul` already exist. Do not create duplicates or assume they exist.
2. Confirm `docs/contracts.md` matches the v1 contract. If missing, coordinate with Amirul to commit it to `main` before parallel development.
3. Download/inspect the UCI data in your own branch if `data/processed/ai4i_cleaned.csv` is not yet generated; use a temporary in-memory canonical-column mapping for development, but **do not implement Amirul's cleaning pipeline in your files**.
4. Write a small input validator for the six feature columns and binary target. Reject bad schema immediately.

### Stage 2 — Engineer interpretable features

Implement `engineer_features()` without modifying its argument in-place. Suggested columns:

- `temperature_difference_k = process_temperature_k - air_temperature_k`
- `torque_speed_product = torque_nm * rotational_speed_rpm` (a proportional operating-load proxy, **not mechanical power in watts** unless you explicitly apply the angular-speed conversion factor)

These added features are **for advanced descriptive analysis** initially. Keep the six-feature prediction contract stable; do not force Amirul to start supplying new features.

### Stage 3 — Train a classification baseline responsibly

1. Build input `X` from the six *allowlisted* features only and `y = machine_failure`.
2. Use `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` (suitable default, not mandatory if both agree otherwise).
3. For `type`, use `OneHotEncoder(handle_unknown="ignore")`. Use a `ColumnTransformer`/`Pipeline`. A Decision Tree generally **does not require scaling**, so do not add a redundant global scaler just for appearances.
4. Train `DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=...)`; choose reasonable depth based on training-only validation or GridSearchCV, not on the held-out test set.
5. Evaluate on the held-out set: accuracy, precision, recall, F1, confusion matrix and failed-case support. Compare against a simple majority-class/no-failure baseline if practical. Avoid claiming that a high accuracy proves good failure detection.
6. Save the *whole fitted pipeline* (including encoder) to `models/failure_model.joblib` with `joblib.dump`, and write real metrics to JSON. Load this same fitted pipeline in `predict_one`.
7. Only if time permits, add SVM as a controlled comparison (scaling in the SVM pipeline, fit on training data only); do **not** change the primary `predict_one` contract or block the minimum viable project.

**Avoid false claims:** The dataset is synthetic, failures are rare, and this is snapshot label classification. It is not a certified preventive-maintenance device nor a forecast of remaining useful life.

### Stage 4 — Advanced analytics and figures

Implement `plot_failure_insights(df, output_dir)` producing two or three useful figures named `fig_y_*.png`, for example:

- `fig_y_01_sensor_by_failure.png`: torque/tool wear/temperature distributions grouped by label.
- `fig_y_02_feature_relation.png`: torque versus RPM with failure as hue, maybe density/alpha to manage overlap.
- `fig_y_03_model_confusion_matrix.png`: generate this **only if model test predictions have been computed** (avoid requiring it from `df` if your function contract contains no model). Alternatively generate the confusion matrix in `train_failure_model` and save it in the same prefix namespace.

Write figures deterministically into the specified directory. Use axis units, labels, readable legends and count denominators. Do not plot `uid` as calendar time or call a failure-rate correlation a causal relationship.

### Stage 5 — Independent unit tests and handover to Amirul

In `tests/test_prediction.py` verify:

- Missing/extra fields and unknown machine type lead to sensible errors.
- Training uses exactly the allowlisted inputs (no `uid`, product IDs, labels or failure flags).
- `train_failure_model` writes the model file and returns the three required keys.
- `predict_one` accepts the agreed JSON-like dict and returns the required keys and correct Python dtypes.
- `predicted_failure in {0, 1}`, `schema_version == "1.0"`; probabilities, if present, are numeric in `[0, 1]`.
- The advanced plotting function writes nonempty PNGs with `fig_y_` names.
- Train/evaluation runs reproducibly with fixed seed.

Send Amirul a short usage example and a PR into `main`. After merge, if integration fails in `prediction.py`/`features.py`/`advanced_analysis.py`, fix it in the Yahya branch and submit another small PR. Amirul should **not** need to copy notebook cells or debug your internals.

## 7. Integration snippet Amirul should be able to run

```python
import pandas as pd
from industrial_analyzer.features import engineer_features
from industrial_analyzer.prediction import train_failure_model, predict_one
from industrial_analyzer.advanced_analysis import plot_failure_insights

clean = pd.read_csv("data/processed/ai4i_cleaned.csv")
features = engineer_features(clean)
model_info = train_failure_model(clean, "models/failure_model.joblib")

example = {
    "type": "L",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}
prediction = predict_one(example, model_info["model_path"])
figure_paths = plot_failure_insights(features, "outputs/figures")

assert model_info["schema_version"] == "1.0"
assert prediction["predicted_failure"] in (0, 1)
assert all("fig_y_" in str(path) for path in figure_paths)
```

**Environment detail:** The imports above assume `src` is on Python's import path (e.g., editable installation or `PYTHONPATH=src`); document the selected approach in `README.md` rather than asking Amirul to add mysterious notebook-only path hacks. This is an *integration example*, not evidence of an actual trained prediction.

## 8. GitHub collaboration workflow

- **Repository:** https://github.com/yjd-lab/dav (check actual remote state when Codex starts).
- **Branches:** `main` stable; Yahya develops on `yahya`; Amirul develops on `amirul`.
- **Before parallel work:** agree on and merge common scaffold, package layout, requirements, `.gitignore`, and `docs/contracts.md` into `main`. Branch from the same baseline or sync from `main`.
- **Yahya owns:** `src/industrial_analyzer/features.py`, `prediction.py`, `advanced_analysis.py`, `notebooks/03_failure_insights.ipynb`, `tests/test_prediction.py`.
- **Avoid shared-file conflicts:** coordinate before touching `README.md`, `requirements.txt`, `__init__.py`, `docs/`, and shared contract tests. **Do not edit** `notebooks/04_final_demo.ipynb` or `scripts/run_pipeline.py` without Amirul's request.
- **Commit/PR sequence:** (1) derived features + tests, (2) baseline classifier + prediction tests, (3) advanced plots + final usage docs. Create small pull requests from `yahya` to `main`; Amirul reviews/coordinates integration and the merged full run.
- **After a merge:** while on your branch, `git fetch origin` and `git merge origin/main` (or a team-agreed rebase strategy). Avoid force-pushing `main`.
- **Git hygiene:** ignore `.venv/`, `__pycache__/`, checkpoints, caches, `models/*.joblib`, throwaway data. Commit reproducible source code and small required artifacts; coordinate whether example output files are committed as evidence.

## 9. Ten-day implementation schedule (suggested)

| Day | Yahya's milestone | Interface checkpoint |
|---|---|---|
| 1 | Review scaffold, UCI schema and agreed contracts | Both branches have baseline |
| 2 | Implement `engineer_features` and its tests | Amirul confirms names and imports |
| 3 | Model input validation + training split + pipeline | Clean CSV schema agreed |
| 4 | Decision Tree fit and hold-out metrics | Baseline output generated |
| 5 | Model serialization + `predict_one` | Amirul can call prediction with six keys |
| 6 | Advanced statistics and plots | PNG filenames don't collide |
| 7 | Contract smoke tests + PR review | Module accepted into `main` |
| 8 | Support Amirul's integration/testing | Full `run_pipeline.py` succeeds |
| 9 | Improve explanations, visuals and viva Q&A | Full notebook is demonstrable |
| 10 | Final bug fixes and backup | Freeze and rehearse |

**Schedule protection:** If anything slips, **keep the Decision Tree + clean CSV + six DAV figures + final notebook**. Skip SVM, extensive tuning, DBs, websites, interactive widgets, and decorative extras before compromising reliability.

## 10. Definition of done for Yahya's portion

- [ ] `features.py`, `prediction.py`, and `advanced_analysis.py` are importable standalone Python modules.
- [ ] Correct canonical input features and target; no failure-mode leakage.
- [ ] `engineer_features` preserves input rows and original fields.
- [ ] Deterministic train/test split and no preprocessing fitted using the held-out test set.
- [ ] Save a trained pipeline to `models/failure_model.joblib` and JSON metrics to `outputs/reports/model_metrics.json`.
- [ ] `predict_one` accepts exactly six sensor/category fields and returns all four defined keys plus `schema_version`.
- [ ] Two or more failure-oriented figures are exported with `fig_y_` prefix.
- [ ] `tests/test_prediction.py` passes and includes an input-contract smoke test.
- [ ] Pull requests merge cleanly; Amirul can import/call your functions without changing internals.
- [ ] Both students can explain the variables, cleaning rules, statistics, prediction and model limits.

## 11. Viva explanation: short, accurate answers

**Q: What is the aim?** To analyze a synthetic industrial machine dataset, perform data preparation and transformations, visualize operating conditions and the recorded failure label, and demonstrate an interpretable failure classifier.

**Q: What does Yahya's module contribute to DAV?** Derived quantitative features, failure-conditioned comparisons, additional charts, and a callable predictive classification demo with evaluation metrics.

**Q: Why Decision Tree?** It classifies from numeric/categorical features and its splitting logic can be interpreted. It is a sensible first baseline for a small synthetic dataset; we verify its effectiveness empirically rather than assuming it will outperform others.

**Q: Why do we need precision/recall?** Failures are the minority class. High accuracy can come from always predicting no failure; recall measures detection of actual failures, and precision measures the fraction of predicted failures that were failures.

**Q: Why not pass scaled CSV from DAV into the model?** Fitting a scaler on all data before training/test splitting leaks information into evaluation. We feed raw physical values and fit transformations only on training partitions inside the saved model pipeline.

**Q: Is this future forecasting?** No. AI4I contains observations without real timestamps. This is a classification of failure state from measured operating conditions, not an estimate of time until breakdown.

**Q: Why split work into `.py` modules?** So Amirul can import independently tested functions into the final notebook and pipeline; notebooks alone are fragile integration boundaries.

## 12. Copy-paste Codex / coding-LLM starting prompt

> I am Yahya, collaborating with Amirul on our third-year fifth-semester **DAV Industrial Machine Failure Analyzer**. Read this entire handoff as the project specification. Repository: `https://github.com/yjd-lab/dav`; I own branch `yahya`, Amirul owns `amirul` and coordinates integration to `main`. Focus only on the DAV project. Our common input is the UCI AI4I synthetic 2020 dataset, cleaned to `data/processed/ai4i_cleaned.csv` containing exactly the canonical fourteen snake_case columns. My files are `src/industrial_analyzer/features.py`, `prediction.py`, `advanced_analysis.py`, `notebooks/03_failure_insights.ipynb`, and `tests/test_prediction.py`. Amirul owns loading, cleaning, DAV transformations, six core plots, `scripts/run_pipeline.py` and final notebook. Implement the exact callable APIs `engineer_features(df) -> DataFrame`, `train_failure_model(df, model_path) -> dict`, `predict_one(readings, model_path) -> dict`, and `plot_failure_insights(df, output_dir) -> list[str]`. Six approved classifier inputs are `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`; target is `machine_failure`. Do not use UID, Product ID or failure-mode flags as predictive inputs. Use a leakage-safe stratified training/test split, Decision Tree as the first model, training-only preprocessing, correct metrics for rare failures, a saved scikit-learn pipeline, JSON-safe output, and `fig_y_` plot filenames. No REST API, web app, fake timestamps, or automatic elimination of true extreme values. **First inspect the actual repo and branches; do not assume files exist.** Then implement in small tested milestones, commit only to `yahya`, and propose a PR for Amirul to merge. Keep beginner-readable code and explain each step. Never claim tests passed unless run.

## 13. Start-here steps before coding

1. Share or commit the **same `docs/contracts.md`** used by both partners and check repo permissions/branches.
2. Install and document a compatible Python environment (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `pytest`, Jupyter).
3. Create one tiny schema/contract test and one read-only data preview.
4. Implement `features.py` first, then `prediction.py` baseline, then `advanced_analysis.py`.
5. Send Amirul a copy-pastable usage snippet as soon as `predict_one` works, rather than waiting for all plots.
6. Preserve computed results; never invent metrics, figures or source observations.

### Trusted references

- UCI AI4I dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Scikit-learn `Pipeline`: https://scikit-learn.org/stable/modules/compose.html#pipeline
- Scikit-learn `ColumnTransformer`: https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data
- Classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- GitHub pull requests: https://docs.github.com/en/pull-requests/get-started/about-pull-requests

**Governance:** Version 1.0 of the canonical schema and public functions in this handoff intentionally matches Amirul's report. Any proposed change to shared filenames, columns, signatures, ownership or integration strategy must be approved by both before implementation.
