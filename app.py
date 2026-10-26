"""Local Streamlit presentation of existing DAV outputs and predictions."""

from pathlib import Path
import sys

# The documented Streamlit command needs no external PYTHONPATH setup.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

from industrial_analyzer import dashboard, prediction
from industrial_analyzer.live_charts import build_live_charts


PAGES = ["Dashboard / Overview", "Data Cleaning & Transformation", "Statistical Analysis", "Data Visualization", "Machine Failure Prediction"]
DISCLAIMER = "Model classification based on synthetic training data. Not a certified industrial equipment diagnosis."
LABELS = {
    "air_temperature_k": "Air temperature (K)",
    "process_temperature_k": "Process temperature (K)",
    "rotational_speed_rpm": "Rotational speed (RPM)",
    "torque_nm": "Torque (Nm)", "tool_wear_min": "Tool wear (minutes)",
}


@st.cache_data(show_spinner=False)
def cached_csv(stamp: tuple) -> pd.DataFrame:
    return dashboard.read_csv(stamp[0])


@st.cache_data(show_spinner=False)
def cached_report(stamp: tuple) -> dict:
    return dashboard.read_report(stamp[0])


def csv(relative_path: str) -> pd.DataFrame:
    return cached_csv(dashboard.file_stamp(relative_path))


def report(relative_path: str) -> dict:
    return cached_report(dashboard.file_stamp(relative_path))


def setup_help() -> None:
    st.info("Generate missing analysis artifacts once from the project root. Navigation never runs training.")
    st.code('$env:PYTHONPATH = "src"\n$env:MPLBACKEND = "Agg"\n.\\.venv\\Scripts\\python.exe scripts/run_pipeline.py', language="powershell")
    st.caption("Without --readings-file, the pipeline generates analysis and the model but reports awaiting_readings. Use the prediction form for a classification, or supply a six-field JSON file with --readings-file to complete the CLI prediction step.")


def overview(clean: pd.DataFrame) -> None:
    st.subheader("A clear view of machine failure patterns")
    st.write("Explore recorded operating conditions, inspect the data preparation process, and classify a set of sensor readings with the existing Decision Tree model.")
    columns = [*st.columns(3), *st.columns(2)]
    for column, (label, value) in zip(columns, dashboard.overview_metrics(clean).items()):
        column.metric(label, f"{value:.2f}%" if "rate" in label else f"{value:,}")
    st.info("Synthetic observations • Snapshot classification • No genuine timestamps")
    st.markdown("**Source:** [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)")
    st.subheader("Canonical cleaned-data preview")
    st.dataframe(clean.head(), hide_index=True, width="stretch")
    with st.expander("How to present this project"):
        st.write("Start with data quality and transformations, review descriptive statistics and nine charts, then enter six sensor readings for an actual model classification. Chart associations do not establish causation.")


def cleaning(clean: pd.DataFrame) -> None:
    st.subheader("Data quality, with the source preserved")
    quality = report("outputs/reports/data_quality.json")
    if quality.get("dataset_kind") != "official" or quality.get("status") != "success":
        raise ValueError("The official data-quality report must describe a successful official-data run")
    counts = quality["row_counts"]
    cols = st.columns(3)
    for column, label, value in zip(cols, ("Source records", "Cleaned records", "Removed records"), (counts["input"], counts["output"], counts["removed"])):
        column.metric(label, f"{value:,}")
    st.caption(f"Canonical shape: {clean.shape[0]:,} rows × {clean.shape[1]} columns. The following audit describes the official source, not deliberately corrupted demo data.")
    tabs = st.tabs(["Missing values & duplicates", "Outlier audit", "Full quality report"])
    with tabs[0]:
        st.dataframe(pd.Series(quality["missing_values_by_column"], name="Source missing values").rename_axis("Column").reset_index(), hide_index=True, width="stretch")
        st.write("Source duplicate audit", quality["duplicates"])
        st.caption("Missing values and duplicates are reported from the source audit; no missing values are invented for demonstration.")
    with tabs[1]:
        fields = quality["outliers"]["fields"]
        st.dataframe(pd.DataFrame([{"sensor": name, **{key: values.get(key) for key in ("lower_fence", "upper_fence", "flagged_count", "status")}} for name, values in fields.items()]), hide_index=True, width="stretch")
        st.write("IQR outliers are flagged only. Meaningful extremes remain in the official dataset; per-sensor flags can overlap.")
    with tabs[2]:
        st.json(quality)
    st.subheader("Descriptive transformations")
    transformed = csv("data/processed/ai4i_transformed.csv")
    pd.testing.assert_frame_equal(transformed.loc[:, clean.columns], clean)
    sensor = st.selectbox("Compare a numerical sensor", list(dashboard.SENSORS), format_func=lambda name: LABELS[name])
    columns = [sensor, f"{sensor}_minmax", f"{sensor}_zscore"]
    st.dataframe(transformed[["uid", *columns]].head(20), hide_index=True, width="stretch")
    st.caption("Min-max normalization maps the observed range to [0, 1]. Z-scores use the descriptive dataset mean and population standard deviation. These globally fitted transformations never enter the prediction model.")
    st.dataframe(transformed[["uid", "type", "type_L", "type_M", "type_H", "tool_wear_min", "tool_wear_quantile_bin"]].head(20), hide_index=True, width="stretch")
    st.caption("L/M/H use one-hot indicator columns. Tool wear uses up to four quantile bins, numbered from 1.")
    with st.expander("Cleaned and transformed previews"):
        st.dataframe(clean.head(), hide_index=True, width="stretch")
        st.dataframe(transformed.head(), hide_index=True, width="stretch")


def statistics() -> None:
    st.subheader("Descriptive statistics")
    tables = {}
    for name, expected in dashboard.TABLE_COLUMNS.items():
        table = csv(f"outputs/tables/{name}.csv")
        if list(table.columns) != expected:
            raise ValueError(f"Unexpected column schema in {name}.csv")
        tables[name] = table
    summary = tables["summary_statistics"]
    selected = st.multiselect("Numerical features", summary["feature"].tolist(), default=summary["feature"].tolist())
    st.dataframe(summary[summary["feature"].isin(selected)], hide_index=True, width="stretch")
    st.caption("Mean and standard deviation summarize location and spread; quartiles describe the middle half of observations. Units remain Kelvin, RPM, Nm and minutes.")
    st.subheader("Failure by product type")
    st.dataframe(tables["failure_by_type"], hide_index=True, width="stretch")
    st.caption("Failure rate (%) = failure_count / total_records within that product type × 100.")
    st.subheader("Failure-mode comparison")
    st.dataframe(tables["failure_modes"], hide_index=True, width="stretch")
    st.caption("TWF: tool wear; HDF: heat dissipation; PWF: power; OSF: overstrain; RNF: random failure. Flags can overlap; percentages use all dataset records as the denominator. Flags are excluded from model inputs.")


def visualizations() -> None:
    st.subheader("Nine views of the recorded data")
    st.caption("Existing PNGs are read from disk, never regenerated by navigation. Select one chart for a larger view.")
    selection = st.selectbox("Large chart view", ["All charts", *[title for _, title, _ in dashboard.FIGURES]])
    tabs = st.tabs(["Amirul · General analysis", "Yahya · Failure insights"])
    for tab, figures in zip(tabs, (dashboard.FIGURES[:6], dashboard.FIGURES[6:])):
        with tab:
            visible = [(name, title, explanation) for name, title, explanation in figures if selection == "All charts" or title == selection]
            if not visible:
                st.caption("The selected chart is in the other tab.")
            columns = st.columns(2) if selection == "All charts" else [st.container()]
            for index, (name, title, explanation) in enumerate(visible):
                with columns[index % len(columns)]:
                    st.markdown(f"#### {title}")
                    path = dashboard.artifact_path(f"outputs/figures/{name}")
                    try:
                        with Image.open(path) as image:
                            image.verify()
                        st.image(str(path), caption=explanation, width="stretch")
                    except (OSError, UnidentifiedImageError) as error:
                        st.error(f"Cannot display {name}: {error}")


def prediction_page(clean: pd.DataFrame) -> None:
    st.subheader("Classify one set of machine readings")
    st.caption("Six raw inputs → saved preprocessing pipeline → Decision Tree classification. No training occurs here.")
    defaults = clean.iloc[0][list(prediction.PREDICTION_FEATURES)].to_dict()
    with st.expander("Prefill from an actual dataset observation"):
        uid = st.selectbox("Observation UID (demonstration only)", clean["uid"].tolist(), key="demo_uid")
        st.caption("Using an existing synthetic dataset record is a demonstration, not an independent evaluation.")
        if st.button("Use selected observation"):
            values = clean.loc[clean["uid"] == uid, list(prediction.PREDICTION_FEATURES)].iloc[0].to_dict()
            for name, value in values.items():
                st.session_state[name] = str(value) if name == "type" else float(value)
            st.session_state.pop("last_prediction", None)
    for name, value in defaults.items():
        if name not in st.session_state:
            st.session_state[name] = str(value) if name == "type" else float(value)
    with st.container(border=True):
        readings = {"type": st.selectbox("Product type", ["L", "M", "H"], key="type")}
        columns = st.columns(2)
        for index, name in enumerate(dashboard.SENSORS):
            with columns[index % 2]:
                readings[name] = st.number_input(LABELS[name], step=1.0 if name in ("rotational_speed_rpm", "tool_wear_min") else 0.1, key=name)
        submitted = st.button("Predict Machine Failure", type="primary", width="stretch")
    previous = st.session_state.get("last_prediction")
    if previous is not None and previous["readings"] != readings:
        st.session_state.pop("last_prediction", None)
        st.session_state["prediction_stale"] = True
    if submitted:
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("prediction_stale", None)
        try:
            # Reuse the existing validation, trusted model loading and class lookup.
            result = prediction.predict_one(readings, str(dashboard.artifact_path("models/failure_model.joblib")))
            st.session_state["last_prediction"] = {"readings": readings, "result": result}
        except (ValueError, OSError, TypeError) as error:
            st.error(f"Prediction could not be completed: {error}")
    if st.session_state.get("prediction_stale"):
        st.info("Readings changed. The previous prediction is hidden; click Predict Machine Failure for these inputs.")
    if "last_prediction" in st.session_state:
        last = st.session_state["last_prediction"]
        result = last["result"]
        if result["predicted_failure"] == 1:
            st.warning("Failure classified · Class 1")
        else:
            st.success("No failure classified · Class 0")
        probability = result["failure_probability"]
        st.metric("Model-estimated failure probability", "Unavailable" if probability is None else f"{probability:.2%}")
        st.caption("Probability is not a calibrated risk estimate. The result refers to the last submitted readings shown below, not a future breakdown date.")
        st.dataframe(pd.DataFrame([last["readings"]]), hide_index=True, width="stretch")
        with st.expander("Actual prediction response"):
            st.json(result)
    st.info(DISCLAIMER)
    st.subheader("Live sensor comparisons")
    st.caption("Edit a sensor and confirm the numeric entry with Enter or by leaving the field. Markers update without clicking Predict. Historical distributions always use all product types and remain fixed. Current-input markers are not model predictions or safety thresholds.")
    try:
        charts = build_live_charts(clean, readings)
        for title, chart, key in zip(
            ("Torque versus rotational speed", "Air and process temperature distributions", "Tool-wear distribution"),
            charts, ("live_torque_rpm", "live_temperatures", "live_tool_wear"),
        ):
            st.markdown(f"#### {title}")
            st.plotly_chart(chart, key=key, width="stretch", config={"displaylogo": False})
    except ValueError as error:
        st.warning(f"Live comparisons need valid sensor readings: {error}")
    with st.expander("Held-out model evaluation"):
        model_report = report("outputs/reports/model_metrics.json")
        metrics = model_report["metrics"]
        st.dataframe(pd.DataFrame([{key: metrics[key] for key in ("accuracy", "precision", "recall", "f1", "train_samples", "test_samples", "failure_support")}]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]), width="stretch")
        st.caption("Positive class = 1. The model uses an 80/20 stratified split, seed 42, balanced class weights and training-only depth selection. Accuracy alone is misleading for uncommon failures.")


def main() -> None:
    st.set_page_config(page_title="Industrial Machine Failure Analyzer", page_icon="⚙️", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 1400px; padding-top: 2rem;}
        [data-testid="stMetric"] {background: #edf4fa; border: 1px solid #d9e6f0; border-radius: 12px; padding: 16px;}
        [data-testid="stMetricLabel"] {color: #314b63;}
        [data-testid="stMetricValue"] {font-size: 1.8rem;}
        h1,h2,h3 {letter-spacing: -.025em;}
        </style>""", unsafe_allow_html=True)
    st.sidebar.markdown("## ⚙ Industrial Analyzer")
    st.sidebar.caption("DAV PROJECT · YAHYA & AMIRUL")
    page = st.sidebar.radio("Explore", PAGES, key="navigation")
    st.sidebar.divider()
    st.sidebar.caption("AI4I 2020 · Synthetic snapshot data\n\nDecision Tree classification, not chronological forecasting.")
    st.title("Industrial Machine Failure Analyzer")
    st.caption("DATA ANALYTICS & VISUALIZATION  /  MACHINE FAILURE INSIGHTS")
    missing = dashboard.missing_artifacts()
    if missing:
        st.warning(f"Setup incomplete: {len(missing)} required artifact(s) missing.")
        with st.expander("Missing files and setup instructions", expanded=True):
            st.code("\n".join(missing), language=None)
            setup_help()
    try:
        if page == PAGES[3]:
            visualizations()
        elif page == PAGES[2]:
            statistics()
        else:
            clean = csv("data/processed/ai4i_cleaned.csv")
            if page == PAGES[0]:
                overview(clean)
            elif page == PAGES[1]:
                cleaning(clean)
            else:
                prediction_page(clean)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        st.error(f"Cannot load the requested analysis: {error}")
        setup_help()


if __name__ == "__main__":
    main()
