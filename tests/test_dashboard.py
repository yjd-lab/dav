"""Connections between Streamlit, existing artifacts and the real classifier."""

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from industrial_analyzer import dashboard, prediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "app.py"
PAGES = ["Dashboard / Overview", "Data Cleaning & Transformation", "Statistical Analysis", "Data Visualization", "Machine Failure Prediction"]
FIGURE_NAMES = [
    "fig_01_failure_rate_by_type.png", "fig_02_sensor_distributions_by_failure.png",
    "fig_03_torque_vs_rotational_speed.png", "fig_04_air_vs_process_temperature.png",
    "fig_05_tool_wear_by_failure.png", "fig_06_sensor_correlation.png",
    "fig_y_01_torque_by_failure.png", "fig_y_02_torque_vs_rpm.png",
    "fig_y_03_failure_rate_by_type.png",
]


def hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    # Copy artifacts only; UI tests never regenerate or train a model.
    for relative_path in dashboard.REQUIRED_ARTIFACTS:
        source = PROJECT_ROOT / relative_path
        if not source.is_file():
            pytest.skip(f"Generate project artifacts before UI integration tests: {relative_path}")
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    protected = [*PROJECT_ROOT.glob("src/industrial_analyzer/*.py"),
                 PROJECT_ROOT / "notebooks/04_final_demo.ipynb",
                 PROJECT_ROOT / "data/raw/ai4i2020.csv"]
    before = hashes(protected)
    monkeypatch.setattr(dashboard, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(prediction, "REPOSITORY_ROOT", tmp_path)

    def forbidden_training(*args, **kwargs):
        pytest.fail("Frontend must not retrain the classifier")

    monkeypatch.setattr(prediction, "train_failure_model", forbidden_training)
    yield tmp_path
    assert hashes(protected) == before


def start(page="Dashboard / Overview"):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    if page != PAGES[0]:
        app.radio(key="navigation").set_value(page).run()
    assert not app.exception
    return app


def submit(app):
    button = next(button for button in app.button if button.label == "Predict Machine Failure")
    return button.click().run()


def test_real_artifact_startup_and_calculated_metrics(isolated_artifacts):
    app = start()
    clean = pd.read_csv(isolated_artifacts / "data/processed/ai4i_cleaned.csv")
    metrics = {metric.label: metric.value for metric in app.metric}
    failure_count = int(clean.machine_failure.sum())
    assert metrics == {
        "Observations": f"{len(clean):,}", "Recorded failures": f"{failure_count:,}",
        "Non-failures": f"{len(clean) - failure_count:,}",
        "Recorded failure rate": f"{failure_count / len(clean) * 100:.2f}%",
        "Product categories": str(clean.type.nunique()),
    }
    assert_frame_equal(app.dataframe[0].value, clean.head())


@pytest.mark.parametrize("page", PAGES)
def test_navigation_does_not_write_or_retrain(isolated_artifacts, page):
    before = hashes(path for path in isolated_artifacts.rglob("*") if path.is_file())
    app = start(page)
    assert not app.error
    assert app.radio(key="navigation").value == page
    assert hashes(Path(path) for path in before) == before


def test_cleaning_and_sensor_comparison(isolated_artifacts):
    app = start(PAGES[1])
    app.selectbox[0].set_value("torque_nm").run()
    transformed = pd.read_csv(isolated_artifacts / "data/processed/ai4i_transformed.csv")
    comparison = next(frame.value for frame in app.dataframe if "torque_nm_zscore" in frame.value.columns and len(frame.value.columns) == 4)
    assert_frame_equal(comparison, transformed[["uid", "torque_nm", "torque_nm_minmax", "torque_nm_zscore"]].head(20))
    assert any("source audit" in caption.value for caption in app.caption)


def test_statistical_tables_and_selection(isolated_artifacts):
    app = start(PAGES[2])
    for rendered, name in zip(app.dataframe, dashboard.TABLE_COLUMNS):
        assert_frame_equal(rendered.value, pd.read_csv(isolated_artifacts / f"outputs/tables/{name}.csv"))
    app.multiselect[0].set_value(["torque_nm"]).run()
    assert app.dataframe[0].value.feature.tolist() == ["torque_nm"]


def test_nine_images_and_large_view(isolated_artifacts):
    app = start(PAGES[3])
    assert [name for name, _, _ in dashboard.FIGURES] == FIGURE_NAMES
    images = app.get("image")
    assert len(images) == 9
    app.selectbox[0].set_value("Within-type failure percentages").run()
    assert len(app.get("image")) == 1
    assert not app.exception


@pytest.mark.parametrize("relative_path", [
    "data/processed/ai4i_cleaned.csv", "data/processed/ai4i_transformed.csv",
    "outputs/reports/data_quality.json", "outputs/tables/summary_statistics.csv",
    "outputs/figures/fig_y_01_torque_by_failure.png", "models/failure_model.joblib",
])
def test_missing_artifacts_are_identified(isolated_artifacts, relative_path):
    (isolated_artifacts / relative_path).unlink()
    app = start()
    assert app.warning
    assert any(relative_path in block.value for block in app.code)
    assert any("run_pipeline.py" in block.value for block in app.code)


def test_prediction_calls_backend_and_matches_direct_result(isolated_artifacts, monkeypatch):
    app = start(PAGES[4])
    input_names = {widget.key for widget in app.number_input}
    assert input_names | {app.selectbox(key="type").key} == set(prediction.PREDICTION_FEATURES)
    readings = {widget.key: widget.value for widget in app.number_input}
    readings["type"] = app.selectbox(key="type").value
    expected = prediction.predict_one(readings, "models/failure_model.joblib")
    original = prediction.predict_one
    calls = []

    def tracked_prediction(inputs, path):
        calls.append((inputs, path))
        return original(inputs, path)

    monkeypatch.setattr(prediction, "predict_one", tracked_prediction)
    submit(app)
    assert not app.exception and not app.error
    assert len(calls) == 1 and calls[0][0] == readings
    assert app.session_state["last_prediction"]["result"] == expected
    assert app.metric[0].value == f"{expected['failure_probability']:.2%}"
    label = "Failure classified" if expected["predicted_failure"] else "No failure classified"
    assert any(label in block.value for block in [*app.success, *app.warning])


@pytest.mark.parametrize("result_label, probability", [(0, 0.18), (1, 0.83), (1, None)])
def test_prediction_displays_exact_backend_response(isolated_artifacts, monkeypatch, result_label, probability):
    def supplied_response(readings, path):
        return {"schema_version": "1.0", "predicted_failure": result_label,
                "failure_probability": probability, "model_name": "decision_tree",
                "message": "Artificial response for UI rendering test"}

    monkeypatch.setattr(prediction, "predict_one", supplied_response)
    app = submit(start(PAGES[4]))
    assert not app.exception
    assert app.metric[0].value == ("Unavailable" if probability is None else f"{probability:.2%}")
    assert bool(app.success) == (result_label == 0)
    assert any("Not a certified industrial equipment diagnosis" in item.value for item in app.info)


def test_dataset_prefill_uses_six_actual_fields(isolated_artifacts):
    app = start(PAGES[4])
    clean = pd.read_csv(isolated_artifacts / "data/processed/ai4i_cleaned.csv")
    record = clean.iloc[10]
    app.selectbox(key="demo_uid").set_value(int(record.uid)).run()
    next(button for button in app.button if button.label == "Use selected observation").click().run()
    for widget in app.number_input:
        assert widget.value == float(record[widget.key])
    assert app.selectbox(key="type").value == record["type"]


@pytest.mark.parametrize("name,value", [("air_temperature_k", 0.0), ("torque_nm", -1.0), ("rotational_speed_rpm", -10.0)])
def test_invalid_readings_clear_success(isolated_artifacts, name, value):
    app = submit(start(PAGES[4]))
    assert "last_prediction" in app.session_state
    app.number_input(key=name).set_value(value)
    submit(app)
    assert app.error and "Prediction could not be completed" in app.error[0].value
    assert not app.success and "last_prediction" not in app.session_state
    assert not app.exception


@pytest.mark.parametrize("broken", ["missing", "corrupt"])
def test_missing_or_corrupt_model_is_not_fake_success(isolated_artifacts, broken):
    model = isolated_artifacts / "models/failure_model.joblib"
    if broken == "missing":
        model.unlink()
    else:
        model.write_bytes(b"artificial corrupt model")
    app = submit(start(PAGES[4]))
    assert not app.exception and app.error
    assert not app.success
    assert "last_prediction" not in app.session_state


def test_nested_working_directory_and_path_containment(isolated_artifacts, monkeypatch):
    nested = isolated_artifacts / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    app = start()
    assert not app.error
    assert dashboard.artifact_path("models/failure_model.joblib") == isolated_artifacts / "models/failure_model.joblib"
    with pytest.raises(ValueError, match="inside the repository"):
        dashboard.artifact_path("../outside.csv")


def test_corrupt_quality_report_is_explained(isolated_artifacts):
    (isolated_artifacts / "outputs/reports/data_quality.json").write_text("not json")
    app = start(PAGES[1])
    assert app.error and not app.exception
    assert "Cannot load the requested analysis" in app.error[0].value


def chart_specs(app):
    return [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]


def test_live_charts_use_all_actual_records(isolated_artifacts):
    app = start(PAGES[4])
    clean = pd.read_csv(isolated_artifacts / "data/processed/ai4i_cleaned.csv")
    scatter, temperatures, wear = chart_specs(app)
    for label, trace in zip((0, 1), scatter["data"][:2]):
        group = clean.loc[clean.machine_failure == label]
        assert trace["x"] == group.rotational_speed_rpm.tolist()
        assert trace["y"] == group.torque_nm.tolist()
    assert sum(len(trace["x"]) for trace in scatter["data"][:2]) == len(clean)
    assert scatter["data"][-1]["name"] == "Current input (unclassified)"
    assert temperatures["data"][0]["x"] == clean.air_temperature_k.tolist()
    assert temperatures["data"][1]["x"] == clean.process_temperature_k.tolist()
    assert wear["data"][0]["x"] == clean.tool_wear_min.tolist()
    assert scatter["layout"]["xaxis"]["title"]["text"] == "Rotational speed (RPM)"
    assert scatter["layout"]["yaxis"]["title"]["text"] == "Torque (Nm)"
    assert temperatures["layout"]["xaxis"]["title"]["text"] == "Temperature (K)"
    assert wear["layout"]["xaxis"]["title"]["text"] == "Tool wear (minutes)"


@pytest.mark.parametrize("sensor,chart_index,position", [
    ("torque_nm", 0, "y"), ("rotational_speed_rpm", 0, "x"),
    ("air_temperature_k", 1, 0), ("process_temperature_k", 1, 1),
    ("tool_wear_min", 2, 0),
])
def test_sensor_edits_move_markers_without_prediction(
    isolated_artifacts, monkeypatch, sensor, chart_index, position,
):
    def forbidden_prediction(*args, **kwargs):
        pytest.fail("Editing a sensor must not submit a model prediction")

    monkeypatch.setattr(prediction, "predict_one", forbidden_prediction)
    app = start(PAGES[4])
    before = chart_specs(app)
    changed_value = app.number_input(key=sensor).value + 7.0
    app.number_input(key=sensor).set_value(changed_value).run()
    after = chart_specs(app)
    assert not app.exception and not app.error
    if chart_index == 0:
        assert after[0]["data"][-1][position] == [changed_value]
    else:
        line = after[chart_index]["layout"]["shapes"][position]
        assert line["x0"] == line["x1"] == changed_value
    assert after[0]["data"][:2] == before[0]["data"][:2]
    assert after[1]["data"] == before[1]["data"]
    assert after[2]["data"] == before[2]["data"]
    assert "last_prediction" not in app.session_state


@pytest.mark.parametrize("field", list(prediction.PREDICTION_FEATURES))
def test_each_input_change_hides_prediction_until_explicit_resubmission(
    isolated_artifacts, monkeypatch, field,
):
    original = prediction.predict_one
    calls = []

    def tracked_prediction(readings, path):
        calls.append(readings.copy())
        return original(readings, path)

    monkeypatch.setattr(prediction, "predict_one", tracked_prediction)
    app = submit(start(PAGES[4]))
    assert len(calls) == 1
    if field == "type":
        value = "H" if app.selectbox(key="type").value != "H" else "L"
        app.selectbox(key="type").set_value(value).run()
    else:
        app.number_input(key=field).set_value(app.number_input(key=field).value + 1.0).run()
    assert len(calls) == 1
    assert "last_prediction" not in app.session_state
    assert not app.success
    assert not any("classified" in warning.value for warning in app.warning)
    assert any("previous prediction is hidden" in message.value for message in app.info)
    submit(app)
    assert len(calls) == 2 and not app.exception
    assert app.session_state["last_prediction"]["result"] == original(calls[-1], "models/failure_model.joblib")


def test_live_invalid_sensor_message_does_not_claim_prediction(isolated_artifacts):
    app = start(PAGES[4])
    app.number_input(key="tool_wear_min").set_value(-1.0).run()
    assert not app.exception and not app.success
    assert any("Live comparisons need valid sensor readings" in warning.value for warning in app.warning)
    assert not chart_specs(app)


@pytest.mark.parametrize("problem", ["missing", "invalid_category", "nonfinite_sensor"])
def test_bad_reference_data_is_explained_before_plotting(isolated_artifacts, problem):
    path = isolated_artifacts / "data/processed/ai4i_cleaned.csv"
    if problem == "missing":
        path.unlink()
    else:
        clean = pd.read_csv(path)
        if problem == "invalid_category":
            clean.loc[0, "type"] = "INVALID"
        else:
            clean.loc[0, "torque_nm"] = float("inf")
        clean.to_csv(path, index=False)
    app = start(PAGES[4])
    assert app.error and not app.exception
    assert not chart_specs(app)


def test_repeated_sensor_edits_use_cached_canonical_data(isolated_artifacts, monkeypatch):
    original = dashboard.read_csv
    calls = []

    def tracked_csv(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(dashboard, "read_csv", tracked_csv)
    app = start(PAGES[4])
    app.number_input(key="torque_nm").set_value(44.0).run()
    app.number_input(key="torque_nm").set_value(45.0).run()
    clean_calls = [path for path in calls if path.endswith("ai4i_cleaned.csv")]
    assert len(clean_calls) == 1
