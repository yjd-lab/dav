"""Read-only access to the dashboard's existing project artifacts."""

import json
from pathlib import Path

import pandas as pd

from industrial_analyzer.data_loader import load_ai4i_csv


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SENSORS = (
    "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min",
)
FIGURES = (
    ("fig_01_failure_rate_by_type.png", "Failure rate by product type", "Recorded failures divided by all observations within each type."),
    ("fig_02_sensor_distributions_by_failure.png", "Sensor distributions", "Physical sensor distributions for recorded failure and non-failure groups."),
    ("fig_03_torque_vs_rotational_speed.png", "Torque and rotational speed", "Torque (Nm) versus RPM, colored by recorded failure status."),
    ("fig_04_air_vs_process_temperature.png", "Air and process temperature", "Temperature associations in Kelvin; associations do not establish causes."),
    ("fig_05_tool_wear_by_failure.png", "Tool wear by failure status", "Recorded tool-wear distributions in minutes."),
    ("fig_06_sensor_correlation.png", "Sensor correlations", "Pearson correlations describe associations, not causal relationships."),
    ("fig_y_01_torque_by_failure.png", "Torque by failure status", "Torque in Nm, with original group sizes and sensor extremes preserved."),
    ("fig_y_02_torque_vs_rpm.png", "Failure-focused torque and speed", "Failure points are drawn last so uncommon failures remain visible."),
    ("fig_y_03_failure_rate_by_type.png", "Within-type failure percentages", "Each type uses its own observation count as the denominator."),
)
TABLE_COLUMNS = {
    "summary_statistics": ["feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    "failure_by_type": ["type", "total_records", "failure_count", "nonfailure_count", "failure_rate_percent"],
    "failure_modes": ["failure_mode", "flagged_records", "percent_of_records", "dataset_records"],
}
REQUIRED_ARTIFACTS = (
    "data/processed/ai4i_cleaned.csv", "data/processed/ai4i_transformed.csv",
    "outputs/reports/data_quality.json", "outputs/reports/model_metrics.json",
    "outputs/reports/run_manifest.json", "models/failure_model.joblib",
    *(f"outputs/tables/{name}.csv" for name in TABLE_COLUMNS),
    *(f"outputs/figures/{name}" for name, _, _ in FIGURES),
)


def artifact_path(relative_path: str) -> Path:
    """Resolve a contained artifact independently of the terminal directory."""
    root = REPOSITORY_ROOT.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dashboard artifacts must stay inside the repository")
    return path


def missing_artifacts() -> list[str]:
    return [name for name in REQUIRED_ARTIFACTS if not artifact_path(name).is_file()]


def file_stamp(relative_path: str) -> tuple[str, int, int]:
    """Include file changes in Streamlit's data-cache key."""
    path = artifact_path(relative_path)
    details = path.stat()
    return str(path), details.st_mtime_ns, details.st_size


def read_csv(path: str) -> pd.DataFrame:
    if Path(path).name == "ai4i_cleaned.csv":
        return load_ai4i_csv(path)
    frame = pd.read_csv(path, encoding="utf-8")
    if frame.empty:
        raise ValueError(f"{Path(path).name} has no records")
    return frame


def read_report(path: str) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    return report


def overview_metrics(clean: pd.DataFrame) -> dict:
    failures = int(clean["machine_failure"].sum())
    return {
        "Observations": len(clean), "Recorded failures": failures,
        "Non-failures": len(clean) - failures,
        "Recorded failure rate": failures / len(clean) * 100,
        "Product categories": int(clean["type"].nunique()),
    }
