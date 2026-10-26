"""Interactive comparisons with fixed, canonical historical observations."""

import pandas as pd
import plotly.graph_objects as go

from industrial_analyzer.features import _validate_canonical
from industrial_analyzer.prediction import _validate_readings


def _layout(figure: go.Figure, x_title: str, y_title: str) -> None:
    figure.update_layout(
        template="plotly_white", height=420,
        xaxis_title=x_title, yaxis_title=y_title,
        margin=dict(l=25, r=20, t=20, b=100),
        legend=dict(orientation="h", y=-0.25, x=0),
        uirevision="fixed-historical-reference",
    )


def build_live_charts(clean: pd.DataFrame, readings: dict) -> tuple:
    """Return three Plotly charts without loading, training or predicting."""
    _validate_canonical(clean)
    _validate_readings(readings)

    scatter = go.Figure()
    # Keep every historical record and draw the current input last.
    for label, name, color in ((0, "Recorded no failure", "#3477A5"), (1, "Recorded failure", "#B44855")):
        group = clean.loc[clean["machine_failure"] == label]
        scatter.add_trace(go.Scattergl(
            x=group["rotational_speed_rpm"].tolist(), y=group["torque_nm"].tolist(),
            mode="markers", name=name,
            marker=dict(color=color, size=5 if label == 0 else 7, opacity=0.3 if label == 0 else 0.8),
            hovertemplate=f"{name}<br>Speed: %{{x}} RPM<br>Torque: %{{y}} Nm<extra></extra>",
        ))
    scatter.add_trace(go.Scatter(
        x=[readings["rotational_speed_rpm"]], y=[readings["torque_nm"]],
        mode="markers", name="Current input (unclassified)",
        marker=dict(color="#EF9B20", size=18, symbol="diamond", line=dict(color="#17324D", width=2)),
        hovertemplate="Current input<br>Speed: %{x} RPM<br>Torque: %{y} Nm<extra></extra>",
    ))
    _layout(scatter, "Rotational speed (RPM)", "Torque (Nm)")

    temperatures = go.Figure()
    for field, title, color in (
        ("air_temperature_k", "Historical air temperature", "#3477A5"),
        ("process_temperature_k", "Historical process temperature", "#6D5A96"),
    ):
        temperatures.add_trace(go.Histogram(
            x=clean[field].tolist(), name=title, opacity=0.55,
            xbins=dict(size=0.5), marker_color=color,
            hovertemplate=f"{title}<br>Temperature: %{{x}} K<br>Records: %{{y}}<extra></extra>",
        ))
    for field, title, color in (
        ("air_temperature_k", "Air input", "#EF9B20"),
        ("process_temperature_k", "Process input", "#BD3370"),
    ):
        temperatures.add_vline(
            x=readings[field], line_color=color, line_width=3, line_dash="dash",
            annotation_text=f"{title}: {readings[field]:g} K",
            annotation_position="top left" if field == "air_temperature_k" else "top right",
        )
    temperatures.update_layout(barmode="overlay")
    _layout(temperatures, "Temperature (K)", "Historical observations per bin")
    temperature_low = min(clean["air_temperature_k"].min(), clean["process_temperature_k"].min(), readings["air_temperature_k"], readings["process_temperature_k"])
    temperature_high = max(clean["air_temperature_k"].max(), clean["process_temperature_k"].max(), readings["air_temperature_k"], readings["process_temperature_k"])
    padding = max(0.5, (temperature_high - temperature_low) * 0.05)
    temperatures.update_xaxes(range=[temperature_low - padding, temperature_high + padding])

    wear = go.Figure(go.Histogram(
        x=clean["tool_wear_min"].tolist(), name="Historical tool wear", marker_color="#3477A5",
        xbins=dict(size=10),
        hovertemplate="Tool wear: %{x} min<br>Records: %{y}<extra></extra>",
    ))
    wear.add_vline(
        x=readings["tool_wear_min"], line_color="#EF9B20", line_width=3, line_dash="dash",
        annotation_text=f"Current input: {readings['tool_wear_min']:g} min", annotation_position="top right",
    )
    _layout(wear, "Tool wear (minutes)", "Historical observations per bin")
    wear_low = min(float(clean["tool_wear_min"].min()), readings["tool_wear_min"])
    wear_high = max(float(clean["tool_wear_min"].max()), readings["tool_wear_min"])
    wear.update_xaxes(range=[wear_low - 10, wear_high + 10])
    return scatter, temperatures, wear
