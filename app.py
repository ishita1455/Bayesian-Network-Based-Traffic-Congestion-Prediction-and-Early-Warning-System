
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pgmpy.inference import VariableElimination


st.set_page_config(
    page_title="TrafficWise | Early Warning",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "traffic_bn_models.joblib"
METR_PATH = BASE_DIR / "METR-LA.csv"
PEMS_PATH = BASE_DIR / "PEMS-BAY.csv"

HORIZONS = {
    "15 minutes": ("congestion_15m", "type_15m"),
    "30 minutes": ("congestion_30m", "type_30m"),
    "60 minutes": ("congestion_60m", "type_60m"),
}

SEVERITY_TARGETS = [
    "congestion_15m", "congestion_30m", "congestion_60m"
]
TYPE_TARGETS = ["type_15m", "type_30m", "type_60m"]


def time_period(hour):
    if 6 <= hour < 10:
        return "Morning"
    if 10 <= hour < 16:
        return "Midday"
    if 16 <= hour < 20:
        return "Evening"
    return "Night"


def severity_from_ratio(ratio):
    if ratio >= 0.70:
        return "Low"
    if ratio >= 0.40:
        return "Medium"
    return "High"


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    required = {"models", "features", "targets"}
    if not required.issubset(artifact):
        raise ValueError(
            "The model artifact must contain models, features and targets."
        )

    return artifact


@st.cache_data(show_spinner="Preparing historical traffic data...")
def load_and_prepare_data():
    frames = []

    for dataset_name, path in [
        ("METR-LA", METR_PATH),
        ("PEMS-BAY", PEMS_PATH),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path.name}")

        raw = pd.read_csv(path, index_col=0)
        raw.index = pd.to_datetime(raw.index, errors="coerce")
        raw = raw.loc[raw.index.notna()].sort_index()

        if raw.empty:
            continue

        sensor_columns = list(raw.columns[:20])
        speeds = raw[sensor_columns].apply(pd.to_numeric, errors="coerce")
        speeds = speeds.replace(0, np.nan)
        speeds = speeds.interpolate(
            method="linear", limit_direction="both"
        ).ffill().bfill()

        for sensor in sensor_columns:
            speed = speeds[sensor].astype(float)
            free_flow = speed.quantile(0.85)

            if not np.isfinite(free_flow) or free_flow <= 0:
                free_flow = speed[speed > 0].median()

            if not np.isfinite(free_flow) or free_flow <= 0:
                continue

            part = pd.DataFrame({"timestamp": speeds.index})
            part["dataset"] = dataset_name
            part["sensor"] = f"{dataset_name}_{sensor}"
            part["speed"] = speed.to_numpy()
            part["speed_ratio"] = (speed / free_flow).clip(0, 1.5).to_numpy()

            part["day_of_week"] = part["timestamp"].dt.dayofweek
            part["slot"] = (
                part["timestamp"].dt.hour * 12
                + part["timestamp"].dt.minute // 5
            )

            part["expected_ratio"] = part.groupby(
                ["day_of_week", "slot"]
            )["speed_ratio"].transform("median")

            part["expected_ratio"] = part["expected_ratio"].fillna(
                part["speed_ratio"].median()
            )

            part["speed_change_15m"] = part["speed_ratio"].diff(3).fillna(0)
            part["time_period"] = part["timestamp"].dt.hour.map(time_period)
            part["day_type"] = np.where(
                part["day_of_week"] >= 5, "Weekend", "Weekday"
            )

            part["peak_hour"] = np.where(
                part["timestamp"].dt.hour.between(7, 9)
                | part["timestamp"].dt.hour.between(16, 18),
                "Yes", "No"
            )

            part["speed_condition"] = part["speed_ratio"].map(
                lambda x: "Normal" if x >= 0.70
                else ("Reduced" if x >= 0.40 else "Severe")
            )

            part["traffic_trend"] = part["speed_change_15m"].map(
                lambda x: "Worsening" if x <= -0.05
                else ("Improving" if x >= 0.05 else "Stable")
            )

            deviation = (
                (part["speed_ratio"] - part["expected_ratio"])
                / part["expected_ratio"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

            part["deviation_category"] = deviation.map(
                lambda x: "High_Deviation" if x <= -20
                else ("Moderate_Deviation" if x <= -10
                      else "Normal_Deviation")
            )

            frames.append(part)

    if not frames:
        raise ValueError("No usable traffic sensor data was found.")

    result = pd.concat(frames, ignore_index=True)
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "timestamp", "speed", "speed_ratio", "expected_ratio",
            "time_period", "day_type", "peak_hour", "speed_condition",
            "traffic_trend", "deviation_category"
        ]
    )


def predict_one(model, evidence):
    target_candidates = [
        node for node in model.nodes() if node not in evidence
    ]

    if len(target_candidates) != 1:
        raise ValueError(
            f"Expected one target node, found: {target_candidates}"
        )

    target = target_candidates[0]
    inference = VariableElimination(model)
    result = inference.query(
        variables=[target],
        evidence=evidence,
        show_progress=False,
    )

    states = list(result.state_names[target])
    probabilities = np.asarray(result.values, dtype=float).reshape(-1)
    distribution = {
        str(states[i]): float(probabilities[i])
        for i in range(len(states))
    }

    prediction = max(distribution, key=distribution.get)
    return prediction, distribution


def averaged_prediction(model, rows, features):
    distributions = []

    for _, row in rows.iterrows():
        evidence = {feature: str(row[feature]) for feature in features}
        _, distribution = predict_one(model, evidence)
        distributions.append(distribution)

    probabilities = pd.DataFrame(distributions).fillna(0).mean()
    probabilities = probabilities / probabilities.sum()

    return probabilities.sort_values(ascending=False)


try:
    artifact = load_artifact()
    models = artifact["models"]
    FEATURES = artifact["features"]
    TARGETS = artifact["targets"]
    data = load_and_prepare_data()
except Exception as exc:
    st.error(f"Could not initialize TrafficWise: {exc}")
    st.info(
        "Check that app.py, traffic_bn_models.joblib, METR-LA.csv "
        "and PEMS-BAY.csv are in the same folder."
    )
    st.stop()


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚦 TrafficWise")
st.subheader("Bayesian Network Traffic Early-Warning System")
st.caption(
    "Explore historical traffic patterns and estimate congestion "
    "severity for the next 15, 30 or 60 minutes."
)

with st.sidebar:
    st.header("Forecast settings")

    dataset_choice = st.selectbox(
        "Traffic dataset",
        sorted(data["dataset"].unique()),
    )

    available_sensors = sorted(
        data.loc[data["dataset"] == dataset_choice, "sensor"].unique()
    )

    sensor_choice = st.selectbox("Road sensor", available_sensors)

    day_choice = st.radio(
        "Day type",
        ["Weekday", "Weekend"],
        horizontal=True,
    )

    time_values = sorted(
        data.loc[
            (data["dataset"] == dataset_choice)
            & (data["sensor"] == sensor_choice)
            & (data["day_type"] == day_choice),
            "timestamp",
        ].dt.strftime("%H:%M").unique()
    )

    if not time_values:
        st.warning("No historical observations match these selections.")
        st.stop()

    time_choice = st.selectbox(
        "Time of day",
        time_values,
        index=min(
            len(time_values) - 1,
            next(
                (i for i, value in enumerate(time_values)
                 if value in ["08:00", "08:05", "08:10"]),
                0,
            ),
        ),
    )

    horizon_choice = st.selectbox(
        "Forecast horizon",
        list(HORIZONS.keys()),
    )

    st.divider()
    st.caption("Data sources: METR-LA and PEMS-BAY")
    st.caption("Inference uses your saved Bayesian Network models.")


selected = data[
    (data["dataset"] == dataset_choice)
    & (data["sensor"] == sensor_choice)
    & (data["day_type"] == day_choice)
    & (data["timestamp"].dt.strftime("%H:%M") == time_choice)
].copy()

if selected.empty:
    st.warning("No matching historical observations were found.")
    st.stop()

st.markdown("### Selected traffic context")

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Historical matches", f"{len(selected):,}")
metric2.metric("Average speed", f"{selected['speed'].mean():.2f}")
metric3.metric("Forecast horizon", horizon_choice)

st.caption(
    "Average speed uses the units supplied by the source dataset. "
    "These observations are historical, not live measurements."
)

severity_target, type_target = HORIZONS[horizon_choice]

missing_targets = [
    target for target in [severity_target, type_target]
    if target not in models or target not in TARGETS
]

if missing_targets:
    st.error(f"Missing targets in saved artifact: {missing_targets}")
    st.stop()

severity_probs = averaged_prediction(
    models[severity_target], selected, FEATURES
)
type_probs = averaged_prediction(
    models[type_target], selected, FEATURES
)

severity_prediction = severity_probs.index[0]
type_prediction = type_probs.index[0]

st.markdown("### Forecast results")

left, right = st.columns(2)

with left:
    st.markdown("#### Congestion severity")
    severity_colors = {
        "Low": "🟢",
        "Medium": "🟠",
        "High": "🔴",
    }
    st.markdown(
        f"## {severity_colors.get(severity_prediction, '🚦')} "
        f"{severity_prediction}"
    )
    st.caption(
        f"Highest model probability: "
        f"{severity_probs.iloc[0] * 100:.2f}%"
    )

    severity_chart = pd.DataFrame({
        "Class": severity_probs.index,
        "Probability (%)": severity_probs.values * 100,
    })
    fig1 = px.bar(
        severity_chart,
        x="Class",
        y="Probability (%)",
        text_auto=".1f",
        title="Congestion probability distribution",
        category_orders={"Class": ["Low", "Medium", "High"]},
    )
    fig1.update_layout(
        yaxis_range=[0, 100],
        showlegend=False,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.markdown("#### Traffic pattern")
    type_colors = {
        "Normal": "🟢",
        "Recurring": "🟠",
        "Non-Recurring": "🔴",
    }
    st.markdown(
        f"## {type_colors.get(type_prediction, '🚦')} "
        f"{type_prediction}"
    )
    st.caption(
        f"Highest model probability: "
        f"{type_probs.iloc[0] * 100:.2f}%"
    )

    type_chart = pd.DataFrame({
        "Class": type_probs.index,
        "Probability (%)": type_probs.values * 100,
    })
    fig2 = px.bar(
        type_chart,
        x="Class",
        y="Probability (%)",
        text_auto=".1f",
        title="Traffic-pattern probability distribution",
        category_orders={
            "Class": ["Normal", "Recurring", "Non-Recurring"]
        },
    )
    fig2.update_layout(
        yaxis_range=[0, 100],
        showlegend=False,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.markdown("### Automatically derived traffic conditions")

feature_columns = [
    "time_period",
    "day_type",
    "peak_hour",
    "speed_condition",
    "traffic_trend",
    "deviation_category",
]

feature_labels = {
    "time_period": "Time period",
    "day_type": "Day type",
    "peak_hour": "Peak hour",
    "speed_condition": "Speed condition",
    "traffic_trend": "Traffic trend",
    "deviation_category": "Deviation category",
}

feature_rows = []

for feature in feature_columns:
    counts = selected[feature].value_counts(normalize=True)
    if counts.empty:
        continue

    value = counts.index[0]
    feature_rows.append({
        "Condition": feature_labels[feature],
        "Typical value": value,
        "Historical match (%)": round(counts.iloc[0] * 100, 1),
    })

st.dataframe(
    pd.DataFrame(feature_rows),
    hide_index=True,
    use_container_width=True,
)

with st.expander("View historical observations"):
    history = selected[
        ["timestamp", "speed"] + feature_columns
    ].sort_values("timestamp", ascending=False)

    st.dataframe(
        history.head(100),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("About this prediction"):
    st.write(
        "TrafficWise applies your saved Bayesian Network models to "
        "historical observations matching the selected dataset, sensor, "
        "day type and time. It averages class probabilities across these "
        "matching observations to estimate a typical historical scenario."
    )
    st.write(
        "Recurring and Non-Recurring labels are derived from speed and "
        "deviation rules. They do not confirm an actual accident or incident."
    )
    st.warning(
        "This prototype does not access live traffic feeds. Its predictions "
        "are not a real-time road forecast, and high confidence does not "
        "guarantee that the predicted outcome will occur."
    )

st.caption(
    "TrafficWise | Bayesian Networks • Historical Data • Early Warning"
)
