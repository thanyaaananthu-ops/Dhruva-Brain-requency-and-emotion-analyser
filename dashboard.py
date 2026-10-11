import streamlit as st
import pandas as pd

from processing import load_data, get_sensor_values
from model import classify_emotion


st.set_page_config(
    page_title="EEG Emotion Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 EEG Emotion Analyzer")

st.write(
    "Analyze a recorded EEG sample using the sensor-based "
    "classification system."
)

st.warning(
    "Demonstration only: this system analyzes recorded EEG data. "
    "It does not measure live brain activity or provide a medical diagnosis."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    df = load_data("emotions.csv")
except Exception as exc:
    st.error(f"Could not load EEG dataset: {exc}")
    st.stop()


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Recorded Samples", len(df))

with col2:
    st.metric("Dataset Columns", len(df.columns))

with col3:
    st.metric(
        "Recorded Categories",
        df.iloc[:, -1].nunique()
    )


st.subheader("Recorded Category Distribution")

category_counts = df.iloc[:, -1].value_counts()

st.bar_chart(category_counts)


# --------------------------------------------------
# SAMPLE SELECTION
# --------------------------------------------------

st.header("🔬 Analyze a Recorded EEG Sample")

sample_number = st.number_input(
    "Select EEG Sample Number",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)


selected_row = df.iloc[sample_number]


try:
    left_mean, right_mean, variance, recorded_category = (
        get_sensor_values(selected_row)
    )

    valence_score, mapped_emotion = classify_emotion(
        left_mean,
        right_mean,
        variance,
        recorded_category
    )

except Exception as exc:
    st.error(f"Could not analyze this sample: {exc}")
    st.stop()


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

st.divider()

st.header("🧠 Analysis Result")

result1, result2, result3 = st.columns(3)

with result1:
    st.metric(
        "Recorded Category",
        str(recorded_category).upper()
    )

with result2:
    st.metric(
        "Mapped Emotion",
        mapped_emotion
    )

with result3:
    st.metric(
        "Balance Score",
        f"{valence_score:.2f}"
    )


# --------------------------------------------------
# SENSOR INFORMATION
# --------------------------------------------------

st.subheader("📡 Sensor Measurements")

sensor1, sensor2, sensor3 = st.columns(3)

with sensor1:
    st.metric(
        "Left Sensor",
        f"{left_mean:.4f}"
    )

with sensor2:
    st.metric(
        "Right Sensor",
        f"{right_mean:.4f}"
    )

with sensor3:
    st.metric(
        "Variance",
        f"{variance:.4f}"
    )


# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------

st.subheader("💡 Interpretation")

st.write(
    f"""
The selected recorded EEG sample is associated with the
category **{str(recorded_category).upper()}** in the dataset.

Using the current rule-based demonstration, the sample is
mapped to **{mapped_emotion}**.

The balance score is **{valence_score:.2f}**, calculated from
the difference between the two selected sensor values.
"""
)


# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------

st.divider()

st.subheader("⚠️ Important Limitation")

st.write(
    """
This dashboard uses previously recorded EEG data from the
provided dataset.

The mapped emotions such as HAPPY, SAD, ANGRY, RELAXED,
CONTENT, or DISTRESSED are produced by predefined heuristic
rules. They are not clinically validated emotion measurements.

The system should therefore be presented as an experimental
demonstration rather than a medical diagnostic tool.
"""
)