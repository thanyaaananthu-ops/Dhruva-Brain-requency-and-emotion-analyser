import streamlit as st

st.set_page_config(
    page_title="Emotion Detection Dashboard",
    layout="wide"
)

st.title("Emotion Detection Dashboard")
st.subheader("Sensor Monitoring System")

st.write("Welcome to your emotion-classification dashboard!")

st.success("Dashboard is running successfully.")
st.header("Upload Sensor Dataset")

uploaded_file = st.file_uploader(
    "Choose your ZIP dataset",
    type=["zip"]
)

if uploaded_file is not None:
    import pandas as pd

    df = pd.read_csv(uploaded_file, compression="zip")

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("Total rows:", df.shape[0])
    st.write("Total columns:", df.shape[1])
 # Emotion Classification
    st.header("Emotion Classification")

    results = []

    for index, row in df.head(26).iterrows():
        left_mean = row.iloc[0]
        right_mean = row.iloc[1]
        variance = row.iloc[2]
        file_tag = str(row.iloc[-1]).strip().upper()

        valence_score = left_mean - right_mean

        if file_tag == "NEUTRAL":
            if variance < 15:
                mapped_emotion = "DROWSY"
            else:
                mapped_emotion = "RELAXED"

        elif file_tag == "POSITIVE":
            if variance > 30:
                mapped_emotion = "HAPPY"
            else:
                mapped_emotion = "CONTENT"

        elif file_tag == "NEGATIVE":
            if variance > 50:
                mapped_emotion = "ANGRY"
            elif valence_score < 0:
                mapped_emotion = "SAD"
            else:
                mapped_emotion = "DISTRESSED"

        else:
            mapped_emotion = "UNDETERMINED"

        results.append({
            "Time Frame": index,
            "Left Sensor": left_mean,
            "Right Sensor": right_mean,
            "Variance": variance,
            "Valence Score": valence_score,
            "Original Label": file_tag,
            "Predicted Emotion": mapped_emotion
        })

    results_df = pd.DataFrame(results)

    st.subheader("Classification Results")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Emotion Distribution")
    emotion_counts = results_df["Predicted Emotion"].value_counts()
    st.bar_chart(emotion_counts)

    st.subheader("Sensor Readings")
    st.line_chart(
        results_df.set_index("Time Frame")[["Left Sensor", "Right Sensor"]]
    )