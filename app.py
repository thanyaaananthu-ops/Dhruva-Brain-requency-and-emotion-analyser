from processing import load_data, get_sensor_values
from model import classify_emotion


def main():

    # EEG dataset in the same project folder
    file_path = "emotions.csv"

    try:

        df = load_data(file_path)

        print("EEG DATA LOADED SUCCESSFULLY")
        print("-" * 64)
        print(f"Number of rows:    {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("-" * 64)

        # Check one recorded EEG sample
        left, right, variance, ground_truth = get_sensor_values(
            df.iloc[1]
        )

        print("SENSOR DATA VERIFICATION")
        print("-" * 64)
        print(f"Sensor value 1: {left:.4f}")
        print(f"Sensor value 2: {right:.4f}")
        print(f"Variance:       {variance:.4f}")
        print(f"Recorded label: {ground_truth}")
        print("-" * 64)

        # Analyze first 26 recorded samples
        for index, (_, row) in enumerate(
            df.head(26).iterrows()
        ):

            left_mean, right_mean, variance, file_tag = (
                get_sensor_values(row)
            )

            valence_score, mapped_emotion = classify_emotion(
                left_mean,
                right_mean,
                variance,
                file_tag
            )

            print(f"Sample: {index:03d}")
            print(
                f"Balance score: {valence_score:.2f}"
            )
            print(
                f"Variance:      {variance:.2f}"
            )
            print(
                f"Recorded category: {file_tag}"
            )
            print(
                f"Mapped emotion:    {mapped_emotion}"
            )
            print("-" * 64)

    except (RuntimeError, ValueError, IndexError) as exc:

        print(f"Error: {exc}")


if __name__ == "__main__":
    main()