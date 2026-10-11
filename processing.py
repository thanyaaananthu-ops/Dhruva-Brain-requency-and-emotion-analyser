import pandas as pd


def load_data(file_path):
    """Load the EEG dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load CSV data from {file_path}: {exc}"
        ) from exc

    if df.shape[1] < 4:
        raise ValueError(
            "The dataset does not contain enough columns."
        )

    if len(df) < 2:
        raise ValueError(
            "The dataset must contain at least two rows."
        )

    return df


def get_sensor_values(row):
    """Get sensor values and the recorded category."""

    left_mean = float(row.iloc[0])
    right_mean = float(row.iloc[1])
    variance = float(row.iloc[2])
    file_tag = row.iloc[-1]

    return left_mean, right_mean, variance, file_tag