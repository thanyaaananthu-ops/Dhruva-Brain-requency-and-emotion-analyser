import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from processing import prepare_training_data
from model import train_model
from model import save_model


# --------------------------------
# LOAD DATASET
# --------------------------------

print("Loading EEG dataset...")

df = pd.read_csv(
    "emotions.csv"
)

print(
    "Dataset shape:",
    df.shape
)


# --------------------------------
# PREPARE DATA
# --------------------------------

X, y = prepare_training_data(
    df
)

print(
    "Number of features:",
    X.shape[1]
)

print("\nEmotion distribution:")
print(
    y.value_counts()
)


# --------------------------------
# TRAIN / TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining model...")


# --------------------------------
# TRAIN MODEL
# --------------------------------

model = train_model(
    X_train,
    y_train
)


# --------------------------------
# TEST MODEL
# --------------------------------

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    "\nModel accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------
# SAVE MODEL
# --------------------------------

save_model(
    model
)