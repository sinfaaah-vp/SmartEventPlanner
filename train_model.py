
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_PATH = "data/events.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "data/events.csv not found. Run python generate_data.py first."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. VALIDATE AND CLEAN DATA
# --------------------------------------------------

required_columns = [
    "EventType",
    "Location",
    "GuestCount",
    "Budget",
    "Style",
    "VenueType",
    "Catering",
    "ActualCost"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.dropna(subset=required_columns).copy()

df = df[
    (df["GuestCount"] > 0)
    & (df["Budget"] > 0)
    & (df["ActualCost"] > 0)
]

if len(df) < 20:
    raise ValueError("Not enough valid event records to train.")

# --------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# --------------------------------------------------

feature_columns = [
    "EventType",
    "Location",
    "GuestCount",
    "Budget",
    "Style",
    "VenueType",
    "Catering"
]

X = df[feature_columns]
y = df["ActualCost"]

categorical_features = [
    "EventType",
    "Location",
    "Style",
    "VenueType",
    "Catering"
]

numeric_features = [
    "GuestCount",
    "Budget"
]

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# --------------------------------------------------
# 5. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 6. TRAIN RANDOM FOREST
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

# --------------------------------------------------
# 7. EVALUATE ON UNSEEN TEST DATA
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Avoid division by zero when computing MAPE.
nonzero_mask = y_test.to_numpy() != 0

mape = (
    np.mean(
        np.abs(
            (
                y_test.to_numpy()[nonzero_mask]
                - predictions[nonzero_mask]
            )
            / y_test.to_numpy()[nonzero_mask]
        )
    ) * 100
)

print("\nMODEL EVALUATION")
print("------------------------------")
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")
print(f"MAE:              ₹{mae:,.2f}")
print(f"RMSE:             ₹{rmse:,.2f}")
print(f"R² score:         {r2:.4f}")
print(f"MAPE:             {mape:.2f}%")

# --------------------------------------------------
# 8. SAVE MODEL AND METRICS
# --------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "event_cost_model.pkl"
)

metrics_path = os.path.join(
    MODEL_DIR,
    "evaluation_metrics.json"
)

joblib.dump(model, model_path)

metrics = {
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2_score": float(r2),
    "mape_percent": float(mape)
}

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel training completed!")
print("Model saved to:", model_path)
print("Metrics saved to:", metrics_path)

print(
    "\nNote: The data is synthetic. These evaluation "
    "metrics do not establish real-world accuracy."
)

