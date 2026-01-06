import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Column names (UCI Heart Disease)
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

# Load raw data
df = pd.read_csv("data/heart_disease.csv", names=columns)

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Handle missing values (median imputation)
df.fillna(df.median(), inplace=True)

# Binary target conversion
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Features & label
X = df.drop("target", axis=1)
y = df["target"]

# ML pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X, y)

# Save model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(pipeline, "artifacts/heart_model.pkl")

print("✅ Model trained & saved successfully (missing values handled)")
