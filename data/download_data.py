import os
import urllib.request
import pandas as pd
import numpy as np

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, "heart_disease.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# Download dataset
urllib.request.urlretrieve(DATA_URL, FILE_PATH)

# Column names as per UCI documentation
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

# Load and preprocess
df = pd.read_csv(FILE_PATH, names=columns)
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.fillna(df.median(), inplace=True)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Save cleaned dataset
df.to_csv(FILE_PATH, index=False)

print("Dataset downloaded and prepared at:", FILE_PATH)
