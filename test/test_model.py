import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "artifacts/heart_model.pkl"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]


def test_model_artifact_exists():
    """Check that trained model artifact exists"""
    assert os.path.exists(MODEL_PATH), "Model artifact not found"


def test_model_loads_successfully():
    """Check that model can be loaded"""
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_prediction_shape():
    """Check that model returns valid prediction output"""
    model = joblib.load(MODEL_PATH)

    sample_input = pd.DataFrame(
        [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(sample_input)
    probability = model.predict_proba(sample_input)

    assert prediction.shape == (1,)
    assert probability.shape == (1, 2)


def test_model_prediction_values():
    """Check prediction values are valid classes"""
    model = joblib.load(MODEL_PATH)

    sample_input = pd.DataFrame(
        [[58, 0, 2, 130, 197, 0, 1, 131, 0, 0.6, 1, 0, 2]],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(sample_input)[0]

    assert prediction in [0, 1]
