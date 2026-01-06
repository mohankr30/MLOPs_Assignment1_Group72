from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os
import pandas as pd
import logging
import time

# -------------------------------------------------------------------
# Logging Configuration (Task 8: Monitoring & Logging)
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML model API for heart disease risk prediction",
    version="1.0.0"
)

# -------------------------------------------------------------------
# Load Trained Model
# -------------------------------------------------------------------
MODEL_PATH = os.path.join("artifacts", "heart_model.pkl")
model = joblib.load(MODEL_PATH)
logger.info("Model loaded successfully")

# -------------------------------------------------------------------
# Request Schema (Python 3.8 compatible)
# -------------------------------------------------------------------
class PatientData(BaseModel):
    features: List[float]

# -------------------------------------------------------------------
# Middleware: Request / Response Logging
# -------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    logger.info(f"Incoming request: {request.method} {request.url}")

    response = await call_next(request)

    process_time = round(time.time() - start_time, 4)
    logger.info(
        f"Completed request: {request.method} {request.url} "
        f"| Status: {response.status_code} "
        f"| Time: {process_time}s"
    )

    return response

# -------------------------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------------------------
@app.post("/predict")
def predict(data: PatientData):
    """
    Predict heart disease risk.
    Returns predicted class and confidence score.
    """

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    # Convert input to DataFrame (prevents sklearn warnings)
    X = pd.DataFrame([data.features], columns=columns)

    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][1]

    logger.info(
        f"Prediction made | Prediction: {prediction} | Confidence: {confidence:.4f}"
    )

    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }

# -------------------------------------------------------------------
# Health Check Endpoint (Kubernetes Ready)
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
