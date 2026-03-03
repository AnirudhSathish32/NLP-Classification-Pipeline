import json
import os 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model once at startup
print("Files in models directory:", os.listdir("models"))
MODEL_PATH = "models/best_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

METRICS_PATH = "models/model_results.json"
try:
    with open(METRICS_PATH) as f:
        model_info = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model metrics: {e}")

app = FastAPI(title="Sentiment Analysis API")

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_used: str
    all_model_metrics: dict

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    # Predict
    prediction = model.predict([request.text])[0]

    # Confidence (only if classifier supports predict_proba)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([request.text])[0]
        confidence = float(np.max(probabilities))
    else:
        confidence = None

    sentiment = "positive" if prediction == 1 else "negative"

    return PredictionResponse(
        prediction=sentiment,
        confidence=confidence,
        model_used=model_info["best_model"],
        all_model_metrics=model_info["metrics"]
    )