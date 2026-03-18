"""
app.py
------
FastAPI serving layer for the sentiment analysis pipeline.
 
What it does:
    Loads three models at startup and exposes a single POST /predict endpoint.
    One review in — three independent sentiment results out, one from each model.
 
    Models loaded at startup:
        1. Best sklearn pipeline  (models/best_model.pkl)
           TextCleaner -> TF-IDF -> best classical model (e.g. LogisticRegression)
 
        2. PyTorch MLP pipeline   (models/pytorch_mlp_pipeline.pkl)
           TextCleaner -> TF-IDF -> TorchTextClassifier
 
        3. BERT                   (models/bert_checkpoints/best_checkpoint/)
           BertForSequenceClassification loaded via HuggingFace from_pretrained()
           Tokeniser loaded from same checkpoint directory.
           Tokenisation handled inline at inference time.
 
    If a model file is missing at startup, that model's result in the response
    will carry an error message rather than crashing the whole application.
 
Connects to:
    src/pipeline.py        — produces best_model.pkl + pytorch_mlp_pipeline.pkl
    bert_classifier.py     — produces models/bert_checkpoints/best_checkpoint/
    bert_config.yaml       — reads max_length for BERT tokenisation
    models/model_results.json — all model metrics displayed in every response
 
Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
    docker compose up api
"""
 
import json
import logging
import os
from typing import Any, Dict, Optional
 
import joblib
import numpy as np
import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizerFast
 
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
 
# ---------------------------------------------------------------------------
# Paths — override via environment variables
# ---------------------------------------------------------------------------
SKLEARN_MODEL_PATH  = os.environ.get("SKLEARN_MODEL_PATH",  "models/best_model.pkl")
PYTORCH_MODEL_PATH  = os.environ.get("PYTORCH_MODEL_PATH",  "models/pytorch_mlp_pipeline.pkl")
BERT_CHECKPOINT_DIR = os.environ.get("BERT_CHECKPOINT_DIR", "models/bert_checkpoints/best_checkpoint")
BERT_CONFIG_PATH    = os.environ.get("BERT_CONFIG_PATH",    "bert_config.yaml")
METRICS_PATH        = os.environ.get("METRICS_PATH",        "models/model_results.json")
 
# ---------------------------------------------------------------------------
# Load BERT max_length from config (defaults to 256 if config missing)
# ---------------------------------------------------------------------------
_BERT_MAX_LENGTH = 256
if os.path.exists(BERT_CONFIG_PATH):
    with open(BERT_CONFIG_PATH) as _f:
        _BERT_MAX_LENGTH = yaml.safe_load(_f).get("max_length", 256)
 
logger.info(f"BERT max_length set to: {_BERT_MAX_LENGTH}")
 
 
# ---------------------------------------------------------------------------
# Model loading at startup
# ---------------------------------------------------------------------------
 
def _load_sklearn_model():
    """
    Load the best sklearn pipeline from disk.
 
    Returns:
        Fitted sklearn Pipeline or None if file is missing.
    """
    if not os.path.exists(SKLEARN_MODEL_PATH):
        logger.error(f"Sklearn model not found: '{SKLEARN_MODEL_PATH}'. Run main.py first.")
        return None
    model = joblib.load(SKLEARN_MODEL_PATH)
    logger.info(f"Sklearn model loaded from '{SKLEARN_MODEL_PATH}'")
    return model
 
 
def _load_pytorch_model():
    """
    Load the PyTorch MLP pipeline from disk.
 
    Returns:
        Fitted sklearn Pipeline (with TorchTextClassifier step) or None.
    """
    if not os.path.exists(PYTORCH_MODEL_PATH):
        logger.error(f"PyTorch model not found: '{PYTORCH_MODEL_PATH}'. Run main.py first.")
        return None
    model = joblib.load(PYTORCH_MODEL_PATH)
    logger.info(f"PyTorch MLP pipeline loaded from '{PYTORCH_MODEL_PATH}'")
    return model
 
 
def _load_bert_model():
    """
    Load the BERT checkpoint and tokenizer from the same checkpoint directory.
 
    What it does:
        Both model weights and tokenizer files must exist in BERT_CHECKPOINT_DIR.
        The tokenizer is saved there by bert_classifier.save_checkpoint() during
        training. Loading from the same directory guarantees the tokenizer matches
        the one used during fine-tuning.
 
    Returns:
        Tuple (BertForSequenceClassification, BertTokenizerFast) or (None, None).
    """
    if not os.path.exists(BERT_CHECKPOINT_DIR):
        logger.error(
            f"BERT checkpoint not found: '{BERT_CHECKPOINT_DIR}'. "
            "Run bert_classifier.py first."
        )
        return None, None
 
    # Verify tokenizer files exist alongside model weights
    tokenizer_config = os.path.join(BERT_CHECKPOINT_DIR, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config):
        logger.warning(
            f"Tokenizer files not found in '{BERT_CHECKPOINT_DIR}'. "
            "The checkpoint may have been saved without the tokenizer. "
            "Falling back to bert-base-uncased tokenizer from HuggingFace cache."
        )
 
    logger.info(f"Loading BERT from '{BERT_CHECKPOINT_DIR}' (this may take a moment)...")
 
    # Log file size to confirm correct checkpoint is loaded
    safetensors_path = os.path.join(BERT_CHECKPOINT_DIR, "model.safetensors")
    if os.path.exists(safetensors_path):
        size_mb = os.path.getsize(safetensors_path) / 1024 / 1024
        logger.info(f"BERT model.safetensors size: {size_mb:.1f} MB")
 
    bert_model     = BertForSequenceClassification.from_pretrained(BERT_CHECKPOINT_DIR)
    bert_tokenizer = BertTokenizerFast.from_pretrained(BERT_CHECKPOINT_DIR)
    bert_model.eval()
    logger.info("BERT model and tokenizer loaded.")
    return bert_model, bert_tokenizer
 
 
def _load_metrics() -> Dict[str, Any]:
    """
    Load model_results.json for display in every response.
 
    Returns:
        dict: Full metrics dict, or empty dict if file is missing.
    """
    if not os.path.exists(METRICS_PATH):
        logger.warning(f"Metrics file not found: '{METRICS_PATH}'")
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)
 
 
# ---------------------------------------------------------------------------
# Load everything once at startup
# ---------------------------------------------------------------------------
sklearn_model              = _load_sklearn_model()
pytorch_model              = _load_pytorch_model()
bert_model, bert_tokenizer = _load_bert_model()
model_info                 = _load_metrics()
 
# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "Submit a review and receive sentiment predictions from three independent models: "
        "the best sklearn classifier, a PyTorch MLP, and a fine-tuned BERT model."
    ),
    version="2.0.0",
)
 
 
# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
 
class ReviewRequest(BaseModel):
    text: str
 
    class Config:
        json_schema_extra = {
            "example": {"text": "This movie was absolutely fantastic!"}
        }
 
 
class ModelResult(BaseModel):
    sentiment:  str
    confidence: float
    error:      Optional[str] = None
 
 
class PredictionResponse(BaseModel):
    review:             str
    sklearn:            ModelResult
    pytorch:            ModelResult
    bert:               ModelResult
    best_sklearn_model: str
    all_model_metrics:  Dict[str, Any]
 
 
# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
 
def _run_sklearn(text: str) -> ModelResult:
    """
    Run inference using the best sklearn pipeline.
 
    Parameters:
        text (str): Raw review string.
 
    Returns:
        ModelResult with sentiment and confidence.
    """
    if sklearn_model is None:
        return ModelResult(sentiment="unknown", confidence=0.0, error="Model not loaded")
 
    try:
        prediction    = sklearn_model.predict([text])[0]
        probabilities = sklearn_model.predict_proba([text])[0]
        sentiment     = "positive" if prediction == 1 else "negative"
        confidence    = float(np.max(probabilities))
        return ModelResult(sentiment=sentiment, confidence=confidence)
    except Exception as exc:
        logger.error(f"Sklearn inference failed: {exc}")
        return ModelResult(sentiment="unknown", confidence=0.0, error=str(exc))
 
 
def _run_pytorch(text: str) -> ModelResult:
    """
    Run inference using the PyTorch MLP pipeline.
 
    Parameters:
        text (str): Raw review string.
 
    Returns:
        ModelResult with sentiment and confidence.
    """
    if pytorch_model is None:
        return ModelResult(sentiment="unknown", confidence=0.0, error="Model not loaded")
 
    try:
        prediction    = pytorch_model.predict([text])[0]
        probabilities = pytorch_model.predict_proba([text])[0]
        sentiment     = "positive" if prediction == 1 else "negative"
        confidence    = float(np.max(probabilities))
        return ModelResult(sentiment=sentiment, confidence=confidence)
    except Exception as exc:
        logger.error(f"PyTorch inference failed: {exc}")
        return ModelResult(sentiment="unknown", confidence=0.0, error=str(exc))
 
 
def _run_bert(text: str) -> ModelResult:
    """
    Run inference using the fine-tuned BERT model.
 
    What it does:
        Tokenises the raw text using the tokenizer loaded from the checkpoint
        directory, runs a forward pass through BertForSequenceClassification,
        applies softmax to the logits, and returns the predicted class and
        confidence score.
 
    Parameters:
        text (str): Raw review string.
 
    Returns:
        ModelResult with sentiment and confidence.
    """
    if bert_model is None or bert_tokenizer is None:
        return ModelResult(sentiment="unknown", confidence=0.0, error="Model not loaded")
 
    try:
        encoding = bert_tokenizer(
            text,
            max_length=_BERT_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
 
        with torch.no_grad():
            outputs = bert_model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                token_type_ids=encoding["token_type_ids"],
            )
 
        probs      = torch.softmax(outputs.logits, dim=1)[0]
        class_idx  = int(torch.argmax(probs).item())
        sentiment  = "positive" if class_idx == 1 else "negative"
        confidence = float(probs[class_idx].item())
 
        return ModelResult(sentiment=sentiment, confidence=confidence)
 
    except Exception as exc:
        logger.error(f"BERT inference failed: {exc}")
        return ModelResult(sentiment="unknown", confidence=0.0, error=str(exc))
 
 
# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
 
@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest) -> PredictionResponse:
    """
    Analyse a review using all three models and return their results.
 
    Submit any movie review and receive three independent sentiment predictions:
    - **sklearn**: Best classical model selected by ROC-AUC (e.g. Logistic Regression)
    - **pytorch**: PyTorch MLP trained on TF-IDF features
    - **bert**: Fine-tuned BERT (bert-base-uncased) on IMDB 50K
 
    Each result includes a sentiment label (positive/negative) and a confidence
    score between 0 and 1. If a model failed to load at startup, its result
    will contain an error message instead.
    """
    text = request.text.strip()
    if not text:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
 
    sklearn_result = _run_sklearn(text)
    pytorch_result = _run_pytorch(text)
    bert_result    = _run_bert(text)
 
    best_sklearn = model_info.get("best_model", "unknown")
    all_metrics  = model_info.get("metrics", {})
 
    return PredictionResponse(
        review=text,
        sklearn=sklearn_result,
        pytorch=pytorch_result,
        bert=bert_result,
        best_sklearn_model=best_sklearn,
        all_model_metrics=all_metrics,
    )