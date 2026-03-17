# Sentiment Analysis Pipeline
 
A modular NLP pipeline that classifies movie reviews using three independent
models — a classical sklearn classifier, a PyTorch MLP, and a fine-tuned BERT
model. Submit any review and receive sentiment predictions from all three in
one API response.
 
---
 
## Architecture
 
```
Raw IMDB Data (50K reviews)
        │
        ▼
prepare_data_csv.py ──► data/raw/dataset.csv
        │
        ▼
main.py — Classical + PyTorch training
        ├── Logistic Regression  ┐
        ├── Linear SVM           ├── best_model.pkl  (best by ROC-AUC)
        ├── Random Forest        ┘
        ├── MLP Classifier       ┘
        └── PyTorch MLP ──────────── pytorch_mlp_pipeline.pkl
        │
        ▼
bert_classifier.py — BERT fine-tuning
        └── bert-base-uncased ────── models/bert_checkpoints/
        │
        ▼
app.py — FastAPI (port 8000)
        └── POST /predict
              ├── sklearn  → best classical model result
              ├── pytorch  → PyTorch MLP result
              └── bert     → fine-tuned BERT result
```
 
---
 
## Model Comparison
 
| Model | Type | Competes for best model |
|-------|------|------------------------|
| Logistic Regression | sklearn | ✅ |
| Linear SVM | sklearn | ✅ |
| Random Forest | sklearn | ✅ |
| MLP Classifier | sklearn | ✅ |
| PyTorch MLP | PyTorch (sklearn interface) | ❌ served independently |
| BERT | HuggingFace Transformers | ❌ served independently |
 
Best model selection uses ROC-AUC on the held-out test set across sklearn
models only. All six model metrics are returned in every API response.
 
---
 
## MLflow Experiment Tracking
 
Every training run is tracked in MLflow — hyperparameters, per-epoch metrics,
and model artifacts. The classical pipeline and BERT fine-tuning both log to
the same sentiment_classifier experiment so all models can be compared
side-by-side.
 
---
 
## Portfolio Components
 
The following components demonstrate production MLOps patterns but are not
part of the running application:
 
- export_onnx.py — exports the PyTorch MLP to ONNX format with onnxruntime validation
- onnx/handler.py — TorchServe handler showing how the ONNX model would be served at scale
 
---
 
## Requirements
 
- Docker + Docker Compose
- NVIDIA GPU with CUDA 12.8 support (RTX 50 series confirmed working)
- NVIDIA Container Toolkit installed on the host
 
Verify your GPU is accessible to Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```
 
---
 
## Setup
 
### Step 1 — Download the IMDB dataset
 
Download from: https://ai.stanford.edu/~amaas/data/sentiment/
 
Extract so the structure looks like:
```
data/
└── raw/
    └── aclImdb_v1/
        └── aclImdb/
            ├── train/
            │   ├── pos/
            │   └── neg/
            └── test/
                ├── pos/
                └── neg/
```
 
### Step 2 — Train all models
```bash
docker compose run --rm trainer
```
 
This runs main.py then bert_classifier.py sequentially inside a GPU-enabled
container. All model files are written to the shared models/ volume.
 
Expected training time on RTX 5070 Ti:
- Classical models + PyTorch MLP: ~10-15 minutes
- BERT fine-tuning (3 epochs, max_length=512): ~30-40 minutes
 
### Step 3 — Start the API
```bash
docker compose up api
```
 
Open http://localhost:8000/docs for the Swagger UI.
 
---
 
## API
 
### POST /predict
 
Submit any movie review and receive independent sentiment predictions from
all three models.
 
**Request:**
```json
{
  "text": "This movie was absolutely fantastic!"
}
```
 
**Response:**
```json
{
  "review": "This movie was absolutely fantastic!",
  "sklearn": {
    "sentiment": "positive",
    "confidence": 0.94
  },
  "pytorch": {
    "sentiment": "positive",
    "confidence": 0.88
  },
  "bert": {
    "sentiment": "positive",
    "confidence": 0.99
  },
  "best_sklearn_model": "logistic_regression",
  "all_model_metrics": {
    "logistic_regression": {"test_auc": 0.9580},
    "linear_svm":          {"test_auc": 0.9529},
    "random_forest":       {"test_auc": 0.9275},
    "mlp":                 {"test_auc": 0.9568},
    "pytorch_mlp":         {"test_auc": 0.9392},
    "bert":                {"test_auc": 0.9750}
  }
}
```
 
If a model failed to load (e.g. training has not been run yet), its result
will contain an error field rather than crashing the application.
 
---
 
## Viewing MLflow Results
 
After training, view all experiment runs in the MLflow UI:
```bash
mlflow ui --backend-store-uri file:./experiments/mlruns --port 5000
```
Open http://localhost:5000
 
---
 
## Rebuilding After Code Changes
 
```bash
docker compose build
docker compose run --rm trainer
docker compose up api
```
 
---
 
## Dataset
 
IMDB Large Movie Review Dataset
Andrew L. Maas et al., 2011
https://ai.stanford.edu/~amaas/data/sentiment/
 