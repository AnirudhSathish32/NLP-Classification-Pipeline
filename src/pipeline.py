import logging
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

from .text_preprocessing import TextCleaner
from .feature_engineering import get_tfidf_vectorizer
from .model_training import get_models, cross_validate_pipeline
from .evaluation import evaluate_model
from src.torch_classifier import TorchTextClassifier
from sklearn.metrics import roc_auc_score
import json


def run_pipeline(X, y):
    """
    End-to-end pipeline orchestration.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    models = get_models()

    results = {}
    trained_pipelines = {}

    for name, model in models.items():
        logging.info(f"Training {name}...")

        pipe = Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", get_tfidf_vectorizer()),
            ("model", model)
        ])

        mean_auc, std_auc = cross_validate_pipeline(pipe, X_train, y_train)

        pipe.fit(X_train, y_train)

        test_auc = evaluate_model(pipe, X_test, y_test, name)

        results[name] = {
            "cv_mean_auc": mean_auc,
            "cv_std_auc": std_auc,
            "test_auc": test_auc
        }

        trained_pipelines[name] = pipe

    logging.info("Training pytorch_mlp...")
    pipe = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", get_tfidf_vectorizer()),
        ("model", TorchTextClassifier(
            hidden_size=128, epochs=10, batch_size=32, lr=0.001
        ))
    ])
    pipe.fit(X_train, y_train)
    # Evaluate using ROC-AUC manually
    probs = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, probs)

    results["pytorch_mlp"] = {
        "cv_mean_auc": None,  # No CV for PyTorch
        "cv_std_auc": None,
        "test_auc": test_auc
    }
    trained_pipelines["pytorch_mlp"] = pipe

    # select best
    best_model_name = max(results, key=lambda k: results[k]["test_auc"])
    best_pipeline = trained_pipelines[best_model_name]

    logging.info(f"Best model: {best_model_name}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_pipeline, "models/best_model.pkl")
    print("writing model reslts to models/model_results.json")
    # Save model comparison metrics as JSON
    
    
    # Convert numpy types to native Python types
    serializable_results = {}

    for model_name, metrics in results.items():
        serializable_results[model_name] = {
            k: (float(v) if v is not None else None)
            for k, v in metrics.items()
        }

    with open("models/model_results.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "metrics": serializable_results
        }, f, indent=4)

    print("\n=== Model Comparison ===")
    for name, vals in results.items():
        cv_mean = f"{vals['cv_mean_auc']:.4f}" if vals['cv_mean_auc'] is not None else "N/A"
        cv_std = f"{vals['cv_std_auc']:.4f}" if vals['cv_std_auc'] is not None else "N/A"
        print(
        f"{name}: CV={cv_mean} ± {cv_std} | Test AUC={vals['test_auc']:.4f}"
    )

    return best_model_name, results
