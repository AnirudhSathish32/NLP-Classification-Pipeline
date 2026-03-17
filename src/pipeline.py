"""
src/pipeline.py
---------------
End-to-end pipeline orchestration for the sentiment classifier.

What it does:
    1. Splits data into train/test.
    2. Builds a sklearn Pipeline (TextCleaner → TF-IDF → model) for each
       candidate model.
    3. Runs stratified cross-validation.
    4. Fits and evaluates each pipeline on the test set.
    5. Trains the PyTorch MLP as a special case (no CV).
    6. Selects the best model by test AUC.
    7. Saves the best pipeline to disk.
    8. Writes a model_results.json summary.

Phase 1 addition:
    All training runs are now tracked in MLflow.  A parent run is opened for
    the full training session; each individual model gets a nested child run
    via src/mlflow_tracking.log_sklearn_run().

Connects to:
    mlflow_config.py         — configure_mlflow() called at module import
    src/mlflow_tracking.py   — log_sklearn_run(), log_test_metrics()
    src/model_training.py    — get_models(), cross_validate_pipeline()
    src/feature_engineering.py — get_tfidf_vectorizer()
    src/text_preprocessing.py  — TextCleaner
    src/evaluation.py          — evaluate_model()
    src/torch_classifier.py    — TorchTextClassifier
"""

import json
import logging
import os
from typing import Tuple, Dict, Any

import joblib
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .evaluation import evaluate_model
from .feature_engineering import get_tfidf_vectorizer
from .model_training import get_models, cross_validate_pipeline
from .text_preprocessing import TextCleaner
from .torch_classifier import TorchTextClassifier
from .mlflow_tracking import log_sklearn_run, log_test_metrics

logger = logging.getLogger(__name__)


def _log_pytorch_run(
    model_name: str,
    pipeline,
    X_test,
    y_test,
) -> str:
    """
    Open an MLflow child run and log everything for the PyTorch MLP.

    What it does:
        PyTorch MLP skips cross-validation so it gets its own lightweight
        logging path: params + test metrics + artifacts.

    Parameters:
        model_name (str):  Run name / artifact prefix (e.g. 'pytorch_mlp').
        pipeline:          A fitted sklearn Pipeline ending with TorchTextClassifier.
        X_test:            Raw text test features.
        y_test:            Test labels.

    Returns:
        str: The MLflow run_id for this child run.

    Connects to:
        run_pipeline() below — called once after the PyTorch pipeline is fit.
        src/mlflow_tracking.py — reuses log_test_metrics() and artifact helpers.
    """
    from .mlflow_tracking import (
        log_tfidf_params,
        log_vectorizer_artifact,
        log_pipeline_artifact,
    )

    with mlflow.start_run(run_name=model_name, nested=True, tags={"model_type": model_name}) as run:
        log_tfidf_params(pipeline.named_steps["tfidf"])

        # Log PyTorch-specific hyperparameters
        torch_model = pipeline.named_steps["model"]
        mlflow.log_params({
            "model_name":       model_name,
            "model_hidden_size": torch_model.hidden_size,
            "model_epochs":      torch_model.epochs,
            "model_batch_size":  torch_model.batch_size,
            "model_lr":          torch_model.lr,
        })

        # Test metrics
        probs = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        log_test_metrics(y_test, y_pred, probs)

        log_pipeline_artifact(pipeline, model_name)
        log_vectorizer_artifact(pipeline, model_name)

        run_id = run.info.run_id
        logger.info(f"MLflow run complete for {model_name}: run_id={run_id}")
        return run_id


def run_pipeline(X, y) -> Tuple[str, Dict[str, Any]]:
    """
    Orchestrate the full training, evaluation, and model-selection pipeline.

    What it does:
        Trains all candidate models, logs every run to MLflow, selects the
        best model by test AUC, and persists it to disk.

    Parameters:
        X: pd.Series of raw review strings.
        y: pd.Series of integer labels (0 = negative, 1 = positive).

    Returns:
        Tuple[str, dict]:
            - best_model_name: key of the winning model.
            - results: dict mapping model_name → {cv_mean_auc, cv_std_auc, test_auc}.

    Connects to:
        main.py                  — entry point calls run_pipeline(X, y).
        mlflow_config.py         — configure_mlflow() must be called before this.
        src/mlflow_tracking.py   — log_sklearn_run(), _log_pytorch_run().
    """
    # Import here to avoid circular imports; configure_mlflow() was already
    # called in main.py before run_pipeline() is invoked.
    from mlflow_config import EXPERIMENT_NAME  # noqa: F401 (side-effect import check)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = get_models()
    sklearn_results = {k: v for k, v in results.items() if k != "pytorch_mlp"}
    best_model_name = max(sklearn_results, key=lambda k: sklearn_results[k]["test_auc"])
    trained_pipelines: Dict[str, Pipeline] = {}

    # -----------------------------------------------------------------------
    # Parent MLflow run — groups all child model runs for this training session
    # -----------------------------------------------------------------------
    with mlflow.start_run(run_name="training_session") as parent_run:
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        logger.info(f"MLflow parent run started: {parent_run.info.run_id}")

        # -------------------------------------------------------------------
        # Classical sklearn models
        # -------------------------------------------------------------------
        joblib.dump(best_pipeline, "models/best_model.pkl")
        joblib.dump(torch_pipe, "models/pytorch_mlp_pipeline.pkl") 
        
        for name, model in models.items():
            logger.info(f"Training {name}...")

            pipe = Pipeline([
                ("cleaner", TextCleaner()),
                ("tfidf", get_tfidf_vectorizer()),
                ("model", model),
            ])

            mean_auc, std_auc = cross_validate_pipeline(pipe, X_train, y_train)
            pipe.fit(X_train, y_train)
            test_auc = evaluate_model(pipe, X_test, y_test, name)

            # MLflow logging (Phase 1 addition)
            log_sklearn_run(
                model_name=name,
                pipeline=pipe,
                mean_auc=mean_auc,
                std_auc=std_auc,
                X_test=X_test,
                y_test=y_test,
            )

            results[name] = {
                "cv_mean_auc": mean_auc,
                "cv_std_auc":  std_auc,
                "test_auc":    test_auc,
            }
            trained_pipelines[name] = pipe

        # -------------------------------------------------------------------
        # PyTorch MLP (no CV — handled separately)
        # -------------------------------------------------------------------
        logger.info("Training pytorch_mlp...")
        torch_pipe = Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf",   get_tfidf_vectorizer()),
            ("model",   TorchTextClassifier(
                hidden_size=128, epochs=10, batch_size=32, lr=0.001
            )),
        ])
        torch_pipe.fit(X_train, y_train)

        probs    = torch_pipe.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, probs)

        # MLflow logging (Phase 1 addition)
        _log_pytorch_run("pytorch_mlp", torch_pipe, X_test, y_test)

        results["pytorch_mlp"] = {
            "cv_mean_auc": None,
            "cv_std_auc":  None,
            "test_auc":    test_auc,
        }
        trained_pipelines["pytorch_mlp"] = torch_pipe

        # -------------------------------------------------------------------
        # Model selection
        # -------------------------------------------------------------------
        best_model_name = max(results, key=lambda k: results[k]["test_auc"])
        best_pipeline   = trained_pipelines[best_model_name]
        logger.info(f"Best model: {best_model_name}")

        # Log the winning model name on the parent run for easy filtering
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_test_auc", results[best_model_name]["test_auc"])

    # -----------------------------------------------------------------------
    # Persist best model to disk (unchanged from original)
    # -----------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/best_model.pkl")

    print("writing model results to models/model_results.json")
    serializable_results = {
        model_name: {
            k: (float(v) if v is not None else None)
            for k, v in metrics.items()
        }
        for model_name, metrics in results.items()
    }

    with open("models/model_results.json", "w") as f:
        json.dump(
            {"best_model": best_model_name, "metrics": serializable_results},
            f,
            indent=4,
        )

    print("\n=== Model Comparison ===")
    for name, vals in results.items():
        cv_mean = f"{vals['cv_mean_auc']:.4f}" if vals["cv_mean_auc"] is not None else "N/A"
        cv_std  = f"{vals['cv_std_auc']:.4f}"  if vals["cv_std_auc"]  is not None else "N/A"
        print(f"{name}: CV={cv_mean} ± {cv_std} | Test AUC={vals['test_auc']:.4f}")

    return best_model_name, results
