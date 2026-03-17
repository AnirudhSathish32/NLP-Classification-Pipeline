"""
src/mlflow_tracking.py
-----------------------
Reusable MLflow logging helpers used during training.

What it does:
    Provides functions for logging hyperparameters, metrics, and artifacts
    (models, vectorizers) inside an active MLflow run. Centralising this
    logic here keeps pipeline.py and bert_classifier.py free of boilerplate.

Parameters:
    See individual function signatures below.

Returns:
    See individual function signatures below.

Connects to:
    src/pipeline.py         — calls log_sklearn_run() for each classical model
    bert_classifier.py      — calls log_bert_epoch_metrics() each epoch
    export_onnx.py          — calls log_onnx_artifact() after export
"""

import logging
import os
import tempfile
from typing import Any, Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def log_tfidf_params(vectorizer) -> None:
    """
    Log TF-IDF vectorizer hyperparameters to the active MLflow run.

    What it does:
        Extracts the settings from a fitted or unfitted TfidfVectorizer and
        logs them as mlflow params with a 'tfidf_' prefix.

    Parameters:
        vectorizer: A sklearn TfidfVectorizer instance.

    Returns:
        None

    Connects to:
        Called from log_sklearn_run() inside src/pipeline.py.
    """
    params = {
        "tfidf_max_features": vectorizer.max_features,
        "tfidf_ngram_range": str(vectorizer.ngram_range),
        "tfidf_stop_words": str(vectorizer.stop_words),
        "tfidf_min_df": vectorizer.min_df,
    }
    mlflow.log_params(params)
    logger.debug(f"Logged TF-IDF params: {params}")


def log_model_params(model_name: str, model) -> None:
    """
    Log model hyperparameters to the active MLflow run.

    What it does:
        Calls get_params() on any sklearn-compatible estimator and logs the
        result with a 'model_' prefix so params from different model types
        don't collide in the same experiment.

    Parameters:
        model_name (str): Human-readable model identifier (e.g. 'logistic_regression').
        model:            Any sklearn estimator exposing get_params().

    Returns:
        None

    Connects to:
        Called from log_sklearn_run() inside src/pipeline.py.
    """
    try:
        raw_params = model.get_params()
        prefixed = {f"model_{k}": str(v) for k, v in raw_params.items()}
        # MLflow param values must be strings ≤ 500 chars
        prefixed = {k: v[:500] for k, v in prefixed.items()}
        mlflow.log_params(prefixed)
        logger.debug(f"Logged params for {model_name}: {prefixed}")
    except Exception as exc:
        logger.warning(f"Could not log params for {model_name}: {exc}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def log_cv_metrics(mean_auc: float, std_auc: float) -> None:
    """
    Log cross-validation ROC-AUC statistics to the active MLflow run.

    Parameters:
        mean_auc (float): Mean ROC-AUC across CV folds.
        std_auc  (float): Std dev of ROC-AUC across CV folds.

    Returns:
        None

    Connects to:
        Called from log_sklearn_run() after cross_validate_pipeline().
    """
    mlflow.log_metrics({
        "cv_mean_auc": mean_auc,
        "cv_std_auc": std_auc,
    })


def log_test_metrics(
    y_test,
    y_pred,
    y_scores,
    prefix: str = "test",
) -> float:
    """
    Log a full suite of classification metrics on the held-out test set.

    What it does:
        Computes and logs accuracy, ROC-AUC, macro F1, macro precision,
        and macro recall.  Returns the ROC-AUC for downstream use.

    Parameters:
        y_test   : True labels (array-like).
        y_pred   : Predicted labels (array-like).
        y_scores : Probability scores for the positive class (array-like).
        prefix   (str): Metric name prefix, default 'test'.

    Returns:
        float: ROC-AUC score on the test set.

    Connects to:
        Called from log_sklearn_run() in src/pipeline.py.
        Also called from bert_classifier.py after final evaluation.
    """
    roc_auc = roc_auc_score(y_test, y_scores)
    metrics = {
        f"{prefix}_roc_auc":  roc_auc,
        f"{prefix}_accuracy": accuracy_score(y_test, y_pred),
        f"{prefix}_f1":       f1_score(y_test, y_pred, average="macro"),
        f"{prefix}_precision":precision_score(y_test, y_pred, average="macro", zero_division=0),
        f"{prefix}_recall":   recall_score(y_test, y_pred, average="macro", zero_division=0),
    }
    mlflow.log_metrics(metrics)
    logger.info(f"Logged test metrics: {metrics}")
    return roc_auc


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def log_pipeline_artifact(pipeline, model_name: str) -> None:
    """
    Serialise and log a fitted sklearn Pipeline as an MLflow artifact.

    What it does:
        Dumps the full pipeline (cleaner + tfidf + model) to a temp .pkl
        file and logs it under artifacts/models/<model_name>/.

    Parameters:
        pipeline:    A fitted sklearn Pipeline object.
        model_name (str): Used as the artifact subdirectory name.

    Returns:
        None

    Connects to:
        Called from log_sklearn_run() in src/pipeline.py.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{model_name}_pipeline.pkl")
        joblib.dump(pipeline, path)
        mlflow.log_artifact(path, artifact_path=f"models/{model_name}")
        logger.info(f"Logged pipeline artifact for {model_name}")


def log_vectorizer_artifact(pipeline, model_name: str) -> None:
    """
    Extract and log the TF-IDF vectorizer from a fitted Pipeline as an artifact.

    What it does:
        Pulls the fitted TfidfVectorizer out of the pipeline's 'tfidf' step
        and saves it separately so it can be reused during ONNX export or
        TorchServe preprocessing (Phase 2).

    Parameters:
        pipeline:    A fitted sklearn Pipeline whose second step is 'tfidf'.
        model_name (str): Used as the artifact subdirectory name.

    Returns:
        None

    Connects to:
        Called from log_sklearn_run() in src/pipeline.py.
        The saved vectorizer is consumed by export_onnx.py (Phase 2).
    """
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tfidf_vectorizer.pkl")
            joblib.dump(vectorizer, path)
            mlflow.log_artifact(path, artifact_path=f"models/{model_name}")
            logger.info(f"Logged TF-IDF vectorizer artifact for {model_name}")
    except KeyError:
        logger.warning("Pipeline has no 'tfidf' step; vectorizer not logged.")


# ---------------------------------------------------------------------------
# High-level convenience wrapper
# ---------------------------------------------------------------------------

def log_sklearn_run(
    model_name: str,
    pipeline,
    mean_auc: float,
    std_auc: float,
    X_test,
    y_test,
    run_tags: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Open an MLflow child run and log everything for one classical ML model.

    What it does:
        1. Starts a nested MLflow run named after model_name.
        2. Logs TF-IDF hyperparameters.
        3. Logs model hyperparameters.
        4. Logs CV metrics (mean/std AUC).
        5. Logs test-set metrics (AUC, accuracy, F1, precision, recall).
        6. Logs the fitted pipeline as an artifact.
        7. Logs the fitted TF-IDF vectorizer as a separate artifact.

    Parameters:
        model_name (str):  Identifier used for run name and artifact paths.
        pipeline:          A fitted sklearn Pipeline (cleaner → tfidf → model).
        mean_auc (float):  Mean CV ROC-AUC from cross_validate_pipeline().
        std_auc  (float):  Std dev CV ROC-AUC from cross_validate_pipeline().
        X_test:            Test feature array (raw text strings).
        y_test:            Test labels.
        run_tags (dict):   Optional extra tags attached to the MLflow run.

    Returns:
        str: The MLflow run_id for this child run.

    Connects to:
        src/pipeline.py  — called once per model inside run_pipeline().
    """
    tags = {"model_type": model_name}
    if run_tags:
        tags.update(run_tags)

    with mlflow.start_run(run_name=model_name, nested=True, tags=tags) as run:
        # --- params ---
        log_tfidf_params(pipeline.named_steps["tfidf"])
        log_model_params(model_name, pipeline.named_steps["model"])
        mlflow.log_param("model_name", model_name)

        # --- cv metrics ---
        log_cv_metrics(mean_auc, std_auc)

        # --- test metrics ---
        y_pred = pipeline.predict(X_test)
        model_step = pipeline.named_steps["model"]
        if hasattr(model_step, "predict_proba"):
            y_scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_scores = pipeline.decision_function(X_test)

        log_test_metrics(y_test, y_pred, y_scores)

        # --- artifacts ---
        log_pipeline_artifact(pipeline, model_name)
        log_vectorizer_artifact(pipeline, model_name)

        run_id = run.info.run_id
        logger.info(f"MLflow run complete for {model_name}: run_id={run_id}")
        return run_id
