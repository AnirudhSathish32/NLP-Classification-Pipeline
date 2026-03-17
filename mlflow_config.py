"""
mlflow_config.py
----------------
Shared MLflow configuration for all training runs.

What it does:
    Sets the tracking URI and experiment name used by every component
    that logs to MLflow (pipeline.py, bert_classifier.py).

Parameters:
    None — all values read from environment variables with sensible defaults.

Returns:
    N/A — imported as a module; call `configure_mlflow()` at the top of
    any training entry point.

Connects to:
    src/pipeline.py     — classical ML training runs
    bert_classifier.py  — BERT fine-tuning runs (Phase 3)
    export_onnx.py      — ONNX artifact logging (Phase 2)
"""

import os
import logging
import mlflow

# ---------------------------------------------------------------------------
# Constants (override via environment variables)
# ---------------------------------------------------------------------------

# Local filesystem URI — all run data lands in experiments/mlruns/
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "file:./experiments/mlruns"
)

EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "sentiment_classifier"
)

logger = logging.getLogger(__name__)


def configure_mlflow() -> str:
    """
    Set the MLflow tracking URI and create/activate the experiment.

    What it does:
        1. Points mlflow at a local directory (or remote URI from env var).
        2. Creates the experiment if it does not already exist.
        3. Sets it as the active experiment for the current process.

    Parameters:
        None

    Returns:
        str: The experiment ID that was activated.

    Connects to:
        Called once at the start of src/pipeline.py and bert_classifier.py.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=os.path.join(
                MLFLOW_TRACKING_URI.replace("file:", ""),
                EXPERIMENT_NAME
            )
        )
        logger.info(f"Created MLflow experiment '{EXPERIMENT_NAME}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{EXPERIMENT_NAME}' (id={experiment_id})")

    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id
