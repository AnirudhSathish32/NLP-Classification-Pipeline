"""
export_onnx.py
--------------
Exports the trained PyTorch MLP to ONNX format and validates it.

IMPORTANT — scope of this export:
    The sklearn Pipeline (TextCleaner → TF-IDF → model) cannot be exported
    to ONNX as a whole because ONNX only represents tensor operations.
    This script therefore exports ONLY the nn.Sequential neural network
    inside TorchTextClassifier.  The TF-IDF vectorizer is saved separately
    as a .pkl so that TorchServe's handler.py can run it as a preprocessing
    step before the ONNX model sees any data.

    Full inference flow after export:
        raw text
          └─> TF-IDF vectorizer (.pkl)   ← sklearn, runs in handler.py
                └─> dense float32 tensor
                      └─> ONNX model     ← this file produces this artifact
                            └─> logits → softmax → predicted class + confidence

What it does:
    1. Loads the pytorch_mlp pipeline from models/pytorch_mlp_pipeline.pkl
       (or best_model.pkl if that is the pytorch pipeline).
    2. Extracts the nn.Sequential model and fitted TF-IDF vectorizer.
    3. Exports the neural net to ONNX using torch.onnx.export().
    4. Validates the exported ONNX model with onnxruntime.
    5. Saves the TF-IDF vectorizer alongside the ONNX model.
    6. Logs both artifacts to MLflow (connects to Phase 1).

Usage:
    python export_onnx.py [--pipeline-path models/pytorch_mlp_pipeline.pkl]
                          [--output-dir model_store]
                          [--mlflow-run-id <run_id>]   # optional, resumes run

Connects to:
    mlflow_config.py         — configure_mlflow() for artifact logging
    src/torch_classifier.py  — TorchTextClassifier whose .model_ is exported
    torchserve/handler.py    — consumes onnx_model.onnx + tfidf_vectorizer.pkl
"""

import argparse
import logging
import os
import tempfile
from typing import Optional

import joblib
import mlflow
import numpy as np
import torch

from mlflow_config import configure_mlflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (all overridable via CLI args or env vars)
# ---------------------------------------------------------------------------
DEFAULT_PIPELINE_PATH = os.environ.get(
    "PYTORCH_PIPELINE_PATH", "models/pytorch_mlp_pipeline.pkl"
)
DEFAULT_OUTPUT_DIR = os.environ.get("ONNX_OUTPUT_DIR", "model_store")
ONNX_MODEL_FILENAME = "onnx_model.onnx"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_pytorch_pipeline(pipeline_path: str):
    """
    Load a fitted sklearn Pipeline that ends with TorchTextClassifier.

    Parameters:
        pipeline_path (str): Path to the serialised pipeline .pkl file.

    Returns:
        sklearn.pipeline.Pipeline: The loaded pipeline.

    Raises:
        FileNotFoundError: If the pipeline file does not exist.
        ValueError: If the pipeline's final step is not TorchTextClassifier.

    Connects to:
        src/torch_classifier.py — TorchTextClassifier must be the 'model' step.
    """
    from src.torch_classifier import TorchTextClassifier  # local import for clarity

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Pipeline not found at '{pipeline_path}'. "
            "Run main.py first to train and save the pytorch_mlp pipeline."
        )

    pipeline = joblib.load(pipeline_path)
    model_step = pipeline.named_steps.get("model")

    if not isinstance(model_step, TorchTextClassifier):
        raise ValueError(
            f"Expected pipeline's 'model' step to be TorchTextClassifier, "
            f"got {type(model_step).__name__}. "
            "Pass the pytorch_mlp pipeline explicitly via --pipeline-path."
        )

    logger.info(f"Loaded PyTorch pipeline from '{pipeline_path}'")
    return pipeline


def export_to_onnx(pipeline, output_dir: str) -> str:
    """
    Export the nn.Sequential inside TorchTextClassifier to ONNX format.

    What it does:
        1. Extracts the fitted TF-IDF vectorizer from the pipeline.
        2. Creates a representative dummy input tensor (batch=1, input_size=vocab).
        3. Calls torch.onnx.export() with named input/output signatures.
        4. Returns the path to the written .onnx file.

    Parameters:
        pipeline:    A fitted sklearn Pipeline with a TorchTextClassifier step.
        output_dir (str): Directory where onnx_model.onnx will be written.

    Returns:
        str: Absolute path to the exported ONNX file.

    Connects to:
        torchserve/handler.py — handler loads this exact file for inference.
    """
    os.makedirs(output_dir, exist_ok=True)

    torch_classifier = pipeline.named_steps["model"]
    nn_model = torch_classifier.model_          # the nn.Sequential
    input_size = torch_classifier.input_size_   # TF-IDF vocab size

    nn_model.eval()

    # Dummy input: batch of 1, full vocab width
    dummy_input = torch.zeros(1, input_size, dtype=torch.float32)

    onnx_path = os.path.join(output_dir, ONNX_MODEL_FILENAME)

    torch.onnx.export(
        nn_model,
        dummy_input,
        onnx_path,
        input_names=["tfidf_features"],     # name visible in ONNX graph
        output_names=["logits"],            # raw pre-softmax scores
        dynamic_axes={
            "tfidf_features": {0: "batch_size"},  # variable batch size
            "logits":         {0: "batch_size"},
        },
        opset_version=17,
        export_params=True,
    )

    logger.info(f"ONNX model exported to '{onnx_path}'")
    return onnx_path


def save_vectorizer(pipeline, output_dir: str) -> str:
    """
    Save the fitted TF-IDF vectorizer from the pipeline to disk.

    What it does:
        Extracts the 'tfidf' step from the pipeline and serialises it with
        joblib.  This file is consumed by torchserve/handler.py at inference
        time to convert raw text to a dense float32 tensor before the ONNX
        model runs.

    Parameters:
        pipeline:    A fitted sklearn Pipeline with a 'tfidf' step.
        output_dir (str): Directory where tfidf_vectorizer.pkl will be written.

    Returns:
        str: Absolute path to the saved vectorizer file.

    Connects to:
        torchserve/handler.py — handler loads this file for preprocessing.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    vec_path = os.path.join(output_dir, VECTORIZER_FILENAME)
    joblib.dump(vectorizer, vec_path)
    logger.info(f"TF-IDF vectorizer saved to '{vec_path}'")
    return vec_path


def validate_onnx(onnx_path: str, pipeline, num_samples: int = 3) -> None:
    """
    Run a smoke-test of the exported ONNX model using onnxruntime.

    What it does:
        1. Loads the ONNX model into an onnxruntime InferenceSession.
        2. Constructs real TF-IDF feature vectors from a tiny dummy corpus.
        3. Runs inference and asserts output shape is (N, 2).
        4. Compares ONNX logits to PyTorch logits and asserts they are close
           (absolute tolerance 1e-4).

    Parameters:
        onnx_path (str):  Path to the .onnx file to validate.
        pipeline:         The original fitted pipeline (for reference outputs).
        num_samples (int): Number of dummy samples to use in the smoke test.

    Returns:
        None

    Raises:
        AssertionError: If output shape or numerical parity checks fail.
        ImportError:    If onnxruntime is not installed.

    Connects to:
        Called immediately after export_to_onnx() in main().
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnxruntime not installed — skipping ONNX validation. "
            "Install with: pip install onnxruntime"
        )
        return

    logger.info("Validating ONNX model with onnxruntime...")

    # Build a tiny real TF-IDF feature matrix for validation
    dummy_texts = [
        "this movie was absolutely fantastic",
        "terrible waste of time",
        "not bad but not great either",
    ][:num_samples]

    vectorizer    = pipeline.named_steps["tfidf"]
    cleaner       = pipeline.named_steps["cleaner"]
    cleaned_texts = cleaner.transform(dummy_texts)
    tfidf_matrix  = vectorizer.transform(cleaned_texts)
    X_dense       = tfidf_matrix.toarray().astype(np.float32)

    # --- ONNX inference ---
    sess    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs  = {sess.get_inputs()[0].name: X_dense}
    outputs = sess.run(None, inputs)
    onnx_logits = outputs[0]  # shape: (N, 2)

    assert onnx_logits.shape == (num_samples, 2), (
        f"Unexpected ONNX output shape: {onnx_logits.shape}"
    )

    # --- PyTorch reference inference ---
    torch_model = pipeline.named_steps["model"].model_
    torch_model.eval()
    with torch.no_grad():
        pt_logits = torch_model(
            torch.tensor(X_dense, dtype=torch.float32)
        ).numpy()

    np.testing.assert_allclose(
        onnx_logits, pt_logits, atol=1e-4,
        err_msg="ONNX and PyTorch logits diverge beyond tolerance"
    )

    logger.info(
        f"✅ ONNX validation passed — output shape {onnx_logits.shape}, "
        f"max deviation from PyTorch: {np.abs(onnx_logits - pt_logits).max():.6f}"
    )


def log_onnx_artifacts_to_mlflow(
    onnx_path: str,
    vec_path: str,
    run_id: Optional[str] = None,
) -> None:
    """
    Log the ONNX model and TF-IDF vectorizer as MLflow artifacts.

    What it does:
        If run_id is given, resumes that existing MLflow run and appends the
        ONNX artifacts to it.  Otherwise opens a new run named 'onnx_export'.
        Artifacts are logged under artifacts/onnx/.

    Parameters:
        onnx_path (str):        Path to the exported .onnx file.
        vec_path  (str):        Path to the saved tfidf_vectorizer.pkl.
        run_id    (str | None): Optional MLflow run ID to resume.

    Returns:
        None

    Connects to:
        mlflow_config.py  — configure_mlflow() must have been called first.
        Phase 1 runs       — links ONNX export back to the originating train run.
    """
    configure_mlflow()

    ctx = (
        mlflow.start_run(run_id=run_id)
        if run_id
        else mlflow.start_run(run_name="onnx_export")
    )

    with ctx:
        mlflow.log_artifact(onnx_path, artifact_path="onnx")
        mlflow.log_artifact(vec_path,  artifact_path="onnx")
        mlflow.log_param("onnx_opset_version", 17)
        logger.info("ONNX artifacts logged to MLflow under 'onnx/' directory")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch MLP to ONNX")
    parser.add_argument(
        "--pipeline-path",
        default=DEFAULT_PIPELINE_PATH,
        help="Path to the fitted pytorch_mlp pipeline .pkl file",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write onnx_model.onnx and tfidf_vectorizer.pkl",
    )
    parser.add_argument(
        "--mlflow-run-id",
        default=None,
        help="Optional MLflow run ID to attach artifacts to (Phase 1 run)",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip MLflow artifact logging (useful for quick local exports)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Orchestrate the full ONNX export workflow.

    Steps: load → export → validate → log to MLflow.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()

    pipeline  = load_pytorch_pipeline(args.pipeline_path)
    onnx_path = export_to_onnx(pipeline, args.output_dir)
    vec_path  = save_vectorizer(pipeline, args.output_dir)

    validate_onnx(onnx_path, pipeline)

    if not args.skip_mlflow:
        log_onnx_artifacts_to_mlflow(onnx_path, vec_path, args.mlflow_run_id)
    else:
        logger.info("MLflow logging skipped (--skip-mlflow flag set)")

    print(f"\n✅ ONNX export complete.")
    print(f"   Model:      {onnx_path}")
    print(f"   Vectorizer: {vec_path}")


if __name__ == "__main__":
    main()
