"""
bert_classifier.py
------------------
BERT fine-tuning pipeline for IMDB sentiment classification.

Changes from Phase 3:
    After training completes, the best test ROC-AUC is written into
    models/model_results.json so app.py can display it alongside the
    sklearn and PyTorch metrics in the compare endpoint response.

Connects to:
    bert_config.yaml        — all hyperparameters
    bert_dataset.py         — IMDBDataset, build_dataloaders, load_tokenizer
    mlflow_config.py        — configure_mlflow()
    src/data_loader.py      — load_dataset()
    models/model_results.json — appends bert entry after training
    app.py                  — loads bert checkpoint at startup
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from bert_dataset import build_dataloaders, load_tokenizer
from mlflow_config import configure_mlflow
from src.data_loader import load_dataset

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.environ.get("BERT_CONFIG_PATH", "bert_config.yaml")
DEFAULT_DATA_CSV    = os.environ.get(
    "BERT_DATA_CSV", os.path.join("data", "raw", "dataset.csv")
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_bert_config(config_path: str) -> Dict[str, Any]:
    """
    Load bert_config.yaml and apply environment variable overrides.

    Parameters:
        config_path (str): Path to bert_config.yaml.

    Returns:
        dict: Fully resolved configuration dictionary.

    Connects to:
        bert_config.yaml — primary source of all hyperparameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"BERT config not found at '{config_path}'.")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_overrides = {
        "BERT_LEARNING_RATE":   ("learning_rate",   float),
        "BERT_BATCH_SIZE":      ("batch_size",       int),
        "BERT_MAX_LENGTH":      ("max_length",       int),
        "BERT_NUM_EPOCHS":      ("num_epochs",       int),
        "BERT_WARMUP_STEPS":    ("warmup_steps",     int),
        "BERT_WEIGHT_DECAY":    ("weight_decay",     float),
        "BERT_MAX_GRAD_NORM":   ("max_grad_norm",    float),
        "BERT_VAL_SPLIT":       ("val_split",        float),
        "BERT_RANDOM_SEED":     ("random_seed",      int),
        "BERT_CHECKPOINT_DIR":  ("checkpoint_dir",   str),
        "BERT_ONNX_OUTPUT_DIR": ("onnx_output_dir",  str),
    }

    for env_key, (config_key, cast) in env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            config[config_key] = cast(val)
            logger.info(f"Config override: {config_key}={config[config_key]}")

    return config


# ---------------------------------------------------------------------------
# Model + device
# ---------------------------------------------------------------------------

def build_bert_model(model_name: str, num_labels: int = 2) -> BertForSequenceClassification:
    """
    Load pre-trained BERT with a classification head.

    Parameters:
        model_name  (str): HuggingFace model identifier.
        num_labels  (int): Number of output classes.

    Returns:
        BertForSequenceClassification
    """
    logger.info(f"Loading pre-trained model: '{model_name}'")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def get_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_epoch(
    model: BertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "val",
) -> Dict[str, float]:
    """
    Full evaluation pass over a DataLoader.

    Parameters:
        model       (BertForSequenceClassification)
        loader      (DataLoader)
        device      (torch.device)
        split_name  (str): Prefix for metric keys.

    Returns:
        dict: ROC-AUC, accuracy, F1, precision, recall.
    """
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            logits = outputs.logits.cpu()
            probs  = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds  = torch.argmax(logits, dim=1).numpy()

            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)

    metrics = {
        f"{split_name}_roc_auc":   roc_auc_score(all_labels, all_probs),
        f"{split_name}_accuracy":  accuracy_score(all_labels, all_preds),
        f"{split_name}_f1":        f1_score(all_labels, all_preds, average="macro"),
        f"{split_name}_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        f"{split_name}_recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    logger.info(
        f"{split_name.upper()} — "
        f"ROC-AUC: {metrics[f'{split_name}_roc_auc']:.4f} | "
        f"Accuracy: {metrics[f'{split_name}_accuracy']:.4f} | "
        f"F1: {metrics[f'{split_name}_f1']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: BertForSequenceClassification,
    checkpoint_dir: str,
    epoch: int,
    metric_value: float,
) -> str:
    """
    Save the current best checkpoint using HuggingFace save_pretrained().

    Parameters:
        model          (BertForSequenceClassification)
        checkpoint_dir (str)
        epoch          (int)
        metric_value   (float)

    Returns:
        str: Path to the saved checkpoint directory.

    Connects to:
        app.py — loads from this exact directory at startup.
    """
    best_dir = os.path.join(checkpoint_dir, "best_checkpoint")
    os.makedirs(best_dir, exist_ok=True)
    model.save_pretrained(best_dir)
    
    with open(os.path.join(checkpoint_dir, "best_checkpoint_meta.txt"), "w") as f:
        f.write(f"epoch={epoch}\nval_roc_auc={metric_value:.6f}\n")

    logger.info(f"Checkpoint saved: epoch={epoch}, val_roc_auc={metric_value:.4f} → '{best_dir}'")
    return best_dir


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def fine_tune(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[str, Dict[str, float]]:
    """
    Fine-tune BERT and checkpoint the best epoch by val ROC-AUC.

    Parameters:
        model        (BertForSequenceClassification)
        train_loader (DataLoader)
        val_loader   (DataLoader)
        config       (dict): From load_bert_config().
        device       (torch.device)

    Returns:
        Tuple[str, dict]: (best_checkpoint_dir, best_val_metrics)

    Connects to:
        run_bert_pipeline() — called inside the MLflow parent run.
    """
    model.to(device)

    num_epochs     = config["num_epochs"]
    warmup_steps   = config["warmup_steps"]
    max_grad_norm  = config["max_grad_norm"]
    checkpoint_dir = config["checkpoint_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=config["learning_rate"])

    total_steps = len(train_loader) * num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_auc        = -1.0
    best_checkpoint_dir = ""
    best_val_metrics: Dict[str, float] = {}

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        model.train()

        epoch_loss  = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} avg train loss: {avg_loss:.4f}")

        val_metrics = evaluate_epoch(model, val_loader, device, split_name="val")

        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v, step=epoch)

        val_auc = val_metrics["val_roc_auc"]
        if val_auc > best_val_auc:
            best_val_auc        = val_auc
            best_val_metrics    = val_metrics
            best_checkpoint_dir = save_checkpoint(model, checkpoint_dir, epoch, val_auc)

    logger.info(f"Training complete. Best val ROC-AUC: {best_val_auc:.4f}")
    return best_checkpoint_dir, best_val_metrics


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_bert_to_onnx(
    checkpoint_dir: str,
    config: Dict[str, Any],
    tokenizer: BertTokenizerFast,
) -> str:
    """
    Export best BERT checkpoint to ONNX.

    Parameters:
        checkpoint_dir (str)
        config         (dict)
        tokenizer      (BertTokenizerFast)

    Returns:
        str: Path to exported .onnx file.
    """
    onnx_output_dir = config["onnx_output_dir"]
    opset_version   = config.get("onnx_opset_version", 17)
    max_length      = config["max_length"]

    os.makedirs(onnx_output_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_output_dir, "bert_sentiment.onnx")

    model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    encoding = tokenizer(
        "This film was surprisingly good",
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    torch.onnx.export(
        model,
        (encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"]),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "token_type_ids": {0: "batch_size"},
            "logits":         {0: "batch_size"},
        },
        opset_version=opset_version,
        export_params=True,
    )

    logger.info(f"BERT ONNX exported to '{onnx_path}'")
    return onnx_path


# ---------------------------------------------------------------------------
# model_results.json update
# ---------------------------------------------------------------------------

def update_model_results(test_auc: float, results_path: str = "models/model_results.json") -> None:
    """
    Append BERT's test ROC-AUC into model_results.json.

    What it does:
        Reads the existing model_results.json written by src/pipeline.py
        and adds a 'bert' entry so app.py can display BERT's metrics
        alongside the sklearn and PyTorch results in the compare endpoint.

        Safe to call even if model_results.json does not yet exist —
        it will create a minimal structure in that case.

    Parameters:
        test_auc     (float): BERT's test ROC-AUC from evaluate_epoch().
        results_path (str):   Path to model_results.json.

    Returns:
        None

    Connects to:
        app.py — reads model_results.json at startup to populate
                 the all_model_metrics field in /predict responses.
        src/pipeline.py — originally creates model_results.json.
    """
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
    else:
        logger.warning(
            f"'{results_path}' not found — creating a new one. "
            "Run main.py before bert_classifier.py for complete results."
        )
        data = {"best_model": "unknown", "metrics": {}}

    data["metrics"]["bert"] = {
        "cv_mean_auc": None,
        "cv_std_auc":  None,
        "test_auc":    float(test_auc),
    }

    with open(results_path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"BERT test_auc={test_auc:.4f} written to '{results_path}'")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_bert_pipeline(
    config_path: str = DEFAULT_CONFIG_PATH,
    data_csv: str    = DEFAULT_DATA_CSV,
    skip_onnx: bool  = False,
) -> None:
    """
    Full BERT fine-tuning pipeline: load → tokenise → train → evaluate → export.

    Parameters:
        config_path (str): Path to bert_config.yaml.
        data_csv    (str): Path to IMDB CSV.
        skip_onnx   (bool): Skip ONNX export if True.

    Connects to:
        mlflow_config.py       — configure_mlflow()
        bert_dataset.py        — build_dataloaders()
        fine_tune()            — training loop
        evaluate_epoch()       — final test evaluation
        update_model_results() — writes BERT AUC to model_results.json
        export_bert_to_onnx()  — ONNX export
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = load_bert_config(config_path)
    configure_mlflow()

    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    logger.info(f"Loading dataset from '{data_csv}'")
    X, y = load_dataset(data_csv, text_col="text", label_col="label")

    tokenizer = load_tokenizer(config["model_name"])

    train_loader, val_loader, test_loader = build_dataloaders(
        X, y,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        batch_size=config["batch_size"],
        val_split=config["val_split"],
        random_seed=config["random_seed"],
    )

    device = get_device()
    model  = build_bert_model(config["model_name"])

    with mlflow.start_run(run_name="bert_fine_tuning") as run:
        mlflow.log_params({
            "model_name":    config["model_name"],
            "max_length":    config["max_length"],
            "num_epochs":    config["num_epochs"],
            "batch_size":    config["batch_size"],
            "learning_rate": config["learning_rate"],
            "weight_decay":  config["weight_decay"],
            "warmup_steps":  config["warmup_steps"],
            "max_grad_norm": config["max_grad_norm"],
            "val_split":     config["val_split"],
            "random_seed":   config["random_seed"],
            "device":        str(device),
        })

        best_checkpoint_dir, best_val_metrics = fine_tune(
            model, train_loader, val_loader, config, device
        )

        logger.info("Loading best checkpoint for final test evaluation...")
        best_model = BertForSequenceClassification.from_pretrained(best_checkpoint_dir)
        best_model.to(device)

        test_metrics = evaluate_epoch(best_model, test_loader, device, split_name="test")

        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics({f"best_{k}": v for k, v in best_val_metrics.items()})
        mlflow.log_artifacts(best_checkpoint_dir, artifact_path="bert_checkpoint")

        # Write BERT's AUC into model_results.json for app.py
        update_model_results(test_metrics["test_roc_auc"])

        if config.get("onnx_export", True) and not skip_onnx:
            onnx_path = export_bert_to_onnx(best_checkpoint_dir, config, tokenizer)
            mlflow.log_artifact(onnx_path, artifact_path="bert_onnx")
        else:
            logger.info("ONNX export skipped.")

        logger.info(f"MLflow run complete: {run.info.run_id}")

    print("\n✅ BERT fine-tuning complete.")
    print(f"   Best checkpoint : {best_checkpoint_dir}")
    print(f"   Test ROC-AUC    : {test_metrics['test_roc_auc']:.4f}")
    print(f"   Test Accuracy   : {test_metrics['test_accuracy']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on IMDB sentiment")
    parser.add_argument("--config",    default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-csv",  default=DEFAULT_DATA_CSV)
    parser.add_argument("--skip-onnx", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bert_pipeline(
        config_path=args.config,
        data_csv=args.data_csv,
        skip_onnx=args.skip_onnx,
    )
