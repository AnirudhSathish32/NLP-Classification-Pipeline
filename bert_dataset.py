"""
bert_dataset.py
---------------
Tokenisation and DataLoader construction for BERT fine-tuning on IMDB.

What it does:
    Provides a PyTorch Dataset that wraps raw text + label pairs and applies
    HuggingFace tokenisation on-the-fly.  Also provides a factory function
    that builds train, validation, and test DataLoaders from a CSV file,
    applying a stratified train/val split internally.

Parameters:
    See IMDBDataset.__init__() and build_dataloaders() below.

Returns:
    See build_dataloaders() — returns three DataLoader objects.

Connects to:
    bert_classifier.py  — calls build_dataloaders() to get train/val/test loaders
    bert_config.yaml    — max_length, batch_size, val_split, random_seed consumed here
    src/data_loader.py  — load_dataset() provides the raw X, y Series
"""

import logging
from typing import Dict, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)


class IMDBDataset(Dataset):
    """
    PyTorch Dataset for IMDB sentiment reviews.

    What it does:
        Stores raw text strings and integer labels.  On __getitem__ it
        tokenises the text using a pre-loaded BertTokenizerFast and returns
        a dict of tensors ready for BERT's forward pass.

        Tokenisation is done per-item (not pre-computed) to keep memory use
        low — the full IMDB dataset tokenised at max_length=256 would require
        several GB of RAM if pre-computed.

    Parameters:
        texts      (list[str]):  Raw review strings.
        labels     (list[int]):  Integer sentiment labels (0 = neg, 1 = pos).
        tokenizer  (BertTokenizerFast): Pre-loaded HuggingFace tokenizer.
        max_length (int):        Maximum token sequence length (default 256).

    Returns (per __getitem__):
        dict with keys:
            input_ids      (LongTensor):  Token IDs, shape (max_length,)
            attention_mask (LongTensor):  1 for real tokens, 0 for padding
            token_type_ids (LongTensor):  Segment IDs (all 0 for single sequence)
            labels         (LongTensor):  Scalar label

    Connects to:
        bert_classifier.py — DataLoader wraps this Dataset.
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: BertTokenizerFast,
        max_length: int = 256,
    ) -> None:
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",     # pad all sequences to max_length
            truncation=True,          # truncate sequences that exceed max_length
            return_tensors="pt",      # return PyTorch tensors
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_dataloaders(
    X,
    y,
    tokenizer: BertTokenizerFast,
    max_length: int = 256,
    batch_size: int = 16,
    val_split: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build stratified train, validation, and test DataLoaders from raw data.

    What it does:
        1. Performs a stratified 80/20 train+val / test split.
        2. Performs a further stratified split on the train+val portion to
           carve out the validation set (val_split fraction).
        3. Wraps each split in an IMDBDataset and then a DataLoader.

        Stratification is applied at every split to preserve the 50/50
        pos/neg class balance that IMDB provides.

    Parameters:
        X           (pd.Series):         Raw review text strings.
        y           (pd.Series):         Integer labels.
        tokenizer   (BertTokenizerFast): Pre-loaded HuggingFace tokenizer.
        max_length  (int):               Max token length passed to IMDBDataset.
        batch_size  (int):               Samples per DataLoader batch.
        val_split   (float):             Fraction of train+val used for validation.
        random_seed (int):               Random state for reproducible splits.
        num_workers (int):               DataLoader worker processes. 0 = main process.
                                         Increase on Linux for faster data loading.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            (train_loader, val_loader, test_loader)

    Connects to:
        bert_classifier.py — unpacks the returned tuple directly into training loop.
        bert_config.yaml   — all parameters originate from config values.
    """
    X_list = list(X)
    y_list = list(y)

    # --- split 1: train+val vs test (80/20) ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_list, y_list,
        test_size=0.2,
        stratify=y_list,
        random_state=random_seed,
    )

    # --- split 2: train vs val ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_split,
        stratify=y_trainval,
        random_state=random_seed,
    )

    logger.info(
        f"Dataset splits — train: {len(X_train)}, "
        f"val: {len(X_val)}, test: {len(X_test)}"
    )

    # --- wrap in Dataset objects ---
    train_ds = IMDBDataset(X_train, y_train, tokenizer, max_length)
    val_ds   = IMDBDataset(X_val,   y_val,   tokenizer, max_length)
    test_ds  = IMDBDataset(X_test,  y_test,  tokenizer, max_length)

    # --- wrap in DataLoaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,              # shuffle training data each epoch
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def load_tokenizer(model_name: str = "bert-base-uncased") -> BertTokenizerFast:
    """
    Load and return a BertTokenizerFast from HuggingFace.

    What it does:
        Downloads the tokenizer vocab and config on first call, then caches
        them in HuggingFace's local cache directory (~/.cache/huggingface/).
        Subsequent calls are instant.

    Parameters:
        model_name (str): HuggingFace model identifier. Must match the
                          model_name in bert_config.yaml.

    Returns:
        BertTokenizerFast: Ready-to-use tokenizer.

    Connects to:
        bert_classifier.py — calls this once at startup before building loaders.
    """
    logger.info(f"Loading tokenizer: '{model_name}'")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    logger.info("Tokenizer loaded.")
    return tokenizer
