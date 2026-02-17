import pandas as pd
import logging


def load_dataset(path: str, text_col: str, label_col: str):
    """
    Load dataset from CSV and return text + labels.
    """
    logging.info("Loading dataset...")
    df = pd.read_csv(path)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError("Specified columns not found.")

    df = df[[text_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(int)

    logging.info(f"Dataset loaded: {len(df)} rows")
    return df[text_col], df[label_col]
