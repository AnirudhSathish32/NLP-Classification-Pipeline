import os
import logging

from src.utils import setup_logging, ensure_dir
from src.data_loader import load_dataset
from src.pipeline import run_pipeline
from prepare_data_csv import prepare_imdb_csv  # import the CSV prep module

# Paths
DATA_DIR = "data/raw"
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
IMDB_DIR = os.path.join(DATA_DIR, "aclImdb_v1", "aclImdb")

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def main():
    setup_logging()
    ensure_dir("models")

    # Convert IMDB .txt to CSV if needed
    if not os.path.exists(CSV_PATH):
        logging.info("CSV not found. Converting IMDB txt files to CSV...")
        prepare_imdb_csv(imdb_dir=IMDB_DIR, output_csv=CSV_PATH)
    else:
        logging.info("Dataset CSV found, skipping conversion.")

    # Load CSV and run pipeline
    X, y = load_dataset(CSV_PATH, text_col=TEXT_COLUMN, label_col=LABEL_COLUMN)
    best_model = run_pipeline(X, y)

    print(f"\nSaved best model: {best_model}")


if __name__ == "__main__":
    main()
