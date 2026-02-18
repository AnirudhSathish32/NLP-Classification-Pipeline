import os
import logging
import pandas as pd

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def prepare_imdb_csv(imdb_dir="data/raw/aclImdb_v1/aclImdb", output_csv="data/raw/dataset.csv"):
    """
    Converts IMDB .txt files (train/test, pos/neg) into a single CSV.
    """
    def load_txt_folder(folder_path, label):
        rows = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), encoding="utf-8") as f:
                    text = f.read().strip()
                    rows.append([text, label])
        return rows

    all_rows = []
    for split in ["train", "test"]:
        for sentiment in ["pos", "neg"]:
            path = os.path.join(imdb_dir, split, sentiment)
            label = 1 if sentiment == "pos" else 0
            logging.info(f"Processing {split}/{sentiment}...")
            rows = load_txt_folder(path, label)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[TEXT_COLUMN, LABEL_COLUMN])
    logging.info(f"Total rows collected: {len(df)}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"CSV saved to {output_csv}")
