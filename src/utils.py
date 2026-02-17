import logging
import os


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
