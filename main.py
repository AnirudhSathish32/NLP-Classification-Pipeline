from src.utils import setup_logging, ensure_dir
from src.data_loader import load_dataset
from src.pipeline import run_pipeline

DATA_PATH = "data/raw/dataset.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def main():
    setup_logging()

    ensure_dir("models")

    X, y = load_dataset(
        DATA_PATH,
        text_col=TEXT_COLUMN,
        label_col=LABEL_COLUMN
    )

    best_model = run_pipeline(X, y)

    print(f"\nSaved best model: {best_model}")


if __name__ == "__main__":
    main()
