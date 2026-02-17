import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

from .text_preprocessing import TextCleaner
from .feature_engineering import get_tfidf_vectorizer
from .model_training import get_models, cross_validate_pipeline
from .evaluation import evaluate_model


def run_pipeline(X, y):
    """
    End-to-end pipeline orchestration.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    models = get_models()

    results = {}
    trained_pipelines = {}

    for name, model in models.items():
        logging.info(f"Training {name}...")

        pipe = Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", get_tfidf_vectorizer()),
            ("model", model)
        ])

        mean_auc, std_auc = cross_validate_pipeline(pipe, X_train, y_train)

        pipe.fit(X_train, y_train)

        test_auc = evaluate_model(pipe, X_test, y_test, name)

        results[name] = {
            "cv_mean_auc": mean_auc,
            "cv_std_auc": std_auc,
            "test_auc": test_auc
        }

        trained_pipelines[name] = pipe

    # select best
    best_model_name = max(results, key=lambda k: results[k]["test_auc"])
    best_pipeline = trained_pipelines[best_model_name]

    logging.info(f"Best model: {best_model_name}")

    joblib.dump(best_pipeline, "models/best_model.pkl")

    print("\n=== Model Comparison ===")
    for name, vals in results.items():
        print(
            f"{name}: CV={vals['cv_mean_auc']:.4f} ± {vals['cv_std_auc']:.4f} "
            f"| Test AUC={vals['test_auc']:.4f}"
        )

    return best_model_name
