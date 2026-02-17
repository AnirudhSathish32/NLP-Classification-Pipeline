import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def get_models():
    """
    Returns candidate models.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            early_stopping=True,
            max_iter=50,
            random_state=42
        ),
    }


def cross_validate_pipeline(pipeline, X, y):
    """
    Stratified 5-fold CV using ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )

    logging.info(
        f"CV ROC-AUC: mean={scores.mean():.4f}, std={scores.std():.4f}"
    )

    return scores.mean(), scores.std()
