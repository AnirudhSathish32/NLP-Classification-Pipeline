import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def evaluate_model(model, X_test, y_test, model_name="model"):
    """
    Evaluate model using multiple metrics.

    ROC-AUC is critical when classes are imbalanced because
    it measures ranking quality independent of threshold.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        # fallback for LinearSVC
        y_scores = model.decision_function(X_test)

    print(f"\n===== {model_name} =====")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_scores)
    print("ROC-AUC:", round(roc_auc, 4))

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend()
    plt.show()

    return roc_auc
