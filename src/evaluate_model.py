import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        logger.info(f"Model Accuracy: {acc:.4f}")
        return {
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report
        }
    except Exception as e:
        logger.exception("Evaluation failed.")
        raise
