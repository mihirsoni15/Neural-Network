import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.neural_network import MLPClassifier
import joblib
from src.logger import get_logger

logger = get_logger(__name__)

def train_and_save_model(X_train, y_train, model_path="models/admission_nn_model.pkl"):
    try:
        model = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        logger.info(f"Model trained and saved to {model_path}")
        return model
    except Exception as e:
        logger.exception("Model training failed.")
        raise
