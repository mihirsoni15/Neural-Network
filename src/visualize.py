import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import get_logger

logger = get_logger(__name__)

def plot_confusion_matrix(cm):
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
        logger.info("Confusion matrix plotted.")
    except Exception as e:
        logger.exception("Failed to plot confusion matrix.")
