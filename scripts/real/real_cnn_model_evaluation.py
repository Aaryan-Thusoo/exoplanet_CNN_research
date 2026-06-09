from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tensorflow import keras
import json

from scripts.helpers.yaml_reading import load_params
from scripts.helpers.layers_helper import *

# Parameters file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "configs"))
yaml_file = PROJECT_ROOT / "configs"/ "real_cnn_params.yaml"

DATA_DIR = PROJECT_ROOT / "data" / "real_model"
MODEL_PATH = PROJECT_ROOT / "models" / "real_model.keras"
RESULTS_DIR = PROJECT_ROOT / "results"


def data_loading():

    test_data = np.load(data_path / "test.npz")

    X_test = test_data["X"][..., np.newaxis]
    y_test = test_data["y"]

    return X_test, y_test


def plot_confusion_matrix(all_true, all_preds):
    names = ["normal", "transit", "eclipsing"]

    conf_matrix = confusion_matrix(all_true, all_preds, normalize='true')
    n_classes = len(names)

    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, cmap='Blues', vmin=0, vmax=1)

    for i in range(n_classes):
        for j in range(n_classes):
            num = conf_matrix[i, j] * 100
            text = f"{num:.2f}%"
            colour = 'white' if conf_matrix[i, j] > 0.6 else 'black'
            plt.text(j, i, text, ha="center", va="center", color=colour)

    accuracy = np.mean(np.array(all_true) == np.array(all_preds)) * 100

    plt.title(f"Accuracy: {accuracy:.4}")
    plt.xticks(range(n_classes), names, rotation=45, ha='right')
    plt.yticks(range(n_classes), names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar(label="Fraction")
    plt.tight_layout()
    plt.show()


def main():

    X_test, y_test = data_loading()

    model = keras.models.load_model("models/real_model.keras")

    y_pred = model.predict(X_train)

    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()

