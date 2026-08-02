from pathlib import Path
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json

from scripts.helpers.yaml_reading import load_params
from scripts.helpers.layers_helper import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Parameters file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "configs"))
YAML_FILE = PROJECT_ROOT / "configs"/ "mc_evaluation_params.yaml"

DATA_DIR = PROJECT_ROOT / "data" / "multi_model"
MODEL_PATH = PROJECT_ROOT / "models" / "multi_model.keras"
RESULTS_DIR = PROJECT_ROOT / "results" / "multiclass"
PAPER_DIR = PROJECT_ROOT / "paper" / "figures"/ "plots" / "multiclass"

def data_loading():

    test_data = np.load(DATA_DIR / "test.npz")

    X_test = test_data["X"][..., np.newaxis]
    y_test = test_data["y"]
    kepid_test = test_data["kepid"]

    return X_test, y_test, kepid_test

def plot_training_history(path):

    # Plotting and saving training history
    with open(path) as f:
        params = json.load(f)
        accuracy = params["accuracy"]
        val_accuracy = params["val_accuracy"]
        loss = params["loss"]
        val_loss = params["val_loss"]

    epochs = np.arange(1, len(accuracy) + 1)

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(PAPER_DIR / "mc_training_loss_history.png")

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(PAPER_DIR / "mc_training_accuracy_history.png")

def plot_confusion_matrix(all_true, all_preds):
    class_names = ["Transit", "Shallow EB", "Deep EB"]

    all_true = np.asarray(all_true).astype(int)
    all_preds = np.asarray(all_preds).astype(int)

    conf_counts = confusion_matrix(all_true, all_preds)
    conf_rates = confusion_matrix(all_true, all_preds, normalize="true")
    accuracy = np.mean(all_true == all_preds) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(conf_rates, cmap="Blues", vmin=0, vmax=1)

    for row in range(conf_rates.shape[0]):
        for col in range(conf_rates.shape[1]):
            rate = conf_rates[row, col] * 100
            count = conf_counts[row, col]
            text_colour = "white" if conf_rates[row, col] >= 0.55 else "#1f2933"

            ax.text(
                col,
                row,
                f"{rate:.1f}%\n(n={count})",
                ha="center",
                va="center",
                color=text_colour,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_title(f"Confusion Matrix - Accuracy: {accuracy:.2f}%", fontsize=14, pad=14)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)

    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of True Class", rotation=270, labelpad=18)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "mc_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def calculate_classification_metrics(
    y_true,
    y_pred,
    y_pred_prob,
) -> dict:
    """
    Calculate binary classification metrics.

    :param y_true: True binary labels.
    :param y_pred: Predicted binary labels after thresholding.
    :param y_pred_prob: Predicted probabilities for the transit/exoplanet class.
    :return: Dictionary of evaluation metrics.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred,)),
        "precision": float(precision_score(y_true, y_pred,  average="macro",zero_division=0)),
        "recall": float(recall_score(y_true, y_pred,  average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred,  average="macro", zero_division=0)),
        #"roc_auc": float(roc_auc_score(y_true, y_pred_prob)),
    }

def plot_confidence_histogram(
    y_pred_prob: np.ndarray
) -> None:
    """
    Plot histogram of model prediction confidence.

    Confidence is defined as the model's distance from uncertainty:
    max(p, 1 - p), where p is the predicted probability for the transit/exoplanet class.
    """

    confidence = np.max(y_pred_prob, axis=1)

    plt.figure(figsize=(8, 5))
    plt.hist(confidence, bins=30, edgecolor="black")

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Samples")
    plt.title("Model Prediction Confidence")
    plt.xlim(0.5, 1.0)
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "plots" / "confidence_histogram.png", dpi=200)
    plt.close()

def plot_correct_incorrect_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    y_pred_prob = np.asarray(y_pred_prob)

    confidence = np.max(y_pred_prob, axis=1)

    correct_confidence = confidence[y_pred == y_true]
    incorrect_confidence = confidence[y_pred != y_true]

    plt.figure(figsize=(8, 5))
    plt.hist(
        correct_confidence,
        bins=30,
        range=(1/3, 1.0),
        alpha=0.7,
        label="Correct",
        edgecolor="black",
    )
    plt.hist(
        incorrect_confidence,
        bins=30,
        range=(1/3, 1.0),
        alpha=0.7,
        label="Incorrect",
        edgecolor="black",
    )

    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Samples")
    plt.title("Prediction Confidence: Correct vs Incorrect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PAPER_DIR / "mc_confidence_correct_vs_incorrect.png", dpi=200)
    plt.close()


def get_misclassified_lightcurves(
    X_test,
    y_test,
    y_pred,
    y_pred_prob,
    kepid_test,
) -> dict:
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)

    confidence = np.max(y_pred_prob, axis=1)
    wrong_mask = y_test != y_pred

    return {
        "X": X_test[wrong_mask],
        "y_true": y_test[wrong_mask],
        "y_pred": y_pred[wrong_mask],
        "y_pred_prob": y_pred_prob[wrong_mask],
        "kepid": kepid_test[wrong_mask],
        "confidence": confidence[wrong_mask],
    }

def main():
    (RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "misclassified_results").mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test, y_test, kepid_test = data_loading()

    params = load_params(YAML_FILE)

    # Load model for testing
    model = keras.models.load_model(MODEL_PATH)

    plot_training_history(RESULTS_DIR / "mc_training_history.json")

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    plot_confusion_matrix(y_test, y_pred)

    plot_confidence_histogram(y_pred_prob)

    plot_correct_incorrect_confidence(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_prob=y_pred_prob,
        results_dir=RESULTS_DIR / "plots",
    )

    metrics = calculate_classification_metrics(y_test, y_pred, y_pred_prob)

    with open(RESULTS_DIR / "evaluation.json", "w") as file:
        json.dump(metrics, file, indent=2)

    misclassified = get_misclassified_lightcurves(
        X_test,
        y_test,
        y_pred,
        y_pred_prob,
        kepid_test,
    )
    np.savez_compressed(
        RESULTS_DIR / "misclassified_results" / "misclassified_lightcurves.npz",
        X=misclassified["X"],
        y_true=misclassified["y_true"],
        y_pred=misclassified["y_pred"],
        y_pred_prob=misclassified["y_pred_prob"],
        kepid=misclassified["kepid"],
        confidence=misclassified["confidence"],
    )

if __name__ == "__main__":
    main()
