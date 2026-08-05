from pathlib import Path
import sys
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.helpers.yaml_reading import *

DATA_DIR = PROJECT_ROOT / "data" / "multi_model"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "multiclass" / "random_forest"
PARAMS_DIR = PROJECT_ROOT / "configs" / "mc_randf_params.yaml"
PAPER_DIR = PROJECT_ROOT / "paper" / "figures" / "plots" / "multiclass"
RF_GREEN = "#2e7d32"


def find_chunk_depth(X: np.ndarray) -> np.ndarray:
    """
    Estimate each chunk depth using the median of the ten lowest flux points.
    :param X: Light curve input array.
    :return: Chunk depth estimate for each row.
    """
    ten_lowest = np.sort(X, axis=1)[:, :10]
    return 1 - np.median(ten_lowest, axis=1)


def make_features(data):
    """
    Create simple summary features from each light curve chunk.
    :param data: Loaded .npz data file.
    :return: Feature dataframe and labels.
    """
    X = data["X"]
    y = data["y"].astype(int)

    features = pd.DataFrame({
        "mean_flux": np.mean(X, axis=1),
        "std_flux": np.std(X, axis=1),
        "min_flux": np.min(X, axis=1),
        "max_flux": np.max(X, axis=1),
        "chunk_depth": find_chunk_depth(X),
    })

    return features, y


def set_up_model(n_est, seed, class_weight):
    return RandomForestClassifier(n_estimators=n_est, random_state=seed, class_weight=class_weight)


def plot_confusion_matrix(all_true, all_preds):
    class_names = ["Transit", "Shallow EB", "Deep EB"]

    all_true = np.asarray(all_true).astype(int)
    all_preds = np.asarray(all_preds).astype(int)

    conf_counts = confusion_matrix(all_true, all_preds)
    conf_rates = confusion_matrix(all_true, all_preds, normalize="true")
    accuracy = np.mean(all_true == all_preds) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(conf_rates, cmap="Greens", vmin=0, vmax=1)

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
                fontsize=11,
                fontweight="bold",
            )

    ax.set_title(f"Random Forest Confusion Matrix - Accuracy: {accuracy:.2f}%", fontsize=14, pad=14)
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
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    fig.savefig(PAPER_DIR / "mc_rf_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, importances, color=RF_GREEN, edgecolor="black")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=200)
    plt.savefig(PAPER_DIR / "mc_feature_importance.png", dpi=200)
    plt.close()


def main():
    params = load_params(PARAMS_DIR)

    model = set_up_model(
        params["n_estimators"],
        params["seed"],
        params["class_weight"],
    )

    train_data = np.load(DATA_DIR / "train.npz")
    val_data = np.load(DATA_DIR / "val.npz")
    test_data = np.load(DATA_DIR / "test.npz")

    train_features, y_train = make_features(train_data)
    val_features, y_val = make_features(val_data)
    test_features, y_test = make_features(test_data)

    X_train_full = pd.concat([train_features, val_features], ignore_index=True)
    y_train_full = np.concatenate([y_train, y_val])
    feature_medians = X_train_full.median(numeric_only=True)

    X_train_full = X_train_full.fillna(feature_medians)
    test_features = test_features.fillna(feature_medians)

    model.fit(X_train_full, y_train_full)

    y_pred = model.predict(test_features)
    y_pred_prob = model.predict_proba(test_features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": float(roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="macro")),
    }

    with open(RESULTS_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    with open(MODEL_DIR / "multiclass_random_forest.pkl", "wb") as file:
        pickle.dump(model, file)

    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, train_features.columns)


if __name__ == "__main__":
    main()
