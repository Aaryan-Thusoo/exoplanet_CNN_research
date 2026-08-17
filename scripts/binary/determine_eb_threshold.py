from pathlib import Path
import sys
import numpy as np
import pandas as pd
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
YAML_FILE = PROJECT_ROOT / "configs"/ "bi_evaluation_params.yaml"

DATA_DIR = PROJECT_ROOT / "data" / "binary_model"
MODEL_PATH = PROJECT_ROOT / "models" / "binary_model.keras"
RESULTS_DIR = PROJECT_ROOT / "results" / "binary"
PAPER_DIR = Path("/Users/aaryanthusoo/Desktop/UCL/exoplanet_CNN_paper/figures/plots")

def load_binary_validation():
    data = np.load(DATA_DIR / "val.npz")

    X, y = data["X"], data["y"]
    chunk_depth = data["chunk_depth"]
    eb_mask = y == 0

    eb_X = X[eb_mask]
    eb_depth = chunk_depth[eb_mask]
    eb_kepid = data["kepid"][eb_mask]
    return eb_kepid, eb_X, eb_depth

def load_binary_model():
    return keras.models.load_model(MODEL_PATH)


def main():
    return -1


def prepare_binning(df):
    depth_bins = load_params(YAML_FILE)["eb_threshold_checks"]

    depth_labels = [f"<{depth_bins[0]}"]

    for i in range(len(depth_bins) - 1):
        depth_labels.append(f"{depth_bins[i]}-{depth_bins[i + 1]}")

    depth_labels.append(f">={depth_bins[-1]}")

    bin_edges = [0] + depth_bins + [np.inf]

    df["depth_bin"] = pd.cut(
        df["eclipse_depth"],
        bins=bin_edges,
        labels=depth_labels,
        include_lowest=True,
        right=False,
    )

def plot_depth_bin_confusion(summary):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        summary["depth_bin"].astype(str),
        summary["confusion_rate"],
        edgecolor="black",
    )

    for bar, count, total in zip(bars, summary["confused_count"], summary["total_ebs"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{str(int(count))}/{str(int(total))}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Estimated Eclipse Depth Bin")
    plt.ylabel("EB Confusion Rate")
    plt.title("Validation EB Confusion Rate by Eclipse Depth")
    plt.ylim(0, 1)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "validation_eb_confusion_by_depth_bin.png", dpi=200)
    plt.savefig(PAPER_DIR / "bi_validation_eb_confusion_by_depth_bin.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    """main()"""

    eb_kepid, eb_X, eb_depth = load_binary_validation()
    model = load_binary_model()

    probs = model.predict(eb_X[..., np.newaxis]).ravel()
    pred = (probs >= 0.5).astype(int)

    df = pd.DataFrame({
        "kepid": eb_kepid,
        "eclipse_depth": eb_depth,
        "pred": pred,
        "prob": probs,
    })

    prepare_binning(df)

    df.to_csv(RESULTS_DIR / "validation_eb_chunk_results.csv", index=False)

    summary = (
        df.groupby("depth_bin", observed=False)
        .agg(
            total_ebs=("pred", "size"),
            confused_count=("pred", "sum"),
            median_prob=("prob", "median"),
            median_depth=("eclipse_depth", "median"),
        )
        .reset_index()
    )

    summary["confusion_rate"] = summary["confused_count"] / summary["total_ebs"]

    summary.to_csv(RESULTS_DIR / "validation_eb_depth_bin_summary.csv", index=False)
    print(summary)

    plot_depth_bin_confusion(summary)
