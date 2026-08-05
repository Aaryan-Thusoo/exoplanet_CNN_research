from pathlib import Path
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.helpers.yaml_reading import load_params

DATA_DIR = PROJECT_ROOT / "data" / "multi_model"
MODEL_PATH = PROJECT_ROOT / "models" / "multi_model.keras"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / "multiclass" / "confidence_group_analysis"
YAML_FILE = PROJECT_ROOT / "configs" / "mc_evaluation_params.yaml"
PAPER_DIR = PROJECT_ROOT / "paper" / "figures" / "plots" / "multiclass"


def load_test_data():
    test_data = np.load(DATA_DIR / "test.npz")

    X_test = test_data["X"][..., np.newaxis]
    y_test = test_data["y"].astype(int)
    kepid_test = test_data["kepid"].astype(int)

    return X_test, y_test, kepid_test


def load_predictions(X_test):
    model = keras.models.load_model(MODEL_PATH)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    confidence = np.max(y_pred_prob, axis=1)

    return y_pred_prob, y_pred, confidence


def calculate_transit_depth(X: np.ndarray) -> np.ndarray:
    """
    Calculate transit depth using the median of the 10 lowest flux values
    in each light-curve chunk.
    """
    lightcurves = np.squeeze(X)

    if lightcurves.ndim == 1:
        lightcurves = lightcurves.reshape(1, -1)

    ten_lowest_values = np.sort(lightcurves, axis=1)[:, :10]

    return np.median(ten_lowest_values, axis=1)


def load_metadata():
    exo_df = pd.read_csv(RAW_DIR / "Kepler_Confirmed_ExoPlanets.csv")
    eb_df = pd.read_csv(RAW_DIR / "Kepler_Confirmed_EB.csv")

    metadata = pd.concat([exo_df, eb_df], ignore_index=True)
    metadata["kepid"] = metadata["kepid"].astype(int)

    return (
        metadata
        .sort_values("koi_model_snr", ascending=False, na_position="last")
        .drop_duplicates(subset="kepid", keep="first")
        .copy()
    )


def create_results_df(X_test, y_test, y_pred, y_pred_prob, confidence, kepid_test):
    lightcurve_std = np.std(X_test.squeeze(), axis=1)
    transit_depth = calculate_transit_depth(X_test)

    results_df = pd.DataFrame({
        "kepid": kepid_test,
        "y_true": y_test,
        "y_pred": y_pred,
        "confidence": confidence,
        "lightcurve_std": lightcurve_std,
        "transit_depth": transit_depth,
    })

    for class_index in range(y_pred_prob.shape[1]):
        results_df[f"y_pred_prob_{class_index}"] = y_pred_prob[:, class_index]

    results_df["correct"] = results_df["y_true"] == results_df["y_pred"]
    results_df["incorrect"] = ~results_df["correct"]

    return results_df


def add_metadata(results_df, metadata):
    results_df = results_df.merge(
        metadata,
        on="kepid",
        how="left",
    )

    results_df["lightcurve_std"] = results_df["lightcurve_std"].replace(0, np.nan)
    results_df["calculated_SNR"] = results_df["koi_depth"] / results_df["lightcurve_std"]
    results_df["transit_depth_abs"] = results_df["transit_depth"].abs()

    return results_df


def add_confidence_group(results_df):
    results_df["confidence_group"] = pd.cut(
        results_df["confidence"],
        bins=[0.0, 0.7, 0.85, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    return results_df


def add_value_group(results_df, column):
    results_df = results_df.dropna(subset=[column]).copy()

    results_df["value_group"] = pd.qcut(
        results_df[column],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    )

    return results_df


def create_summary(results_df, plot_name):
    summary = (
        results_df
        .groupby(["value_group", "confidence_group"], observed=False)
        .agg(
            total=("correct", "size"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )

    summary["incorrect"] = summary["total"] - summary["correct"]
    summary["correct_ratio"] = summary["correct"] / summary["total"]
    summary["incorrect_ratio"] = summary["incorrect"] / summary["total"]
    summary["plot_name"] = plot_name

    return summary


def clean_file_name(name):
    return name.lower().replace(" ", "_")


def clear_old_plots():
    for plot_file in RESULTS_DIR.glob("*.png"):
        plot_file.unlink()


def plot_ratio_axis(ax, summary, ratio_column, count_column, ylabel, title):
    colours = {
        "Low": "#4C78A8",
        "Medium": "#F58518",
        "High": "#E45756",
    }

    value_groups = ["Low", "Medium", "High"]
    confidence_groups = ["Low", "Medium", "High"]

    x = np.arange(len(value_groups))
    width = 0.25

    for i, confidence_group in enumerate(confidence_groups):
        group_data = summary[summary["confidence_group"] == confidence_group]

        ratios = []
        counts = []
        for value_group in value_groups:
            row = group_data[group_data["value_group"] == value_group]

            if len(row) == 0:
                ratios.append(0)
                counts.append(0)
            else:
                ratios.append(row[ratio_column].iloc[0])
                counts.append(int(row[count_column].iloc[0]))

        bars = ax.bar(
            x + (i - 1) * width,
            ratios,
            width=width,
            label=f"{confidence_group} Confidence",
            color=colours[confidence_group],
            edgecolor="black",
        )

        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x, value_groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_mistake_ratio(summary, plot_name):
    file_name = clean_file_name(plot_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_ratio_axis(
        ax,
        summary,
        ratio_column="incorrect_ratio",
        count_column="incorrect",
        ylabel="All Incorrect Ratio",
        title=f"Mistake Ratios by {plot_name} and Confidence",
    )
    ax.set_xlabel(f"{plot_name} Group")
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / f"mc_mistake_ratios_by_{file_name}_and_confidence.png",
        dpi=200,
        bbox_inches="tight",
    )

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        PAPER_DIR / f"mc_mistake_ratios_by_{file_name}_and_confidence_bottom.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clear_old_plots()

    X_test, y_test, kepid_test = load_test_data()
    y_pred_prob, y_pred, confidence = load_predictions(X_test)

    metadata = load_metadata()

    results_df = create_results_df(
        X_test,
        y_test,
        y_pred,
        y_pred_prob,
        confidence,
        kepid_test,
    )

    results_df = add_metadata(results_df, metadata)
    results_df = add_confidence_group(results_df)

    results_df.to_csv(RESULTS_DIR / "all_test_predictions_with_groups.csv", index=False)

    params = load_params(YAML_FILE)
    summaries = []

    for plot_info in params["plots_info"]:
        plot_name = plot_info["name"]
        column = plot_info["col"]

        grouped_df = add_value_group(results_df, column)

        summary = create_summary(grouped_df, plot_name)
        summaries.append(summary)

        plot_mistake_ratio(summary, plot_name)

    all_summaries = pd.concat(summaries, ignore_index=True)
    all_summaries.to_csv(RESULTS_DIR / "group_summary.csv", index=False)


if __name__ == "__main__":
    main()
