from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pandas import DataFrame

from scipy.stats import kruskal, mannwhitneyu

from scripts.helpers.misclassified_helper import *
from scripts.helpers.yaml_reading import *

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="seaborn"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results" / "binary" / "misclassified_results"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REAL_MODEL_DIR = PROJECT_ROOT / "data" / "binary_model"
YAML_DIR = PROJECT_ROOT / "configs" / "bi_evaluation_params.yaml"
PAPER_DIR = Path("/Users/aaryanthusoo/Desktop/UCL/exoplanet_CNN_paper/figures/plots")

def load_data() -> tuple[DataFrame, DataFrame, Any, Any]:
    misclassified = np.load(RESULTS_DIR / "misclassified_lightcurves.npz")

    fp_df = pd.DataFrame({
        "kepid": misclassified["false_positive_kepid"],
        "y_true": misclassified["false_positive_y_true"],
        "y_pred": misclassified["false_positive_y_pred"],
        "y_pred_prob": misclassified["false_positive_y_pred_prob"],
        "confidence": misclassified["false_positive_confidence"],
        "chunk_depth": misclassified["false_positive_chunk_depth"],
    })

    fn_df = pd.DataFrame({
        "kepid": misclassified["false_negative_kepid"],
        "y_true": misclassified["false_negative_y_true"],
        "y_pred": misclassified["false_negative_y_pred"],
        "y_pred_prob": misclassified["false_negative_y_pred_prob"],
        "confidence": misclassified["false_negative_confidence"],
        "chunk_depth": misclassified["false_negative_chunk_depth"],
    })

    fp_X = misclassified["false_positive_X"]
    fn_X = misclassified["false_negative_X"]

    return fp_df, fn_df, fp_X, fn_X


def read_exo_eb_csv() -> tuple[DataFrame, DataFrame]:
    exo_df = pd.read_csv(RAW_DIR / "Kepler_Confirmed_ExoPlanets.csv")
    eb_df = pd.read_csv(RAW_DIR / "Kepler_Confirmed_EB.csv")
    return exo_df, eb_df


def calculate_transit_depth(X: np.ndarray) -> np.ndarray:
    """
    Calculate transit depth using the median of the 10 lowest flux values
    in each light-curve chunk.

    Each returned value corresponds to one model input instance.
    """
    lightcurves = np.squeeze(X)

    if lightcurves.ndim == 1:
        lightcurves = lightcurves.reshape(1, -1)

    ten_lowest_values = np.sort(lightcurves, axis=1)[:, :10]

    return np.median(ten_lowest_values, axis=1)


def plot_low_med_high_conf_hist(low_data, med_data, high_data, bins, value_name):

    plt.figure()
    plt.title("Low vs. High Confidence Histogram")

    plt.hist(low_data, bins=bins, label=f"Low Confidence", alpha=1, edgecolor="black", histtype="stepfilled")
    plt.hist(med_data, bins=bins, label=f"Medium Confidence", alpha=0.75, edgecolor="black", histtype="stepfilled")
    plt.hist(high_data, bins=bins, label=f"High Confidence", alpha=0.5, edgecolor="black", histtype="stepfilled")

    plt.xlabel(f"{value_name}")
    plt.ylabel("Density")
    plt.legend()

    stat, p_value = kruskal(low_data, med_data, high_data)
    print(stat, p_value)

    pairs = {
    "low_vs_med": mannwhitneyu(low_data, med_data, alternative="two-sided"),
    "low_vs_high": mannwhitneyu(low_data, high_data, alternative="two-sided"),
    "med_vs_high": mannwhitneyu(med_data, high_data, alternative="two-sided"),
    }

    for name, result in pairs.items():
        print(name)
        print("U statistic:", result.statistic)
        print("raw p-value:", result.pvalue)
        print("Bonferroni p-value:", min(result.pvalue * 3, 1.0))
        print()


def plot_misclassified_eb_depths(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot chunk-depth distribution for false positives.

    In the binary model, false positives are true eclipsing binaries
    predicted as transits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_depth = df["chunk_depth"].dropna()

    def save_depth_histogram(depth_data: pd.Series, output_file: Path, title: str) -> None:
        plt.figure(figsize=(9, 6))
        plt.hist(
            depth_data,
            bins=80,
            edgecolor="black",
            color="#4c78a8",
            alpha=0.85,
        )

        plt.xlabel("Chunk Depth")
        plt.ylabel("Number of Misclassified EB Chunks")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_file, dpi=200)
        plt.close()

    save_depth_histogram(
        chunk_depth,
        output_dir / "misclassified_eb_chunk_depth_histogram_full.png",
        "Depth Distribution of Misclassified Eclipsing Binaries",
    )
    save_depth_histogram(
        chunk_depth[chunk_depth <= 0.1],
        output_dir / "misclassified_eb_chunk_depth_histogram_zoomed.png",
        "Depth Distribution of Misclassified Eclipsing Binaries (Depth <= 0.1)",
    )

def main():

    params = load_params(YAML_DIR)["plots_info"]

    fp_df, fn_df, fp_X, fn_X = load_data()

    fp_std = np.std(fp_X.squeeze(), axis=1)
    fn_std = np.std(fn_X.squeeze(), axis=1)

    fp_transit_depth = calculate_transit_depth(fp_X)
    fn_transit_depth = calculate_transit_depth(fn_X)

    exo_df, eb_df = read_exo_eb_csv()

    fp_df["kepid"] = fp_df["kepid"].astype(int)
    exo_df["kepid"] = exo_df["kepid"].astype(int)

    fn_df["kepid"] = fn_df["kepid"].astype(int)
    eb_df["kepid"] = eb_df["kepid"].astype(int)

    exo_df_unique = (
        exo_df
        .sort_values("koi_model_snr", ascending=False, na_position="last")
        .drop_duplicates(subset="kepid", keep="first")
        .copy()
    )

    eb_df_unique = (
        eb_df
        .sort_values("koi_model_snr", ascending=False, na_position="last")
        .drop_duplicates(subset="kepid", keep="first")
        .copy()
    )

    # False positives are true eclipsing binaries predicted as transits, so join EB metadata.
    fp_with_eb = fp_df.merge(
        eb_df_unique,
        on="kepid",
        how="left"
    )

    # False negatives are true transits predicted as eclipsing binaries, so join exoplanet metadata.
    fn_with_exo = fn_df.merge(
        exo_df_unique,
        on="kepid",
        how="left"
    )

    fp_with_eb["std"] = fp_std
    fn_with_exo["std"] = fn_std

    fp_with_eb["transit_depth"] = fp_transit_depth
    fn_with_exo["transit_depth"] = fn_transit_depth

    low_conf = 0.65
    med_conf = 0.8

    box_results_dir = RESULTS_DIR / "misclassified_plots" / "boxplots"
    hist_results_dir = RESULTS_DIR / "misclassified_plots" / "histograms"
    ecdf_results_dir = RESULTS_DIR / "misclassified_plots" / "ecdfs"

    box_results_dir.mkdir(parents=True, exist_ok=True)
    hist_results_dir.mkdir(parents=True, exist_ok=True)
    ecdf_results_dir.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    plot_misclassified_eb_depths(fp_with_eb, hist_results_dir)
    plot_misclassified_eb_depths(fp_with_eb, PAPER_DIR)

    for df, case in zip([fp_with_eb, fn_with_exo], ["fp", "fn"]):
        df["std"] = df["std"].replace(0, np.nan)
        df["calculated_SNR"] = df["koi_depth"] / df["std"]

        df_clean = df.dropna().copy()
        df_clean["transit_depth_abs"] = df_clean["transit_depth"].abs()

        df_clean["confidence_group"] = np.select(
            [
                df_clean["confidence"] <= low_conf,
                (df_clean["confidence"] > low_conf) & (df_clean["confidence"] <= med_conf),
                df_clean["confidence"] > med_conf,
            ],
            ["Low", "Medium", "High"],
            default="unknown",
        )

        for par in params:
            if case == "fp":
                box_jitter(df_clean, par["col"], f"Binary Model False Positive by {par["name"]}", PAPER_DIR, case, par["log"])
                ecdf_plotting(df_clean, par["col"], par["name"], ecdf_results_dir, case, par["log"])
                ecdf_plotting(df_clean, par["col"], par["name"], PAPER_DIR, case, par["log"])
                write_stats(df_clean, par["col"], RESULTS_DIR / f"{case}_distribution_stats.json")
            else:
                box_jitter(df_clean, par["col"], f"Binary Model False Negative by {par["name"]}", PAPER_DIR, case, par["log"])
                ecdf_plotting(df_clean, par["col"], par["name"], ecdf_results_dir, case, par["log"])
                ecdf_plotting(df_clean, par["col"], par["name"], PAPER_DIR, case, par["log"])
                write_stats(df_clean, par["col"], RESULTS_DIR / f"{case}_distribution_stats.json")


if __name__ == "__main__":
    main()
        
