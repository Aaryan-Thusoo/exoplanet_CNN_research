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

RESULTS_DIR = PROJECT_ROOT / 'results' / "misclassified_results"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REAL_MODEL_DIR = PROJECT_ROOT / "data" / "real_model"
YAML_DIR = PROJECT_ROOT / "configs" / "evaluation_params.yaml"

def load_data() -> tuple[DataFrame, DataFrame, Any, Any]:
    misclassified = np.load(RESULTS_DIR / "misclassified_lightcurves.npz")

    fp_df = pd.DataFrame({
        "kepid": misclassified["false_positive_kepid"],
        "y_true": misclassified["false_positive_y_true"],
        "y_pred": misclassified["false_positive_y_pred"],
        "y_pred_prob": misclassified["false_positive_y_pred_prob"],
        "confidence": misclassified["false_positive_confidence"],
    })

    fn_df = pd.DataFrame({
        "kepid": misclassified["false_negative_kepid"],
        "y_true": misclassified["false_negative_y_true"],
        "y_pred": misclassified["false_negative_y_pred"],
        "y_pred_prob": misclassified["false_negative_y_pred_prob"],
        "confidence": misclassified["false_negative_confidence"],
    })

    fp_X = misclassified["false_positive_X"]
    fn_X = misclassified["false_negative_X"]

    return fp_df, fn_df, fp_X, fn_X


def read_exo_eb_csv() -> tuple[DataFrame, DataFrame]:
    exo_df = pd.read_csv(RAW_DIR / "Kepler_Confirmed_ExoPlanets.csv")
    eb_df = pd.read_csv(RAW_DIR / "Kepler_Confirmed_EB.csv")
    return exo_df, eb_df


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

def main():

    params = load_params(YAML_DIR)["plots_info"]

    fp_df, fn_df, fp_X, fn_X = load_data()

    fp_std = np.std(fp_X.squeeze(), axis=1)
    fn_std = np.std(fn_X.squeeze(), axis=1)

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

    low_conf = 0.65
    med_conf = 0.8

    box_results_dir = RESULTS_DIR / "misclassified_plots" / "boxplots"
    hist_results_dir = RESULTS_DIR / "misclassified_plots" / "histograms"

    box_results_dir.mkdir(parents=True, exist_ok=True)
    hist_results_dir.mkdir(parents=True, exist_ok=True)

    for df, case in zip([fp_with_eb, fn_with_exo], ["fp", "fn"]):
        df["std"] = df["std"].replace(0, np.nan)
        df["calculated_SNR"] = df["koi_depth"] / df["std"]

        df_clean = df.dropna().copy()

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
            box_jitter(df_clean, par["col"], par["name"], box_results_dir, case, par["log"])
            hist_plotting(df_clean, par["col"], par["name"], 10, hist_results_dir, case, par["log"])
            write_stats(df_clean, par["col"], RESULTS_DIR / f"{case}_distribution_stats.json")


if __name__ == "__main__":
    main()
        