from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pandas import DataFrame

from scripts.helpers.misclassified_helper import *
from scripts.helpers.yaml_reading import *

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="seaborn"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "multiclass" / "misclassified_results"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
YAML_DIR = PROJECT_ROOT / "configs" / "mc_evaluation_params.yaml"
PAPER_DIR = PROJECT_ROOT / "paper" / "figures" / "plots" / "multiclass"


def load_data() -> tuple[DataFrame, Any]:
    misclassified = np.load(RESULTS_DIR / "misclassified_lightcurves.npz")

    mistake_df = pd.DataFrame({
        "kepid": misclassified["kepid"],
        "y_true": misclassified["y_true"],
        "y_pred": misclassified["y_pred"],
        "confidence": misclassified["confidence"],
    })

    return mistake_df, misclassified["X"]


def read_metadata_csv() -> DataFrame:
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


def main():
    params = load_params(YAML_DIR)["plots_info"]

    mistake_df, mistake_X = load_data()
    metadata = read_metadata_csv()

    mistake_df["kepid"] = mistake_df["kepid"].astype(int)

    mistake_df = mistake_df.merge(
        metadata,
        on="kepid",
        how="left",
    )

    mistake_df["std"] = np.std(mistake_X.squeeze(), axis=1)
    mistake_df["std"] = mistake_df["std"].replace(0, np.nan)
    mistake_df["calculated_SNR"] = mistake_df["koi_depth"] / mistake_df["std"]
    mistake_df["transit_depth"] = calculate_transit_depth(mistake_X)
    mistake_df["transit_depth_abs"] = mistake_df["transit_depth"].abs()

    low_conf = 0.65
    med_conf = 0.8

    mistake_df["confidence_group"] = np.select(
        [
            mistake_df["confidence"] <= low_conf,
            (mistake_df["confidence"] > low_conf) & (mistake_df["confidence"] <= med_conf),
            mistake_df["confidence"] > med_conf,
        ],
        ["Low", "Medium", "High"],
        default="unknown",
    )

    box_results_dir = RESULTS_DIR / "misclassified_plots" / "boxplots"
    ecdf_results_dir = RESULTS_DIR / "misclassified_plots" / "ecdfs"

    box_results_dir.mkdir(parents=True, exist_ok=True)
    ecdf_results_dir.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = mistake_df.dropna().copy()

    for par in params:
        box_jitter(
            df_clean,
            par["col"],
            f"Multiclass Misclassified {par['name']}",
            PAPER_DIR,
            "mc",
            par["log"],
        )
        ecdf_plotting(
            df_clean,
            par["col"],
            par["name"],
            ecdf_results_dir,
            "mc",
            par["log"],
        )
        write_stats(
            df_clean,
            par["col"],
            RESULTS_DIR / "mc_distribution_stats.json",
        )


if __name__ == "__main__":
    main()
