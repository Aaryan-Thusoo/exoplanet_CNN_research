import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

import json
from scipy.stats import kruskal, mannwhitneyu

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="seaborn"
)

def prettify_label(text: str) -> str:
    abbreviations = {"snr": "SNR", "id": "ID", "koi": "KOI"}

    words = text.replace("_", " ").lower().split()

    return " ".join(
        abbreviations.get(word, word.capitalize())
        for word in words
    )

def box_jitter(df: pd.DataFrame, column: str, title: str, output_dir: Path, case: str, log_scale: bool = False) -> None:

    plt.figure(figsize=(8, 8))

    order = ["Low", "Medium", "High"]

    box_palette = {
        "Low": "#bcd7f0",
        "Medium": "#ffd8a8",
        "High": "#f5b7b1",
    }

    point_palette = {
        "Low": "#1f77b4",
        "Medium": "#ff7f0e",
        "High": "#d62728",
    }

    plt.title(title)

    np.random.seed(42)
    sns.boxplot(
        data=df,
        x="confidence_group",
        y=column,
        order=order,
        showfliers=False,
        palette=box_palette,
        hue="confidence_group",
    )

    sns.stripplot(
        data=df,
        x="confidence_group",
        y=column,
        order=order,
        palette=point_palette,
        jitter=True,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.4,
        hue="confidence_group",
    )

    plt.ylabel(title)
    plt.xlabel("Confidence Group")

    if log_scale:
        plt.yscale("log")

    plt.savefig(output_dir / f"{case}_boxplot_{column}.png", bbox_inches="tight")


def ecdf_plotting(df: pd.DataFrame, column: str, title: str, output_dir: Path, case: str, log_scale=False):

    plot_df = df.dropna(subset=[column, "confidence_group"]).copy()

    if log_scale:
        plot_df = plot_df[plot_df[column] > 0].copy()

    if plot_df.empty:
        return

    plt.figure(figsize=(9, 6))

    order = ["Low", "Medium", "High"]
    palette = {
        "Low": "#1f77b4",
        "Medium": "#ff7f0e",
        "High": "#d62728",
    }

    for confidence_group in order:
        group_values = (
            plot_df.loc[plot_df["confidence_group"] == confidence_group, column]
            .sort_values()
            .to_numpy()
        )

        if len(group_values) == 0:
            continue

        y_values = np.arange(1, len(group_values) + 1) / len(group_values)
        x_step = np.insert(group_values, 0, group_values[0])
        y_step = np.insert(y_values, 0, 0)

        plt.step(
            x_step,
            y_step,
            where="post",
            color=palette[confidence_group],
            linewidth=2,
            label=confidence_group,
        )
        plt.fill_between(
            x_step,
            y_step,
            step="post",
            color=palette[confidence_group],
            alpha=0.15,
        )

    if log_scale:
        plt.xscale("log")

    plt.xlabel(title)
    plt.ylabel("Cumulative Fraction of Misclassified Samples")
    plt.title(f"{title} ECDF by Confidence Group")
    plt.legend(title="Confidence Group")
    plt.tight_layout()
    plt.savefig(output_dir / f"{case}_ecdf_{column}.png", bbox_inches="tight")
    plt.close()

def write_stats(df: pd.DataFrame, column: str, output_file: Path) -> None:
    low_data = df.loc[df["confidence_group"] == "Low", column].dropna()
    med_data = df.loc[df["confidence_group"] == "Medium", column].dropna()
    high_data = df.loc[df["confidence_group"] == "High", column].dropna()

    stat, p_value = kruskal(low_data, med_data, high_data)

    pairs = {
        "low_vs_med": mannwhitneyu(low_data, med_data, alternative="two-sided"),
        "low_vs_high": mannwhitneyu(low_data, high_data, alternative="two-sided"),
        "med_vs_high": mannwhitneyu(med_data, high_data, alternative="two-sided"),
    }

    column_stats = {
        "kruskal_wallis": {
            "statistic": float(stat),
            "p_value": float(p_value),
        },
        "pairwise_mann_whitney_u": {
            name: {
                "u_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "bonferroni_p_value": float(min(result.pvalue * 3, 1.0)),
            }
            for name, result in pairs.items()
        },
        "groups": {
            "Low": {
                "count": int(len(low_data)),
                "median": float(low_data.median()),
                "mean": float(low_data.mean()),
            },
            "Medium": {
                "count": int(len(med_data)),
                "median": float(med_data.median()),
                "mean": float(med_data.mean()),
            },
            "High": {
                "count": int(len(high_data)),
                "median": float(high_data.median()),
                "mean": float(high_data.mean()),
            },
        },
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        with open(output_file, "r") as file:
            all_stats = json.load(file)
    else:
        all_stats = {}

    all_stats[column] = column_stats

    with open(output_file, "w") as file:
        json.dump(all_stats, file, indent=2)
