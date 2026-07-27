import base64
from io import BytesIO
import numpy as np

from pathlib import Path
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow import keras
import json

from scripts.helpers.yaml_reading import load_params
from scripts.helpers.layers_helper import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Parameters file path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "configs"))
YAML_FILE = PROJECT_ROOT / "configs"/ "evaluation_params.yaml"

DATA_DIR = PROJECT_ROOT / "data" / "real_model"
MODEL_PATH = PROJECT_ROOT / "models" / "real_model.keras"
RESULTS_DIR = PROJECT_ROOT / "results" / "misclassified_results"


def lightcurve_plot_to_base64(lightcurve, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(np.squeeze(lightcurve), linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Normalized flux")

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)

    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return image_base64

def save_misclassified_html_report(misclassified: dict, output_path: Path) -> None:
    html = """
    <html>
    <head>
        <title>Misclassified Light Curves</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .sample { margin-bottom: 36px; border-bottom: 1px solid #ddd; padding-bottom: 24px; }
            img { max-width: 100%; }
            code { background: #f2f2f2; padding: 2px 4px; }
        </style>
    </head>
    <body>
    <h1>Misclassified Light Curves</h1>
    """

    sections = [
        ("False Positives", "false_positives", "True eclipsing binary, predicted transit"),
        ("False Negatives", "false_negatives", "True transit, predicted eclipsing binary"),
    ]

    for heading, key, description in sections:
        group = misclassified[key]

        html += f"<h2>{heading}</h2>"
        html += f"<p>{description}</p>"

        for i in range(len(group["X"])):
            kepid = group["kepid"][i]
            y_true = group["y_true"][i]
            y_pred = group["y_pred"][i]
            prob = group["y_pred_prob"][i]
            confidence = group["confidence"][i]

            title = (
                f"KepID {kepid} | "
                f"true={y_true}, pred={y_pred}, "
                f"p={prob:.3f}, confidence={confidence:.3f}"
            )

            image = lightcurve_plot_to_base64(group["X"][i], title)

            html += f"""
            <div class="sample">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{image}">
            </div>
            """

    html += """
    </body>
    </html>
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def save_single_section_html_report(
    group: dict,
    output_path: Path,
    heading: str,
    description: str,
) -> None:
    html = f"""
    <html>
    <head>
        <title>{heading}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .sample {{ margin-bottom: 36px; border-bottom: 1px solid #ddd; padding-bottom: 24px; }}
            img {{ max-width: 100%; }}
            code {{ background: #f2f2f2; padding: 2px 4px; }}
        </style>
    </head>
    <body>
    <h1>{heading}</h1>
    <p>{description}</p>
    """

    for i in range(len(group["X"])):
        kepid = group["kepid"][i]
        y_true = group["y_true"][i]
        y_pred = group["y_pred"][i]
        prob = group["y_pred_prob"][i]
        confidence = group["confidence"][i]

        title = (
            f"KepID {kepid} | "
            f"true={y_true}, pred={y_pred}, "
            f"p={prob:.3f}, confidence={confidence:.3f}"
        )

        image = lightcurve_plot_to_base64(group["X"][i], title)

        html += f"""
        <div class="sample">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{image}">
        </div>
        """

    html += """
    </body>
    </html>
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

def main():
    misclassified = np.load(RESULTS_DIR / "misclassified_lightcurves.npz")

    # True eclipsing binary, predicted transit
    false_positives = {
        "X": misclassified["false_positive_X"],
        "y_true": misclassified["false_positive_y_true"],
        "y_pred": misclassified["false_positive_y_pred"],
        "y_pred_prob": misclassified["false_positive_y_pred_prob"],
        "kepid": misclassified["false_positive_kepid"],
        "confidence": misclassified["false_positive_confidence"],
    }

    # True transit, predicted eclipsing binary
    false_negatives = {
        "X": misclassified["false_negative_X"],
        "y_true": misclassified["false_negative_y_true"],
        "y_pred": misclassified["false_negative_y_pred"],
        "y_pred_prob": misclassified["false_negative_y_pred_prob"],
        "kepid": misclassified["false_negative_kepid"],
        "confidence": misclassified["false_negative_confidence"],
    }

    misclassified_groups = {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    save_misclassified_html_report(
        misclassified_groups,
        RESULTS_DIR / "misclassified_lightcurves.html",
    )

    # Looking at specific examples of high confidence results which are misclassified
    fp_high_confidence_filter = false_positives["confidence"] > 0.8
    fp_high_confidence = {
        "X": misclassified["false_positive_X"][fp_high_confidence_filter],
        "y_true": misclassified["false_positive_y_true"][fp_high_confidence_filter],
        "y_pred": misclassified["false_positive_y_pred"][fp_high_confidence_filter],
        "y_pred_prob": misclassified["false_positive_y_pred_prob"][fp_high_confidence_filter],
        "kepid": misclassified["false_positive_kepid"][fp_high_confidence_filter],
        "confidence": misclassified["false_positive_confidence"][fp_high_confidence_filter],
    }

    save_single_section_html_report(
        fp_high_confidence,
        RESULTS_DIR / "false_positive_high_confidence.html",
        heading="High-Confidence False Positives",
        description="True eclipsing binary, predicted transit. Eclipsing binary examples mistaken as exoplanet/transit cases.",
    )

    fn_high_confidence_filter = false_negatives["confidence"] > 0.8
    fn_high_confidence = {
        "X": misclassified["false_negative_X"][fn_high_confidence_filter],
        "y_true": misclassified["false_negative_y_true"][fn_high_confidence_filter],
        "y_pred": misclassified["false_negative_y_pred"][fn_high_confidence_filter],
        "y_pred_prob": misclassified["false_negative_y_pred_prob"][fn_high_confidence_filter],
        "kepid": misclassified["false_negative_kepid"][fn_high_confidence_filter],
        "confidence": misclassified["false_negative_confidence"][fn_high_confidence_filter],
    }

    save_single_section_html_report(
        fn_high_confidence,
        RESULTS_DIR / "false_negative_high_confidence.html",
        heading="High-Confidence False Negatives",
        description="True transit, predicted eclipsing binary. Exoplanet/transit examples mistaken as eclipsing binaries.",
    )

if __name__ == "__main__":
    main()