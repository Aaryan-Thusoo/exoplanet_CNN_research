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
RESULTS_DIR = PROJECT_ROOT / "results"


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
        ("False Positives", "false_positives", "True 0, predicted 1"),
        ("False Negatives", "false_negatives", "True 1, predicted 0"),
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

def main():
    misclassified = np.load(RESULTS_DIR / "misclassified_lightcurves.npz")

    false_positives = {
        "X": misclassified["false_positive_X"],
        "y_true": misclassified["false_positive_y_true"],
        "y_pred": misclassified["false_positive_y_pred"],
        "y_pred_prob": misclassified["false_positive_y_pred_prob"],
        "kepid": misclassified["false_positive_kepid"],
        "confidence": misclassified["false_positive_confidence"],
    }

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

if __name__ == "__main__":
    main()