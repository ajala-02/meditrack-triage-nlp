"""Evaluation artifacts prioritising recall for high-acuity synthetic examples."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .constants import HIGH, LABELS


def evaluate_predictions(actual: pd.Series, predicted, probabilities, output_dir: Path) -> dict:
    """Write machine-readable metrics, a confusion matrix, and error-review data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(actual, predicted, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "high_recall": round(float(report[HIGH]["recall"]), 4),
        "per_class": {label: {key: round(float(report[label][key]), 4)
                     for key in ("precision", "recall", "f1-score", "support")} for label in LABELS},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis("actual").to_csv(
        output_dir / "confusion_matrix.csv"
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=LABELS,
             yticklabels=LABELS, xlabel="Predicted triage", ylabel="Actual triage",
             title="Confusion matrix — synthetic test split")
    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    return metrics


def make_prediction_frame(test: pd.DataFrame, predicted, probabilities, classes) -> pd.DataFrame:
    """Create reviewable prediction rows with confidence and explicit error flags."""
    result = test.copy().reset_index(drop=True)
    result["predicted_triage"] = predicted
    result["confidence"] = probabilities.max(axis=1).round(4)
    for index, label in enumerate(classes):
        result[f"probability_{label}"] = probabilities[:, index].round(4)
    result["is_error"] = result["triage"] != result["predicted_triage"]
    result["is_high_miss"] = (result["triage"] == HIGH) & (result["predicted_triage"] != HIGH)
    return result.sort_values(["is_high_miss", "is_error", "confidence"], ascending=[False, False, False])


def write_error_analysis(predictions: pd.DataFrame, output_dir: Path, limit: int = 20) -> Path:
    """Persist the highest-confidence mistakes for transparent manual review."""
    errors = predictions[predictions["is_error"]].head(limit)
    errors.to_csv(output_dir / "worst_errors.csv", index=False)
    counts = (errors.groupby(["triage", "predicted_triage"], dropna=False).size()
              .rename("count").reset_index())
    counts.to_csv(output_dir / "error_pairs.csv", index=False)
    return output_dir / "worst_errors.csv"
