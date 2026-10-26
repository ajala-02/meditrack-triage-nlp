"""TF-IDF + logistic regression baseline and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .constants import LABELS


def build_baseline(random_state: int = 42) -> Pipeline:
    """Build an interpretable text-classification baseline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ]
    )


def train_baseline(train: pd.DataFrame, random_state: int = 42) -> Pipeline:
    model = build_baseline(random_state=random_state)
    model.fit(train["text"], train["triage"])
    return model


def save_baseline(model: Pipeline, output_dir: Path, metadata: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline.joblib"
    joblib.dump(model, model_path)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def top_features(model: Pipeline, top_n: int = 12) -> dict[str, list[dict[str, float | str]]]:
    """Expose features with the greatest positive logistic-regression weight per class."""
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    classifier: LogisticRegression = model.named_steps["classifier"]
    terms = vectorizer.get_feature_names_out()
    return {
        label: [
            {"term": str(terms[index]), "weight": round(float(weights[index]), 4)}
            for index in weights.argsort()[-top_n:][::-1]
        ]
        for label, weights in zip(classifier.classes_, classifier.coef_)
        if label in LABELS
    }
