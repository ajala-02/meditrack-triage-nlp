"""Command-line workflow for data generation, training, evaluation, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baseline import save_baseline, top_features, train_baseline
from .data import write_generated_data
from .evaluation import evaluate_predictions, make_prediction_frame, write_error_analysis
from .transformer import train_transformer


def _paths(project_root: Path) -> tuple[Path, Path]:
    return project_root / "data", project_root / "artifacts"


def generate(args: argparse.Namespace) -> None:
    paths = write_generated_data(Path(args.data_dir), rows_per_label=args.rows_per_label, seed=args.seed)
    print("Generated synthetic data:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def train(args: argparse.Namespace) -> None:
    data_dir, artifacts_dir = _paths(Path(args.project_root))
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        write_generated_data(data_dir, seed=args.seed)
    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(data_dir / "validation.csv")
    model = train_baseline(train_frame, random_state=args.seed)
    probabilities = model.predict_proba(validation_frame["text"])
    predicted = model.predict(validation_frame["text"])
    output_dir = artifacts_dir / "baseline"
    metrics = evaluate_predictions(validation_frame["triage"], predicted, probabilities, output_dir / "validation")
    metadata = {"model": "TF-IDF (1-2 gram) + logistic regression", "seed": args.seed,
                "data": "synthetic only", "validation_metrics": metrics}
    model_path = save_baseline(model, output_dir, metadata)
    (output_dir / "top_features.json").write_text(json.dumps(top_features(model), indent=2), encoding="utf-8")
    print(f"Saved baseline to {model_path}")
    print(json.dumps(metrics, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    import joblib

    data_dir, artifacts_dir = _paths(Path(args.project_root))
    test = pd.read_csv(data_dir / "test.csv")
    model = joblib.load(artifacts_dir / "baseline" / "baseline.joblib")
    probabilities = model.predict_proba(test["text"])
    predicted = model.predict(test["text"])
    output_dir = artifacts_dir / "baseline" / "test"
    metrics = evaluate_predictions(test["triage"], predicted, probabilities, output_dir)
    rows = make_prediction_frame(test, predicted, probabilities, model.classes_)
    rows.to_csv(output_dir / "predictions.csv", index=False)
    write_error_analysis(rows, output_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Review errors in {output_dir / 'worst_errors.csv'}")


def transformer(args: argparse.Namespace) -> None:
    data_dir, artifacts_dir = _paths(Path(args.project_root))
    if not (data_dir / "train.csv").exists():
        write_generated_data(data_dir, seed=args.seed)
    path = train_transformer(data_dir / "train.csv", data_dir / "validation.csv", data_dir / "test.csv",
                             artifacts_dir / "transformer", model_name=args.model_name,
                             epochs=args.epochs, learning_rate=args.learning_rate)
    print(f"Saved fine-tuned model to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MediTrack synthetic symptom-triage NLP prototype")
    parser.add_argument("--project-root", default=".", help="Project directory (default: current directory)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate-data", help="Create synthetic data and fixed splits")
    generate_parser.add_argument("--data-dir", default="data")
    generate_parser.add_argument("--rows-per-label", type=int, default=180)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.set_defaults(func=generate)
    for name, handler in [("train-baseline", train), ("evaluate-baseline", evaluate)]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--seed", type=int, default=42)
        sub.set_defaults(func=handler)
    transformer_parser = subparsers.add_parser("train-transformer", help="Fine-tune optional DistilBERT model")
    transformer_parser.add_argument("--seed", type=int, default=42)
    transformer_parser.add_argument("--model-name", default="distilbert-base-uncased")
    transformer_parser.add_argument("--epochs", type=int, default=3)
    transformer_parser.add_argument("--learning-rate", type=float, default=2e-5)
    transformer_parser.set_defaults(func=transformer)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
