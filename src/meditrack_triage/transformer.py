"""Optional DistilBERT fine-tuning path.

Imports for this module are intentionally lazy so the baseline remains lightweight.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .constants import LABELS
from .evaluation import evaluate_predictions, make_prediction_frame, write_error_analysis


def train_transformer(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    output_dir: Path,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    learning_rate: float = 2e-5,
) -> Path:
    """Fine-tune DistilBERT and save a Hugging Face model directory.

    Requires the ``transformer`` optional dependency group and access to the model files.
    Metrics include macro F1 and high-tier recall, the same decision-support priorities as
    the baseline. This function is for an educational prototype, not clinical use.
    """
    try:
        import numpy as np
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "Transformer dependencies are missing. Install with: pip install -e '.[transformer]'"
        ) from error

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    label_to_id = {label: index for index, label in enumerate(LABELS)}

    def to_dataset(frame: pd.DataFrame):
        data = frame[["text", "triage"]].rename(columns={"triage": "label"}).copy()
        data["label"] = data["label"].map(label_to_id)
        return Dataset.from_pandas(data, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_dataset = to_dataset(train).map(tokenize, batched=True)
    validation_dataset = to_dataset(validation).map(tokenize, batched=True)
    test_dataset = to_dataset(test).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label={i: label for i, label in enumerate(LABELS)},
        label2id=label_to_id,
    )

    def compute_metrics(prediction):
        from sklearn.metrics import f1_score, recall_score

        labels = prediction.label_ids
        predictions = np.argmax(prediction.predictions, axis=-1)
        return {
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "high_recall": recall_score(labels, predictions, labels=[label_to_id["high"]],
                                          average="macro", zero_division=0),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to=[],
        seed=42,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=train_dataset,
                      eval_dataset=validation_dataset, processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    test_prediction = trainer.predict(test_dataset)
    logits = test_prediction.predictions
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    predicted = [LABELS[index] for index in np.argmax(probabilities, axis=1)]
    test_output = output_dir / "test"
    metrics = evaluate_predictions(test["triage"], predicted, probabilities, test_output)
    rows = make_prediction_frame(test, predicted, probabilities, LABELS)
    rows.to_csv(test_output / "predictions.csv", index=False)
    write_error_analysis(rows, test_output)
    (output_dir / "training_config.json").write_text(json.dumps({
        "base_model": model_name, "epochs": epochs, "learning_rate": learning_rate,
        "labels": LABELS, "data": "synthetic only", "test_metrics": metrics,
    }, indent=2), encoding="utf-8")
    return output_dir / "model"

