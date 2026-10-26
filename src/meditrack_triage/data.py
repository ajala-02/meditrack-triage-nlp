"""Deterministic generator for explicitly synthetic, non-clinical training examples."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .constants import LABELS

# These are illustrative prompts, not medical rules. They intentionally include informal wording,
# spelling variants, negation, and a limited set of multilingual wording for error analysis.
EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "low": [
        ("minor_cold", "I have a stuffy nose and a little sore throat but I can eat and drink."),
        ("mild_headache", "I have a mild headache after staring at screens all day."),
        ("minor_rash", "A small itchy patch appeared on my arm and it is not spreading."),
        ("muscle_ache", "My shoulder feels sore after carrying bags yesterday."),
        ("seasonal_allergy", "My eyes are itchy and I keep sneezing when I go outside."),
        ("minor_stomach", "My stomach feels slightly upset after eating spicy food, no vomiting."),
        ("mild_cough", "I have an occasional dry cough but I am breathing normally."),
    ],
    "medium": [
        ("persistent_fever", "I have had a fever for three days and feel very tired."),
        ("worsening_cough", "My cough is getting worse and I am bringing up yellow mucus."),
        ("dehydration", "I have been vomiting since yesterday and cannot keep much water down."),
        ("abdominal_pain", "I have steady pain on the lower right side of my belly."),
        ("migraine", "I have a bad headache with nausea and light makes it worse."),
        ("infection", "The cut on my leg is red, warm, swollen and has pus."),
        ("breathing_change", "I get short of breath walking upstairs, which is new for me."),
    ],
    "high": [
        ("chest_pain", "I have crushing pressure in my chest and I am sweating."),
        ("severe_breathing", "I am struggling to breathe even while sitting still."),
        ("stroke_signs", "My face suddenly drooped and my speech is slurred."),
        ("severe_bleeding", "I cannot stop heavy bleeding from a deep cut."),
        ("unconscious", "My partner fainted and is hard to wake up."),
        ("anaphylaxis", "My lips are swelling quickly and my throat feels tight after food."),
        ("severe_headache", "This is the worst sudden headache of my life with confusion."),
    ],
}

OPENERS = ["", "Since this morning, ", "Please help, ", "I am worried because ", "For the last few hours, "]
ENDING = ["", " What should I do?", " Is this serious?", " I am not sure if I should worry."]
TYPOS = {"breathing": "breathng", "headache": "head ache", "stomach": "tummy", "because": "bc"}
MULTILINGUAL = {
    "low": "Mujhe halka headache hai after work, but I am okay otherwise.",
    "medium": "Mujhe teen din se fever hai aur I feel very weak.",
    "high": "Mere se saans nahi li ja rahi, chest feels tight.",
}


def _variant(text: str, rng: random.Random, label: str) -> tuple[str, str]:
    """Add surface variation without changing the synthetic example's intended label."""
    language = "en"
    if rng.random() < 0.08:
        return MULTILINGUAL[label], "mixed_hi_en"
    if rng.random() < 0.18:
        for source, replacement in TYPOS.items():
            if source in text.lower():
                text = text.replace(source, replacement)
                break
    if rng.random() < 0.20:
        text = text.lower()
    return f"{rng.choice(OPENERS)}{text}{rng.choice(ENDING)}".strip(), language


def generate_synthetic_dataset(rows_per_label: int = 180, seed: int = 42) -> pd.DataFrame:
    """Create a balanced, reproducible dataset with no patient-derived records."""
    if rows_per_label < 15:
        raise ValueError("rows_per_label must be at least 15 to support stratified splits")
    rng = random.Random(seed)
    rows: list[dict[str, str | int]] = []
    row_id = 1
    for label in LABELS:
        choices = list(EXAMPLES[label])
        rng.shuffle(choices)
        for index in range(rows_per_label):
            scenario, template = choices[index % len(choices)]
            text, language = _variant(template, rng, label)
            rows.append(
                {"id": f"synthetic-{row_id:04d}", "text": text, "triage": label,
                 "scenario": scenario, "language": language, "source": "synthetic"}
            )
            row_id += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def create_splits(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Return stratified, scenario-disjoint 70/10/20 train/validation/test splits.

    Holding a whole synthetic scenario out prevents near-identical template phrasing from
    appearing in both train and test data. This is stricter and more honest than a random
    row split for a generated dataset.
    """
    required = {"text", "triage", "scenario"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dataset must contain {sorted(required)}")
    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for label in LABELS:
        scenarios = sorted(frame.loc[frame["triage"] == label, "scenario"].unique())
        if len(scenarios) < 5:
            raise ValueError("Each label needs at least five distinct scenarios for group-disjoint splits")
        rng.shuffle(scenarios)
        train_end = max(1, round(len(scenarios) * 0.70))
        validation_end = max(train_end + 1, round(len(scenarios) * 0.80))
        for scenario in scenarios[:train_end]:
            assignments[f"{label}:{scenario}"] = "train"
        for scenario in scenarios[train_end:validation_end]:
            assignments[f"{label}:{scenario}"] = "validation"
        for scenario in scenarios[validation_end:]:
            assignments[f"{label}:{scenario}"] = "test"
    tagged = frame.copy()
    tagged["split"] = [assignments[f"{label}:{scenario}"] for label, scenario in zip(
        tagged["triage"], tagged["scenario"]
    )]
    return {name: tagged[tagged["split"] == name].drop(columns="split").reset_index(drop=True)
            for name in ("train", "validation", "test")}


def write_generated_data(output_dir: Path, rows_per_label: int = 180, seed: int = 42) -> dict[str, Path]:
    """Generate and persist the complete dataset and the fixed stratified splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_dataset(rows_per_label=rows_per_label, seed=seed)
    paths = {"all": output_dir / "synthetic_symptoms.csv"}
    dataset.to_csv(paths["all"], index=False)
    for name, split in create_splits(dataset, seed=seed).items():
        paths[name] = output_dir / f"{name}.csv"
        split.to_csv(paths[name], index=False)
    return paths
