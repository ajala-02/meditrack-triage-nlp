# MediTrack Triage NLP Prototype

An educational NLP project that classifies **synthetic** free-text symptom descriptions into three illustrative triage tiers: `low`, `medium`, and `high`.

> **Safety boundary:** This is a learning project, not a medical device. It is not validated for clinical use and must never be used to diagnose, triage, or make emergency decisions. The repository contains no real patient records; its generated examples are synthetic.

## Why this project

Patients rarely describe symptoms in tidy clinical language. They use informal wording, partial information, typos, and sometimes mixed languages. This project asks whether a model trained on symptom-like text can assign an illustrative urgency tier, and how to assess whether it is useful.

The work is intentionally structured like an applied NLP study:

1. Generate a reproducible, balanced synthetic dataset and fixed scenario-disjoint train/validation/test splits.
2. Train an interpretable **TF-IDF + logistic regression** baseline.
3. Fine-tune an optional **DistilBERT** model and compare it fairly with the baseline.
4. Evaluate held-out data with precision, recall, F1, and a confusion matrix.
5. Prioritise recall for the `high` tier: a high-tier miss is more important to inspect than an extra alert.
6. Export the highest-confidence errors for manual analysis.

## Repository layout

```text
src/meditrack_triage/
  data.py          synthetic-data generator and fixed splits
  baseline.py      TF-IDF + logistic-regression model
  transformer.py   optional DistilBERT fine-tuning workflow
  evaluation.py    metrics, confusion matrix, error-review files
  api.py           optional FastAPI service
  cli.py           reproducible command-line workflow
tests/             data and baseline smoke tests
data/              generated CSV files (gitignored)
artifacts/         models, metrics, matrices, errors (gitignored)
```

## Quick start

Requires Python 3.10–3.13. Create an isolated environment, install the baseline dependencies, then run the full baseline workflow:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

meditrack-triage generate-data --rows-per-label 180
meditrack-triage train-baseline
meditrack-triage evaluate-baseline
pytest
```

The commands produce:

- `data/{train,validation,test}.csv`: deterministic scenario-disjoint, stratified splits. A synthetic symptom scenario is held entirely in one split, so near-identical templates cannot leak from training into testing.
- `artifacts/baseline/baseline.joblib`: saved baseline model.
- `artifacts/baseline/{validation,test}/metrics.json`: macro/weighted F1, per-tier metrics, and `high_recall`.
- `artifacts/baseline/test/confusion_matrix.{csv,png}`: held-out error pattern.
- `artifacts/baseline/test/worst_errors.csv`: high-confidence mistakes, with high-tier misses first.
- `artifacts/baseline/top_features.json`: most influential words/n-grams per class.

## Evaluation protocol

The final test split is untouched during baseline development. It is scenario-disjoint from training, a deliberately tougher test that avoids template leakage in generated data. Validation metrics are used when training; report the test metrics only once the workflow is stable. Do not present accuracy alone: this dataset is balanced, and accuracy can still conceal dangerous-looking high-tier misses.

| Metric | Why it matters |
| --- | --- |
| Precision | How often a predicted tier is correct. |
| Recall | How much of each actual tier the model identifies. `high` recall is the priority metric here. |
| F1-score | Balances precision and recall. Macro F1 weights each tier equally. |
| Confusion matrix | Shows exactly which tiers are being mixed up. |
| Worst errors | Makes qualitative failure analysis repeatable rather than anecdotal. |

When the optional transformer has been evaluated on the same untouched test file, add the measured values here:

| Model | Macro F1 | High-tier recall | Notes |
| --- | ---: | ---: | --- |
| TF-IDF + logistic regression | 0.2928 | 0.1154 | Measured on the scenario-disjoint synthetic test split, seed 42 |
| DistilBERT fine-tuned | To be measured | To be measured | Compare against the same test split |

The baseline's validation Macro F1 was 0.9111 but its held-out scenario-disjoint Macro F1 was only 0.2928. This gap is the main finding of the initial run: a small synthetic template dataset can look excellent when phrases overlap, while failing to generalise to unseen symptom scenarios. The project deliberately retains this result and its error files rather than reporting only the favourable validation score.

## Optional DistilBERT model

Install the transformer dependencies and fine-tune on the generated training/validation data:

```bash
python -m pip install -e ".[transformer]"
meditrack-triage train-transformer --epochs 3
```

The first run downloads `distilbert-base-uncased` from Hugging Face. It needs network access and a supported PyTorch Python version (3.10–3.13; Python 3.15 is not yet supported by PyTorch). It is appreciably more resource-intensive than the baseline. The command evaluates the saved best model once on the same held-out test file and writes the same metrics, confusion matrix, predictions, and error-review files under `artifacts/transformer/test/`.

## Error analysis guide

Open `worst_errors.csv` after every model run and label each mistake before changing the model. Useful categories include:

- vague or underspecified text;
- multiple symptoms in one description;
- negation and context (for example, “no vomiting”);
- typographical mistakes or informal phrasing;
- mixed-language descriptions;
- synthetic-template overlap or gaps.

For each category, record the actual tier, predicted tier, model confidence, possible reason, and a proposed data/model change. A confident high-tier miss is the most important kind of error to investigate, not something to silently remove.

## Optional API

Train the baseline first, then launch the local development service:

```bash
uvicorn meditrack_triage.api:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"symptom_description":"I have a mild headache after working all day"}'
```

The response includes a tier, class probabilities, confidence, and an explicit educational-use disclaimer. A production service would require clinical governance, real consented and representative data, external validation, calibration analysis, monitoring, privacy controls, security review, and human clinical oversight.

## Limitations and responsible use

- Labels and examples are deliberately synthetic; performance here does **not** indicate clinical performance.
- The label set and phrase templates are simplistic and cannot represent real patient complexity.
- A classifier cannot establish diagnosis, account for full medical history, or replace emergency care.
- Confidence is not proof of correctness; model calibration still needs dedicated analysis.
- Language coverage is intentionally limited. Mixed-language examples are included to expose, not solve, that limitation.

See `MODEL_CARD.md` for the intended use, prohibited use, evaluation context, and risk notes.

## Portfolio narrative

This project demonstrates an end-to-end applied NLP workflow: reproducible data generation, a strong classical baseline, a transformer comparison, risk-aware evaluation, explainability, error analysis, model persistence, and a service boundary. Its contribution is methodological—not a claim of clinically safe automated triage.

