<div align="center">

# 🩺 MediTrack Triage NLP

**Can a model read a patient's own words and flag how urgent they are, and how do we know if it can?**

![Python](https://img.shields.io/badge/Python-3.10--3.13-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Model](https://img.shields.io/badge/Baseline-TF--IDF%20%2B%20LogReg-orange)
![Data](https://img.shields.io/badge/Data-synthetic%20only-lightgrey)
![Status](https://img.shields.io/badge/Status-educational%20prototype-yellow)

</div>

> ⚠️ **Safety boundary:** This is a learning project, not a medical device. It is not validated for clinical use and must never be used to diagnose, triage, or make emergency decisions. The repository contains **no real patient data**. Every example is synthetic.

---

## 📌 Headline result

| Split | Macro F1 | High-tier recall |
|---|---|---|
| Validation | 0.9111 | 0.9200 |
| **Held-out test (unseen scenarios)** | **0.2928** | **0.1154** |

The baseline looks excellent on validation and performs close to chance on unseen scenarios. For three balanced classes, random guessing gives a macro F1 of about 0.33.

**What this means:** Both validation and test contain scenarios excluded from training, but this small synthetic split is sensitive to which scenarios land in each set. The large gap suggests the baseline relies heavily on template-specific wording and does not generalise reliably to different symptom scenarios. On the held-out test set, it identified only 3 of 26 high-tier examples, so it is not suitable for any triage use.

I kept this result and its error files instead of reporting only the favourable validation score.

---

## 🎯 Why this project

Patients rarely describe symptoms in tidy clinical language. They use informal wording, partial information, typos, and sometimes mixed languages. This project is a small applied NLP study of three questions:

1. Can a model trained on symptom-like text assign an illustrative urgency tier (low, medium, high)?
2. How should we evaluate it, especially for the cases where a miss matters most?
3. Where and why does it fail?

It was built as the NLP component for [MediTrack](https://github.com/ajala-02/meditrack), my post-discharge patient monitoring app.

---

## 🧪 What's inside

| Component | Status |
|---|---|
| Synthetic dataset generator with fixed, scenario-disjoint splits | ✅ Done |
| TF-IDF + logistic regression baseline | ✅ Done |
| Metrics: precision, recall, macro F1, confusion matrix | ✅ Done |
| Worst-error export for manual analysis | ✅ Done |
| FastAPI service (`/health`, `/predict`) serving the baseline | ✅ Done |
| DistilBERT fine-tuning workflow | 🛠 Implemented, **not yet evaluated** |
| Fix for the generalisation gap (more varied data) | 🔜 Planned |

---

## 📊 Model comparison

| Model | Macro F1 | High-tier recall | Notes |
|---|---|---|---|
| TF-IDF + logistic regression | 0.2928 | 0.1154 | Scenario-disjoint synthetic test split, seed 42 |
| DistilBERT (fine-tuned) | To be measured | To be measured | Will use the same test split |

I prioritise **high-tier recall** because missing a serious case matters more than raising an extra alert. Accuracy alone is not enough here, since it can hide dangerous misses.

---

## 🔍 Error analysis

### API spot checks

| Input | Predicted | Top probability | Comment |
|---|---|---|---|
| "my chest feels heavy" | high | 0.69 | Wording associated with high-tier training features |
| "I feel off since this morning" | high | 0.36 | Probabilities almost uniform (0.36 / 0.32 / 0.32), so this is essentially a guess |

A confident-looking label can hide an uncertain prediction. A natural next step is an "uncertain" outcome when the top probability falls below a threshold, flagged for human review.

### Worst errors (from `worst_errors.csv`)

| Text | Actual | Predicted | Confidence | Category | Likely reason |
|---|---|---|---|---|---|
| "My lips are swelling quickly and my throat feels tight after food." | high | low | 0.5502 | Unseen high-tier scenario | The held-out anaphylaxis scenario was absent from training. |
| "Since this morning, I get short of breath walking upstairs, which is new for me. What should I do?" | medium | high | 0.4936 | Unseen medium-tier scenario | The model over-weighted "short of breath" without understanding the fuller context. |

**Categories to label:** vague text · multiple symptoms · negation ("no vomiting") · typos and informal wording · mixed language · template overlap or gaps.

---

## 🗂 Repository layout

```
src/meditrack_triage/
  data.py          synthetic-data generator and fixed splits
  baseline.py      TF-IDF + logistic-regression model
  transformer.py   DistilBERT fine-tuning workflow
  evaluation.py    metrics, confusion matrix, error-review files
  api.py           FastAPI service
  cli.py           reproducible command-line workflow
tests/             data and baseline smoke tests
data/              generated CSV files (gitignored)
artifacts/         models, metrics, matrices, errors (gitignored)
```

---

## 🚀 Quick start

Requires **Python 3.10–3.13**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

meditrack-triage generate-data --rows-per-label 180
meditrack-triage train-baseline
meditrack-triage evaluate-baseline
pytest
```

**Outputs**
- `data/{train,validation,test}.csv`: deterministic, scenario-disjoint, stratified splits
- `artifacts/baseline/baseline.joblib`: saved model
- `artifacts/baseline/{validation,test}/metrics.json`: macro and weighted F1, per-tier metrics, high-tier recall
- `artifacts/baseline/test/confusion_matrix.{csv,png}`: held-out error pattern
- `artifacts/baseline/test/worst_errors.csv`: highest-confidence mistakes, high-tier misses first
- `artifacts/baseline/top_features.json`: most influential words per class

### Optional: DistilBERT

```bash
python -m pip install -e ".[transformer]"
meditrack-triage train-transformer --epochs 3
```

Needs network access (downloads `distilbert-base-uncased`) and a PyTorch-supported Python (3.10–3.13).

---

## 🌐 API

```bash
uvicorn meditrack_triage.api:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptom_description":"I have a mild headache after working all day"}'
```

Returns a triage tier, class probabilities, confidence and a disclaimer. The service currently serves the **TF-IDF baseline**.



---

## ⚖️ Evaluation protocol

- The test split is held out during baseline development and is **scenario-disjoint** from training, so near-identical templates cannot leak across splits.
- Validation metrics guide training; test metrics are reported once the workflow is stable.
- Reported metrics: precision, recall, F1, confusion matrix.

---

## ⚠️ Limitations

- All labels and examples are synthetic. Performance here says nothing about clinical performance.
- The label set and templates are simplistic and cannot represent real patient complexity.
- A classifier cannot diagnose, account for medical history, or replace emergency care.
- Confidence is not correctness. Calibration needs its own analysis.
- Language coverage is limited. Mixed-language examples are included to expose this gap, not to solve it.
- The DistilBERT comparison has not been run yet.

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, prohibited use and risk notes.

---

## 🛣 Next steps

- [ ] Run DistilBERT on the same held-out test split and fill in the comparison table
- [ ] Add more varied, paraphrased and mixed-language data to test whether the generalisation gap closes
- [ ] Add an "uncertain" outcome below a confidence threshold
- [ ] Label more rows from `worst_errors.csv` in the error-analysis table

---

## 👤 Author

**Ajala Nalawade**: [GitHub](https://github.com/ajala-02)
