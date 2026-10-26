# Model card — MediTrack Triage NLP Prototype

## Model details

The initial baseline is a TF-IDF word/bi-gram vectorizer plus class-balanced logistic regression. The comparison model path fine-tunes `distilbert-base-uncased`. Labels are illustrative `low`, `medium`, and `high` tiers.

## Intended use

This repository is an educational demonstration of text classification, baseline comparison, evaluation, and error analysis. It may be used to learn reproducible NLP workflows on the supplied synthetic data.

## Not intended for use

It must not be used for clinical triage, diagnosis, treatment advice, emergency routing, or any decision affecting a patient. It must not be connected to a patient-facing or clinician-facing workflow.

## Data

All records are generated from hand-authored illustrative templates. They are not patient records, do not represent clinical prevalence, and do not establish medically correct triage rules. Limited typo and Hindi-English mixed-language variants exist solely to support error analysis.

## Evaluation context

Data is divided by synthetic scenario, so no scenario appears in both train and test data. In the initial seed-42 baseline run, validation Macro F1 was 0.9111, while the scenario-disjoint test Macro F1 was 0.2928 and high-tier recall was 0.1154. This is evidence of limited generalisation, not a performance claim.

## Known risks and limitations

- Text lacks medical history, physical examination, vital signs, time course, and clinician judgement.
- The three labels collapse complex decisions into simplistic categories.
- Confidence scores are uncalibrated and can be confidently wrong.
- Performance on synthetic English-dominant prompts says nothing about real populations, languages, access needs, or demographic fairness.
- A high-tier miss is especially concerning. Such errors are explicitly exported for review.

## Required safeguards before any future real-world study

Ethics approval where applicable, consented and privacy-protected data governance, clinician-designed labelling, representative external validation, calibration and subgroup analysis, human oversight, security review, monitoring, and formal regulatory assessment would all be required. None are supplied by this educational prototype.
