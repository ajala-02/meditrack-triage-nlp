"""Shared labels and project constants."""

LOW = "low"
MEDIUM = "medium"
HIGH = "high"
LABELS = [LOW, MEDIUM, HIGH]

# This order is deliberately used in reports and confusion matrices.
LABEL_DISPLAY = {LOW: "Low", MEDIUM: "Medium", HIGH: "High"}
DISCLAIMER = (
    "Educational prototype only. It uses synthetic data and must not be used for diagnosis, "
    "clinical triage, or emergency decisions. Seek urgent medical help for emergencies."
)

