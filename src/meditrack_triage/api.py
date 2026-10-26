"""Small HTTP interface for the saved baseline model.

The service deliberately labels every response as an educational prototype.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .constants import DISCLAIMER

DEFAULT_MODEL_PATH = Path(os.getenv("MEDI_TRIAGE_MODEL_PATH", "artifacts/baseline/baseline.joblib"))


class PredictionRequest(BaseModel):
    symptom_description: Annotated[str, Field(min_length=3, max_length=1000)]


class PredictionResponse(BaseModel):
    triage_level: str
    confidence: float
    probabilities: dict[str, float]
    disclaimer: str = DISCLAIMER


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None
    yield


app = FastAPI(
    title="MediTrack Triage NLP Prototype",
    version="0.1.0",
    description=DISCLAIMER,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": app.state.model is not None, "scope": "educational prototype"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="No saved baseline model. Train it before starting the API.")
    probabilities = model.predict_proba([request.symptom_description])[0]
    classes = model.classes_
    best_index = int(probabilities.argmax())
    return PredictionResponse(
        triage_level=str(classes[best_index]),
        confidence=round(float(probabilities[best_index]), 4),
        probabilities={str(label): round(float(value), 4) for label, value in zip(classes, probabilities)},
    )

