"""FastAPI service exposing the readmission risk model.

Loads the calibrated pipeline and the training feature list at start-up, so a
request only needs to send the fields it has - anything missing is left blank
and handled by the pipeline's imputers. Returns a calibrated probability and a
simple risk band.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import NUMERIC

MODELS = Path(__file__).resolve().parents[1] / "models"
model = joblib.load(MODELS / "readmission_model.joblib")
COLUMNS = json.loads((MODELS / "feature_columns.json").read_text())

app = FastAPI(title="30-day readmission risk", version="1.0")


class Encounter(BaseModel):
    features: dict


def _band(p: float) -> str:
    if p >= 0.20:
        return "high"
    if p >= 0.10:
        return "elevated"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(COLUMNS)}


@app.post("/predict")
def predict(encounter: Encounter):
    row = {c: np.nan for c in COLUMNS}
    row.update({k: v for k, v in encounter.features.items() if k in COLUMNS})
    frame = pd.DataFrame([row], columns=COLUMNS)
    for col in NUMERIC:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    proba = float(model.predict_proba(frame)[:, 1][0])
    return {"readmission_probability": round(proba, 4), "risk_band": _band(proba)}
