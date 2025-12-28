# src/contracts/predictions.py
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    patient_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    generated_at_iso: str


class AdmitReq(BaseModel):
    patient_id: str
    encounter_id: str


class AdmitResp(BaseModel):
    patient_id: str
    encounter_id: str
    probability: float
    label: int
