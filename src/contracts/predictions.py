from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    patient_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    generated_at_iso: str
