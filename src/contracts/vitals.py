from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

class VitalEvent(BaseModel):
    patient_id: str
    encounter_id: str
    timestamp: datetime
    heart_rate: int = Field(..., ge=0)
    spo2: int | None = Field(default=None, ge=0, le=100)
