from datetime import datetime

from pydantic import BaseModel, Field


class VitalEvent(BaseModel):
    patient_id: str = Field(..., min_length=1)
    timestamp: datetime
    heart_rate: int = Field(..., ge=0, le=300)
    systolic_bp: int | None = Field(default=None, ge=0, le=400)
    diastolic_bp: int | None = Field(default=None, ge=0, le=300)
    spo2: int | None = Field(default=None, ge=0, le=100)
