from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError


class EventMeta(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    schema: Literal["vitals.v1"] = "vitals.v1"
    schema_version: int = 1
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VitalEvent(BaseModel):
    meta: EventMeta = Field(default_factory=EventMeta)

    patient_id: str
    encounter_id: str
    timestamp: datetime
    heart_rate: Optional[float] = None
    resp_rate: Optional[float] = None
    spo2: Optional[float] = None

    @classmethod
    def parse_json(cls, raw: str) -> "VitalEvent":
        try:
            return cls.model_validate_json(raw)
        except ValidationError as e:
            # keep error text short for logs/DLQ
            raise ValueError(
                f"schema_validation_failed: {e.errors()[0].get('msg', 'invalid')}"
            ) from e
