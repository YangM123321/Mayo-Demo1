from __future__ import annotations

import json
from pathlib import Path

from src.streaming.schemas import VitalEvent

AUDIT_PATH = Path("/tmp/vitals_audit.log")


def handle_vital(evt: VitalEvent) -> None:
    # In real life this would upsert into DB / BigQuery / FHIR, etc.
    # For demo: append an auditable record.
    record = {
        "event_id": evt.meta.event_id,
        "schema": evt.meta.schema,
        "schema_version": evt.meta.schema_version,
        "patient_id": evt.patient_id,
        "encounter_id": evt.encounter_id,
        "timestamp": evt.timestamp.isoformat(),
        "heart_rate": evt.heart_rate,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
