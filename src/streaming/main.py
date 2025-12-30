from __future__ import annotations

import json
import os

from src.streaming.consumer import run_forever
from src.streaming.schemas import VitalEvent

AUDIT_PATH = os.getenv("VITALS_AUDIT_PATH", "/tmp/vitals_audit.log")


def handler(evt: VitalEvent) -> None:
    # minimal "business logic" for CI: write an audit record
    rec = {
        "event_id": evt.meta.event_id,
        "patient_id": evt.patient_id,
        "encounter_id": evt.encounter_id,
        "timestamp": evt.timestamp,
        "heart_rate": evt.heart_rate,
    }
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    run_forever(handler)
