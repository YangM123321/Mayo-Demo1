from __future__ import annotations

import json

from kafka import KafkaProducer

from src.common.logging import get_logger
from src.contracts.vitals import VitalEvent
from src.observability.metrics import KAFKA_MESSAGES_TOTAL

log = get_logger("kafka_producer")


def publish_vital(
    producer: KafkaProducer,
    topic: str,
    patient_id: str,
    encounter_id: str,
    timestamp: str,
    heart_rate: int | None = None,
    spo2: int | None = None,
) -> None:
    event = VitalEvent(
        patient_id=patient_id,
        encounter_id=encounter_id,
        timestamp=timestamp,
        heart_rate=heart_rate,
        spo2=spo2,
    )

    payload = json.dumps(event.model_dump()).encode("utf-8")
    producer.send(topic, payload)
    KAFKA_MESSAGES_TOTAL.labels(direction="produced", topic=topic).inc()
    log.info("published vital", topic=topic, patient_id=patient_id, encounter_id=encounter_id)
