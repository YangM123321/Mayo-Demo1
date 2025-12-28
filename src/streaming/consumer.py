from __future__ import annotations

import json

from kafka import KafkaConsumer, KafkaProducer
from pydantic import ValidationError

from src.common.logging import get_logger
from src.contracts.vitals import VitalEvent
from src.observability.metrics import KAFKA_MESSAGES_TOTAL

log = get_logger("kafka_consumer")


def consume_vitals(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    dlq_topic: str,
) -> None:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    dlq_producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

    for msg in consumer:
        try:
            obj = json.loads(msg.value.decode("utf-8"))
            event = VitalEvent(**obj)  # ✅ validation here
        except (json.JSONDecodeError, ValidationError):
            dlq_producer.send(dlq_topic, msg.value)
            KAFKA_MESSAGES_TOTAL.labels(direction="dlq", topic=dlq_topic).inc()
            log.warning("invalid message sent to dlq", topic=topic, dlq_topic=dlq_topic)
            continue

        KAFKA_MESSAGES_TOTAL.labels(direction="consumed", topic=topic).inc()
        log.info(
            "consumed vital",
            patient_id=event.patient_id,
            encounter_id=event.encounter_id,
            timestamp=str(event.timestamp),
        )
