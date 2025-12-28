import json

from src.common.logging import get_logger
from src.observability.metrics import KAFKA_MESSAGES_TOTAL

log = get_logger("kafka_consumer")

def handle_message(topic: str, raw: bytes):
    try:
        event = json.loads(raw.decode("utf-8"))
        KAFKA_MESSAGES_TOTAL.labels(topic=topic, direction="consumed").inc()
        log.info("kafka_consume_success", topic=topic, patient_id=event.get("patient_id"))
        return event
    except Exception as e:
        log.exception("kafka_consume_failed", topic=topic, error_type=type(e).__name__)
        raise
