import json

from src.common.logging import get_logger
from src.observability.metrics import KAFKA_MESSAGES_TOTAL

log = get_logger("kafka_producer")

def produce_vital_event(topic: str, event: dict):
    # Replace with your real Kafka client code
    payload = json.dumps(event).encode("utf-8")

    log.info("kafka_produce_attempt", topic=topic, bytes=len(payload), patient_id=event.get("patient_id"))
    # kafka_client.produce(topic, payload)  # your real call here
    # kafka_client.flush()

    KAFKA_MESSAGES_TOTAL.labels(topic=topic, direction="produced").inc()
    log.info("kafka_produce_success", topic=topic, patient_id=event.get("patient_id"))
