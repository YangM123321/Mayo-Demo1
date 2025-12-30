from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str
    topic_in: str
    topic_dlq: str
    group_id: str
    enable_idempotency_producer: bool
    max_retries: int

    @staticmethod
    def from_env() -> "KafkaConfig":
        return KafkaConfig(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic_in=os.getenv("KAFKA_TOPIC_IN", "vitals.in"),
            topic_dlq=os.getenv("KAFKA_TOPIC_DLQ", "vitals.dlq"),
            group_id=os.getenv("KAFKA_GROUP_ID", "mayo-demo1-vitals-consumer"),
            enable_idempotency_producer=os.getenv("KAFKA_IDEMPOTENT_PRODUCER", "1").strip() in {"1", "true", "yes"},
            max_retries=int(os.getenv("KAFKA_MAX_RETRIES", "5")),
        )
