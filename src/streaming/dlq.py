from __future__ import annotations

import json
import time
from typing import Any, Dict

from confluent_kafka import Producer

from src.streaming.config import KafkaConfig


def _producer(conf: KafkaConfig) -> Producer:
    pconf = {
        "bootstrap.servers": conf.bootstrap_servers,
        # idempotent producer is nice for DLQ too
        "enable.idempotence": conf.enable_idempotency_producer,
        "acks": "all",
        "linger.ms": 5,
    }
    return Producer(pconf)


def publish_dlq(conf: KafkaConfig, *, raw: str, reason: str, extra: Dict[str, Any] | None = None) -> None:
    payload = {
        "raw": raw,
        "reason": reason,
        "extra": extra or {},
        "dlq_ts": int(time.time()),
    }
    p = _producer(conf)
    p.produce(conf.topic_dlq, value=json.dumps(payload).encode("utf-8"))
    p.flush(10)
