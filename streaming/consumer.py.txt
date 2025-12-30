from __future__ import annotations

import logging
import os
from typing import Callable

from confluent_kafka import Consumer
from tenacity import retry, stop_after_attempt, wait_exponential

from src.streaming.config import KafkaConfig
from src.streaming.dlq import publish_dlq
from src.streaming.idempotency import mark_seen, seen
from src.streaming.schemas import VitalEvent

logger = logging.getLogger("mayo.streaming")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _consumer(conf: KafkaConfig) -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": conf.bootstrap_servers,
            "group.id": conf.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,  # commit only after success
            "max.poll.interval.ms": 600000,
        }
    )


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
def _process_with_retry(handler: Callable[[VitalEvent], None], evt: VitalEvent) -> None:
    handler(evt)


def run_forever(handler: Callable[[VitalEvent], None]) -> None:
    conf = KafkaConfig.from_env()
    c = _consumer(conf)
    c.subscribe([conf.topic_in])

    logger.info("consumer_start", extra={"topic": conf.topic_in, "group": conf.group_id})

    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("kafka_error", extra={"err": str(msg.error())})
                continue

            raw = msg.value().decode("utf-8", errors="replace")

            # schema validation
            try:
                evt = VitalEvent.parse_json(raw)
            except Exception as e:
                publish_dlq(conf, raw=raw, reason=str(e), extra={"stage": "schema"})
                c.commit(message=msg, asynchronous=False)
                continue

            # idempotency
            if seen(evt.meta.event_id):
                logger.info("duplicate_event_skip", extra={"event_id": evt.meta.event_id})
                c.commit(message=msg, asynchronous=False)
                continue

            # processing + retries
            try:
                _process_with_retry(handler, evt)
                mark_seen(evt.meta.event_id)
                c.commit(message=msg, asynchronous=False)
            except Exception as e:
                publish_dlq(
                    conf,
                    raw=raw,
                    reason=f"processing_failed: {e}",
                    extra={"stage": "processing", "event_id": evt.meta.event_id},
                )
                c.commit(message=msg, asynchronous=False)
    finally:
        c.close()
