# streaming/producer.py
from __future__ import annotations

import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone

from confluent_kafka import Producer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS") or os.getenv("KAFKA_BROKER") or "localhost:9092"
TOPIC = os.getenv("KAFKA_TOPIC", "vitals.in")

p = Producer({"bootstrap.servers": BOOTSTRAP})

# ... rest unchanged


def msg():
    return {
        "event_id": str(uuid.uuid4()),
        "patient_id": random.randint(1000, 1020),
        "ts": datetime.now(timezone.utc).isoformat(),
        "BP_SYS": random.randint(95, 165),
        "BP_DIA": random.randint(55, 105),
    }


def delivered(err, rec):
    if err:
        print("❌ delivery failed:", err, file=sys.stderr)


try:
    while True:
        p.produce(TOPIC, json.dumps(msg()).encode("utf-8"), callback=delivered)
        p.poll(0)
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    p.flush(5)
