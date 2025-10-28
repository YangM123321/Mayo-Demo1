# streaming/producer.py
import json, time, random, uuid, os, sys
from datetime import datetime, timezone
from confluent_kafka import Producer

BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
TOPIC  = os.getenv("KAFKA_TOPIC", "vitals")

p = Producer({"bootstrap.servers": BROKER})

def msg():
    return {
        "event_id": str(uuid.uuid4()),
        "patient_id": random.randint(1000, 1020),
        "ts": datetime.now(timezone.utc).isoformat(),  # no deprecation
        "BP_SYS": random.randint(95, 165),
        "BP_DIA": random.randint(55, 105),
    }

def delivered(err, rec):
    if err:
        print("❌ delivery failed:", err, file=sys.stderr)

try:
    while True:
        p.produce(TOPIC, json.dumps(msg()).encode("utf-8"), callback=delivered)
        p.poll(0)            # serve callbacks
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    p.flush(5)              # ✅ drain before exit

