# streaming/consumer.py
import os, json
from confluent_kafka import Consumer

BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
TOPIC  = os.getenv("KAFKA_TOPIC", "vitals")

c = Consumer({
    "bootstrap.servers": BROKER,
    "group.id": "vitals-consumer-2",   # new group reads from start
    "auto.offset.reset": "earliest"    # read backlog if no committed offsets
})

c.subscribe([TOPIC])
print(f"Consuming from {TOPIC} on {BROKER}...\n")

try:
    while True:
        msg = c.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print("❌", msg.error()); continue
        print(json.loads(msg.value().decode("utf-8")))
except KeyboardInterrupt:
    pass
finally:
    c.close()
