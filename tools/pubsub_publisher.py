import json, os, time, uuid, random
from datetime import datetime, timezone
from google.cloud import pubsub_v1

PROJECT_ID = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
TOPIC_ID   = os.getenv("PUBSUB_TOPIC", "vitals")

if not PROJECT_ID:
    raise SystemExit("Set GCP_PROJECT or GOOGLE_CLOUD_PROJECT")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

def make_msg():
    return {
        "event_id": str(uuid.uuid4()),
        "patient_id": random.randint(1000, 1020),
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "BP_SYS": random.randint(95, 165),
        "BP_DIA": random.randint(55, 105),
    }

print(f"Publishing to {topic_path} ... Ctrl+C to stop.")
try:
    while True:
        payload = json.dumps(make_msg()).encode("utf-8")
        future = publisher.publish(topic_path, payload, content_type="application/json")
        future.result(timeout=30)   # ensure backpressure/raise if errors
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nStopped.")
