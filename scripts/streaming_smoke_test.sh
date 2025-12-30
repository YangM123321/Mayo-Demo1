#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export KAFKA_TOPIC_IN="vitals.in"
export KAFKA_TOPIC_DLQ="vitals.dlq"
export KAFKA_GROUP_ID="ci-consumer-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}"
export IDEMPOTENCY_DB="/tmp/seen_events_ci.db"
export VITALS_AUDIT_PATH="/tmp/vitals_audit.log"

rm -f /tmp/vitals_audit.log /tmp/seen_events_ci.db /tmp/consumer.log || true

fail() {
  echo "❌ streaming smoke test failed"

  echo "---- consumer log file stats ----"
  ls -l /tmp/consumer.log || true
  wc -c /tmp/consumer.log 2>/dev/null || true

  echo "---- consumer log (tail) ----"
  tail -n 300 /tmp/consumer.log 2>/dev/null || true

  echo "---- VitalEvent json schema ----"
  python - <<'PY' || true
from src.streaming.schemas import VitalEvent
try:
    print(VitalEvent.model_json_schema())
except Exception as e:
    print("schema_dump_failed:", e)
PY

  echo "---- DLQ sample (vitals.dlq) ----"
  python - <<'PY' || true
import os, time
from confluent_kafka import Consumer

c = Consumer({
    "bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"],
    "group.id": "ci-dlq-debug",
    "auto.offset.reset": "earliest",
})
c.subscribe([os.environ["KAFKA_TOPIC_DLQ"]])

msgs = []
deadline = time.time() + 8
while time.time() < deadline and len(msgs) < 5:
    m = c.poll(1.0)
    if m is None:
        continue
    if m.error():
        continue
    msgs.append(m.value().decode("utf-8", errors="replace"))

c.close()
print("DLQ_messages_count=", len(msgs))
for i, v in enumerate(msgs, 1):
    print(f"--- dlq msg {i} ---")
    print(v)
PY

  echo "---- redpanda logs ----"
  docker compose -f docker-compose.kafka.yml logs --no-color redpanda || true

  echo "---- audit file ----"
  ls -l /tmp/vitals_audit.log || true
  tail -n 50 /tmp/vitals_audit.log 2>/dev/null || true

  # don't leave a running consumer
  kill "${CONS_PID:-0}" 2>/dev/null || true

  exit 1
}

# Wait for broker (metadata, no produce)
python - <<'PY'
import os, time
from confluent_kafka.admin import AdminClient

bs = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
a = AdminClient({"bootstrap.servers": bs})
for i in range(90):
    try:
        md = a.list_topics(timeout=2)
        if md.brokers:
            print("broker_ready")
            raise SystemExit(0)
    except Exception:
        time.sleep(1)
raise SystemExit("broker_not_ready")
PY

# Start consumer in background (capture logs, unbuffered!)
python -u -m src.streaming.main > /tmp/consumer.log 2>&1 &
CONS_PID=$!
sleep 2

# Send "valid" event (your best guess)
python - <<'PY'
import os, json, time
from datetime import datetime, timezone
from confluent_kafka import Producer

bs = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
topic = os.environ["KAFKA_TOPIC_IN"]
p = Producer({"bootstrap.servers": bs})

evt = {
  "meta": {
    "event_id": "ci-evt-1",
    "schema": "vitals.v1",
    "schema_version": 1,
    "produced_at": datetime.now(timezone.utc).isoformat(),
  },
  "patient_id": "p1",
  "encounter_id": "e1",
  "timestamp": datetime.now(timezone.utc).isoformat(),
  "heart_rate": 80,
}

payload = json.dumps(evt).encode("utf-8")
p.produce(topic, payload)
p.flush(10)

# send twice in case consumer is still joining
time.sleep(1)
p.produce(topic, payload)
p.flush(10)
PY

# Send invalid event (should go to DLQ)
python - <<'PY'
import os
from confluent_kafka import Producer

bs = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
topic = os.environ["KAFKA_TOPIC_IN"]
p = Producer({"bootstrap.servers": bs})
p.produce(topic, b'{"bad":"payload"}')
p.flush(10)
PY

# Wait up to 40s for audit record; if missing, dump logs
for i in {1..40}; do
  if test -f /tmp/vitals_audit.log && grep -q '"event_id": "ci-evt-1"' /tmp/vitals_audit.log; then
    echo "✅ audit record found"
    break
  fi
  if ! kill -0 "$CONS_PID" 2>/dev/null; then
    fail
  fi
  sleep 1
done

test -f /tmp/vitals_audit.log || fail
grep -q '"event_id": "ci-evt-1"' /tmp/vitals_audit.log || fail

kill "$CONS_PID" 2>/dev/null || true
sleep 1

echo "✅ streaming smoke test passed"
