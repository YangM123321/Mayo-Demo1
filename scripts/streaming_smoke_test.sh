#!/usr/bin/env bash
set -euo pipefail

export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export KAFKA_TOPIC_IN="vitals.in"
export KAFKA_TOPIC_DLQ="vitals.dlq"
export KAFKA_GROUP_ID="ci-consumer"
export IDEMPOTENCY_DB="/tmp/seen_events_ci.db"
export VITALS_AUDIT_PATH="/tmp/vitals_audit.log"

rm -f /tmp/vitals_audit.log /tmp/seen_events_ci.db || true

# Wait for broker
python - <<'PY'
import time
from confluent_kafka import Producer
p = Producer({"bootstrap.servers": "localhost:9092"})
for i in range(30):
    try:
        p.produce("vitals.in", b"{}")
        p.flush(2)
        print("broker_ready")
        raise SystemExit(0)
    except Exception:
        time.sleep(1)
raise SystemExit("broker_not_ready")
PY

# start consumer in background
python -m src.streaming.main &
CONS_PID=$!
sleep 2

# send valid event
python - <<'PY'
import json
from datetime import datetime, timezone
from confluent_kafka import Producer
p=Producer({"bootstrap.servers":"localhost:9092"})
evt={
  "meta":{"event_id":"ci-evt-1","schema":"vitals.v1","schema_version":1,"produced_at":datetime.now(timezone.utc).isoformat()},
  "patient_id":"p1",
  "encounter_id":"e1",
  "timestamp":datetime.now(timezone.utc).isoformat(),
  "heart_rate":80
}
p.produce("vitals.in", json.dumps(evt).encode("utf-8"))
p.flush(5)
PY

# send invalid event
python - <<'PY'
from confluent_kafka import Producer
p=Producer({"bootstrap.servers":"localhost:9092"})
p.produce("vitals.in", b'{"bad":"payload"}')
p.flush(5)
PY

sleep 3

# stop consumer (if still alive)
kill "$CONS_PID" 2>/dev/null || true
sleep 1

# verify audit exists (valid processed)
test -f /tmp/vitals_audit.log
grep -q '"event_id": "ci-evt-1"' /tmp/vitals_audit.log

echo "✅ streaming smoke test passed"
