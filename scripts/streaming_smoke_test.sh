#!/usr/bin/env bash
set -euo pipefail

export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export KAFKA_TOPIC_IN="vitals.in"
export KAFKA_TOPIC_DLQ="vitals.dlq"
export KAFKA_GROUP_ID="ci-consumer"
export IDEMPOTENCY_DB="/tmp/seen_events_ci.db"

# create topics (redpanda supports kafka api; easiest is rpk but not installed)
# We'll use Python producer to implicitly create topics (works if auto-create enabled).
python -c "from confluent_kafka import Producer; p=Producer({'bootstrap.servers':'localhost:9092'}); p.produce('vitals.in', b'{}'); p.produce('vitals.dlq', b'{}'); p.flush(5)"

# start consumer in background
python -m src.streaming.main &
CONS_PID=$!
sleep 2

# send valid event
python - <<'PY'
import json
from datetime import datetime, timezone
from confluent_kafka import Producer
p=Producer({'bootstrap.servers':'localhost:9092'})
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

# send invalid event (missing patient_id)
python - <<'PY'
from confluent_kafka import Producer
p=Producer({'bootstrap.servers':'localhost:9092'})
p.produce("vitals.in", b'{"bad":"payload"}')
p.flush(5)
PY

sleep 3
kill $CONS_PID || true

# verify audit exists (valid processed)
test -f /tmp/vitals_audit.log
grep -q '"event_id": "ci-evt-1"' /tmp/vitals_audit.log

echo "✅ streaming smoke test passed"
