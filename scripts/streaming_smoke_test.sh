#!/usr/bin/env bash
set -euo pipefail

export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export KAFKA_TOPIC_IN="vitals.in"
export KAFKA_TOPIC_DLQ="vitals.dlq"

export KAFKA_GROUP_ID="ci-consumer-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}"


export IDEMPOTENCY_DB="/tmp/seen_events_ci.db"
export VITALS_AUDIT_PATH="/tmp/vitals_audit.log"

rm -f /tmp/vitals_audit.log /tmp/seen_events_ci.db || true

fail() {
  echo "❌ streaming smoke test failed"
  echo "---- docker compose ps ----"
  docker compose -f docker-compose.kafka.yml ps || true
  echo "---- redpanda logs ----"
  docker compose -f docker-compose.kafka.yml logs --no-color redpanda || true
  echo "---- consumer alive? ----"
  if kill -0 "${CONS_PID:-0}" 2>/dev/null; then
    echo "consumer still running (pid=$CONS_PID)"
  else
    echo "consumer NOT running"
  fi
  echo "---- audit file (if any) ----"
  ls -l /tmp/vitals_audit.log || true
  tail -n 50 /tmp/vitals_audit.log 2>/dev/null || true
  exit 1
}

# Wait for broker
python - <<'PY'
import os, time
from confluent_kafka import Producer

p = Producer({"bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"]})
for i in range(60):
    try:
        p.produce(os.environ.get("KAFKA_TOPIC_IN","vitals.in"), b"{}")
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
import os, json, time
from datetime import datetime, timezone
from confluent_kafka import Producer

p = Producer({"bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"]})
topic = os.environ["KAFKA_TOPIC_IN"]

def send():
    evt = {
      "meta":{"event_id":"ci-evt-1","schema":"vitals.v1","schema_version":1,"produced_at":datetime.now(timezone.utc).isoformat()},
      "patient_id":"p1",
      "encounter_id":"e1",
      "timestamp":datetime.now(timezone.utc).isoformat(),
      "heart_rate":80
    }
    p.produce(topic, json.dumps(evt).encode("utf-8"))
    p.flush(5)

# send once, wait, send again (helps if consumer not ready yet)
send()
time.sleep(2)
send()
PY

# send invalid event
python - <<'PY'
import os
from confluent_kafka import Producer

p = Producer({"bootstrap.servers": os.environ["KAFKA_BOOTSTRAP_SERVERS"]})
topic = os.environ["KAFKA_TOPIC_IN"]
p.produce(topic, b'{"bad":"payload"}')
p.flush(5)
PY


# wait up to 30s for consumer to write audit
for i in {1..30}; do
  if test -f /tmp/vitals_audit.log && grep -q '"event_id": "ci-evt-1"' /tmp/vitals_audit.log; then
    echo "✅ audit record found"
    break
  fi
  # if consumer died, fail fast with diagnostics
  if ! kill -0 "$CONS_PID" 2>/dev/null; then
    fail
  fi
  sleep 1
done

# final assertion (and diagnostics if missing)
test -f /tmp/vitals_audit.log || fail
grep -q '"event_id": "ci-evt-1"' /tmp/vitals_audit.log || fail

# stop consumer
kill "$CONS_PID" 2>/dev/null || true
sleep 1

echo "✅ streaming smoke test passed"
