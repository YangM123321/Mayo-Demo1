# Failure Scenarios (Platform Readiness)

## 1) Kafka down
**Goal:** confirm retries + clear error logs
- Stop broker
- Run producer/consumer
**Expected:**
- logs: `kafka_*_failed` with error_type
- API stays healthy (if decoupled)
- alert/dashboard shows message rate drop

## 2) Bad payload (schema violation)
Send event missing required fields.
**Expected:**
- structured log: validation_failed
- metrics: error counter increments (optional)
- message routed to DLQ or discarded with explicit reason

## 3) API timeout
Simulate a slow endpoint.
**Expected:**
- latency histogram increases
- p95 dashboard shows spike
