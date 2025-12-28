from prometheus_client import Counter, Gauge, Histogram

# API
API_REQUEST_LATENCY = Histogram(
    "mayo_api_request_latency_seconds",
    "Latency of API requests (seconds)",
    ["route", "method", "status_code"],
)

API_ERRORS_TOTAL = Counter(
    "mayo_api_errors_total",
    "Total API errors",
    ["route", "method", "error_type"],
)

# Kafka / streaming
KAFKA_MESSAGES_TOTAL = Counter(
    "mayo_kafka_messages_total",
    "Kafka messages processed",
    ["topic", "direction"],  # direction: produced|consumed
)

KAFKA_CONSUMER_LAG = Gauge(
    "mayo_kafka_consumer_lag",
    "Approx consumer lag (if you can compute it)",
    ["topic", "consumer_group"],
)

# Airflow (optional push; also can be scraped from airflow exporter if you add one later)
AIRFLOW_TASK_FAILURES = Counter(
    "mayo_airflow_task_failures_total",
    "Airflow task failures (if emitted manually)",
    ["dag_id", "task_id"],
)
