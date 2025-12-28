# Adding a New Data Source

## 1) Input contract (required)
Define your event schema in `src/contracts/` using Pydantic.

Example:
- `src/contracts/vitals.py` -> `VitalEvent`

## 2) Topic / ingestion
- Define topic name via env var: `KAFKA_TOPIC_<NAME>`
- Producer lives in: `src/streaming/producer.py`
- Consumer handler lives in: `src/streaming/consumer.py`

## 3) Validation
All incoming messages must validate against a contract.
Pattern:
1) parse JSON
2) `ContractModel.model_validate(payload)`
3) if fails -> log structured error + (optionally) send to DLQ

## 4) Feature generation
Transform layer lives in:
- `src/fhir/transform.py` (or `src/features/` if you add it)

## 5) Orchestration (Airflow)
Add a task to your DAG in `dags/`:
- ingestion -> validation -> transform -> load -> train/eval

## 6) Observability checklist
Every new source must add:
- structured logs: `source`, `patient_id`, `stage`
- metrics: messages produced/consumed counters
- dashboard: panel or label for new topic
