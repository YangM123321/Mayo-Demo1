# Observability

## Run locally
1) Start API on port 8080
2) Start Prometheus + Grafana:
   docker compose -f docker-compose.observability.yml up -d

## URLs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (anonymous enabled)

## Evidence
Save screenshots here:
- docs/observability/grafana_dashboard.png
- docs/observability/prometheus_targets.png
