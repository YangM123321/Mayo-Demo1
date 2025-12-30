#!/bin/sh
set -eu

PORT="${PORT:-8080}"
MODE="${MODE:-api}"

echo "Starting MODE=$MODE on PORT=$PORT"

if [ "$MODE" = "api" ]; then
  exec uvicorn src.service:app --host 0.0.0.0 --port "$PORT"
elif [ "$MODE" = "mlflow" ]; then
  exec uvicorn src.app_mlflow:app --host 0.0.0.0 --port "$PORT"
else
  echo "Unknown MODE=$MODE"
  exit 2
fi
