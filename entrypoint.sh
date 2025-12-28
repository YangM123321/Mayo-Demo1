

#!/bin/sh
set -eu

PORT="${PORT:-8080}"
MODE="${MODE:-api}"

echo "Starting MODE=$MODE on PORT=$PORT"

if [ "$MODE" = "api" ]; then
  exec uvicorn src.service:app --host 0.0.0.0 --port "$PORT"
elif [ "$MODE" = "mlflow" ]; then
  exec uvicorn src.app_mlflow:app --host 0.0.0.0 --port "$PORT"
  #exec uvicorn src.service:app --host 0.0.0.0 --port "${PORT:-8080}"

else
  echo "Unknown MODE=$MODE" >&2
  exit 2
fi
