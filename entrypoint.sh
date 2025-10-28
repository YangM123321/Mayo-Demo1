#!/bin/sh
set -e

echo "[entrypoint] MODE=${MODE}"

case "${MODE}" in
  api)
    echo "[entrypoint] Starting FastAPI (uvicorn)..."
    exec uvicorn src.app:main --host 0.0.0.0 --port 8000
    ;;
  jupyter)
    echo "[entrypoint] Starting JupyterLab on 0.0.0.0:8888..."
    exec jupyter lab \
      --ServerApp.ip=0.0.0.0 \
      --ServerApp.port=8888 \
      --ServerApp.open_browser=False \
      --ServerApp.token='' \
      --ServerApp.allow_origin='*' \
      --ServerApp.allow_root=True
    ;;
  script)
    if [ -z "${SCRIPT}" ]; then
      echo "[entrypoint] ERROR: SCRIPT env var is empty (e.g., SCRIPT=src/etl_pipeline.py)." >&2
      exit 1
    fi
    echo "[entrypoint] Running script: ${SCRIPT}"
    exec python "${SCRIPT}"
    ;;
  shell)
    echo "[entrypoint] Opening shell..."
    exec sh
    ;;
  *)
    echo "[entrypoint] Unknown MODE '${MODE}'. Use one of: api | jupyter | script | shell" >&2
    exit 2
    ;;
esac