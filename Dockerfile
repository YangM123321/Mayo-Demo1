# ---- Dockerfile (multi-mode) ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (helpful for pandas/pyarrow/sklearn/mlflow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Fix mlflow/pkg_resources
RUN pip install --upgrade pip setuptools wheel

# Install Python deps first (better layer cache)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project (includes entrypoint.sh)
COPY . .

# Make sure entrypoint is executable and has Linux line endings
# RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh
RUN sed -i '1s/^\xEF\xBB\xBF//' /app/entrypoint.sh \
 && sed -i 's/\r$//' /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh


EXPOSE 8000 8888
ENV MODE=api SCRIPT=""
ENTRYPOINT ["/app/entrypoint.sh"]


FROM python:3.12-slim

WORKDIR /app

# If you need system deps for pandas/pyarrow, add them here
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY out/features_matrix.parquet ./out/features_matrix.parquet

ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8000"]


