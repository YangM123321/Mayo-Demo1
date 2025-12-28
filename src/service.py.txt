# src/service.py
from __future__ import annotations

import json
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from google.cloud import storage
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from starlette.requests import Request

print("[BOOT] importing src.service", file=sys.stderr, flush=True)

# -----------------------------
# Paths (inside container /app)
# -----------------------------
BASE = Path("/app")
OUT = BASE / "out"
MODELS = BASE / "models"

MODEL_PATH = MODELS / "admit_lr.joblib"
FEATURES_PATH = MODELS / "feature_list.json"

# Default bundled location (if you bake the parquet into the image)
DEFAULT_FEATURES_TABLE = OUT / "features_matrix.parquet"
FEATURES_TABLE = None  # will be loaded during startup


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    # gs://bucket/path/to/blob
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    no_scheme = uri[5:]
    bucket, _, blob = no_scheme.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return bucket, blob


def _ensure_features_local() -> Path:
    """
    Decide where the features parquet is and ensure it exists locally.

    Priority:
      1) FEATURES_URI=gs://...  -> download to /tmp/features_matrix.parquet
      2) FEATURES_URI=/some/local/path -> use that
      3) default bundled file: /app/out/features_matrix.parquet
    """
    features_uri = os.getenv("FEATURES_URI", "").strip()

    # 3) default bundled
    if not features_uri:
        return DEFAULT_FEATURES_TABLE

    # 2) local path
    if features_uri.startswith("/"):
        return Path(features_uri)

    # 1) gs:// download
    if features_uri.startswith("gs://"):
        dst = Path("/tmp/features_matrix.parquet")
        dst.parent.mkdir(parents=True, exist_ok=True)

        # download only if missing/empty
        if (not dst.exists()) or dst.stat().st_size == 0:
            bucket_name, blob_name = _parse_gs_uri(features_uri)

            proj = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
            print(
                f"[BOOT] GOOGLE_CLOUD_PROJECT={os.getenv('GOOGLE_CLOUD_PROJECT')} GCLOUD_PROJECT={os.getenv('GCLOUD_PROJECT')}",
                file=sys.stderr,
                flush=True,
            )
            print("[BOOT] service.py VERSION=2025-12-27", file=sys.stderr, flush=True)

            client = storage.Client(project=proj)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(dst))
            print(f"[BOOT] downloaded FEATURES_URI -> {dst}", file=sys.stderr, flush=True)

        return dst

    # Unknown scheme -> fallback
    return DEFAULT_FEATURES_TABLE


# -----------------------------
# Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = None

    skip = os.getenv("SKIP_MODEL_LOAD", "").strip().lower() in {"1", "true", "yes"}

    if skip:

        class _Dummy:
            def predict_proba(self, X):
                import numpy as np

                n = len(X)
                return np.c_[np.zeros(n), np.zeros(n)]  # shape (n,2)

        app.state.model_bundle = {
            "model": _Dummy(),
            "features": [],
            "features_df": None,
        }
        print("[BOOT] SKIP_MODEL_LOAD=1 -> dummy model bundle", file=sys.stderr, flush=True)
        yield
        return

    # Resolve features table location (may download from GCS)
    features_table = _ensure_features_local()

    # Validate artifacts
    if not MODEL_PATH.exists():
        raise RuntimeError(f"MODEL missing: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise RuntimeError(f"FEATURE LIST missing: {FEATURES_PATH}")
    if not features_table.exists():
        raise RuntimeError(f"FEATURES TABLE missing: {features_table}")

    # Load
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEATURES_PATH.read_text())
    features_df = pd.read_parquet(features_table)

    app.state.model_bundle = {
        "model": model,
        "features": features,
        "features_df": features_df,
        "features_table": str(features_table),
    }
    print(f"[BOOT] model + features loaded (table={features_table})", file=sys.stderr, flush=True)

    yield


app = FastAPI(title="Mayo Demo API", lifespan=lifespan)


# -----------------------------
# Error handler (single)
# -----------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print(f"[ERROR] {request.method} {request.url} -> {exc}", file=sys.stderr, flush=True)
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": "internal_error"})


# -----------------------------
# Schemas
# -----------------------------
class AdmitReq(BaseModel):
    patient_id: str
    encounter_id: str


class AdmitResp(BaseModel):
    patient_id: str
    encounter_id: str
    probability: float
    label: int


# -----------------------------
# Helpers
# -----------------------------
def _bundle():
    b = getattr(app.state, "model_bundle", None)
    if b is not None:
        return b

    if os.getenv("SKIP_MODEL_LOAD", "").strip().lower() in {"1", "true", "yes"}:

        class _Dummy:
            def predict_proba(self, X):
                import numpy as np

                n = len(X)
                return np.c_[np.zeros(n), np.zeros(n)]

        b = {"model": _Dummy(), "features": [], "features_df": None}
        app.state.model_bundle = b
        return b

    raise HTTPException(503, "Model not initialized")


def _row_for(pid: str, enc: str, feat_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    row = feat_df[(feat_df["patient_id"] == pid) & (feat_df["encounter_id"] == enc)]
    if row.empty:
        raise HTTPException(404, f"No features for patient_id={pid} encounter_id={enc}")
    return row[features] if features else row.iloc[:, 0:0]


def _predict_one(pid: str, enc: str) -> Dict[str, Any]:
    b = _bundle()
    model = b["model"]
    features: List[str] = b["features"]
    feat_df: Optional[pd.DataFrame] = b["features_df"]

    if model is None:
        raise HTTPException(503, "Model not available")

    if feat_df is None:
        # Dummy mode
        feat_df = pd.DataFrame([{"patient_id": pid, "encounter_id": enc}])

    X = _row_for(pid, enc, feat_df=feat_df, features=features)
    try:
        p = float(model.predict_proba(X)[:, 1][0])
    except Exception:
        p = 0.0
    return {"patient_id": pid, "encounter_id": enc, "probability": p, "label": int(p >= 0.5)}


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    _ = _bundle()
    return {"status": "healthy", "title": app.title}


@app.get("/_routes")
def list_routes():
    return sorted([getattr(r, "path", "") for r in app.routes])





@app.get("/readyz")
def readyz():
    _ = _bundle()
    return {"ok": True}


@app.post("/predict/admission/batch")
def predict_admission_batch(items: List[AdmitReq]):
    if not items:
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        try:
            out.append(_predict_one(it.patient_id, it.encounter_id))
        except HTTPException as e:
            out.append(
                {
                    "patient_id": it.patient_id,
                    "encounter_id": it.encounter_id,
                    "error": str(e.detail),
                }
            )
        except Exception as e:
            out.append(
                {"patient_id": it.patient_id, "encounter_id": it.encounter_id, "error": str(e)}
            )
    return out


@app.get("/predict/admission/{patient_id}/{encounter_id}", response_model=AdmitResp)
def predict_admission(patient_id: str, encounter_id: str):
    return _predict_one(patient_id, encounter_id)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}



    