# src/service.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import json, joblib, pandas as pd



import sys
print("[BOOT] importing src.service", file=sys.stderr, flush=True)

from fastapi import FastAPI
app = FastAPI(title="Mayo Demo API (service.py)")


BASE   = Path(__file__).resolve().parents[1]
OUT    = BASE / "out"
MODELS = BASE / "models"

MODEL_PATH     = MODELS / "admit_lr.joblib"
FEATURES_PATH  = MODELS / "feature_list.json"
FEATURES_TABLE = OUT / "features_matrix.parquet"

MODEL    = joblib.load(MODEL_PATH)
FEATURES = json.loads(FEATURES_PATH.read_text())

class AdmitReq(BaseModel):
    patient_id: str
    encounter_id: str

class AdmitResp(BaseModel):
    patient_id: str
    encounter_id: str
    probability: float
    label: int

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/_whoami")
def whoami():
    return {"module": "src.service", "title": app.title}

@app.get("/_routes")
def list_routes():
    return sorted([getattr(r, "path", "") for r in app.routes])

def _load_features_df() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_TABLE)

def row_for(pid: str, enc: str, feat_df: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    if feat_df is None:
        feat_df = _load_features_df()
    row = feat_df[(feat_df["patient_id"] == pid) & (feat_df["encounter_id"] == enc)]
    if row.empty:
        raise HTTPException(404, f"No features for patient_id={pid} encounter_id={enc}")
    return row[FEATURES]

def predict_one(pid: str, enc: str, feat_df: Optional[pd.DataFrame]=None) -> Dict[str, Any]:
    X = row_for(pid, enc, feat_df=feat_df)
    p = float(MODEL.predict_proba(X)[:, 1][0])
    return {"patient_id": pid, "encounter_id": enc, "probability": p, "label": int(p >= 0.5)}

# --- Register BATCH FIRST (extra safe) ---
@app.post("/predict/admission/batch")
def predict_admission_batch(items: List[AdmitReq]):
    if not items:
        return []
    feat_df = _load_features_df()
    out: List[Dict[str, Any]] = []
    for it in items:
        try:
            out.append(predict_one(it.patient_id, it.encounter_id, feat_df=feat_df))
        except HTTPException as e:
            out.append({"patient_id": it.patient_id, "encounter_id": it.encounter_id, "error": str(e.detail)})
        except Exception as e:
            out.append({"patient_id": it.patient_id, "encounter_id": it.encounter_id, "error": str(e)})
    return out

# --- Then the dynamic single GET ---
@app.get("/predict/admission/{patient_id}/{encounter_id}", response_model=AdmitResp)
def predict_admission(patient_id: str, encounter_id: str):
    return predict_one(patient_id, encounter_id)

@app.get("/_whoami")
def whoami():
    return {"module": "src.service", "title": app.title}

@app.get("/_routes")
def list_routes():
    return sorted([getattr(r, "path", "") for r in app.routes])
# fresh, unique tag
$TAG = (Get-Date -Format "yyyyMMdd-HHmmss") + "-forcecmd"

# build
docker build -f Dockerfile.api `
  -t "us-central1-docker.pkg.dev/innate-mix-432320-h1/docker-repo/mayo-api:$TAG" .

# push
docker push "us-central1-docker.pkg.dev/innate-mix-432320-h1/docker-repo/mayo-api:$TAG"

# deploy with explicit command + args (PowerShell-friendly)
gcloud run deploy mayo-api `
  --image "us-central1-docker.pkg.dev/innate-mix-432320-h1/docker-repo/mayo-api:$TAG" `
  --region us-central1 `
  --platform managed `
  --allow-unauthenticated `
  --command uvicorn `
  --args "src.service:app,--host,0.0.0.0,--port,8080"

