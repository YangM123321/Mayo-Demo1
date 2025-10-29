# src/service.py
from __future__ import annotations

import json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

print("[BOOT] importing src.service", file=sys.stderr, flush=True)

# --- Paths
BASE   = Path(__file__).resolve().parents[1]
OUT    = BASE / "out"
MODELS = BASE / "models"
MODEL_PATH     = MODELS / "admit_lr.joblib"
FEATURES_PATH  = MODELS / "feature_list.json"
FEATURES_TABLE = OUT / "features_matrix.parquet"

# --- Lifespan: initialize shared state safely (works in prod + CI)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Always create the attribute so routes/tests never crash
    app.state.model_bundle = None

    skip = os.getenv("SKIP_MODEL_LOAD", "").strip() in {"1", "true", "True"}
    if skip:
        class _Dummy:
            def predict_proba(self, X):
                import numpy as np
                n = len(X)
                return np.c_[np.zeros(n), np.zeros(n)]  # shape (n,2)

        app.state.model_bundle = {
            "model": _Dummy(),
            "features": [],          # no real features in CI
            "features_df": None,     # synthesize row later
        }
        print("[BOOT] SKIP_MODEL_LOAD=1 -> using dummy model bundle", file=sys.stderr, flush=True)
    else:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"MODEL missing: {MODEL_PATH}")
        if not FEATURES_PATH.exists():
            raise RuntimeError(f"FEATURE LIST missing: {FEATURES_PATH}")
        if not FEATURES_TABLE.exists():
            raise RuntimeError(f"FEATURES TABLE missing: {FEATURES_TABLE}")

        model = joblib.load(MODEL_PATH)
        features = json.loads(FEATURES_PATH.read_text())
        features_df = pd.read_parquet(FEATURES_TABLE)
        app.state.model_bundle = {"model": model, "features": features, "features_df": features_df}
        print("[BOOT] model + features loaded", file=sys.stderr, flush=True)

    yield

# Attach lifespan at construction time (important!)
app = FastAPI(title="Mayo Demo API", lifespan=lifespan)

# --- Schemas
class AdmitReq(BaseModel):
    patient_id: str
    encounter_id: str

class AdmitResp(BaseModel):
    patient_id: str
    encounter_id: str
    probability: float
    label: int

# --- Helpers
def _bundle():
    """
    Return the loaded model bundle. In CI (SKIP_MODEL_LOAD=1),
    this returns the dummy bundle set by lifespan. If the attribute
    somehow isn’t present, create a safe stub in CI; error otherwise.
    """
    b = getattr(app.state, "model_bundle", None)
    if b is not None:
        return b

    # Defensive fallback: if tests didn’t trigger startup yet
    if os.getenv("SKIP_MODEL_LOAD", "").strip() in {"1", "true", "True"}:
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
        # In dummy mode, synthesize the id columns
        feat_df = pd.DataFrame([{"patient_id": pid, "encounter_id": enc}])

    X = _row_for(pid, enc, feat_df=feat_df, features=features)
    try:
        p = float(model.predict_proba(X)[:, 1][0])
    except Exception:
        p = 0.0
    return {"patient_id": pid, "encounter_id": enc, "probability": p, "label": int(p >= 0.5)}

# --- Routes
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    _ = _bundle()  # ensures state exists
    return {"status": "healthy", "title": app.title}

@app.get("/_routes")
def list_routes():
    return sorted([getattr(r, "path", "") for r in app.routes])

@app.post("/predict/admission/batch")
def predict_admission_batch(items: List[AdmitReq]):
    if not items:
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        try:
            out.append(_predict_one(it.patient_id, it.encounter_id))
        except HTTPException as e:
            out.append({"patient_id": it.patient_id, "encounter_id": it.encounter_id, "error": str(e.detail)})
        except Exception as e:
            out.append({"patient_id": it.patient_id, "encounter_id": it.encounter_id, "error": str(e)})
    return out

@app.get("/predict/admission/{patient_id}/{encounter_id}", response_model=AdmitResp)
def predict_admission(patient_id: str, encounter_id: str):
    return _predict_one(patient_id, encounter_id)
