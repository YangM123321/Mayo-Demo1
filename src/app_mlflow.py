# src/app_mlflow.py
from pathlib import Path
from typing import Optional, Dict, Any
import os, traceback
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow, mlflow.pyfunc
import joblib

APP_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_ROOT / "models"
JOBLIB_FALLBACK = MODELS_DIR / "admit_mlflow_lr.joblib"

# Point to the same local registry DB you used in Stage 3
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
os.environ.setdefault("MLFLOW_REGISTRY_URI", "sqlite:///mlflow.db")

MODEL_URI = "models:/admission_lr@champion"  # registry alias
FEATURE_ORDER = ["2345-7", "718-7"]          # keep same order as training

app = FastAPI(title="Mayo Demo – MLflow Model API", version="1.0.0")

class AdmissionRequest(BaseModel):
    loinc_2345_7: Optional[float] = Field(None, description="Glucose (mg/dL)")
    loinc_718_7:  Optional[float] = Field(None, description="Hemoglobin (g/dL)")

def _load_model() -> Dict[str, Any]:
    # 1) Try MLflow Registry
    try:
        m = mlflow.pyfunc.load_model(MODEL_URI)
        return {"model": m, "source": MODEL_URI, "note": "mlflow.pyfunc"}
    except Exception as e:
        ml_err = f"MLflow load failed: {e}"

    # 2) Fallback to local joblib
    try:
        if JOBLIB_FALLBACK.exists():
            m = joblib.load(JOBLIB_FALLBACK)
            return {"model": m, "source": str(JOBLIB_FALLBACK), "note": "joblib fallback"}
        raise RuntimeError(f"{ml_err}; joblib file not found: {JOBLIB_FALLBACK}")
    except Exception as e:
        raise RuntimeError(f"{ml_err}; joblib fallback failed: {e}")

@app.on_event("startup")
def startup_load():
    app.state.model_bundle = _load_model()

@app.get("/health")
def health():
    b = app.state.model_bundle
    return {"ok": True, "model_source": b["source"], "note": b["note"], "features": FEATURE_ORDER}

@app.get("/model-info")
def model_info():
    info = {"model_source": app.state.model_bundle["source"], "features": FEATURE_ORDER}
    try:
        meta = getattr(app.state.model_bundle["model"], "metadata", None)
        if meta and hasattr(meta, "flavors"):
            info["flavors"] = list(meta.flavors.keys())
    except Exception:
        pass
    return info

@app.post("/predict/admission")
def predict(inp: AdmissionRequest):
    try:
        f_glu = float(inp.loinc_2345_7 or 0.0)
        f_hgb = float(inp.loinc_718_7  or 0.0)
        row = pd.DataFrame([[f_glu, f_hgb]], columns=FEATURE_ORDER)

        model = app.state.model_bundle["model"]

        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(row)[:, 1][0])
        else:
            yhat = np.asarray(model.predict(row)).ravel()
            proba = float(yhat[0]) if (yhat.size and 0.0 <= float(yhat[0]) <= 1.0) \
                    else float(yhat[0] >= 0.5)

        label = int(proba >= 0.5)
        return {
            "ok": True,
            "model_source": app.state.model_bundle["source"],
            "features": {"2345-7": f_glu, "718-7": f_hgb},
            "probability_admit": proba,
            "predicted_label": label,
            "details": {"threshold": 0.5, "feature_order": FEATURE_ORDER},
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
