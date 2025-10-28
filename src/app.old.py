# ---- src/app.py ----
from pathlib import Path
from typing import Any, Dict, Optional, List
import os, json, re, requests
from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd, json, joblib

from fastapi import FastAPI, HTTPException, Body, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, RootModel, Field
from py2neo import Graph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
from datetime import datetime
import joblib, numpy as np, pandas as pd

# ---------------- Main app (FHIR + KG + NLP + local ML) ----------------
main = FastAPI(title="Clinical KG + NLP demo")
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()

def fhir_base() -> str:
    base = os.getenv("FHIR_BASE_URL", "").strip()
    if not base:
        raise RuntimeError("Set FHIR_BASE_URL env var or .env")
    return base

def fhir_params(params: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in params.items() if v is not None}

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")  # e.g., bolt://host.docker.internal:7687
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "testpass")
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)) if NEO4J_URI else None

# Tiny text classifier
train_text = ["polyuria high glucose", "low hemoglobin", "normal check"]
train_y    = ["diabetes","anemia","other"]
vec = TfidfVectorizer().fit(train_text)
clf = LogisticRegression(max_iter=500).fit(vec.transform(train_text), train_y)

def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+"," ", t).strip()

class NoteIn(BaseModel):
    text: str

class FHIRResource(RootModel[Dict[str, Any]]):
    pass

@main.get("/dx_by_loinc/{loinc}")
def dx_by_loinc(loinc: str):
    if graph is None:
        raise HTTPException(status_code=503, detail="Neo4j not configured (set NEO4J_URI).")
    q = "MATCH (:Lab {loinc:$loinc})-[]->(d:Diagnosis) RETURN collect(d.code) AS dx"
    res = graph.run(q, loinc=loinc).data()
    return {"loinc": loinc, "dx_codes": (res[0]["dx"] if res else [])}

@main.post("/classify_note")
def classify(note: NoteIn):
    X = vec.transform([clean_text(note.text)])
    yhat = clf.predict(X)[0]
    return {"label": yhat}

# --- Local FHIR store ---
FHIR_DIR   = BASE_DIR / "out" / "fhir"
FHIR_INDEX = FHIR_DIR / "index.json"

def load_fhir_index():
    if FHIR_INDEX.exists():
        with open(FHIR_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

fhir_index_cache = load_fhir_index()

@main.get("/fhir/observation/{obs_id}")
def fhir_observation(obs_id: str):
    path = FHIR_DIR / f"{obs_id}.json"
    if not path.exists():
        return JSONResponse({"detail": "Observation not found"}, status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))

@main.get("/fhir/observation/by_loinc/{loinc}")
def fhir_by_loinc(loinc: str, limit: int = 10):
    matches = [x for x in fhir_index_cache if x.get("loinc") == loinc][:max(1, min(limit, 100))]
    return JSONResponse({
        "loinc": loinc,
        "count": len(matches),
        "observations": [{"id": m["id"], "date": m["date"], "patient_id": m["patient_id"]} for m in matches]
    })

def _is_iso_datetime(s: str) -> bool:
    try:
        if len(s) == 10:
            datetime.strptime(s, "%Y-%m-%d")
            return True
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except Exception:
        return False

def _validate_observation(obs: dict):
    if obs.get("resourceType") != "Observation":
        raise HTTPException(400, "resourceType must be 'Observation'")
    if not isinstance(obs.get("id"), str) or not obs["id"]:
        raise HTTPException(400, "Observation.id is required")
    if obs.get("status") not in {"final", "amended", "corrected", "preliminary"}:
        raise HTTPException(400, "Observation.status invalid or missing")
    try:
        coding0 = obs["code"]["coding"][0]
    except Exception:
        raise HTTPException(400, "Observation.code.coding[0] is required")
    if coding0.get("system") != "http://loinc.org":
        raise HTTPException(400, "Observation.code.coding[0].system must be http://loinc.org")
    if not isinstance(coding0.get("code"), str) or not coding0["code"]:
        raise HTTPException(400, "Observation.code.coding[0].code (LOINC) is required")
    try:
        subj_ref = obs["subject"]["reference"]
    except Exception:
        raise HTTPException(400, "Observation.subject.reference is required")
    if not isinstance(subj_ref, str) or not subj_ref.startswith("Patient/"):
        raise HTTPException(400, "subject.reference must be like 'Patient/<id>'")
    eff = obs.get("effectiveDateTime")
    if not isinstance(eff, str) or not _is_iso_datetime(eff):
        raise HTTPException(400, "effectiveDateTime must be ISO date or datetime")
    try:
        vq = obs["valueQuantity"]
        float(vq["value"])
        if not vq.get("unit") or not vq.get("system") or not vq.get("code"):
            raise ValueError
    except Exception:
        raise HTTPException(400, "valueQuantity must include numeric value, unit, system, code")

@main.post("/fhir/observation", status_code=status.HTTP_201_CREATED)
def create_fhir_observation(obs: dict = Body(...)):
    _validate_observation(obs)
    obs_id = obs["id"]
    out_path = FHIR_DIR / f"{obs_id}.json"
    FHIR_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obs, f, ensure_ascii=False, indent=2)

    loinc_code = obs["code"]["coding"][0]["code"]
    eff_str = obs["effectiveDateTime"]
    date_only = eff_str[:10] if len(eff_str) >= 10 else eff_str
    patient_id = obs["subject"]["reference"].split("/", 1)[-1]

    global fhir_index_cache
    fhir_index_cache = [x for x in fhir_index_cache if x.get("id") != obs_id]
    fhir_index_cache.append({
        "id": obs_id,
        "loinc": loinc_code,
        "patient_id": patient_id,
        "date": date_only,
        "path": str(out_path),
    })
    with open(FHIR_INDEX, "w", encoding="utf-8") as f:
        json.dump(fhir_index_cache, f, ensure_ascii=False, indent=2)
    return {"detail": "created", "id": obs_id, "path": str(out_path)}

@main.get("/remote/fhir/observations/by_loinc/{loinc}")
def remote_fhir_by_loinc(loinc: str, limit: int = 10):
    base = fhir_base()
    url = f"{base}/Observation"
    params = fhir_params({"code": f"http://loinc.org|{loinc}", "_count": str(max(1, min(limit, 100)))})
    try:
        r = requests.get(url, params=params, timeout=15, headers={"Accept": "application/fhir+json"})
        r.raise_for_status()
        bundle = r.json()
    except requests.RequestException as e:
        raise HTTPException(502, f"FHIR GET failed: {e}")

    entries = bundle.get("entry", []) or []
    out = []
    for e in entries[:limit]:
        res = e.get("resource", {}) or {}
        out.append({
            "id": res.get("id"),
            "status": res.get("status"),
            "effectiveDateTime": res.get("effectiveDateTime") or res.get("issued"),
            "valueQuantity": res.get("valueQuantity"),
            "subject": res.get("subject"),
            "code": res.get("code"),
        })
    return {"server": base, "loinc": loinc, "count": len(out), "observations": out}

from src.deid import deid_observation

@main.post("/remote/fhir/submit_observation")
def remote_fhir_submit_observation(payload: FHIRResource, deid: bool = True):
    base = fhir_base()
    url = f"{base}/Observation"
    obs = payload.root
    if deid:
        obs = deid_observation(obs)
    if obs.get("resourceType") != "Observation":
        raise HTTPException(400, "resourceType must be 'Observation'")
    try:
        r = requests.post(url, json=obs, headers={"Content-Type": "application/fhir+json"}, timeout=15)
        r.raise_for_status()
        outcome = r.json()
    except requests.RequestException as e:
        raise HTTPException(502, f"FHIR POST failed: {e}")
    return {"server": base, "status": "submitted", "id": outcome.get("id"), "outcome": outcome}

@main.post("/deid/observation")
def deid_observation_api(obs: dict):
    try:
        return deid_observation(obs)
    except Exception as e:
        raise HTTPException(400, f"De-id failed: {e}")

# ----- Local joblib LR admission model -----
MODEL_PATH = BASE_DIR / "models" / "admit_lr.joblib"
FEAT_PATH  = BASE_DIR / "models" / "feature_list.json"
FEATURES_PARQUET = BASE_DIR / "data" / "processed" / "features.parquet"

if not (MODEL_PATH.exists() and FEAT_PATH.exists()):
    print("[WARN] ML artifacts not found; /predict/admission will 503 until you save models.")
    _model = None
    _feat_names: List[str] = []
else:
    _model = joblib.load(MODEL_PATH)
    _feat_names = json.loads(FEAT_PATH.read_text())

class AdmissionRequest(BaseModel):
    features: Dict[str, float] = Field(default_factory=dict)
    patient_id: Optional[int] = None

def _align_features(feat_in: Dict[str, float]) -> pd.DataFrame:
    x = {k: np.nan for k in _feat_names}
    for k, v in feat_in.items():
        if k in x:
            x[k] = v
    return pd.DataFrame([x])

@main.post("/predict/admission")
def predict_admission(payload: AdmissionRequest):
    if _model is None or not _feat_names:
        raise HTTPException(503, "Model not loaded. Train & save artifacts first.")
    if payload.features:
        X = _align_features(payload.features)
    elif payload.patient_id is not None:
        try:
            feat = pd.read_parquet(FEATURES_PARQUET)
        except Exception as e:
            raise HTTPException(500, f"Could not read features table: {e}")
        row = feat.loc[feat["patient_id"] == payload.patient_id]
        if row.empty:
            raise HTTPException(404, f"No features for patient_id={payload.patient_id}")
        row = row.drop(columns=["patient_id"]).iloc[0].to_dict()
        X = _align_features(row)
    else:
        raise HTTPException(400, "Provide either features or patient_id")
    proba = float(_model.predict_proba(X)[:, 1][0])
    
    label = int(proba >= 0.5)
    row_dict = X.iloc[0].to_dict()
    return {
        "label": int(label),
        "probability": float(proba),
        "features_used": {k: float(v) for k, v in row_dict.items()}
    }




# ---------------- MLflow sub-app (mounted) ----------------
from fastapi import FastAPI as _FastAPI
from pydantic import BaseModel as _BaseModel
import mlflow, mlflow.pyfunc

FEATURES_MLFLOW = ["2345-7", "718-7"]
MODELS_DIR = Path("models")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

def _load_mlflow_model():
    try:
        return mlflow.pyfunc.load_model("models:/admission_lr@champion")
    except Exception:
        return joblib.load(MODELS_DIR / "admit_mlflow_lr.joblib")

_mlflow_model = _load_mlflow_model()

mlflow_app = _FastAPI(title="Mayo Demo API (MLflow)")

class ObsIn(_BaseModel):
    loinc_2345_7: float | None = None
    loinc_718_7:  float | None = None

def to_feature_vec(x: ObsIn):
    vals = {"2345-7": x.loinc_2345_7 or 0.0, "718-7": x.loinc_718_7 or 0.0}
    return np.array([[vals[f] for f in FEATURES_MLFLOW]])

@mlflow_app.get("/")
def root():
    return {"status": "ok", "model": "admission_lr", "features": FEATURES_MLFLOW}

@mlflow_app.post("/predict/admission")
def predict_mlflow(x: ObsIn):
    X = to_feature_vec(x)
    try:
        if hasattr(_mlflow_model, "predict_proba"):
            p = float(_mlflow_model.predict_proba(X)[:, 1][0])
        else:
            p = float(np.ravel(_mlflow_model.predict(X))[0])
    except Exception:
        p = float(np.ravel(_mlflow_model.predict(X))[0])
    return {"admission_risk": p, "features_order": FEATURES_MLFLOW}

# mount the MLflow sub-app
main.mount("/mlflow", mlflow_app)

# ---- unify both apps ----
from fastapi import APIRouter

# Create a router for the live prediction feature
live_router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
STREAM_DIR = BASE_DIR / "data" / "stream_features"

# Try loading the live baseline model if it exists
LIVE_MODEL_PATH = BASE_DIR / "models" / "baseline.pkl"
_live_model = None
if LIVE_MODEL_PATH.exists():
    try:
        _live_model = joblib.load(LIVE_MODEL_PATH)
    except Exception as e:
        print(f"[WARN] Could not load live model: {e}")

FEATURES = ["BP_SYS_mean", "BP_SYS_last", "BP_DIA_mean"]  # example list

def _align_features_live(d: dict):
    return pd.DataFrame([[d.get(f) for f in FEATURES]], columns=FEATURES)

# ---- BigQuery-backed live features ----
import os
from google.cloud import bigquery

BQ_PROJECT = os.getenv("BQ_PROJECT", "").strip() or None  # None => default creds project
BQ_DATASET = os.getenv("BQ_DATASET", "mayo").strip()
BQ_TABLE   = os.getenv("BQ_TABLE", "features_latest").strip()
FEATURES   = ["BP_SYS_mean", "BP_SYS_last", "BP_DIA_mean"]  # must match training

def latest_features_for_bq(pid: int) -> dict:
    if not BQ_DATASET or not BQ_TABLE:
        raise HTTPException(500, "BQ env not set (BQ_DATASET/BQ_TABLE).")

    client = bigquery.Client(project=BQ_PROJECT)
    table = f"`{client.project}.{BQ_DATASET}.{BQ_TABLE}`"
    sql = f"""
    WITH last_win AS (
      SELECT MAX(window_end_ts) AS ts FROM {table}
    )
    SELECT *
    FROM {table}, last_win
    WHERE window_end_ts = ts AND CAST(patient_id AS INT64) = @pid
    LIMIT 1
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("pid", "INT64", int(pid))]
        ),
    )
    rows = list(job.result())
    if not rows:
        raise HTTPException(404, f"No features in latest window for patient_id={pid}")
    r = dict(rows[0].items())
    out = {k: float(r.get(k) or 0.0) for k in FEATURES}
    return out

@live_router.get("/predict/admission/live/{patient_id}")
def predict_admission_live(patient_id: int):
    if _live_model is None:
        raise HTTPException(503, "Live model not loaded. Please check models/baseline.pkl or LIVE_MODEL_GCS_URI.")
    feats = latest_features_for_bq(patient_id)
    X = _align_features_live(feats)  # uses FEATURES ordering
    proba = float(_live_model.predict_proba(X)[:, 1][0])
    return {"patient_id": patient_id, "probability": proba, "label": int(proba >= 0.5)}


@live_router.get("/debug/bq/latest_ids")
def debug_latest_ids():
    client = bigquery.Client(project=BQ_PROJECT)
    q = f"""
    SELECT ARRAY_AGG(DISTINCT CAST(patient_id AS STRING) ORDER BY patient_id LIMIT 20) ids,
           COUNT(*) rows,
           MAX(window_end_ts) max_ts
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    WHERE window_end_ts=(SELECT MAX(window_end_ts)
                         FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`)
    """
    rows = client.query(q).result()
    out = [dict(r) for r in rows]
    if not out:
        raise HTTPException(status_code=404, detail="No rows found")
    return out



# ---- GCS-aware loader for live model (drop-in) ----
import os
from google.cloud import storage

def _maybe_download_from_gcs(gcs_uri: str, local_path: Path):
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        return None
    bucket_name, _, blob_name = gcs_uri[5:].partition("/")
    try:
        st = storage.Client()
        b = st.bucket(bucket_name).blob(blob_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        b.download_to_filename(str(local_path))
        return local_path
    except Exception as e:
        print(f"[WARN] GCS download failed: {e}")
        return None

# Override LIVE_MODEL_PATH if env is set
LIVE_MODEL_GCS_URI = os.getenv("LIVE_MODEL_GCS_URI", "").strip()
if LIVE_MODEL_GCS_URI:
    tmp_local = LIVE_MODEL_PATH  # reuse: BASE_DIR / "models" / "baseline.pkl"
    p = _maybe_download_from_gcs(LIVE_MODEL_GCS_URI, tmp_local)
    if p and p.exists():
        try:
            _live_model = joblib.load(p)
            print(f"[live] loaded model from {LIVE_MODEL_GCS_URI}")
        except Exception as e:
            print(f"[WARN] Could not load downloaded live model: {e}")

# ---- GCS-aware loader for live model (drop-in) ----
import os
from google.cloud import storage
from pathlib import Path

def _maybe_download_from_gcs(gcs_uri: str, local_path: Path):
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        return None
    bucket_name, _, blob_name = gcs_uri[5:].partition("/")
    try:
        st = storage.Client()
        b = st.bucket(bucket_name).blob(blob_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        b.download_to_filename(str(local_path))
        return local_path
    except Exception as e:
        print(f"[WARN] GCS download failed: {e}")
        return None

# Override LIVE_MODEL_PATH if env is set
LIVE_MODEL_GCS_URI = os.getenv("LIVE_MODEL_GCS_URI", "").strip()
if LIVE_MODEL_GCS_URI:
    tmp_local = LIVE_MODEL_PATH  # reuse: BASE_DIR / "models" / "baseline.pkl"
    p = _maybe_download_from_gcs(LIVE_MODEL_GCS_URI, tmp_local)
    if p and p.exists():
        try:
            _live_model = joblib.load(p)
            print(f"[live] loaded model from {LIVE_MODEL_GCS_URI}")
        except Exception as e:
            print(f"[WARN] Could not load downloaded live model: {e}")





# Mount router to main app
main.include_router(live_router)
