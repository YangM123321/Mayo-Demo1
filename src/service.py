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
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.requests import Request

from src.bootstrap.artifacts import ArtifactError, bootstrap_stage5
from src.bootstrap.manifest import bootstrap_from_manifest
from src.contracts.predictions import AdmitReq, AdmitResp

print("[BOOT] importing src.service", file=sys.stderr, flush=True)


def _require_file(p: Path, label: str) -> None:
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"{label} missing or empty: {p}")


def _as_path(x: Optional[str]) -> Optional[Path]:
    return Path(x) if x else None


def _fix_legacy_path(p: Optional[Path]) -> Optional[Path]:
    """
    Old images used /app/out/*. New runtime uses /tmp/artifacts/**.
    If bootstrap returns a legacy path, try to map it to the new layout.
    """
    if not p:
        return None

    # If it exists and is non-empty, keep it
    try:
        if p.exists() and p.stat().st_size > 0:
            return p
    except Exception:
        pass

    # Legacy mapping for old paths like /app/out/features_matrix.parquet
    if str(p).startswith("/app/out/"):
        name = p.name  # e.g., features_matrix.parquet
        candidates = [
            Path("/tmp/artifacts/features") / name,
            Path("/tmp/artifacts/model") / name,
            Path("/tmp/artifacts") / name,
        ]
        for c in candidates:
            try:
                if c.exists() and c.stat().st_size > 0:
                    return c
            except Exception:
                continue

    return p


def _list_artifacts(max_items: int = 80) -> list[str]:
    """Best-effort listing of /tmp/artifacts contents for debugging startup crashes."""
    root = Path("/tmp/artifacts")
    if not root.exists():
        return []
    out: list[str] = []
    try:
        for p in root.rglob("*"):
            out.append(str(p))
            if len(out) >= max_items:
                break
    except Exception:
        return out
    return out


# -----------------------------
# Lifespan (BOOTSTRAP FIRST)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = None
    app.state.bootstrap = {}

    skip = os.getenv("SKIP_MODEL_LOAD", "").strip().lower() in {"1", "true", "yes"}
    if skip:

        class _Dummy:
            def predict_proba(self, X):
                import numpy as np

                n = len(X)
                return np.c_[np.zeros(n), np.zeros(n)]  # shape (n,2)

        app.state.model_bundle = {"model": _Dummy(), "features": [], "features_df": None}
        print("[BOOT] SKIP_MODEL_LOAD=1 -> dummy model bundle", file=sys.stderr, flush=True)
        yield
        return

    # ✅ Stage 5 MUST run before any file checks/loads
    try:
        if os.getenv("ARTIFACT_MANIFEST_URI"):
            boot = bootstrap_from_manifest()
        else:
            boot = bootstrap_stage5()
        app.state.bootstrap = boot
    except ArtifactError as e:
        raise RuntimeError(f"Stage 5 bootstrap failed: {e}") from e

    # Stage 5 returns local paths (should be /tmp/artifacts/...)
    model_path = _fix_legacy_path(_as_path(app.state.bootstrap.get("model_path")))
    features_table_path = _fix_legacy_path(_as_path(app.state.bootstrap.get("features_path")))

    # You ALSO need feature_list.json as an artifact.
    # Supported:
    # 1) feature_list_path set by bootstrap
    # 2) fallback env FEATURE_LIST_PATH
    feature_list_path = _fix_legacy_path(
        _as_path(app.state.bootstrap.get("feature_list_path"))
        or _as_path(os.getenv("FEATURE_LIST_PATH"))
    )

    # Validate "present" (not existence yet)
    if not model_path:
        raise RuntimeError("bootstrap did not provide model_path")
    if not features_table_path:
        raise RuntimeError("bootstrap did not provide features_path (features parquet)")
    if not feature_list_path:
        raise RuntimeError(
            "feature_list_path not provided. Add it to Stage 5 manifest or set FEATURE_LIST_PATH."
        )

    # Validate existence with rich debug output
    try:
        _require_file(model_path, "MODEL")
        _require_file(feature_list_path, "FEATURE LIST")
        _require_file(features_table_path, "FEATURES TABLE")
    except Exception as e:
        found = _list_artifacts()
        raise RuntimeError(
            f"{e}. bootstrap={app.state.bootstrap}. /tmp/artifacts_found={found}"
        ) from e

    # Load
    model = joblib.load(model_path)
    features = json.loads(feature_list_path.read_text(encoding="utf-8"))
    features_df = pd.read_parquet(features_table_path)

    app.state.model_bundle = {
        "model": model,
        "features": features,
        "features_df": features_df,
        "features_table": str(features_table_path),
        "model_path": str(model_path),
        "feature_list_path": str(feature_list_path),
    }

    print(
        f"[BOOT] model+features loaded "
        f"(model={model_path}, feature_list={feature_list_path}, table={features_table_path})",
        file=sys.stderr,
        flush=True,
    )

    yield


app = FastAPI(title="Mayo Demo API", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    boot = getattr(app.state, "bootstrap", {}) or {}
    b = getattr(app.state, "model_bundle", None)
    return {
        "ok": True,
        "bootstrap": {
            "model_path": boot.get("model_path"),
            "features_path": boot.get("features_path"),
            "config_path": boot.get("config_path"),
            "feature_list_path": boot.get("feature_list_path"),
        },
        "loaded": bool(b),
        "loaded_paths": (b or {}),
    }


# -----------------------------
# Error handler (single)
# -----------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print(f"[ERROR] {request.method} {request.url} -> {exc}", file=sys.stderr, flush=True)
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": "internal_error"})


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


@app.post("/predict/admission/batch", response_model=list[AdmitResp])
def predict_admission_batch(items: List[AdmitReq]):
    if not items:
        return []
    out: List[AdmitResp] = []
    for it in items:
        out.append(AdmitResp(**_predict_one(it.patient_id, it.encounter_id)))
    return out


@app.get("/predict/admission/{patient_id}/{encounter_id}", response_model=AdmitResp)
def predict_admission(patient_id: str, encounter_id: str):
    return AdmitResp(**_predict_one(patient_id, encounter_id))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
