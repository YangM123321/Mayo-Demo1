# src/train_admission_mlflow.py
from pathlib import Path
import os, json, joblib
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

ROOT = Path.cwd()
OUT = ROOT / "out"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# Clear any confusing env overrides, then use SQLite for both tracking and registry
os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ.pop("MLFLOW_REGISTRY_URI", None)
MLFLOW_DB = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_registry_uri(MLFLOW_DB)

mlflow.set_experiment("admission_risk_demo")

# ----- Load data and build label
df = pd.read_parquet(OUT / "labs_curated.parquet")
df["admit_label"] = (
    ((df["loinc"] == "2345-7") & (df["lab_value"] >= 150)) |
    ((df["loinc"] == "718-7")  & (df["lab_value"] < 11.5))
).astype(int)

feat = (df.pivot_table(index=["patient_id","encounter_id"],
                       columns="loinc", values="lab_value", aggfunc="mean")
          .reset_index().rename_axis(None, axis=1)).fillna(0.0)

feature_cols = ["2345-7","718-7"] if {"2345-7","718-7"}.issubset(feat.columns) else feat.columns.tolist()[2:]
X = feat[feature_cols].values
y = (df.groupby(["patient_id","encounter_id"])["admit_label"]
        .max()
        .reindex(list(zip(feat["patient_id"], feat["encounter_id"])))
        .astype(int)
        .values)

# ----- Tiny-data safe split
counts = Counter(y)
min_class = min(counts.values()) if counts else 0
if len(y) < 4 or min_class < 2:
    Xtr, Xte, ytr, yte = X, X, y, y
else:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

with mlflow.start_run(run_name="logreg_baseline"):
    model = LogisticRegression(max_iter=200).fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    try:
        auc = roc_auc_score(yte, proba)
    except Exception:
        auc = float("nan")

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("features", ",".join(feature_cols))
    mlflow.log_metric("roc_auc", auc)

    # optional text report
    try:
        report = classification_report(yte, model.predict(Xte), zero_division=0)
        (OUT / "ml_report.txt").write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(OUT / "ml_report.txt"))
    except Exception:
        pass

    # log to MLflow and also save local copy for API fallback
    mlflow.sklearn.log_model(model, artifact_path="model")
    joblib.dump(model, MODELS / "admit_mlflow_lr.joblib")

print("Logged to MLflow. AUC:", auc)
print("Artifacts:", MODELS / "admit_mlflow_lr.joblib")
