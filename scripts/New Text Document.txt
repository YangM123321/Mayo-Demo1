# scripts/train_from_bq.py
from __future__ import annotations
import argparse, json, os, sys, datetime as dt, tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

FEATURE_COLS = ["BP_SYS_mean", "BP_SYS_last", "BP_DIA_mean"]

def read_latest_window(client: bigquery.Client, project: str, dataset: str, table: str, lookback_min: int):
    tbl = f"`{project}.{dataset}.{table}`"
    q = f"""
    -- Take the most recent window (or last {lookback_min} minutes if specified)
    WITH base AS (
      SELECT * FROM {tbl}
      WHERE window_end_ts >= TIMESTAMP_SUB(
        (SELECT MAX(window_end_ts) FROM {tbl}),
        INTERVAL @lookback_min MINUTE
      )
    )
    SELECT * FROM base
    """
    job = client.query(q, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("lookback_min", "INT64", lookback_min)]
    ))
    return job.result().to_dataframe(create_bqstorage_client=True)

def simple_label(df: pd.DataFrame) -> pd.Series:
    # Synthetic “admission-ish risk”: stage-2-ish thresholds
    return ((df["BP_SYS_mean"] >= 140) | (df["BP_DIA_mean"] >= 90)).astype(int)

def train(df: pd.DataFrame):
    X = df[FEATURE_COLS].astype(float).fillna(0.0)
    y = simple_label(df)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    # quick metrics on training window (fine for demo)
    p = clf.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, p)) if len(y.unique()) > 1 else float("nan")
    acc = float(accuracy_score(y, (p >= 0.5).astype(int)))
    return clf, {"auc": auc, "acc": acc, "n": int(len(df))}

def upload_local_to_gcs(local_path: Path, gcs_uri: str):
    assert gcs_uri.startswith("gs://")
    bucket_name, _, obj = gcs_uri[5:].partition("/")
    st = storage.Client()
    bucket = st.bucket(bucket_name)
    blob = bucket.blob(obj)
    blob.upload_from_filename(str(local_path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", default="mayo")
    ap.add_argument("--table",   default="features_latest")
    ap.add_argument("--bucket",  required=True, help="GCS bucket for models, e.g. mayo-demo-artifacts-<PROJECT>")
    ap.add_argument("--prefix",  default="models", help="GCS folder prefix")
    ap.add_argument("--lookback_min", type=int, default=60, help="use last N minutes of windows")
    ap.add_argument("--tag", default=None, help="optional model tag suffix")
    args = ap.parse_args()

    bq = bigquery.Client(project=args.project)
    df = read_latest_window(bq, args.project, args.dataset, args.table, args.lookback_min)
    if df.empty:
        print("No rows in BigQuery window; aborting.")
        sys.exit(2)

    # ensure columns exist
    for c in FEATURE_COLS + ["patient_id"]:
        if c not in df.columns:
            raise RuntimeError(f"Column missing in BQ: {c}")

    model, metrics = train(df)
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or ts

    local_dir = Path("models"); local_dir.mkdir(exist_ok=True)
    local_model = local_dir / f"baseline-{tag}.pkl"
    local_latest = local_dir / "baseline.pkl"
    joblib.dump(model, local_model)
    joblib.dump(model, local_latest)
    (local_dir / "feature_list.json").write_text(json.dumps(FEATURE_COLS, indent=2))

    # upload versioned + "latest"
    gcs_ver = f"gs://{args.bucket}/{args.prefix}/baseline-{tag}.pkl"
    gcs_latest = f"gs://{args.bucket}/{args.prefix}/baseline.pkl"
    upload_local_to_gcs(local_model, gcs_ver)
    upload_local_to_gcs(local_latest, gcs_latest)

    # small metrics file
    mpath = local_dir / f"metrics-{tag}.json"
    mpath.write_text(json.dumps(metrics, indent=2))
    upload_local_to_gcs(mpath, f"gs://{args.bucket}/{args.prefix}/metrics-{tag}.json")

    print("== TRAIN OK ==")
    print({"metrics": metrics, "gcs_model_latest": gcs_latest, "gcs_model_version": gcs_ver})

if __name__ == "__main__":
    main()

