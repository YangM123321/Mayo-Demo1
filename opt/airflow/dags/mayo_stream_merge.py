from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
import pandas as pd
import glob, os, joblib

BASE = Path("/opt/project")           # mounted project root in your compose
SRC  = BASE / "data" / "stream_features"
OUT  = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def merge_microbatches():
    files = sorted(glob.glob(str(SRC / "features_*.parquet")))
    if not files:
        print("No microbatches yet"); return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    outp = OUT / "features.parquet"
    df.to_parquet(outp, index=False)
    print(f"Wrote {len(df)} rows to {outp}")

def retrain_baseline():
    # Example retrain stub – replace with your real train() code
    feats = pd.read_parquet(OUT / "features.parquet")
    if feats.empty: 
        print("No data to train"); return
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    y = (feats["BP_SYS_mean"] > 140).astype(int)  # dummy label for demo
    X = feats[["BP_SYS_mean","BP_SYS_last","BP_DIA_mean"]]
    m = LogisticRegression().fit(X, y)
    OUT.parent.joinpath("models").mkdir(exist_ok=True, parents=True)
    joblib.dump(m, OUT.parent / "models" / "baseline.pkl")
    print("Retrained baseline model")

with DAG(
    "mayo_stream_merge",
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 * * * *",  # hourly
    catchup=False,
    default_args={"retries": 0},
) as dag:
    merge = PythonOperator(task_id="merge_microbatches", python_callable=merge_microbatches)
    # optional nightly retrain at 02:00
    retrain = PythonOperator(
        task_id="retrain_baseline",
        python_callable=retrain_baseline,
        execution_timeout=timedelta(minutes=30),
        )
    merge >> retrain
