# labels/build_labels_revisit.py
import pandas as pd
from pathlib import Path

EDSTAYS = Path("data/physionet.org/files/mimic-iv-ed-demo/2.2/ed/edstays.csv.gz")
OUT     = Path("data/processed/labels.parquet")   # merge with existing labels if present

def main():
    df = pd.read_csv(EDSTAYS, compression="infer")
    # Try common id/time column names in the demo
    id_col = "subject_id" if "subject_id" in df.columns else "patient_id"
    in_col = "intime"
    out_col = "outtime"
    for c in (in_col, out_col):
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # Keep rows with valid times
    df = df.dropna(subset=[id_col, in_col, out_col]).copy()
    df = df.sort_values([id_col, in_col])

    # For each stay, find the *next* ED arrival for the same patient
    df["next_intime"] = df.groupby(id_col)[in_col].shift(-1)

    # Compute time to next visit
    dt = (df["next_intime"] - df[out_col]).dt.total_seconds() / 3600.0  # hours

    # Labels
    df["revisit_72h"] = ((dt >= 0) & (dt <= 72)).astype("Int64")
    df["revisit_7d"]  = ((dt >= 0) & (dt <= 7*24)).astype("Int64")
    df["revisit_30d"] = ((dt >= 0) & (dt <= 30*24)).astype("Int64")

    # One row per index stay → label belongs to the *current* stay’s patient
    labels = df[[id_col, "revisit_72h", "revisit_7d", "revisit_30d"]].copy()
    labels = labels.rename(columns={id_col: "patient_id"}).drop_duplicates()

    # If you already have labels.parquet (e.g., admitted), merge; else write fresh
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        merged = old.merge(labels, on="patient_id", how="outer")
        merged.to_parquet(OUT, index=False)
        print(f"Merged labels → {OUT} (shape={merged.shape})")
    else:
        labels.to_pa_
