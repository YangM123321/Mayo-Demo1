# src/train_admission.py
from pathlib import Path
from collections import Counter
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

OUT = Path("out")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)

DATA = OUT / "labs_curated.parquet"  # created by your ETL
REPORT_TXT = OUT / "ml_report.txt"

def main():
    # ---- 1) Load curated data and create label
    df = pd.read_parquet(DATA)

    # demo rule: high glucose >=150 OR low hgb <11.5 -> positive
    df["admit_label"] = (
        ((df["loinc"] == "2345-7") & (df["lab_value"] >= 150)) |
        ((df["loinc"] == "718-7")  & (df["lab_value"] < 11.5))
    ).astype(int)

    # ---- 2) Wide features by LOINC per encounter
    feat = (
        df.pivot_table(
            index=["patient_id", "encounter_id"],
            columns="loinc",
            values="lab_value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .fillna(0.0)
    )

    # features: use specific LOINCs if present, else all numeric after IDs
    if {"2345-7", "718-7"}.issubset(set(feat.columns)):
        feature_cols = ["2345-7", "718-7"]
    else:
        feature_cols = feat.columns.tolist()[2:]

    X = feat[feature_cols].values

    # y: max label per encounter
    y = (
        df.groupby(["patient_id", "encounter_id"])["admit_label"]
          .max()
          .reindex(list(zip(feat["patient_id"], feat["encounter_id"])))
          .astype(int)
          .values
    )

    # ---- 3) Tiny-data safe split
    counts = Counter(y)
    min_class = min(counts.values()) if counts else 0
    if len(y) < 4 or min_class < 2:
        print(f"[tiny-data mode] n={len(y)} class_counts={dict(counts)} -> train on all")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    # ---- 4) Train models
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # ---- 5) Evaluate (best-effort for tiny sets)
    def safe_report(name, model):
        try:
            y_pred = model.predict(X_test)
            rep = classification_report(y_test, y_pred, zero_division=0)
            print(f"\n{name} report:\n{rep}")
            return rep
        except Exception as e:
            msg = f"{name} evaluation skipped: {e}"
            print(msg)
            return msg

    rep_lr = safe_report("LogisticRegression", lr)
    rep_rf = safe_report("RandomForest", rf)

    # ---- 6) Save artifacts
    joblib.dump(lr, MODELS / "admit_lr.joblib")
    joblib.dump(rf, MODELS / "admit_rf.joblib")
    with open(MODELS / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    # Save combined report
    REPORT_TXT.write_text(
        "=== LogisticRegression ===\n"
        + rep_lr
        + "\n\n=== RandomForest ===\n"
        + rep_rf
        + "\n",
        encoding="utf-8",
    )

    print("\nSaved:")
    print("  models/admit_lr.joblib")
    print("  models/admit_rf.joblib")
    print("  models/feature_list.json")
    print(f"  {REPORT_TXT}")

if __name__ == "__main__":
    main()
