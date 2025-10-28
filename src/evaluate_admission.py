# src/evaluate_admission.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

OUT = Path("out")
MODELS = Path("models")
PLOTS = OUT / "ml_plots"
PLOTS.mkdir(parents=True, exist_ok=True)

DATA = OUT / "labs_curated.parquet"

def load_features_and_labels():
    df = pd.read_parquet(DATA)

    # same label definition as training
    df["admit_label"] = (
        ((df["loinc"] == "2345-7") & (df["lab_value"] >= 150)) |
        ((df["loinc"] == "718-7")  & (df["lab_value"] < 11.5))
    ).astype(int)

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

    # align with training logic
    if {"2345-7", "718-7"}.issubset(set(feat.columns)):
        feature_cols = ["2345-7", "718-7"]
    else:
        feature_cols = feat.columns.tolist()[2:]

    X = feat[feature_cols].values
    y = (
        df.groupby(["patient_id", "encounter_id"])["admit_label"]
          .max()
          .reindex(list(zip(feat["patient_id"], feat["encounter_id"])))
          .astype(int)
          .values
    )
    return X, y, feature_cols

def plot_curves(model, X, y, name: str):
    # get scores
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    else:
        yhat = np.asarray(model.predict(X)).ravel()
        p = yhat if (yhat.size and 0.0 <= float(yhat[0]) <= 1.0) else (yhat >= 0.5).astype(float)

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS / f"roc_{name}.png", dpi=150)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PLOTS / f"pr_{name}.png", dpi=150)
    plt.close()

    print(f"Wrote ROC/PR for {name}:")
    print(f"  {PLOTS / f'roc_{name}.png'}")
    print(f"  {PLOTS / f'pr_{name}.png'}")

def main():
    X, y, _ = load_features_and_labels()

    # load models saved by training step
    lr_path = MODELS / "admit_lr.joblib"
    rf_path = MODELS / "admit_rf.joblib"

    if lr_path.exists():
        lr = joblib.load(lr_path)
        plot_curves(lr, X, y, "logreg")
    else:
        print("Skip: models/admit_lr.joblib not found")

    if rf_path.exists():
        rf = joblib.load(rf_path)
        plot_curves(rf, X, y, "random_forest")
    else:
        print("Skip: models/admit_rf.joblib not found")

if __name__ == "__main__":
    main()
