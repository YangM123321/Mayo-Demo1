# train/train_lr.py
import pathlib, json, joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
feat   = pd.read_parquet(ROOT / "data/processed/features.parquet")
labels = pd.read_parquet(ROOT / "data/processed/labels.parquet")

data = feat.merge(labels, on="patient_id", how="inner").dropna(subset=["admitted"])
X = data.drop(columns=["patient_id", "admitted"])
y = data["admitted"].astype(int)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=42)),
])

pipe.fit(X, y)

models = ROOT / "models"
models.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, models / "admit_lr.joblib")
(models / "feature_list.json").write_text(json.dumps(list(X.columns)))
print("Saved model + features to", models)
