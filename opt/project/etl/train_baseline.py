# etl/train_baseline.py
from pathlib import Path
import joblib, pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = Path("/opt/project")
feat_path = BASE / "data" / "processed" / "features.parquet"
model_dir = BASE / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "baseline.pkl"

df = pd.read_parquet(feat_path)
# toy label: high systolic BP => 1
y = (df["BP_SYS_mean"] > 140).astype(int)
X = df[["BP_SYS_mean","BP_SYS_last","BP_DIA_mean"]]

m = LogisticRegression(max_iter=500).fit(X, y)
joblib.dump(m, model_path)
print("Saved model to", model_path)
