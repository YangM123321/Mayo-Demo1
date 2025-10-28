# scripts/train_baseline.py
from pathlib import Path
import pandas as pd, json, joblib
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parents[1]
STREAM_DIR = BASE / "data" / "stream_features"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

# use latest micro-batch parquet
files = sorted(STREAM_DIR.glob("features_*.parquet"))
if not files:
    raise SystemExit("No stream feature files found in data/stream_features")

df = pd.read_parquet(files[-1])
X = df[["BP_SYS_mean","BP_SYS_last","BP_DIA_mean"]]

# simple synthetic label: hypertension-ish thresholding
y = ((X["BP_SYS_mean"] >= 140) | (X["BP_DIA_mean"] >= 90)).astype(int)

clf = LogisticRegression(max_iter=1000).fit(X, y)
joblib.dump(clf, MODELS_DIR / "baseline.pkl")
(MODELS_DIR / "feature_list.json").write_text(json.dumps(list(X.columns)))
print("Saved models/baseline.pkl and models/feature_list.json")
