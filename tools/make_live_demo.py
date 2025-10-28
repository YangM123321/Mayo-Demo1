# tools/make_live_demo.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import datetime

BASE = Path(__file__).resolve().parents[1]  # repo root
FEATURES = ["BP_SYS_mean", "BP_SYS_last", "BP_DIA_mean"]

# 1) make a trivial classifier expecting those FEATURES
X = np.array([[120, 118, 78], [160, 158, 100], [110, 110, 70], [150, 148, 95]], dtype=float)
y = np.array([0, 1, 0, 1])
clf = LogisticRegression().fit(X, y)

# save to models/baseline.pkl
(BASE / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(clf, BASE / "models" / "baseline.pkl")

# 2) make a single microbatch with patient_id=1
df = pd.DataFrame([{
    "patient_id": 1,
    "BP_SYS_mean": 130.0,
    "BP_SYS_last": 128.0,
    "BP_DIA_mean": 85.0,
}])

outdir = BASE / "data" / "stream_features"
outdir.mkdir(parents=True, exist_ok=True)
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
df.to_parquet(outdir / f"features_{ts}.parquet")

print("Wrote models/baseline.pkl and data/stream_features/features_*.parquet")
