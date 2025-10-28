# etl/build_features.py
from pathlib import Path
import pandas as pd

BASE = Path("/opt/project")
obs_csv = BASE / "data" / "interim" / "observations.csv"
out_dir = BASE / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)
out_parquet = out_dir / "features.parquet"

df = pd.read_csv(obs_csv, parse_dates=["timestamp"])

# pivot vitals -> columns, compute simple features per patient
wide = (df.pivot_table(index=["patient_id","timestamp"],
                       columns="code", values="value", aggfunc="mean")
          .reset_index())

feat = (wide.sort_values("timestamp")
             .groupby("patient_id")
             .agg(
                 BP_SYS_mean=("BP_SYS","mean"),
                 BP_SYS_last=("BP_SYS","last"),
                 BP_DIA_mean=("BP_DIA","mean"),
             )
             .reset_index())

feat.to_parquet(out_parquet, index=False)
print(f"Wrote {len(feat)} rows to {out_parquet}")
