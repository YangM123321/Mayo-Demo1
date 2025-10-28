import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CURATED = Path("out") / "labs_curated.parquet"
OUTDIR  = Path("out") / "viz"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPNG  = OUTDIR / "lab_counts.png"

df = pd.read_parquet(CURATED)

# Expect columns like: loinc, value, unit, patient_id, date
counts = (
    df.groupby("loinc")
      .size()
      .sort_values(ascending=False)
)

plt.figure(figsize=(6,4))
counts.plot(kind="bar")          # keep default matplotlib style
plt.title("Lab observations per LOINC")
plt.xlabel("LOINC code")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPNG, dpi=200)
print(f"Saved {OUTPNG}")
