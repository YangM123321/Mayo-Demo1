import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_parquet("out/labs_curated.parquet")
Path("out").mkdir(exist_ok=True)

ax = (df["loinc"].value_counts().sort_values(ascending=False).head(10)
      .plot(kind="bar", figsize=(8,4), title="Top LOINC counts"))
plt.tight_layout(); plt.savefig("out/loinc_top10.png")

valid_rate = df["is_value_valid"].mean()
with open("out/summary.txt","w") as f:
    f.write(f"Valid value rate: {valid_rate:.2%}\n")
print("Wrote out/loinc_top10.png and out/summary.txt")
