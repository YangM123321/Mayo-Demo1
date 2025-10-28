# opt/project/etl/flatten_mimic_labs.py
from pathlib import Path
import pandas as pd

OUT = Path("/opt/project/data/raw")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame({"patient_id":[1001,1002], "lab_code":["2345-7","718-7"], "value":[120,14.0]})
df.to_csv(OUT/"labs.csv", index=False)
print("wrote labs.csv")
