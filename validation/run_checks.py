import pandas as pd
from great_expectations.dataset import PandasDataset

df = pd.read_csv("data/interim/observations.csv")
gdf = PandasDataset(df)

required_cols = ["patient_id", "timestamp", "code", "value", "unit"]
for col in required_cols:
    gdf.expect_column_to_exist(col)

gdf.expect_column_values_to_be_between("value", min_value=30, max_value=220)

for col in ["patient_id", "timestamp", "code"]:
    gdf.expect_column_values_to_not_be_null(col)


# ---- SUMMARY ----
results = gdf.validate()
print(results)

# Save a JSON report (serialize via GE helper)
import json, pathlib
pathlib.Path("validation").mkdir(exist_ok=True)
with open("validation/results.json", "w") as f:
    json.dump(results.to_json_dict(), f, indent=2)

print("Validation report written to validation/results.json")


