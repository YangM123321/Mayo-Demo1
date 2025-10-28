import pandas as pd
from dateutil import parser
from pathlib import Path

LOCAL_TO_LOINC = {"HGB_LOCAL": "718-7", "GLU_LOCAL": "2345-7"}
UNIT_NORMALIZATION = {"g/dl": "g/dL", "mg_dl": "mg/dL", "mg/dl": "mg/dL"}

def safe_parse_date(x):
    try:
        return parser.parse(str(x)).date()
    except Exception:
        return pd.NaT

def main(in_path="data/labs_raw.csv", out_path="out/labs_clean.parquet"):
    df = pd.read_csv(in_path)
    keep = ["patient_id","encounter_id","lab_code","lab_value","unit","collected_time"]
    df = df[[c for c in keep if c in df.columns]].copy()

    df["collected_date"] = df["collected_time"].apply(safe_parse_date)
    df = df.dropna(subset=["patient_id","lab_code","lab_value","collected_date"])

    df["unit"] = df["unit"].astype(str).str.strip().str.lower().map(UNIT_NORMALIZATION).fillna(df["unit"])
    df["loinc"] = df["lab_code"].map(LOCAL_TO_LOINC).fillna(df["lab_code"])

    df = (df.sort_values("collected_date")
            .drop_duplicates(subset=["patient_id","encounter_id","loinc"], keep="last"))

    Path("out").mkdir(exist_ok=True)
    df.to_parquet(out_path)
    print("Cleaned rows:", len(df))
    print(df.head())

if __name__ == "__main__":
    main()
