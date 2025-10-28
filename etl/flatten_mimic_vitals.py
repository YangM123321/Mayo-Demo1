import pandas as pd
import pathlib

SRC = r"data/physionet.org/files/mimic-iv-ed-demo/2.2/ed/vitalsign.csv.gz"
OUT = "data/interim/observations.csv"
NROWS = None      # set to 50000 for a quick sample if you want

# Map vitals -> (code, unit)
VITAL_MAP = {
    "heart_rate": ("HR", "bpm"),
    "sbp": ("BP_SYS", "mmHg"),
    "dbp": ("BP_DIA", "mmHg"),
    "resp_rate": ("RR", "breaths/min"),
    "temp_c": ("TEMP_C", "°C"),
    "spo2": ("SPO2", "%"),
    "glucose": ("GLUCOSE", "mg/dL"),
}

def main():
    # Load a slice (pandas handles .gz automatically)
    df = pd.read_csv(SRC, nrows=NROWS)

    # Pick columns that actually exist in this file
    available = [c for c in VITAL_MAP.keys() if c in df.columns]
    if not available:
        raise ValueError(f"No expected vital columns found in {SRC}. Got: {list(df.columns)}")

    # Basic id/time columns used in the ED demo
    # (present in MIMIC-IV-ED vitals file)
    id_cols = [c for c in ["subject_id", "stay_id", "charttime"] if c in df.columns]

    # Melt wide -> long for the available vitals
    long = df[id_cols + available].melt(
        id_vars=id_cols,
        value_vars=available,
        var_name="vital_name",
        value_name="value"
    )

    # Drop missing values
    long = long.dropna(subset=["value"])

    # Map to our standard schema
    long["code"] = long["vital_name"].map(lambda v: VITAL_MAP[v][0])
    long["unit"] = long["vital_name"].map(lambda v: VITAL_MAP[v][1])

    # Rename and select final columns
    long = long.rename(columns={"subject_id": "patient_id", "charttime": "timestamp"})
    for col in ["patient_id", "timestamp"]:
        if col not in long.columns:
            # Fall back to stay_id as patient id if subject_id missing
            if col == "patient_id" and "stay_id" in long.columns:
                long["patient_id"] = long["stay_id"]
            else:
                raise ValueError(f"Required column {col} not found in source.")

    long["timestamp"] = pd.to_datetime(long["timestamp"], errors="coerce")
    long = long.dropna(subset=["timestamp"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # keep only our standard five columns
    obs = long[["patient_id", "timestamp", "code", "value", "unit"]]

    pathlib.Path("data/interim").mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT, index=False)
    print(f"Wrote {len(obs):,} rows to {OUT}")

if __name__ == "__main__":
    main()
