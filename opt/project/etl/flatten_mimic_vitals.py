from pathlib import Path
import pandas as pd

# Resolve project root regardless of working directory
BASE = Path(__file__).resolve().parents[1]   # -> /opt/project

SRC = BASE / "data" / "physionet.org" / "files" / "mimic-iv-ed-demo" / "2.2" / "ed" / "vitalsign.csv.gz"
OUT = BASE / "data" / "interim" / "observations.csv"
NROWS = None  # set to an int (e.g., 50000) for a quick sample

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
    # Load CSV (pandas handles .gz)
    df = pd.read_csv(SRC, nrows=NROWS)

    # Only columns that exist
    available = [c for c in VITAL_MAP if c in df.columns]
    if not available:
        raise ValueError(f"No expected vital columns found in {SRC}. Got: {list(df.columns)}")

    id_cols = [c for c in ["subject_id", "stay_id", "charttime"] if c in df.columns]

    long = df[id_cols + available].melt(
        id_vars=id_cols,
        value_vars=available,
        var_name="vital_name",
        value_name="value",
    ).dropna(subset=["value"])

    long["code"] = long["vital_name"].map(lambda v: VITAL_MAP[v][0])
    long["unit"] = long["vital_name"].map(lambda v: VITAL_MAP[v][1])

    long = long.rename(columns={"subject_id": "patient_id", "charttime": "timestamp"})
    if "patient_id" not in long.columns and "stay_id" in long.columns:
        long["patient_id"] = long["stay_id"]
    if "patient_id" not in long.columns or "timestamp" not in long.columns:
        raise ValueError("Required columns 'patient_id' and/or 'timestamp' missing after rename.")

    long["timestamp"] = pd.to_datetime(long["timestamp"], errors="coerce")
    long = long.dropna(subset=["timestamp"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    obs = long[["patient_id", "timestamp", "code", "value", "unit"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT, index=False)
    print(f"Wrote {len(obs):,} rows to {OUT}")

if __name__ == "__main__":
    main()
