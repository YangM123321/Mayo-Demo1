import pandas as pd
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"

# Use MIMIC-IV demo hospital labs (the ED demo doesn't include labs)
SRC  = DATA / "physionet.org/files/mimic-iv-demo/2.2/hosp/labevents.csv.gz"
OUT  = DATA / "interim/labs.csv"
NROWS = None  # e.g. 100_000 for a quick run

def main():
    if not SRC.exists():
        raise FileNotFoundError(
            f"Expected file not found: {SRC}\n"
            "Download the MIMIC-IV *demo* (not ED) hosp/labevents.csv.gz under data/physionet.org/files/mimic-iv-demo/2.2/hosp/"
        )

    usecols = [c for c in ["subject_id","hadm_id","charttime","itemid","valuenum","valueuom"]]
    df = pd.read_csv(SRC, nrows=NROWS, usecols=lambda c: c in usecols)

    df = df.rename(columns={"subject_id":"patient_id", "charttime":"timestamp",
                            "valuenum":"value", "valueuom":"unit", "itemid":"code"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    out = df[["patient_id","timestamp","code","value","unit"]].dropna(subset=["value"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out):,} rows to {OUT}")

if __name__ == "__main__":
    main()
