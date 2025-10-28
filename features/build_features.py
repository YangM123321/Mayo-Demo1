
# features/build_features.py
import pandas as pd
import pathlib

SRC = "data/interim/observations.csv"
OUT = "data/processed/features.parquet"   # small + fast to load

def main():
    df = pd.read_csv(SRC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # basic QC: keep rows with a value
    df = df.dropna(subset=["value"])

    # recent value per (patient, code)
    df = df.sort_values(["patient_id", "code", "timestamp"])
    last_val = df.groupby(["patient_id", "code"])["value"].last().rename("last")

    # simple stats per (patient, code)
    agg = df.groupby(["patient_id", "code"])["value"].agg(mean="mean", std="std", min="min", max="max", count="count")

    # combine + pivot wide: one row per patient, columns like HR_mean, HR_last, …
    wide = pd.concat([agg, last_val], axis=1).reset_index()
    wide["feature_prefix"] = wide["code"].astype(str)
    # build wide column names
    wide_cols = {}
    for stat in ["mean", "std", "min", "max", "count", "last"]:
        wide_cols[stat] = wide["feature_prefix"] + f"_{stat}"
    # construct a tidy table for pivot
    tidy = pd.melt(
        wide,
        id_vars=["patient_id", "code", "feature_prefix"],
        value_vars=["mean", "std", "min", "max", "count", "last"],
        var_name="stat",
        value_name="val",
    )
    tidy["feature"] = tidy["feature_prefix"] + "_" + tidy["stat"]
    feat = tidy.pivot(index="patient_id", columns="feature", values="val").reset_index()

    pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
    feat.to_parquet(OUT, index=False)
    print(f"Wrote features: {feat.shape[0]} rows x {feat.shape[1]} cols -> {OUT}")

if __name__ == "__main__":
    main()
