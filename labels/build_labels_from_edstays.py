# labels/build_labels_from_edstays.py
import pandas as pd
import pathlib

EDSTAYS = "data/physionet.org/files/mimic-iv-ed-demo/2.2/ed/edstays.csv.gz"
OUT = "data/processed/labels.parquet"

def main():
    df = pd.read_csv(EDSTAYS, compression="infer")

    # Try to be robust to column names across demo versions
    cols = {c.lower(): c for c in df.columns}
    # subject_id is the MIMIC patient; map to our patient_id
    subj_col = cols.get("subject_id") or cols.get("patient_id") or "subject_id"

    # Heuristics for "admission":
    # - hadm_id present (linked hospital admission)
    # - or disposition text contains 'admit'
    hadm_col = cols.get("hadm_id")
    disp_col = None
    for k in ["hospitaldisposition", "disposition", "eddisposition"]:
        if k in cols:
            disp_col = cols[k]; break

    admitted = pd.Series(False, index=df.index)
    if hadm_col and hadm_col in df:
        admitted = admitted | df[hadm_col].notna()
    if disp_col and disp_col in df:
        admitted = admitted | df[disp_col].astype(str).str.lower().str.contains("admit")

    labels = (
        df.assign(admitted=admitted.astype(int))
          .groupby(subj_col, as_index=False)["admitted"].max()
          .rename(columns={subj_col: "patient_id"})
    )

    pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
    labels.to_parquet(OUT, index=False)
    print(f"Wrote labels: {labels.shape[0]} rows -> {OUT}")

if __name__ == "__main__":
    main()
