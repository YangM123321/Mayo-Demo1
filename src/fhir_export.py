import json
from pathlib import Path
from datetime import datetime
import pandas as pd

IN_PATH = Path("out/labs_curated.parquet")
OUT_DIR = Path("out/fhir")
INDEX_PATH = OUT_DIR / "index.json"

def uom_to_ucum(unit: str):
    # map common lab units to UCUM coding
    unit_norm = (unit or "").strip()
    # already normalized earlier to mg/dL or g/dL in your cleaner
    if unit_norm in ("mg/dL", "mg/dl", "mg_dl"):
        return {"value_unit":"mg/dL", "system":"http://unitsofmeasure.org", "code":"mg/dL"}
    if unit_norm in ("g/dL", "g/dl"):
        return {"value_unit":"g/dL", "system":"http://unitsofmeasure.org", "code":"g/dL"}
    # fallback
    return {"value_unit": unit_norm or "1", "system":"http://unitsofmeasure.org", "code": unit_norm or "1"}

def to_fhir_observation(row):
    # Build a stable ID: obs-<patient>-<loinc>-<yyyymmdd>
    date = pd.to_datetime(row["collected_date"]).date()
    obs_id = f"obs-{row['patient_id']}-{row['loinc']}-{date.strftime('%Y%m%d')}"
    ucum = uom_to_ucum(row["unit"])

    return obs_id, {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "laboratory",
                "display": "Laboratory"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": str(row["loinc"]),
                "display": "Lab test"
            }],
            "text": "Lab Observation"
        },
        "subject": {
            "reference": f"Patient/{row['patient_id']}"
        },
        "effectiveDateTime": f"{date.isoformat()}T00:00:00Z",
        "valueQuantity": {
            "value": float(row["lab_value"]),
            "unit": ucum["value_unit"],
            "system": ucum["system"],
            "code": ucum["code"]
        }
    }

def main():
    if not IN_PATH.exists():
        raise SystemExit("Missing out/labs_curated.parquet. Run etl_pipeline.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PATH)
    required = {"patient_id","loinc","lab_value","unit","collected_date"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    index = []
    for _, row in df.iterrows():
        obs_id, obs = to_fhir_observation(row)
        p = OUT_DIR / f"{obs_id}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obs, f, ensure_ascii=False, indent=2)
        index.append({
            "id": obs_id,
            "loinc": str(row["loinc"]),
            "patient_id": str(row["patient_id"]),
            "date": str(pd.to_datetime(row["collected_date"]).date()),
            "path": str(p)
        })

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(index)} FHIR Observations to {OUT_DIR}")
    print(f"Index: {INDEX_PATH}")

if __name__ == "__main__":
    main()

