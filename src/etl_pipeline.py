# src/etl_pipeline.py
import os
import pandas as pd
from py2neo import Graph

# --- Neo4j config from env (works from Docker) ---
uri  = os.getenv("NEO4J_URI",  "bolt://host.docker.internal:7687")  # or bolt://mayo-neo4j:7687 on a docker network
user = os.getenv("NEO4J_USER", "neo4j")
pwd  = os.getenv("NEO4J_PASS", "testpass")

print(f"[etl] Connecting to Neo4j at {uri} as {user}")
graph = Graph(uri, auth=(user, pwd))

REQUIRED = {"patient_id", "encounter_id", "loinc", "lab_value", "unit", "collected_date"}

def fetch_dx_for_loinc(g: Graph, loinc: str):
    q = "MATCH (:Lab {loinc:$loinc})-[]->(d:Diagnosis) RETURN collect(d.code) AS dx"
    res = g.run(q, loinc=loinc).data()
    return res[0]["dx"] if res else []

def main():
    # input/output paths (ensure 'out/' exists)
    in_path  = "out/labs_clean.parquet"
    out_path = "out/labs_curated.parquet"

    # load
    df = pd.read_parquet(in_path)
    assert REQUIRED.issubset(df.columns), f"Schema mismatch: {set(df.columns)} vs {REQUIRED}"

    # ⚠️ DO NOT create a new Graph here; reuse the env-driven global `graph`
    df["dx_codes"] = df["loinc"].apply(lambda x: fetch_dx_for_loinc(graph, x))

    # simple validity check
    df["is_value_valid"] = df["lab_value"].apply(lambda v: bool(pd.notna(v) and v > 0))

    # save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[etl] Wrote {out_path}")

if __name__ == "__main__":
    main()
