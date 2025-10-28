# src/kg_load.py
import os
from py2neo import Graph, Node, Relationship

URI  = os.getenv("NEO4J_URI",  "bolt://host.docker.internal:7687")  # or bolt://mayo-neo4j:7687 on a docker network
USER = os.getenv("NEO4J_USER", "neo4j")
PASS = os.getenv("NEO4J_PASS", "testpass")

print(f"[kg_load] Connecting to Neo4j at {URI} as {USER}")
graph = Graph(URI, auth=(USER, PASS))

# dev only: wipe graph
graph.run("MATCH (n) DETACH DELETE n")

diagnoses = [
    {"code": "250.00", "name": "Type 2 Diabetes"},
    {"code": "285.9",  "name": "Anemia"},
]
labs = [
    {"loinc": "2345-7", "name": "Glucose [Mass/volume] in Serum or Plasma"},
    {"loinc": "718-7",  "name": "Hemoglobin [Mass/volume] in Blood"},
]
links = [
    ("2345-7", "250.00", "INDICATES"),
    ("718-7",  "285.9",  "ASSOCIATED_WITH"),
]

tx = graph.begin()
for d in diagnoses:
    tx.merge(Node("Diagnosis", code=d["code"], name=d["name"]), "Diagnosis", "code")
for l in labs:
    tx.merge(Node("Lab", loinc=l["loinc"], name=l["name"]), "Lab", "loinc")
for loinc, icd, rel in links:
    lab = tx.evaluate("MATCH (l:Lab {loinc: $loinc}) RETURN l", loinc=loinc)
    dx  = tx.evaluate("MATCH (d:Diagnosis {code: $code}) RETURN d", code=icd)
    tx.create(Relationship(lab, rel, dx))
tx.commit()
print("KG loaded.")
