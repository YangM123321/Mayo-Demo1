import pathlib
import sys

BANNED_EXT = {".parquet", ".pkl", ".joblib", ".onnx", ".bin"}
BANNED_DIRS = {"mlruns", "artifacts", "data"}

root = pathlib.Path(__file__).resolve().parents[1]

bad = []
for p in root.rglob("*"):
    if p.is_dir() and p.name in BANNED_DIRS:
        bad.append(str(p))
    if p.is_file() and p.suffix.lower() in BANNED_EXT:
        bad.append(str(p))

if bad:
    print("Found banned artifacts in repo:")
    for x in bad[:200]:
        print(" -", x)
    sys.exit(1)

print("OK: no baked artifacts found.")
