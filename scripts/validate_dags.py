import os
import sys
import traceback
from pathlib import Path

DAGS_DIR = Path("dags")

def main() -> int:
    if not DAGS_DIR.exists():
        print(f"[validate_dags] No dags/ directory found at {DAGS_DIR.resolve()}")
        return 1

    failures = 0
    for pyfile in DAGS_DIR.glob("*.py"):
        try:
            # Import by executing the file in isolated namespace
            code = pyfile.read_text(encoding="utf-8")
            exec(compile(code, str(pyfile), "exec"), {})
            print(f"[validate_dags] OK: {pyfile.name}")
        except Exception:
            failures += 1
            print(f"[validate_dags] FAIL: {pyfile.name}")
            traceback.print_exc()

    if failures:
        print(f"[validate_dags] {failures} DAG file(s) failed import/parse.")
        return 1

    print("[validate_dags] All DAGs validated successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
