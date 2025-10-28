import sys, pathlib
from fastapi.testclient import TestClient

# Add repo root to sys.path so imports like 'src.service' work on CI
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Try to import the FastAPI app
app = None
try:
    from src.service import app  # your current app
except Exception:
    try:
        from src.app_mlflow import app  # fallback if you move files later
    except Exception as e:
        raise ImportError(f'Could not import FastAPI app: {e}')

def test_health_route():
    client = TestClient(app)
    # Check a few common endpoints; pass if any returns 200
    for path in ("/health", "/", "/openapi.json"):
        r = client.get(path)
        if r.status_code == 200:
            return
    assert False, "No 200 from /health, /, or /openapi.json"
