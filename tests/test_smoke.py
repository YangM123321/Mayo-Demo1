# tests/test_smoke.py
import sys, pathlib
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service import app

def test_health_route():
    client = TestClient(app)
    for path in ("/health", "/", "/openapi.json"):
        r = client.get(path)
        if r.status_code == 200:
            return
    assert False, "No 200 from /health, /, or /openapi.json"

