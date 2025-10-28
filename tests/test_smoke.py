def test_app_imports():
    import importlib
    m = importlib.import_module("src.app_mlflow")
    assert hasattr(m, "app")

def test_health_route():
    from src.app_mlflow import app
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

