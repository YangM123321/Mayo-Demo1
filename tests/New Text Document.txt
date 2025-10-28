def test_app_imports():
    import importlib
    m = importlib.import_module("src.app_mlflow")
    assert hasattr(m, "app"), "FastAPI app not found"
