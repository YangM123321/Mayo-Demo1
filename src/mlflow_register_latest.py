# src/mlflow_register_latest.py
import time
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "admission_risk_demo"
MODEL_NAME = "admission_lr"
ALIAS = "champion"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
client = MlflowClient()

exp = client.get_experiment_by_name(EXPERIMENT_NAME)
assert exp is not None, f"Experiment '{EXPERIMENT_NAME}' not found."

runs = client.search_runs(
    [exp.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
assert runs, "No runs found to register."

run_id = runs[0].info.run_id
source = f"runs:/{run_id}/model"  # path logged in train script

print(f"Registering {source} as {MODEL_NAME} ...")
mv = mlflow.register_model(source, MODEL_NAME)

# Wait until READY
while True:
    cur = client.get_model_version(MODEL_NAME, mv.version)
    if cur.status == "READY":
        break
    time.sleep(1)

client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
print(f"Registered {MODEL_NAME} v{mv.version} and set alias '{ALIAS}'.")

