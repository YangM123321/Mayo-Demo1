import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {"owner": "you", "depends_on_past": False, "retries": 0}

with DAG(
    dag_id="mayo_pipeline",
    start_date=pendulum.datetime(2025, 10, 1, tz="America/New_York"),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["demo"],
) as dag:

    etl_vitals = BashOperator(
        task_id="etl_vitals",
        bash_command="/bin/bash -lc 'set -euo pipefail; cd /opt/project && python etl/flatten_mimic_vitals.py'",
    )

    etl_labs = BashOperator(
        task_id="etl_labs",
        bash_command="/bin/bash -lc 'set -euo pipefail; cd /opt/project && python etl/flatten_mimic_labs.py'",
    )

    validate_with_ge = BashOperator(
        task_id="validate_with_ge",
        bash_command="/bin/bash -lc 'set -euo pipefail; echo VALIDATING && sleep 1'",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="/bin/bash -lc 'set -euo pipefail; cd /opt/project && python etl/build_features.py'",
    )

    build_labels = BashOperator(
        task_id="build_labels",
        bash_command="/bin/bash -lc 'set -euo pipefail; echo (labels embedded in train step for demo)'",
    )

    train_lr = BashOperator(
        task_id="train_lr",
        bash_command="/bin/bash -lc 'set -euo pipefail; cd /opt/project && python etl/train_baseline.py'",
    )

    # dependencies
    [etl_vitals, etl_labs] >> validate_with_ge >> build_features >> build_labels >> train_lr
