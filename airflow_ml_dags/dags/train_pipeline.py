from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from datetime import timedelta
from airflow.utils.dates import days_ago

FEATURES_DIR = "data/raw/{{ ds }}"
DATA_DIR = "/home/mivankin/airflow_ml_dags/data"
DATA_MODEL_DIR = "data/models/{{ ds }}"
DATA_PREDICTIONS_DIR = "data/predictions/{{ ds }}"
AIRFLOW_BASE_DIR = "usr/local/airflow"

with DAG(
    "train_pipeline",
    default_args={
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    },
    schedule_interval="@weekly",
    start_date=days_ago(1),
    tags=["custom"],
) as dag:

    features_check = FileSensor(
        task_id="features-check",
        filepath=f"{FEATURES_DIR}/features.csv",
        timeout=500,
        poke_interval=10,
        retries=50,
        mode="reschedule",
        fs_conn_id="fs_default",
    )

    target_check = FileSensor(
        task_id="target-check",
        filepath=f"{FEATURES_DIR}/target.csv",
        timeout=500,
        poke_interval=10,
        retries=50,
        mode="reschedule",
        fs_conn_id="fs_default",
    )

    load = DockerOperator(
        image="airflow-load",
        command=f"--input_dir={FEATURES_DIR} "
                f"--output_dir={FEATURES_DIR} ",
        network_mode="bridge",
        task_id="load",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}/:/data"],
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input_dir={FEATURES_DIR} "
                f"--output_dir={FEATURES_DIR}",
        network_mode="bridge",
        task_id="split",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}/:/data"],
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input_dir={FEATURES_DIR} "
                f"--output_dir={DATA_MODEL_DIR}",
        network_mode="bridge",
        task_id="train",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}/:/data"],
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--input_dir={FEATURES_DIR} "
                f"--model_dir={DATA_MODEL_DIR}",
        network_mode="bridge",
        task_id="validate",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}/:/data"],
    )

    [features_check, target_check] >> load >> split >> train >> validate
