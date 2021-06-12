from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import timedelta
from airflow.utils.dates import days_ago


FEATURES_DIR = "data/raw/{{ ds }}"
DATA_DIR = "/home/mivankin/airflow_ml_dags/data"


with DAG(
    "download_data",
    default_args={
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    },
    schedule_interval="@hourly",
    start_date=days_ago(1),
    tags=["custom"],
) as dag:
    download_data = DockerOperator(
        image="airflow-download-data",
        command=f"{FEATURES_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-download-data",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}:/data"],
    )
