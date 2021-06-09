from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from constants import DEFAULT_ARGS, START_DATE, DATA_RAW_DIR, LOCAL_FS_DATA_DIR


with DAG(
    "download_data",
    default_args=DEFAULT_ARGS,
    schedule_interval="@hourly",
    start_date=START_DATE,
    tags=["custom"],
) as dag:
    download_data = DockerOperator(
        image="airflow-download",
        command=f"{DATA_RAW_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=[LOCAL_FS_DATA_DIR],
    )
