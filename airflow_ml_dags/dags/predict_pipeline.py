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
    "predict_pipeline",
    default_args={
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    },
    schedule_interval="@daily",
    start_date=days_ago(1),
    tags=["custom"],
) as dag:

    data_check = FileSensor(
        task_id="data-check",
        filepath=f"{FEATURES_DIR}/features.csv",
        timeout=500,
        poke_interval=10,
        retries=50,
        mode="reschedule",
        fs_conn_id="fs_default",
    )

    model_check = FileSensor(
        task_id="model-check",
        filepath=f"{DATA_MODEL_DIR}/model.pkl",
        timeout=500,
        poke_interval=10,
        retries=50,
        mode="reschedule",
        fs_conn_id="fs_default",
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input_dir={FEATURES_DIR} "
                f"--output_dir={DATA_PREDICTIONS_DIR} "
                f"--model_dir={DATA_MODEL_DIR}",
        network_mode="bridge",
        task_id="predict",
        do_xcom_push=False,
        volumes=[f"{DATA_DIR}/:/data"],
    )

    [data_check, model_check] >> predict
