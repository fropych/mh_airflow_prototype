from datetime import datetime, timedelta

import pandas as pd
from airflow.decorators import task
from airflow.operators.python_operator import PythonOperator
from pyexpat import model

from airflow import DAG

# from scripts.models import Model


@task(task_id=f"load_data_from_db")
def load_data_from_db():
    print("LOAD DATA")
    data = pd.read_csv("/opt/airflow/data/data.csv")
    return data.to_dict()


@task(task_id=f"transform_data")
def transform_data(data):
    print("TRANSFORM DATA")
    return data


@task(task_id=f"train_count_model")
def train_count_model(data):
    # model = Model().fit(data, 'sold_count', '/opt/airflow/models/pipe_count.zip')
    # pred = model.count_agg_predict(data, '/opt/airflow/models/pipe_count.zip')
    print("TRAIN COUNT")


@task(task_id=f"train_volume_model")
def train_volume_model(data):
    # model = Model().fit(data, 'sold_volume', '/opt/airflow/models/pipe_volume.zip')
    # pred = model.volume_agg_predict(data, '/opt/airflow/models/pipe_volume.zip')
    print("TRAIN VOLUME")


@task(task_id=f"predict_count_model")
def predict_count_model(data):
    print("PREDICT COUNT")

    return 1


@task(task_id=f"predict_volume_model")
def predict_volume_model(data):
    print("PREDICT VOLUME")

    return 2


@task(task_id=f"load_to_db")
def load_to_db(count_pred, volume_pred):
    print(f"LOAD {count_pred} AND {volume_pred}")


default_args = {
    "owner": "marking hack",
    "depends_on_past": False,
    "start_date": datetime(2023, 3, 25),
    "email": ["hack@hack.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=15),
}

with DAG(
    "train_model",
    default_args=default_args,
    catchup=False,
    dagrun_timeout=timedelta(minutes=10),
    schedule_interval=timedelta(seconds=15),
) as dag:
    data = load_data_from_db()
    transformed_data = transform_data(data)
    trained_count = train_count_model(transformed_data)
    trained_volume = train_volume_model(transformed_data)

    count_pred = predict_count_model(transformed_data)
    volume_pred = predict_volume_model(transformed_data)

    load = load_to_db(count_pred, volume_pred)

    trained_count >> count_pred
    trained_volume >> volume_pred

    count_pred >> load
    volume_pred >> load
